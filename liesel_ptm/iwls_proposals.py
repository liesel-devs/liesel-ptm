from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from liesel.goose.types import ModelInterface, ModelState

from .logprob import FlatLogProb
from .var import PTMCoef


@dataclass
class GaussianLocCholInfo:
    basis_name: str
    smooth_name: str
    smooth_scale_name: str
    scale_name: str
    penalty: ArrayLike
    model: ModelInterface
    model_state: ModelState
    n: int | None = None
    noncentered: bool = False

    @classmethod
    def from_smooth(
        cls,
        smooth,
        model: ModelInterface,
        scale_name: str = "$\\sigma$",
        n: int | None = None,
    ):
        try:
            noncentered = smooth.noncentered
        except AttributeError:
            noncentered = False

        cinfo = cls(
            basis_name=smooth.basis.name,
            smooth_name=smooth.name,
            smooth_scale_name=smooth.scale.name,
            scale_name=scale_name,
            penalty=smooth.coef.dist_node["penalty"].value,
            model=model,
            model_state=smooth.model.state,
            n=n,
            noncentered=noncentered,
        )
        return cinfo

    def __post_init__(self):
        pos = self.model.extract_position([self.basis_name], self.model_state)
        self.n = self.n or pos[self.basis_name].shape[0]

    def working_weights(self, model_state: ModelState) -> Array:
        pos = self.model.extract_position([self.scale_name], model_state)
        scale = pos[self.scale_name]
        eps = jnp.sqrt(jnp.finfo(jnp.asarray(scale).dtype).eps)
        return 1 / (jnp.clip(scale, a_min=eps) ** 2)

    def precision(self, model_state: ModelState) -> Array:
        pos = self.model.extract_position(
            [self.basis_name, self.smooth_scale_name], model_state
        )
        Z = pos[self.basis_name]
        scale = pos[self.smooth_scale_name]

        # Weights: support scalar or vector without materializing a diagonal
        w = jnp.asarray(self.working_weights(model_state), dtype=Z.dtype)
        # if scalar, broadcasts; if vector, row-weights
        ZW = Z * (w[:, None] if w.ndim == 1 else w)

        # Z^T W Z without constructing W
        ZTWZ = Z.T @ ZW

        eps = jnp.sqrt(jnp.finfo(Z.dtype).eps)  # small but not too small
        inv_scale2 = 1.0 / jnp.clip(scale, a_min=eps) ** 2

        P = ZTWZ + inv_scale2 * self.penalty
        return P + 1e-6 * jnp.mean(jnp.diag(P)) * jnp.eye(P.shape[0], P.shape[1])

    def chol_info(self, model_state: ModelState) -> Array:
        chol = jnp.linalg.cholesky(self.precision(model_state))
        if self.noncentered:
            pos = self.model.extract_position([self.smooth_scale_name], model_state)
            scale = pos[self.smooth_scale_name]
            return scale * chol

        return chol


@dataclass
class GaussianScaleCholInfo:
    basis_name: str
    smooth_name: str
    smooth_scale_name: str
    penalty: ArrayLike
    model: ModelInterface
    model_state: ModelState
    n: int | None = 0

    @classmethod
    def from_smooth(cls, smooth, model: ModelInterface, n: int | None = None):
        return cls(
            basis_name=smooth.basis.name,
            smooth_name=smooth.name,
            smooth_scale_name=smooth.scale.name,
            penalty=smooth.coef.dist_node["penalty"].value,
            model=model,
            model_state=smooth.model.state,
            n=n,
        )

    def __post_init__(self):
        pos = self.model.extract_position([self.basis_name], self.model_state)
        self.n = self.n or pos[self.basis_name].shape[0]

    def working_weights(self, model_state: ModelState) -> Array:
        return jnp.array(2.0)

    def precision(self, model_state: ModelState) -> Array:
        pos = self.model.extract_position(
            [self.basis_name, self.smooth_scale_name], model_state
        )
        Z = pos[self.basis_name]
        scale = pos[self.smooth_scale_name]

        # Weights: support scalar or vector without materializing a diagonal
        w = jnp.asarray(self.working_weights(model_state), dtype=Z.dtype)
        ZW = Z * (
            w[:, None] if w.ndim == 1 else w
        )  # if scalar, broadcasts; if vector, row-weights

        # Z^T W Z without constructing W
        ZTWZ = Z.T @ ZW

        eps = jnp.sqrt(jnp.finfo(Z.dtype).eps)  # small but not too small
        inv_scale2 = 1.0 / jnp.clip(scale, a_min=eps) ** 2

        P = ZTWZ + inv_scale2 * self.penalty
        return P + 1e-6 * jnp.mean(jnp.diag(P)) * jnp.eye(P.shape[0], P.shape[1])

    def chol_info(self, model_state: ModelState) -> Array:
        return jnp.linalg.cholesky(self.precision(model_state))


@dataclass
class PTMCholInfo:
    coef_name: str
    coef_scale_name: str
    penalty: ArrayLike
    model: ModelInterface
    model_state: ModelState

    @classmethod
    def from_coef(cls, coef: PTMCoef, model: ModelInterface):
        return cls(
            coef_name=coef.latent_coef.name,
            coef_scale_name=coef.latent_coef.dist_node["scale"].name,  # type: ignore
            penalty=coef.penalty,
            model=model,
            model_state=coef.model.state,  # type: ignore
        )

    def __post_init__(self):
        self.fisher_info = self.current_fisher_info(self.model_state)

    def current_fisher_info(self, model_state: ModelState) -> Array:
        pos = self.model.extract_position([self.coef_name], model_state)

        flat_position, _ = jax.flatten_util.ravel_pytree(pos[self.coef_name])
        logprob = FlatLogProb(self.model, model_state, [self.coef_name])
        info = -logprob.hessian(flat_position, model_state)

        def make_pd(A, jitter=1e-6):
            A = 0.5 * (A + A.T)
            w_min = jnp.linalg.eigvalsh(A).min()
            # If w_min < jitter, raise all eigenvalues by (jitter - w_min)
            shift = jnp.maximum(0.0, jitter - w_min)
            return A + shift * jnp.eye(A.shape[0], dtype=A.dtype)

        info = make_pd(jnp.atleast_2d(info))

        return info

    def precision(self, model_state: ModelState) -> Array:
        pos = self.model.extract_position([self.coef_scale_name], model_state)
        scale = pos[self.coef_scale_name]

        eps = jnp.sqrt(jnp.finfo(scale.dtype).eps)  # small but not too small
        inv_scale2 = 1.0 / jnp.clip(scale, a_min=eps) ** 2

        P = self.fisher_info  # + inv_scale2 * self.penalty
        return jnp.asarray(P * inv_scale2)

    def chol_info(self, model_state: ModelState) -> Array:
        return jnp.linalg.cholesky(self.precision(model_state))


@dataclass
class PTMCholInfoFixed:
    coef_name: str
    coef_scale_name: str
    penalty: ArrayLike
    model: ModelInterface
    model_state: ModelState
    fisher_info: ArrayLike | None = None
    chol_fisher_info: ArrayLike | None = None
    _fisher_info_unprocessed: ArrayLike | None = None

    @classmethod
    def from_coef(cls, coef: PTMCoef, model: ModelInterface):
        return cls(
            coef_name=coef.latent_coef.name,
            coef_scale_name=coef.latent_coef.dist_node["scale"].name,  # type: ignore
            penalty=coef.penalty,
            model=model,
            model_state=coef.model.state,  # type: ignore
        )

    def __post_init__(self):
        f1, f2 = self.current_fisher_info(self.model_state)
        self._fisher_info_unprocessed = f1
        self.fisher_info = f2
        self.chol_fisher_info = jnp.linalg.cholesky(self.fisher_info)

    def current_fisher_info(self, model_state: ModelState) -> tuple[Array, Array]:
        pos = self.model.extract_position([self.coef_name], model_state)

        flat_position, _ = jax.flatten_util.ravel_pytree(pos[self.coef_name])
        logprob = FlatLogProb(self.model, model_state, [self.coef_name])
        info = -logprob.hessian(flat_position, model_state)

        fisher_info_unprocessed = info

        def make_pd(A, jitter=1e-6):
            A = 0.5 * (A + A.T)
            w_min = jnp.linalg.eigvalsh(A).min()
            # If w_min < jitter, raise all eigenvalues by (jitter - w_min)
            shift = jnp.maximum(0.0, jitter - w_min)
            return A + shift * jnp.eye(A.shape[0], dtype=A.dtype)

        info_pd = make_pd(jnp.atleast_2d(info))

        return fisher_info_unprocessed, info_pd

    def chol_info(self, model_state: ModelState) -> Array:
        return jnp.asarray(self.chol_fisher_info)

    @property
    def nan_in_cholesky_of_unprocessed_finfo(self) -> bool:
        chol = jnp.linalg.cholesky(self._fisher_info_unprocessed)
        return bool(jnp.any(jnp.isnan(chol)))


@dataclass
class CholInfo:
    coef_name: str
    model: ModelInterface
    model_state: ModelState
    _fisher_info_unprocessed: ArrayLike | None = None
    fisher_info: ArrayLike | None = None
    chol_fisher_info: ArrayLike | None = None

    @classmethod
    def from_smooth(cls, smooth, model: ModelInterface):
        return cls(
            coef_name=smooth.coef.name,
            model=model,
            model_state=smooth.model.state,
        )

    def __post_init__(self):
        f1, f2 = self.current_fisher_info(self.model_state)
        self._fisher_info_unprocessed = f1
        self.fisher_info = f2
        self.chol_fisher_info = jnp.linalg.cholesky(self.fisher_info)

    def current_fisher_info(self, model_state: ModelState) -> tuple[Array, Array]:
        pos = self.model.extract_position([self.coef_name], model_state)

        flat_position, _ = jax.flatten_util.ravel_pytree(pos[self.coef_name])
        logprob = FlatLogProb(self.model, model_state, [self.coef_name])
        info = -logprob.hessian(flat_position, model_state)
        info += jnp.eye(info.shape[0]) * 1e-6

        fisher_info_unprocessed = info

        def make_pd(A, jitter=1e-6):
            A = 0.5 * (A + A.T)
            w_min = jnp.linalg.eigvalsh(A).min()
            # If w_min < jitter, raise all eigenvalues by (jitter - w_min)
            shift = jnp.maximum(0.0, jitter - w_min)
            return A + shift * jnp.eye(A.shape[0], dtype=A.dtype)

        fisher_info_pd = make_pd(jnp.atleast_2d(info))

        return fisher_info_unprocessed, fisher_info_pd

    def chol_info(self, model_state: ModelState) -> Array:
        return jnp.asarray(self.chol_fisher_info)

    @property
    def nan_in_cholesky_of_unprocessed_finfo(self) -> bool:
        chol = jnp.linalg.cholesky(self._fisher_info_unprocessed)
        return bool(jnp.any(jnp.isnan(chol)))


@dataclass
class ObservedCholInfoOrIdentity:
    coef_name: str
    model: ModelInterface
    model_state: ModelState
    _fisher_info_unprocessed: ArrayLike | None = None
    fisher_info: ArrayLike | None = None
    chol_fisher_info: ArrayLike | None = None

    @classmethod
    def from_smooth(cls, smooth, model: ModelInterface):
        return cls(
            coef_name=smooth.coef.name,
            model=model,
            model_state=smooth.model.state,
        )

    def __post_init__(self):
        f1, f2 = self.current_fisher_info(self.model_state)
        self._fisher_info_unprocessed = f1
        self.fisher_info = f2
        self.chol_fisher_info = jnp.linalg.cholesky(self.fisher_info)
        self.eye = jnp.eye(self.chol_fisher_info.shape[0])

    def current_fisher_info(self, model_state: ModelState) -> tuple[Array, Array]:
        pos = self.model.extract_position([self.coef_name], model_state)

        flat_position, _ = jax.flatten_util.ravel_pytree(pos[self.coef_name])
        logprob = FlatLogProb(self.model, model_state, [self.coef_name])
        info = -logprob.hessian(flat_position, model_state)
        info += jnp.eye(info.shape[0]) * 1e-6

        fisher_info_unprocessed = info

        def make_pd(A, jitter=1e-6):
            A = 0.5 * (A + A.T)
            w_min = jnp.linalg.eigvalsh(A).min()
            # If w_min < jitter, raise all eigenvalues by (jitter - w_min)
            shift = jnp.maximum(0.0, jitter - w_min)
            return A + shift * jnp.eye(A.shape[0], dtype=A.dtype)

        fisher_info_pd = make_pd(jnp.atleast_2d(info))

        return fisher_info_unprocessed, fisher_info_pd

    def chol_info_or_identity(self, model_state: ModelState) -> tuple[Array, Array]:
        pos = self.model.extract_position([self.coef_name], model_state)

        flat_position, _ = jax.flatten_util.ravel_pytree(pos[self.coef_name])
        logprob = FlatLogProb(self.model, model_state, [self.coef_name])
        info = -logprob.hessian(flat_position, model_state)
        info += self.eye * 1e-6
        chol_info = jnp.linalg.cholesky(info)
        return jax.lax.cond(
            jnp.any(jnp.isnan(chol_info)), lambda: self.eye, lambda: chol_info
        )

    def chol_info(self, model_state: ModelState) -> Array:
        return jnp.asarray(self.chol_info_or_identity(model_state))

    @property
    def nan_in_cholesky_of_unprocessed_finfo(self) -> bool:
        chol = jnp.linalg.cholesky(self._fisher_info_unprocessed)
        return bool(jnp.any(jnp.isnan(chol)))
