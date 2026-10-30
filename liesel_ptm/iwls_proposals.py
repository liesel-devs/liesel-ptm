from dataclasses import dataclass

import jax
import jax.flatten_util
import jax.numpy as jnp
import liesel.goose as gs
from jax import Array
from jax.typing import ArrayLike
from liesel.goose.types import ModelInterface, ModelState

from .var import PTMCoef


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
        logprob = gs.FlatInterfaceLogProb(self.model, model_state, [self.coef_name])
        info = -logprob.hessian(flat_position)

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
        logprob = gs.FlatInterfaceLogProb(self.model, model_state, [self.coef_name])
        info = -logprob.hessian(flat_position)
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
