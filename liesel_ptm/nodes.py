from __future__ import annotations

import logging
import warnings
from abc import abstractmethod, abstractproperty
from collections.abc import Callable, Sequence
from functools import partial
from itertools import chain

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import scipy
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.distributions import MultivariateNormalDegenerate
from liesel.goose.kernel import Kernel
from liesel.model.nodes import no_model_setter
from sklearn.preprocessing import LabelBinarizer

from .bsplines import EquidistantKnots, OnionCoef, OnionKnots
from .custom_types import Array, KeyArray, TFPDistribution
from .liesel_internal import splines
from .sampling import summarise_by_quantiles, summarise_by_samples
from .var import Var

bspline_basis = splines.build_design_matrix_b_spline
bspline_basis_d = splines.build_design_matrix_b_spline_derivative
kn = splines.create_equidistant_knots

logger = logging.getLogger("liesel")


class TransformationDist(lsl.Dist):
    """A transformation-distribution node for a conditional transformation model."""

    def __init__(
        self,
        transformed_variable: lsl.Calc,
        transformation_derivative: lsl.Calc,
        refdist: TFPDistribution,
        _name: str = "",
        _needs_seed: bool = False,
    ):
        super(lsl.Dist, self).__init__(
            transformed_variable,
            transformation_derivative,
            _name=_name,
            _needs_seed=_needs_seed,
        )

        self._per_obs = True
        self.refdist = refdist
        self._transformed_variable = transformed_variable
        self._transformation_derivative = transformation_derivative

    @property
    def log_prob(self) -> Array:
        """The log-probability of the distribution."""
        return self.value

    @property
    def per_obs(self) -> bool:
        """Whether the log-probability is stored per observation or summed up."""
        return self._per_obs

    @per_obs.setter
    @no_model_setter
    def per_obs(self, per_obs: bool):
        self._per_obs = per_obs

    def update(self) -> TransformationDist:
        base_log_prob = self.refdist.log_prob(self._transformed_variable.value)
        deriv = self._transformation_derivative.value
        deriv = jnp.maximum(deriv, 1e-30)
        log_prob_adjustment = jnp.log(deriv)
        log_prob = jnp.add(base_log_prob, log_prob_adjustment)

        if not self.per_obs and hasattr(log_prob, "sum"):
            log_prob = log_prob.sum()

        self._value = log_prob
        self._outdated = False
        return self


class TransformationDistLogDeriv(lsl.Dist):
    """A transformation-distribution node for a conditional transformation model."""

    def __init__(
        self,
        transformed_variable: lsl.Calc,
        log_transformation_derivative: lsl.Calc,
        refdist: TFPDistribution,
        _name: str = "",
        _needs_seed: bool = False,
    ):
        super(lsl.Dist, self).__init__(
            transformed_variable,
            log_transformation_derivative,
            _name=_name,
            _needs_seed=_needs_seed,
        )

        self._per_obs = True
        self.refdist = refdist
        self._transformed_variable = transformed_variable
        self._log_transformation_derivative = log_transformation_derivative

    @property
    def log_prob(self) -> Array:
        """The log-probability of the distribution."""
        return self.value

    @property
    def per_obs(self) -> bool:
        """Whether the log-probability is stored per observation or summed up."""
        return self._per_obs

    @per_obs.setter
    @no_model_setter
    def per_obs(self, per_obs: bool):
        self._per_obs = per_obs

    def update(self) -> TransformationDist:
        base_log_prob = self.refdist.log_prob(self._transformed_variable.value)
        log_prob_adjustment = self._log_transformation_derivative.value
        log_prob = jnp.add(base_log_prob, log_prob_adjustment)

        if not self.per_obs and hasattr(log_prob, "sum"):
            log_prob = log_prob.sum()

        self._value = log_prob
        self._outdated = False
        return self


def model_matrix(*args: Array, intercept: bool = True) -> Array:
    x = np.vstack(args)
    if intercept:
        icol = np.ones(x.shape[1])
        return np.vstack((icol, x)).T
    return x.T


def _matrix(x: Array) -> Array:
    if not np.shape(x):
        x = np.atleast_2d(x)
    elif len(np.shape(x)) == 1:
        x = np.expand_dims(x, axis=1)
    elif len(np.shape(x)) == 2:
        pass
    else:
        raise ValueError(f"Shape of x is unsupported: {np.shape(x)}")
    return x


def dot(x: Array, coef: Array):
    return x @ coef


def scaled_dot(x: Array, coef: Array, scale: Array):
    return x @ (scale * coef)


@partial(jnp.vectorize, signature="(n,p),(p)->(n,p)")
def elementwise_dot(x: Array, coef: Array):
    return x * coef


class Dot(lsl.Calc):
    """
    A dot product. Assumes that ``x`` is fixed and that ``coef`` is strong.
    """

    def __init__(
        self, x: lsl.Var | lsl.Node, coef: lsl.Var | lsl.Node, _name: str = ""
    ) -> None:
        if not coef.strong:
            raise ValueError("coef must be a strong node.")
        super().__init__(dot, x=x, coef=coef, _name=_name)
        self.update()

    def predict(self, samples: dict[str, Array], x: Array | None = None) -> Array:
        x = x if x is not None else self.kwinputs["x"].value
        coef_samples = samples[self.kwinputs["coef"].var.name]
        coef_shape = self.kwinputs["coef"].value.shape
        if coef_shape[0] == 1 and len(coef_samples.shape) < 3:
            coef_samples = np.expand_dims(coef_samples, axis=-1)
        smooth = np.einsum("...pj,...j->...p", _matrix(x), coef_samples)
        return smooth

    def predict_elementwise(
        self, samples: dict[str, Array], x: Array | None = None
    ) -> Array:
        x = x if x is not None else self.kwinputs["x"].value
        coef_samples = samples[self.kwinputs["coef"].var.name]
        return elementwise_dot(_matrix(x), coef_samples)


class IncDot(lsl.Calc):
    """
    A dot product for an increasing spline. Assumes that ``x`` is fixed and that
    ``coef`` is strong.
    """

    def __init__(
        self, x: lsl.Var | lsl.Node, coef: lsl.Var | lsl.Node, _name: str = ""
    ) -> None:
        super().__init__(dot, x=x, coef=coef, _name=_name)
        self.update()

    def predict(self, samples: dict[str, Array], x: Array | None = None) -> Array:
        x = x if x is not None else self.kwinputs["x"].value

        log_coef_name = self.kwinputs["coef"].var.value_node.inputs[0].var.name
        log_coef_samples = samples[log_coef_name]

        exp_fn = self.kwinputs["coef"].var.value_node.function
        coef_samples = exp_fn(log_coef_samples)

        coef_shape = self.kwinputs["coef"].value.shape
        if coef_shape[0] == 1 and len(coef_samples.shape) < 3:
            coef_samples = np.expand_dims(coef_samples, axis=-1)
        smooth = np.einsum("...pj,...j->...p", _matrix(x), coef_samples)
        return smooth


class ScaledDot(lsl.Calc):
    def __init__(
        self,
        x: lsl.Var | lsl.Node,
        coef: lsl.Var | lsl.Node,
        scale: ExpParam,
        _name: str = "",
    ) -> None:
        super().__init__(scaled_dot, x=x, coef=coef, scale=scale, _name=_name)
        self.update()
        self.x = x
        self.scale = scale
        self.coef = coef

    def predict(self, samples: dict[str, Array], x: Array | None = None) -> Array:
        if not self.coef.strong:
            raise ValueError("To use predict(), coef must be a strong node.")

        coef_samples = samples[self.coef.name]
        coef_samples = np.atleast_3d(coef_samples)

        scale_samples = self.scale.predict(samples)
        scale_samples = np.atleast_3d(scale_samples)

        scaled_coef_samples = scale_samples * coef_samples

        x = x if x is not None else self.x.value
        smooth = np.tensordot(_matrix(x), scaled_coef_samples, axes=([1], [-1]))
        return np.moveaxis(smooth, 0, -1)


class ScaledBasisDot(lsl.Calc):
    def __init__(
        self,
        x: lsl.Var | lsl.Node,
        coef: lsl.Var | lsl.Node,
        scale: ExpParam,
        basis_fn: Callable[[Array], Array] | None = None,
        _name: str = "",
    ) -> None:
        if not coef.strong:
            raise ValueError("coef must be a strong node.")
        super().__init__(scaled_dot, x=x, coef=coef, scale=scale, _name=_name)
        self.update()

        self.x = x
        self.scale = scale
        self.coef = coef
        self.basis_fn = lambda x: x

        if basis_fn is not None:
            self.basis_fn = basis_fn
        else:
            try:
                self.basis_fn = self.kwinputs["x"].var.evaluate
            except AttributeError:
                pass

    def predict(self, samples: dict[str, Array], x: Array | None = None) -> Array:
        coef_samples = samples[self.coef.name]
        coef_samples = np.atleast_3d(coef_samples)

        scale_samples = self.scale.predict(samples)
        scale_samples = np.atleast_3d(scale_samples)

        scaled_coef_samples = scale_samples * coef_samples
        basis = self.basis_fn(x) if x is not None else self.x.value
        smooth = np.tensordot(basis, scaled_coef_samples, axes=([1], [-1]))
        return np.moveaxis(smooth, 0, -1)


class BasisDot(Dot):
    def predict(self, samples: dict[str, Array], x: Array | None = None) -> Array:
        coef_samples = samples[self.kwinputs["coef"].var.name]
        coef_samples = np.atleast_3d(coef_samples)
        basis = self.kwinputs["x"].var.evaluate(x)
        smooth = np.tensordot(basis, coef_samples, axes=([1], [-1]))
        return np.moveaxis(smooth, 0, -1)


def nullspace_remover(pen: Array) -> Array:
    """
    Constructs a reparameterization matrix fo removing the nullspace from a penalty
    matrix of a structured additive predictor term.

    Parameters
    ----------
    pen
        Penalty matrix.

    Examples
    --------

    Example usage::

        import numpy as np
        import tensorflow_probability.substrates.jax.distributions as tfd
        import liesel.model as lsl
        import liesel_ptm as ptm

        np.random.seed(2407)
        x = np.random.uniform(low=-1.0, high=1.0, size=300)
        nparam = 10
        knots = ptm.kn(x, n_params=nparam)

        K = ptm.diffpen(nparam, diff=2)
        Z = ptm.nullspace_remover(K)
        Kz = Z.T @ K  @ Z

        def basis_fn(x):
            return ptm.bspline_basis(x, knots, 3) @ Z

        tau2 = ptm.VarWeibull(10.0, scale=0.05, name="tau2")

        term = ptm.StructuredAdditiveTerm(
            x=x, basis_fn=basis_fn, K=Kz, tau2=tau2, name="my_term"
        )

    """
    ker = scipy.linalg.null_space(pen)
    Q, _ = np.linalg.qr(ker, mode="complete")
    Z = Q[:, ker.shape[-1] :]
    return Z


def sumzero_coef(nparam: int) -> Array:
    """
    Matrix ``Z`` for reparameterization for sum-to-zero-constraint of coefficients.

    The reparameterization matrix returned by this function
    applies a sum-to-zero constraint on the coefficients of a spline.

    Parameters
    ----------
    nparam
        Number of parameters.

    See Also
    --------
    sumzero_term : Matrix ``Z`` for reparameterization for sum-to-zero-constraint of
        term.

    Examples
    --------

    Example usage::

        import numpy as np
        import tensorflow_probability.substrates.jax.distributions as tfd
        import liesel.model as lsl
        import liesel_ptm as ptm

        np.random.seed(2407)
        x = np.random.uniform(low=-1.0, high=1.0, size=300)
        nparam = 10
        knots = ptm.kn(x, n_params=nparam)

        Z = ptm.sumzero_coef(nparam)

        K = Z.T @ ptm.diffpen(nparam, diff=2) @ Z

        def basis_fn(x):
            return ptm.bspline_basis(x, knots, 3) @ Z

        tau2 = ptm.VarWeibull(10.0, scale=0.05, name="tau2")

        term = ptm.StructuredAdditiveTerm(
            x=x, basis_fn=basis_fn, K=K, tau2=tau2, name="my_term"
        )

    """
    j = np.ones(shape=(nparam, 1), dtype=np.float32)
    q, _ = np.linalg.qr(j, mode="complete")
    return q[:, 1:]


# for backwards-compatibility
sumzero = sumzero_coef


def sumzero_term(basis: Array) -> Array:
    """
    Matrix ``Z`` for reparameterization for sum-to-zero-constraint of a structured
    additive term.

    The reparameterization matrix returned by this function applies a sum-to-zero
    constraint on the evaluations of a spline.

    Parameters
    ----------
    basis
        Basis matrix to work on.

    See Also
    --------
    sumzero_coef : Matrix ``Z`` for reparameterization for sum-to-zero-constraint of
        coefficients.

    Examples
    --------

    Example usage::

        import numpy as np
        import tensorflow_probability.substrates.jax.distributions as tfd
        import liesel.model as lsl
        import liesel_ptm as ptm

        np.random.seed(2407)
        x = np.random.uniform(low=-1.0, high=1.0, size=300)
        nparam = 10
        knots = ptm.kn(x, n_params=nparam)

        Z = ptm.sumzero_term(ptm.bspline_basis(x, knots, 3))
        K = Z.T @ ptm.diffpen(nparam, diff=2) @ Z

        def basis_fn(x):
            return ptm.bspline_basis(x, knots, 3) @ Z

        tau2 = ptm.VarWeibull(10.0, scale=0.05, name="tau2")

        term = ptm.StructuredAdditiveTerm(
            x=x, basis_fn=basis_fn, K=K, tau2=tau2, name="my_term"
        )

    """
    nobs = basis.shape[0]
    j = np.ones(shape=nobs, dtype=np.float32)
    A = jnp.expand_dims(j @ basis, 0)
    q, _ = np.linalg.qr(A.T, mode="complete")
    return q[:, 1:]


def cholesky_ltinv(a: Array) -> Array:
    L = np.linalg.cholesky(a)
    return np.linalg.inv(L).T


def diffpen(ncol: int, diff: int = 2):
    """A P-spline penalty matrix based on ``diff``-order differences."""
    D = np.diff(np.identity(ncol), diff, axis=0)
    return np.array(D.T @ D, dtype=np.float32)


# ----------------------------------------------------------------------
# Basis matrices for univariate splines
# ----------------------------------------------------------------------


class BSplineBasis(lsl.Data):
    """A design matrix of B-spline basis function evaluations."""

    observed = True

    def __init__(
        self,
        knots: Array,
        value: Array,
        order: int = 3,
        centered: bool = False,
        name: str = "",
    ) -> None:
        basis = bspline_basis(value, knots=knots, order=order)

        self.min = knots[3]
        self.max = knots[-4]
        #: The observed values at which the Bspline is evaluated.
        self.observed_value = value

        #: Number of parameters associated with this basis matrix.
        self.nparam = basis.shape[-1]

        self._reparam_matrix = np.eye(self.nparam, dtype=np.float32)

        #: Array of knots that were used to create the BasisMatrix.
        self.knots = knots

        #: Order of the B-splines used in this basis matrix.
        self.order = order

        #: Column means of the uncentered basis matrix, based on :attr:`.observed_value`.
        self.colmeans = basis.mean(axis=0)

        #: If ``True``, the column means are subtracted from the basis matrix
        #: evaluations. Centering also applies to new evaluations computed via
        # :meth:`.evaluate`.
        self.centered = centered

        super().__init__(value=(basis - self.centered * self.colmeans), _name=name)

    @classmethod
    def auto(
        cls,
        value: Array,
        nparam: int,
        order: int = 3,
        centered: bool = False,
        name: str = "",
    ) -> BSplineBasis:
        knots = kn(value, order=order, n_params=nparam)
        basis = cls(knots, value, order, centered, name)
        return basis

    @classmethod
    def from_basis(
        cls, basis: BSplineBasis, value: Array, name: str = ""
    ) -> BSplineBasis:
        """
        Constructs a new basis matrix based on an existing one.
        Can be useful if you want the columns of the new basis matrix to be centered
        according to the input basis.
        """

        new_basis = cls(
            knots=basis.knots, value=value, order=basis.order, centered=False, name=name
        )
        new_basis.colmeans = basis.colmeans

        new_basis.value = new_basis.value - (  # type: ignore
            basis.centered * basis.colmeans
        )
        new_basis.colmeans = basis.colmeans

        return new_basis

    @property
    def reparam_matrix(self) -> Array:
        """Reparameterisation matrix. Gets set by :meth:`.reparam`."""
        return self._reparam_matrix

    def reparam(self, z: Array) -> BSplineBasis:
        """
        Adds a reparameterisation matrix. Updates the number of parameters, since
        this number may decrease depending on the reparameterization.
        """
        self._reparam_matrix = self.reparam_matrix @ z
        self.value = self.value @ z  # type: ignore
        self.nparam = self.value.shape[-1]
        return self

    def evaluate(self, value: Array | float | None) -> Array:
        """
        Evaluate the B-spline basis functions with :attr:`.knots` and :attr:`.order` for
        new values. If the basis has been reparameterized using :meth:`.reparam`, this
        reparameterization is applied, too.

        If ``value=None``, evaluate the derivative at :attr:`.observed_value`.
        """
        if value is None:
            return self.value
        value = np.atleast_1d(np.array(value, dtype=np.float32))
        basis = bspline_basis(value, knots=self.knots, order=self.order)
        return (basis - self.centered * self.colmeans) @ self.reparam_matrix

    def deriv(self, value: Array | float | None = None) -> Array:
        """
        Evaluate the derivative of the B-spline basis functions with :attr:`.knots` and
        :attr:`.order` for new values. If the basis has been reparameterized using
        :meth:`.reparam`, this reparameterization is applied, too.

        If ``value=None``, evaluate the derivative at :attr:`.obs_value`.
        """
        value = value if value is not None else self.observed_value
        value = np.atleast_1d(np.array(value, dtype=np.float32))
        basis = bspline_basis_d(value, knots=self.knots, order=self.order)
        return basis @ self.reparam_matrix


class ExpParam(lsl.Var):
    def __init__(self, value, distribution: lsl.Dist, name: str = "") -> None:
        InputDist = distribution.distribution
        inputs = distribution.inputs
        kwinputs = distribution.kwinputs

        bijector = tfb.Exp()
        bijector_inv = tfb.Invert(bijector)

        def exp_dist(*args, **kwargs):
            return tfd.TransformedDistribution(InputDist(*args, **kwargs), bijector_inv)

        dist = lsl.Dist(
            exp_dist,
            *inputs,
            **kwinputs,
            _name=distribution.name,
            _needs_seed=distribution.needs_seed,
        )

        self.log_var = lsl.Var(
            bijector.inverse(value), dist, name=f"{name}_transformed"
        )
        self.log_var.parameter = True
        super().__init__(lsl.Calc(bijector.forward, self.log_var), name=name)
        self.value_node.monitor = True
        self.update()

    def predict(self, samples: dict[str, Array]) -> Array:
        samps = samples[self.log_var.name]
        return self.value_node.function(samps)


class Term(lsl.Group):
    @abstractmethod
    def predict(self, samples, x: Array | None) -> Array:
        ...

    @abstractproperty
    def value(self) -> lsl.Var | lsl.Node:
        ...

    @abstractproperty
    def observed_value(self) -> Array:
        ...

    @abstractproperty
    def parameters(self) -> list[str]:
        ...

    @abstractproperty
    def hyper_parameters(self) -> list[str]:
        ...


class NonlinearPSpline(Term):
    """
    Group of nodes for a univariate nonlinear PSpline.

    This Pspline assumes B-spline bases of order 3 and a second-order random walk prior
    for smoothness. The null space of the prior penalty matrix is completely removed,
    such that the PSpline contains no constant and no linear trend.

    .. warning::
        Even though the null space is removed, this does not mean that the PSpline
        cannot contain a linear effect given observed covariate values. This is the case
        for example, if the covariate is not uniformly distributed. In this case, it
        is possible for the PSpline to contain a linear effect, even though the null
        space is removed. To fully remove the linear effect, the linear part would
        need to be removed manually.

    The PSpline's coefficient is reparameterized to have a standard normal prior.

    Parameters
    ----------
    knots
        Knots of the B-spline basis functions.
    x
        Observed covariate values.
    scale
        Scale of the random walk prior.
    name
        Name of the group.
    """

    def __init__(self, knots: Array, x: Array, scale: Var, name: str) -> None:
        self.scale = scale

        self._observed_value = x
        self.knots = knots

        basis = BSplineBasis(knots, x, order=3)

        pen = diffpen(ncol=basis.nparam, diff=2)
        Z = nullspace_remover(pen)
        Ltinv = cholesky_ltinv(Z.T @ pen @ Z)

        def basis_fn(x):
            basis = bspline_basis(x, knots=knots, order=3)
            return basis @ Z @ Ltinv

        basis = basis_fn(x)
        X = np.c_[np.ones_like(x), x]

        P = np.eye(X.shape[0]) - (X @ np.linalg.inv(X.T @ X) @ X.T)
        basis_p = P @ basis

        self.basis = Var(basis_p, name=f"{name}_basis")
        self.nparam = basis_p.shape[-1]

        prior = lsl.Dist(tfd.Normal, loc=0.0, scale=1.0)
        self.coef = lsl.param(
            np.zeros(self.nparam), prior, name=f"{name}_coef_transformed"
        )

        self.smooth = Var(
            ScaledBasisDot(
                x=self.basis, coef=self.coef, scale=self.scale, basis_fn=basis_fn
            ),
            name=f"{name}_smooth",
        )

        scale_param = find_param(self.scale)

        self._hyper_parameters: list[str] = []
        if scale_param is not None:
            self._hyper_parameters.append(scale_param.name)

        super().__init__(
            name=name,
            smooth=self.smooth,
            basis=self.basis,
            coef=self.coef,
            scale=self.scale,
        )
        self.mcmc_kernels: list[Kernel] = self._default_kernels()

    @classmethod
    def from_nparam(
        cls, x: Array, nparam: int, scale: Var, name: str
    ) -> NonlinearPSpline:
        knots = kn(x, order=3, n_params=nparam)
        return cls(knots, x, scale, name)

    @property
    def hyper_parameters(self) -> list[str]:
        return self._hyper_parameters

    @property
    def parameters(self) -> list[str]:
        return [self.coef.name]

    @property
    def value(self) -> Var:
        return self.smooth

    def predict(
        self,
        samples: dict[str, Array],
        x: Array | None = None,
        center: bool = False,
        scale: bool = False,
    ) -> Array:
        fx = self.smooth.predict(samples=samples, x=x)

        if center:
            fx = fx - fx.mean(axis=-1, keepdims=True)
        if scale:
            fx = fx / fx.std(axis=-1, keepdims=True)
        return fx

    def posterior_coef(self, samples: dict[str, Array]) -> Array:
        calc = ScaledDot(self.basis.reparam_matrix, self.coef, self.scale)
        return calc.predict(samples)

    @property
    def observed_value(self) -> Array:
        return self._observed_value

    def summarise_by_quantiles(
        self,
        samples: dict[str, Array],
        x: Array | None = None,
        axis: Sequence[int] | int = (0, 1),
        lo: float = 0.1,
        hi: float = 0.9,
        center: bool = False,
        scale: bool = False,
    ) -> pd.DataFrame:
        x = x if x is not None else self.observed_value
        fx = self.predict(samples, x=x, center=center, scale=scale)

        df = summarise_by_quantiles(fx, axis=axis, lo=lo, hi=hi)

        df["x_value"] = np.asarray(jnp.squeeze(x))
        df["name"] = self.name

        return df

    def summarise_by_samples(
        self,
        key: KeyArray,
        samples: dict[str, Array],
        n: int = 100,
        x: Array | None = None,
        center: bool = False,
        scale: bool = False,
    ) -> pd.DataFrame:
        x = x if x is not None else self.observed_value
        fx = self.predict(samples, x=x, center=center, scale=scale)

        df = summarise_by_samples(key, fx, "value", n=n)

        x = jnp.squeeze(x)
        if jnp.atleast_1d(x).shape[-1] == 1:
            df[self.name] = np.asarray(x)
        else:
            df[self.name] = np.asarray(np.tile(x, n))

        return df

    def _default_kernels(self) -> list[Kernel]:
        return [gs.NUTSKernel(self.parameters), gs.NUTSKernel(self.hyper_parameters)]


def array_to_dict(
    x: Array, names_prefix: str = "x", prefix_1d: bool = False
) -> dict[str, Array]:
    """Turns a 2d-array into a dict."""

    if isinstance(x, float) or x.ndim == 1:
        if prefix_1d:
            return {f"{names_prefix}0": x}
        else:
            return {names_prefix: x}
    elif x.ndim == 2:
        return {f"{names_prefix}{i}": x[:, i] for i in range(x.shape[-1])}
    else:
        raise ValueError(f"x should have ndim <= 2, but it has x.ndim={x.ndim}")


def bs(
    knots: Array, order: int = 3, Z: Array | None = None
) -> Callable[[Array], Array]:
    """Returns a function that evaluates the B-spline basis functions."""

    if Z is None:
        nparam = len(knots) - order - 1
        Z = jnp.eye(nparam, dtype=np.float32)

    def bs_(x: Array) -> Array:
        return bspline_basis(x, knots=knots, order=order) @ Z

    return bs_


def find_param(var: lsl.Var) -> lsl.Var | None:
    if var.parameter:
        if not var.strong:
            raise ValueError(f"{var} is marked as a parameter but it is not strong.")
        return var

    if not var.value_node.inputs:
        return None

    var_value_node = var.value_node.inputs[0]
    value_var = var_value_node.inputs[0].var
    return find_param(value_var)


class StructuredAdditiveTerm(Term):
    """
    Term in a structured additive predictor.

    This term has the form ``basis_matrix @ coef_vector``, where ``coef_vector``
    is equipped with a potentially singular multivariate normal prior, constructed
    using a penalty matrix ``K`` and a variance parameter ``tau2``.

    The user supplies a callable for the basis matrix, because this allows the class
    to easily evaluate the basis with manually chosen covariate values for predictions.


    Parameters
    ----------
    x
        Covariate array.
    basis_fn
        A function that takes ``x`` and returns an array of basis function evaluations.
    K
        Penalty matrix.
    tau2
        Variance parameter for the singular normal prior.
    name
        Unique name of the term.
    mcmc_kernel
        Kernel class to use by default.


    See Also
    --------
    .bspline_basis : Evaluates B-spline basis functions.
    .kn : Creates equidistant knots.
    .diffpen : Evaluates a differences-penalty matrix.
    .sumzero_term : Reparameterization matrix for sum-to-zero constraint.
    .sumzero_coef : Reparameterization matrix for sum-to-zero constraint.
    .nullspace_remover : Reparameterization matrix for sum-to-zero constraint.

    Examples
    --------

    In this example, we define a P-spline term using B-spline bases and a second-order
    random walk penalty matrix. We reparameterize both the basis and
    the penalty matrix ``K`` to enforce a sum-to-zero constraint on the
    term::

        import numpy as np
        import tensorflow_probability.substrates.jax.distributions as tfd
        import liesel.model as lsl
        import liesel_ptm as ptm

        np.random.seed(2407)
        x = np.random.uniform(low=-1.0, high=1.0, size=300)
        nparam = 10
        knots = ptm.kn(x, n_params=nparam)

        Z = ptm.sumzero_term(ptm.bspline_basis(x, knots, 3))
        K = Z.T @ ptm.diffpen(nparam, diff=2) @ Z

        def basis_fn(x):
            return ptm.bspline_basis(x, knots, 3) @ Z

        tau2 = ptm.VarWeibull(10.0, scale=0.05, name="tau2")

        term = ptm.StructuredAdditiveTerm(
            x=x, basis_fn=basis_fn, K=K, tau2=tau2, name="my_term"
        )

    """

    def __init__(
        self,
        x: Array,
        basis_fn: Callable[[Array], Array],
        K: Array,
        tau2: lsl.Var,
        name: str,
        mcmc_kernel: Kernel = gs.NUTSKernel,
        combined_kernels: bool = False,
    ) -> None:
        self._default_kernel = mcmc_kernel
        self.x = lsl.obs(x, name=f"{name}_covariate")
        """Covariate node."""

        self.basis_fn = jnp.vectorize(basis_fn, signature="(n)->(n,p)")
        """Basis function."""

        self.basis = lsl.obs(basis_fn(x), name=f"{name}_basis")
        """Basis matrix node."""

        self.K = lsl.Data(K, _name=f"{name}_K")
        """Penalty matrix node."""

        self.tau2 = tau2
        """Variance parameter node."""

        self.evals = jnp.linalg.eigvalsh(K)
        self.rank = lsl.Data(jnp.sum(self.evals > 0.0), _name=f"{name}_K_rank")
        _log_pdet = jnp.log(jnp.where(self.evals > 0.0, self.evals, 1.0)).sum()
        self.log_pdet = lsl.Data(_log_pdet, _name=f"{name}_K_log_pdet")

        prior = lsl.Dist(
            MultivariateNormalDegenerate.from_penalty,
            loc=0.0,
            var=tau2,
            pen=self.K,
            rank=self.rank,
            log_pdet=self.log_pdet,
        )

        self.nparam = jnp.shape(K)[-1]
        start_value = jnp.zeros(self.nparam)
        self.coef = lsl.param(start_value, prior, name=f"{name}_coef")
        """Coefficient node."""

        self.smooth = Var(Dot(self.basis, self.coef), name=f"{name}_smooth")
        """Smooth node."""

        tau2_param = find_param(self.tau2)

        self._hyper_parameters: list[str] = []
        if tau2_param is not None:
            self._hyper_parameters.append(tau2_param.name)

        self._parameters: list[str] = [self.coef.name]

        super().__init__(
            name=name,
            smooth=self.smooth,
            basis=self.basis,
            x=self.x,
            coef=self.coef,
            tau2=self.tau2,
            K=self.K,
            rank=self.rank,
            log_pdet=self.log_pdet,
        )

        self.mcmc_kernels: list[Kernel] = self._default_kernels(combined_kernels)
        """
        List of :class:`liesel.goose.kernel.Kernel` MCMC kernel classes.
        These kernels are used when setting up a :class:`liesel.goose.EngineBuilder`
        with :meth:`.ptm_ls.PTMLocScale.setup_engine_builder`.
        """

    @classmethod
    def pspline(
        cls,
        x: Array,
        nparam: int,
        tau2: lsl.Var | Var,
        name: str,
        combined_kernels: bool = False,
    ) -> StructuredAdditiveTerm:
        """
        Alternative constructor for quickly setting up a P-spline.

        Parameters
        ----------
        x
            Covariate array.
        nparam
            Number of parameters for the P-spline.
        tau2
            Variance parameter for the singular normal prior.
        name
            Unique name of the term.
        """

        knots = kn(x, order=3, n_params=nparam)
        basis = bspline_basis(x, knots, order=3)

        K = diffpen(nparam)
        Z = sumzero_term(basis)
        Kz = Z.T @ K @ Z

        basis_fn = bs(knots, order=3, Z=Z)

        star = cls(x, basis_fn, Kz, tau2, name, combined_kernels=combined_kernels)
        return star

    def _default_kernels(self, combined_kernels: bool = False) -> list[Kernel]:
        kernels: list[Kernel] = []

        if combined_kernels:
            param_names = [self.coef.name]
            tau2_param = find_param(self.tau2)
            if tau2_param is not None:
                param_names.append(tau2_param.name)

            kernels.append(self._default_kernel(param_names))

            return kernels

        kernels.append(self._default_kernel([self.coef.name]))

        if not self.hyper_parameters:
            return kernels

        if self.tau2.has_dist:
            if self.tau2.dist_node.distribution is tfd.InverseGamma:
                transition_fn = ig_gibbs_transition_fn(self, var_name="tau2")
                name = self.tau2.name
                kernels.append(gs.GibbsKernel([name], transition_fn))
        else:
            tau2_param = find_param(self.tau2)
            if tau2_param is not None:
                kernels.append(gs.NUTSKernel([tau2_param.name]))

        return kernels

    @property
    def value(self) -> Var:
        """Evaluation of this term."""
        return self.smooth

    def predict(
        self,
        samples: dict[str, Array],
        x: Array | None = None,
        center: bool = False,
        scale: bool = False,
    ) -> Array:
        """
        Computes predicted values given an array of posterior samples.

        Can be thought of as the sum of the predictions for the individual terms
        obtained with :meth:`.StructuredAdditiveTerm.predict_elementwise`.

        Parameters
        ----------
        samples
            Array of posterior samples.
        x
            Array of covariate values.
        center
            Whether to center the predictions.
        scale
            Whether to scale the predictions.

        Returns
        -------
        Array of prediction values.
        """
        fx = self.smooth.predict(samples=samples, x=self.basis_fn(x))
        if center:
            fx = fx - fx.mean(axis=-1, keepdims=True)
        if scale:
            fx = fx / fx.std(axis=-1, keepdims=True)
        return fx

    @property
    def observed_value(self) -> Array:
        """Array of observed ``x`` values."""
        return self.x.value

    @property
    def parameters(self) -> list[str]:
        """List of parameter names associated with this term."""
        return self._parameters

    @property
    def hyper_parameters(self) -> list[str]:
        """List of hyperparameter names associated with this term."""
        return self._hyper_parameters

    def summarise_by_quantiles(
        self,
        samples: dict[str, Array],
        x: Array | None = None,
        axis: Sequence[int] | int = (0, 1),
        lo: float = 0.1,
        hi: float = 0.9,
        center: bool = False,
        scale: bool = False,
    ) -> pd.DataFrame:
        """
        Computes a posterior summary for this term based on quantiles.

        Parameters
        ----------
        samples
            Array of posterior samples.
        x
            Array of covariate values.
        axis
            Indicates the axes of ``samples`` that index the posterior samples.
        lo
            Probability level for lower quantile to include in summary.
        hi
            Probability level for upper quantile to include in summary.
        center
            Whether to center the predictions.
        scale
            Whether to scale the predictions.
        indices
            If not None, only the columns of ``x`` with these indices are used.
        """
        x = x if x is not None else self.observed_value
        fx = self.predict(samples, x=x, center=center, scale=scale)

        df = summarise_by_quantiles(fx, axis=axis, lo=lo, hi=hi)

        df["x_value"] = np.asarray(jnp.squeeze(x))
        df["name"] = self.name

        return df

    def summarise_by_samples(
        self,
        key: KeyArray,
        samples: dict[str, Array],
        n: int = 100,
        x: Array | None = None,
        center: bool = False,
        scale: bool = False,
    ) -> pd.DataFrame:
        """
        Computes a posterior summary for this term based on a subsample from the
        posterior.

        Parameters
        ----------
        key
            A ``jax.random.PRNGKey`` for reproducibility.
        samples
            Array of posterior samples.
        n
            The number of samples to draw.
        x
            Array of covariate values.
        center
            Whether to center the predictions.
        scale
            Whether to scale the predictions.
        indices
            If not None, only the columns of ``x`` with these indices are used.
        """
        x = x if x is not None else self.observed_value
        fx = self.predict(samples, x=x, center=center, scale=scale)

        df = summarise_by_samples(key, fx, "value", n=n)

        x = jnp.squeeze(x)
        if jnp.atleast_1d(x).shape[-1] == 1:
            df[self.name] = np.asarray(x)
        else:
            df[self.name] = np.asarray(np.tile(x, n))

        return df


class PSpline(StructuredAdditiveTerm):
    """
    A P-spline with second-order random walk prior.

    Parameters
    ----------
    x
        Covariate array.
    nparam
        Number of parameters for the P-spline.
    tau2
        Variance parameter for the singular normal prior.
    name
        Unique name of the term.
    """

    def __init__(self, x: Array, nparam: int, tau2: lsl.Var | Var, name: str) -> None:
        knots = kn(x, order=3, n_params=nparam)
        basis = bspline_basis(x, knots, order=3)

        K = diffpen(nparam)
        Z = sumzero_term(basis)
        Kz = Z.T @ K @ Z

        basis_fn = bs(knots, order=3, Z=Z)
        super().__init__(x, basis_fn, Kz, tau2, name)


class RandomIntercept(StructuredAdditiveTerm):
    """
    A random intercept with iid normal prior in noncentered parameterization.
    """

    def __init__(self, x: Array, tau: lsl.Var, name: str) -> None:
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(x)
        self.x = lsl.obs(x, name=f"{name}_covariate")
        self.basis_fn = self.label_binarizer.transform
        self.basis = lsl.Data(self.basis_fn(x), _name=f"{name}_basis")
        self.tau = tau
        self.nparam = self.basis.value.shape[-1]

        prior = lsl.Dist(tfd.Normal, loc=0.0, scale=1.0)
        self.coef = lsl.param(np.zeros(self.nparam), prior, name=f"{name}_coef")

        self.smooth = Var(
            ScaledDot(x=self.basis, coef=self.coef, scale=self.tau),
            name=f"{name}_smooth",
        )

        tau_param = find_param(self.tau)
        self._hyper_parameters: list[str] = []
        if tau_param is not None:
            self._hyper_parameters.append(tau_param.name)

        self._parameters: list[str] = [self.coef.name]

        super(StructuredAdditiveTerm, self).__init__(
            name=name,
            smooth=self.smooth,
            basis=self.basis,
            x=self.x,
            coef=self.coef,
            tau=self.tau,
        )

        self._default_kernel = gs.NUTSKernel
        self.mcmc_kernels: list[Kernel] = self._default_kernels()

    def _default_kernels(self, combined_kernels: bool = False) -> list[Kernel]:
        if combined_kernels:
            raise NotImplementedError

        kernels: list[Kernel] = []

        kernels.append(self._default_kernel([self.coef.name]))

        if not self.hyper_parameters:
            return kernels

        tau_param = find_param(self.tau)
        if tau_param is not None:
            kernels.append(gs.NUTSKernel([tau_param.name]))

        return kernels

    @classmethod
    def pspline(
        cls,
        x: Array,
        nparam: int,
        tau2: lsl.Var | Var,
        name: str,
        combined_kernels: bool = False,
    ):
        raise NotImplementedError


class RandomInterceptSumZero(RandomIntercept):
    def __init__(self, x: Array, tau: lsl.Var, name: str) -> None:
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(x)
        self.x = lsl.obs(x, name=f"{name}_covariate")

        nparam = self.label_binarizer.transform(x).shape[-1]
        K = jnp.eye(nparam)
        Z = sumzero_coef(nparam)
        Kz = Z.T @ K @ Z

        self.nparam = Kz.shape[-1]
        self.basis_fn = lambda x: self.label_binarizer.transform(x) @ Z
        self.basis = lsl.Data(self.basis_fn(x), _name=f"{name}_basis")
        self.tau = tau

        self.evals = jnp.linalg.eigvalsh(Kz)
        self.rank = lsl.Data(jnp.sum(self.evals > 0.0), _name=f"{name}_K_rank")
        _log_pdet = jnp.log(jnp.where(self.evals > 0.0, self.evals, 1.0)).sum()
        self.log_pdet = lsl.Data(_log_pdet, _name=f"{name}_K_log_pdet")

        prior = lsl.Dist(
            MultivariateNormalDegenerate.from_penalty,
            loc=0.0,
            var=tau,
            pen=Kz,
            rank=self.rank,
            log_pdet=self.log_pdet,
        )

        self.coef = lsl.param(np.zeros(self.nparam), prior, name=f"{name}_coef")

        self.smooth = Var(
            ScaledDot(x=self.basis, coef=self.coef, scale=self.tau),
            name=f"{name}_smooth",
        )

        tau_param = find_param(self.tau)
        self._hyper_parameters: list[str] = []
        if tau_param is not None:
            self._hyper_parameters.append(tau_param.name)

        self._parameters: list[str] = [self.coef.name]

        super(StructuredAdditiveTerm, self).__init__(
            name=name,
            smooth=self.smooth,
            basis=self.basis,
            x=self.x,
            coef=self.coef,
            tau=self.tau,
        )

        self._default_kernel = gs.NUTSKernel
        self.mcmc_kernels: list[Kernel] = self._default_kernels()


class MISpline(StructuredAdditiveTerm):
    def __init__(
        self,
        x: Array,
        nparam: int,
        tau2: lsl.Var,
        name: str,
        mcmc_kernel: Kernel = gs.NUTSKernel,
    ) -> None:
        self._default_kernel = mcmc_kernel
        self.x = lsl.obs(x, name=f"{name}_covariate")

        knots = kn(x, order=3, n_params=nparam)

        basis = bspline_basis(x, knots=knots, order=3)
        cumsum_matrix = np.tril(np.ones((nparam, nparam)))
        basis_cumsum = basis @ cumsum_matrix
        colmeans = basis_cumsum.mean(axis=0)

        def basis_fn(x):
            basis = bspline_basis(x, knots=knots, order=3)
            cumsum_matrix = np.tril(np.ones((nparam, nparam)))
            basis_cumsum = basis @ cumsum_matrix
            return basis_cumsum - colmeans

        self.basis_fn = jnp.vectorize(basis_fn, signature="(n)->(n,p)")
        self.basis = lsl.Data(basis_fn(x), _name=f"{name}_basis")

        K = diffpen(nparam, diff=1)
        self.K = lsl.Data(K, _name=f"{name}_K")
        self.tau2 = tau2
        self.evals = jnp.linalg.eigvalsh(K)
        self.rank = lsl.Data(jnp.sum(self.evals > 0.0), _name=f"{name}_K_rank")
        _log_pdet = jnp.log(jnp.where(self.evals > 0.0, self.evals, 1.0)).sum()
        self.log_pdet = lsl.Data(_log_pdet, _name=f"{name}_K_log_pdet")

        prior = lsl.Dist(
            MultivariateNormalDegenerate.from_penalty,
            loc=0.0,
            var=tau2,
            pen=self.K,
            rank=self.rank,
            log_pdet=self.log_pdet,
        )
        self.nparam = jnp.shape(K)[-1]
        start_value = jnp.zeros(self.nparam)
        self.coef = lsl.param(start_value, prior, name=f"{name}_coef")
        self.exp_coef = lsl.Var(
            lsl.Calc(jnp.exp, self.coef), name=f"{name}_exp_coef"
        ).update()

        self.smooth = Var(IncDot(self.basis, self.exp_coef), name=f"{name}_smooth")

        tau2_param = find_param(self.tau2)

        self._hyper_parameters: list[str] = []
        if tau2_param is not None:
            self._hyper_parameters.append(tau2_param.name)

        self._parameters: list[str] = [self.coef.name]

        super(StructuredAdditiveTerm, self).__init__(
            name=name,
            smooth=self.smooth,
            basis=self.basis,
            x=self.x,
            coef=self.coef,
            exp_coef=self.exp_coef,
            tau2=self.tau2,
            K=self.K,
            rank=self.rank,
            log_pdet=self.log_pdet,
        )

        self.mcmc_kernels: list[Kernel] = self._default_kernels()


class StrAT(StructuredAdditiveTerm):
    pass


class S(StructuredAdditiveTerm):
    pass


class LinearTerm(Term):
    """
    Linear function of one or more covariates.

    Parameters
    ----------
    x
        Covariate array.
    name
        Unique name of the covariate term.
    prior
        Joint prior distribution for the regression coefficients. If ``None``, uses\
        a constant prior.
    """

    def __init__(self, x: Array, name: str, prior: lsl.Dist | None = None) -> None:
        x = _matrix(x)
        self.x: lsl.Var = lsl.obs(x, name=f"{name}_covariate")
        """Covariate node."""

        self.nparam = self.x.value.shape[1]
        self.coef: lsl.Var = lsl.param(
            np.zeros(self.nparam), prior, name=f"{name}_coef"
        )
        """Regression coefficient node. Initialized with zeros."""
        self.smooth: lsl.Var = Var(Dot(self.x, self.coef), name=f"{name}_smooth")
        """Node for the evaluated linear term ``x @ coef``."""
        self._nuts_params = [self.coef.name]
        super().__init__(name=name, smooth=self.smooth, x=self.x, coef=self.coef)

        self.mcmc_kernels: list[Kernel] = []
        """
        List of :class:`liesel.goose.kernel.Kernel` MCMC kernel classes.
        These kernels are used when setting up a :class:`liesel.goose.EngineBuilder`
        with :meth:`.ptm_ls.PTMLocScale.setup_engine_builder`.
        """
        self.mcmc_kernels.append(gs.NUTSKernel([self.coef.name]))

        self._parameters: list[str] = [self.coef.name]
        self._hyper_parameters: list[str] = []

    @property
    def nuts_params(self) -> list[str]:
        warnings.warn(
            "nuts_params is deprecated. Use parameters and hyper_parameters instead.",
            FutureWarning,
        )
        return self._nuts_params

    @property
    def parameters(self) -> list[str]:
        """List of parameter names associated with this term."""
        return self._parameters

    @property
    def hyper_parameters(self) -> list[str]:
        """List of hyperparameter names associated with this term."""
        return self._hyper_parameters

    @property
    def value(self) -> Var:
        """Evaluation of this term."""
        return self.smooth

    def predict(
        self,
        samples: dict[str, Array],
        x: Array | None = None,
        center: bool = False,
        scale: bool = False,
    ) -> Array:
        """
        Computes predicted values given an array of posterior samples.

        Can be thought of as the sum of the predictions for the individual terms
        obtained with :meth:`.predict_elementwise`.

        Parameters
        ----------
        samples
            Array of posterior samples.
        x
            Array of covariate values.
        center
            Whether to center the predictions.
        scale
            Whether to scale the predictions.

        Returns
        -------
        Array of prediction values.
        """
        fx = self.smooth.predict(samples=samples, x=x)
        if center:
            fx = fx - fx.mean(axis=-1, keepdims=True)
        if scale:
            fx = fx / fx.std(axis=-1, keepdims=True)
        return fx

    def predict_elementwise(
        self,
        samples: dict[str, Array],
        x: Array | None = None,
        indices: Sequence[int] | None = None,
        center: bool = False,
        scale: bool = False,
    ) -> Array:
        """
        Computes predicted values given an array of posterior samples.

        Can be thought of as the individual terms in :meth:`.predict`.

        Parameters
        ----------
        samples
            Array of posterior samples.
        x
            Array of covariate values.
        indices
            Sequence of indices, indicating which elements of ``x`` to return \
            predictions for. If ``None`` predictions are returned for all covariates.
        center
            Whether to center the predictions.
        scale
            Whether to scale the predictions.

        Returns
        -------
        Array of prediction values.
        """
        x = x if x is not None else self.observed_value
        x = _matrix(x)

        if indices is not None:
            if not len(indices):
                raise ValueError("indices must be None or a sequence of integers.")
            x = x[..., indices]
            samples[self.coef.name] = samples[self.coef.name][..., indices]

        fx = self.smooth.predict_elementwise(samples=samples, x=x)

        if center:
            fx = fx - fx.mean(axis=-2, keepdims=True)
        if scale:
            fx = fx / fx.std(axis=-2, keepdims=True)

        return fx

    def summarise_by_quantiles(
        self,
        samples: dict[str, Array],
        x: Array | None = None,
        axis: Sequence[int] | int = (0, 1),
        lo: float = 0.1,
        hi: float = 0.9,
        center: bool = False,
        scale: bool = False,
        indices: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """
        Computes a posterior summary for this term based on quantiles.

        Parameters
        ----------
        samples
            Array of posterior samples.
        x
            Array of covariate values.
        axis
            Indicates the axes of ``samples`` that index the posterior samples.
        lo
            Probability level for lower quantile to include in summary.
        hi
            Probability level for upper quantile to include in summary.
        center
            Whether to center the predictions.
        scale
            Whether to scale the predictions.
        indices
            If not None, only the columns of ``x`` with these indices are used.
        """
        x = x if x is not None else self.observed_value
        x = _matrix(x)

        if indices is not None:
            if not len(indices):
                raise ValueError("indices must be None or a sequence of integers.")
            x = x[..., indices]
            samples[self.coef.name] = samples[self.coef.name][..., indices]

        fx = self.predict_elementwise(samples, x=x, center=center, scale=scale)

        df_list = []
        for k in range(x.shape[-1]):
            df = summarise_by_quantiles(fx[..., k], axis=axis, lo=lo, hi=hi)

            df["x_value"] = x[..., k]
            df["name"] = self.name
            df["id"] = f"{self.name}[{k}]"
            df["id_index"] = k
            df_list.append(df)

        return pd.concat(df_list).reset_index(drop=True)

    def summarise_by_samples(
        self,
        key: KeyArray,
        samples: dict[str, Array],
        n: int = 100,
        x: Array | None = None,
        center: bool = False,
        scale: bool = False,
        indices: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """
        Computes a posterior summary for this term based on a subsample from the
        posterior.

        Parameters
        ----------
        key
            A ``jax.random.PRNGKey`` for reproducibility.
        samples
            Array of posterior samples.
        n
            The number of samples to draw.
        x
            Array of covariate values.
        center
            Whether to center the predictions.
        scale
            Whether to scale the predictions.
        indices
            If not None, only the columns of ``x`` with these indices are used.
        """
        x = x if x is not None else self.observed_value
        x = _matrix(x)

        if indices is not None:
            if not len(indices):
                raise ValueError("indices must be None or a sequence of integers.")
            x = x[..., indices]
            samples[self.coef.name] = samples[self.coef.name][..., indices]

        fx = self.predict_elementwise(samples, x=x, center=center, scale=scale)

        df_list = []
        for k in range(x.shape[-1]):
            key, subkey = jax.random.split(key)
            df = summarise_by_samples(subkey, fx[..., k], "fx_value", n=n)

            if jnp.atleast_1d(x[..., k]).shape[-1] == 1:
                df["x_value"] = x[..., k]
            else:
                df["x_value"] = np.tile(x[..., k], n)

            df["name"] = self.name
            df["id"] = f"{self.name}[{k}]"
            df["id_index"] = k
            df_list.append(df)

        return pd.concat(df_list).reset_index(drop=True)

    @property
    def observed_value(self) -> Array:
        """Array of observed ``x`` values."""
        return self.x.value


class LinearSmooth(LinearTerm):
    """Alias for :class:`.LinearTerm` for backwards-compatibility."""

    pass


class Intercept(Term):
    def __init__(self, name: str, distribution: lsl.Dist | None = None) -> None:
        self.intercept = lsl.Var(value=0.0, distribution=distribution, name=name)
        self._nuts_params = [self.intercept.name]
        super().__init__(name=name, intercept=self.intercept)

    @property
    def nuts_params(self) -> list[str]:
        warnings.warn(
            "nuts_params is deprecated. Use parameters and hyper_parameters instead.",
            FutureWarning,
        )
        return self._nuts_params

    @property
    def hyper_parameters(self) -> list[str]:
        return []

    @property
    def parameters(self) -> list[str]:
        return [self.intercept.name]

    @property
    def value(self) -> Var:
        return self.intercept

    def predict(self, samples: dict[str, Array], x: Array | None = None) -> Array:
        return samples[self.name]

    @property
    def observed_value(self) -> Array:
        return jnp.array(1.0)


def addition(*args, **kwargs):
    return sum(args) + sum(kwargs.values())


class Predictor(lsl.Var):
    """
    Organizes the terms of a structured additive predictor.

    Parameters
    ----------
    name
        Name of the predictor.

    See Also
    --------
    .StructuredAdditiveTerm : Term in a structured additive predictor.
    .LinearTerm : Linear term.
    .RandomIntercept : Random intercept term.

    Examples
    --------

    Terms must be added to the predictor using the ``+=`` syntax::

        import liesel_ptm as ptm
        import numpy as np

        x = np.linspace(size=10)

        predictor = ptm.Predictor(name="pred")
        predictor += ptm.LinearSmooth(x, name="term1")

    Terms can be accessed using both bracket- and dot-syntax, using the names of
    the terms::

        predictor["term1"]
        predictor.term1

    """

    def __init__(self, name: str):
        super().__init__(lsl.Calc(addition), name=name)
        self.terms: dict[str, Term] = {}
        """Dictionary of terms in this predictor."""
        self.intercept: Intercept | None = None
        """Intercept variable (if any)."""

    @classmethod
    def with_intercept(cls, name: str):
        """Alternative constructor for initializing the predictor with an intercept."""
        predictor = cls(name)
        intercept = Intercept(f"{name}_intercept")
        return predictor + intercept

    def __add__(self, other: Term) -> Predictor:
        self.value_node.add_inputs(other.value)
        self.terms[other.name] = other
        if isinstance(other, Intercept):
            if self.intercept is not None:
                raise ValueError(f"Intercept already present on {self}.")
            self.intercept = other
        return self.update()

    def __iadd__(self, other: Term) -> Predictor:
        self.value_node.add_inputs(other.value)
        self.terms[other.name] = other
        if isinstance(other, Intercept):
            if self.intercept is not None:
                raise ValueError(f"Intercept already present on {self}.")
            self.intercept = other
        return self.update()

    def __getitem__(self, name) -> Term:
        return self.terms[name]

    def __getattr__(self, name) -> Term:
        if name.startswith("__"):  # ensures, for example, that copying works.
            raise AttributeError
        try:
            return self.terms[name]
        except KeyError:
            raise AttributeError(name)

    @property
    def nuts_params(self) -> list[str]:
        """
        (Deprecated) List of parameters to be sampled via the No-U-Turn Sampler (NUTS).
        """
        warnings.warn(
            "nuts_params is deprecated. Use parameters and hyper_parameters instead.",
            FutureWarning,
        )
        param_generator = (term.nuts_params for term in self.terms.values())
        return list(chain.from_iterable(param_generator))

    @property
    def parameters(self) -> list[str]:
        """
        Collects the names of the parameters in this predictor and returns them in a
        list.
        """
        param_generator = (term.parameters for term in self.terms.values())
        return list(chain.from_iterable(param_generator))

    @property
    def hyper_parameters(self) -> list[str]:
        """
        Collects the names of the hyperparameters in this predictor and returns them in
        a list.
        """
        param_generator = (term.hyper_parameters for term in self.terms.values())
        return list(chain.from_iterable(param_generator))

    def predict(self, samples: dict[str, Array], **kwargs) -> Array:
        """
        Returns predicted values for this predictor.

        Parameters
        ----------
        samples
            Dictionary with arrays of posterior samples.
        **kwargs
            Covariate values at which to evaluate the predictor. Supplied in the form
            ``name=value``, where ``name`` is the name of a term in the predictor,
            and ``value`` is the covariate value at which this term should be evaluated
            to enter the predictor.

        Returns
        -------
        Array of predicted values.
        """
        term_predictions = []
        for name, term in self.terms.items():
            if self.intercept is not None and name == self.intercept.name:
                continue
            prediction = term.predict(samples, x=kwargs.pop(name))
            term_predictions.append(prediction)

        if self.intercept is not None:
            term_predictions.append(np.atleast_3d(samples[self.intercept.name]))

        return sum(term_predictions)


def ig_gibbs_transition_fn(
    group: lsl.Group, var_name: str = "tau2"
) -> Callable[[KeyArray, gs.ModelState], dict[str, Array]]:
    """
    Gibbs transition function for an inverse smoothing parameter with an
    inverse gamma prior.
    """

    var = group[var_name]

    a = var.dist_node.kwinputs["concentration"]
    b = var.dist_node.kwinputs["scale"]

    for hyper_param in [a, b]:
        try:
            if hyper_param.parameter:
                raise ValueError(
                    f"{hyper_param} is marked as parameter, but this function assumes"
                    " it to be fixed."
                )
        except AttributeError:
            pass

    a_prior = a.value
    b_prior = b.value

    K = group["K"].value
    rank = group["rank"].value

    a_gibbs = jnp.squeeze(a_prior + 0.5 * rank)

    assert group["coef"]

    def transition(prng_key, model_state):
        coef = group.value_from(model_state, "coef")
        b_gibbs = jnp.squeeze(b_prior + 0.5 * (coef @ K @ coef))

        draw = b_gibbs / jax.random.gamma(prng_key, a_gibbs)

        return {var.name: draw}

    return transition


class VarInverseGamma(Var):
    """
    A variable with an inverse gamma prior.

    Parameters
    ----------
    value
        Initial value of the variable.
    concentration
        Concentration parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``a``.
    scale
        Scale parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``b``.
    name
        Name of the variable.
    bijector
        A tensorflow bijector instance.\
        If a bijector is supplied, the variable will be transformed using the bijector.\
        This renders the variable itself weak, meaning that it is a deterministic\
        function of the newly created transformed variable. The prior is transferred\
        to this transformed variable and transformed according to the \
        change-of-variables theorem.

    """

    def __init__(
        self,
        value: float,
        concentration: float | lsl.Var | lsl.Node,
        scale: float | lsl.Var | lsl.Node,
        name: str,
        bijector: tfb.Bijector | None = tfb.Softplus(),
    ) -> None:
        try:
            scale.name = f"{name}_scale"  # type: ignore
        except AttributeError:
            scale = lsl.Data(scale, _name=f"{name}_scale")

        try:
            concentration.name = f"{name}_concentration"  # type: ignore
        except AttributeError:
            concentration = lsl.Data(concentration, _name=f"{name}_concentration")

        prior = lsl.Dist(tfd.InverseGamma, concentration=concentration, scale=scale)
        super().__init__(value, prior, name=name)
        self.parameter = True
        self.update()

        val = self.value
        log_prob = self.log_prob

        self.transformed: Var | None = None
        """The transformed variable (if any)."""
        if bijector is not None:
            self.transformed = self.transform(bijector)
            self.transformed.update()
            self.update()

            val = self.transformed.value
            log_prob = self.transformed.log_prob

        assert jnp.isfinite(val)
        assert jnp.isfinite(log_prob)

    def predict(self, samples: dict[str, Array]) -> Array:
        """
        Returns posterior samples of this variable. If the variable is transformed
        using a bijector, this method automatically takes care of applying the
        bijector to the transformed samples.

        Parameters
        ----------
        samples
            Dictionary with arrays of posterior samples.

        Returns
        -------
        Array of posterior samples.
        """
        if self.transformed is not None:
            return self.value_node.function(samples[self.transformed.name])

        return samples[self.name]


class ScaleInverseGamma(Var):
    """
    A variable with an inverse gamma prior on its square.

    Parameters
    ----------
    value
        Initial value of the variable.
    concentration
        Concentration parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``a``.
    scale
        Scale parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``b``.
    name
        Name of the variable.
    bijector
        A tensorflow bijector instance.\
        If a bijector is supplied, the variable will be transformed using the bijector.\
        This renders the variable itself weak, meaning that it is a deterministic\
        function of the newly created transformed variable. The prior is transferred\
        to this transformed variable and transformed according to the \
        change-of-variables theorem.

    Notes
    -----
    The variable itself is weak, meaning that it is always defined as a deterministic
    function of a random variable, in this case the square root. This random variable
    is available as :attr:`.variance_param`. If a bijector is supplied, this
    bijector is applied to :attr:`.variance_param`, such that :attr:`.variance_param`
    becomes weak, too, and the random variable is given by :attr:`.transformed`.
    """

    def __init__(
        self,
        value: float,
        concentration: float | lsl.Var | lsl.Node,
        scale: float | lsl.Var | lsl.Node,
        name: str,
        bijector: tfb.Bijector | None = tfb.Softplus(),
    ) -> None:
        if isinstance(scale, float):
            scale = lsl.Data(scale, _name=f"{name}_scale")
        else:
            scale.name = f"{name}_scale"

        if isinstance(concentration, float):
            concentration = lsl.Data(concentration, _name=f"{name}_concentration")
        else:
            concentration.name = f"{name}_concentration"

        prior = lsl.Dist(tfd.InverseGamma, concentration=concentration, scale=scale)
        self.variance_param = Var(value, prior, name=f"{name}2")
        """Variance parameter node."""
        self.variance_param.parameter = True
        super().__init__(lsl.Calc(jnp.sqrt, self.variance_param), name=name)
        self.transformed = None
        """The transformed variable (if any)."""
        self.update()

        val = self.variance_param.value
        log_prob = self.variance_param.log_prob

        if bijector is not None:
            self.transformed = self.variance_param.transform(bijector)
            self.transformed.update()
            self.variance_param.update()
            self.update()

            val = self.transformed.value
            log_prob = self.transformed.log_prob

        assert jnp.isfinite(val)
        assert jnp.isfinite(log_prob)

    def predict(self, samples: dict[str, Array]) -> Array:
        """
        Returns posterior samples of this variable. If the variance parameter
        is transformed using a bijector, this method automatically takes care of
        applying the bijector to the transformed samples.

        Parameters
        ----------
        samples
            Dictionary with arrays of posterior samples.

        Returns
        -------
        Array of posterior samples.
        """
        if self.transformed is not None:
            samps = samples[self.transformed.name]
            variance_samples = self.variance_param.value_node.function(samps)
            return self.value_node.function(variance_samples)

        variance_samples = samples[self.variance_param.name]
        return self.value_node.function(variance_samples)


class VarHalfCauchy(Var):
    """
    A variance parameter with a half Cauchy prior on its square root.

    Parameters
    ----------
    value
        Initial value of the variable.
    scale
        Scale parameter of the half Cauchy distribution.
    name
        Name of the variable.
    bijector
        A tensorflow bijector instance.\
        If a bijector is supplied, the variable will be transformed using the bijector.\
        This renders the variable itself weak, meaning that it is a deterministic\
        function of the newly created transformed variable. The prior is transferred\
        to this transformed variable and transformed according to the \
        change-of-variables theorem.

    Notes
    -----
    Note that the half Cauchy prior is placed not directly on the variance parameter,
    but on the standard deviation, i.e. the square root of the variance.
    Futher, the standard deviation is transformed to the positive real line for
    MCMC sampling using a bijector, which defaults to softplus.
    """

    def __init__(
        self,
        value: Array,
        scale: float | lsl.Var | lsl.Node,
        name: str,
        bijector: tfb.Bijector | None = tfb.Softplus(),
    ) -> None:
        if isinstance(scale, float):
            scale = lsl.Data(scale, _name=f"{name}_scale")
        else:
            scale.name = f"{name}_scale"

        loc = lsl.Data(0.0, _name=f"{name}_loc")

        prior = lsl.Dist(tfd.HalfCauchy, loc=loc, scale=scale)

        self.scale_param = Var(jnp.sqrt(value), prior, name=f"{name}_root")
        """The scale parameter (if any)."""

        self.scale_param.parameter = True

        self.transformed = None
        """The transformed variable (if any)."""

        var_calc = lsl.Calc(jnp.square, self.scale_param)
        super().__init__(var_calc, name=name)

        self.update()

        val = self.scale_param.value
        log_prob = self.scale_param.log_prob

        if bijector is not None:
            self.transformed = self.scale_param.transform(bijector)
            self.transformed.update()
            self.scale_param.update()
            self.update()
            val = self.transformed.value
            log_prob = self.transformed.log_prob

        assert jnp.isfinite(val)
        assert jnp.isfinite(log_prob)

    def predict(self, samples: dict[str, Array]) -> Array:
        """
        Returns posterior samples of this variable. If the scale variable is transformed
        using a bijector, this method automatically takes care of applying the
        bijector to the transformed samples.

        Parameters
        ----------
        samples
            Dictionary with arrays of posterior samples.

        Returns
        -------
        Array of posterior samples.
        """
        if self.transformed is not None:
            samps = samples[self.transformed.name]
            scale_samples = self.scale_param.value_node.function(samps)
            return self.value_node.function(scale_samples)

        scale_samples = samples[self.scale_param.name]
        return self.value_node.function(scale_samples)


class VarHalfNormal(Var):
    """
    A variance parameter with a half normal prior on its square root.

    Parameters
    ----------
    value
        Initial value of the variable.
    scale
        Scale parameter of the half Cauchy distribution.
    name
        Name of the variable.
    bijector
        A tensorflow bijector instance.\
        If a bijector is supplied, the variable will be transformed using the bijector.\
        This renders the variable itself weak, meaning that it is a deterministic\
        function of the newly created transformed variable. The prior is transferred\
        to this transformed variable and transformed according to the \
        change-of-variables theorem.

    Notes
    -----
    Note that the half normal prior is placed not directly on the variance parameter,
    but on the standard deviation, i.e. the square root of the variance.
    Futher, the standard deviation is transformed to the positive real line for
    MCMC sampling using a bijector, which defaults to softplus.
    """

    def __init__(
        self,
        value: Array,
        scale: float | lsl.Var | lsl.Node,
        name: str,
        low: float | lsl.Var | lsl.Node = 0.0,
        bijector: tfb.Bijector | None = tfb.Softplus(),
    ) -> None:
        if isinstance(scale, float):
            scale = lsl.Data(scale, _name=f"{name}_scale")
        else:
            scale.name = f"{name}_scale"

        if isinstance(low, float):
            low = lsl.Data(low, _name=f"{name}_low")
        else:
            low.name = f"{name}_low"

        loc = lsl.Data(0.0, _name=f"{name}_loc")

        prior = lsl.Dist(
            tfd.TruncatedNormal, loc=loc, scale=scale, low=low, high=jnp.inf
        )

        self.scale_param = Var(jnp.sqrt(value), prior, name=f"{name}_root")
        """The scale parameter (if any)."""

        self.scale_param.parameter = True

        self.transformed = None
        """The transformed variable (if any)."""

        var_calc = lsl.Calc(jnp.square, self.scale_param)
        super().__init__(var_calc, name=name)

        self.update()

        val = self.scale_param.value
        log_prob = self.scale_param.log_prob

        if bijector is not None:
            self.transformed = self.scale_param.transform(bijector)
            self.transformed.update()
            self.scale_param.update()
            self.update()
            val = self.transformed.value
            log_prob = self.transformed.log_prob

        assert jnp.isfinite(val)
        assert jnp.isfinite(log_prob)

    def predict(self, samples: dict[str, Array]) -> Array:
        """
        Returns posterior samples of this variable. If the scale variable is transformed
        using a bijector, this method automatically takes care of applying the
        bijector to the transformed samples.

        Parameters
        ----------
        samples
            Dictionary with arrays of posterior samples.

        Returns
        -------
        Array of posterior samples.
        """
        if self.transformed is not None:
            samps = samples[self.transformed.name]
            scale_samples = self.scale_param.value_node.function(samps)
            return self.value_node.function(scale_samples)

        scale_samples = samples[self.scale_param.name]
        return self.value_node.function(scale_samples)


class ScaleHalfCauchy(Var):
    """
    A scale parameter with a half Cauchy prior.

    Parameters
    ----------
    value
        Initial value of the variable.
    scale
        Scale parameter of the half Cauchy distribution.
    name
        Name of the variable.
    bijector
        A tensorflow bijector instance.\
        If a bijector is supplied, the variable will be transformed using the bijector.\
        This renders the variable itself weak, meaning that it is a deterministic\
        function of the newly created transformed variable. The prior is transferred\
        to this transformed variable and transformed according to the \
        change-of-variables theorem.
    """

    def __init__(
        self,
        value: Array,
        scale: float | lsl.Var | lsl.Node,
        name: str,
        bijector: tfb.Bijector | None = tfb.Softplus(),
    ) -> None:
        if isinstance(scale, float):
            scale = lsl.Data(scale, _name=f"{name}_scale")
        else:
            scale.name = f"{name}_scale"

        loc = lsl.Data(0.0, _name=f"{name}_loc")
        prior = lsl.Dist(tfd.HalfCauchy, loc=loc, scale=scale)

        super().__init__(value, prior, name=name)
        self.parameter = True
        self.transformed = None
        """The transformed variable (if any)."""

        val = self.value
        log_prob = self.log_prob

        if bijector is not None:
            self.transformed = self.transform(bijector)
            self.transformed.update()
            self.update()

            val = self.transformed.value
            log_prob = self.transformed.log_prob

        assert jnp.isfinite(val)
        assert jnp.isfinite(log_prob)

    def predict(self, samples: dict[str, Array]) -> Array:
        """
        Returns posterior samples of this variable. If the variable is transformed
        using a bijector, this method automatically takes care of applying the
        bijector to the transformed samples.

        Parameters
        ----------
        samples
            Dictionary with arrays of posterior samples.

        Returns
        -------
        Array of posterior samples.
        """
        if self.transformed is not None:
            return self.value_node.function(samples[self.transformed.name])

        return samples[self.name]


class VarWeibull(Var):
    """
    A variable with a Weibull prior.

    Parameters
    ----------
    value
        Initial value of the variable.
    concentration
        Concentration parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``a``.
    scale
        Scale parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``b``.
    name
        Name of the variable.
    bijector
        A tensorflow bijector instance.\
        If a bijector is supplied, the variable will be transformed using the bijector.\
        This renders the variable itself weak, meaning that it is a deterministic\
        function of the newly created transformed variable. The prior is transferred\
        to this transformed variable and transformed according to the \
        change-of-variables theorem.

    """

    def __init__(
        self,
        value: Array,
        scale: float | lsl.var | lsl.Node,
        name: str,
        bijector: tfb.Bijector | None = tfb.Softplus(),
    ):
        if isinstance(scale, float):
            scale = lsl.Data(scale, _name=f"{name}_scale")
        else:
            scale.name = f"{name}_scale"
        concentration = lsl.Data(0.5, _name=f"{name}_concentration")
        prior = lsl.Dist(tfd.Weibull, concentration=concentration, scale=scale)
        super().__init__(value, prior, name=name)

        val = self.value
        log_prob = self.log_prob

        self.parameter = True
        self.transformed = None
        """The transformed variable (if any)."""
        if bijector is not None:
            self.transformed = self.transform(bijector)
            self.transformed.update()
            self.update()

            val = self.transformed.value
            log_prob = self.transformed.log_prob

        assert jnp.isfinite(val)
        assert jnp.isfinite(log_prob)

    def predict(self, samples: dict[str, Array]) -> Array:
        """
        Returns posterior samples of this variable. If the variable is transformed
        using a bijector, this method automatically takes care of applying the
        bijector to the transformed samples.

        Parameters
        ----------
        samples
            Dictionary with arrays of posterior samples.

        Returns
        -------
        Array of posterior samples.
        """
        if self.transformed is not None:
            return self.value_node.function(samples[self.transformed.name])

        return samples[self.name]


class ScaleWeibull(Var):
    """
    A variable with a Weibull prior on its square.

    Parameters
    ----------
    value
        Initial value of the variable.
    concentration
        Concentration parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``a``.
    scale
        Scale parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``b``.
    name
        Name of the variable.
    bijector
        A tensorflow bijector instance.\
        If a bijector is supplied, the variable will be transformed using the bijector.\
        This renders the variable itself weak, meaning that it is a deterministic\
        function of the newly created transformed variable. The prior is transferred\
        to this transformed variable and transformed according to the \
        change-of-variables theorem.

    Notes
    -----
    The variable itself is weak, meaning that it is always defined as a deterministic
    function of a random variable, in this case the square root. This random variable
    is available as :attr:`.variance_param`. If a bijector is supplied, this
    bijector is applied to :attr:`.variance_param`, such that :attr:`.variance_param`
    becomes weak, too, and the random variable is given by
    :attr:`.ScaleWeibull.transformed`.
    """

    def __init__(
        self,
        value: Array,
        scale: float | lsl.Var | lsl.Node,
        name: str,
        bijector: tfb.Bijector | None = tfb.Softplus(),
    ):
        if isinstance(scale, float):
            scale = lsl.Data(scale, _name=f"{name}_scale")
        else:
            scale.name = f"{name}_scale"
        concentration = lsl.Data(0.5, _name=f"{name}_concentration")
        prior = lsl.Dist(tfd.Weibull, concentration=concentration, scale=scale)
        self.variance_param = Var(value, prior, name=f"{name}2")
        self.variance_param.parameter = True
        super().__init__(lsl.Calc(jnp.sqrt, self.variance_param), name=name)
        self.update()

        self.transformed = None

        val = self.variance_param.value
        log_prob = self.variance_param.log_prob

        if bijector is not None:
            self.transformed = self.variance_param.transform(bijector)
            self.transformed.update()
            self.variance_param.update()
            self.update()

            val = self.transformed.value
            log_prob = self.transformed.log_prob

        assert jnp.isfinite(val)
        assert jnp.isfinite(log_prob)

    def predict(self, samples: dict[str, Array]) -> Array:
        """
        Returns posterior samples of this variable. If the variance parameter
        is transformed using a bijector, this method automatically takes care of
        applying the bijector to the transformed samples.

        Parameters
        ----------
        samples
            Dictionary with arrays of posterior samples.

        Returns
        -------
        Array of posterior samples.
        """
        if self.transformed is not None:
            samps = samples[self.transformed.name]
            variance_samples = self.variance_param.value_node.function(samps)
            return self.value_node.function(variance_samples)

        variance_samples = samples[self.variance_param.name]
        return self.value_node.function(variance_samples)


# ----------------------------------------------------------------------
# Shape parameter
# ----------------------------------------------------------------------


def cumsum_leading_zero(exp_shape: Array) -> Array:
    """Cumulative sum with a leading zero."""
    zeros_shape = jnp.shape(exp_shape)[:-1] + (1,)
    exp_shape = jnp.concatenate((jnp.zeros(zeros_shape), exp_shape), axis=-1)
    return jnp.cumsum(exp_shape, axis=-1)


def sfn(exp_shape):
    order = 3
    p = jnp.shape(exp_shape)[-1] + 1

    outer_border = exp_shape[..., jnp.array([0, -1])] / 6
    inner_border = 5 * exp_shape[..., jnp.array([1, -2])] / 6
    middle = exp_shape[..., 2:-2]
    summed_exp_shape = (
        outer_border.sum(axis=-1, keepdims=True)
        + inner_border.sum(axis=-1, keepdims=True)
        + middle.sum(axis=-1, keepdims=True)
    )

    return (1 / (p - order)) * summed_exp_shape


def normalization_coef(shape: Array, dknots: Array) -> Array:
    """
    Constructs the spline coefficients sucht that the average slope over the domain
    is one.
    """
    exp_shape = jnp.exp(shape)
    cumsum_exp_shape = cumsum_leading_zero(exp_shape)
    coef = (dknots / sfn(exp_shape)) * cumsum_exp_shape
    return coef


class ShapeParam(lsl.Var):
    def __init__(self, nparam: int, scale: ExpParam, name: str = "") -> None:
        pen = diffpen(nparam, diff=1)
        Z = sumzero_coef(nparam)
        Ltinv = cholesky_ltinv(Z.T @ pen @ Z)

        nparam_reparameterized = Ltinv.shape[-1]

        prior = lsl.Dist(tfd.Normal, loc=0.0, scale=1.0)
        self.shape_reparam = lsl.param(
            np.zeros(nparam_reparameterized), prior, name=f"{name}_transformed"
        )
        self.transformation_matrix = Z @ Ltinv
        self.shape_calc = ScaledDot(
            x=lsl.Data(self.transformation_matrix, _name="Z_Ltinv"),
            coef=self.shape_reparam,
            scale=scale,
        )

        super().__init__(self.shape_calc, name=name)
        self.transformed_name = self.shape_reparam.name


class PSplineCoef(lsl.Var):
    def __init__(self, nparam: int, tau2: Var, diff: int = 2, name: str = "") -> None:
        pen = diffpen(nparam, diff=diff)
        Z = sumzero_coef(nparam)
        pen_z = Z.T @ pen @ Z

        nparam_reparameterized = pen_z.shape[-1]

        prior = lsl.Dist(
            MultivariateNormalDegenerate.from_penalty,
            loc=0.0,
            var=tau2,
            pen=pen_z,
        )

        self.shape_reparam = lsl.param(
            np.zeros(nparam_reparameterized, dtype=np.float32),
            prior,
            name=f"{name}_transformed",
        )
        self.transformation_matrix = Z

        self.shape_calc = Dot(
            x=lsl.Data(self.transformation_matrix, _name="Z"),
            coef=self.shape_reparam,
        )

        super().__init__(self.shape_calc, name=name)
        self.transformed_name = self.shape_reparam.name


class PSplineCoefUnconstrained(lsl.Var):
    """Alternative for the ``PSplineCoef`` shape parameter variable."""

    def __init__(self, nparam: int, tau2: Var, diff: int = 2, name: str = "") -> None:
        pen = diffpen(nparam, diff=diff)

        prior = lsl.Dist(
            MultivariateNormalDegenerate.from_penalty,
            loc=0.0,
            var=tau2,
            pen=pen,
        )

        self.shape_reparam = lsl.param(
            np.zeros(nparam), prior, name=f"{name}_transformed"
        )
        self.transformation_matrix = np.eye(nparam)

        self.shape_calc = Dot(
            x=lsl.Data(self.transformation_matrix, _name="Z"),
            coef=self.shape_reparam,
        )

        super().__init__(self.shape_calc, name=name)
        self.transformed_name = self.shape_reparam.name


class NormalCoef(lsl.Var):
    """Alternative for the ``PSplineCoef`` shape parameter variable."""

    def __init__(self, nparam: int, tau2: Var, name: str = "") -> None:
        pen = jnp.eye(nparam)
        nparam_reparameterized = pen.shape[-1]

        prior = lsl.Dist(
            MultivariateNormalDegenerate.from_penalty,
            loc=0.0,
            var=tau2,
            pen=pen,
        )

        self.shape_reparam = lsl.param(
            np.zeros(nparam_reparameterized), prior, name=f"{name}_transformed"
        )
        self.transformation_matrix = pen

        self.shape_calc = Dot(
            x=lsl.Data(self.transformation_matrix, _name="Z"),
            coef=self.shape_reparam,
        )

        super().__init__(self.shape_calc, name=name)
        self.transformed_name = self.shape_reparam.name


class SymmetricallyBoundedScalar(Var):
    """
    Class for defining a scalar :math:`\\omega` with symmetrically bounded support
    around 1.

    With this class, the support of this scalar is :math:`\\omega \\in [1 - d, 1 + d]`.
    The scalar itself is specified as :math:`\\omega = 1 + d(1 - 2\\nu)`,
    where :math:`\\nu \\sim \\text{Beta}(a, b)`. This ensures that the boundaries of
    support are enforced.

    Parameters
    ----------
    allowed_dist_from_one
        The allowed distance from 1, which is written as :math:`d` in the description\
        above.
    name
        Name of the variable.
    a, b
        Parameters :math:`a, b` of the beta prior for the variable :math:`\\nu`.
    bijector
        A tensorflow bijector instance.\
        If a bijector is supplied, the variable :math:`\\nu` will be transformed using\
        the bijector.\
        This renders the variable itself weak, meaning that it is a deterministic\
        function of the newly created transformed variable. The prior is transferred\
        to this transformed variable and transformed according to the \
        change-of-variables theorem.

    """

    def __init__(
        self,
        allowed_dist_from_one: float,
        name: str = "",
        a: float | lsl.Var | lsl.Node = 15.0,
        b: float | lsl.Var | lsl.Node = 15.0,
        bijector: tfb.Bijector | None = tfb.Sigmoid(),
    ) -> None:
        if isinstance(a, float):
            a = lsl.Data(a, _name=f"{name}_a")
        else:
            a.name = f"{name}_a"

        if isinstance(b, float):
            b = lsl.Data(b, _name=f"{name}_b")
        else:
            b.name = f"{name}_b"

        self.a = a
        self.b = b
        self.allowed_dist_from_one = allowed_dist_from_one
        prior = lsl.Dist(tfd.Beta, concentration1=a, concentration0=b)
        self.unit_var = Var(0.5, prior, name=f"{name}_unit")
        """The variable :math:`\\nu \\sim \\text{Beta}(a, b)`."""
        self.unit_var.parameter = True

        def compute_scaling_factor(unit_var):
            return 1.0 + allowed_dist_from_one * (1 - 2 * unit_var)

        super().__init__(lsl.Calc(compute_scaling_factor, self.unit_var), name=name)

        self.transformed = None
        """The transformed variable (if any)."""

        if bijector is not None:
            self.transformed = self.unit_var.transform(bijector)
            self.unit_var.update()

        self.update()

    def predict(self, samples: dict[str, Array]) -> Array:
        """
        Returns posterior samples of this variable. If the variable is transformed
        using a bijector, this method automatically takes care of applying the
        bijector to the transformed samples.

        Parameters
        ----------
        samples
            Dictionary with arrays of posterior samples.

        Returns
        -------
        Array of posterior samples.
        """
        if self.transformed is not None:
            samps = samples[self.transformed.name]
            unit_samples = self.unit_var.value_node.function(samps)
        else:
            unit_samples = samples[self.unit_var.name]

        return self.value_node.function(unit_samples)

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}({self.allowed_dist_from_one}, a={self.a.value},"
            f" b={self.b.value})"
        )


class TransformedVar(Var):
    """
    Class for defining a possibly transformed variable.

    Parameters
    ----------
    value
        The value of the variable.
    prior
        The probability distribution of the variable.
    name
        Name of the variable.
    bijector
        A tensorflow bijector instance.\
        If a bijector is supplied, the variable :math:`\\nu` will be transformed using\
        the bijector.\
        This renders the variable itself weak, meaning that it is a deterministic\
        function of the newly created transformed variable. The prior is transferred\
        to this transformed variable and transformed according to the \
        change-of-variables theorem.
    """

    def __init__(
        self,
        value: float,
        prior: lsl.Dist | None = None,
        name: str = "",
        bijector: tfb.Bijector | None = tfb.Softplus(),
    ) -> None:
        self.transformed = None
        """The transformed variable (if any)."""

        if bijector is not None and prior is not None:
            super().__init__(value, prior, name=name)
            self.parameter = True
            self.transformed = self.transform(bijector)
            self.update()
        elif bijector is not None and prior is None:
            self.transformed = lsl.param(
                value=bijector.inverse(value), name=f"{name}_transformed"
            )
            super().__init__(lsl.Calc(bijector.forward, self.transformed), name=name)
            self.update()

    def predict(self, samples: dict[str, Array]) -> Array:
        """
        Returns posterior samples of this variable. If the variable is transformed
        using a bijector, this method automatically takes care of applying the
        bijector to the transformed samples.

        Parameters
        ----------
        samples
            Dictionary with arrays of posterior samples.

        Returns
        -------
        Array of posterior samples.
        """
        if self.transformed is not None:
            samps = samples[self.transformed.name]
            return self.value_node.function(samps)

        return samples[self.name]

    def __str__(self) -> str:
        return f"{type(self).__name__}()"


class TruncatedNormalOmega(TransformedVar):
    def __init__(
        self,
        value: Array = 1.0,
        loc: Array = 1.0,
        scale: Array = 0.1,
        low: Array = 0.01,
        high: Array = jnp.inf,
        name: str = "",
    ) -> None:
        if isinstance(scale, float):
            scale = lsl.Data(scale, _name=f"{name}_scale")
        else:
            scale.name = f"{name}_scale"

        if isinstance(low, float):
            low = lsl.Data(low, _name=f"{name}_low")
        else:
            low.name = f"{name}_low"

        if isinstance(high, float):
            high = lsl.Data(high, _name=f"{name}_high")
        else:
            high.name = f"{name}_high"

        if isinstance(loc, float):
            loc = lsl.Data(loc, _name=f"{name}_loc")
        else:
            loc.name = f"{name}_loc"

        prior = lsl.Dist(tfd.TruncatedNormal, loc=loc, scale=scale, low=low, high=high)

        super().__init__(value, prior, name=name, bijector=tfb.Softplus())


class ConstantPriorScalingFactor(Var):
    def __init__(
        self,
        value: float = 1.0,
        name: str = "scaling_factor",
        bijector: tfb.Bijector = tfb.Softplus(),
    ) -> None:
        self.transformed = lsl.Var(bijector.inverse(value), name=f"{name}_transformed")
        self.transformed.parameter = True

        super().__init__(lsl.Calc(bijector.forward, self.transformed), name=name)
        self.update()

    def predict(self, samples: dict[str, Array]) -> Array:
        samps = samples[self.transformed.name]
        return self.value_node.function(samps)

    def __str__(self) -> str:
        return f"{type(self).__name__}()"


def brownian_motion_mat(nrows: int, ncols: int):
    r = jnp.arange(nrows)[:, None] + 1
    c = jnp.arange(ncols)[None, :] + 1
    return jnp.minimum(r, c)


def rw_weight_matrix(nparam: int, center: bool = False, nfixed: int = 0):
    nparam_flex = nparam - nfixed

    B = brownian_motion_mat(nparam_flex, nparam_flex)
    L = jnp.linalg.cholesky(B, upper=False)

    for _ in range(nfixed):
        L = jnp.r_[jnp.zeros((1, L.shape[-1])), L]

    if not center:
        return L

    C = jnp.eye(nparam) - jnp.ones(nparam) / (nparam)
    W = C @ L
    return W


def _rank(eigenvalues: Array, tol: float = 1e-6) -> Array | float:
    """
    Computes the rank of a matrix based on the provided eigenvalues. The rank is taken
    to be the number of non-zero eigenvalues.

    Can handle batches.
    """
    mask = eigenvalues > tol
    rank = jnp.sum(mask, axis=-1)
    return rank


def _log_pdet(
    eigenvalues: Array, rank: Array | float | None = None, tol: float = 1e-6
) -> Array | float:
    """
    Computes the log of the pseudo-determinant of a matrix based on the provided
    eigenvalues. If the rank is provided, it is used to select the non-zero eigenvalues.
    If the rank is not provided, it is computed by counting the non-zero eigenvalues. An
    eigenvalue is deemed to be non-zero if it is greater than the numerical tolerance
    ``tol``.

    Can handle batches.
    """
    if rank is None:
        mask = eigenvalues > tol
    else:
        max_index = eigenvalues.shape[-1] - rank

        def fn(i, x):
            return x.at[..., i].set(i >= max_index)

        mask = jax.lax.fori_loop(0, eigenvalues.shape[-1], fn, eigenvalues)

    selected = jnp.where(mask, eigenvalues, 1.0)
    log_pdet = jnp.sum(jnp.log(selected), axis=-1)
    return log_pdet


class IncreasingCoefLogIncrements(lsl.Var):
    def __init__(
        self,
        nparam: int,
        tau2: TransformedVar,
        name: str = "",
        nfixed: int = 0,
        center: bool = False,
        reparameterize: bool = True,
    ) -> None:
        self.W = rw_weight_matrix(nparam=nparam, center=center, nfixed=nfixed)
        self.tau2 = tau2
        self.reparameterize = reparameterize

        if self.reparameterize:
            self.transformed = lsl.param(
                jnp.zeros(nparam - nfixed),
                distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
                name=f"{name}_param",
            )

            def fn(latent, tau2):
                return jnp.sqrt(tau2) * jnp.dot(self.W, latent)

        else:
            pen = jnp.linalg.inv(self.W @ self.W.T)
            evals = jax.numpy.linalg.eigvalsh(pen)
            rank = _rank(evals)
            log_pdet = _log_pdet(evals, rank=rank)

            prior = lsl.Dist(
                MultivariateNormalDegenerate.from_penalty,
                loc=0.0,
                var=self.tau2,
                pen=pen,
                rank=rank,
                log_pdet=log_pdet,
            )

            self.transformed = lsl.param(
                jnp.zeros(nparam),
                distribution=prior,
                name=f"{name}_param",
            )

            def fn(latent, tau2):
                return latent

        super().__init__(
            lsl.Calc(
                fn,
                self.transformed,
                self.tau2,
            ),
            name=name,
        )
        self.update()

    def predict(self, samples: dict[str, Array]) -> Array:
        transformed_samples = samples[self.transformed.name]
        if not self.reparameterize:
            return transformed_samples

        tau2_samples = self.tau2.predict(samples)
        tau_samples = jnp.expand_dims(jnp.sqrt(tau2_samples), -1)
        return tau_samples * jnp.einsum("jp,...p->...j", self.W, transformed_samples)


class OnionCoefParam(lsl.Var):
    def __init__(
        self,
        knots: OnionKnots,
        tau2: TransformedVar,
        intercept: lsl.Var | None = None,
        slope: TransformedVar | None = None,
        name: str = "",
        combined_kernels: bool = True,
        reparameterize: bool = True,
    ) -> None:
        self.log_increments = IncreasingCoefLogIncrements(
            nparam=knots.nparam,
            tau2=tau2,
            name=f"{name}_log_increments",
            nfixed=0,
            center=False,
            reparameterize=reparameterize,
        )
        self.nparam = knots.nparam
        self.order = knots.order
        self._knots = knots.knots
        self.fn = OnionCoef(knots)
        self.tau2 = tau2
        self.intercept = intercept if intercept is not None else float(0.0)
        self.slope = slope if slope is not None else float(1.0)
        self.combined_kernels = combined_kernels

        def compute_coef(log_increments, intercept, slope):
            standardized_coef = self.fn(log_increments)
            coef = standardized_coef.at[..., 1:].set(standardized_coef[..., 1:] * slope)
            coef = intercept + coef
            return coef

        super().__init__(
            lsl.Calc(
                compute_coef, self.log_increments, self.intercept, self.slope
            ).update(),
            name=name,
        )

        # setup mcmc kernels
        param_names: list[list[str]] = []

        param_names.append([self.log_increments.transformed.name])

        intercept_and_slope: list[str] = []
        if intercept is not None:
            intercept_param = find_param(intercept)
            if intercept_param is not None:
                intercept_and_slope.append(intercept_param.name)
        if slope is not None:
            slope_param = find_param(slope)
            if slope_param is not None:
                intercept_and_slope.append(slope_param.name)
        if intercept_and_slope:
            param_names.append(intercept_and_slope)

        tau2_param_name = (
            self.tau2.transformed.name
            if self.tau2.transformed is not None
            else self.tau2.name
        )

        param_names.append([tau2_param_name])

        if combined_kernels:
            # the nested structure is intentional
            param_names_list: list[list[str]] = [
                [item for sublist in param_names for item in sublist]
            ]
        else:
            param_names_list = param_names

        self.mcmc_kernels: list[gs.NUTSKernel] = []
        for param_name_list in param_names_list:
            self.mcmc_kernels.append(gs.NUTSKernel(param_name_list))

    @property
    def knots(self) -> Array:
        return self._knots

    @knots.setter
    def knots(self, value: Array) -> None:
        knots = OnionKnots.new_from_knots_array(value, order=self.order)

        if not knots.nparam == self.nparam:
            raise ValueError(
                f"New knots imply a different number of parameters than {self.nparam=}."
            )
        if not knots.order == self.order:
            raise ValueError(f"Order of new knots does not match {self.order=}.")

        self._knots = knots.knots

        self.fn = OnionCoef(knots)

        def compute_coef(log_increments, intercept, slope):
            standardized_coef = self.fn(log_increments)
            coef = standardized_coef.at[..., 1:].set(standardized_coef[..., 1:] * slope)
            coef = intercept + coef
            return coef

        self.value_node.function = compute_coef
        self.update()

    def predict(self, samples: dict[str, Array]) -> Array:
        log_increments = self.log_increments.predict(samples)

        if isinstance(self.intercept, float):
            intercept = self.intercept
        else:
            intercept = jnp.expand_dims(samples[self.intercept.name], -1)  # type: ignore

        if isinstance(self.slope, float):
            slope = self.slope
        else:
            slope = jnp.expand_dims(self.slope.predict(samples), -1)  # type: ignore

        return self.value_node.function(log_increments, intercept, slope)


def find_coef_offset(knots: Array, nparam: int, order: int = 3):
    dknots = jnp.diff(knots).mean()
    log_increments = jnp.zeros((nparam + 1,))

    temp_coef = normalization_coef(log_increments, dknots)

    fx_at_zero = bspline_basis(x=jnp.zeros(1), knots=knots, order=order) @ temp_coef

    return -fx_at_zero.squeeze()


class PTMCoef(lsl.Var):
    """
    Classic PTM coefficient with intercept coefficient fixed to zero.
    """

    def __init__(
        self,
        knots: EquidistantKnots,
        tau2: TransformedVar,
        intercept: lsl.Var | None = None,
        slope: TransformedVar | None = None,
        name: str = "",
        combined_kernels: bool = True,
        reparameterize: bool = True,
    ) -> None:
        self.log_increments = IncreasingCoefLogIncrements(
            nparam=knots.nparam + 1,
            tau2=tau2,
            name=f"{name}_log_increments",
            nfixed=1,
            center=True,
            reparameterize=reparameterize,
        )
        self.nparam = knots.nparam
        self.order = knots.order
        self._knots = knots.knots
        dknots = jnp.diff(knots.knots).mean()
        self.fn = partial(normalization_coef, dknots=dknots)
        self.tau2 = tau2
        self.intercept = intercept if intercept is not None else 0.0
        self.slope = slope if slope is not None else 1.0
        self.combined_kernels = combined_kernels

        self._coef_offset = find_coef_offset(self.knots, self.nparam, self.order)

        def compute_coef(log_increments, intercept, slope):
            standardized_coef = self.fn(log_increments) + self._coef_offset
            coef = standardized_coef.at[..., 1:].set(standardized_coef[..., 1:] * slope)
            coef = intercept + coef
            return coef

        super().__init__(
            lsl.Calc(
                compute_coef, self.log_increments, self.intercept, self.slope
            ).update(),
            name=name,
        )

        # setup mcmc kernels
        param_names: list[list[str]] = []

        param_names.append([self.log_increments.transformed.name])

        intercept_and_slope: list[str] = []
        if intercept is not None:
            intercept_param = find_param(intercept)
            if intercept_param is not None:
                intercept_and_slope.append(intercept_param.name)
        if slope is not None:
            slope_param = find_param(slope)
            if slope_param is not None:
                intercept_and_slope.append(slope_param.name)
        if intercept_and_slope:
            param_names.append(intercept_and_slope)

        tau2_param_name = (
            self.tau2.transformed.name
            if self.tau2.transformed is not None
            else self.tau2.name
        )

        param_names.append([tau2_param_name])

        if combined_kernels:
            # the nested structure is intentional
            param_names_list: list[list[str]] = [
                [item for sublist in param_names for item in sublist]
            ]
        else:
            param_names_list = param_names

        self.mcmc_kernels: list[gs.NUTSKernel] = []
        for param_name_list in param_names_list:
            self.mcmc_kernels.append(gs.NUTSKernel(param_name_list))

    @property
    def knots(self) -> Array:
        return self._knots

    @knots.setter
    def knots(self, value: Array) -> None:
        knots = EquidistantKnots.new_from_knots_array(value, order=self.order)
        if not knots.nparam == self.nparam:
            raise ValueError(
                f"New knots imply a different number of parameters than {self.nparam=}."
            )
        if not knots.order == self.order:
            raise ValueError(f"Order of new knots does not match {self.order=}.")

        self._knots = knots.knots

        dknots = jnp.diff(self._knots).mean()
        self.fn = partial(normalization_coef, dknots=dknots)

        def compute_coef(log_increments, intercept, slope):
            standardized_coef = self.fn(log_increments) + self._coef_offset
            coef = standardized_coef.at[..., 1:].set(standardized_coef[..., 1:] * slope)
            coef = intercept + coef
            return coef

        self.value_node.function = compute_coef

    def predict(self, samples: dict[str, Array]) -> Array:
        log_increments = self.log_increments.predict(samples)

        if isinstance(self.intercept, float):
            intercept = self.intercept
        else:
            intercept = jnp.expand_dims(samples[self.intercept.name], -1)  # type: ignore

        if isinstance(self.slope, float):
            slope = self.slope
        else:
            slope = jnp.expand_dims(self.slope.predict(samples), -1)  # type: ignore

        return self.value_node.function(log_increments, intercept, slope)
