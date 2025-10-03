from __future__ import annotations

from collections.abc import Callable
from functools import cache, partial
from typing import Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates.jax import tf2jax as tf

from .bspline import OnionSpline, PTMSpline

KeyArray = Any
Array = Any


def integrate_simpson(
    f: Callable[[Array], Array], a: float | Array, b: float | Array, N: int = 20
) -> Array:
    """
    Implementation from:
    https://jax-cosmo.readthedocs.io/en/latest/_modules/jax_cosmo/scipy/integrate.html#romb
    """
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = jnp.linspace(a, b, N + 1)
    y = f(x)
    S = (
        dx
        / 3
        * jnp.sum(
            y[..., 0:-1:2] + 4 * y[..., 1::2] + y[..., 2::2], axis=-1, keepdims=True
        )
    )
    return S


class TransformationDist(tfd.Distribution):
    """
    Transformation distribution using a spline and optional parametric component.

    This combines a spline-based monotonically increasing transformation with a
    parametric distribution, using a reference distribution (default: standard
    normal) for mapping and likelihoods.

    Parameters
    ----------
    coef
        Coefficients for the spline basis.
    bspline
        Spline object providing transformation and its inverse/derivative.
    parametric_distribution
        Parametric distribution class to include in the model.
    reference_distribution
        Reference distribution used for transformations; defaults to Normal(0, 1).
    validate_args
        Whether to validate input arguments.
    allow_nan_stats
        Whether to allow NaN statistics.
    name
        Name of the distribution.
    centered
        If True, the transformation is centered such that any side-effect the \
        spline transformation might have on the location of the distribution is \
        negated.
    scaled
        If True, the transformation is scaled such that any side-effect the \
        spline transformation might have on the scale of the distribution is \
        negated.
    batched
        If True, allow for batched computations (might be slightly less efficient in \
        the scalar case).
    **parametric_distribution_kwargs
        Additional keyword arguments passed to the parametric distribution.

    Attributes
    ----------
    coef
        Coefficients for the spline basis.
    bspline
        Spline object used for transformations.
    reference_distribution
        Reference distribution used for mapping and likelihoods.
    parametric_distribution
        Instantiated parametric distribution if provided, else None.
    parametric_distribution_kwargs
        Keyword arguments used to construct the parametric distribution.
    centered
        Indicates whether centering is applied.
    scaled
        Indicates whether scaling is applied.
    """

    def __init__(
        self,
        coef: Array,
        bspline: PTMSpline | OnionSpline,
        parametric_distribution: type[tfd.Distribution] | None = None,
        reference_distribution: tfd.Distribution | None = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "TransformationDist",
        centered: bool = False,
        scaled: bool = False,
        batched: bool = True,
        **parametric_distribution_kwargs,
    ):
        parameters = dict(locals())

        self.coef = coef
        self.parametric_distribution_kwargs = parametric_distribution_kwargs
        self.centered = centered
        self.scaled = scaled

        self.bspline = bspline
        self.knots = self.bspline.knots

        if reference_distribution is None:
            self.reference_distribution = tfd.Normal(loc=0.0, scale=1.0)
        else:
            self.reference_distribution = reference_distribution

        if parametric_distribution is None and parametric_distribution_kwargs:
            raise ValueError(
                "Provided 'parametric_distribution_kwargs', but no value for"
                " 'parametric_distribution'."
            )

        if parametric_distribution is None:
            self.parametric_distribution = None
        else:
            self.parametric_distribution = parametric_distribution(
                **parametric_distribution_kwargs
            )

        if batched:
            self.dot_and_deriv = self.bspline.dot_and_deriv
        else:
            self.dot_and_deriv = self.bspline.dot_and_deriv_n_fullbatch

        super().__init__(
            dtype=coef.dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    def _mean(self, **kwargs) -> Array:
        if self.parametric_distribution is None:
            parametric_mean = 0.0
        else:
            parametric_mean = self.parametric_distribution._mean(**kwargs)

        if self.centered:
            return parametric_mean

        return parametric_mean + self.transformation_spline_mean()

    def _stddev(self, **kwargs) -> Array:
        if self.parametric_distribution is None:
            parametric_stddev: float | Array = 1.0
        else:
            try:
                parametric_stddev = self.parametric_distribution._stddev(**kwargs)
            except NotImplementedError:
                parametric_stddev = jnp.sqrt(self.parametric_distribution._variance())

        if self.scaled:
            return parametric_stddev

        return parametric_stddev * jnp.sqrt(self.transformation_spline_variance())

    def _cdf(self, value: Array) -> Array | float:
        z, _ = self.transformation_and_logdet(value)
        return self.reference_distribution.cdf(z)

    def _log_cdf(self, value: Array) -> Array | float:
        return jnp.log(self._cdf(value))

    @partial(jax.jit, static_argnums=0)
    def _log_prob(self, value: Array) -> Array | float:
        z, logdet = self.transformation_and_logdet(value)
        return self.reference_distribution.log_prob(z) + logdet

    def _prob(self, value: Array) -> Array | float:
        return jnp.exp(self._log_prob(value))

    def _sample_n(self, n: int | Array, seed: KeyArray | None = None) -> Array:
        shape = [n] + self.batch_shape + self.event_shape
        # ensure 0 will be > 0 to avoid numerical instability
        eps = jnp.finfo(jnp.dtype(self.coef)).eps
        u = jax.random.uniform(seed, shape=shape, minval=eps)  # type: ignore

        shape = [n] + self.batch_shape + self.event_shape
        return tf.reshape(self._quantile(u), shape)

    @partial(jax.jit, static_argnums=0)
    def _quantile(self, value: Array) -> Array:
        z = self.reference_distribution.quantile(value)
        if jnp.ndim(value) == 0:  # scalar case
            z = jnp.reshape(z, (1,) * len(self.batch_shape))
            return self.inverse_transformation(z)

        return self.inverse_transformation(z)

    def quantile_spline(self, value: Array) -> Array:
        """
        Quantile function using only the spline transformation.

        Parameters
        ----------
        value
            Quantile levels in (0, 1).

        Returns
        -------
            Values under the spline-only model at the requested quantiles.
        """
        z = self.reference_distribution.quantile(value)
        return self.inverse_transformation_spline(z)

    def _event_shape(self):
        # if self.rowwise_dot:
        #     return tf.TensorShape([self.coef.shape[-2]])
        return tf.TensorShape([])

    def _event_shape_tensor(self):
        # if self.rowwise_dot:
        #     return jnp.array([self.coef.shape[-2]], dtype=jnp.int32)
        return jnp.array([], dtype=jnp.int32)

    def _batch_shape(self):
        shape = tuple()
        for param in self.parametric_distribution_kwargs.values():
            shape = tf.broadcast_static_shape(shape, jnp.shape(param))

        coef_shape = self.coef.shape[:-1]

        coef_shape = tf.TensorShape(coef_shape)

        return tf.broadcast_static_shape(coef_shape, shape)

    def _batch_shape_tensor(self):
        shape = tuple()
        for param in self.parametric_distribution_kwargs.values():
            shape = tf.broadcast_static_shape(shape, jnp.shape(param))

        coef_shape = self.coef.shape[:-1]

        coef_shape = tf.TensorShape(coef_shape)

        return tf.broadcast_dynamic_shape(coef_shape, shape)

    def log_prob_spline(self, value: Array):
        """
        Log probability under the spline-only transformation.

        Parameters
        ----------
        value
            Values at which to evaluate the log density.

        Returns
        -------
            Log probability evaluated using the spline transform.
        """
        z, logdet = self.transformation_and_logdet_spline(value)
        return self.reference_distribution.log_prob(z) + logdet

    def prob_spline(self, value: Array):
        """
        Probability under the spline-only transformation.

        Parameters
        ----------
        value
            Values at which to evaluate the density.

        Returns
        -------
            Probability density evaluated using the spline transform.
        """
        return jnp.exp(self.log_prob_spline(value))

    def cdf_spline(self, value: Array):
        """
        CDF under the spline-only transformation.

        Parameters
        ----------
        value
            Values at which to evaluate the CDF.

        Returns
        -------
            Cumulative distribution evaluated using the spline transform.
        """
        z, _ = self.transformation_and_logdet_spline(value)
        return self.reference_distribution.cdf(z)

    def transformation_and_logdet_parametric(self, value: Array) -> tuple[Array, Array]:
        """
        Apply the parametric transformation and its log determinant.

        Parameters
        ----------
        value
            Input values.

        Returns
        -------
            A pair of transformed values and the corresponding log determinant.
        """
        if self.parametric_distribution is None:
            return value, jnp.zeros_like(value)

        F_apriori = self.parametric_distribution
        Fz = self.reference_distribution

        # Use jnp.finfo to get machine epsilon and min/max float values
        eps = jnp.finfo(value.dtype).eps
        tiny = jnp.finfo(value.dtype).tiny
        max_float = 1.0 - eps

        u = F_apriori.cdf(value)
        u = jnp.where(u >= 1.0, max_float, u)  # safeguard using max float
        u = jnp.where(u <= 0.0, tiny, u)  # safeguard using smallest positive float

        transf = Fz.quantile(u)
        logdet = F_apriori.log_prob(value) - Fz.log_prob(transf)

        return transf, logdet

    def _transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        nan_mask = jnp.isnan(value)
        transf, deriv = self.dot_and_deriv(value, self.coef)
        transf = jnp.where(nan_mask, jnp.nan, transf)
        deriv = jnp.where(nan_mask, jnp.nan, deriv)
        tiny = jnp.finfo(value.dtype).tiny
        deriv = jnp.clip(deriv, min=tiny)  # safeguard against numerical issues
        return transf, jnp.log(deriv)

    def transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        """
        Apply spline transformation with centering/scaling and compute logdet.

        Parameters
        ----------
        value
            Input values.

        Returns
        -------
            A pair of transformed values and the corresponding log determinant.
        """
        if self.centered:
            ymean = self.transformation_spline_mean()  # intercept / expected val.
        else:
            ymean = jnp.array(0.0)

        if self.scaled and not self.centered:
            _ymean = self.transformation_spline_mean()  # intercept / expected val.
            # ystd = jnp.sqrt(self.transformation_spline_variance(_ymean))
            ystd = jnp.sqrt(self.transformation_spline_variance())
        elif self.scaled and self.centered:
            # ystd = jnp.sqrt(self.transformation_spline_variance(ymean))
            ystd = jnp.sqrt(self.transformation_spline_variance())
        else:
            ystd = jnp.array(1.0)

        value = value * ystd + ymean
        logdet = jnp.log(ystd)

        z, transf_logdet = self._transformation_and_logdet_spline(value)

        z_logdet = transf_logdet + logdet

        return z, z_logdet

    @partial(jax.jit, static_argnums=0)
    def transformation_and_logdet(self, value: Array) -> tuple[Array, Array]:
        """Apply parametric then spline transforms; return value and logdet."""
        transf_param, logdet_param = self.transformation_and_logdet_parametric(value)
        transf_spline, logdet_spline = self.transformation_and_logdet_spline(
            transf_param
        )

        logdet = logdet_param + logdet_spline

        return transf_spline, logdet

    # @cache
    def transformation_spline_mean(self) -> Array:
        """Expected value under the spline transformation."""
        return self._transformation_spline_mean_simple()

    def _transformation_spline_mean_simple(self) -> Array:
        def fn(x):
            z, logdet = self._transformation_and_logdet_spline(x)
            return x * self.reference_distribution.prob(z) * jnp.exp(logdet)

        mom = integrate_simpson(fn, a=self.knots[0], b=self.knots[-1], N=1024)

        return mom

    def _transformation_spline_mean_for(self) -> Array:
        state = (
            jnp.inf,  # convergence criterion
            jnp.zeros(self.batch_shape),
            0,  # iteration counter
        )
        maxiter = 1

        def body_fun(state):
            m_before = state[1]

            def fn(x):
                z, logdet = self._transformation_and_logdet_spline(x + m_before)
                return x * self.reference_distribution.prob(z) * jnp.exp(logdet)

            m_after = integrate_simpson(fn, a=self.knots[0], b=self.knots[-1], N=1024)
            m_after = jnp.reshape(m_after, self.batch_shape)
            diff = jnp.abs(m_after).sum()

            return (diff, m_before + m_after, state[2] + 1)

        state = jax.lax.fori_loop(
            lower=0,
            upper=maxiter,
            init_val=state,
            body_fun=lambda i, val: body_fun(val),
        )

        return state[1]

    def _transformation_spline_mean_while(self) -> Array:
        """
        Potentially more accurate, but cannot be differentiated in reverse mode by jax.
        Unused for now, kept for reference.
        """
        state = (
            jnp.inf,  # convergence criterion
            jnp.zeros(self.batch_shape),
            0,  # iteration counter
        )
        tol = 1e-3
        maxiter = 20

        def cond_fun(state):
            return jnp.logical_and(state[0] > tol, state[2] <= maxiter)

        def body_fun(state):
            m_before = state[1]

            def fn(x):
                z, logdet = self._transformation_and_logdet_spline(x + m_before)
                return x * self.reference_distribution.prob(z) * jnp.exp(logdet)

            m_after = integrate_simpson(fn, a=self.knots[0], b=self.knots[-1], N=1024)
            m_after = jnp.reshape(m_after, self.batch_shape)
            diff = jnp.abs(m_after).sum()

            return (diff, m_before + m_after, state[2] + 1)

        state = jax.lax.while_loop(body_fun=body_fun, cond_fun=cond_fun, init_val=state)

        return state[1]

    # @cache
    def transformation_spline_variance(self) -> Array:
        """Variance under the spline transformation."""
        return self._transformation_spline_variance_simple()

    def _transformation_spline_variance_simple(
        self, mean: Array | None = None
    ) -> Array:
        if mean is None:
            mean = self.transformation_spline_mean()

        def fn(x):
            z, logdet = self._transformation_and_logdet_spline(x)
            return (
                (x - mean) ** 2 * self.reference_distribution.prob(z) * jnp.exp(logdet)
            )

        var = integrate_simpson(fn, a=self.knots[0], b=self.knots[-1], N=1024)

        return var

    def _transformation_spline_variance_for(self) -> Array:
        mean = self.transformation_spline_mean()

        state = (
            jnp.inf,  # convergence criterion
            jnp.ones(self.batch_shape),
            0,  # iteration counter
        )

        maxiter = 2

        def pdf(x, v):
            s = jnp.sqrt(v)
            zt = s * x + mean
            z, logdet = self._transformation_and_logdet_spline(zt)
            return self.reference_distribution.prob(z) * jnp.exp(logdet) * s

        def body_fun(state):
            v_before = state[1]

            def fn(x):
                return x**2 * pdf(x, v_before)

            v_after = integrate_simpson(fn, a=self.knots[0], b=self.knots[-1], N=1024)
            v_after = jnp.reshape(v_after, self.batch_shape)
            diff = jnp.abs(1.0 - v_after).sum()
            state = (diff, v_before * v_after, state[2] + 1)

            return state

        state = jax.lax.fori_loop(
            lower=0,
            upper=maxiter,
            init_val=state,
            body_fun=lambda i, val: body_fun(val),
        )

        return state[1]

    def _transformation_spline_variance_while(self) -> Array:
        """
        Potentially more accurate, but cannot be differentiated in reverse mode by jax.
        Unused for now, kept for reference.
        """
        mean = self.transformation_spline_mean()

        state = (
            jnp.inf,  # convergence criterion
            jnp.ones(self.batch_shape),
            0,  # iteration counter
        )
        tol = 1e-3
        maxiter = 20

        def cond_fun(state):
            return jnp.logical_and(state[0] > tol, state[2] <= maxiter)

        def pdf(x, v):
            s = jnp.sqrt(v)
            zt = s * x + mean
            z, logdet = self._transformation_and_logdet_spline(zt)
            return self.reference_distribution.prob(z) * jnp.exp(logdet) * s

        def body_fun(state):
            v_before = state[1]

            def fn(x):
                return x**2 * pdf(x, v_before)

            v_after = integrate_simpson(fn, a=self.knots[0], b=self.knots[-1], N=1024)
            v_after = jnp.reshape(v_after, self.batch_shape)
            diff = jnp.abs(1.0 - v_after).sum()
            state = (diff, v_before * v_after, state[2] + 1)

            return state

        state = jax.lax.while_loop(body_fun=body_fun, cond_fun=cond_fun, init_val=state)

        return state[1]

    def inverse_transformation_spline(self, value: Array) -> Array:
        """
        Inverse of the spline transformation.

        Parameters
        ----------
        value
            Inputs on the reference scale.

        Returns
        -------
            Values on the original scale using the spline inverse.
        """
        if self.centered:
            ymean = self.transformation_spline_mean()  # intercept / expected val.
        else:
            ymean = jnp.array(0.0)

        if self.scaled and not self.centered:
            _ymean = self.transformation_spline_mean()  # intercept / expected val.
            # ystd = jnp.sqrt(self.transformation_spline_variance(_ymean))
            ystd = jnp.sqrt(self.transformation_spline_variance())
        elif self.scaled and self.centered:
            # ystd = jnp.sqrt(self.transformation_spline_variance(ymean))
            ystd = jnp.sqrt(self.transformation_spline_variance())
        else:
            ystd = jnp.array(1.0)

        return (self.bspline.dot_inverse(value, self.coef) - ymean) / ystd

    def inverse_transformation_parametric(self, value: Array) -> Array:
        """
        Inverse of the parametric transformation.

        Parameters
        ----------
        value
            Inputs on the reference scale.

        Returns
        -------
            Values mapped back via the parametric distribution.
        """
        if self.parametric_distribution is None:
            return value

        # Use jnp.finfo to get machine epsilon and min/max float values
        eps = jnp.finfo(value.dtype).eps
        tiny = jnp.finfo(value.dtype).tiny
        max_float = 1.0 - eps

        u = self.reference_distribution.cdf(value)
        u = jnp.where(u >= 1.0, max_float, u)  # safeguard against numerical issues
        u = jnp.where(u <= 0.0, tiny, u)  # safeguard against numerical issues
        y = self.parametric_distribution.quantile(u)

        return y

    @partial(jax.jit, static_argnums=0)
    def inverse_transformation(self, value: Array) -> Array:
        """
        Inverse transformation combining parametric and spline parts.

        Parameters
        ----------
        value
            Inputs on the reference scale.

        Returns
        -------
            Values on the original scale after both inverses.
        """
        y_tilde = self.inverse_transformation_spline(value)
        y = self.inverse_transformation_parametric(y_tilde)

        return y


class LocScaleTransformationDist(TransformationDist):
    """
    Location–scale specialization of :class:`.TransformationDist`.

    Uses a Normal parametric layer with location ``loc`` and scale ``scale``,
    combined with a spline transformation and reference Normal(0, 1).

    Parameters
    ----------
    coef
        Coefficients for the spline basis.
    loc
        Location parameter for the Normal layer.
    scale
        Scale parameter for the Normal layer.
    bspline
        Spline object providing the transformation.
    validate_args
        Whether to validate input arguments.
    allow_nan_stats
        Whether to allow NaN statistics.
    name
        Name of the distribution.
    centered
        If True, the transformation is centered.
    scaled
        If True, the transformation is scaled.
    batched
        If True, use batched computations.

    Notes
    -----
    Inherits public attributes from :class:`TransformationDist`.
    """

    def __init__(
        self,
        coef: Array,
        loc: Array,
        scale: Array,
        bspline: PTMSpline | OnionSpline,
        parametric_distribution: type[tfd.Distribution] | None = tfd.Normal,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "LocScaleTransformationDist",
        centered: bool = False,
        scaled: bool = False,
        batched: bool = True,
        reference_distribution=tfd.Normal(loc=0.0, scale=1.0),
    ) -> None:
        super().__init__(
            coef=coef,
            parametric_distribution=parametric_distribution,
            reference_distribution=reference_distribution,
            bspline=bspline,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            loc=loc,
            scale=scale,
            centered=centered,
            scaled=scaled,
            batched=batched,
        )

    def transformation_and_logdet_parametric(self, value: Array) -> tuple[Array, Array]:
        """
        Apply location–scale normalization and its log-determinant.

        Parameters
        ----------
        value
            Input values on the original scale.

        Returns
        -------
        transf, logdet
            Normalized values and corresponding log-determinant.
        """
        if self.parametric_distribution is None:
            raise RuntimeError

        sd = self.parametric_distribution.stddev()
        transf = (value - self.parametric_distribution.mean()) / sd

        logdet = -jnp.log(sd)

        return transf, logdet

    def inverse_transformation_parametric(self, value: Array) -> Array:
        """
        Invert the location–scale normalization.

        Parameters
        ----------
        value
            Values on the normalized (reference) scale.

        Returns
        -------
        y
            Values mapped back to the original scale.
        """
        if self.parametric_distribution is None:
            raise RuntimeError

        sd = self.parametric_distribution.stddev()
        m = self.parametric_distribution.mean()
        y = value * sd + m

        return y


class GaussianPseudoTransformationDist(LocScaleTransformationDist):
    """
    Gaussian pseudo-transformation distribution.

    A simplified version of :class:`LocScaleTransformationDist` with
    identity spline behavior. This class is used to be compatible in interface to
    :class:`LocScaleTransformationDist` while conveniently representing a Gaussian
    distribution.


    Parameters
    ----------
    coef
        Coefficients for the spline basis (kept for consistency).
    loc
        Location parameter for the Normal layer.
    scale
        Scale parameter for the Normal layer.
    validate_args
        Whether to validate input arguments.
    allow_nan_stats
        Whether to allow NaN statistics.
    name
        Name of the distribution.
    centered
        If True, the transformation is centered.
    scaled
        If True, the transformation is scaled.
    batched
        If True, use batched computations.

    Notes
    -----
    - Inherits attributes from :class:`LocScaleTransformationDist`.
    - The spline transformation is effectively the identity.
    """

    knots = jnp.linspace(-3.0, 3.0, 10)
    bspline = PTMSpline(knots)

    def __init__(
        self,
        coef: Array,
        loc: Array,
        scale: Array,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "GaussianPseudoTransformationDist",
        centered: bool = False,
        scaled: bool = False,
        batched: bool = True,
    ) -> None:
        super().__init__(
            coef=coef,
            bspline=self.bspline,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            loc=jnp.atleast_1d(loc),
            scale=jnp.atleast_1d(scale),
            centered=centered,
            scaled=scaled,
            batched=batched,
        )

    @partial(jax.jit, static_argnums=0)
    def transformation_and_logdet(self, value: Array) -> tuple[Array, Array]:
        return self.transformation_and_logdet_parametric(value)

    @partial(jax.jit, static_argnums=0)
    def inverse_transformation(self, value: Array) -> Array:
        return self.inverse_transformation_parametric(value)

    @cache
    def transformation_spline_mean(self):
        return 0.0

    @cache
    def transformation_spline_variance(self) -> Array:
        return 1.0

    def transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        return value, tf.zeros_like(value)

    def inverse_transformation_spline(self, value: Array) -> Array:
        return value


class PseudoTransformationDist(TransformationDist):
    """
    Oseudo-transformation distribution.

    A simplified version of :class:`TransformationDist` with
    identity spline behavior. This class is used to be compatible in interface to
    :class:`TransformationDist` while conveniently representing a parametric
    distribution.


    Parameters
    ----------
    coef
        Coefficients for the spline basis (kept for consistency).
    loc
        Location parameter for the Normal layer.
    scale
        Scale parameter for the Normal layer.
    validate_args
        Whether to validate input arguments.
    allow_nan_stats
        Whether to allow NaN statistics.
    name
        Name of the distribution.
    centered
        If True, the transformation is centered.
    scaled
        If True, the transformation is scaled.
    batched
        If True, use batched computations.

    Notes
    -----
    - Inherits attributes from :class:`LocScaleTransformationDist`.
    - The spline transformation is effectively the identity.
    """

    knots = jnp.linspace(-3.0, 3.0, 10)
    bspline = PTMSpline(knots)

    def __init__(
        self,
        coef: Array,
        parametric_distribution: type[tfd.Distribution] | None = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "PseudoTransformationDist",
        centered: bool = False,
        scaled: bool = False,
        batched: bool = True,
        reference_distribution=tfd.Normal(loc=0.0, scale=1.0),
        **parametric_distribution_kwargs,
    ) -> None:
        super().__init__(
            coef=coef,
            bspline=self.bspline,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reference_distribution=reference_distribution,
            name=name,
            parametric_distribution=parametric_distribution,
            centered=centered,
            scaled=scaled,
            batched=batched,
            **parametric_distribution_kwargs,
        )

    @partial(jax.jit, static_argnums=0)
    def transformation_and_logdet(self, value: Array) -> tuple[Array, Array]:
        return self.transformation_and_logdet_parametric(value)

    @partial(jax.jit, static_argnums=0)
    def inverse_transformation(self, value: Array) -> Array:
        return self.inverse_transformation_parametric(value)

    @cache
    def transformation_spline_mean(self):
        return 0.0

    @cache
    def transformation_spline_variance(self) -> Array:
        return 1.0

    def transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        return value, tf.zeros_like(value)

    def inverse_transformation_spline(self, value: Array) -> Array:
        return value


class LocScalePseudoTransformationDist(TransformationDist):
    """
    Location–scale specialization of :class:`.PseudoTransformationDist`.

    Parameters
    ----------
    coef
        Coefficients for the spline basis.
    loc
        Location parameter for the Normal layer.
    scale
        Scale parameter for the Normal layer.
    validate_args
        Whether to validate input arguments.
    allow_nan_stats
        Whether to allow NaN statistics.
    name
        Name of the distribution.
    centered
        If True, the transformation is centered.
    scaled
        If True, the transformation is scaled.
    batched
        If True, use batched computations.

    Notes
    -----
    Inherits public attributes from :class:`TransformationDist`.
    """

    knots = jnp.linspace(-3.0, 3.0, 10)
    bspline = PTMSpline(knots)

    def __init__(
        self,
        coef: Array,
        loc: Array,
        scale: Array,
        parametric_distribution: type[tfd.Distribution] | None = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "LocScalePseudoTransformationDist",
        centered: bool = False,
        scaled: bool = False,
        batched: bool = True,
    ) -> None:
        super().__init__(
            coef=coef,
            parametric_distribution=parametric_distribution,
            reference_distribution=tfd.Normal(loc=0.0, scale=1.0),
            bspline=self.bspline,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            loc=loc,
            scale=scale,
            centered=centered,
            scaled=scaled,
            batched=batched,
        )

    def transformation_and_logdet_parametric(self, value: Array) -> tuple[Array, Array]:
        """
        Apply location–scale normalization and its log-determinant.

        Parameters
        ----------
        value
            Input values on the original scale.

        Returns
        -------
        transf, logdet
            Normalized values and corresponding log-determinant.
        """
        if self.parametric_distribution is None:
            raise RuntimeError

        sd = self.parametric_distribution.stddev()
        transf = (value - self.parametric_distribution.mean()) / sd

        logdet = -jnp.log(sd)

        return transf, logdet

    def inverse_transformation_parametric(self, value: Array) -> Array:
        """
        Invert the location–scale normalization.

        Parameters
        ----------
        value
            Values on the normalized (reference) scale.

        Returns
        -------
        y
            Values mapped back to the original scale.
        """
        if self.parametric_distribution is None:
            raise RuntimeError

        sd = self.parametric_distribution.stddev()
        m = self.parametric_distribution.mean()
        y = value * sd + m

        return y

    @partial(jax.jit, static_argnums=0)
    def transformation_and_logdet(self, value: Array) -> tuple[Array, Array]:
        return self.transformation_and_logdet_parametric(value)

    @partial(jax.jit, static_argnums=0)
    def inverse_transformation(self, value: Array) -> Array:
        return self.inverse_transformation_parametric(value)

    @cache
    def transformation_spline_mean(self):
        return 0.0

    @cache
    def transformation_spline_variance(self) -> Array:
        return 1.0

    def transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        return value, tf.zeros_like(value)

    def inverse_transformation_spline(self, value: Array) -> Array:
        return value
