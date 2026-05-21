from __future__ import annotations

import operator
from collections.abc import Callable
from functools import cache, partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates.jax import tf2jax as tf

from .bspline import OnionKnots, OnionSpline, PTMSpline

KeyArray = Any
Array = Any


def _validate_gauss_legendre_order(order: int) -> int:
    try:
        order = operator.index(order)
    except TypeError as err:
        raise TypeError("gauss_legendre_order must be a positive integer.") from err

    if order <= 0:
        raise ValueError("gauss_legendre_order must be a positive integer.")

    return order


def _as_scalar_float(value: float | Array, name: str) -> float:
    value_array = np.asarray(value)
    if value_array.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")

    value = float(value_array)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def _validate_integration_bounds(
    bounds: tuple[float, float] | None,
    default_bounds: tuple[float, float],
) -> tuple[float, float]:
    if bounds is None:
        lower = _as_scalar_float(default_bounds[0], "integration lower bound")
        upper = _as_scalar_float(default_bounds[1], "integration upper bound")
    else:
        if len(bounds) != 2:
            raise ValueError("integration_bounds must contain exactly two values.")
        lower = _as_scalar_float(bounds[0], "integration lower bound")
        upper = _as_scalar_float(bounds[1], "integration upper bound")

    if not lower < upper:
        raise ValueError("integration_bounds must satisfy lower < upper.")

    return lower, upper


def _integration_breaks_from_knots(
    knots: Array,
    bounds: tuple[float, float],
    dtype: Any,
) -> Array:
    lower, upper = bounds
    knots_np = np.asarray(knots, dtype=float)
    internal = knots_np[(knots_np > lower) & (knots_np < upper)]
    breaks_np = np.concatenate((np.asarray([lower]), internal, np.asarray([upper])))
    breaks_np = np.unique(breaks_np)
    breaks_np.sort()

    return jnp.asarray(breaks_np, dtype=dtype)


def _gauss_legendre_nodes_and_weights(order: int, dtype: Any) -> tuple[Array, Array]:
    order = _validate_gauss_legendre_order(order)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return jnp.asarray(nodes, dtype=dtype), jnp.asarray(weights, dtype=dtype)


def _as_unused_pseudo_coef(coef: Array) -> Array:
    coef = jnp.asarray(coef)
    if coef.ndim < 2:
        return jnp.reshape(coef, (1, 1))
    return coef


def integrate_piecewise_gauss_legendre(
    f: Callable[[Array], Array],
    breaks: Array,
    nodes: Array,
    weights: Array,
    batch_ndims: int = 0,
) -> Array:
    """
    Integrate ``f`` with fixed Gauss-Legendre quadrature over piecewise intervals.

    The first axis of ``f`` is the quadrature/sample axis. Additional batch axes
    are broadcast by expanding the evaluation points with trailing singleton
    dimensions.
    """
    breaks = jnp.asarray(breaks)
    nodes = jnp.asarray(nodes)
    weights = jnp.asarray(weights)
    batch_ndims = operator.index(batch_ndims)
    if batch_ndims < 0:
        raise ValueError("batch_ndims must be nonnegative.")

    lower = breaks[:-1]
    upper = breaks[1:]
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    points = midpoint[:, None] + half_width[:, None] * nodes[None, :]
    flat_points = jnp.reshape(points, (-1,))

    if batch_ndims:
        flat_points = jnp.reshape(flat_points, flat_points.shape + (1,) * batch_ndims)

    y = f(flat_points)
    y = jnp.reshape(y, points.shape + jnp.shape(y)[1:])
    integration_weights = half_width[:, None] * weights[None, :]
    integration_weights = jnp.reshape(
        integration_weights, integration_weights.shape + (1,) * (jnp.ndim(y) - 2)
    )

    return jnp.sum(y * integration_weights, axis=(0, 1))


def _identity_moment_quadrature_diagnostic(dtype: Any) -> dict[str, Array]:
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    return {
        "mean": zero,
        "variance": one,
        "mean_reference": zero,
        "variance_reference": one,
        "mean_abs_error": zero,
        "variance_abs_error": zero,
        "mean_rel_error": zero,
        "variance_rel_error": zero,
        "ok": jnp.asarray(True),
    }


def onion_dist(
    a: float = -4.0,
    b: float = 4.0,
    nparam: int = 20,
    *,
    knots: Array | OnionKnots | None = None,
    order: int = 3,
    loc_scale: bool = True,
    **kwargs,
) -> Callable[..., "TransformationDist"]:
    """
    Return a reusable constructor for an OnionSpline transformation distribution.

    The OnionSpline is instantiated once in this factory and then reused by each
    distribution created by the returned constructor.
    """
    if knots is None:
        knots = OnionKnots(a, b, nparam=nparam, order=order).knots
    elif isinstance(knots, OnionKnots):
        knots = knots.knots

    bspline = OnionSpline(knots)
    dist_class = LocScaleTransformationDist if loc_scale else TransformationDist

    return partial(dist_class, bspline=bspline, **kwargs)


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
        Accepted for backward compatibility. Computation always follows TFP \
        scalar-event batching.
    gauss_legendre_order
        Number of Gauss-Legendre nodes per knot interval for spline moments.
    integration_bounds
        Optional lower and upper integration bounds. Defaults to the first and \
        last spline knot.
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
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
        **parametric_distribution_kwargs,
    ):
        coef = jnp.asarray(coef)
        if not jnp.issubdtype(coef.dtype, jnp.floating):
            raise TypeError("Spline coefficients must have a floating-point dtype.")

        gauss_legendre_order = _validate_gauss_legendre_order(gauss_legendre_order)
        parameters = dict(locals())

        self.coef = coef
        self.parametric_distribution_kwargs = parametric_distribution_kwargs
        self.centered = centered
        self.scaled = scaled

        self.bspline = bspline
        self.knots = self.bspline.knots
        self.bspline._check_coef_core_shape(coef)
        self.gauss_legendre_order = gauss_legendre_order
        default_integration_bounds = (
            self.bspline._outer_knot_left,
            self.bspline._outer_knot_right,
        )
        self.integration_bounds = _validate_integration_bounds(
            integration_bounds, default_integration_bounds
        )
        if hasattr(self.bspline, "_knots_np"):
            knots_np = self.bspline._knots_np
        else:
            knots_np = jax.device_get(self.knots)
        self.integration_breaks = _integration_breaks_from_knots(
            knots_np, self.integration_bounds, coef.dtype
        )
        self._gl_nodes, self._gl_weights = _gauss_legendre_nodes_and_weights(
            gauss_legendre_order, coef.dtype
        )

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

        self.batched = batched

        super().__init__(
            dtype=coef.dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    def _batch_shape_tuple(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in tuple(self.batch_shape))

    def _broadcast_to_batch(self, value: Array) -> Array:
        return jnp.broadcast_to(jnp.asarray(value), self._batch_shape_tuple())

    def _spline_forward_tfp(self, value: Array) -> tuple[Array, Array]:
        return self.bspline.dot_and_deriv_tfp(
            value, self.coef, batch_shape=self._batch_shape_tuple()
        )

    def _spline_inverse_tfp(self, value: Array) -> Array:
        return self.bspline.dot_inverse_tfp(
            value, self.coef, batch_shape=self._batch_shape_tuple()
        )

    def _mean(self, **kwargs) -> Array:
        if self.parametric_distribution is None:
            parametric_mean = jnp.array(0.0, dtype=self.dtype)
        else:
            parametric_mean = self.parametric_distribution._mean(**kwargs)

        parametric_mean = self._broadcast_to_batch(parametric_mean)

        if self.centered:
            return parametric_mean

        return self._broadcast_to_batch(
            parametric_mean + self.transformation_spline_mean()
        )

    def _stddev(self, **kwargs) -> Array:
        if self.parametric_distribution is None:
            parametric_stddev: float | Array = jnp.array(1.0, dtype=self.dtype)
        else:
            try:
                parametric_stddev = self.parametric_distribution._stddev(**kwargs)
            except NotImplementedError:
                parametric_stddev = jnp.sqrt(self.parametric_distribution._variance())

        parametric_stddev = self._broadcast_to_batch(parametric_stddev)

        if self.scaled:
            return parametric_stddev

        return self._broadcast_to_batch(
            parametric_stddev * jnp.sqrt(self.transformation_spline_variance())
        )

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
        shape = (n,) + self._batch_shape_tuple()
        # ensure 0 will be > 0 to avoid numerical instability
        eps = jnp.finfo(jnp.dtype(self.coef)).eps
        u = jax.random.uniform(  # type: ignore
            seed, shape=shape, minval=eps, maxval=1.0 - eps
        )

        return self._quantile(u)

    @partial(jax.jit, static_argnums=0)
    def _quantile(self, value: Array) -> Array:
        z = self.reference_distribution.quantile(value)
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
        return tf.TensorShape([])

    def _event_shape_tensor(self):
        return jnp.array([], dtype=jnp.int32)

    def _batch_shape(self):
        coef_shape = tf.TensorShape(self.bspline._coef_batch_shape(self.coef))

        if self.parametric_distribution is None:
            parametric_shape = tf.TensorShape([])
        else:
            parametric_shape = self.parametric_distribution.batch_shape

        return tf.broadcast_static_shape(coef_shape, parametric_shape)

    def _batch_shape_tensor(self):
        coef_shape = jnp.asarray(
            self.bspline._coef_batch_shape(self.coef), dtype=jnp.int32
        )

        if self.parametric_distribution is None:
            parametric_shape = jnp.asarray([], dtype=jnp.int32)
        else:
            parametric_shape = self.parametric_distribution.batch_shape_tensor()

        return tf.broadcast_dynamic_shape(coef_shape, parametric_shape)

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

        value = jnp.asarray(value, dtype=self.dtype)
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
        value = jnp.asarray(value, dtype=self.dtype)
        nan_mask = jnp.isnan(value)
        transf, deriv = self._spline_forward_tfp(value)
        transf = jnp.where(nan_mask, jnp.nan, transf)
        deriv = jnp.where(nan_mask, jnp.nan, deriv)
        tiny = jnp.finfo(value.dtype).tiny
        deriv = jnp.clip(deriv, min=tiny)  # safeguard against numerical issues
        return transf, jnp.log(deriv)

    def _spline_center_scale(self) -> tuple[Array, Array]:
        if self.centered:
            ymean = self.transformation_spline_mean()
        else:
            ymean = jnp.array(0.0, dtype=self.dtype)

        if self.scaled:
            mean_for_variance = ymean if self.centered else None
            ystd = jnp.sqrt(self.transformation_spline_variance(mean_for_variance))
        else:
            ystd = jnp.array(1.0, dtype=self.dtype)

        return ymean, ystd

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
        ymean, ystd = self._spline_center_scale()

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

    def transformation_spline_mean(self) -> Array:
        """Expected value under the spline transformation."""
        return self._transformation_spline_mean_gl()

    def _integrate_piecewise_gauss_legendre(
        self,
        fn: Callable[[Array], Array],
        order: int | None = None,
    ) -> Array:
        if order is None:
            nodes = self._gl_nodes
            weights = self._gl_weights
        else:
            nodes, weights = _gauss_legendre_nodes_and_weights(order, self.dtype)

        return integrate_piecewise_gauss_legendre(
            fn,
            breaks=self.integration_breaks,
            nodes=nodes,
            weights=weights,
            batch_ndims=len(self._batch_shape_tuple()),
        )

    def _transformation_spline_mean_gl(self, order: int | None = None) -> Array:
        def fn(x):
            z, logdet = self._transformation_and_logdet_spline(x)
            return x * self.reference_distribution.prob(z) * jnp.exp(logdet)

        return self._integrate_piecewise_gauss_legendre(fn, order=order)

    def transformation_spline_variance(self, mean: Array | None = None) -> Array:
        """Variance under the spline transformation."""
        return self._transformation_spline_variance_gl(mean=mean)

    def _transformation_spline_variance_gl(
        self, mean: Array | None = None, order: int | None = None
    ) -> Array:
        if mean is None:
            mean = self._transformation_spline_mean_gl(order=order)

        def fn(x):
            z, logdet = self._transformation_and_logdet_spline(x)
            return (
                (x - mean) ** 2 * self.reference_distribution.prob(z) * jnp.exp(logdet)
            )

        return self._integrate_piecewise_gauss_legendre(fn, order=order)

    def moment_quadrature_diagnostic(
        self,
        reference_order: int | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> dict[str, Array]:
        """
        Compare spline moments at the configured order against a higher-order rule.

        This diagnostic is opt-in and intentionally separate from likelihood and
        moment computation so it does not add work.
        """
        if reference_order is None:
            reference_order = max(
                2 * self.gauss_legendre_order, self.gauss_legendre_order + 8
            )
        reference_order = _validate_gauss_legendre_order(reference_order)
        if reference_order <= self.gauss_legendre_order:
            raise ValueError("reference_order must exceed gauss_legendre_order.")

        mean = self._transformation_spline_mean_gl(order=self.gauss_legendre_order)
        mean_reference = self._transformation_spline_mean_gl(order=reference_order)
        variance = self._transformation_spline_variance_gl(
            mean=mean, order=self.gauss_legendre_order
        )
        variance_reference = self._transformation_spline_variance_gl(
            mean=mean_reference, order=reference_order
        )

        mean_abs_error = jnp.abs(mean - mean_reference)
        variance_abs_error = jnp.abs(variance - variance_reference)
        mean_rel_error = mean_abs_error / jnp.maximum(
            jnp.abs(mean_reference), jnp.asarray(atol, dtype=self.dtype)
        )
        variance_rel_error = variance_abs_error / jnp.maximum(
            jnp.abs(variance_reference), jnp.asarray(atol, dtype=self.dtype)
        )
        mean_ok = mean_abs_error <= atol + rtol * jnp.abs(mean_reference)
        variance_ok = variance_abs_error <= atol + rtol * jnp.abs(variance_reference)

        return {
            "mean": mean,
            "variance": variance,
            "mean_reference": mean_reference,
            "variance_reference": variance_reference,
            "mean_abs_error": mean_abs_error,
            "variance_abs_error": variance_abs_error,
            "mean_rel_error": mean_rel_error,
            "variance_rel_error": variance_rel_error,
            "ok": jnp.all(mean_ok & variance_ok),
        }

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
        ymean, ystd = self._spline_center_scale()

        return (self._spline_inverse_tfp(value) - ymean) / ystd

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
        Accepted for backward compatibility. Computation always follows TFP \
        scalar-event batching.
    gauss_legendre_order
        Number of Gauss-Legendre nodes per knot interval for spline moments.
    integration_bounds
        Optional lower and upper integration bounds. Defaults to the first and \
        last spline knot.

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
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
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
            gauss_legendre_order=gauss_legendre_order,
            integration_bounds=integration_bounds,
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
        Accepted for backward compatibility. Computation always follows TFP \
        scalar-event batching.
    gauss_legendre_order
        Number of Gauss-Legendre nodes per knot interval for spline moments.
    integration_bounds
        Optional lower and upper integration bounds. Defaults to the first and \
        last spline knot.

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
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(
            coef=_as_unused_pseudo_coef(coef),
            bspline=self.bspline,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            loc=jnp.atleast_1d(loc),
            scale=jnp.atleast_1d(scale),
            centered=centered,
            scaled=scaled,
            batched=batched,
            gauss_legendre_order=gauss_legendre_order,
            integration_bounds=integration_bounds,
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
    def transformation_spline_variance(self, mean: Array | None = None) -> Array:
        return 1.0

    def moment_quadrature_diagnostic(
        self,
        reference_order: int | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> dict[str, Array]:
        return _identity_moment_quadrature_diagnostic(self.dtype)

    def transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        return value, tf.zeros_like(value)

    def inverse_transformation_spline(self, value: Array) -> Array:
        return value


class PseudoTransformationDist(TransformationDist):
    """
    Pseudo-transformation distribution.

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
        Accepted for backward compatibility. Computation always follows TFP \
        scalar-event batching.
    gauss_legendre_order
        Number of Gauss-Legendre nodes per knot interval for spline moments.
    integration_bounds
        Optional lower and upper integration bounds. Defaults to the first and \
        last spline knot.

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
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
        **parametric_distribution_kwargs,
    ) -> None:
        super().__init__(
            coef=_as_unused_pseudo_coef(coef),
            bspline=self.bspline,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reference_distribution=reference_distribution,
            name=name,
            parametric_distribution=parametric_distribution,
            centered=centered,
            scaled=scaled,
            batched=batched,
            gauss_legendre_order=gauss_legendre_order,
            integration_bounds=integration_bounds,
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
    def transformation_spline_variance(self, mean: Array | None = None) -> Array:
        return 1.0

    def moment_quadrature_diagnostic(
        self,
        reference_order: int | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> dict[str, Array]:
        return _identity_moment_quadrature_diagnostic(self.dtype)

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
        Accepted for backward compatibility. Computation always follows TFP \
        scalar-event batching.
    gauss_legendre_order
        Number of Gauss-Legendre nodes per knot interval for spline moments.
    integration_bounds
        Optional lower and upper integration bounds. Defaults to the first and \
        last spline knot.

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
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(
            coef=_as_unused_pseudo_coef(coef),
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
            gauss_legendre_order=gauss_legendre_order,
            integration_bounds=integration_bounds,
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
    def transformation_spline_variance(self, mean: Array | None = None) -> Array:
        return 1.0

    def moment_quadrature_diagnostic(
        self,
        reference_order: int | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> dict[str, Array]:
        return _identity_moment_quadrature_diagnostic(self.dtype)

    def transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        return value, tf.zeros_like(value)

    def inverse_transformation_spline(self, value: Array) -> Array:
        return value
