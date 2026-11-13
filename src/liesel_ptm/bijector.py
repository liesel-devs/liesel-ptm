"""Bijectors for the spline transformation of penalized transformation models."""

from __future__ import annotations

import operator
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from interpax import interp1d
from tensorflow_probability.substrates.jax.internal import (
    parameter_properties as tfp_parameter_properties,
)

if TYPE_CHECKING:
    from .bspline import OnionSpline
    from .bspline.util import TransformationSpline

Array = Any

# Coefficient rows per block when a spline is inverted on a grid.
INVERSE_BLOCK_ROWS = 256


def _validate_gauss_legendre_order(
    order: int, name: str = "gauss_legendre_order"
) -> int:
    try:
        order = operator.index(order)
    except TypeError as err:
        raise TypeError(f"{name} must be a positive integer.") from err

    if order <= 0:
        raise ValueError(f"{name} must be a positive integer.")

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


@cache
def _numpy_gauss_legendre_nodes_and_weights(
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Cache host arrays only: caching JAX arrays can retain tracers across jit calls.
    return np.polynomial.legendre.leggauss(order)


def _gauss_legendre_nodes_and_weights(order: int, dtype: Any) -> tuple[Array, Array]:
    order = _validate_gauss_legendre_order(order)
    nodes, weights = _numpy_gauss_legendre_nodes_and_weights(order)
    return jnp.asarray(nodes, dtype=dtype), jnp.asarray(weights, dtype=dtype)


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


def _moment_diagnostic(
    moments: tuple[Array, Array],
    reference_moments: tuple[Array, Array],
    rtol: float,
    atol: float,
    dtype: Any,
) -> dict[str, Array]:
    """Compare moments against moments from a higher-order quadrature rule."""
    mean, variance = moments
    mean_reference, variance_reference = reference_moments
    mean_abs_error = jnp.abs(mean - mean_reference)
    variance_abs_error = jnp.abs(variance - variance_reference)
    mean_rel_error = mean_abs_error / jnp.maximum(
        jnp.abs(mean_reference), jnp.asarray(atol, dtype=dtype)
    )
    variance_rel_error = variance_abs_error / jnp.maximum(
        jnp.abs(variance_reference), jnp.asarray(atol, dtype=dtype)
    )
    mean_ok = (
        jnp.isfinite(mean)
        & jnp.isfinite(mean_reference)
        & (mean_abs_error <= atol + rtol * jnp.abs(mean_reference))
    )
    variance_ok = (
        jnp.isfinite(variance)
        & jnp.isfinite(variance_reference)
        & (variance_abs_error <= atol + rtol * jnp.abs(variance_reference))
    )

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


def _validate_reference_order(reference_order: int | None, order: int) -> int:
    if reference_order is None:
        reference_order = max(2 * order, order + 8)
    reference_order = _validate_gauss_legendre_order(reference_order, "reference_order")
    if reference_order <= order:
        raise ValueError("reference_order must exceed the configured quadrature order.")
    return reference_order


@jax.jit
def _onion_core_inverse_rows(
    bspline: OnionSpline, value_rows: Array, coef_rows: Array
) -> Array:
    """
    Invert an onion spline row by row, on a fixed grid over its core.

    ``value_rows`` has shape ``(B, n)`` and ``coef_rows`` has shape ``(B, p)``.
    Values outside the core are returned unchanged, because the spline is the
    identity there.
    """
    dtype = value_rows.dtype
    lower = jnp.asarray(bspline.min_knot, dtype=dtype)
    upper = jnp.asarray(bspline.max_knot, dtype=dtype)
    xgrid = jnp.linspace(lower, upper, bspline._ngrid_inverse, dtype=dtype)

    def invert_row(row: tuple[Array, Array]) -> Array:
        value_row, coef_row = row
        ygrid = bspline.evaluate(xgrid, coef_row)
        # h maps the core onto itself; pin the ends against rounding.
        ygrid = ygrid.at[0].set(lower).at[-1].set(upper)
        in_core = (value_row > lower) & (value_row < upper)
        clipped = jnp.clip(value_row, lower, upper)
        inverse = interp1d(clipped, ygrid, xgrid, method="monotonic")
        return jnp.where(in_core, inverse, value_row)

    n_rows = value_rows.shape[0]
    batch_size = min(INVERSE_BLOCK_ROWS, n_rows)
    return jax.lax.map(invert_row, (value_rows, coef_rows), batch_size=batch_size)


class SplineBijector(tfb.AutoCompositeTensorBijector):
    """
    Monotone spline transformation of a transformation model as a TFP bijector.

    In a transformation model, the spline ``h`` maps a response ``y`` to the
    reference scale, ``z = h(y)``, where ``z`` follows the reference distribution.
    TFP bijectors map the other way, from the base (reference) distribution to the
    data. So this bijector's ``inverse`` is ``h`` and its ``forward`` is ``h⁻¹``:

    - ``inverse(y)`` evaluates ``h(y)``. This is the cheap direction, used for
      ``log_prob``, ``cdf`` and the survival functions.
    - ``inverse_log_det_jacobian(y)`` is ``log h'(y)``.
    - ``forward(z)`` evaluates ``h⁻¹(z)`` numerically. It is used for quantiles
      and sampling.

    Pass it to ``tfd.TransformedDistribution`` with the reference distribution as
    base distribution, and chain it with other bijectors, for example
    ``tfb.Chain([tfb.Exp(), spline_bijector])`` for a model on the log scale.

    Parameters
    ----------
    coef
        Unconstrained spline coefficients with shape ``batch_shape + (n_param,)``.
        The spline maps them to the coefficients of a monotonically increasing
        spline. A leading axis with one entry per observation is an ordinary batch
        dimension.
    bspline
        Spline object that evaluates the transformation, its derivative and its
        inverse. Reuse one spline across parameter draws, so its compiled
        functions are reused.
    centered
        If True, shift the spline input by the finite-bound raw mean
        [mean_raw][liesel_ptm.SplineBijector.mean_raw], so that the forward
        image of the reference distribution has mean zero.
    scaled
        If True, scale the spline input by the square root of the finite-bound
        raw variance [variance_raw][liesel_ptm.SplineBijector.variance_raw], so
        that the forward image of the reference distribution has variance one.
    gauss_legendre_order
        Number of Gauss-Legendre nodes per knot interval for the raw moments.
    integration_bounds
        Lower and upper bounds for the raw moments. Defaults to the first and last
        spline knot.
    reference_distribution
        Distribution of ``z = h(y)``, used only for centering, scaling and the raw
        moments. Defaults to ``Normal(0, 1)``. Must have no batch shape. Pass the
        same distribution that serves as base distribution; otherwise a centered
        or scaled bijector is not actually centered or scaled.
    validate_args
        Whether to validate input arguments.
    name
        Name of the bijector.

    Attributes
    ----------
    coef
        Unconstrained spline coefficients.
    bspline
        Spline object used for evaluation.
    centered
        Whether the spline input is centered.
    scaled
        Whether the spline input is scaled.
    gauss_legendre_order
        Number of Gauss-Legendre nodes per knot interval for the raw moments.
    integration_bounds
        Lower and upper bounds for the raw moments.
    integration_breaks
        Bounds and the knots between them, the intervals of the quadrature rule.
    reference_distribution
        Distribution of ``z = h(y)`` used for the raw moments.

    Notes
    -----
    With centering and scaling, the bijector evaluates
    ``inverse(y) = h(y * s + m)`` with raw mean ``m`` and raw standard deviation
    ``s``. Like other TFP bijectors, it supports batch slicing such as
    ``bijector[rows]`` and passing into ``jax.jit``. It has no ``mean()`` or
    ``variance()``: full response moments belong to the distribution.
    """

    def __init__(
        self,
        coef: Array,
        bspline: TransformationSpline,
        centered: bool = False,
        scaled: bool = False,
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
        reference_distribution: tfd.Distribution | None = None,
        validate_args: bool = False,
        name: str = "SplineBijector",
    ):
        coef = jnp.asarray(coef)
        parameters = dict(locals())
        if not jnp.issubdtype(coef.dtype, jnp.floating):
            raise TypeError("Spline coefficients must have a floating-point dtype.")
        bspline._check_coef_shape(coef)

        self.coef = coef
        self.bspline = bspline
        self.centered = centered
        self.scaled = scaled
        self.gauss_legendre_order = _validate_gauss_legendre_order(gauss_legendre_order)
        self.integration_bounds = _validate_integration_bounds(
            integration_bounds,
            (bspline._outer_knot_left, bspline._outer_knot_right),
        )
        self.integration_breaks = _integration_breaks_from_knots(
            bspline._knots_np, self.integration_bounds, coef.dtype
        )
        if reference_distribution is None:
            reference_distribution = tfd.Normal(loc=0.0, scale=1.0)
        self.reference_distribution = reference_distribution
        if centered or scaled:
            self._check_reference_distribution()

        super().__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            parameters=parameters,
            name=name,
            dtype=coef.dtype,
        )

    @classmethod
    def _parameter_properties(cls, dtype):
        return {"coef": tfp_parameter_properties.ParameterProperties(event_ndims=1)}

    @classmethod
    def _is_increasing(cls, **kwargs):
        return True

    def _check_reference_distribution(self) -> None:
        if tuple(self.reference_distribution.batch_shape) != ():
            raise ValueError(
                "The reference distribution of a SplineBijector must have no batch "
                "shape for centering, scaling and raw moments."
            )

    def _coef_batch_shape(self) -> tuple[int, ...]:
        return self.bspline._coef_batch_shape(self.coef)

    # Raw spline h, without centering and scaling.

    def _raw_value(self, y: Array) -> Array:
        y = jnp.asarray(y, dtype=self.dtype)
        transf = self.bspline.evaluate(
            y, self.coef, batch_shape=self._coef_batch_shape()
        )
        return jnp.where(jnp.isnan(y), jnp.nan, transf)

    def _raw_value_and_log_deriv(self, y: Array) -> tuple[Array, Array]:
        y = jnp.asarray(y, dtype=self.dtype)
        nan_mask = jnp.isnan(y)
        transf, deriv = self.bspline.evaluate_and_deriv(
            y, self.coef, batch_shape=self._coef_batch_shape()
        )
        transf = jnp.where(nan_mask, jnp.nan, transf)
        deriv = jnp.where(nan_mask, jnp.nan, deriv)
        # Safeguard against a numerically zero derivative.
        deriv = jnp.clip(deriv, min=jnp.finfo(self.dtype).tiny)
        return transf, jnp.log(deriv)

    def _raw_inverse_value(self, z: Array) -> Array:
        z = jnp.asarray(z, dtype=self.dtype)
        return self.bspline.inverse(z, self.coef, batch_shape=self._coef_batch_shape())

    # Centering and scaling.

    def _center_scale(self) -> tuple[Array, Array]:
        if self.centered:
            mean = self.mean_raw()
        else:
            mean = jnp.zeros((), dtype=self.dtype)

        if self.scaled:
            sd = jnp.sqrt(self.variance_raw(mean if self.centered else None))
        else:
            sd = jnp.ones((), dtype=self.dtype)

        return mean, sd

    def _forward(self, x: Array) -> Array:
        mean, sd = self._center_scale()
        return (self._raw_inverse_value(x) - mean) / sd

    def _inverse(self, y: Array) -> Array:
        mean, sd = self._center_scale()
        return self._raw_value(y * sd + mean)

    def _inverse_log_det_jacobian(self, y: Array) -> Array:
        return self._inverse_and_log_det_jacobian(y)[1]

    def _inverse_and_log_det_jacobian(self, y: Array) -> tuple[Array, Array]:
        """Evaluate ``inverse`` and its log-det-Jacobian in one pass."""
        mean, sd = self._center_scale()
        transf, log_deriv = self._raw_value_and_log_deriv(y * sd + mean)
        return transf, log_deriv + jnp.log(sd)

    # Raw moments used for centering and scaling.

    def _integrate(self, fn: Callable[[Array], Array], order: int | None) -> Array:
        self._check_reference_distribution()
        if order is None:
            order = self.gauss_legendre_order
        nodes, weights = _gauss_legendre_nodes_and_weights(order, self.dtype)
        return integrate_piecewise_gauss_legendre(
            fn,
            breaks=self.integration_breaks,
            nodes=nodes,
            weights=weights,
            batch_ndims=len(self._coef_batch_shape()),
        )

    def _raw_density(self, x: Array) -> Array:
        z, log_deriv = self._raw_value_and_log_deriv(x)
        return self.reference_distribution.prob(z) * jnp.exp(log_deriv)

    def _mean_raw(self, order: int | None = None) -> Array:
        return self._integrate(lambda x: x * self._raw_density(x), order)

    def _variance_raw(self, mean: Array | None = None, order: int | None = None):
        if mean is None:
            mean = self._mean_raw(order)
        return self._integrate(lambda x: (x - mean) ** 2 * self._raw_density(x), order)

    def mean_raw(self) -> Array:
        """
        Finite-bound mean of the raw spline distribution, used for centering.

        The raw spline distribution is the distribution of ``h⁻¹(Z)`` for the raw
        spline ``h``, without centering and scaling, where ``Z`` follows the
        reference distribution. The integral covers only the integration bounds,
        so this is not the full mean.

        Returns
        -------
        mean
            Raw mean with the coefficients' batch shape.
        """
        return self._mean_raw()

    def variance_raw(self, mean: Array | None = None) -> Array:
        """
        Finite-bound variance of the raw spline distribution, used for scaling.

        Parameters
        ----------
        mean
            Mean to compute the variance around. Defaults to
            [mean_raw][liesel_ptm.SplineBijector.mean_raw].

        Returns
        -------
        variance
            Raw variance with the coefficients' batch shape.
        """
        return self._variance_raw(mean)

    def moment_quadrature_diagnostic(
        self,
        reference_order: int | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> dict[str, Array]:
        """
        Compare the raw moments against a higher-order quadrature rule.

        Agreement between the two rules does not prove that omitted tails beyond
        the integration bounds are negligible.

        Parameters
        ----------
        reference_order
            Number of Gauss-Legendre nodes per knot interval for the comparison.
            Defaults to ``max(2 * order, order + 8)``. Must exceed
            ``gauss_legendre_order``.
        rtol
            Relative tolerance for ``ok``.
        atol
            Absolute tolerance for ``ok``.

        Returns
        -------
        diagnostic
            Raw moments (``mean``, ``variance``), their higher-order counterparts
            (``mean_reference``, ``variance_reference``), absolute and relative
            errors, and ``ok``, whether all errors are within tolerance.
        """
        order = self.gauss_legendre_order
        reference_order = _validate_reference_order(reference_order, order)
        mean = self._mean_raw(order)
        mean_reference = self._mean_raw(reference_order)
        variance = self._variance_raw(mean, order)
        variance_reference = self._variance_raw(mean_reference, reference_order)
        return _moment_diagnostic(
            (mean, variance),
            (mean_reference, variance_reference),
            rtol=rtol,
            atol=atol,
            dtype=self.dtype,
        )


class OnionSplineBijector(SplineBijector):
    """
    Onion spline transformation as a TFP bijector.

    Works like [SplineBijector][liesel_ptm.SplineBijector], with two properties
    specific to the [OnionSpline][liesel_ptm.OnionSpline]:

    - The coefficients are log increments. The spline adds the onion correction,
      so that ``h`` maps the core ``[min_knot, max_knot]`` onto itself.
    - Outside the core, ``h`` is exactly the identity: ``forward(z) = z`` and the
      inverse log-det-Jacobian is zero. So ``forward`` only inverts the spline
      numerically inside the core, on a fixed grid.

    Parameters
    ----------
    coef
        Spline log increments with shape ``batch_shape + (n_param,)``.
    bspline
        Onion spline that evaluates the transformation.
    centered
        If True, center the spline input. See
        [SplineBijector][liesel_ptm.SplineBijector].
    scaled
        If True, scale the spline input. See
        [SplineBijector][liesel_ptm.SplineBijector].
    gauss_legendre_order
        Number of Gauss-Legendre nodes per knot interval for the raw moments.
    integration_bounds
        Lower and upper bounds for the raw moments. Defaults to the first and last
        spline knot.
    reference_distribution
        Distribution of ``z = h(y)``, used only for centering, scaling and the raw
        moments. Defaults to ``Normal(0, 1)``.
    validate_args
        Whether to validate input arguments.
    name
        Name of the bijector.
    """

    def __init__(
        self,
        coef: Array,
        bspline: OnionSpline,
        centered: bool = False,
        scaled: bool = False,
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
        reference_distribution: tfd.Distribution | None = None,
        validate_args: bool = False,
        name: str = "OnionSplineBijector",
    ):
        super().__init__(
            coef=coef,
            bspline=bspline,
            centered=centered,
            scaled=scaled,
            gauss_legendre_order=gauss_legendre_order,
            integration_bounds=integration_bounds,
            reference_distribution=reference_distribution,
            validate_args=validate_args,
            name=name,
        )

    def _raw_inverse_value(self, z: Array) -> Array:
        z = jnp.asarray(z, dtype=self.dtype)
        bspline = self.bspline
        batch_shape = self._coef_batch_shape()

        value, sample_shape, value_batch_shape, result_batch_shape = (
            bspline._value_layout(z, batch_shape)
        )
        sample_size = int(np.prod(sample_shape)) if sample_shape else 1
        value = bspline._to_batch_last(value, value_batch_shape, sample_shape)
        # The layout pads value_batch_shape to the length of the coefficient batch
        # shape, so both broadcast to result_batch_shape directly.
        n_param = self.coef.shape[-1]
        value = jnp.broadcast_to(value, result_batch_shape + (sample_size,))
        value_rows = jnp.reshape(value, (-1, sample_size))
        coef = jnp.broadcast_to(self.coef, result_batch_shape + (n_param,))
        coef_rows = jnp.reshape(coef, (-1, n_param))

        inverse_rows = _onion_core_inverse_rows(bspline, value_rows, coef_rows)
        inverse = jnp.reshape(inverse_rows, result_batch_shape + (sample_size,))
        return bspline._from_batch_last(inverse, result_batch_shape, sample_shape)
