from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import jax.numpy as jnp
import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from .bspline import PTMSpline
from .dist import LocScaleTransformationDist

Array = Any
KeyArray = Any


def _as_float_array(value: Array) -> Array:
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype(jnp.float32)
    return value


def _nan_like(value: Array) -> Array:
    return jnp.full_like(value, jnp.nan)


def uncensored(value: Array) -> Array:
    """Create uncensored survival records ``[time, nan, nan]``."""
    value = _as_float_array(value)
    nan = _nan_like(value)
    return jnp.stack((value, nan, nan), axis=-1)


def left_censored(upper: Array) -> Array:
    """Create left-censored records ``[nan, nan, upper]``."""
    upper = _as_float_array(upper)
    nan = _nan_like(upper)
    return jnp.stack((nan, nan, upper), axis=-1)


def right_censored(lower: Array) -> Array:
    """Create right-censored records ``[nan, lower, nan]``."""
    lower = _as_float_array(lower)
    nan = _nan_like(lower)
    return jnp.stack((nan, lower, nan), axis=-1)


def interval_censored(lower: Array, upper: Array) -> Array:
    """Create interval-censored records ``[nan, lower, upper]``."""
    lower = _as_float_array(lower)
    upper = _as_float_array(upper)
    lower, upper = jnp.broadcast_arrays(lower, upper)
    nan = _nan_like(lower)
    return jnp.stack((nan, lower, upper), axis=-1)


def _logdiffexp(log_a: Array, log_b: Array) -> Array:
    """
    Compute ``log(exp(log_a) - exp(log_b))`` for ``log_a > log_b``.

    Invalid or zero differences are returned as ``-inf``. The implementation is
    NaN-safe for equal ``-inf`` inputs, which occur naturally in extreme tails.
    """
    neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.result_type(log_a, log_b))
    delta = log_b - log_a
    delta = jnp.where(jnp.isnan(delta), 0.0, delta)
    delta = jnp.minimum(delta, 0.0)
    value = log_a + jnp.log1p(-jnp.exp(delta))
    return jnp.where(log_a > log_b, value, neg_inf)


class CensoredDistribution(tfd.Distribution):
    """
    Generic censoring-record wrapper for scalar-event TFP distributions.

    The wrapped distribution models uncensored event times on its native scale.
    This wrapper models censoring records with event shape ``(3,)`` and layout
    ``[time, lower, upper]``:

    - ``[time, nan, nan]`` for uncensored observations.
    - ``[nan, nan, upper]`` for left-censored observations.
    - ``[nan, lower, nan]`` for right-censored observations.
    - ``[nan, lower, upper]`` for interval-censored observations.
    """

    def __init__(
        self,
        distribution: Callable[..., tfd.Distribution],
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "CensoredDistribution",
        **distribution_kwargs,
    ) -> None:
        parameters = dict(locals())
        self.distribution = distribution
        self.distribution_kwargs = distribution_kwargs
        self.base_distribution = distribution(
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            **distribution_kwargs,
        )

        if tuple(self.base_distribution.event_shape) != ():
            raise ValueError(
                "CensoredDistribution requires a scalar-event base distribution."
            )

        super().__init__(
            dtype=self.base_distribution.dtype,
            reparameterization_type=self.base_distribution.reparameterization_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    def _event_shape(self):
        return tf.TensorShape([3])

    def _event_shape_tensor(self):
        return jnp.array([3], dtype=jnp.int32)

    def _batch_shape(self):
        return self.base_distribution.batch_shape

    def _batch_shape_tensor(self):
        return self.base_distribution.batch_shape_tensor()

    def _as_record_value(self, value: Array) -> Array:
        if self.dtype is None:
            value = jnp.asarray(value)
        else:
            value = jnp.asarray(value, dtype=self.dtype)

        if value.ndim == 0 or value.shape[-1] != 3:
            raise ValueError(
                "Censored records must have trailing event shape (3,), with "
                "layout [time, lower, upper]."
            )

        return value

    def _base_log_cdf(self, value: Array) -> Array:
        try:
            return self.base_distribution.log_cdf(value)
        except (AttributeError, NotImplementedError):
            prob = self.base_distribution.cdf(value)
            tiny = jnp.finfo(jnp.asarray(prob).dtype).tiny
            return jnp.log(jnp.clip(prob, tiny, 1.0))

    def _base_log_survival_function(self, value: Array) -> Array:
        try:
            return self.base_distribution.log_survival_function(value)
        except (AttributeError, NotImplementedError):
            if hasattr(self.base_distribution, "survival_function"):
                prob = self.base_distribution.survival_function(value)
            else:
                prob = 1.0 - self.base_distribution.cdf(value)
            tiny = jnp.finfo(jnp.asarray(prob).dtype).tiny
            return jnp.log(jnp.clip(prob, tiny, 1.0))

    def _interval_log_prob(self, lower: Array, upper: Array) -> Array:
        log_cdf_lower = self._base_log_cdf(lower)
        log_cdf_upper = self._base_log_cdf(upper)
        log_sf_lower = self._base_log_survival_function(lower)
        log_sf_upper = self._base_log_survival_function(upper)

        cdf_diff = _logdiffexp(log_cdf_upper, log_cdf_lower)
        survival_diff = _logdiffexp(log_sf_lower, log_sf_upper)
        interval_log_prob = jnp.maximum(cdf_diff, survival_diff)

        return jnp.where(upper > lower, interval_log_prob, -jnp.inf)

    def log_prob_uncensored(self, value: Array) -> Array:
        """Log probability density/mass of uncensored event times."""
        return self.base_distribution.log_prob(value)

    def prob_uncensored(self, value: Array) -> Array:
        """Probability density/mass of uncensored event times."""
        return jnp.exp(self.log_prob_uncensored(value))

    def _log_prob(self, value: Array) -> Array:
        value = self._as_record_value(value)
        time = value[..., 0]
        lower = value[..., 1]
        upper = value[..., 2]

        time_nan = jnp.isnan(time)
        lower_nan = jnp.isnan(lower)
        upper_nan = jnp.isnan(upper)

        is_uncensored = ~time_nan & lower_nan & upper_nan
        is_left_censored = time_nan & lower_nan & ~upper_nan
        is_right_censored = time_nan & ~lower_nan & upper_nan
        is_interval_censored = time_nan & ~lower_nan & ~upper_nan

        zero = jnp.asarray(0.0, dtype=value.dtype)
        time_safe = jnp.where(is_uncensored, time, zero)
        lower_safe = jnp.where(is_right_censored, lower, zero)
        upper_safe = jnp.where(is_left_censored, upper, zero)
        interval_lower_safe = jnp.where(is_interval_censored, lower, zero - 1.0)
        interval_upper_safe = jnp.where(is_interval_censored, upper, zero + 1.0)

        uncensored_log_prob = self.base_distribution.log_prob(time_safe)
        left_censored_log_prob = self._base_log_cdf(upper_safe)
        right_censored_log_prob = self._base_log_survival_function(lower_safe)
        interval_censored_log_prob = self._interval_log_prob(
            interval_lower_safe, interval_upper_safe
        )

        finite_interval = jnp.where(
            jnp.isfinite(interval_censored_log_prob), interval_censored_log_prob, 0.0
        )
        shape_ref = (
            uncensored_log_prob
            + left_censored_log_prob
            + right_censored_log_prob
            + finite_interval
        )
        result = jnp.full_like(shape_ref, -jnp.inf)
        result = jnp.where(is_uncensored, uncensored_log_prob, result)
        result = jnp.where(is_left_censored, left_censored_log_prob, result)
        result = jnp.where(is_right_censored, right_censored_log_prob, result)
        result = jnp.where(is_interval_censored, interval_censored_log_prob, result)

        return result

    def _prob(self, value: Array) -> Array:
        return jnp.exp(self._log_prob(value))

    def _sample_n(self, n: int | Array, seed: KeyArray | None = None) -> Array:
        samples = self.base_distribution.sample(n, seed=seed)
        nan = jnp.full_like(samples, jnp.nan)
        return jnp.stack((samples, nan, nan), axis=-1)

    def _cdf(self, value: Array) -> Array:
        raise NotImplementedError(
            "CensoredDistribution.cdf is not defined for censoring records. "
            "Use .base_distribution.cdf(...) for uncensored event times."
        )

    def _quantile(self, value: Array) -> Array:
        raise NotImplementedError(
            "CensoredDistribution.quantile is not defined for censoring records. "
            "Use .base_distribution.quantile(...) for uncensored event times."
        )


class CensoredPTMDist(lsl.Dist):
    """
    Liesel distribution helper for censored location-scale PTMs.

    This wraps :class:`.LocScaleTransformationDist` in :class:`CensoredDistribution`.
    Observed response values must be censoring records with trailing shape ``(3,)``.
    """

    def __init__(
        self,
        knots: Array,
        loc: lsl.Var,
        scale: lsl.Var,
        shape: lsl.Var,
        centered: bool = False,
        scaled: bool = False,
        trafo_target_slope: Literal["continue_linearly", "identity"] = "identity",
        trafo_lambda: float | None = None,
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
        **kwargs,
    ) -> None:
        if trafo_target_slope not in ("continue_linearly", "identity"):
            raise ValueError(
                "trafo_target_slope must be either 'continue_linearly' or 'identity'."
            )

        eps = 0.1 if trafo_lambda is None else float(trafo_lambda)
        bspline = PTMSpline(
            knots=knots,
            eps=eps,
            continue_linearly=trafo_target_slope == "continue_linearly",
        )

        partial_dist_class = partial(
            CensoredDistribution,
            distribution=LocScaleTransformationDist,
            bspline=bspline,
            centered=centered,
            scaled=scaled,
            gauss_legendre_order=gauss_legendre_order,
            integration_bounds=integration_bounds,
        )

        super().__init__(partial_dist_class, loc=loc, scale=scale, coef=shape, **kwargs)
