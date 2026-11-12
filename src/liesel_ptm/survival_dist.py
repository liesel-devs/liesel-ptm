from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, NamedTuple, Self, cast

import jax
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.typing import ArrayLike
from liesel.model.nodes import VarValue
from tensorflow_probability.substrates.jax import tf2jax as tf

Array = Any
KeyArray = Any
type ValuesLike = ArrayLike | Sequence[Any]


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


CENSORING_STATUS_LABELS = (
    "uncensored",
    "left_censored",
    "right_censored",
    "interval_censored",
)


def _record_status_masks(
    time: Array, lower: Array, upper: Array, xp: Any = jnp
) -> tuple[Array, Array, Array, Array]:
    """Classify records by their missing-value pattern.

    Returns masks in the order of ``CENSORING_STATUS_LABELS``. Every other
    pattern is invalid. The likelihood, record splitting and
    :func:`censoring_status` all use this classification.
    """
    time_nan = xp.isnan(time)
    lower_nan = xp.isnan(lower)
    upper_nan = xp.isnan(upper)
    return (
        ~time_nan & lower_nan & upper_nan,
        time_nan & lower_nan & ~upper_nan,
        time_nan & ~lower_nan & upper_nan,
        time_nan & ~lower_nan & ~upper_nan,
    )


def _input_array(value: ValuesLike) -> tuple[Array, bool]:
    """Return a value at its input precision and whether it is concrete.

    Concrete values are classified with NumPy before any conversion to the JAX
    default dtype can erase a positive interval width. Traced values are
    classified with JAX at their traced precision and cannot be validated.
    """
    try:
        array = np.asarray(value)
    except jax.errors.TracerArrayConversionError:
        return jnp.asarray(value), False
    if not np.issubdtype(array.dtype, np.inexact):
        array = array.astype(np.float64)
    return array, True


def _event_mask(event_observed: ValuesLike) -> tuple[Array, bool]:
    try:
        mask = np.asarray(event_observed)
    except jax.errors.TracerArrayConversionError:
        return jnp.asarray(event_observed) != 0, False
    if mask.dtype == np.bool_:
        return mask, True
    if not (np.issubdtype(mask.dtype, np.number) and np.isin(mask, (0, 1)).all()):
        raise ValueError(
            "event_observed must be a Boolean value or mask containing only 0 and 1."
        )
    return mask != 0, True


def _encode(
    *,
    exact: Array,
    left: Array,
    right: Array,
    interval: Array,
    time: ValuesLike,
    lower: ValuesLike,
    upper: ValuesLike,
) -> Array:
    time = _as_float_array(time)
    lower = _as_float_array(lower)
    upper = _as_float_array(upper)
    dtype = jnp.result_type(time, lower, upper)
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    time = jnp.where(exact, time.astype(dtype), nan)
    lower = jnp.where(right | interval, lower.astype(dtype), nan)
    upper = jnp.where(left | interval, upper.astype(dtype), nan)
    time, lower, upper = jnp.broadcast_arrays(time, lower, upper)
    return jnp.stack((time, lower, upper), axis=-1)


def _require(invalid: Array, message: str) -> None:
    if np.any(invalid):
        raise ValueError(message)


def _one_sided_censored(
    value: ValuesLike, event_observed: ValuesLike, *, side: str
) -> Array:
    array, concrete_value = _input_array(value)
    observed, concrete_mask = _event_mask(event_observed)
    if concrete_value and concrete_mask:
        _require(
            observed & ~np.isfinite(array),
            "Observed event times (event_observed=True) must be finite.",
        )
    censored_rows = ~observed
    no_rows = jnp.zeros((), dtype=bool)
    return _encode(
        exact=observed,
        left=censored_rows if side == "left" else no_rows,
        right=censored_rows if side == "right" else no_rows,
        interval=no_rows,
        time=value,
        lower=value,
        upper=value,
    )


def left_censored(upper: ValuesLike, *, event_observed: ValuesLike = False) -> Array:
    """Encode left-censored observations, optionally mixed with exact ones.

    Parameters
    ----------
    upper
        Upper bounds of censored observations. Where ``event_observed`` is true,
        the same value is the exact event time instead.
    event_observed
        Scalar flag or mask broadcastable with ``upper``, either Boolean or
        containing only 0 and 1. By default, every observation is left-censored.

    Returns
    -------
    jax.Array
        Records with trailing columns ``[time, lower, upper]``:
        ``[nan, nan, upper]`` for censored and ``[upper, nan, nan]`` for observed
        rows.

    Raises
    ------
    ValueError
        If ``event_observed`` is not binary or an observed time is not finite.

    Examples
    --------
    >>> from liesel_ptm import survival_dist as surv
    >>> surv.left_censored([1.0, 2.0], event_observed=[False, True])
    Array([[nan, nan,  1.],
           [ 2., nan, nan]], dtype=float32)
    """
    return _one_sided_censored(upper, event_observed, side="left")


def right_censored(lower: ValuesLike, *, event_observed: ValuesLike = False) -> Array:
    """Encode right-censored observations, optionally mixed with exact ones.

    Parameters
    ----------
    lower
        Lower bounds of censored observations, such as follow-up times. Where
        ``event_observed`` is true, the same value is the exact event time
        instead.
    event_observed
        Scalar flag or mask broadcastable with ``lower``, either Boolean or
        containing only 0 and 1. By default, every observation is right-censored.

    Returns
    -------
    jax.Array
        Records with trailing columns ``[time, lower, upper]``:
        ``[nan, lower, nan]`` for censored and ``[lower, nan, nan]`` for observed
        rows.

    Raises
    ------
    ValueError
        If ``event_observed`` is not binary or an observed time is not finite.

    Examples
    --------
    >>> from liesel_ptm import survival_dist as surv
    >>> surv.right_censored([1.0, 2.0], event_observed=[True, False])
    Array([[ 1., nan, nan],
           [nan,  2., nan]], dtype=float32)
    """
    return _one_sided_censored(lower, event_observed, side="right")


def interval_censored(
    lower: ValuesLike, upper: ValuesLike, *, event_observed: ValuesLike = False
) -> Array:
    """Encode interval-censored observations, optionally mixed with exact ones.

    Parameters
    ----------
    lower, upper
        Broadcastable interval bounds. Censored rows require
        ``lower < upper``; an infinite endpoint is kept as an interval bound.
        Observed rows require matching finite bounds, the exact event time.
    event_observed
        Scalar flag or mask broadcastable with the bounds, either Boolean or
        containing only 0 and 1. By default, every observation is
        interval-censored.

    Returns
    -------
    jax.Array
        Records with trailing columns ``[time, lower, upper]``:
        ``[nan, lower, upper]`` for censored and ``[lower, nan, nan]`` for
        observed rows.

    Raises
    ------
    ValueError
        If ``event_observed`` is not binary, a censored row lacks positive
        width, or an observed row lacks matching finite bounds.

    Notes
    -----
    Widths are checked at input precision. The interval status is kept even if
    conversion to the output dtype collapses the bounds;
    [PTM.from_censored][liesel_ptm.PTM.from_censored] then rejects the records.

    Examples
    --------
    >>> from liesel_ptm import survival_dist as surv
    >>> surv.interval_censored([0.0, 1.0], [2.0, 1.0], event_observed=[0, 1])
    Array([[nan,  0.,  2.],
           [ 1., nan, nan]], dtype=float32)
    """
    lower_array, concrete_lower = _input_array(lower)
    upper_array, concrete_upper = _input_array(upper)
    observed, concrete_mask = _event_mask(event_observed)
    if concrete_lower and concrete_upper and concrete_mask:
        observed, lower_array, upper_array = np.broadcast_arrays(
            observed, lower_array, upper_array
        )
        _require(
            observed & ~((lower_array == upper_array) & np.isfinite(lower_array)),
            "Observed rows (event_observed=True) require matching finite bounds.",
        )
        _require(
            ~observed & ~(upper_array > lower_array),
            "Interval-censored rows require positive-width intervals, lower < upper.",
        )
    no_rows = jnp.zeros((), dtype=bool)
    return _encode(
        exact=observed,
        left=no_rows,
        right=no_rows,
        interval=~observed,
        time=lower,
        lower=lower,
        upper=upper,
    )


def censored(
    lower: ValuesLike, upper: ValuesLike, *, event_observed: ValuesLike | None = None
) -> Array:
    """Encode observations from their bounds as censoring records.

    Parameters
    ----------
    lower, upper
        Broadcastable bounds with ``lower <= upper``. Use ``-inf`` as the lower
        bound of a left-censored and ``inf`` as the upper bound of a
        right-censored observation. Matching finite bounds give an exact time.
    event_observed
        ``None`` infers every observation type from the bounds. Otherwise, a
        scalar flag or mask broadcastable with the bounds, either Boolean or
        containing only 0 and 1. True rows are exact and require matching finite
        bounds; false rows are censored and require ``lower < upper``.

    Returns
    -------
    jax.Array
        Records with trailing columns ``[time, lower, upper]``. Exact,
        left-censored, right-censored, and interval-censored rows are
        ``[t, nan, nan]``, ``[nan, nan, upper]``, ``[nan, lower, nan]``, and
        ``[nan, lower, upper]``.

    Raises
    ------
    ValueError
        If ``event_observed`` is not binary, or bounds are missing, decreasing,
        both infinite, infinite on the wrong side, or inconsistent with
        ``event_observed``.

    Notes
    -----
    Bounds are compared with exact equality at input precision, before
    conversion to the output dtype. An interval whose bounds collapse during
    that conversion stays interval-censored; it never becomes exact.

    Examples
    --------
    >>> import numpy as np
    >>> from liesel_ptm import survival_dist as surv
    >>> records = surv.censored([1.0, -np.inf, 2.0, 3.0], [1.0, 4.0, np.inf, 5.0])
    >>> surv.censoring_status(records).tolist()
    ['uncensored', 'left_censored', 'right_censored', 'interval_censored']
    """
    lower_array, concrete_lower = _input_array(lower)
    upper_array, concrete_upper = _input_array(upper)
    if event_observed is None:
        observed, concrete_mask = None, True
    else:
        observed, concrete_mask = _event_mask(event_observed)
    concrete = concrete_lower and concrete_upper and concrete_mask
    xp: Any = np if concrete else jnp
    if not concrete:
        lower_array = jnp.asarray(lower_array)
        upper_array = jnp.asarray(upper_array)
    finite_lower = xp.isfinite(lower_array)
    finite_upper = xp.isfinite(upper_array)
    matching = finite_lower & (lower_array == upper_array)
    exact = matching if observed is None else xp.asarray(observed)
    censored_rows = ~exact
    left = censored_rows & (lower_array == -xp.inf) & finite_upper
    right = censored_rows & finite_lower & (upper_array == xp.inf)
    interval = censored_rows & finite_lower & finite_upper & (lower_array < upper_array)
    if concrete:
        _require(
            exact & ~matching,
            "Observed rows (event_observed=True) require matching finite bounds.",
        )
        _require(
            censored_rows & ~(left | right | interval),
            "Censored rows require lower < upper, without NaN and with at most one "
            "infinite endpoint: -inf for lower or inf for upper.",
        )
    return _encode(
        exact=exact,
        left=left,
        right=right,
        interval=interval,
        time=lower,
        lower=lower,
        upper=upper,
    )


def censoring_status(records: ValuesLike) -> np.ndarray:
    """Report the observation type encoded in censoring records.

    Parameters
    ----------
    records
        Records with trailing columns ``[time, lower, upper]``.

    Returns
    -------
    numpy.ndarray
        String labels with the record batch shape: ``"uncensored"``,
        ``"left_censored"``, ``"right_censored"``, ``"interval_censored"``, or
        ``"invalid"`` for any other missing-value pattern. Interval widths are
        not checked.

    Raises
    ------
    ValueError
        If the records lack a trailing dimension of size 3.

    Examples
    --------
    >>> from liesel_ptm import survival_dist as surv
    >>> surv.censoring_status(surv.right_censored([1.0, 2.0], event_observed=[1, 0]))
    array(['uncensored', 'right_censored'], dtype='<U17')
    """
    array = _as_record_array(records)
    masks = _record_status_masks(array[..., 0], array[..., 1], array[..., 2], xp=np)
    return np.select(masks, CENSORING_STATUS_LABELS, default="invalid")


def _as_record_array(records: ValuesLike) -> np.ndarray:
    array = np.asarray(records)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(
            "Censored records must have trailing event shape (3,), with "
            "layout [time, lower, upper]."
        )
    if not np.issubdtype(array.dtype, np.inexact):
        array = array.astype(np.float64)
    return array


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


def _log_from_probability(prob: Array) -> Array:
    prob = jnp.asarray(prob)
    neg_inf = jnp.asarray(-jnp.inf, dtype=prob.dtype)
    nan = jnp.asarray(jnp.nan, dtype=prob.dtype)
    valid = jnp.isfinite(prob) & (prob >= 0.0) & (prob <= 1.0)
    positive = valid & (prob > 0.0)
    zero = valid & (prob == 0.0)
    safe_prob = jnp.where(positive, prob, 1.0)
    log_prob = jnp.log(safe_prob)
    return jnp.where(positive, log_prob, jnp.where(zero, neg_inf, nan))


def _distribution_log_cdf(distribution: tfd.Distribution, value: Array) -> Array:
    try:
        return distribution.log_cdf(value)
    except (AttributeError, NotImplementedError):
        # If the public log-CDF is unavailable, fall back to the public CDF.
        # Zero probability stays -inf; it is not clipped to a finite value.
        prob = distribution.cdf(value)
        return _log_from_probability(prob)


def _distribution_log_survival_function(
    distribution: tfd.Distribution, value: Array
) -> Array:
    try:
        return distribution.log_survival_function(value)
    except (AttributeError, NotImplementedError):
        # If the public log-survival method is unavailable, fall back to public
        # probability methods. Returned -inf values from an available base
        # log-survival method are trusted: the wrapper cannot distinguish true
        # zero probability from distribution-specific tail underflow without
        # second-guessing the base distribution.
        try:
            prob = distribution.survival_function(value)
        except (AttributeError, NotImplementedError):
            prob = 1.0 - distribution.cdf(value)
        return _log_from_probability(prob)


class _BaseCensoringDistribution(tfd.Distribution):
    _event_shape_tuple: tuple[int, ...] = ()

    def __init__(
        self,
        distribution: Callable[..., tfd.Distribution],
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "BaseCensoringDistribution",
        **distribution_kwargs,
    ) -> None:
        parameters = dict(locals())
        self.distribution = distribution
        self.distribution_kwargs = distribution_kwargs
        self._base_validate_args = validate_args
        self._base_allow_nan_stats = allow_nan_stats
        self.base_distribution = distribution(
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            **distribution_kwargs,
        )

        if tuple(self.base_distribution.event_shape) != ():
            raise ValueError(
                f"{type(self).__name__} requires a scalar-event base distribution."
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
        return tf.TensorShape(self._event_shape_tuple)

    def _event_shape_tensor(self):
        return jnp.asarray(self._event_shape_tuple, dtype=jnp.int32)

    def _batch_shape(self):
        return self.base_distribution.batch_shape

    def _batch_shape_tensor(self, **kwargs):
        return self.base_distribution.batch_shape_tensor()

    def _as_base_value(self, value: Array) -> Array:
        if self.dtype is None:
            return jnp.asarray(value)
        return jnp.asarray(value, dtype=self.dtype)

    def _base_log_cdf(self, value: Array) -> Array:
        return _distribution_log_cdf(self.base_distribution, value)

    def _base_log_survival_function(self, value: Array) -> Array:
        return _distribution_log_survival_function(self.base_distribution, value)

    def _interval_log_prob(self, lower: Array, upper: Array) -> Array:
        log_cdf_lower = self._base_log_cdf(lower)
        log_cdf_upper = self._base_log_cdf(upper)
        log_sf_lower = self._base_log_survival_function(lower)
        log_sf_upper = self._base_log_survival_function(upper)

        cdf_diff = _logdiffexp(log_cdf_upper, log_cdf_lower)
        survival_diff = _logdiffexp(log_sf_lower, log_sf_upper)
        interval_log_prob = jnp.maximum(cdf_diff, survival_diff)

        return jnp.where(upper > lower, interval_log_prob, -jnp.inf)

    def _prob(self, value: Array) -> Array:
        return jnp.exp(self.log_prob(value))

    def _sample_n(
        self, n: int | Array, seed: KeyArray | None = None, **kwargs
    ) -> Array:
        raise NotImplementedError(
            f"{type(self).__name__}.sample is not defined for censoring bounds. "
            "Use .base_distribution.sample(...) for uncensored event times."
        )

    def _cdf(self, value: Array) -> Array:
        raise NotImplementedError(
            f"{type(self).__name__}.cdf is not defined for censoring records. "
            "Use .base_distribution.cdf(...) for uncensored event times."
        )

    def _quantile(self, value: Array, **kwargs) -> Array:
        raise NotImplementedError(
            f"{type(self).__name__}.quantile is not defined for censoring records. "
            "Use .base_distribution.quantile(...) for uncensored event times."
        )


class LeftCensoredDistribution(_BaseCensoringDistribution):
    """Wrapper for observations known only to satisfy ``T <= upper``."""

    def __init__(
        self,
        distribution: Callable[..., tfd.Distribution],
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "LeftCensoredDistribution",
        **distribution_kwargs,
    ) -> None:
        super().__init__(
            distribution=distribution,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            **distribution_kwargs,
        )

    def _log_prob(self, value: Array) -> Array:
        return self._base_log_cdf(self._as_base_value(value))


class RightCensoredDistribution(_BaseCensoringDistribution):
    """Wrapper for observations known only to satisfy ``T > lower``."""

    def __init__(
        self,
        distribution: Callable[..., tfd.Distribution],
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "RightCensoredDistribution",
        **distribution_kwargs,
    ) -> None:
        super().__init__(
            distribution=distribution,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            **distribution_kwargs,
        )

    def _log_prob(self, value: Array) -> Array:
        return self._base_log_survival_function(self._as_base_value(value))


class IntervalCensoredDistribution(_BaseCensoringDistribution):
    """Wrapper for interval observations with event layout ``[lower, upper]``."""

    _event_shape_tuple = (2,)

    def __init__(
        self,
        distribution: Callable[..., tfd.Distribution],
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "IntervalCensoredDistribution",
        **distribution_kwargs,
    ) -> None:
        super().__init__(
            distribution=distribution,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            **distribution_kwargs,
        )

    def _as_interval_value(self, value: Array) -> Array:
        value = self._as_base_value(value)
        if value.ndim == 0 or value.shape[-1] != 2:
            raise ValueError(
                "Interval-censored observations must have trailing event shape "
                "(2,), with layout [lower, upper]."
            )
        return value

    def _log_prob(self, value: Array) -> Array:
        value = self._as_interval_value(value)
        lower = value[..., 0]
        upper = value[..., 1]
        return self._interval_log_prob(lower, upper)


class CensoredDistribution(_BaseCensoringDistribution):
    """
    Generic censoring-record wrapper for scalar-event TFP distributions.

    The wrapped distribution models uncensored event times on its native scale.
    This wrapper models censoring records with event shape ``(3,)`` and layout
    ``[time, lower, upper]``:

    - ``[time, nan, nan]`` for uncensored observations.
    - ``[nan, nan, upper]`` for left-censored observations.
    - ``[nan, lower, nan]`` for right-censored observations.
    - ``[nan, lower, upper]`` for interval-censored observations.

    Optionally, pass ``censoring_records`` at construction time to cache the
    censoring indicators for a fixed observed data set. In cached mode,
    ``log_prob(value)`` still expects full records with trailing event shape
    ``(3,)``; the cached indicators determine which likelihood branch is
    evaluated along the cached observation axis.
    """

    _event_shape_tuple = (3,)

    def __init__(
        self,
        distribution: Callable[..., tfd.Distribution],
        censoring_records: ArrayLike | None = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "CensoredDistribution",
        **distribution_kwargs,
    ) -> None:
        super().__init__(
            distribution=distribution,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            **distribution_kwargs,
        )
        self.censoring_records: Array | None = None
        self._cached_record_shape: tuple[int, ...] | None = None
        self._cached_record_batch_shape: tuple[int, ...] | None = None
        self._cached_result_batch_shape: tuple[int, ...] | None = None
        self._cached_padded_record_batch_shape: tuple[int, ...] | None = None
        self._cached_observation_axis = 0
        self._cached_n_observations = 0
        self._cached_uncensored_indices = jnp.asarray([], dtype=jnp.int32)
        self._cached_left_censored_indices = jnp.asarray([], dtype=jnp.int32)
        self._cached_right_censored_indices = jnp.asarray([], dtype=jnp.int32)
        self._cached_interval_censored_indices = jnp.asarray([], dtype=jnp.int32)
        self._cached_invalid_indices = jnp.asarray([], dtype=jnp.int32)
        self._n_cached_uncensored = 0
        self._n_cached_left_censored = 0
        self._n_cached_right_censored = 0
        self._n_cached_interval_censored = 0
        self._n_cached_invalid = 0

        if censoring_records is not None:
            self._set_cached_censoring_records(censoring_records)

    def _set_cached_censoring_records(self, censoring_records: ArrayLike) -> None:
        try:
            records_np = np.asarray(censoring_records)
        except Exception as err:
            raise TypeError(
                "censoring_records must be a concrete array available at "
                "distribution construction time."
            ) from err

        if not np.issubdtype(records_np.dtype, np.inexact):
            records_np = records_np.astype(np.float32)

        if records_np.ndim < 2 or records_np.shape[-1] != 3:
            raise ValueError(
                "censoring_records must have trailing event shape (3,) and "
                "at least one observation batch axis."
            )

        record_batch_shape = tuple(int(dim) for dim in records_np.shape[:-1])
        base_batch_shape = tuple(
            int(dim) for dim in tuple(self.base_distribution.batch_shape)
        )
        try:
            result_batch_shape = tuple(
                int(dim)
                for dim in np.broadcast_shapes(record_batch_shape, base_batch_shape)
            )
        except ValueError as err:
            raise ValueError(
                "censoring_records batch shape must broadcast with the base "
                f"distribution batch shape. Got {record_batch_shape} and "
                f"{base_batch_shape}."
            ) from err

        rank = len(result_batch_shape)
        padded_record_batch_shape = (1,) * (
            rank - len(record_batch_shape)
        ) + record_batch_shape
        non_singleton_axes = [
            axis for axis, dim in enumerate(padded_record_batch_shape) if dim != 1
        ]
        if not non_singleton_axes:
            raise ValueError(
                "censoring_records must include a non-singleton observation batch axis."
            )

        observation_axis = non_singleton_axes[-1]
        if any(dim != 1 for dim in padded_record_batch_shape[:observation_axis]):
            raise ValueError(
                "Cached censoring records may only vary along one compact "
                "observation axis; leading record batch axes must be singleton."
            )

        records_padded = np.reshape(
            records_np, padded_record_batch_shape + records_np.shape[-1:]
        )
        records_compact = np.reshape(
            np.moveaxis(records_padded, observation_axis, -2), (-1, 3)
        )

        (
            is_uncensored,
            is_left_censored,
            is_right_censored,
            is_interval_censored,
        ) = _record_status_masks(
            records_compact[..., 0],
            records_compact[..., 1],
            records_compact[..., 2],
            xp=np,
        )
        is_valid = (
            is_uncensored | is_left_censored | is_right_censored | is_interval_censored
        )

        self.censoring_records = jnp.asarray(records_np, dtype=self.dtype)
        self._cached_record_shape = tuple(int(dim) for dim in records_np.shape)
        self._cached_record_batch_shape = record_batch_shape
        self._cached_result_batch_shape = result_batch_shape
        self._cached_padded_record_batch_shape = padded_record_batch_shape
        self._cached_observation_axis = observation_axis
        self._cached_n_observations = int(padded_record_batch_shape[observation_axis])
        self._cached_uncensored_indices = jnp.asarray(
            np.flatnonzero(is_uncensored), dtype=jnp.int32
        )
        self._cached_left_censored_indices = jnp.asarray(
            np.flatnonzero(is_left_censored), dtype=jnp.int32
        )
        self._cached_right_censored_indices = jnp.asarray(
            np.flatnonzero(is_right_censored), dtype=jnp.int32
        )
        self._cached_interval_censored_indices = jnp.asarray(
            np.flatnonzero(is_interval_censored), dtype=jnp.int32
        )
        self._cached_invalid_indices = jnp.asarray(
            np.flatnonzero(~is_valid), dtype=jnp.int32
        )
        self._n_cached_uncensored = int(np.sum(is_uncensored))
        self._n_cached_left_censored = int(np.sum(is_left_censored))
        self._n_cached_right_censored = int(np.sum(is_right_censored))
        self._n_cached_interval_censored = int(np.sum(is_interval_censored))
        self._n_cached_invalid = int(np.sum(~is_valid))

    def _as_record_value(self, value: Array) -> Array:
        value = self._as_base_value(value)
        if value.ndim == 0 or value.shape[-1] != 3:
            raise ValueError(
                "Censored records must have trailing event shape (3,), with "
                "layout [time, lower, upper]."
            )

        return value

    def log_prob_uncensored(self, value: Array) -> Array:
        """Log probability density/mass of uncensored event times."""
        return self.base_distribution.log_prob(value)

    def prob_uncensored(self, value: Array) -> Array:
        """Probability density/mass of uncensored event times."""
        return jnp.exp(self.log_prob_uncensored(value))

    def _log_prob(self, value: Array) -> Array:
        value = self._as_record_value(value)
        if self.censoring_records is not None:
            return self._log_prob_cached(value)

        return self._log_prob_mixed(value)

    def _log_prob_mixed(self, value: Array) -> Array:
        time = value[..., 0]
        lower = value[..., 1]
        upper = value[..., 2]

        (
            is_uncensored,
            is_left_censored,
            is_right_censored,
            is_interval_censored,
        ) = _record_status_masks(time, lower, upper)

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

    @staticmethod
    def _left_pad_shape(shape: tuple[int, ...], rank: int) -> tuple[int, ...]:
        if len(shape) > rank:
            raise ValueError(f"Cannot left-pad shape {shape} to shorter rank {rank}.")

        return (1,) * (rank - len(shape)) + shape

    def _cached_value_padded(self, value: Array) -> Array:
        if self._cached_result_batch_shape is None:
            raise RuntimeError("Cached censoring records are not initialized.")

        value_batch_shape = tuple(int(dim) for dim in value.shape[:-1])
        try:
            result_batch_shape = tuple(
                int(dim)
                for dim in np.broadcast_shapes(
                    value_batch_shape, self._cached_result_batch_shape
                )
            )
        except ValueError as err:
            raise ValueError(
                "Cached CensoredDistribution requires value batch shape to "
                "broadcast with cached records and base distribution batch "
                f"shape. Got {value_batch_shape} and "
                f"{self._cached_result_batch_shape}."
            ) from err

        if result_batch_shape != self._cached_result_batch_shape:
            raise ValueError(
                "Cached CensoredDistribution does not support additional sample "
                "axes beyond the cached/base batch shape in this path. Got "
                f"{value_batch_shape}; expected broadcast result "
                f"{self._cached_result_batch_shape}."
            )

        padded_value_batch_shape = self._left_pad_shape(
            value_batch_shape, len(self._cached_result_batch_shape)
        )
        return jnp.reshape(value, padded_value_batch_shape + (3,))

    def _cached_branch_value(
        self, value_padded: Array, component: int, indices: Array
    ) -> Array:
        if value_padded.shape[self._cached_observation_axis] == 1:
            indices = jnp.zeros_like(indices)

        return jnp.take(
            value_padded[..., component],
            indices,
            axis=self._cached_observation_axis,
        )

    def _cached_kwarg_batch_shape_and_core_ndims(
        self, key: str, value_array: Array
    ) -> tuple[tuple[int, ...], int] | None:
        if self._cached_result_batch_shape is None:
            return None

        candidate_core_ndims = (1, 0) if key == "coef" else (0, 1)
        for core_ndims in candidate_core_ndims:
            if value_array.ndim < core_ndims:
                continue

            if core_ndims:
                batch_shape = tuple(int(dim) for dim in value_array.shape[:-core_ndims])
            else:
                batch_shape = tuple(int(dim) for dim in value_array.shape)

            try:
                broadcast_shape = tuple(
                    int(dim)
                    for dim in np.broadcast_shapes(
                        batch_shape, self._cached_result_batch_shape
                    )
                )
            except ValueError:
                continue

            if broadcast_shape == self._cached_result_batch_shape:
                return batch_shape, core_ndims

        return None

    def _subset_cached_distribution_kwarg(
        self, key: str, value: Any, indices: Array
    ) -> Any:
        try:
            value_array = jnp.asarray(value)
        except (TypeError, ValueError):
            return value

        batch_and_core = self._cached_kwarg_batch_shape_and_core_ndims(key, value_array)
        if batch_and_core is None or self._cached_result_batch_shape is None:
            return value

        batch_shape, _ = batch_and_core
        padded_batch_shape = self._left_pad_shape(
            batch_shape, len(self._cached_result_batch_shape)
        )
        if padded_batch_shape[self._cached_observation_axis] != (
            self._cached_n_observations
        ):
            return value

        array_axis = self._cached_observation_axis - (
            len(self._cached_result_batch_shape) - len(batch_shape)
        )
        if array_axis < 0:
            return value

        return jnp.take(value_array, indices, axis=array_axis)

    def _cached_base_distribution(self, indices: Array) -> tfd.Distribution:
        distribution_kwargs = {
            key: self._subset_cached_distribution_kwarg(key, value, indices)
            for key, value in self.distribution_kwargs.items()
        }
        return self.distribution(
            validate_args=self._base_validate_args,
            allow_nan_stats=self._base_allow_nan_stats,
            **distribution_kwargs,
        )

    def _scatter_cached_log_prob(
        self, result: Array, indices: Array, branch_log_prob: Array
    ) -> Array:
        branch_shape = (
            result.shape[: self._cached_observation_axis]
            + (indices.shape[0],)
            + result.shape[self._cached_observation_axis + 1 :]
        )
        branch_log_prob = jnp.broadcast_to(jnp.asarray(branch_log_prob), branch_shape)
        result_moved = jnp.moveaxis(result, self._cached_observation_axis, -1)
        branch_moved = jnp.moveaxis(branch_log_prob, self._cached_observation_axis, -1)
        result_moved = result_moved.at[..., indices].set(branch_moved)
        return jnp.moveaxis(result_moved, -1, self._cached_observation_axis)

    def _log_prob_cached(self, value: Array) -> Array:
        if self._cached_record_shape is None or self._cached_result_batch_shape is None:
            raise RuntimeError("Cached censoring records are not initialized.")

        value_padded = self._cached_value_padded(value)
        result = jnp.full(self._cached_result_batch_shape, -jnp.inf, dtype=value.dtype)

        if self._n_cached_uncensored:
            indices = self._cached_uncensored_indices
            branch_distribution = self._cached_base_distribution(indices)
            branch_value = self._cached_branch_value(value_padded, 0, indices)
            result = self._scatter_cached_log_prob(
                result, indices, branch_distribution.log_prob(branch_value)
            )

        if self._n_cached_left_censored:
            indices = self._cached_left_censored_indices
            branch_distribution = self._cached_base_distribution(indices)
            branch_value = self._cached_branch_value(value_padded, 2, indices)
            result = self._scatter_cached_log_prob(
                result,
                indices,
                _distribution_log_cdf(branch_distribution, branch_value),
            )

        if self._n_cached_right_censored:
            indices = self._cached_right_censored_indices
            branch_distribution = self._cached_base_distribution(indices)
            branch_value = self._cached_branch_value(value_padded, 1, indices)
            result = self._scatter_cached_log_prob(
                result,
                indices,
                _distribution_log_survival_function(branch_distribution, branch_value),
            )

        if self._n_cached_interval_censored:
            indices = self._cached_interval_censored_indices
            branch_distribution = self._cached_base_distribution(indices)
            lower = self._cached_branch_value(value_padded, 1, indices)
            upper = self._cached_branch_value(value_padded, 2, indices)
            lower_log_cdf = _distribution_log_cdf(branch_distribution, lower)
            upper_log_cdf = _distribution_log_cdf(branch_distribution, upper)
            lower_log_sf = _distribution_log_survival_function(
                branch_distribution, lower
            )
            upper_log_sf = _distribution_log_survival_function(
                branch_distribution, upper
            )
            cdf_diff = _logdiffexp(upper_log_cdf, lower_log_cdf)
            survival_diff = _logdiffexp(lower_log_sf, upper_log_sf)
            interval_log_prob = jnp.where(
                upper > lower, jnp.maximum(cdf_diff, survival_diff), -jnp.inf
            )
            result = self._scatter_cached_log_prob(result, indices, interval_log_prob)

        return result

    def _sample_n(
        self, n: int | Array, seed: KeyArray | None = None, **kwargs
    ) -> Array:
        samples = self.base_distribution.sample(n, seed=seed)
        nan = jnp.full_like(samples, jnp.nan)
        return jnp.stack((samples, nan, nan), axis=-1)


class CensoredDist(lsl.Dist):
    """Liesel distribution node that evaluates censoring records.

    Wraps a scalar-event distribution in a
    [CensoredDistribution][liesel_ptm.survival_dist.CensoredDistribution]. Its
    variable holds records with trailing columns ``[time, lower, upper]``, such as
    those from [censored][liesel_ptm.survival_dist.censored].

    Parameters
    ----------
    distribution
        Constructor of the uncensored, scalar-event distribution.
    censoring_records
        Optional records from which to cache each row's observation type. The
        cache evaluates only the likelihood branch each row needs, but assumes
        that row types stay fixed. Values may change while every row keeps its
        type. By default, row types are read from the records at each evaluation.
    **kwinputs
        Keyword inputs of the distribution, and other keyword arguments of
        `liesel.model.Dist`. Positional inputs are not supported.

    Raises
    ------
    ValueError
        If ``distribution`` is already censored or ``censoring_records`` lacks a
        trailing dimension of size 3 and an observation axis.

    Notes
    -----
    With the cache on, each eager update checks that the row types of the
    variable's records still match the cached ones, raising a `ValueError`
    otherwise. Values traced by JAX cannot be checked.

    Summary and plotting functions use the uncensored distribution.

    See Also
    --------
    [PTM.from_censored][liesel_ptm.PTM.from_censored]
        Censored PTM with an uncached `CensoredDist`.

    Examples
    --------
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel_ptm import survival_dist as surv
    >>> records = surv.right_censored([0.5, 1.0], event_observed=[True, False])
    >>> uncached = surv.CensoredDist(tfd.Normal, loc=0.0, scale=1.0)
    >>> cached = surv.CensoredDist(
    ...     tfd.Normal, censoring_records=records, loc=0.0, scale=1.0
    ... )
    >>> y = lsl.Var.new_obs(records, uncached, name="y").update()
    >>> y_cached = lsl.Var.new_obs(records, cached, name="y").update()
    >>> [round(float(value), 3) for value in y.log_prob]
    [-1.044, -1.841]
    >>> bool((y_cached.log_prob == y.log_prob).all())
    True
    """

    def __init__(
        self,
        distribution: Callable[..., Any],
        *,
        censoring_records: ValuesLike | None = None,
        **kwinputs: Any,
    ) -> None:
        if _is_censored(distribution):
            raise ValueError(
                "The distribution is already censored; pass the uncensored "
                "distribution instead."
            )
        self._censoring_records: np.ndarray | None = None
        self._cached_status: np.ndarray | None = None
        keywords: dict[str, Any] = {"distribution": distribution}
        if censoring_records is not None:
            records = _as_record_array(censoring_records)
            if records.ndim < 2:
                raise ValueError(
                    "censoring_records must have trailing event shape (3,) and "
                    "at least one observation axis."
                )
            self._censoring_records = records
            self._cached_status = censoring_status(records)
            keywords["censoring_records"] = records
        super().__init__(partial(CensoredDistribution, **keywords), **kwinputs)

    @classmethod
    def from_dist(
        cls, dist: lsl.Dist, *, censoring_records: ValuesLike | None = None
    ) -> Self:
        """Censor an existing distribution node, which is left unchanged.

        The new node shares the keyword inputs and ``per_obs`` setting of
        ``dist``, which must not belong to a variable yet.

        Parameters
        ----------
        dist
            Distribution node with keyword inputs only, such as a
            [PTMDist][liesel_ptm.PTMDist].
        censoring_records
            As in [CensoredDist][liesel_ptm.survival_dist.CensoredDist].

        Raises
        ------
        ValueError
            If ``dist`` has positional inputs, belongs to a variable, or is
            already censored.
        """
        if dist.inputs:
            raise ValueError(
                "CensoredDist supports keyword inputs only; pass the inputs of "
                "dist by keyword."
            )
        if dist.var is not None or dist.at is not None:
            raise ValueError("dist already belongs to a variable.")
        node = cls(
            dist.distribution, censoring_records=censoring_records, **dist.kwinputs
        )
        node.per_obs = dist.per_obs
        return node

    @property
    def base_distribution(self) -> Callable[..., Any]:
        """Constructor of the uncensored distribution."""
        return _uncensored_constructor(self.distribution)

    @property
    def censoring_records(self) -> np.ndarray | None:
        """Records whose row types are cached, or `None` without a cache."""
        return self._censoring_records

    def update(self) -> Self:
        if self._cached_status is not None and self.at is not None:
            self._check_cached_status(self.at.value)
        super().update()
        return self

    def _check_cached_status(self, value: Any) -> None:
        try:
            records = np.asarray(value)
        except jax.errors.TracerArrayConversionError:
            return
        cached = self._cached_status
        assert cached is not None
        status = censoring_status(records)
        if status.shape != cached.shape:
            raise ValueError(
                f"The records have batch shape {status.shape}, but the cached "
                f"row types have shape {cached.shape}. Rebuild the distribution "
                "with the new records or without censoring_records."
            )
        changed = np.argwhere(status != cached)
        if changed.size:
            rows = ", ".join(str(tuple(map(int, row))) for row in changed[:5])
            raise ValueError(
                "The records' row types no longer match the cached censoring "
                f"records at index {rows}{', ...' if len(changed) > 5 else ''}. "
                "Rebuild the distribution with the new records or without "
                "censoring_records."
            )


def _is_censored(distribution: Callable[..., Any]) -> bool:
    return distribution is CensoredDistribution or (
        isinstance(distribution, partial) and distribution.func is CensoredDistribution
    )


def _as_record_value(value: ArrayLike) -> Array:
    value = jnp.asarray(value)
    if value.ndim == 0 or value.shape[-1] != 3:
        raise ValueError(
            "Censored records must have trailing event shape (3,), with "
            "layout [time, lower, upper]."
        )

    return value


def subset_var(
    value: lsl.Var,
    indicators: Array,
    suffix: str,
    dist: lsl.Dist | None = None,
    transform: Callable[[Array], Array] | None = None,
) -> lsl.Var:
    """
    Return a Liesel variable restricted to selected observations.

    The helper subsets ``value`` along its first axis wherever ``indicators`` is
    true. If a distribution is supplied, its positional inputs and keyword
    inputs are copied and any array-like input whose first axis has the same
    length as the indicator vector is subset in the same way. Inputs with
    different leading dimensions and non-array static inputs are left unchanged.

    Parameters
    ----------
    value
        Variable whose value is subset along the observation axis.
    indicators
        One-dimensional boolean mask selecting observations.
    suffix
        Suffix appended to the names of the returned variable and copied inputs.
    dist
        Optional distribution to attach to the returned variable. Defaults to
        ``value.dist_node``.
    transform
        Optional transformation applied after subsetting the variable value.

    Returns
    -------
    lsl.Var
        A variable backed by a transient calculation. If no observations are
        selected, the returned variable has value ``None`` and no distribution.
    """
    indicators_np = np.asarray(indicators, dtype=bool)
    if indicators_np.ndim != 1:
        raise ValueError("indicators must be a one-dimensional boolean array.")

    n_obs = indicators_np.shape[0]
    indices = jnp.asarray(np.flatnonzero(indicators_np), dtype=jnp.int32)

    if not np.any(indicators_np):
        return lsl.Var(None, name=value.name + suffix)

    if transform is None:

        def transform(x):
            return x

    def subset_first_axis_if_observed(x):
        try:
            x_array = jnp.asarray(x)
        except (TypeError, ValueError):
            return x

        if x_array.ndim > 0 and x_array.shape[0] == n_obs:
            return jnp.take(x_array, indices, axis=0)
        return x_array

    dist = dist if dist is not None else value.dist_node

    if dist is not None:
        _inputs = []
        for iv in dist.inputs:
            iv_node: Any = iv
            if isinstance(iv, VarValue):
                if iv.var is None:
                    raise ValueError("Cannot subset a detached VarValue input.")
                iv_node = iv.var
            iv_calc = lsl.TransientCalc(subset_first_axis_if_observed, iv_node)
            # iv_var = lsl.Var(iv_calc, name=iv_node.name + suffix)
            # _inputs.append(iv_var)
            _inputs.append(iv_calc)

        _kwinputs: dict[str, Any] = {}
        for kw, kwiv in dist.kwinputs.items():
            kwiv_node: Any = kwiv
            if isinstance(kwiv, VarValue):
                if kwiv.var is None:
                    raise ValueError("Cannot subset a detached VarValue input.")
                kwiv_node = kwiv.var
            kwiv_calc = lsl.TransientCalc(subset_first_axis_if_observed, kwiv_node)
            # kwiv_var = lsl.Var(kwiv_calc, name=kwiv_node.name + suffix)
            # _kwinputs[kw] = kwiv_var
            _kwinputs[kw] = kwiv_calc

        subset_dist = lsl.Dist(dist.distribution, *_inputs, **_kwinputs)
    else:
        subset_dist = None

    def subset_value(x):
        return transform(subset_first_axis_if_observed(x))

    var_calc = lsl.TransientCalc(subset_value, value)
    var = lsl.Var(var_calc, distribution=subset_dist, name=value.name + suffix)
    var.observed = value.observed
    return var


class CensoredVars(NamedTuple):
    uncensored: lsl.Var
    left_censored: lsl.Var
    right_censored: lsl.Var
    interval_censored: lsl.Var


def _base_distribution_from_dist(
    dist: lsl.Dist,
) -> Callable[..., tfd.Distribution]:
    return cast(
        Callable[..., tfd.Distribution], _uncensored_constructor(dist.distribution)
    )


def _uncensored_constructor(distribution: Callable[..., Any]) -> Callable[..., Any]:
    """Return the event-time constructor wrapped by a ``CensoredDistribution``.

    Other constructors are returned unchanged.
    """
    if _is_censored(distribution) and isinstance(distribution, partial):
        keywords = distribution.keywords or {}
        try:
            base_distribution = keywords["distribution"]
        except KeyError as err:
            raise ValueError(
                "A CensoredDistribution partial must fix its base 'distribution'; "
                "the uncensored distribution cannot be recovered."
            ) from err

        fixed_kwargs = {
            key: value
            for key, value in keywords.items()
            if key not in ("distribution", "censoring_records")
        }
        if fixed_kwargs:
            return partial(base_distribution, **fixed_kwargs)
        return base_distribution

    return distribution


def setup_censored_vars(
    censoring_records: lsl.Var,
    dist: lsl.Dist | None = None,
) -> CensoredVars:
    """
    Split mixed censoring records into specialized Liesel variables.

    ``censoring_records`` must contain records with shape ``(n, 3)`` and layout
    ``[time, lower, upper]``. Missing entries are encoded as ``nan``:
    ``[time, nan, nan]`` is uncensored, ``[nan, nan, upper]`` is left censored,
    ``[nan, lower, nan]`` is right censored, and ``[nan, lower, upper]`` is
    interval censored.

    If ``dist`` is a ``CensoredDistribution`` Liesel distribution, this helper
    unwraps its base distribution and creates four specialized variables:
    uncensored observations use the base distribution directly, while left,
    right, and interval censored observations use the corresponding specialized
    censoring distribution. Keyword inputs of the original distribution are
    preserved and subset along the first axis when they are observation-shaped.

    Parameters
    ----------
    censoring_records
        Variable containing the mixed censoring records.
    dist
        Optional distribution node to split. Defaults to
        ``censoring_records.dist_node``. Positional distribution inputs are not
        supported because observation-aligned inputs must be identifiable by
        keyword.

    Returns
    -------
    CensoredVars
        Named tuple with variables for uncensored, left-censored,
        right-censored, and interval-censored observations.
    """
    dist = dist if dist is not None else censoring_records.dist_node
    value = _as_record_value(censoring_records.value)
    if value.ndim != 2:
        raise ValueError("setup_censored_vars expects records with shape (n, 3).")

    (
        is_uncensored,
        is_left_censored,
        is_right_censored,
        is_interval_censored,
    ) = _record_status_masks(value[..., 0], value[..., 1], value[..., 2])

    if dist is not None and dist.inputs:
        raise ValueError(
            "Please specify inputs to the distribution only via keyword-inputs."
        )

    uncensored_dist_cls = None
    left_censored_dist_cls = None
    right_censored_dist_cls = None
    interval_censored_dist_cls = None

    if dist is not None:
        base_distribution = _base_distribution_from_dist(dist)
        kwinputs = cast(dict[str, Any], dict(dist.kwinputs))
        uncensored_dist_cls = lsl.Dist(base_distribution, **kwinputs)
        left_censored_dist_cls = lsl.Dist(
            distribution=partial(
                LeftCensoredDistribution, distribution=base_distribution
            ),
            **kwinputs,
        )
        right_censored_dist_cls = lsl.Dist(
            distribution=partial(
                RightCensoredDistribution, distribution=base_distribution
            ),
            **kwinputs,
        )
        interval_censored_dist_cls = lsl.Dist(
            distribution=partial(
                IntervalCensoredDistribution, distribution=base_distribution
            ),
            **kwinputs,
        )

    uncensored_var = subset_var(
        censoring_records,
        indicators=is_uncensored,
        dist=uncensored_dist_cls,
        suffix="_uncensored",
        transform=lambda x: x[..., 0],
    )

    left_censored_var = subset_var(
        censoring_records,
        indicators=is_left_censored,
        dist=left_censored_dist_cls,
        suffix="_left_censored",
        transform=lambda x: x[..., 2],
    )

    right_censored_var = subset_var(
        censoring_records,
        indicators=is_right_censored,
        dist=right_censored_dist_cls,
        suffix="_right_censored",
        transform=lambda x: x[..., 1],
    )

    interval_censored_var = subset_var(
        censoring_records,
        indicators=is_interval_censored,
        dist=interval_censored_dist_cls,
        suffix="_interval_censored",
        transform=lambda x: x[..., 1:3],
    )

    return CensoredVars(
        uncensored=uncensored_var,
        left_censored=left_censored_var,
        right_censored=right_censored_var,
        interval_censored=interval_censored_var,
    )
