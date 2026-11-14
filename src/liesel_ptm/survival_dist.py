from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Self, cast

import jax
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.typing import ArrayLike
from tensorflow_probability.substrates.jax import tf2jax as tf
from tensorflow_probability.substrates.jax.internal import slicing as tfp_slicing

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
    """Create uncensored records ``[time, nan, nan]``."""
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
    pattern is invalid. The likelihood, the record cache, validation and
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


def _censored_mask(censored: ValuesLike) -> tuple[Array, bool]:
    try:
        mask = np.asarray(censored)
    except jax.errors.TracerArrayConversionError:
        return jnp.asarray(censored) != 0, False
    if mask.dtype == np.bool_:
        return mask, True
    if not (np.issubdtype(mask.dtype, np.number) and np.isin(mask, (0, 1)).all()):
        raise ValueError(
            "censored must be a Boolean value or mask containing only 0 and 1."
        )
    return mask != 0, True


def _encode(
    *,
    uncensored_rows: Array,
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
    time = jnp.where(uncensored_rows, time.astype(dtype), nan)
    lower = jnp.where(right | interval, lower.astype(dtype), nan)
    upper = jnp.where(left | interval, upper.astype(dtype), nan)
    time, lower, upper = jnp.broadcast_arrays(time, lower, upper)
    return jnp.stack((time, lower, upper), axis=-1)


def _require(invalid: Array, message: str) -> None:
    if np.any(invalid):
        raise ValueError(message)


def _one_sided_censored(value: ValuesLike, censored: ValuesLike, *, side: str) -> Array:
    array, concrete_value = _input_array(value)
    censored_rows, concrete_mask = _censored_mask(censored)
    uncensored_rows = ~censored_rows
    if concrete_value and concrete_mask:
        _require(
            uncensored_rows & ~np.isfinite(array),
            "Uncensored times (censored=False) must be finite.",
        )
    no_rows = jnp.zeros((), dtype=bool)
    return _encode(
        uncensored_rows=uncensored_rows,
        left=censored_rows if side == "left" else no_rows,
        right=censored_rows if side == "right" else no_rows,
        interval=no_rows,
        time=value,
        lower=value,
        upper=value,
    )


def left_censored(upper: ValuesLike, *, censored: ValuesLike = True) -> Array:
    """Encode left-censored observations, optionally mixed with uncensored ones.

    Parameters
    ----------
    upper
        Upper bounds of censored observations. Where ``censored`` is false, the
        same value is the uncensored observation instead.
    censored
        Scalar flag or mask broadcastable with ``upper``, either Boolean or
        containing only 0 and 1. By default, every observation is left-censored.

    Returns
    -------
    jax.Array
        Records with trailing columns ``[time, lower, upper]``:
        ``[nan, nan, upper]`` for censored and ``[upper, nan, nan]`` for
        uncensored rows.

    Raises
    ------
    ValueError
        If ``censored`` is not binary or an uncensored value is not finite.

    Examples
    --------
    >>> from liesel_ptm import survival_dist as surv
    >>> surv.left_censored([1.0, 2.0], censored=[True, False])
    Array([[nan, nan,  1.],
           [ 2., nan, nan]], dtype=float32)
    """
    return _one_sided_censored(upper, censored, side="left")


def right_censored(lower: ValuesLike, *, censored: ValuesLike = True) -> Array:
    """Encode right-censored observations, optionally mixed with uncensored ones.

    Parameters
    ----------
    lower
        Lower bounds of censored observations, such as follow-up times. Where
        ``censored`` is false, the same value is the uncensored observation,
        such as an observed event time, instead.
    censored
        Scalar flag or mask broadcastable with ``lower``, either Boolean or
        containing only 0 and 1. By default, every observation is
        right-censored. For survival data with an event indicator ``event``,
        pass ``censored=~event``.

    Returns
    -------
    jax.Array
        Records with trailing columns ``[time, lower, upper]``:
        ``[nan, lower, nan]`` for censored and ``[lower, nan, nan]`` for
        uncensored rows.

    Raises
    ------
    ValueError
        If ``censored`` is not binary or an uncensored value is not finite.

    Examples
    --------
    >>> from liesel_ptm import survival_dist as surv
    >>> surv.right_censored([1.0, 2.0], censored=[False, True])
    Array([[ 1., nan, nan],
           [nan,  2., nan]], dtype=float32)
    """
    return _one_sided_censored(lower, censored, side="right")


def interval_censored(
    lower: ValuesLike, upper: ValuesLike, *, censored: ValuesLike = True
) -> Array:
    """Encode interval-censored observations, optionally mixed with uncensored ones.

    Parameters
    ----------
    lower, upper
        Broadcastable interval bounds. Censored rows require
        ``lower < upper``; an infinite endpoint is kept as an interval bound.
        Uncensored rows require matching finite bounds, the observed value.
    censored
        Scalar flag or mask broadcastable with the bounds, either Boolean or
        containing only 0 and 1. By default, every observation is
        interval-censored.

    Returns
    -------
    jax.Array
        Records with trailing columns ``[time, lower, upper]``:
        ``[nan, lower, upper]`` for censored and ``[lower, nan, nan]`` for
        uncensored rows.

    Raises
    ------
    ValueError
        If ``censored`` is not binary, a censored row lacks positive width, or
        an uncensored row lacks matching finite bounds.

    Notes
    -----
    Widths are checked at input precision. The interval status is kept even if
    conversion to the output dtype collapses the bounds;
    [CensoredDist][liesel_ptm.survival_dist.CensoredDist] then rejects the
    records.

    Examples
    --------
    >>> from liesel_ptm import survival_dist as surv
    >>> surv.interval_censored([0.0, 1.0], [2.0, 1.0], censored=[1, 0])
    Array([[nan,  0.,  2.],
           [ 1., nan, nan]], dtype=float32)
    """
    lower_array, concrete_lower = _input_array(lower)
    upper_array, concrete_upper = _input_array(upper)
    censored_rows, concrete_mask = _censored_mask(censored)
    if concrete_lower and concrete_upper and concrete_mask:
        censored_rows, lower_array, upper_array = np.broadcast_arrays(
            censored_rows, lower_array, upper_array
        )
        _require(
            ~censored_rows & ~((lower_array == upper_array) & np.isfinite(lower_array)),
            "Uncensored rows (censored=False) require matching finite bounds.",
        )
        _require(
            censored_rows & ~(upper_array > lower_array),
            "Interval-censored rows require positive-width intervals, lower < upper.",
        )
    no_rows = jnp.zeros((), dtype=bool)
    return _encode(
        uncensored_rows=~censored_rows,
        left=no_rows,
        right=no_rows,
        interval=censored_rows,
        time=lower,
        lower=lower,
        upper=upper,
    )


def censored(
    lower: ValuesLike, upper: ValuesLike, *, censored: ValuesLike | None = None
) -> Array:
    """Encode observations from their bounds as censoring records.

    Parameters
    ----------
    lower, upper
        Broadcastable bounds with ``lower <= upper``. Use ``-inf`` as the lower
        bound of a left-censored and ``inf`` as the upper bound of a
        right-censored observation. Matching finite bounds give an uncensored
        observation.
    censored
        ``None`` infers every observation type from the bounds. Otherwise, a
        scalar flag or mask broadcastable with the bounds, either Boolean or
        containing only 0 and 1. False rows are uncensored and require matching
        finite bounds; true rows are censored and require ``lower < upper``.

    Returns
    -------
    jax.Array
        Records with trailing columns ``[time, lower, upper]``. Uncensored,
        left-censored, right-censored, and interval-censored rows are
        ``[t, nan, nan]``, ``[nan, nan, upper]``, ``[nan, lower, nan]``, and
        ``[nan, lower, upper]``.

    Raises
    ------
    ValueError
        If ``censored`` is not binary, or bounds are missing, decreasing, both
        infinite, infinite on the wrong side, or inconsistent with
        ``censored``.

    Notes
    -----
    Bounds are compared with exact equality at input precision, before
    conversion to the output dtype. An interval whose bounds collapse during
    that conversion stays interval-censored; it never becomes uncensored.

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
    if censored is None:
        censored_mask, concrete_mask = None, True
    else:
        censored_mask, concrete_mask = _censored_mask(censored)
    concrete = concrete_lower and concrete_upper and concrete_mask
    xp: Any = np if concrete else jnp
    if not concrete:
        lower_array = jnp.asarray(lower_array)
        upper_array = jnp.asarray(upper_array)
    finite_lower = xp.isfinite(lower_array)
    finite_upper = xp.isfinite(upper_array)
    matching = finite_lower & (lower_array == upper_array)
    uncensored_rows = matching if censored_mask is None else ~xp.asarray(censored_mask)
    censored_rows = ~uncensored_rows
    left = censored_rows & (lower_array == -xp.inf) & finite_upper
    right = censored_rows & finite_lower & (upper_array == xp.inf)
    interval = censored_rows & finite_lower & finite_upper & (lower_array < upper_array)
    if concrete:
        _require(
            uncensored_rows & ~matching,
            "Uncensored rows (censored=False) require matching finite bounds.",
        )
        _require(
            censored_rows & ~(left | right | interval),
            "Censored rows require lower < upper, without NaN and with at most one "
            "infinite endpoint: -inf for lower or inf for upper.",
        )
    return _encode(
        uncensored_rows=uncensored_rows,
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
    >>> surv.censoring_status(surv.right_censored([1.0, 2.0], censored=[0, 1]))
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


def _row_indices(mask: np.ndarray) -> str:
    if mask.ndim == 0:
        return "() (a single record)"
    rows = np.argwhere(mask)
    labels = [
        str(int(row[0])) if mask.ndim == 1 else str(tuple(map(int, row)))
        for row in rows[:5]
    ]
    return ", ".join(labels) + (", ..." if len(rows) > 5 else "")


def _check_records(records: ValuesLike, dtype: Any = None) -> None:
    """Raise if concrete records are malformed; traced records are skipped.

    Checks the layout, the missing-value patterns and, in ``dtype``, the
    interval widths. The likelihood converts records to its dtype, which can
    collapse narrow intervals.
    """
    try:
        array = np.asarray(records)
    except jax.errors.TracerArrayConversionError:
        return
    array = _as_record_array(array)
    if dtype is not None:
        array = array.astype(np.dtype(dtype))
    status = censoring_status(array)
    invalid = status == "invalid"
    if invalid.any():
        raise ValueError(
            "Censoring records have invalid missing-value patterns at row "
            f"indices {_row_indices(invalid)}. Encode them with "
            "liesel_ptm.survival_dist.censored() or a dedicated helper."
        )
    collapsed = (status == "interval_censored") & ~(array[..., 2] > array[..., 1])
    if collapsed.any():
        raise ValueError(
            "Interval-censored records must have positive width (upper > lower) "
            f"in the likelihood dtype {array.dtype.name}; check row indices "
            f"{_row_indices(collapsed)}. Fix the bounds, or rescale them or "
            "enable 64-bit JAX if a narrow interval lost its width through "
            "precision."
        )


_LOG_2 = math.log(2.0)


def _log1mexp(x: Array) -> Array:
    """Compute ``log(1 - exp(x))`` for ``x < 0``, accurately on both sides of
    ``-log(2)``. Each branch only sees valid inputs, so gradients stay finite.
    """
    near_zero = x > -_LOG_2
    x_near = jnp.where(near_zero, x, -1.0)
    x_far = jnp.where(near_zero, -1.0, x)
    return jnp.where(near_zero, jnp.log(-jnp.expm1(x_near)), jnp.log1p(-jnp.exp(x_far)))


def _logdiffexp(log_a: Array, log_b: Array) -> Array:
    """Compute ``log(exp(log_a) - exp(log_b))``, or ``-inf`` unless ``log_a > log_b``.

    Invalid inputs are replaced before the computation, so masked results do not
    produce NaN gradients.
    """
    neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.result_type(log_a, log_b))
    delta = log_b - log_a
    valid = (log_a > log_b) & (delta < 0.0)
    safe_delta = jnp.where(valid, delta, -1.0)
    safe_log_a = jnp.where(valid, log_a, 0.0)
    return jnp.where(valid, safe_log_a + _log1mexp(safe_delta), neg_inf)


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


def _first_finite(*values: Array) -> Array:
    """Return the first finite value per element, or 0 where none is finite."""
    result = jnp.zeros_like(values[0])
    for value in reversed(values):
        result = jnp.where(jnp.isfinite(value), value, result)
    return result


def _interval_log_prob(
    distribution: tfd.Distribution,
    lower: Array,
    upper: Array,
    anchor: Array | None = None,
) -> Array:
    """Log probability of ``lower < Y <= upper`` for scalar-event distributions.

    An infinite endpoint gives a one-sided probability, and ``(-inf, inf)``
    gives zero. Rows without valid bounds give ``-inf``. Unused evaluations
    happen at ``anchor``, a finite value of the row, so masked rows do not
    produce NaN gradients.
    """
    finite_lower = jnp.isfinite(lower)
    finite_upper = jnp.isfinite(upper)
    lower_open = jnp.isneginf(lower)
    upper_open = jnp.isposinf(upper)
    two_sided = finite_lower & finite_upper & (upper > lower)
    left_open = lower_open & finite_upper
    right_open = finite_lower & upper_open
    if anchor is None:
        anchor = _first_finite(lower, upper)

    upper_eval = jnp.where(two_sided | left_open, upper, anchor)
    lower_eval = jnp.where(two_sided | right_open, lower, anchor)
    log_cdf_upper = _distribution_log_cdf(distribution, upper_eval)
    log_cdf_lower = _distribution_log_cdf(distribution, lower_eval)
    log_sf_upper = _distribution_log_survival_function(distribution, upper_eval)
    log_sf_lower = _distribution_log_survival_function(distribution, lower_eval)

    # The survival difference is more accurate in the right tail.
    two_sided_log_prob = jnp.maximum(
        _logdiffexp(log_cdf_upper, log_cdf_lower),
        _logdiffexp(log_sf_lower, log_sf_upper),
    )
    result = jnp.full_like(two_sided_log_prob, -jnp.inf)
    result = jnp.where(lower_open & upper_open, 0.0, result)
    result = jnp.where(right_open, log_sf_lower, result)
    result = jnp.where(left_open, log_cdf_upper, result)
    return jnp.where(two_sided, two_sided_log_prob, result)


class CensoredDistribution(tfd.Distribution):
    """
    Censoring-record wrapper for a scalar-event TFP distribution.

    The wrapped distribution, available as ``distribution``, models uncensored
    observations on their native scale. This wrapper models censoring records
    with event shape ``(3,)`` and layout ``[time, lower, upper]``:

    - ``[time, nan, nan]`` for uncensored observations.
    - ``[nan, nan, upper]`` for left-censored observations.
    - ``[nan, lower, nan]`` for right-censored observations.
    - ``[nan, lower, upper]`` for interval-censored observations.

    Optionally, pass concrete ``censoring_records`` to cache each row's type
    for a fixed data set. In cached mode, ``log_prob(value)`` still expects
    full records with trailing event shape ``(3,)``; the cached types decide
    which likelihood contribution each row gets, and only that one is
    evaluated. The cache selects rows of ``distribution`` with TFP batch
    slicing, so a base distribution with an observation axis must support it.

    Censored records contribute ``P(Y <= upper)`` (left), ``P(Y > lower)``
    (right) and ``P(lower < Y <= upper)`` (interval). The inequalities only
    matter for discrete distributions. Infinite interval endpoints give the
    corresponding one-sided probability. With ``validate_args=True``,
    ``log_prob`` raises for concrete records with invalid missing-value
    patterns or intervals without positive width; otherwise, such rows have
    log probability ``-inf``.

    The wrapper supports batch slicing, and an uncached wrapper can be passed
    into a jitted function. A cached wrapper must be built inside it, because
    it reads the record values when it is built.

    Parameters
    ----------
    distribution
        Built, scalar-event distribution with a floating-point dtype, such as
        ``tfd.Normal(0.0, 1.0)``. It keeps its own ``validate_args`` setting.
    censoring_records
        Optional concrete records whose row types to cache.
    validate_args
        Whether ``log_prob`` checks concrete records.
    allow_nan_stats, name
        As in TFP distributions.

    Raises
    ------
    TypeError
        If ``distribution`` is not a built distribution, or cached records are
        not concrete.
    ValueError
        If the base distribution does not have a scalar event and a
        floating-point dtype, or cannot be batch-sliced although the cache
        needs to select its rows.

    Examples
    --------
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel_ptm import survival_dist as surv
    >>> records = surv.right_censored([0.5, 1.0], censored=[False, True])
    >>> dist = surv.CensoredDistribution(tfd.Normal(0.0, 1.0))
    >>> [round(float(value), 3) for value in dist.log_prob(records)]
    [-1.044, -1.841]
    """

    def __init__(
        self,
        distribution: tfd.Distribution,
        censoring_records: ArrayLike | None = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "CensoredDistribution",
    ) -> None:
        parameters = dict(locals())
        if not isinstance(distribution, tfd.Distribution):
            raise TypeError(
                f"{type(self).__name__} wraps a built distribution, such as "
                f"tfd.Normal(0.0, 1.0); got {distribution!r}."
            )
        self._distribution = distribution

        if tuple(distribution.event_shape) != ():
            raise ValueError(
                f"{type(self).__name__} requires a scalar-event base distribution."
            )
        base_dtype = distribution.dtype
        if base_dtype is None or not jnp.issubdtype(base_dtype, jnp.floating):
            raise ValueError(
                f"{type(self).__name__} requires a floating-point base "
                f"distribution, because records mark unused fields with nan; "
                f"got dtype {base_dtype}."
            )

        super().__init__(
            dtype=base_dtype,
            reparameterization_type=distribution.reparameterization_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

        self.censoring_records: Array | None = None
        self._cached_record_shape: tuple[int, ...] | None = None
        self._cached_record_batch_shape: tuple[int, ...] | None = None
        self._cached_result_batch_shape: tuple[int, ...] | None = None
        self._cached_padded_record_batch_shape: tuple[int, ...] | None = None
        self._cached_observation_axis = 0
        self._cached_base_observation_axis: int | None = None
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

    def _event_shape(self):
        return tf.TensorShape((3,))

    def _event_shape_tensor(self):
        return jnp.asarray((3,), dtype=jnp.int32)

    @property
    def distribution(self) -> tfd.Distribution:
        """The uncensored distribution."""
        return self._distribution

    def _batch_shape(self):
        return self.distribution.batch_shape

    def _batch_shape_tensor(self, **kwargs):
        return self.distribution.batch_shape_tensor()

    def __getitem__(self, slices: Any) -> Self:
        if not isinstance(slices, Sequence):
            slices = (slices,)
        distribution = self.distribution[slices]
        records = self.censoring_records
        if records is not None:
            records = tfp_slicing._slice_single_param(
                records, 1, slices, self.batch_shape_tensor()
            )
        return type(self)(
            distribution,
            censoring_records=records,
            validate_args=self.validate_args,
            allow_nan_stats=self.allow_nan_stats,
            name=self.name,
        )

    def _set_cached_censoring_records(self, censoring_records: ArrayLike) -> None:
        try:
            records_np = np.asarray(censoring_records)
        except jax.errors.TracerArrayConversionError as err:
            raise TypeError(
                "censoring_records must be concrete when the distribution is built. "
                "A cached CensoredDistribution cannot be passed into a jitted "
                "function; build it inside the function instead."
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
            int(dim) for dim in tuple(self.distribution.batch_shape)
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
        # A single observation uses the last axis.
        observation_axis = non_singleton_axes[-1] if non_singleton_axes else rank - 1
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

        base_axis = observation_axis - (rank - len(base_batch_shape))
        if base_axis >= 0 and base_batch_shape[base_axis] != 1:
            self._cached_base_observation_axis = base_axis
            try:
                self._cached_base_distribution(jnp.zeros((1,), dtype=jnp.int32))
            except (NotImplementedError, ValueError, TypeError) as err:
                raise ValueError(
                    "Caching censoring records needs a base distribution that "
                    f"supports batch slicing, but {type(self.distribution).__name__} "
                    "does not. Define _parameter_properties for it, or evaluate "
                    "the records without the cache."
                ) from err

    def _as_record_value(self, value: Array) -> Array:
        value = jnp.asarray(value, dtype=self.dtype)
        if value.ndim == 0 or value.shape[-1] != 3:
            raise ValueError(
                "Censored records must have trailing event shape (3,), with "
                "layout [time, lower, upper]."
            )
        return value

    def _prob(self, value: Array) -> Array:
        return jnp.exp(self.log_prob(value))

    def _log_prob(self, value: Array) -> Array:
        value = self._as_record_value(value)
        if self.validate_args:
            _check_records(value)
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

        # Every row is evaluated as uncensored and as an interval. One-sided
        # rows become intervals with an infinite endpoint. Unused evaluations
        # happen at a finite value of the same row, which lies near the data.
        nan = jnp.asarray(jnp.nan, dtype=value.dtype)
        inf = jnp.asarray(jnp.inf, dtype=value.dtype)
        anchor = _first_finite(time, lower, upper)
        bound_lower = jnp.where(
            is_right_censored | is_interval_censored,
            lower,
            jnp.where(is_left_censored, -inf, nan),
        )
        bound_upper = jnp.where(
            is_left_censored | is_interval_censored,
            upper,
            jnp.where(is_right_censored, inf, nan),
        )
        uncensored_log_prob = self.distribution.log_prob(
            jnp.where(is_uncensored, time, anchor)
        )
        censored_log_prob = _interval_log_prob(
            self.distribution, bound_lower, bound_upper, anchor
        )
        return jnp.where(is_uncensored, uncensored_log_prob, censored_log_prob)

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

    def _cached_base_distribution(self, indices: Array) -> tfd.Distribution:
        """Select the base distribution's rows along the cached observation axis.

        A base without that axis, or with size one along it, broadcasts and is
        returned unchanged.
        """
        axis = self._cached_base_observation_axis
        if axis is None:
            return self.distribution
        rank = len(tuple(self.distribution.batch_shape))
        slices = (slice(None),) * axis + (indices,) + (slice(None),) * (rank - axis - 1)
        return self.distribution[slices]

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
            result = self._scatter_cached_log_prob(
                result,
                indices,
                _interval_log_prob(branch_distribution, lower, upper),
            )

        return result

    def _sample_n(
        self, n: int | Array, seed: KeyArray | None = None, **kwargs
    ) -> Array:
        samples = self.distribution.sample(n, seed=seed)
        nan = jnp.full_like(samples, jnp.nan)
        return jnp.stack((samples, nan, nan), axis=-1)

    def _cdf(self, value: Array) -> Array:
        raise NotImplementedError(
            f"{type(self).__name__}.cdf is not defined for censoring records. "
            "Use .distribution.cdf(...) for uncensored observations."
        )

    def _quantile(self, value: Array, **kwargs) -> Array:
        raise NotImplementedError(
            f"{type(self).__name__}.quantile is not defined for censoring records. "
            "Use .distribution.quantile(...) for uncensored observations."
        )


def _flatten_censored_distribution(
    dist: CensoredDistribution,
) -> tuple[tuple[Any, Any], tuple[bool, bool, str]]:
    return (
        (dist.distribution, dist.censoring_records),
        (dist.validate_args, dist.allow_nan_stats, dist.name),
    )


def _unflatten_censored_distribution(
    aux: tuple[bool, bool, str], children: Sequence[Any]
) -> CensoredDistribution:
    distribution, censoring_records = children
    validate_args, allow_nan_stats, name = aux
    return CensoredDistribution(
        distribution,
        censoring_records=censoring_records,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name,
    )


jax.tree_util.register_pytree_node(
    CensoredDistribution,
    _flatten_censored_distribution,
    _unflatten_censored_distribution,
)


class CensoredDist(lsl.Dist):
    """Liesel distribution node that evaluates censoring records.

    Wraps a scalar-event distribution in a
    [CensoredDistribution][liesel_ptm.survival_dist.CensoredDistribution]. Its
    variable holds records with trailing columns ``[time, lower, upper]``, such as
    those from [censored][liesel_ptm.survival_dist.censored].

    Parameters
    ----------
    distribution
        Constructor of the uncensored, scalar-event distribution with a
        floating-point dtype.
    cache
        Whether to cache each row's observation type from the variable's
        records at the first update. The cache evaluates only the likelihood
        branch each row needs, which is several times faster, but assumes that
        row types stay fixed. Values may change while every row keeps its type.
        By default, row types are read from the records at each evaluation.
    **kwinputs
        Keyword inputs of the distribution, and other keyword arguments of
        `liesel.model.Dist`. Positional inputs are not supported.

    Raises
    ------
    ValueError
        If ``distribution`` is already censored. On update, if the records
        have the wrong layout, invalid missing-value patterns, or an interval
        without positive width in the likelihood dtype, or if the cache is on
        and a row type changed.

    Notes
    -----
    Records are validated whenever the node is updated with concrete values.
    Values traced by JAX cannot be checked. With the cache on, the first
    update must use concrete records.

    Censored records contribute ``P(Y <= upper)`` (left), ``P(Y > lower)``
    (right) and ``P(lower < Y <= upper)`` (interval). The inequalities only
    matter for discrete distributions.

    Summary and plotting functions use the uncensored distribution.

    See Also
    --------
    [PTM.from_censored][liesel_ptm.PTM.from_censored]
        Censored PTM built with a `CensoredDist`.

    Examples
    --------
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel_ptm import survival_dist as surv
    >>> records = surv.right_censored([0.5, 1.0], censored=[False, True])
    >>> uncached = surv.CensoredDist(tfd.Normal, loc=0.0, scale=1.0)
    >>> cached = surv.CensoredDist(tfd.Normal, cache=True, loc=0.0, scale=1.0)
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
        cache: bool = False,
        **kwinputs: Any,
    ) -> None:
        if _is_censored(distribution):
            raise ValueError(
                "The distribution is already censored; pass the uncensored "
                "distribution instead."
            )
        self._base_distribution = distribution
        self._cache = bool(cache)
        self._cached_records: np.ndarray | None = None
        self._cached_status: np.ndarray | None = None
        super().__init__(partial(_censor, distribution), **kwinputs)

    @classmethod
    def from_dist(cls, dist: lsl.Dist, *, cache: bool = False) -> Self:
        """Censor an existing distribution node, which is left unchanged.

        The new node shares the keyword inputs and ``per_obs`` setting of
        ``dist``, which must not belong to a variable yet.

        Parameters
        ----------
        dist
            Distribution node with keyword inputs only, such as a
            [PTMDist][liesel_ptm.PTMDist].
        cache
            As in [CensoredDist][liesel_ptm.survival_dist.CensoredDist].

        Raises
        ------
        ValueError
            If ``dist`` has positional inputs, belongs to a variable, or is
            already censored.
        """
        if isinstance(dist, CensoredDist):
            raise ValueError(
                "The distribution is already censored; pass the uncensored "
                "distribution instead."
            )
        if dist.inputs:
            raise ValueError(
                "CensoredDist supports keyword inputs only; pass the inputs of "
                "dist by keyword."
            )
        if dist.var is not None or dist.at is not None:
            raise ValueError("dist already belongs to a variable.")
        node = cls(dist.distribution, cache=cache, **dist.kwinputs)
        node.per_obs = dist.per_obs
        return node

    @property
    def base_distribution(self) -> Callable[..., Any]:
        """Constructor of the uncensored distribution."""
        return self._base_distribution

    @property
    def cache(self) -> bool:
        """Whether row types are cached at the first update."""
        return self._cache

    def init_dist(self) -> CensoredDistribution:
        if self._cached_records is None:
            return cast(CensoredDistribution, super().init_dist())
        kwargs = {kw: node.value for kw, node in self.kwinputs.items()}
        dist = self.distribution(censoring_records=self._cached_records, **kwargs)
        return cast(CensoredDistribution, dist)

    def update(self) -> Self:
        if self.at is not None:
            self._check_value(self.at.value)
        super().update()
        return self

    def _check_value(self, value: Any) -> None:
        try:
            records = np.asarray(value)
        except jax.errors.TracerArrayConversionError:
            if self._cache and self._cached_records is None:
                raise ValueError(
                    "CensoredDist with cache=True needs concrete records at its "
                    "first update; update the model before tracing it."
                ) from None
            return
        _check_records(records, dtype=super().init_dist().dtype)
        if not self._cache:
            return
        status = censoring_status(records)
        if self._cached_status is None:
            self._cached_records = records
            self._cached_status = status
            return
        if status.shape != self._cached_status.shape:
            raise ValueError(
                f"The records have batch shape {status.shape}, but the cached "
                f"row types have shape {self._cached_status.shape}. Rebuild the "
                "distribution for the new records, or use cache=False."
            )
        changed = status != self._cached_status
        if changed.any():
            raise ValueError(
                "The records' row types no longer match the cached ones at index "
                f"{_row_indices(changed)}. Rebuild the distribution for the new "
                "records, or use cache=False."
            )


def _censor(
    distribution: Callable[..., Any],
    censoring_records: ArrayLike | None = None,
    **kwargs: Any,
) -> CensoredDistribution:
    """Build the uncensored distribution and wrap it in censoring."""
    return CensoredDistribution(
        distribution(**kwargs), censoring_records=censoring_records
    )


def _is_censored(distribution: Callable[..., Any]) -> bool:
    return distribution is CensoredDistribution or (
        isinstance(distribution, partial)
        and distribution.func in (CensoredDistribution, _censor)
    )
