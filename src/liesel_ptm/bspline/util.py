from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from interpax import interp1d
from jax import Array
from tensorflow_probability.substrates.jax import distributions as tfd

from ..bijector import INVERSE_BLOCK_ROWS, SplineBijector
from .approx import BSplineApprox


def _broadcast_leading_core(x, target_batch_shape, core_ndims):
    """
    Prepend singleton axes so x can broadcast to target_batch_shape + core dims.
    """
    trailing = x.shape[-core_ndims:]
    need = len(target_batch_shape) - (x.ndim - core_ndims)
    shape = (1,) * need + x.shape
    x_reshaped = jnp.reshape(x, shape)
    return jnp.broadcast_to(x_reshaped, target_batch_shape + trailing)


class TransformationSpline:
    def __init__(self, knots: Array):
        """
        Parent class for monotonically increasing splines.

        Assumes usage of a cubic B-spline.

        Parameters
        ----------
        knots
            Spline knot sequence.

        Attributes
        ----------
        knots
            Spline knot sequence.
        min_knot
            Minimum knot value.
        max_knot
            Maximum knot value.
        bspline
            :class:`.BSplineApprox` instance for basis operations.

        Notes
        -----
        Splines are JAX pytrees, so they can be passed into jitted functions.
        Splines with the same settings share compiled kernels, and a discarded
        spline is freed. Treat a spline as immutable: create a new one to change
        its knots or settings.
        """
        knots = jnp.asarray(knots)
        self.knots = knots
        self._knots_tuple = tuple(
            float(k) for k in np.asarray(jax.device_get(knots), dtype=float)
        )

        self._nparam = int(knots.size) - 4  # len(knots) - order - 1; order is 3
        S = jnp.tril(jnp.ones((self._nparam, self._nparam)))
        self.bspline = BSplineApprox(knots, order=3, ngrid=1000, postmultiply_by=S)

        self.min_knot = self.bspline.min_knot
        self.max_knot = self.bspline.max_knot

        self._outer_knot_left = self._knots_tuple[0]
        self._outer_knot_right = self._knots_tuple[-1]
        self._ngrid_inverse = int(
            int(self.bspline.knots[-1] - self.bspline.knots[0]) * 100
        )

    # Splines are pytrees: arrays are leaves and hashable host-side settings are
    # static aux data. JAX's caches then hold no spline objects, splines with
    # equal settings share compiled kernels, and coefficients stay dynamic.
    # Subclasses extend both tuples with their own fields.
    _array_fields: tuple[str, ...] = ("knots", "bspline", "min_knot", "max_knot")
    _static_fields: tuple[str, ...] = (
        "_knots_tuple",
        "_nparam",
        "_outer_knot_left",
        "_outer_knot_right",
        "_ngrid_inverse",
    )

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node_class(cls)

    def tree_flatten(self) -> tuple[tuple[object, ...], tuple[object, ...]]:
        children = tuple(getattr(self, name) for name in self._array_fields)
        aux = tuple(getattr(self, name) for name in self._static_fields)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls, aux: tuple[object, ...], children: tuple[object, ...]
    ) -> TransformationSpline:
        obj = object.__new__(cls)
        for name, value in zip(cls._static_fields, aux, strict=True):
            setattr(obj, name, value)
        for name, value in zip(cls._array_fields, children, strict=True):
            setattr(obj, name, value)
        return obj

    @property
    def _knots_np(self) -> np.ndarray:
        return np.asarray(self._knots_tuple)

    def _compute_coef(self, raw_coef: Array) -> Array:
        """Map raw coefficients with shape ``(..., n_param)`` to spline coefficients."""
        raise NotImplementedError

    def bijector(
        self,
        coef: Array,
        centered: bool = False,
        scaled: bool = False,
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
        reference_distribution: tfd.Distribution | None = None,
        validate_args: bool = False,
    ) -> SplineBijector:
        """
        Return the spline transformation as a TFP bijector.

        Parameters
        ----------
        coef
            Unconstrained spline coefficients with shape
            ``batch_shape + (n_param,)``.
        centered
            If True, center the spline input by its finite-bound raw mean.
        scaled
            If True, scale the spline input by its finite-bound raw standard
            deviation.
        gauss_legendre_order
            Number of Gauss-Legendre nodes per knot interval for the raw moments.
        integration_bounds
            Lower and upper bounds for the raw moments. Defaults to the first and
            last spline knot.
        reference_distribution
            Distribution of the reference scale, used for centering, scaling and
            the raw moments. Defaults to ``Normal(0, 1)``.
        validate_args
            Whether to validate input arguments.

        Returns
        -------
        bijector
            A [SplineBijector][liesel_ptm.SplineBijector] that uses this spline.
        """
        return SplineBijector(
            coef,
            bspline=self,
            centered=centered,
            scaled=scaled,
            gauss_legendre_order=gauss_legendre_order,
            integration_bounds=integration_bounds,
            reference_distribution=reference_distribution,
            validate_args=validate_args,
        )

    @staticmethod
    def _shape_tuple(shape) -> tuple[int, ...]:
        return tuple(int(dim) for dim in tuple(shape))

    def _resolve_batch_shape(
        self, coef: Array, batch_shape: tuple[int, ...] | None
    ) -> tuple[int, ...]:
        if batch_shape is None:
            return self._coef_batch_shape(coef)

        return self._shape_tuple(batch_shape)

    @staticmethod
    def _check_coef_shape(coef: Array) -> None:
        if jnp.ndim(coef) < 1:
            raise ValueError(
                "Spline coefficients must have shape (..., n_param). "
                "Use shape (n_param,) for a shared, unbatched spline."
            )

    def _coef_batch_shape(self, coef: Array) -> tuple[int, ...]:
        self._check_coef_shape(coef)
        return self._shape_tuple(jnp.shape(coef)[:-1])

    def _constrained_coef(self, raw_coef: Array) -> Array:
        raw_coef = jnp.asarray(raw_coef)
        self._check_coef_shape(raw_coef)
        coef = self._compute_coef(raw_coef)
        return jnp.expand_dims(coef, axis=-2)

    @staticmethod
    def _value_layout(
        value: Array, batch_shape: tuple[int, ...]
    ) -> tuple[Array, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        value = jnp.asarray(value)
        value_shape = tuple(jnp.shape(value))

        if not batch_shape:
            sample_shape = value_shape
            value_batch_shape: tuple[int, ...] = ()
            result_batch_shape: tuple[int, ...] = ()
        elif len(value_shape) > len(batch_shape):
            sample_shape = value_shape[: -len(batch_shape)]
            value_batch_shape = value_shape[-len(batch_shape) :]
            result_batch_shape = jnp.broadcast_shapes(value_batch_shape, batch_shape)
        else:
            sample_shape = ()
            value_batch_shape = value_shape
            result_batch_shape = jnp.broadcast_shapes(value_batch_shape, batch_shape)

        padded_value_batch_shape = (1,) * (
            len(batch_shape) - len(value_batch_shape)
        ) + value_batch_shape

        return value, sample_shape, padded_value_batch_shape, result_batch_shape

    @staticmethod
    def _to_batch_last(
        value: Array,
        value_batch_shape: tuple[int, ...],
        sample_shape: tuple[int, ...],
    ) -> Array:
        sample_rank = len(sample_shape)
        batch_rank = len(value_batch_shape)
        sample_size = int(np.prod(sample_shape)) if sample_shape else 1

        value = jnp.reshape(value, sample_shape + value_batch_shape)
        axes = tuple(range(sample_rank, sample_rank + batch_rank)) + tuple(
            range(sample_rank)
        )
        value = jnp.transpose(value, axes)
        return jnp.reshape(value, value_batch_shape + (sample_size,))

    @staticmethod
    def _from_batch_last(
        value: Array, batch_shape: tuple[int, ...], sample_shape: tuple[int, ...]
    ) -> Array:
        sample_rank = len(sample_shape)
        batch_rank = len(batch_shape)

        value = jnp.reshape(value, batch_shape + sample_shape)
        axes = tuple(range(batch_rank, batch_rank + sample_rank)) + tuple(
            range(batch_rank)
        )
        return jnp.transpose(value, axes)

    def _evaluate_spline(self, value: Array, coef: Array) -> tuple[Array, Array]:
        """
        Evaluate constrained, broadcasted spline coefficients at broadcasted values.
        """
        return self.bspline.dot_and_deriv_n(value, coef)

    def _evaluate_spline_value(self, value: Array, coef: Array) -> Array:
        """
        Evaluate constrained, broadcasted spline coefficients at broadcasted values.
        """
        return self.bspline.dot_n(value, coef)

    def evaluate_and_deriv(
        self,
        value: Array,
        coef: Array,
        batch_shape: tuple[int, ...] | None = None,
    ) -> tuple[Array, Array]:
        """
        Evaluate the monotone spline f and its derivative, f(x) and f'(x).

        Parameters
        ----------
        value
            Values x at which to evaluate. Leading axes are sample axes;
            trailing axes broadcast against ``batch_shape``.
        coef
            Unconstrained spline coefficients with shape
            ``batch_shape + (n_param,)``.
        batch_shape
            Batch shape to broadcast against. Defaults to the batch shape of
            ``coef``.

        Returns
        -------
        value
            f(x), with shape ``broadcast(value.shape, batch_shape)``.
        deriv
            f'(x), with the same shape.
        """
        batch_shape = self._resolve_batch_shape(coef, batch_shape)
        value, sample_shape, value_batch_shape, result_batch_shape = self._value_layout(
            value, batch_shape
        )
        coef = self._constrained_coef(coef)

        value = self._to_batch_last(value, value_batch_shape, sample_shape)
        coef = _broadcast_leading_core(coef, result_batch_shape, core_ndims=2)
        dot, deriv = self._evaluate_spline(value, coef)

        dot = self._from_batch_last(dot, result_batch_shape, sample_shape)
        deriv = self._from_batch_last(deriv, result_batch_shape, sample_shape)

        return dot, deriv

    def evaluate(
        self,
        value: Array,
        coef: Array,
        batch_shape: tuple[int, ...] | None = None,
    ) -> Array:
        """
        Evaluate the monotone spline f at x, giving f(x).

        Parameters
        ----------
        value
            Values x at which to evaluate. Leading axes are sample axes;
            trailing axes broadcast against ``batch_shape``.
        coef
            Unconstrained spline coefficients with shape
            ``batch_shape + (n_param,)``.
        batch_shape
            Batch shape to broadcast against. Defaults to the batch shape of
            ``coef``.

        Returns
        -------
        value
            f(x), with shape ``broadcast(value.shape, batch_shape)``.
        """
        batch_shape = self._resolve_batch_shape(coef, batch_shape)
        value, sample_shape, value_batch_shape, result_batch_shape = self._value_layout(
            value, batch_shape
        )
        coef = self._constrained_coef(coef)

        value = self._to_batch_last(value, value_batch_shape, sample_shape)
        coef = _broadcast_leading_core(coef, result_batch_shape, core_ndims=2)
        dot = self._evaluate_spline_value(value, coef)

        return self._from_batch_last(dot, result_batch_shape, sample_shape)

    def inverse(
        self,
        value: Array,
        coef: Array,
        batch_shape: tuple[int, ...] | None = None,
    ) -> Array:
        """
        Evaluate the inverse of the monotone spline f, giving f⁻¹(z).

        The inverse is computed numerically: f is evaluated on a grid and
        interpolated, so ``f(f⁻¹(z))`` matches ``z`` up to interpolation error.
        [SplineBijector][liesel_ptm.SplineBijector] follows TFP's direction, so
        the bijector's ``inverse`` evaluates f and its ``forward`` evaluates f⁻¹.

        Parameters
        ----------
        value
            Values z at which to evaluate f⁻¹. Leading axes are sample axes;
            trailing axes broadcast against ``batch_shape``.
        coef
            Unconstrained spline coefficients with shape
            ``batch_shape + (n_param,)``.
        batch_shape
            Batch shape to broadcast against. Defaults to the batch shape of
            ``coef``.

        Returns
        -------
        value
            f⁻¹(z), with shape ``broadcast(value.shape, batch_shape)``.
        """
        batch_shape = self._resolve_batch_shape(coef, batch_shape)
        value, sample_shape, value_batch_shape, result_batch_shape = self._value_layout(
            value, batch_shape
        )
        coef = self._constrained_coef(coef)
        sample_size = int(np.prod(sample_shape)) if sample_shape else 1

        value = self._to_batch_last(value, value_batch_shape, sample_shape)
        value = _broadcast_leading_core(value, result_batch_shape, core_ndims=1)
        value_rows = jnp.reshape(value, (-1, sample_size))

        coef = _broadcast_leading_core(coef, result_batch_shape, core_ndims=2)
        _, p = jnp.shape(coef)[-2:]
        coef_rows = jnp.reshape(coef, (-1, 1, p))
        inverse_rows = _inverse_rows_chunked(self, value_rows, coef_rows)
        inverse = jnp.reshape(inverse_rows, result_batch_shape + (sample_size,))

        return self._from_batch_last(inverse, result_batch_shape, sample_shape)


def _inverse_rows_with_shared_grid(
    spline: TransformationSpline, value_rows: Array, coef_rows: Array
) -> Array:
    value_min = jnp.min(value_rows, axis=-1)
    value_max = jnp.max(value_rows, axis=-1)
    dtype = jnp.asarray(value_rows).dtype

    def fn(x: Array) -> Array:
        return spline._evaluate_spline(jnp.atleast_1d(x), coef_rows)[0]

    xlo_start = jnp.asarray(spline._outer_knot_left, dtype=dtype)
    left_shift = jnp.asarray(0.1, dtype=dtype)
    min_grid = jnp.min(fn(xlo_start - left_shift), axis=-1)

    def left_cond(val):
        _, min_grid = val
        return jnp.any(min_grid >= value_min)

    def left_body(val):
        left_shift, _ = val
        left_shift = left_shift + jnp.asarray(0.5, dtype=dtype)
        min_grid = jnp.min(fn(xlo_start - left_shift), axis=-1)
        return left_shift, min_grid

    left_shift, _ = jax.lax.while_loop(left_cond, left_body, (left_shift, min_grid))
    xlo = xlo_start - left_shift

    xhi_start = jnp.asarray(spline._outer_knot_right, dtype=dtype)
    right_shift = jnp.asarray(0.1, dtype=dtype)
    max_grid = jnp.max(fn(xhi_start + right_shift), axis=-1)

    def right_cond(val):
        _, max_grid = val
        return jnp.any(max_grid <= value_max)

    def right_body(val):
        right_shift, _ = val
        right_shift = right_shift + jnp.asarray(0.5, dtype=dtype)
        max_grid = jnp.max(fn(xhi_start + right_shift), axis=-1)
        return right_shift, max_grid

    right_shift, _ = jax.lax.while_loop(right_cond, right_body, (right_shift, max_grid))
    xhi = xhi_start + right_shift

    xgrid = jnp.linspace(xlo, xhi, spline._ngrid_inverse)
    ygrid = fn(xgrid)

    return jax.vmap(
        lambda value_row, ygrid_row: interp1d(
            value_row, ygrid_row, xgrid, method="monotonic"
        )
    )(value_rows, ygrid)


@jax.jit
def _inverse_rows_chunked(
    spline: TransformationSpline, value_rows: Array, coef_rows: Array
) -> Array:
    B, n = value_rows.shape
    p = coef_rows.shape[-1]

    chunk_size = min(INVERSE_BLOCK_ROWS, B)
    pad = (-B) % chunk_size

    if pad:
        value_pad = jnp.broadcast_to(value_rows[:1, :], (pad, n))
        coef_pad = jnp.broadcast_to(coef_rows[:1, :, :], (pad, 1, p))
        value_rows = jnp.concatenate((value_rows, value_pad), axis=0)
        coef_rows = jnp.concatenate((coef_rows, coef_pad), axis=0)

    n_blocks = value_rows.shape[0] // chunk_size
    value_blocks = jnp.reshape(value_rows, (n_blocks, chunk_size, n))
    coef_blocks = jnp.reshape(coef_rows, (n_blocks, chunk_size, 1, p))

    def body(carry, inputs):
        value_block, coef_block = inputs
        inverse_block = _inverse_rows_with_shared_grid(spline, value_block, coef_block)
        return carry, inverse_block

    _, inverse_blocks = jax.lax.scan(body, None, (value_blocks, coef_blocks))
    inverse_rows = jnp.reshape(inverse_blocks, (n_blocks * chunk_size, n))

    return inverse_rows[:B, :]
