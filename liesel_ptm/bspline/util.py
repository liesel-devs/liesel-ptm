from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..util.inverse_interpax import inv1d
from .approx import BSplineApprox

inv1d = jax.jit(inv1d, static_argnums=(1, 2, 3, 4, 5))


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
    def __init__(
        self,
        knots: Array,
        subscripts: Literal[
            "...nj,...j->...n", "...nj,...nj->...n"
        ] = "...nj,...j->...n",
    ):
        """
        Parent class for monotonically increasing splines.

        Assumes usage of a cubic B-spline.

        Parameters
        ----------
        knots
            Spline knot sequence.

        Attributes
        ----------
        n_chunks
            Compatibility attribute; no longer controls evaluation chunking.
        knots
            Spline knot sequence.
        min_knot
            Minimum knot value.
        max_knot
            Maximum knot value.
        bspline
            :class:`.BSplineApprox` instance for basis operations.
        """
        self.n_chunks = 1024
        self.supports_rowwise_coef = False
        self.knots = knots
        self._knots_np = np.asarray(jax.device_get(knots), dtype=float)

        self._nparam = knots.size - 4  # len(knots) - order - 1; order is fixed to 3
        S = jnp.tril(jnp.ones((self._nparam, self._nparam)))
        self.bspline = BSplineApprox(
            knots, order=3, ngrid=1000, postmultiply_by=S, subscripts=subscripts
        )

        self.min_knot = self.bspline.min_knot
        self.max_knot = self.bspline.max_knot

        self._outer_knot_left = float(self.bspline.knots[0])
        self._outer_knot_right = float(self.bspline.knots[-1])
        self._ngrid_inverse = int(
            int(self.bspline.knots[-1] - self.bspline.knots[0]) * 100
        )

        def _compute_coef_default(x):
            raise NotImplementedError

        self._compute_coef = _compute_coef_default

    def compute_coef(self, raw_coef: Array) -> Array:
        """
        Compute contrained spline coefficients from raw, real-valued coefficients.
        """
        self._check_coef_core_shape(raw_coef)
        return self._compute_coef(raw_coef)

    @staticmethod
    def _shape_tuple(shape) -> tuple[int, ...]:
        return tuple(int(dim) for dim in tuple(shape))

    @staticmethod
    def _check_coef_core_shape(coef: Array) -> None:
        if jnp.ndim(coef) < 2:
            raise ValueError(
                "Spline coefficients must have shape (..., n_coef, n_param). "
                "Use shape (1, n_param) for a shared, unbatched spline."
            )

    def _coef_batch_shape(self, coef: Array) -> tuple[int, ...]:
        self._check_coef_core_shape(coef)
        coef_shape = self._shape_tuple(jnp.shape(coef))
        batch_shape = coef_shape[:-2]
        n_coef = coef_shape[-2]

        if n_coef == 1:
            return batch_shape

        return batch_shape + (n_coef,)

    def _coef_leading_batch_shape(self, coef: Array) -> tuple[int, ...]:
        self._check_coef_core_shape(coef)
        return self._shape_tuple(jnp.shape(coef)[:-2])

    def _value_matches_shared_coef_batch(self, value: Array, coef: Array) -> bool:
        """
        Whether value is one scalar per leading batch member of shared coefficients.
        """
        return (
            jnp.ndim(value) > 0
            and jnp.shape(coef)[-2] == 1
            and self._shape_tuple(jnp.shape(value))
            == self._coef_leading_batch_shape(coef)
        )

    def _tfp_batch_shape(
        self, coef: Array, batch_shape: tuple[int, ...] | None
    ) -> tuple[int, ...]:
        if batch_shape is None:
            return self._coef_batch_shape(coef)

        return self._shape_tuple(batch_shape)

    @staticmethod
    def _broadcast_tfp_value(
        value: Array, batch_shape: tuple[int, ...]
    ) -> tuple[Array, tuple[int, ...], tuple[int, ...]]:
        value = jnp.asarray(value)
        value_shape = tuple(jnp.shape(value))

        if not batch_shape:
            sample_shape = value_shape
            result_batch_shape: tuple[int, ...] = ()
        elif len(value_shape) > len(batch_shape):
            sample_shape = value_shape[: -len(batch_shape)]
            value_batch_shape = value_shape[-len(batch_shape) :]
            result_batch_shape = jnp.broadcast_shapes(value_batch_shape, batch_shape)
        else:
            sample_shape = ()
            result_batch_shape = jnp.broadcast_shapes(value_shape, batch_shape)

        output_shape = sample_shape + result_batch_shape
        value = jnp.broadcast_to(value, output_shape)

        return value, sample_shape, result_batch_shape

    @staticmethod
    def _tfp_to_legacy_batch_last(
        value: Array, batch_shape: tuple[int, ...], sample_shape: tuple[int, ...]
    ) -> Array:
        sample_rank = len(sample_shape)
        batch_rank = len(batch_shape)
        sample_size = int(np.prod(sample_shape)) if sample_shape else 1

        axes = tuple(range(sample_rank, sample_rank + batch_rank)) + tuple(
            range(sample_rank)
        )
        value = jnp.transpose(value, axes)
        return jnp.reshape(value, batch_shape + (sample_size,))

    @staticmethod
    def _legacy_batch_last_to_tfp(
        value: Array, batch_shape: tuple[int, ...], sample_shape: tuple[int, ...]
    ) -> Array:
        sample_rank = len(sample_shape)
        batch_rank = len(batch_shape)

        value = jnp.reshape(value, batch_shape + sample_shape)
        axes = tuple(range(batch_rank, batch_rank + sample_rank)) + tuple(
            range(batch_rank)
        )
        return jnp.transpose(value, axes)

    def _coef_uses_rowwise_eval(self, coef: Array) -> bool:
        return jnp.shape(coef)[-2] > 1

    def _coef_for_eval(self, x: Array, coef: Array) -> Array:
        n_eval = 1 if jnp.ndim(x) == 0 else jnp.shape(x)[-1]
        n_coef = jnp.shape(coef)[-2]

        if n_coef == 1:
            return coef

        if n_coef != n_eval:
            raise ValueError(
                "Spline coefficients with n_coef > 1 must match the evaluation "
                f"axis length. Got {n_coef=} and {n_eval=}."
            )

        if not self.supports_rowwise_coef:
            raise ValueError(
                "Spline coefficients with n_coef > 1 require a spline that "
                "supports rowwise coefficients."
            )

        return coef

    def _broadcast_value_and_coef(
        self, value: Array, coef: Array
    ) -> tuple[Array, Array, bool]:
        value = jnp.asarray(value)
        was_scalar = jnp.ndim(value) == 0
        if was_scalar:
            value = jnp.reshape(value, (1,))

        coef = self.compute_coef(raw_coef=coef)

        if self._value_matches_shared_coef_batch(value, coef):
            value = jnp.expand_dims(value, -1)
            was_scalar = True

        batch_value = value.shape[:-1]
        batch_coef = self._coef_leading_batch_shape(coef)
        target_batch = jnp.broadcast_shapes(batch_value, batch_coef)

        value = _broadcast_leading_core(value, target_batch, core_ndims=1)
        coef = _broadcast_leading_core(coef, target_batch, core_ndims=2)

        return value, coef, was_scalar

    def _squeeze_scalar_result(self, value: Array, was_scalar: bool) -> Array:
        if was_scalar:
            return value.squeeze(axis=-1)
        return value

    def _evaluate_spline(self, value: Array, coef: Array) -> tuple[Array, Array]:
        """
        Evaluate constrained, broadcasted spline coefficients at broadcasted values.
        """
        coef = self._coef_for_eval(value, coef)
        return self.bspline.dot_and_deriv_n(value, coef)

    def _dot_and_deriv_broadcast(
        self, value: Array, coef: Array
    ) -> tuple[Array, Array]:
        value, coef, was_scalar = self._broadcast_value_and_coef(value, coef)
        dot, deriv = self._evaluate_spline(value, coef)
        return (
            self._squeeze_scalar_result(dot, was_scalar),
            self._squeeze_scalar_result(deriv, was_scalar),
        )

    def _dot_inverse_broadcast(self, value: Array, coef: Array) -> Array:
        value, coef, was_scalar = self._broadcast_value_and_coef(value, coef)

        target_batch = value.shape[:-1]
        n = value.shape[-1]
        n_coef, p = coef.shape[-2:]
        B = int(np.prod(target_batch)) if target_batch else 1

        if n_coef != 1:
            if n_coef != n:
                raise ValueError(
                    "Spline coefficients with n_coef > 1 must match the evaluation "
                    f"axis length. Got {n_coef=} and {n=}."
                )

            if not self.supports_rowwise_coef:
                raise ValueError(
                    "Spline coefficients with n_coef > 1 require a spline that "
                    "supports rowwise coefficients."
                )

        value_flat = jnp.reshape(value, (B, n))
        coef_flat = jnp.reshape(coef, (B, n_coef, p))

        def inv_shared(value_row: Array, coef_row: Array) -> Array:
            def fn(x):
                return self._evaluate_spline(jnp.atleast_1d(x), coef_row)[0]

            return inv1d(
                value_row,
                fn,
                self._outer_knot_left,
                self._outer_knot_right,
                self._ngrid_inverse,
                "monotonic",
            )

        def inv_rowwise(value_row: Array, coef_row: Array) -> Array:
            if coef_row.shape[-2] == 1:
                return inv_shared(value_row, coef_row)

            def inv_one(value_i: Array, coef_i: Array) -> Array:
                coef_i = jnp.expand_dims(coef_i, axis=0)
                return inv_shared(jnp.atleast_1d(value_i), coef_i)[0]

            return jax.vmap(inv_one)(value_row, coef_row)

        inverse_flat = jax.vmap(inv_rowwise)(value_flat, coef_flat)
        inverse = jnp.reshape(inverse_flat, target_batch + (n,))
        return self._squeeze_scalar_result(inverse, was_scalar)

    def dot_and_deriv_n_fullbatch(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative without chunking over observations.
        """
        return self._dot_and_deriv_broadcast(x, coef)

    def dot_inverse_n(self, x: Array, coef: Array) -> Array:
        """
        Compute inverse spline values for x.
        """
        return self._dot_inverse_broadcast(x, coef)

    def dot_inverse_n_fullbatch(self, x: Array, coef: Array) -> Array:
        """
        Compute inverse spline values without chunking over observations.
        """
        return self._dot_inverse_broadcast(x, coef)

    def dot_and_deriv_n(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative for x.
        """
        return self._dot_and_deriv_broadcast(x, coef)

    def dot_and_deriv(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative in the legacy spline layout.

        For standard splines, ``x`` has shape ``(...batch, n)`` and ``coef`` has
        shape ``(...batch, 1, p)``. For rowwise splines, ``x`` has shape
        ``(...batch, n)`` and ``coef`` has shape ``(...batch, n, p)``.
        """
        return self._dot_and_deriv_broadcast(x, coef)

    def dot_inverse(self, x: Array, coef: Array) -> Array:
        """
        Compute inverse spline values in the legacy spline layout.

        For standard splines, ``x`` has shape ``(...batch, n)`` and ``coef`` has
        shape ``(...batch, 1, p)``. For rowwise splines, ``x`` has shape
        ``(...batch, n)`` and ``coef`` has shape ``(...batch, n, p)``.
        """
        return self._dot_inverse_broadcast(x, coef)

    def dot_and_deriv_tfp(
        self,
        value: Array,
        coef: Array,
        batch_shape: tuple[int, ...] | None = None,
    ) -> tuple[Array, Array]:
        """
        Compute dot product and derivative in TFP scalar-event layout.

        ``value`` follows TensorFlow Probability's scalar-event convention:
        leading axes are sample axes and trailing axes broadcast against
        ``batch_shape``. The returned arrays have shape
        ``broadcast(value.shape, batch_shape)``.
        """
        batch_shape = self._tfp_batch_shape(coef, batch_shape)
        value, sample_shape, result_batch_shape = self._broadcast_tfp_value(
            value, batch_shape
        )

        if self._coef_uses_rowwise_eval(coef):
            return self.dot_and_deriv_n_fullbatch(value, coef)

        value = self._tfp_to_legacy_batch_last(value, result_batch_shape, sample_shape)
        dot, deriv = self.dot_and_deriv(value, coef)

        dot = self._legacy_batch_last_to_tfp(dot, result_batch_shape, sample_shape)
        deriv = self._legacy_batch_last_to_tfp(deriv, result_batch_shape, sample_shape)

        return dot, deriv

    def dot_inverse_tfp(
        self,
        value: Array,
        coef: Array,
        batch_shape: tuple[int, ...] | None = None,
    ) -> Array:
        """
        Compute inverse spline values in TFP scalar-event layout.

        ``value`` follows TensorFlow Probability's scalar-event convention:
        leading axes are sample axes and trailing axes broadcast against
        ``batch_shape``. The returned array has shape
        ``broadcast(value.shape, batch_shape)``.
        """
        batch_shape = self._tfp_batch_shape(coef, batch_shape)
        value, sample_shape, result_batch_shape = self._broadcast_tfp_value(
            value, batch_shape
        )

        if self._coef_uses_rowwise_eval(coef):
            return self.dot_inverse_n_fullbatch(value, coef)

        value = self._tfp_to_legacy_batch_last(value, result_batch_shape, sample_shape)
        inverse = self.dot_inverse(value, coef)

        return self._legacy_batch_last_to_tfp(inverse, result_batch_shape, sample_shape)
