from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..util.inverse_interpax import inv1d
from .approx import BSplineApprox

inv1d = jax.jit(inv1d, static_argnums=(1, 2, 3, 4, 5))


def _broadcast_leading(x, target_batch_shape):
    """
    Prepend singleton axes so x can broadcast to target_batch_shape + x.shape[-1:].
    """
    return _broadcast_leading_core(x, target_batch_shape, core_ndims=1)


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
            Chunk size for batching operations when streaming over large numbers of
            observations.
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
        self.knots = knots

        self._nparam = knots.size - 4  # len(knots) - order - 1; order is fixed to 3
        S = jnp.tril(jnp.ones((self._nparam, self._nparam)))
        self.bspline = BSplineApprox(
            knots, order=3, ngrid=1000, postmultiply_by=S, subscripts=subscripts
        )

        self.min_knot = self.bspline.min_knot
        self.max_knot = self.bspline.max_knot

        self._dot_and_deriv_n = self._vmap_over_n_chunked(
            self._dot_and_deriv_n_fullbatch
        )
        self._dot_and_deriv = self._batch_in_chunks(self.dot_and_deriv_n)

        self._dot_inverse = self._batch_in_chunks_inverse(self.dot_inverse_n)

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
    ) -> tuple[Array, tuple[int, ...]]:
        value = jnp.asarray(value)
        output_shape = jnp.broadcast_shapes(jnp.shape(value), batch_shape)
        value = jnp.broadcast_to(value, output_shape)

        if batch_shape:
            sample_shape = output_shape[: -len(batch_shape)]
        else:
            sample_shape = output_shape

        return value, sample_shape

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

    def _is_rowwise_spline(self) -> bool:
        return self.bspline.subscripts == "...nj,...nj->...n"

    def _coef_for_eval(self, x: Array, coef: Array) -> Array:
        n_eval = 1 if jnp.ndim(x) == 0 else jnp.shape(x)[-1]
        n_coef = jnp.shape(coef)[-2]

        if n_coef == 1:
            shared_coef = coef[..., 0, :]

            if self._is_rowwise_spline():
                return jnp.broadcast_to(
                    jnp.expand_dims(shared_coef, -2),
                    jnp.shape(shared_coef)[:-1] + (n_eval, jnp.shape(shared_coef)[-1]),
                )

            return shared_coef

        if n_coef != n_eval:
            raise ValueError(
                "Spline coefficients with n_coef > 1 must match the evaluation "
                f"axis length. Got {n_coef=} and {n_eval=}."
            )

        if not self._is_rowwise_spline():
            raise ValueError(
                "Spline coefficients with n_coef > 1 require a rowwise spline "
                "with subscripts='...nj,...nj->...n'."
            )

        return coef

    def _dot_and_deriv_n_fullbatch(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Assumes coef is already computed.
        """
        raise NotImplementedError

    def dot_and_deriv_n_fullbatch(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative without batching over observation axis.
        """
        was_scalar = jnp.ndim(x) == 0
        if was_scalar:
            x = jnp.atleast_1d(x)
        coef = self.compute_coef(raw_coef=coef)

        if self._value_matches_shared_coef_batch(x, coef):
            x = jnp.expand_dims(x, -1)
            was_scalar = True

        batch_x = x.shape[:-1]
        batch_c = self._coef_leading_batch_shape(coef)
        target_batch = jnp.broadcast_shapes(batch_x, batch_c)

        x_bt = _broadcast_leading_core(x, target_batch, core_ndims=1)
        coef_bt = _broadcast_leading_core(coef, target_batch, core_ndims=2)

        B = int(np.prod(target_batch)) if target_batch else 1
        n = x_bt.shape[-1]
        n_coef, p = coef_bt.shape[-2:]

        x_flat = jnp.reshape(x_bt, (B, n))
        coef_flat = jnp.reshape(coef_bt, (B, n_coef, p))

        def body(carry, inputs):
            x_row, coef_row = inputs
            dot_row, deriv_row = self._dot_and_deriv_n_fullbatch(x_row, coef_row)
            return carry, (dot_row, deriv_row)

        _, (dot_flat, deriv_flat) = jax.lax.scan(body, None, (x_flat, coef_flat))
        dot = jnp.reshape(dot_flat, target_batch + (n,))
        deriv = jnp.reshape(deriv_flat, target_batch + (n,))

        if was_scalar:
            return dot.squeeze(axis=-1), deriv.squeeze(axis=-1)

        return dot, deriv

    def dot_inverse_n(self, y: Array, coef: Array) -> Array:
        """
        Compute inverse spline for y with shape (n,).
        """

        def fn(x):
            return self.dot_and_deriv_n(x, coef)[0]

        x = inv1d(
            y,
            fn,
            self._outer_knot_left,
            self._outer_knot_right,
            self._ngrid_inverse,
            "monotonic",
        )

        return x

    def dot_inverse_n_fullbatch(self, x: Array, coef: Array) -> Array:
        """
        Compute inverse spline for y with shape (n,).
        """
        was_scalar = jnp.ndim(x) == 0
        if was_scalar:
            y = jnp.atleast_1d(x)

        coef = self.compute_coef(raw_coef=coef)

        if self._value_matches_shared_coef_batch(y, coef):
            y = jnp.expand_dims(y, -1)
            was_scalar = True

        batch_y = y.shape[:-1]
        batch_c = self._coef_leading_batch_shape(coef)
        target_batch = jnp.broadcast_shapes(batch_y, batch_c)

        y_bt = _broadcast_leading_core(y, target_batch, core_ndims=1)
        coef_bt = _broadcast_leading_core(coef, target_batch, core_ndims=2)

        B = int(np.prod(target_batch)) if target_batch else 1
        n = y_bt.shape[-1]
        n_coef, p = coef_bt.shape[-2:]

        y_flat = jnp.reshape(y_bt, (B, n))
        coef_flat = jnp.reshape(coef_bt, (B, n_coef, p))

        def _inv(y: Array, coef: Array) -> Array:
            def fn(x):
                return self._dot_and_deriv_n_fullbatch(jnp.atleast_1d(x), coef)[0]

            x = inv1d(
                y,
                fn,
                self._outer_knot_left,
                self._outer_knot_right,
                self._ngrid_inverse,
                "monotonic",
            )

            return x

        def _inv_row(y: Array, coef: Array) -> Array:
            if coef.shape[-2] == 1:
                return _inv(y, coef)

            def _inv_one(y_i: Array, coef_i: Array) -> Array:
                return _inv(jnp.atleast_1d(y_i), jnp.expand_dims(coef_i, 0))[0]

            return jax.vmap(_inv_one)(y, coef)

        def body(carry, inputs):
            y_row, coef_row = inputs
            return carry, _inv_row(y_row, coef_row)

        _, out_flat = jax.lax.scan(body, None, (y_flat, coef_flat))
        out = jnp.reshape(out_flat, target_batch + (n,))

        if was_scalar:
            return out.squeeze(axis=-1)

        return out

    def dot_and_deriv_n(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative for x with shape (n,).
        """
        coef = self.compute_coef(raw_coef=coef)
        return self._dot_and_deriv_n(x, coef)

    def _vmap_over_n_chunked(self, fn):
        """
        Vectorize function over chunks.
        """
        n_chunk = self.n_chunks

        # vmap over a small chunk of length n_chunk
        bdd_over_chunk = fn

        def bdd_over_n_chunked(x_row: jnp.ndarray, coef_row: jnp.ndarray):
            """
            x_row:    (n,) or scalar (())
            coef_row: (p,)
            returns:  (n,), (n,)  or  scalar (), scalar () if x_row is scalar
            """
            # Accept scalars and vectors
            was_scalar = jnp.ndim(x_row) == 0
            x_vec = jnp.atleast_1d(x_row)

            n = x_vec.shape[0]  # static at trace-time
            pad = (-n) % n_chunk  # static int

            if pad:
                x_vec = jnp.pad(x_vec, (0, pad))

            num_blocks = x_vec.shape[0] // n_chunk  # static int

            # (num_blocks, n_chunk)
            x_blocks = x_vec.reshape((num_blocks, n_chunk))

            # map chunk-by-chunk; avoids a single (n, p) intermediate
            dot_blocks, der_blocks = jax.vmap(bdd_over_chunk, in_axes=(0, None))(
                x_blocks, coef_row
            )  # -> (num_blocks, n_chunk) each

            # stitch blocks back together and drop padding
            dot = dot_blocks.reshape((num_blocks * n_chunk,))[:n]
            deriv = der_blocks.reshape((num_blocks * n_chunk,))[:n]

            # If the input was scalar, return scalars
            if was_scalar:
                return dot[0], deriv[0]
            return dot, deriv

        return bdd_over_n_chunked

    def _batch_in_chunks(self, fn):
        """
        Batch function in chunks.
        """
        """
        x:    (...Bx..., n)
        coef: (...Bc..., p)
        Returns:
        dot, deriv with shape broadcast(...Bx..., ...Bc...) + (n,)
        """
        bdd_over_n = fn

        def batched_fn(x, coef):
            # Allow scalar x by promoting to length-1 vector so a trailing axis exists
            x_was_scalar = jnp.ndim(x) == 0
            if x_was_scalar:
                x = jnp.reshape(x, (1,))  # n = 1

            self._check_coef_core_shape(coef)

            if self._value_matches_shared_coef_batch(x, coef):
                x = jnp.expand_dims(x, -1)
                x_was_scalar = True

            # Extract leading (batch) shapes
            batch_x = x.shape[:-1]
            batch_c = coef.shape[:-2]

            # Compute common leading batch shape via broadcasting rules
            target_batch = jnp.broadcast_shapes(batch_x, batch_c)  # tuple

            # Broadcast both to the same leading batch shape
            x_bt = _broadcast_leading(x, target_batch)  # target_batch + (n,)
            coef_bt = _broadcast_leading_core(
                coef, target_batch, core_ndims=2
            )  # target_batch + (n_coef, p)

            # Flatten the leading batch to a single dimension B
            B = int(np.prod(target_batch)) if target_batch else 1
            n_coef, p = coef_bt.shape[-2:]
            x_flat = jnp.reshape(x_bt, (B, x.shape[-1]))  # (B, n)
            coef_flat = jnp.reshape(coef_bt, (B, n_coef, p))  # (B, n_coef, p)

            # Scan over the flattened batch to avoid (B, n, p) temporaries
            def body(carry, inputs):
                x_row, coef_row = inputs
                dot_row, deriv_row = bdd_over_n(x_row, coef_row)  # (n,), (n,)
                return carry, (dot_row, deriv_row)

            carry = None
            carry, (dot_flat, deriv_flat) = jax.lax.scan(
                body, carry, (x_flat, coef_flat)
            )  # (B, n)

            # Reshape back to the broadcast batch shape
            out_shape = target_batch + (x.shape[-1],)  # (..., n)
            dot = jnp.reshape(dot_flat, out_shape)
            deriv = jnp.reshape(deriv_flat, out_shape)
            if x_was_scalar:
                return dot.squeeze(axis=-1), deriv.squeeze(axis=-1)
            return dot, deriv

        return batched_fn

    def _batch_in_chunks_inverse(self, fn):
        """
        Batch inverse function in chunks.
        """
        bdd_over_n = fn

        def batched_fn(x, coef):
            # Allow scalar x by promoting to length-1 vector so a trailing axis exists
            x_was_scalar = jnp.ndim(x) == 0
            if x_was_scalar:
                x = jnp.reshape(x, (1,))  # n = 1

            self._check_coef_core_shape(coef)

            if self._value_matches_shared_coef_batch(x, coef):
                x = jnp.expand_dims(x, -1)
                x_was_scalar = True

            # Leading (batch) shapes
            batch_x = x.shape[:-1]  # possibly ()
            batch_c = coef.shape[:-2]  # possibly ()

            # Broadcast to common leading batch shape
            target_batch = jnp.broadcast_shapes(batch_x, batch_c)

            x_bt = _broadcast_leading(x, target_batch)  # target_batch + (n,)
            coef_bt = _broadcast_leading_core(
                coef, target_batch, core_ndims=2
            )  # target_batch + (n_coef, p)

            # Flatten leading batch
            B = int(np.prod(target_batch)) if target_batch else 1
            n = x_bt.shape[-1]
            n_coef, p = coef_bt.shape[-2:]

            x_flat = jnp.reshape(x_bt, (B, n))  # (B, n)
            coef_flat = jnp.reshape(coef_bt, (B, n_coef, p))  # (B, n_coef, p)

            # Scan over flattened batch (avoids (B, n, p) temporaries)
            def body(carry, inputs):
                x_row, coef_row = inputs
                dot_row = bdd_over_n(x_row, coef_row)  # -> (n,)
                return carry, dot_row

            _, dot_flat = jax.lax.scan(body, None, (x_flat, coef_flat))  # (B, n)

            # Reshape back to broadcast batch shape
            out = jnp.reshape(dot_flat, target_batch + (n,))  # (..., n)

            if x_was_scalar:
                return out.squeeze(axis=-1)
            return out

        return batched_fn

    def dot_and_deriv(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative in the legacy spline layout.

        For standard splines, ``x`` has shape ``(...batch, n)`` and ``coef`` has
        shape ``(...batch, 1, p)``. For rowwise splines, ``x`` has shape
        ``(...batch, n)`` and ``coef`` has shape ``(...batch, n, p)``.
        """
        return self._dot_and_deriv(x, coef)

    def dot_inverse(self, x: Array, coef: Array) -> Array:
        """
        Compute inverse spline values in the legacy spline layout.

        For standard splines, ``x`` has shape ``(...batch, n)`` and ``coef`` has
        shape ``(...batch, 1, p)``. For rowwise splines, ``x`` has shape
        ``(...batch, n)`` and ``coef`` has shape ``(...batch, n, p)``.
        """
        return self._dot_inverse(x, coef)

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
        value, sample_shape = self._broadcast_tfp_value(value, batch_shape)

        if self._is_rowwise_spline():
            return self.dot_and_deriv_n_fullbatch(value, coef)

        value = self._tfp_to_legacy_batch_last(value, batch_shape, sample_shape)
        dot, deriv = self.dot_and_deriv(value, coef)

        dot = self._legacy_batch_last_to_tfp(dot, batch_shape, sample_shape)
        deriv = self._legacy_batch_last_to_tfp(deriv, batch_shape, sample_shape)

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
        value, sample_shape = self._broadcast_tfp_value(value, batch_shape)

        if self._is_rowwise_spline():
            return self.dot_inverse_n_fullbatch(value, coef)

        value = self._tfp_to_legacy_batch_last(value, batch_shape, sample_shape)
        inverse = self.dot_inverse(value, coef)

        return self._legacy_batch_last_to_tfp(inverse, batch_shape, sample_shape)
