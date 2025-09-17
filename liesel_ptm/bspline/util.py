from __future__ import annotations

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
    trailing = x.shape[-1:]  # (n,) for x, (p,) for coef
    need = len(target_batch_shape) - (x.ndim - 1)
    shape = (1,) * need + x.shape
    x_reshaped = jnp.reshape(x, shape)
    return jnp.broadcast_to(x_reshaped, target_batch_shape + trailing)


class TransformationSpline:
    def __init__(self, knots: Array):
        """
        Parent class for monotonically transformation splines.

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
        self.bspline = BSplineApprox(knots, order=3, ngrid=1000, postmultiply_by=S)

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
        return self._compute_coef(raw_coef)

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
        dot, deriv = self._dot_and_deriv_n_fullbatch(x, coef)
        if was_scalar:
            return dot[0], deriv[0]

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

            coef = jnp.atleast_2d(coef)
            if coef.shape[-2] == 1:
                coef = jnp.squeeze(coef, -2)

            # Extract leading (batch) shapes
            batch_x = x.shape[:-1]
            batch_c = coef.shape[:-1]

            # Compute common leading batch shape via broadcasting rules
            target_batch = jnp.broadcast_shapes(batch_x, batch_c)  # tuple

            # Broadcast both to the same leading batch shape
            x_bt = _broadcast_leading(x, target_batch)  # target_batch + (n,)
            coef_bt = _broadcast_leading(coef, target_batch)  # target_batch + (p,)

            # Flatten the leading batch to a single dimension B
            B = int(np.prod(target_batch)) if target_batch else 1
            x_flat = jnp.reshape(x_bt, (B, x.shape[-1]))  # (B, n)
            coef_flat = jnp.reshape(coef_bt, (B, coef.shape[-1]))  # (B, p)

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

            coef = jnp.atleast_2d(coef)
            if coef.shape[-2] == 1:
                coef = jnp.squeeze(coef, -2)

            # Leading (batch) shapes
            batch_x = x.shape[:-1]  # possibly ()
            batch_c = coef.shape[:-1]  # possibly ()

            # Broadcast to common leading batch shape
            target_batch = jnp.broadcast_shapes(batch_x, batch_c)

            x_bt = _broadcast_leading(x, target_batch)  # target_batch + (n,)
            coef_bt = _broadcast_leading(coef, target_batch)  # target_batch + (p,)

            # Flatten leading batch
            B = int(np.prod(target_batch)) if target_batch else 1
            n = x_bt.shape[-1]
            p = coef_bt.shape[-1]

            x_flat = jnp.reshape(x_bt, (B, n))  # (B, n)
            coef_flat = jnp.reshape(coef_bt, (B, p))  # (B, p)

            # Scan over flattened batch (avoids (B, n, p) temporaries)
            def body(carry, inputs):
                x_row, coef_row = inputs
                dot_row = bdd_over_n(x_row, coef_row)  # -> (n,)
                return carry, dot_row

            _, dot_flat = jax.lax.scan(body, None, (x_flat, coef_flat))  # (B, n)

            # Reshape back to broadcast batch shape
            out = jnp.reshape(dot_flat, target_batch + (n,))  # (..., n)

            # Keep the documented (..., n) contract: scalar x -> (..., 1)
            if x_was_scalar:
                return out.squeeze(axis=-1)
            return out

        return batched_fn

    def dot_and_deriv(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative for x with shape (B,n) and coef with
        shape (B,P) or (B, n, P).
        """
        return self._dot_and_deriv(x, coef)

    def dot_inverse(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative for x with shape (B,n) and coef with
        shape (B,P) or (B, n, P).
        """
        return self._dot_inverse(x, coef)
