from __future__ import annotations

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax import Array
from jax.tree_util import Partial as partial

from .approx import BSplineApprox, bspline_basis
from .util import TransformationSpline


def cumsum_leading_zero(exp_shape: Array) -> Array:
    """
    Cumulative sum with a leading zero.
    """
    zeros_shape = jnp.shape(exp_shape)[:-1] + (1,)
    exp_shape = jnp.concatenate((jnp.zeros(zeros_shape), exp_shape), axis=-1)
    return jnp.cumsum(exp_shape, axis=-1)


class MiCoef:
    def __init__(self, knots: Array, bijector: tfb.Bijector = tfb.Exp()) -> None:
        """
        Initialize PTMCoef functionality with spline knots.

        Assumes knots were created for a cubic spline.

        Parameters
        ----------
        knots
            Spline knot sequence for a cubic spline.

        Attributes
        ----------
        knots
            Array of spline knots.
        k1
            Value of knots[3].
        B
            Basis matrix for cumulative sum.
        B0
            Basis matrix at zero for cumulative sum.
        step
            Step size between knots.
        """
        self.knots = knots
        self.k1 = knots[3]
        B = bspline_basis(jnp.atleast_1d(knots[3]), knots, 3)
        B0 = bspline_basis(jnp.zeros(1), knots, 3)
        S = jnp.tril(jnp.ones((B.shape[-1], B.shape[-1])))  # for cumulative sum
        self.B = B @ S
        self.B0 = B0 @ S
        self.bijector = bijector
        self.zeros = jnp.zeros(self.B.shape[-1])

        self.step = jnp.diff(knots).mean()

    def _add_intercept_and_biject(self, log_increments):
        """
        Add intercept and exponentiate log increments.
        """
        exp_coef = self.bijector.forward(log_increments)
        prelim_coef = self.zeros.at[1:].set(exp_coef)
        fx_at_zero = (self.B0 @ prelim_coef).squeeze()

        full_coef = prelim_coef.at[0].set(-fx_at_zero)
        return full_coef


class MiSpline(TransformationSpline):
    """
    PTM spline transformation using given knots.

    Parameters
    ----------
    knots
        Increasing, equidistant spline knot sequence.

    Attributes
    ----------
    knots
        Array of spline knots.
    transition_width
        Width of the transition interval.
    min_eps
        Left boundary for tail transition.
    max_eps
        Right boundary for tail transition.
    """

    def __init__(
        self,
        knots: Array,
        bijector: tfb.Bijector = tfb.Exp(),
    ) -> None:
        self.n_chunks = 1024
        self.knots = knots

        self._nparam = knots.size - 4  # len(knots) - order - 1; order is fixed to 3
        S = jnp.tril(jnp.ones((self._nparam, self._nparam)))
        self.bspline = BSplineApprox(knots, order=3, ngrid=1000, postmultiply_by=S)

        self.min_knot = self.bspline.min_knot
        self.max_knot = self.bspline.max_knot

        self._outer_knot_left = float(self.bspline.knots[0])
        self._outer_knot_right = float(self.bspline.knots[-1])
        self._ngrid_inverse = int(
            int(self.bspline.knots[-1] - self.bspline.knots[0]) * 100
        )

        micoef = MiCoef(knots, bijector=bijector)
        self._compute_coef = jax.jit(micoef._add_intercept_and_biject)

        eps = 100000.0

        self.transition_width = eps * (self.max_knot - self.min_knot)
        self.min_eps = self.min_knot - self.transition_width
        self.max_eps = self.max_knot + self.transition_width

        def target_slope(coef):
            return 1.0

        self._target_slope_left = target_slope
        self._target_slope_right = target_slope

        self._boundaries = jnp.array([self.min_knot, self.max_knot])

        self._dot_and_deriv_n = self._vmap_over_n_chunked(
            self._dot_and_deriv_n_fullbatch
        )
        self._dot_and_deriv = self._batch_in_chunks(self.dot_and_deriv_n)
        self._dot_inverse = self._batch_in_chunks_inverse(self.dot_inverse_n)

    def _left_transition_and_deriv(self, x, coef, value_left, deriv_left):
        """
        Compute left transition value and derivative.
        """
        poly = x * self.min_knot - 0.5 * x * x

        target_slope_left = self._target_slope_left(coef)

        unsh = (target_slope_left / self.transition_width) * poly + deriv_left * (
            x - poly / self.transition_width
        )
        x0 = self.min_knot
        poly0 = x0 * self.min_knot - 0.5 * x0 * x0
        const = value_left - (
            (target_slope_left / self.transition_width) * poly0
            + deriv_left * (x0 - poly0 / self.transition_width)
        )

        value = unsh + const

        dist = (self.min_knot - x) / self.transition_width
        deriv = (1.0 - dist) * deriv_left + target_slope_left * dist
        return value, deriv

    def _right_transition_and_deriv(self, x, coef, value_right, deriv_right):
        """
        Compute right transition value and derivative.
        """
        poly = 0.5 * x * x - x * self.max_knot

        target_slope_right = self._target_slope_right(coef)

        unsh = (target_slope_right / self.transition_width) * poly + deriv_right * (
            x - poly / self.transition_width
        )
        x0 = self.max_knot
        poly0 = 0.5 * x0 * x0 - x0 * self.max_knot
        const = value_right - (
            (target_slope_right / self.transition_width) * poly0
            + deriv_right * (x0 - poly0 / self.transition_width)
        )
        value = unsh + const
        dist = (x - self.max_knot) / self.transition_width
        der = (1.0 - dist) * deriv_right + target_slope_right * dist

        return value, der

    def _left_tail_and_deriv(self, x, coef, fx_at_linear_start):
        """
        Compute left tail value and derivative.
        """
        target_slope_left = self._target_slope_left(coef)
        val = fx_at_linear_start - target_slope_left * (self.min_eps - x)
        return val, target_slope_left

    def _right_tail_and_deriv(self, x, coef, fx_at_linear_start):
        """
        Compute right tail value and derivative.
        """
        target_slope_right = self._target_slope_right(coef)
        val = fx_at_linear_start + target_slope_right * (x - self.max_eps)
        return val, target_slope_right

    def _dot_and_deriv_n_fullbatch(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative for batch.
        """
        fx_n, deriv_n = self.bspline.dot_and_deriv_n(x, coef)
        boundary_values, boundary_derivs = self.bspline.dot_and_deriv_n(
            self._boundaries, coef
        )

        left_transition = partial(
            self._left_transition_and_deriv,
            value_left=boundary_values[0],
            deriv_left=boundary_derivs[0],
        )

        right_transition = partial(
            self._right_transition_and_deriv,
            value_right=boundary_values[1],
            deriv_right=boundary_derivs[1],
        )

        left_tail = partial(
            self._left_tail_and_deriv,
            fx_at_linear_start=left_transition(self.min_eps, coef)[0],
        )

        right_tail = partial(
            self._right_tail_and_deriv,
            fx_at_linear_start=right_transition(self.max_eps, coef)[0],
        )

        def branches(x, fx_n, deriv_n):
            def fxderiv(x, coef):
                return fx_n, deriv_n

            code = jnp.where(
                # check most common case first
                (x >= self.min_knot) & (x <= self.max_knot),
                2,
                jnp.where(
                    x < self.min_eps,
                    0,
                    jnp.where(
                        x < self.min_knot,
                        1,
                        jnp.where(x < self.max_eps, 3, 4),
                    ),
                ),
            )

            value, deriv = jax.lax.switch(
                code,
                (
                    left_tail,
                    left_transition,
                    fxderiv,
                    right_transition,
                    right_tail,
                ),
                x,
                coef,
            )

            return value, deriv

        value, deriv = jax.vmap(branches)(x, fx_n, deriv_n)
        return value, deriv
