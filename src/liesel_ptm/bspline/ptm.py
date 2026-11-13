from __future__ import annotations

from collections.abc import Callable
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .approx import BSplineApprox, bspline_basis, equidistant_knots
from .util import TransformationSpline

# Slope of the identity tails; the tail transitions move towards it.
_TAIL_SLOPE = 1.0


class PTMKnots:
    """
    Knots for a monotonically increasing PTM spline.

    Parameters
    ----------
    a
        Left boundary of the core interval.
    b
        Right boundary of the core interval.
    nparam
        Number of parameters for the spline.
    order
        Spline order.
    eps
        Stretch factor for knot spacing.

    Attributes
    ----------
    a
        Left boundary of the core interval.
    b
        Right boundary of the core interval.
    nparam
        Number of parameters for the spline.
    order
        Spline order.
    knots
        Array of spline knots.
    step
        Step size between knots.
    """

    def __init__(
        self, a: float, b: float, nparam: int, order: int = 3, eps: float = 0.0
    ) -> None:
        self.a = a
        self.b = b
        self.nparam = nparam
        self.order = order

        self.knots = equidistant_knots(
            jnp.array([a, b]), order=order, n_param=nparam + 1, eps=eps
        )

        self.step = jnp.diff(self.knots).mean()


def log_sfn(shape):
    """
    Compute log normalization factor for PTM spline coefficients.

    Parameters
    ----------
    shape
        Shape parameters.

    Returns
    -------
    Log normalization factor.
    """
    order = 3
    J = jnp.shape(shape)[-1] + 1

    a = jnp.full((J - 1,), fill_value=1.0 / 6.0)
    a = a.at[-2:].set(0.0)

    b = jnp.full((J - 1,), fill_value=2.0 / 3.0)
    b = b.at[0].set(0.0)
    b = b.at[-1].set(0.0)

    c = jnp.full((J - 1,), fill_value=1.0 / 6.0)
    c = c.at[:2].set(0.0)

    log_w = jnp.log(a + b + c)

    log_T = jax.scipy.special.logsumexp(shape + log_w)

    return log_T - jnp.log(J - order)


def normalization_coef_log(shape: Array, dknots: Array) -> Array:
    """
    Construct log spline coefficients with average slope one over the domain.

    Parameters
    ----------
    shape
        Shape parameters.
    dknots
        Knot differences.

    Returns
    -------
    Log spline coefficients.
    """
    log_s = log_sfn(shape)
    corrected_coef = shape - log_s + jnp.log(dknots)
    return corrected_coef


def get_ptm_coef_fn(knots: Array) -> Callable[[Array], Array]:
    """
    Get the function that computes PTM spline coefficients.

    Assumes knots were created for a cubic spline.

    Parameters
    ----------
    knots
        Spline knot sequence for a cubic spline.

    Returns
    -------
    compute_coef
        Function that maps log increments with shape ``(..., nparam)`` to
        cumulative-sum spline coefficients with shape ``(..., nparam + 1)``.
        The spline has average slope one and passes through ``knots[3]`` at
        ``knots[3]``.
    """
    k1 = knots[3]
    B = bspline_basis(jnp.atleast_1d(knots[3]), knots, 3)
    S = jnp.tril(jnp.ones((B.shape[-1], B.shape[-1])))  # for cumulative sum
    B = B @ S
    step = jnp.diff(knots).mean()

    def compute_row(log_increments: Array) -> Array:
        log_increments = normalization_coef_log(log_increments, step)
        exp_coef = jnp.exp(log_increments)
        prelim_coef = jnp.concatenate((jnp.zeros(1), exp_coef), axis=-1)
        offset = (B @ prelim_coef) - k1
        return jnp.concatenate((-offset, exp_coef), axis=-1)

    def compute_coef(log_increments: Array) -> Array:
        leading_shape = jnp.shape(log_increments)[:-1]
        rows = jnp.reshape(log_increments, (-1, jnp.shape(log_increments)[-1]))
        coef = jax.vmap(compute_row)(rows)
        return jnp.reshape(coef, leading_shape + coef.shape[-1:])

    return compute_coef


class PTMSpline(TransformationSpline):
    """
    PTM spline transformation using given knots.

    Parameters
    ----------
    knots
        Increasing, equidistant spline knot sequence.
    eps
        Parameter controlling the sharpness of transition to tail extrapolation. \
        Is used to compute ``transition_width = eps * (knots[3] - knots[-4])``, \
        where ``transition_width`` indicates the width of the transition interval.
    continue_linearly
        If True, there is no transition to the identity function. Instead, \
        the spline will continue linearly in the tails with the slope fixed \
        to the slopes at ``knots[3]`` and ``knots[-4]`` for left and right \
        extrapolation, respectively.

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

    Notes
    -----
    Set geometry and numerical settings before first evaluation; JIT kernels
    capture them. Reuse the spline across parameter draws. Changing settings
    requires a new spline, which creates a separate compilation-cache entry.
    """

    def __init__(
        self,
        knots: Array,
        eps: float = 0.1,
        continue_linearly: bool = False,
    ) -> None:
        if eps < 1e-6:
            raise ValueError(f"{eps=} is < 1e-6; that is numerically unstable.")

        self.knots = knots
        self._knots_np = np.asarray(jax.device_get(knots), dtype=float)

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

        self._compute_coef = jax.jit(get_ptm_coef_fn(knots))

        if continue_linearly:
            eps = 100000.0

        self.transition_width = eps * (self.max_knot - self.min_knot)
        self.min_eps = self.min_knot - self.transition_width
        self.max_eps = self.max_knot + self.transition_width

        self._boundaries = jnp.array([self.min_knot, self.max_knot])

    @classmethod
    def from_bounds(
        cls,
        a: float,
        b: float,
        nparam: int,
        eps: float = 0.1,
        continue_linearly: bool = False,
    ) -> Self:
        """
        Create a PTM spline from the bounds of its core interval.

        Parameters
        ----------
        a
            Left boundary of the core interval.
        b
            Right boundary of the core interval.
        nparam
            Number of parameters for the spline.
        eps
            Parameter controlling the sharpness of the transition to tail \
            extrapolation. See [PTMSpline][liesel_ptm.PTMSpline].
        continue_linearly
            If True, continue linearly in the tails instead of transitioning to \
            the identity function.

        Returns
        -------
        spline
            A PTM spline with [PTMKnots][liesel_ptm.PTMKnots].
        """
        knots = PTMKnots(a, b, nparam=nparam).knots
        return cls(knots, eps=eps, continue_linearly=continue_linearly)

    def _left_transition_and_deriv(self, x, value_left, deriv_left):
        """
        Compute left transition value and derivative.
        """
        poly = x * self.min_knot - 0.5 * x * x

        unsh = (_TAIL_SLOPE / self.transition_width) * poly + deriv_left * (
            x - poly / self.transition_width
        )
        x0 = self.min_knot
        poly0 = x0 * self.min_knot - 0.5 * x0 * x0
        const = value_left - (
            (_TAIL_SLOPE / self.transition_width) * poly0
            + deriv_left * (x0 - poly0 / self.transition_width)
        )

        value = unsh + const

        dist = (self.min_knot - x) / self.transition_width
        deriv = (1.0 - dist) * deriv_left + _TAIL_SLOPE * dist
        return value, deriv

    def _left_transition_value(self, x, value_left, deriv_left):
        """
        Compute left transition value.
        """
        poly = x * self.min_knot - 0.5 * x * x

        unsh = (_TAIL_SLOPE / self.transition_width) * poly + deriv_left * (
            x - poly / self.transition_width
        )
        x0 = self.min_knot
        poly0 = x0 * self.min_knot - 0.5 * x0 * x0
        const = value_left - (
            (_TAIL_SLOPE / self.transition_width) * poly0
            + deriv_left * (x0 - poly0 / self.transition_width)
        )

        return unsh + const

    def _right_transition_and_deriv(self, x, value_right, deriv_right):
        """
        Compute right transition value and derivative.
        """
        poly = 0.5 * x * x - x * self.max_knot

        unsh = (_TAIL_SLOPE / self.transition_width) * poly + deriv_right * (
            x - poly / self.transition_width
        )
        x0 = self.max_knot
        poly0 = 0.5 * x0 * x0 - x0 * self.max_knot
        const = value_right - (
            (_TAIL_SLOPE / self.transition_width) * poly0
            + deriv_right * (x0 - poly0 / self.transition_width)
        )
        value = unsh + const
        dist = (x - self.max_knot) / self.transition_width
        der = (1.0 - dist) * deriv_right + _TAIL_SLOPE * dist

        return value, der

    def _right_transition_value(self, x, value_right, deriv_right):
        """
        Compute right transition value.
        """
        poly = 0.5 * x * x - x * self.max_knot

        unsh = (_TAIL_SLOPE / self.transition_width) * poly + deriv_right * (
            x - poly / self.transition_width
        )
        x0 = self.max_knot
        poly0 = 0.5 * x0 * x0 - x0 * self.max_knot
        const = value_right - (
            (_TAIL_SLOPE / self.transition_width) * poly0
            + deriv_right * (x0 - poly0 / self.transition_width)
        )
        return unsh + const

    def _left_tail_and_deriv(self, x, fx_at_linear_start):
        """
        Compute left tail value and derivative.
        """
        val = fx_at_linear_start - _TAIL_SLOPE * (self.min_eps - x)
        return val, _TAIL_SLOPE

    def _left_tail_value(self, x, fx_at_linear_start):
        """
        Compute left tail value.
        """
        return fx_at_linear_start - _TAIL_SLOPE * (self.min_eps - x)

    def _right_tail_and_deriv(self, x, fx_at_linear_start):
        """
        Compute right tail value and derivative.
        """
        val = fx_at_linear_start + _TAIL_SLOPE * (x - self.max_eps)
        return val, _TAIL_SLOPE

    def _right_tail_value(self, x, fx_at_linear_start):
        """
        Compute right tail value.
        """
        return fx_at_linear_start + _TAIL_SLOPE * (x - self.max_eps)

    def _evaluate_spline(self, value: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative for broadcasted values and coefficients.
        """
        x = value
        fx_n, deriv_n = self.bspline.dot_and_deriv_n(x, coef)
        boundary_values, boundary_derivs = self.bspline.dot_and_deriv_n(
            self._boundaries, coef
        )

        value_left = jnp.expand_dims(boundary_values[..., 0], -1)
        deriv_left = jnp.expand_dims(boundary_derivs[..., 0], -1)
        value_right = jnp.expand_dims(boundary_values[..., 1], -1)
        deriv_right = jnp.expand_dims(boundary_derivs[..., 1], -1)

        left_transition, left_transition_deriv = self._left_transition_and_deriv(
            x, value_left=value_left, deriv_left=deriv_left
        )
        right_transition, right_transition_deriv = self._right_transition_and_deriv(
            x, value_right=value_right, deriv_right=deriv_right
        )

        fx_left_start = self._left_transition_and_deriv(
            self.min_eps, value_left=value_left, deriv_left=deriv_left
        )[0]
        fx_right_start = self._right_transition_and_deriv(
            self.max_eps, value_right=value_right, deriv_right=deriv_right
        )[0]

        left_tail, left_tail_deriv = self._left_tail_and_deriv(
            x, fx_at_linear_start=fx_left_start
        )
        right_tail, right_tail_deriv = self._right_tail_and_deriv(
            x, fx_at_linear_start=fx_right_start
        )

        in_core = (x >= self.min_knot) & (x <= self.max_knot)
        value = jnp.where(
            in_core,
            fx_n,
            jnp.where(
                x < self.min_eps,
                left_tail,
                jnp.where(
                    x < self.min_knot,
                    left_transition,
                    jnp.where(x < self.max_eps, right_transition, right_tail),
                ),
            ),
        )
        deriv = jnp.where(
            in_core,
            deriv_n,
            jnp.where(
                x < self.min_eps,
                left_tail_deriv,
                jnp.where(
                    x < self.min_knot,
                    left_transition_deriv,
                    jnp.where(
                        x < self.max_eps, right_transition_deriv, right_tail_deriv
                    ),
                ),
            ),
        )
        return value, deriv

    def _evaluate_spline_value(self, value: Array, coef: Array) -> Array:
        """
        Compute dot product for broadcasted values and coefficients.
        """
        x = value
        fx_n = self.bspline.dot_n(x, coef)
        boundary_values, boundary_derivs = self.bspline.dot_and_deriv_n(
            self._boundaries, coef
        )

        value_left = jnp.expand_dims(boundary_values[..., 0], -1)
        deriv_left = jnp.expand_dims(boundary_derivs[..., 0], -1)
        value_right = jnp.expand_dims(boundary_values[..., 1], -1)
        deriv_right = jnp.expand_dims(boundary_derivs[..., 1], -1)

        left_transition = self._left_transition_value(
            x, value_left=value_left, deriv_left=deriv_left
        )
        right_transition = self._right_transition_value(
            x, value_right=value_right, deriv_right=deriv_right
        )

        fx_left_start = self._left_transition_value(
            self.min_eps, value_left=value_left, deriv_left=deriv_left
        )
        fx_right_start = self._right_transition_value(
            self.max_eps, value_right=value_right, deriv_right=deriv_right
        )

        left_tail = self._left_tail_value(x, fx_at_linear_start=fx_left_start)
        right_tail = self._right_tail_value(x, fx_at_linear_start=fx_right_start)

        in_core = (x >= self.min_knot) & (x <= self.max_knot)
        return jnp.where(
            in_core,
            fx_n,
            jnp.where(
                x < self.min_eps,
                left_tail,
                jnp.where(
                    x < self.min_knot,
                    left_transition,
                    jnp.where(x < self.max_eps, right_transition, right_tail),
                ),
            ),
        )
