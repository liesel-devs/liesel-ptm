from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import Partial as partial

from .approx import bspline_basis, equidistant_knots
from .util import TransformationSpline


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


LogIncKnots = PTMKnots  # Alias


def sfn(exp_shape):
    """
    Compute normalization factor for PTM spline coefficients.

    Parameters
    ----------
    exp_shape
        Exponentiated shape parameters.

    Returns
    -------
    Normalization factor.
    """
    order = 3
    p = jnp.shape(exp_shape)[-1] + 1

    outer_border = exp_shape[..., jnp.array([0, -1])] / 6
    inner_border = 5 * exp_shape[..., jnp.array([1, -2])] / 6
    middle = exp_shape[..., 2:-2]
    summed_exp_shape = (
        outer_border.sum(axis=-1, keepdims=True)
        + inner_border.sum(axis=-1, keepdims=True)
        + middle.sum(axis=-1, keepdims=True)
    )
    return (1 / (p - order)) * summed_exp_shape


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


def cumsum_leading_zero(exp_shape: Array) -> Array:
    """
    Cumulative sum with a leading zero.
    """
    zeros_shape = jnp.shape(exp_shape)[:-1] + (1,)
    exp_shape = jnp.concatenate((jnp.zeros(zeros_shape), exp_shape), axis=-1)
    return jnp.cumsum(exp_shape, axis=-1)


def normalization_coef(shape: Array, dknots: Array) -> Array:
    """
    Construct spline coefficients with average slope one over the domain.

    Parameters
    ----------
    shape
        Shape parameters.
    dknots
        Knot differences.

    Returns
    -------
    Spline coefficients.
    """
    exp_shape = jnp.exp(shape)
    cumsum_exp_shape = cumsum_leading_zero(exp_shape)
    coef = (dknots / sfn(exp_shape)) * cumsum_exp_shape
    return coef


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


class PTMCoef:
    def __init__(self, knots: Array) -> None:
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

        self.step = jnp.diff(knots).mean()

    def _add_intercept_and_exponentiate(self, log_increments, intercept, log_slope):
        """
        Add intercept and exponentiate log increments.
        """
        exp_coef = jnp.exp(log_increments)
        prelim_coef = jnp.concatenate((jnp.zeros(1), exp_coef), axis=-1)
        offset = (self.B @ prelim_coef) - self.k1

        full_coef = jnp.concatenate((-offset + intercept, exp_coef), axis=-1)

        fx_at_zero = self.B0 @ full_coef

        coef2 = jnp.exp(log_increments + log_slope)
        full_coef2 = jnp.concatenate((-offset + intercept, coef2), axis=-1)

        fx_at_zero2 = self.B0 @ full_coef2

        diff_at_zero = (fx_at_zero2 - fx_at_zero).squeeze()
        coef3 = full_coef2.at[..., 0].set(full_coef2[..., 0] - diff_at_zero)

        return coef3

    def get_ptm_fn(self) -> Callable[[Array, Array, Array], Array]:
        """
        Get function to compute PTM spline coefficients.
        """
        nparam = len(self.knots) - 4 - 1
        zeros = jnp.zeros((nparam,))

        add_intercept_and_exponentiate = jax.vmap(self._add_intercept_and_exponentiate)

        def compute_coef(log_increments, intercept, log_slope):
            log_increments = jnp.atleast_2d(log_increments + zeros)
            intercept = jnp.expand_dims(jnp.atleast_1d(intercept), -1)
            log_slope = jnp.expand_dims(jnp.atleast_1d(log_slope), -1)

            log_increments_slope_one = jax.vmap(normalization_coef_log, (0, None))(
                log_increments, self.step
            )
            full_coef = add_intercept_and_exponentiate(
                log_increments_slope_one, intercept, log_slope
            )
            return full_coef

        return compute_coef

    def get_ptm_fn_squeeze(self) -> Callable[[Array, Array, Array], Array]:
        """
        Get function to compute PTM spline coefficients and squeeze output.
        """
        fn = self.get_ptm_fn()

        def compute_coef(log_increments, intercept, log_slope):
            return fn(log_increments, intercept, log_slope).squeeze(-2)

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
    """

    def __init__(
        self,
        knots: Array,
        eps: float = 0.1,
        continue_linearly: bool = False,
    ) -> None:
        if eps < 1e-6:
            raise ValueError(f"{eps=} is < 1e-6; that is numerically unstable.")

        super().__init__(knots)

        self._compute_coef = jax.jit(
            partial(PTMCoef(knots).get_ptm_fn_squeeze(), intercept=0.0, log_slope=0.0)
        )  # type: ignore

        if continue_linearly:
            eps = 100000.0

        self.transition_width = eps * (self.max_knot - self.min_knot)
        self.min_eps = self.min_knot - self.transition_width
        self.max_eps = self.max_knot + self.transition_width

        def target_slope(coef):
            return 1.0

        self._target_slope_left = target_slope
        self._target_slope_right = target_slope

        self._boundaries = jnp.array([self.min_knot, self.max_knot])

        # if continue_linearly:

        #     def slope_at_min_knot(coef):
        #         return self.bspline.dot_and_deriv_n(self.min_knot, coef)[0]

        #     def slope_at_max_knot(coef):
        #         return self.bspline.dot_and_deriv_n(self.max_knot, coef)[0]

        #     self._target_slope_left = slope_at_min_knot
        #     self._target_slope_right = slope_at_max_knot

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
