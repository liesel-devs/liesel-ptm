from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum, auto
from functools import partial

import jax
import jax.numpy as jnp

from .custom_types import Array
from .liesel_internal import splines

kn = splines.create_equidistant_knots

logger = logging.getLogger(__name__)


class Knots:
    def __init__(
        self,
        a: float,
        b: float,
        nparam: int,
        order: int = 3,
        nfixed_left: int = 0,
        nfixed_right: int = 0,
    ) -> None:
        self.a = a
        self.b = b
        self.nparam = nparam
        self.order = order
        self.nfixed_left = nfixed_left
        self.nfixed_right = nfixed_right

        knots = splines.create_equidistant_knots(
            jnp.array([a, b]), order=order, n_params=nparam
        )
        self.step = jnp.diff(knots).mean()

        leftmost_knot = knots[0]
        rightmost_knot = knots[-1]

        left_additional_knots = jnp.linspace(
            leftmost_knot - ((nfixed_left) * self.step),
            (leftmost_knot - self.step),
            nfixed_left,
        )

        right_additional_knots = jnp.linspace(
            (rightmost_knot + self.step),
            rightmost_knot + ((nfixed_right) * self.step),
            nfixed_right,
        )

        knots = jnp.concatenate((left_additional_knots, knots, right_additional_knots))

        intercept_knot = jnp.atleast_1d(knots[0] - self.step)

        self.knots = jnp.concatenate((intercept_knot, knots))
        self.nparam_full_domain = self.nparam + self.nfixed_left + self.nfixed_right + 1

    def _average_slope(self, coef: Array) -> Array:
        dcoef = jnp.log(jnp.diff(coef))
        return self._average_slope_latent(dcoef)

    def _average_slope_latent(self, latent_coef: Array) -> Array:
        p = jnp.shape(latent_coef)[-1] + 1
        dcoef = jnp.exp(latent_coef)

        outer_border = dcoef[..., jnp.array([0, -1])] / 6
        inner_border = 5 * dcoef[..., jnp.array([1, -2])] / 6
        middle = dcoef[..., 2:-2]

        numerator = (
            outer_border.sum(axis=-1, keepdims=True)
            + inner_border.sum(axis=-1, keepdims=True)
            + middle.sum(axis=-1, keepdims=True)
        )

        return numerator / (self.step * (p - self.order))

    def average_slope_in_ab_latent(self, latent_coef: Array) -> float:
        assert jnp.shape(latent_coef)[-1] == self.nparam
        return self._average_slope_latent(latent_coef)

    def average_slope_right_of_a_latent(self, latent_coef: Array) -> float:
        assert jnp.shape(latent_coef)[-1] == (self.nparam + self.nfixed_right)
        return self._average_slope_latent(latent_coef)

    def average_slope_in_full_domain_latent(self, latent_coef: Array) -> float:
        assert jnp.shape(latent_coef)[-1] == (
            self.nparam + self.nfixed_left + self.nfixed_right
        )
        return self._average_slope_latent(latent_coef)

    def average_slope_in_full_domain(self, coef: Array) -> float:
        assert jnp.shape(coef)[-1] == (
            self.nparam + self.nfixed_left + self.nfixed_right + 1
        )
        return self._average_slope(coef)


class IncreasingCoef:

    """
    Assuming that latent_coef[..., 0] is -Inf.
    """

    def __init__(self, knots: Knots):
        self.knots = knots
        self.target_log_increment = self.setup_target_log_increment()
        self.intercept_coef = self.setup_intercept_coef()
        self.target_coef = self.setup_target_coef()
        self.target_latent_coef = self.setup_target_latent_coef()
        self.init_params = self.setup_target_latent_coef_params()

        # assert self.knots.nfixed_left >= 1

    def setup_target_latent_coef(self) -> Array:
        latent_coef_b0 = jnp.array([-jnp.inf])
        latent_coef_left = jnp.full(
            (self.knots.nfixed_left,), fill_value=self.target_log_increment
        )
        latent_coef_params = jnp.full(
            (self.knots.nparam,), fill_value=self.target_log_increment
        )
        latent_coef_right = jnp.full(
            (self.knots.nfixed_right,), fill_value=self.target_log_increment
        )
        latent_coef = jnp.concatenate(
            (latent_coef_b0, latent_coef_left, latent_coef_params, latent_coef_right)
        )
        return latent_coef

    def setup_target_latent_coef_params(self) -> Array:
        latent_coef_params = jnp.full(
            (self.knots.nparam,), fill_value=self.target_log_increment
        )
        return latent_coef_params

    @partial(jnp.vectorize, excluded=[0], signature="(p)->(k)")
    def assemble_latent_coef(self, latent_params: Array) -> Array:
        il = self.knots.nfixed_left + 1
        ir = il + self.knots.nparam
        latent_coef = self.target_latent_coef.at[il:ir].set(latent_params)
        return latent_coef

    def normalize_in_ab(self, latent_coef: Array) -> Array:
        il = self.knots.nfixed_left + 1
        ir = il + self.knots.nparam

        latent_coef_in_ab = latent_coef[..., il:ir]

        avg_slope = self.knots.average_slope_in_ab_latent(latent_coef_in_ab)

        latent_coef = latent_coef.at[..., il:ir].set(
            latent_coef_in_ab - jnp.log(avg_slope)
        )
        return latent_coef

    def normalize_in_full_domain(self, latent_coef: Array) -> Array:
        avg_slope = self.knots.average_slope_in_full_domain_latent(latent_coef[..., 1:])
        return latent_coef - jnp.log(avg_slope)

    def normalize_right_of_a(self, latent_coef: Array) -> Array:
        latent_coef_right_of_a = latent_coef[..., (self.knots.nfixed_left + 1) :]
        avg_slope = self.knots.average_slope_right_of_a_latent(latent_coef_right_of_a)

        latent_coef = latent_coef.at[..., (self.knots.nfixed_left + 1) :].set(
            latent_coef_right_of_a - jnp.log(avg_slope)
        )
        return latent_coef

    def setup_target_coef(self):
        latent_coef = jnp.full(
            (self.knots.nparam_full_domain - 1,), fill_value=self.target_log_increment
        )
        coef = jnp.concatenate(
            (jnp.atleast_1d(self.intercept_coef), jnp.exp(latent_coef))
        ).cumsum()
        return coef

    def setup_intercept_coef(self):
        latent_coef = jnp.full(
            (self.knots.nparam_full_domain - 1,), fill_value=self.target_log_increment
        )
        coef = jnp.concatenate((jnp.zeros(1), jnp.exp(latent_coef))).cumsum()
        basis_matrix = bspline_basis(jnp.zeros(1), self.knots.knots, self.knots.order)
        return -jnp.squeeze(basis_matrix @ coef)

    def setup_target_log_increment(self):
        latent_coef = jnp.zeros((self.knots.nparam_full_domain - 1,))
        coef = jnp.concatenate((jnp.zeros(1), jnp.exp(latent_coef))).cumsum()
        avg_slope = self.knots.average_slope_in_full_domain(coef)

        return (latent_coef - jnp.log(avg_slope)).mean()


class EnforceSlope1(StrEnum):
    NO = auto()
    IN_AB = auto()
    RIGHT_OF_A = auto()
    GLOBALLY = auto()


class CoefWeights:
    def __init__(self, knots: Knots):
        self.knots = knots
        self._base_weights = jnp.ones(knots.nparam_full_domain)

    def sigmoid(self, nleft, nright, nmid=0, k=10):
        if nleft > 0:
            xleft = jnp.linspace(0, 1, nleft)
            left = jnp.flip(1 / (1 + jnp.exp(-k * (xleft - 0.5))))
        else:
            left = jnp.array([])

        if nright > 0:
            xright = jnp.linspace(0, 1, nright)
            right = 1 / (1 + jnp.exp(-k * (xright - 0.5)))
        else:
            right = jnp.array([])

        mid = jnp.zeros(nmid)
        weights = jnp.concatenate((left, mid, right))

        il = self.knots.nfixed_left
        ir = il + len(weights)
        weights = self._base_weights.at[il:ir].set(weights)

        return weights


class OnionCoef(IncreasingCoef):
    def __init__(
        self,
        knots: Knots,
        enforce_slope1: EnforceSlope1 = EnforceSlope1.IN_AB,
        min_increment: float = 1e-4,
    ):
        super().__init__(knots=knots)
        self.enforce_slope1 = enforce_slope1
        self.min_increment = min_increment

    def compute_coef(self, latent_params: Array, weights: Array) -> Array:
        latent_coef = self.assemble_latent_coef(latent_params)

        latent_coef = self.normalize_right_of_a(latent_coef)

        def body_fun(j, val):
            latent_coef, weights = val

            coef = jnp.exp(latent_coef).cumsum(axis=-1) + self.intercept_coef
            target = self.target_coef[j] + jnp.exp(self.target_log_increment)
            diff_j = target - coef[..., j]

            latent_target_j = jnp.log(
                (diff_j > 0) * diff_j + (1 - (diff_j > 0)) * self.min_increment
            )

            updated_latent_coef_j = (
                weights[j] * latent_target_j
                + (1 - weights[j]) * latent_coef[..., (j + 1)]
            )
            latent_coef = latent_coef.at[..., (j + 1)].set(updated_latent_coef_j)
            return latent_coef, weights

        latent_coef, _ = jax.lax.fori_loop(
            lower=1,
            upper=self.knots.nparam_full_domain,
            body_fun=body_fun,
            init_val=(latent_coef, weights),
        )

        if self.enforce_slope1 == EnforceSlope1.IN_AB:
            latent_coef = self.normalize_in_ab(latent_coef)
            latent_coef = latent_coef.at[..., -(self.knots.nfixed_right + 1) :].set(
                self.target_log_increment
            )

            latent_coef, _ = jax.lax.fori_loop(
                lower=self.knots.nfixed_left + self.knots.nparam - 1,
                upper=self.knots.nparam_full_domain,
                body_fun=body_fun,
                init_val=(latent_coef, weights),
            )
        elif self.enforce_slope1 == EnforceSlope1.RIGHT_OF_A:
            latent_coef = self.normalize_right_of_a(latent_coef)
        elif self.enforce_slope1 == EnforceSlope1.GLOBALLY:
            latent_coef = self.normalize_in_full_domain(latent_coef)
        elif self.enforce_slope1 == EnforceSlope1.NO:
            pass

        coef = jnp.exp(latent_coef).cumsum(axis=-1) + self.intercept_coef

        return coef


class StreamCoef(IncreasingCoef):
    def __init__(
        self, knots: Knots, enforce_slope1: EnforceSlope1 = EnforceSlope1.IN_AB
    ):
        super().__init__(knots=knots)
        self.enforce_slope1 = enforce_slope1

    def compute_coef(self, latent_params: Array, weights: Array) -> Array:
        latent_coef = self.assemble_latent_coef(latent_params)

        latent_coef = self.normalize_right_of_a(latent_coef)

        base = weights * self.target_log_increment
        deviation = (1.0 - weights) * latent_coef

        latent_coef = latent_coef.at[..., (1 + self.knots.nfixed_left) :].set(
            (base + deviation)[..., (1 + self.knots.nfixed_left) :]
        )

        if self.enforce_slope1 == EnforceSlope1.IN_AB:
            latent_coef = self.normalize_in_ab(latent_coef)
        elif self.enforce_slope1 == EnforceSlope1.RIGHT_OF_A:
            latent_coef = self.normalize_right_of_a(latent_coef)
        elif self.enforce_slope1 == EnforceSlope1.GLOBALLY:
            latent_coef = self.normalize_in_full_domain(latent_coef)
        elif self.enforce_slope1 == EnforceSlope1.NO:
            pass

        coef = jnp.exp(latent_coef).cumsum(axis=-1) + self.intercept_coef

        return coef


@partial(jax.jit, static_argnums=2)
@partial(jnp.vectorize, excluded=(1, 2), signature="(n)->(n,p)")
def bspline_basis(x, knots, order):
    """
    Vectorized B-spline basis function evaluation.

    Parameters
    ----------
    x
        Input array.
    knots
        Array of knots.
    order
        Order of the spline (``order=3`` for a cubic spline).

    Returns
    -------
    B-spline basis matrix.
    """
    min_knot = knots[order]
    max_knot = knots[-(order + 1)]
    basis = splines.build_design_matrix_b_spline(x, knots, order)
    mask = jnp.logical_or(x < min_knot, x > max_knot)
    mask = jnp.expand_dims(mask, -1)
    return jnp.where(mask, 0.0, basis)


@partial(jax.jit, static_argnums=2)
@partial(jnp.vectorize, excluded=(1, 2), signature="(n)->(n,p)")
def bspline_basis_deriv(x, knots, order):
    min_knot = knots[order]
    max_knot = knots[-(order + 1)]

    basis = splines.build_design_matrix_b_spline(x, knots[1:-1], order - 1)
    dknots = jnp.diff(knots).mean()
    D = jnp.diff(jnp.identity(jnp.shape(knots)[-1] - order - 1)).T
    basis_grad = basis @ (D / dknots)

    mask = jnp.logical_or(x < min_knot, x > max_knot)
    mask = jnp.expand_dims(mask, -1)
    return jnp.where(mask, 0.0, basis_grad)


@partial(jax.jit, static_argnums=2)
@partial(jnp.vectorize, excluded=(1, 2), signature="(n)->(n,p)")
def bspline_basis_deriv2(x, knots, order):
    min_knot = knots[order]
    max_knot = knots[-(order + 1)]
    basis = splines.build_design_matrix_b_spline(x, knots[2:-2], order - 2)

    dknots = jnp.diff(knots).mean()
    D = jnp.diff(jnp.identity(jnp.shape(knots)[-1] - order - 1)).T
    basis_grad = basis @ D[1::, 1:] @ (D / (dknots**2))

    mask = jnp.logical_or(x < min_knot, x > max_knot)
    mask = jnp.expand_dims(mask, -1)
    return jnp.where(mask, 0.0, basis_grad)


class BSplineApprox:
    def __init__(self, knots: Array, order: Array, ngrid: int = 1000) -> None:
        self.knots = knots
        self.dknots = jnp.mean(jnp.diff(knots))
        self.order = order

        self.min_knot = self.knots[order]
        self.max_knot = self.knots[-(order + 1)]

        grid = jnp.linspace(self.min_knot, self.max_knot, ngrid)
        self.step = (self.max_knot - self.min_knot) / ngrid
        prepend = jnp.array([self.min_knot - self.step])
        append = jnp.array([self.max_knot + self.step])
        self.grid = jnp.concatenate((prepend, grid, append))

        self.basis = bspline_basis(self.grid, self.knots, self.order)
        self.basis_deriv = bspline_basis_deriv(self.grid, self.knots, self.order)
        self.basis_deriv2 = bspline_basis_deriv2(self.grid, self.knots, self.order)

    @partial(jax.jit, static_argnums=0)
    @partial(jnp.vectorize, excluded=[0], signature="(n)->(n,p)")
    def get_basis(self, x: Array) -> Array:
        i = jnp.searchsorted(self.grid, x, side="right") - 1
        lo = self.grid[i]
        k = jnp.expand_dims((x - lo) / self.step, -1)
        basis = (1.0 - k) * self.basis[i, :] + (k * self.basis[i + 1, :])
        return basis

    @partial(jax.jit, static_argnums=0)
    @partial(jnp.vectorize, excluded=[0], signature="(n)->(n,p),(n,p)")
    def get_basis_and_deriv(self, x: Array) -> tuple[Array, Array]:
        """
        Returns the basis matrix approximation and its gradient with
        respect to the data.
        """
        i = jnp.searchsorted(self.grid, x, side="right") - 1
        lo = self.grid[i]
        k = jnp.expand_dims((x - lo) / self.step, -1)
        basis = (1.0 - k) * self.basis[i, :] + (k * self.basis[i + 1, :])
        basis_deriv = (1.0 - k) * self.basis_deriv[i, :] + (
            k * self.basis_deriv[i + 1, :]
        )
        return basis, basis_deriv

    @partial(jax.jit, static_argnums=0)
    @partial(jnp.vectorize, excluded=[0], signature="(n)->(n,p),(n,p),(n,p)")
    def get_basis_deriv_and_deriv2(self, x: Array) -> tuple[Array, Array, Array]:
        """
        Returns the basis matrix approximation and its first and second
        derivative with respect to the data.
        """
        i = jnp.searchsorted(self.grid, x, side="right") - 1
        lo = self.grid[i]
        k = jnp.expand_dims((x - lo) / self.step, -1)
        basis = (1.0 - k) * self.basis[i, :] + (k * self.basis[i + 1, :])
        basis_deriv = (1.0 - k) * self.basis_deriv[i, :] + (
            k * self.basis_deriv[i + 1, :]
        )
        basis_deriv2 = (1.0 - k) * self.basis_deriv2[i, :] + (
            k * self.basis_deriv2[i + 1, :]
        )
        return basis, basis_deriv, basis_deriv2

    def get_basis_dot_fn(self) -> Callable[[Array, Array], Array]:
        @jax.custom_jvp
        def _basis_dot(
            x: Array,
            coef: Array,
        ) -> Array:
            x = jnp.atleast_1d(x)
            basis = self.get_basis(x)
            smooth = jnp.einsum("...ip,...p->...i", basis, coef)
            return smooth

        @_basis_dot.defjvp
        def _basis_dot_jvp(primals, tangents):
            x, coef = primals
            x_dot, coef_dot = tangents

            basis, basis_deriv = self.get_basis_and_deriv(x)
            smooth = jnp.einsum("...ip,...p->...i", basis, coef)

            tangent_x = jnp.einsum("...ip,...p->...i", basis_deriv, coef) * x_dot
            tangent_coef = jnp.einsum("...ip,...p->...i", basis, coef_dot)

            tangent = tangent_x + tangent_coef

            return smooth, tangent

        return jax.jit(_basis_dot)

    def get_basis_dot_and_deriv_fn(
        self,
    ) -> Callable[[Array, Array], tuple[Array, Array]]:
        @jax.custom_jvp
        def _basis_dot_and_deriv(
            x: Array,
            coef: Array,
        ) -> tuple[Array, Array]:
            x = jnp.atleast_1d(x)
            basis, basis_deriv = self.get_basis_and_deriv(x)
            smooth = jnp.einsum("...ip,...p->...i", basis, coef)
            smooth_deriv = jnp.einsum("...ip,...p->...i", basis_deriv, coef)
            return smooth, smooth_deriv

        @_basis_dot_and_deriv.defjvp
        def _basis_dot_and_deriv_jvp(primals, tangents):
            x, coef = primals
            x = jnp.atleast_1d(x)
            x_dot, coef_dot = tangents

            basis, basis_deriv, basis_deriv2 = self.get_basis_deriv_and_deriv2(x)
            smooth = jnp.einsum("...ip,...p->...i", basis, coef)
            smooth_deriv = jnp.einsum("...ip,...p->...i", basis_deriv, coef)
            smooth_deriv2 = jnp.einsum("...ip,...p->...i", basis_deriv2, coef)

            primal_out = (smooth, smooth_deriv)

            tangent_bdot_x = jnp.einsum("...ip,...p->...i", basis_deriv, coef) * x_dot
            tangent_bdot_coef = jnp.einsum("...ip,...p->...i", basis, coef_dot)
            tangent_bdot = tangent_bdot_x + tangent_bdot_coef

            tangent_deriv_x = smooth_deriv2 * x_dot
            tangent_deriv_coef = jnp.einsum("...ip,...p->...i", basis_deriv, coef_dot)
            tangent_deriv = tangent_deriv_x + tangent_deriv_coef

            tangent_out = (tangent_bdot, tangent_deriv)

            return primal_out, tangent_out

        return jax.jit(_basis_dot_and_deriv)


class ExtrapBSplineApprox(BSplineApprox):
    """
    Extrapolating B-Spline approximation.

    Params
    ------
    knots
        The knots of the B-Spline. Assumed to be equidistant.
    order
        The order of the B-Spline. A cubic B-Spline is given by ``order=3``.
    ngrid
        The number of grid points used for the approximation.
    eps
        Controls the width of the transition from the B-Spline to linear \
        extrapolation. This is a factor applied to the range of the knots. \
        The default of ``0.1`` means that the transition interval width is \
        ``0.1 * (max_knot - min_knot)``.
    """

    def __init__(
        self, knots: Array, order: Array, ngrid: int = 1000, eps: float = 0.3
    ) -> None:
        super().__init__(knots, order, ngrid)
        self.basis_min, self.basis_grad_min = self.get_basis_and_deriv(
            jnp.atleast_1d(self.knots[order])
        )
        self.basis_max, self.basis_grad_max = self.get_basis_and_deriv(
            jnp.atleast_1d(self.knots[-(order + 1)])
        )

        self.eps = eps * (self.max_knot - self.min_knot)
        self.min_eps = self.min_knot - self.eps
        self.max_eps = self.max_knot + self.eps

    def get_extrap_basis_dot_fn(
        self, target_slope: float | None = None
    ) -> Callable[[Array, Array], Array]:
        basis_dot_and_deriv_fn = self.get_basis_dot_and_deriv_fn()
        basis_dot_fn = self.get_basis_dot_fn()

        target_slope_fn = (
            (lambda knots, coef, order: target_slope)
            if target_slope is not None
            else avg_slope_bspline
        )

        @partial(jnp.vectorize, signature="(n),(p)->(n)")
        def basis_dot(x: Array, coef: Array) -> Array:
            basis_dot_left, deriv_left = basis_dot_and_deriv_fn(self.min_knot, coef)
            basis_dot_right, deriv_right = basis_dot_and_deriv_fn(self.max_knot, coef)

            target_slope = target_slope_fn(self.knots, coef, self.order)

            # -----------------------------------------------------------------
            # Helper functions
            # -----------------------------------------------------------------

            def _extrap_left_transition(self, x: Array) -> Array:
                def _unshifted_extrap(x: Array) -> Array:
                    polynomial = x * self.min_knot - (x**2) / 2
                    term1 = (target_slope / self.eps) * polynomial
                    term2 = jnp.squeeze(deriv_left) * (x - polynomial / self.eps)
                    return term1 + term2

                const = jnp.squeeze(basis_dot_left) - jnp.squeeze(
                    _unshifted_extrap(self.min_knot)
                )

                return _unshifted_extrap(x) + const

            def _extrap_right_transition(self, x: Array) -> Array:
                def _unshifted_extrap(x: Array) -> Array:
                    polynomial = (x**2) / 2 - x * self.max_knot
                    term1 = (target_slope / self.eps) * polynomial
                    term2 = jnp.squeeze(deriv_right) * (x - polynomial / self.eps)
                    return term1 + term2

                const = jnp.squeeze(basis_dot_right) - jnp.squeeze(
                    _unshifted_extrap(self.max_knot)
                )

                return _unshifted_extrap(x) + const

            # -----------------------------------------------------------------
            # Function segments
            # -----------------------------------------------------------------

            outl = x < self.min_eps
            outr = x > self.max_eps
            transitl = (self.min_eps <= x) & (x < self.min_knot)
            transitr = (self.max_knot < x) & (x <= self.max_eps)
            center = (self.min_knot <= x) & (x <= self.max_knot)

            # -----------------------------------------------------------------
            # Core spline
            # -----------------------------------------------------------------

            value_center = center * basis_dot_fn(x, coef)

            # -----------------------------------------------------------------
            # Main values
            # -----------------------------------------------------------------

            # start points of linear extrapolation
            linl_start = _extrap_left_transition(self, self.min_eps)
            linr_start = _extrap_right_transition(self, self.max_eps)

            # linear transition
            val_linl = outl * (linl_start - target_slope * (self.min_eps - x))
            val_linr = outr * (linr_start + target_slope * (x - self.max_eps))

            # transition
            val_transitl = transitl * _extrap_left_transition(self, x)
            val_transitr = transitr * _extrap_right_transition(self, x)

            # put everything together
            value = val_linl + val_transitl + value_center + val_transitr + val_linr

            return value

        # Return jitted function
        return jax.jit(basis_dot)

    def get_extrap_basis_dot_and_deriv_fn(
        self, target_slope: float | None = None
    ) -> Callable[[Array, Array], tuple[Array, Array]]:
        basis_dot_and_deriv_fn = self.get_basis_dot_and_deriv_fn()
        minmax = jnp.array([self.min_knot, self.max_knot])

        target_slope_fn = (
            (lambda knots, coef, order: target_slope)
            if target_slope is not None
            else avg_slope_bspline
        )

        @partial(jnp.vectorize, signature="(n),(p)->(n),(n)")
        def basis_dot_and_deriv(x: Array, coef: Array) -> tuple[Array, Array]:
            basis_dot_left, deriv_left = basis_dot_and_deriv_fn(self.min_knot, coef)
            basis_dot_right, deriv_right = basis_dot_and_deriv_fn(self.max_knot, coef)

            target_slope = target_slope_fn(self.knots, coef, self.order)

            # -----------------------------------------------------------------
            # Helper functions
            # -----------------------------------------------------------------

            def _extrap_left_transition(self, x: Array) -> Array:
                def _unshifted_extrap(x: Array) -> Array:
                    polynomial = x * self.min_knot - (x**2) / 2
                    term1 = (target_slope / self.eps) * polynomial
                    term2 = jnp.squeeze(deriv_left) * (x - polynomial / self.eps)
                    return term1 + term2

                const = jnp.squeeze(basis_dot_left) - jnp.squeeze(
                    _unshifted_extrap(self.min_knot)
                )

                return _unshifted_extrap(x) + const

            def _extrap_right_transition(self, x: Array) -> Array:
                def _unshifted_extrap(x: Array) -> Array:
                    polynomial = (x**2) / 2 - x * self.max_knot
                    term1 = (target_slope / self.eps) * polynomial
                    term2 = jnp.squeeze(deriv_right) * (x - polynomial / self.eps)
                    return term1 + term2

                const = jnp.squeeze(basis_dot_right) - jnp.squeeze(
                    _unshifted_extrap(self.max_knot)
                )

                return _unshifted_extrap(x) + const

            # -----------------------------------------------------------------
            # Function segments
            # -----------------------------------------------------------------

            outl = x < self.min_eps
            outr = x > self.max_eps
            transitl = (self.min_eps <= x) & (x < self.min_knot)
            transitr = (self.max_knot < x) & (x <= self.max_eps)
            center = (self.min_knot <= x) & (x <= self.max_knot)

            # -----------------------------------------------------------------
            # Core spline and derivative
            # -----------------------------------------------------------------

            value_center, deriv_center = basis_dot_and_deriv_fn(x, coef)
            value_center = center * value_center
            deriv_center = center * deriv_center

            # -----------------------------------------------------------------
            # Main values
            # -----------------------------------------------------------------

            # start points of linear extrapolation
            linl_start = _extrap_left_transition(self, self.min_eps)
            linr_start = _extrap_right_transition(self, self.max_eps)

            # linear transition
            val_linl = outl * (linl_start - target_slope * (self.min_eps - x))
            val_linr = outr * (linr_start + target_slope * (x - self.max_eps))

            # transition
            val_transitl = transitl * _extrap_left_transition(self, x)
            val_transitr = transitr * _extrap_right_transition(self, x)

            # put everything together
            value = val_linl + val_transitl + value_center + val_transitr + val_linr

            # -----------------------------------------------------------------
            # Derivative
            # -----------------------------------------------------------------

            # parts for evaluating transition derivatives
            _, (deriv_at_min, deriv_at_max) = basis_dot_and_deriv_fn(minmax, coef)
            distl = (self.min_knot - x) / self.eps
            distr = (x - self.max_knot) / self.eps

            # transition derivatives
            derivl = transitl * ((1.0 - distl) * deriv_at_min + target_slope * distl)
            derivr = transitr * ((1.0 - distr) * deriv_at_max + target_slope * distr)

            # put everything together
            deriv = deriv_center + (outl + outr) * target_slope + derivl + derivr

            return value, deriv

        # Return jitted function
        return jax.jit(basis_dot_and_deriv)


def avg_slope_bspline(knots: Array, coef: Array, order: int):
    dk = jnp.diff(knots).mean()
    p = jnp.shape(coef)[-1]
    coef = jnp.diff(coef)

    outer_border = coef[..., jnp.array([0, -1])] / 6
    inner_border = 5 * coef[..., jnp.array([1, -2])] / 6
    middle = coef[..., 2:-2]
    summed_coef = (
        outer_border.sum(axis=-1, keepdims=True)
        + inner_border.sum(axis=-1, keepdims=True)
        + middle.sum(axis=-1, keepdims=True)
    )

    return summed_coef / (dk * (p - order))


def _extrapolate_bspline_linearly_left(
    x,
    smooth: Array,
    coef: Array,
    knots: Array,
    slope: float | Array = 1.0,
    order: int = 3,
    reparam_matrix: Array | None = None,
):
    """
    Can handle batched x and coef. The batch dimensions must be leading dimensions.
    Batching of x and smooth must be the same.
    """
    reparam_matrix = (
        reparam_matrix if reparam_matrix is not None else jnp.eye(coef.shape[-1])
    )
    min_knot = knots[order]
    min_output = _basis_dot(
        min_knot, knots, coef  # , order=order, reparam_matrix=reparam_matrix
    )
    # min_basis = bspline_basis(min_knot, knots, order)
    # min_basis = jnp.einsum("...ij,...jp->...ip", min_basis, reparam_matrix)
    # min_output = jnp.einsum("...ij,...j->...i", min_basis, coef)

    diffmin = min_knot - x
    linear_min = min_output - slope * diffmin
    smaller_than_min = x < min_knot
    smooth = jnp.where(smaller_than_min, linear_min, smooth)

    return smooth


def _extrapolate_bspline_linearly_right(
    x,
    smooth: Array,
    coef: Array,
    knots: Array,
    slope: float | Array = 1.0,
    order: int = 3,
    reparam_matrix: Array | None = None,
):
    """
    Can handle batched x and coef. The batch dimensions must be leading dimensions.
    Batching of x and smooth must be the same.
    """
    reparam_matrix = (
        reparam_matrix if reparam_matrix is not None else jnp.eye(coef.shape[-1])
    )

    max_knot = jnp.atleast_1d(knots[-(order + 1)])
    max_basis = bspline_basis(max_knot, knots, order)
    max_basis = jnp.einsum("...ij,...jp->...ip", max_basis, reparam_matrix)
    max_output = jnp.einsum("...ij,...j->...i", max_basis, coef)

    diffmax = x - max_knot
    linear_max = max_output + slope * diffmax
    larger_than_max = x > max_knot
    smooth = jnp.where(larger_than_max, linear_max, smooth)

    return smooth


def _extrapolate_bspline_linearly(
    x: Array,
    smooth: Array,
    coef: Array,
    knots: Array,
    slope: float = 1.0,
    order: int = 3,
    reparam_matrix: Array | None = None,
) -> Array:
    smooth = _extrapolate_bspline_linearly_left(
        x, smooth, coef, knots, slope=slope, order=order, reparam_matrix=reparam_matrix
    )
    smooth = _extrapolate_bspline_linearly_right(
        x, smooth, coef, knots, slope=slope, order=order, reparam_matrix=reparam_matrix
    )
    return smooth


def _extrapolate_bspline_grad_constant_left(
    x: Array, smooth: Array, knots: Array, const: float, order: int = 3
):
    min_knot = knots[order]
    smaller_than_min = x < min_knot
    smooth = jnp.where(smaller_than_min, const, smooth)

    return smooth


def _extrapolate_bspline_grad_constant_right(
    x: Array, smooth: Array, knots: Array, const: float, order: int = 3
):
    max_knot = knots[-(order + 1)]
    larger_than_max = x > max_knot
    smooth = jnp.where(larger_than_max, const, smooth)

    return smooth


def _extrapolating_basis_dot_fixed_slope(
    x: Array,
    knots: Array,
    coef: Array,
    order: int = 3,
    slope: float = 1.0,
    reparam_matrix: Array | None = None,
):
    """
    Can handle batched x and coef. The batch dimensions must be leading dimensions.
    """
    # smooth = _basis_dot(x, knots, coef, order=order, reparam_matrix=reparam_matrix)
    smooth = _basis_dot(x, knots, coef)
    smooth = _extrapolate_bspline_linearly(
        x, smooth, coef, knots, slope=slope, order=3, reparam_matrix=reparam_matrix
    )

    return smooth


def _average_slope_in_segment(coef: Array, knots: Array) -> Array:
    """
    Important! Assumes B-Spline of order 3!
    Also important! Assume to work on the coefs "g" of a full spline "Bg",
    where "B" is the basis matrix and "g" are the coefs.
    """
    dknots = jnp.diff(knots).mean()
    dcoef = jnp.diff(coef)
    weighted_coefs = dcoef[..., 0] / 6 + 2 * dcoef[..., 1] / 3 + dcoef[..., 2] / 6
    return jnp.expand_dims(weighted_coefs / dknots, -1)


def _average_slope_left(coef: Array, knots: Array) -> Array:
    return _average_slope_in_segment(coef[:4], knots)


def _average_slope_right(coef: Array, knots: Array) -> Array:
    return _average_slope_in_segment(coef[-4:], knots)


_average_slope_left_jac = jax.jacobian(_average_slope_left)
_average_slope_right_jac = jax.jacobian(_average_slope_right)


def _extrapolating_dot_jac_coef(x: Array, knots: Array, coef: Array) -> Array:
    left = jnp.expand_dims(x < knots[3], -1)
    right = jnp.expand_dims(x > knots[-4], -1)
    inside = jnp.logical_not(jnp.logical_or(left, right))

    # jax.jacobian(_basis_dot, argnums=2)(jnp.atleast_1d(knots[3]), knots, coef)

    bl = bspline_basis(jnp.atleast_1d(knots[3]), knots, 3)
    sl = _average_slope_left_jac(coef, knots)
    left_jac = bl - sl * jnp.expand_dims((knots[3] - x), -1)

    br = bspline_basis(jnp.atleast_1d(knots[-4]), knots, 3)
    sr = _average_slope_right_jac(coef, knots)
    right_jac = br + sr * jnp.expand_dims((x - knots[-4]), -1)

    jac = jnp.where(
        inside,
        bspline_basis(x, knots, 3),
        jnp.where(
            left,
            left_jac,
            right_jac,
        ),
    )
    return jac


@jax.custom_jvp
def _extrapolating_basis_dot_continue_average_slope(
    x: Array,
    knots: Array,
    coef: Array,
):
    """
    Extrapolates the B-Spline with straight lines with the slopes set to the
    average slopes in the boundary segments.
    """
    # smooth = _basis_dot(x, knots, coef, order=order, reparam_matrix=reparam_matrix)
    smooth = _basis_dot(x, knots, coef)
    slope_left = _average_slope_left(coef, knots)
    slope_right = _average_slope_right(coef, knots)
    # slope_left = _average_slope_in_segment(coef[:4], knots)
    # slope_right = _average_slope_in_segment(coef[-4:], knots)

    smooth = _extrapolate_bspline_linearly_left(
        x, smooth, coef, knots, slope=slope_left, order=3, reparam_matrix=None
    )
    smooth = _extrapolate_bspline_linearly_right(
        x,
        smooth,
        coef,
        knots,
        slope=slope_right,
        order=3,
        reparam_matrix=None,
    )

    return smooth


@_extrapolating_basis_dot_continue_average_slope.defjvp
def _extrapolating_basis_dot_continue_average_slope_jvp(primals, tangents):
    x, knots, coef = primals
    x_dot, _, coef_dot = tangents

    x = jnp.atleast_1d(x)
    primal_out = _extrapolating_basis_dot_continue_average_slope(x, knots, coef)
    tangent_x = (
        _extrapolating_basis_dot_grad_continue_average_slope(x, knots, coef) * x_dot
    )
    tangent_coef = _extrapolating_dot_jac_coef(x, knots, coef) @ coef_dot

    tangent_out = tangent_x + tangent_coef

    return primal_out, tangent_out


def _extrapolating_basis_dot_continue_point_slope(
    x: Array,
    knots: Array,
    coef: Array,
    order: int = 3,
    reparam_matrix: Array | None = None,
):
    """
    Extrapolates the B-Spline with straight lines with the slopes at the
    boundary knots.
    """

    # smooth = _basis_dot(x, knots, coef, order=order, reparam_matrix=reparam_matrix)
    smooth = _basis_dot(x, knots, coef)
    min_basis = bspline_basis_deriv(jnp.atleast_1d(knots[order]), knots, order)
    max_basis = bspline_basis_deriv(jnp.atleast_1d(knots[-(order + 1)]), knots, order)
    slope_left = jnp.einsum("...ip,...p->...i", min_basis, coef)
    slope_right = jnp.einsum("...ip,...p->...i", max_basis, coef)

    smooth = _extrapolate_bspline_linearly_left(
        x, smooth, coef, knots, slope=slope_left, order=3, reparam_matrix=reparam_matrix
    )
    smooth = _extrapolate_bspline_linearly_right(
        x,
        smooth,
        coef,
        knots,
        slope=slope_right,
        order=3,
        reparam_matrix=reparam_matrix,
    )

    return smooth


@jax.custom_jvp
def _basis_dot(
    x: Array,
    knots: Array,
    coef: Array,
):
    x = jnp.atleast_1d(x)
    basis = bspline_basis(x, knots, 3)
    smooth = jnp.einsum("...ip,...p->...i", basis, coef)
    return smooth


@_basis_dot.defjvp
def _basis_dot_jvp(primals, tangents):
    x, knots, coef = primals
    x_dot, _, coef_dot = tangents

    x = jnp.atleast_1d(x)
    primal_out = _basis_dot(x, knots, coef)
    tangent_x = _basis_dot_grad(x, knots, coef) * x_dot
    tangent_coef = bspline_basis(x, knots, 3) @ coef_dot

    tangent_out = tangent_x + tangent_coef

    return primal_out, tangent_out


def pad0(original_array, num_zeros_begin, num_zeros_end):
    # Get the shape of the original array
    shape = original_array.shape

    # Construct the pad width based on the number of zeros to prepend at each end
    pad_width = [
        (0, 0) if i < len(shape) - 1 else (num_zeros_begin, num_zeros_end)
        for i in range(len(shape))
    ]

    # Pad the last axis with the specified number of zeros at each end
    zero_prepended_array = jnp.pad(original_array, tuple(pad_width), mode="constant")

    return zero_prepended_array


@jax.custom_jvp
def _basis_dot_grad(
    x: Array,
    knots: Array,
    coef: Array,
):
    x = jnp.atleast_1d(x)

    dcoef = jnp.diff(coef) / jnp.diff(knots).mean()
    basis = bspline_basis(x, knots, 2)

    mask = jnp.logical_or(x < knots[3], x > knots[-4])
    mask = jnp.expand_dims(mask, -1)
    basis = jnp.where(mask, 0.0, basis)

    smooth = jnp.einsum("...ip,...p->...i", basis, pad0(dcoef, 1, 1))
    return smooth


@_basis_dot_grad.defjvp
def _basis_dot_grad_jvp(primals, tangents):
    x, knots, coef = primals
    x_dot, _, coef_dot = tangents

    x = jnp.atleast_1d(x)
    primal_out = _basis_dot_grad(x, knots, coef)
    tangent_x = _basis_dot_grad2(x, knots, coef) * x_dot
    tangent_coef = bspline_basis_deriv(x, knots, 3) @ coef_dot

    tangent_out = tangent_x + tangent_coef

    return primal_out, tangent_out


def _basis_dot_grad2(
    x: Array,
    knots: Array,
    coef: Array,
):
    # TODO Übergang stimmt nicht!
    x = jnp.atleast_1d(x)
    dcoef = jnp.diff(coef) / jnp.diff(knots).mean()
    dcoef = jnp.diff(dcoef) / jnp.diff(knots).mean()

    basis = bspline_basis(x, knots, 1)
    mask = jnp.logical_or(x < knots[3], x > knots[-4])
    mask = jnp.expand_dims(mask, -1)
    basis = jnp.where(mask, 0.0, basis)
    smooth = jnp.einsum("...ip,...p->...i", basis, pad0(dcoef, 2, 2))
    return smooth
    # x = jnp.atleast_1d(x)

    # basis = bspline_basis_grad2(x, knots, 3)
    # smooth = jnp.einsum("...ip,...p->...i", basis, coef)
    # return smooth


def _extrapolating_basis_dot_grad_fixed_slope(
    x: Array,
    knots: Array,
    coef: Array,
    order: int = 3,
    slope_left: float | Array = 1.0,
    slope_right: float | Array = 1.0,
    reparam_matrix: Array | None = None,
):
    # smooth = _basis_dot_grad(x, knots, coef, order=order, reparam_matrix=reparam_matrix)
    smooth = _basis_dot_grad(x, knots, coef)
    smooth = _extrapolate_bspline_grad_constant_left(
        x, smooth, knots, slope_left, order=order
    )
    smooth = _extrapolate_bspline_grad_constant_right(
        x, smooth, knots, slope_right, order=order
    )
    return smooth


@jax.custom_jvp
def _extrapolating_basis_dot_grad_continue_average_slope(
    x: Array,
    knots: Array,
    coef: Array,
):
    slope_left = _average_slope_left(coef, knots)
    slope_right = _average_slope_right(coef, knots)
    smooth = _extrapolating_basis_dot_grad_fixed_slope(
        x,
        knots,
        coef,
        order=3,
        slope_left=slope_left,
        slope_right=slope_right,
        reparam_matrix=None,
    )
    return smooth


@_extrapolating_basis_dot_grad_continue_average_slope.defjvp
def _extrapolating_basis_dot_grad_continue_average_slope_jvp(primals, tangents):
    x, knots, coef = primals
    x_dot, _, coef_dot = tangents
    x = jnp.atleast_1d(x)

    primal_out = _extrapolating_basis_dot_grad_continue_average_slope(x, knots, coef)
    tangent_x = _basis_dot_grad2(x, knots, coef) * x_dot

    sl = _average_slope_left_jac(coef, knots)
    b = bspline_basis_deriv(x, knots, 3)
    sr = _average_slope_right_jac(coef, knots)

    left = jnp.expand_dims(x < knots[3], -1)
    right = jnp.expand_dims(x > knots[-4], -1)
    inside = jnp.logical_not(jnp.logical_or(left, right))

    tangent_coef = jnp.where(inside, b, jnp.where(left, sl, sr)) @ coef_dot

    tangent_out = tangent_x + tangent_coef

    return primal_out, tangent_out
