from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from .approx import equidistant_knots
from .util import TransformationSpline


class OnionKnots:
    def __init__(self, a: float, b: float, nparam: int, order: int = 3) -> None:
        """
        Initialize OnionKnots with interval, number of parameters, and spline order.

        Parameters
        ----------
        a : float
            Left boundary of the flexible interval.
        b : float
            Right boundary of the flexible interval.
        nparam : int
            Number of flexible parameters for the spline.
        order : int, optional
            Spline order (default is 3 for a cubic spline).

        Attributes
        ----------
        a
            Left boundary of the flexible interval.
        b
            Right boundary of the flexible interval.
        nparam
            Number of parameters for the spline.
        order
            Spline order.
        step
            Step size between knots.
        knots
            Array of spline knots.
        """
        self.a = a
        self.b = b
        self.nparam = nparam
        self.order = order

        m = nparam + 5

        self.step = (self.b - self.a) / (m - 3)

        k1 = a - self.step
        k2 = b + self.step

        self.knots = equidistant_knots(
            jnp.array([k1, k2]), order=order, n_param=nparam + 7, eps=0.0
        )


def get_onion_fn(knots) -> Callable[[Array], Array]:
    """
    Create a function to compute onion spline coefficients from log increments.

    Parameters
    ----------
    knots
        Spline knot sequence.

    Returns
    -------
    compute_coef
        Function that computes coefficients from log increments.
    """
    m = len(knots) - 6
    dk = jnp.diff(knots).mean()

    dk_array = jnp.full((3,), fill_value=dk)
    dk_array = jnp.log(dk_array)
    latent_correction_denominator = jnp.log((m - 5) * dk)

    nparam = len(knots) - 11
    zeros = jnp.zeros((nparam,))

    def onion_correction(latent_coef):
        """
        Apply onion correction to latent coefficients.
        """
        latent_correction_numerator = jax.nn.logsumexp(latent_coef)
        latent_correction = latent_correction_numerator - latent_correction_denominator

        latent_coef = latent_coef - latent_correction
        full_latent_coef = jnp.concatenate((dk_array, latent_coef, dk_array))
        return full_latent_coef

    def compute_coef(log_increments):
        """
        Compute full spline coefficients from log increments.
        """
        log_increments = log_increments + zeros

        log_increments_slope_one = onion_correction(log_increments)
        exp_coef = jnp.exp(log_increments_slope_one)

        full_coef = jnp.r_[(knots[2], exp_coef)]

        return full_coef

    return compute_coef


class OnionSpline(TransformationSpline):
    def __init__(self, knots: Array) -> None:
        """
        Onion spline transformation using given knots.

        Parameters
        ----------
        knots
            Increasing, equidistant spline knot sequence.

        Attributes
        ----------
        knots
            Array of spline knots.
        """
        super().__init__(knots)
        self._compute_coef = jax.jit(get_onion_fn(knots))  # type: ignore

    def _dot_and_deriv_n_fullbatch(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative for batch.
        """
        fx_n, deriv_n = self.bspline.dot_and_deriv_n(x, coef)
        in_core = (x >= self.min_knot) & (x <= self.max_knot)
        fx_n = jnp.where(in_core, fx_n, x)
        deriv_n = jnp.where(in_core, deriv_n, 1.0)
        return fx_n, deriv_n
