from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Self

import jax
import jax.numpy as jnp
from jax import Array
from tensorflow_probability.substrates.jax import distributions as tfd

from ..bijector import OnionSplineBijector
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

    return jnp.vectorize(compute_coef, signature="(k)->(d)")

    # return compute_coef


class OnionSpline(TransformationSpline):
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

    Notes
    -----
    Set geometry and numerical settings before first evaluation; JIT kernels
    capture them. Reuse the spline across parameter draws. Changing settings
    requires a new spline, which creates a separate compilation-cache entry.
    """

    def __init__(
        self,
        knots: Array,
        subscripts: Literal[
            "...nj,...j->...n", "...nj,...nj->...n"
        ] = "...nj,...j->...n",
    ) -> None:
        super().__init__(knots, subscripts=subscripts)
        self.supports_rowwise_coef = True
        self._compute_coef = jax.jit(get_onion_fn(knots))

    @classmethod
    def from_bounds(cls, a: float, b: float, nparam: int, order: int = 3) -> Self:
        """
        Create an onion spline from the bounds of its flexible interval.

        Parameters
        ----------
        a
            Left boundary of the flexible interval.
        b
            Right boundary of the flexible interval.
        nparam
            Number of flexible parameters.
        order
            Spline order.

        Returns
        -------
        spline
            An onion spline with [OnionKnots][liesel_ptm.OnionKnots].
        """
        return cls(OnionKnots(a, b, nparam=nparam, order=order).knots)

    def bijector(
        self,
        coef: Array,
        centered: bool = False,
        scaled: bool = False,
        gauss_legendre_order: int = 8,
        integration_bounds: tuple[float, float] | None = None,
        reference_distribution: tfd.Distribution | None = None,
        validate_args: bool = False,
    ) -> OnionSplineBijector:
        """
        Return the onion spline transformation as a TFP bijector.

        Parameters
        ----------
        coef
            Spline log increments with shape ``batch_shape + (n_param,)``.
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
            An [OnionSplineBijector][liesel_ptm.OnionSplineBijector] that uses
            this spline.
        """
        return OnionSplineBijector(
            coef,
            bspline=self,
            centered=centered,
            scaled=scaled,
            gauss_legendre_order=gauss_legendre_order,
            integration_bounds=integration_bounds,
            reference_distribution=reference_distribution,
            validate_args=validate_args,
        )

    def _evaluate_spline(self, value: Array, coef: Array) -> tuple[Array, Array]:
        """
        Compute dot product and derivative for broadcasted values and coefficients.
        """
        x = value
        coef = self._coef_for_eval(x, coef)
        fx_n, deriv_n = self.bspline.dot_and_deriv_n(x, coef)
        in_core = (x >= self.min_knot) & (x <= self.max_knot)
        fx_n = jnp.where(in_core, fx_n, x)
        deriv_n = jnp.where(in_core, deriv_n, 1.0)
        return fx_n, deriv_n

    def _evaluate_spline_value(self, value: Array, coef: Array) -> Array:
        """
        Compute dot product for broadcasted values and coefficients.
        """
        x = value
        coef = self._coef_for_eval(x, coef)
        fx_n = self.bspline.dot_n(x, coef)
        in_core = (x >= self.min_knot) & (x <= self.max_knot)
        return jnp.where(in_core, fx_n, x)

    def _evaluate_rowwise_shared_value(
        self, value: Array, coef: Array
    ) -> tuple[Array, Array]:
        """
        Evaluate rowwise coefficients at values shared along the rowwise axis.
        """
        x = value
        fx_n, deriv_n = super()._evaluate_rowwise_shared_value(x, coef)
        in_core = (x >= self.min_knot) & (x <= self.max_knot)
        fx_n = jnp.where(in_core, fx_n, x)
        deriv_n = jnp.where(in_core, deriv_n, 1.0)
        return fx_n, deriv_n

    def _evaluate_rowwise_shared_value_only(self, value: Array, coef: Array) -> Array:
        """
        Evaluate rowwise coefficients at values shared along the rowwise axis.
        """
        x = value
        fx_n = super()._evaluate_rowwise_shared_value_only(x, coef)
        in_core = (x >= self.min_knot) & (x <= self.max_knot)
        return jnp.where(in_core, fx_n, x)
