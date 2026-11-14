from __future__ import annotations

from typing import Self

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


def _onion_coef_constants(knots: Array) -> tuple[Array, Array, Array, Array]:
    """Arrays that the onion coefficient map needs."""
    m = len(knots) - 6
    dk = jnp.diff(knots).mean()

    log_dk = jnp.log(jnp.full((3,), fill_value=dk))
    log_correction_denominator = jnp.log((m - 5) * dk)

    nparam = len(knots) - 11
    zeros = jnp.zeros((nparam,))
    return log_dk, log_correction_denominator, zeros, knots[2]


@jax.jit
def _onion_coef(
    log_increments: Array,
    log_dk: Array,
    log_correction_denominator: Array,
    zeros: Array,
    first_coef: Array,
) -> Array:
    """
    Compute full onion spline coefficients from log increments.

    The onion correction rescales the increments so that the spline maps its core
    onto itself.
    """

    def compute_coef(log_increments):
        log_increments = log_increments + zeros
        correction = jax.nn.logsumexp(log_increments) - log_correction_denominator
        full_latent_coef = jnp.concatenate(
            (log_dk, log_increments - correction, log_dk)
        )
        return jnp.r_[(first_coef, jnp.exp(full_latent_coef))]

    return jnp.vectorize(compute_coef, signature="(k)->(d)")(log_increments)


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
    Splines are JAX pytrees. Splines with the same settings share compiled
    kernels. Treat a spline as immutable: create a new one to change its knots
    or settings.
    """

    def __init__(self, knots: Array) -> None:
        super().__init__(knots)
        (
            self._coef_log_dk,
            self._coef_log_denominator,
            self._coef_zeros,
            self._coef_first,
        ) = _onion_coef_constants(self.knots)

    _array_fields = TransformationSpline._array_fields + (
        "_coef_log_dk",
        "_coef_log_denominator",
        "_coef_zeros",
        "_coef_first",
    )

    def _compute_coef(self, raw_coef: Array) -> Array:
        return _onion_coef(
            raw_coef,
            self._coef_log_dk,
            self._coef_log_denominator,
            self._coef_zeros,
            self._coef_first,
        )

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
        fx_n = self.bspline.dot_n(x, coef)
        in_core = (x >= self.min_knot) & (x <= self.max_knot)
        return jnp.where(in_core, fx_n, x)
