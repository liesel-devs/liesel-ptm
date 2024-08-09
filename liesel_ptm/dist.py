from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates.jax import tf2jax as tf

from .bsplines import ExtrapBSplineApprox
from .custom_types import Array, KeyArray
from .inverse_fn import approximate_inverse


def integrate_simpson(
    f: Callable[[Array], Array], a: float, b: float, N: int = 20
) -> Array:
    """
    Implementation from:
    https://jax-cosmo.readthedocs.io/en/latest/_modules/jax_cosmo/scipy/integrate.html#romb
    """
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = jnp.linspace(a, b, N + 1)
    y = f(x)
    S = (
        dx
        / 3
        * jnp.sum(
            y[..., 0:-1:2] + 4 * y[..., 1::2] + y[..., 2::2], axis=-1, keepdims=True
        )
    )
    return S


class TransformationDist(tfd.Distribution):
    def __init__(
        self,
        knots: Array,
        coef: Array,
        apriori_distribution: type[tfd.Distribution] | None = None,
        reference_distribution: tfd.Distribution | None = None,
        basis_dot_and_deriv_fn: (
            Callable[[Array, Array], tuple[Array, Array]] | None
        ) = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "TransformationDist",
        centered: bool = False,
        scaled: bool = False,
        **apriori_distribution_kwargs,
    ):
        parameters = dict(locals())

        self.knots = knots
        self.coef = coef
        self.apriori_distribution_kwargs = apriori_distribution_kwargs
        self.centered = centered
        self.scaled = scaled

        if basis_dot_and_deriv_fn is None:
            bspline = ExtrapBSplineApprox(knots=knots, order=3)
            fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)
            self.bdot_and_deriv_fn = fn
        else:
            self.bdot_and_deriv_fn = basis_dot_and_deriv_fn

        if reference_distribution is None:
            self.reference_distribution = tfd.Normal(loc=0.0, scale=1.0)
        else:
            self.reference_distribution = reference_distribution

        if apriori_distribution is None and apriori_distribution_kwargs:
            raise ValueError(
                "Provided 'apriori_distribution_kwargs', but no value for"
                " 'apriori_distribution'."
            )

        if apriori_distribution is None:
            self.apriori_distribution = tfd.Normal(loc=0.0, scale=1.0)
        else:
            self.apriori_distribution = apriori_distribution(
                **apriori_distribution_kwargs
            )

        super().__init__(
            dtype=coef.dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    def _mean(self) -> Array:
        apriori_mean = self.apriori_distribution._mean()
        if self.centered:
            return apriori_mean

        return apriori_mean + self._transformation_spline_mean()

    def _stddev(self) -> Array:
        try:
            apriori_stddev = self.apriori_distribution._stddev()
        except NotImplementedError:
            apriori_stddev = jnp.sqrt(self.apriori_distribution._variance())

        if self.scaled:
            return apriori_stddev

        mean = self._transformation_spline_mean()
        return apriori_stddev * jnp.sqrt(self._transformation_spline_variance(mean))

    def _cdf(self, value: Array) -> Array | float:
        z, _ = self.transformation_and_logdet(value)
        return self.reference_distribution.cdf(z)

    def _log_cdf(self, value: Array) -> Array | float:
        return jnp.log(self._cdf(value))

    def _log_prob(self, value: Array) -> Array | float:
        z, logdet = self.transformation_and_logdet(value)
        return self.reference_distribution.log_prob(z) + logdet

    def _prob(self, value: Array) -> Array | float:
        return jnp.exp(self._log_prob(value))

    def _sample_n(self, n: int | Array, seed: KeyArray | None = None) -> Array:
        shape = [n] + self.batch_shape + self.event_shape
        u = jax.random.uniform(seed, shape=shape)  # type: ignore
        return self._quantile(u)

    def _quantile(self, value: Array) -> Array:
        z = self.reference_distribution.quantile(value)
        y = self.inverse_transformation(z)
        return y

    def _event_shape(self):
        return tf.TensorShape([])

    def _event_shape_tensor(self):
        return jnp.array([], dtype=jnp.int32)

    def _batch_shape(self):
        kwargs_shapes = [
            param.shape for param in self.apriori_distribution_kwargs.values()
        ]
        shape = jnp.broadcast_shapes(self.coef.shape[:-1], *kwargs_shapes)
        return tf.TensorShape(shape)

    def _batch_shape_tensor(self):
        kwargs_shapes = [
            param.shape for param in self.apriori_distribution_kwargs.values()
        ]
        shape = jnp.broadcast_shapes(self.coef.shape[:-1], *kwargs_shapes)
        return jnp.array(shape, dtype=jnp.int32)

    def transformation_and_logdet_parametric(self, value: Array) -> tuple[Array, Array]:
        F_apriori = self.apriori_distribution
        Fz = self.reference_distribution

        u = F_apriori.cdf(value)
        u = jnp.where(u >= 1.0, 1 - 1e-7, u)  # safeguard against numerical issues
        u = jnp.where(u <= 0.0, 1e-30, u)  # safeguard against numerical issues

        transf = Fz.quantile(u)
        logdet = F_apriori.log_prob(value) - Fz.log_prob(transf)

        return transf, logdet

    def _transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        transf, deriv = self.bdot_and_deriv_fn(value, self.coef)
        deriv = jnp.clip(deriv, min=1e-30)  # safeguard against numerical issues
        return transf, jnp.log(deriv)

    def transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        if self.centered:
            ymean = self._transformation_spline_mean()  # intercept / expected val.
        else:
            ymean = jnp.zeros(1)

        if self.scaled and not self.centered:
            _ymean = self._transformation_spline_mean()  # intercept / expected val.
            ystd = jnp.sqrt(self._transformation_spline_variance(_ymean))
        elif self.scaled and self.centered:
            ystd = jnp.sqrt(self._transformation_spline_variance(ymean))
        else:
            ystd = jnp.ones(1)

        value = value * ystd + ymean
        logdet = jnp.log(ystd)

        z, transf_logdet = self._transformation_and_logdet_spline(value)

        z_logdet = transf_logdet + logdet

        return z, z_logdet

    def transformation_and_logdet(self, value: Array) -> tuple[Array, Array]:
        transf_param, logdet_param = self.transformation_and_logdet_parametric(value)
        transf_spline, logdet_spline = self.transformation_and_logdet_spline(
            transf_param
        )

        logdet = logdet_param + logdet_spline

        return transf_spline, logdet

    def _transformation_spline_mean(self) -> Array:
        def fn(x):
            z, logdet = self._transformation_and_logdet_spline(x)
            return x * self.reference_distribution.prob(z) * jnp.exp(logdet)

        mom = integrate_simpson(fn, a=self.knots[0], b=self.knots[-1], N=1024)

        return mom

    def _transformation_spline_variance(self, mean: Array) -> Array:
        def fn(x):
            z, logdet = self._transformation_and_logdet_spline(x)
            return (
                (x - mean) ** 2 * self.reference_distribution.prob(z) * jnp.exp(logdet)
            )

        var = integrate_simpson(fn, a=self.knots[0], b=self.knots[-1], N=1024)

        return var

    def _inverse_fn_newton(self, z, initial_guess, tol, max_iter):
        def newton_step(y, z):
            h, logdet = self.transformation_and_logdet(y)
            h_deriv = jnp.exp(logdet)
            h_deriv = jnp.clip(h_deriv, min=1e-30)  # safeguard against numerical issues
            return y - (h - z) / h_deriv

        def cond_fn(state):
            y, iter_count, z = state
            h, _ = self.transformation_and_logdet(y)
            return jnp.logical_and(
                jnp.any(jnp.abs(h - z) >= tol), iter_count < max_iter
            )

        def body_fn(state):
            y, iter_count, z = state
            y_new = newton_step(y, z)
            return y_new, iter_count + 1, z

        state = (initial_guess, 0, z)
        y_new, _, _ = jax.lax.while_loop(cond_fn, body_fn, state)

        return y_new

    def _inverse_transformation_initial_guess(self, value: Array) -> Array:
        if self.centered:
            ymean = self._transformation_spline_mean()  # intercept / expected val.
        else:
            ymean = jnp.zeros(1)
        if self.scaled and not self.centered:
            _ymean = self._transformation_spline_mean()  # intercept / expected val.
            ystd = jnp.sqrt(self._transformation_spline_variance(_ymean))
        elif self.scaled and self.centered:
            ystd = jnp.sqrt(self._transformation_spline_variance(ymean))
        else:
            ystd = jnp.ones(1)

        scaled_value = (value - ymean) / ystd

        y_tilde_grid = jnp.linspace(self.knots[0], self.knots[-1], 500)
        zgrid, _ = self.transformation_and_logdet_spline(y_tilde_grid)
        initial_guess = approximate_inverse(y_tilde_grid, zgrid, value)
        initial_guess = jnp.where(
            (scaled_value <= self.knots[0]) | (scaled_value >= self.knots[-1]),
            scaled_value,
            initial_guess,
        )

        return initial_guess

    def inverse_transformation(
        self, value: Array, tol: float = 1e-6, max_iter: int = 100
    ) -> Array:
        value = jnp.moveaxis(value, 0, -1)

        initial_guess = self._inverse_transformation_initial_guess(value)

        y_tilde = self._inverse_fn_newton(
            value,
            initial_guess,
            tol,
            max_iter,
        )

        y_tilde = jnp.moveaxis(y_tilde, -1, 0)

        u = self.reference_distribution.cdf(y_tilde)
        u = jnp.where(u >= 1.0, 1 - 1e-7, u)  # safeguard against numerical issues
        u = jnp.where(u <= 0.0, 1e-30, u)  # safeguard against numerical issues
        y = self.apriori_distribution.quantile(u)
        return y


class LocScaleTransformationDist(TransformationDist):
    def __init__(
        self,
        knots: Array,
        coef: Array,
        loc: Array,
        scale: Array,
        basis_dot_and_deriv_fn: (
            Callable[[Array, Array], tuple[Array, Array]] | None
        ) = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "LocScaleTransformationDist",
        centered: bool = True,
        scaled: bool = True,
    ) -> None:
        super().__init__(
            knots=knots,
            coef=coef,
            apriori_distribution=tfd.Normal,
            reference_distribution=tfd.Normal(loc=0.0, scale=1.0),
            basis_dot_and_deriv_fn=basis_dot_and_deriv_fn,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            loc=jnp.atleast_1d(loc),
            scale=jnp.atleast_1d(scale),
            centered=centered,
            scaled=scaled,
        )

    def transformation_and_logdet_parametric(self, value: Array) -> tuple[Array, Array]:
        sd = self.apriori_distribution.stddev()
        transf = (value - self.apriori_distribution.mean()) / sd

        logdet = -jnp.log(sd)

        return transf, logdet

    def inverse_transformation(
        self, value: Array, tol: float = 1e-6, max_iter: int = 100
    ) -> Array:
        y_tilde_grid = self._grid()
        zgrid, _ = self.transformation_and_logdet_spline(y_tilde_grid)
        value = jnp.moveaxis(value, 0, -1)
        initial_guess = approximate_inverse(y_tilde_grid, zgrid, value)

        y_tilde = self._inverse_fn_newton(
            value,
            initial_guess,
            tol,
            max_iter,
        )

        y_tilde = jnp.moveaxis(y_tilde, -1, 0)

        y = (
            y_tilde * self.apriori_distribution.stddev()
            + self.apriori_distribution.mean()
        )

        return y