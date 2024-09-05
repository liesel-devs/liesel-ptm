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
        basis_dot_and_deriv_fn: (
            Callable[[Array, Array], tuple[Array, Array]] | None
        ) = None,
        parametric_distribution: type[tfd.Distribution] | None = None,
        reference_distribution: tfd.Distribution | None = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "TransformationDist",
        centered: bool = False,
        scaled: bool = False,
        **parametric_distribution_kwargs,
    ):
        parameters = dict(locals())

        self.knots = knots
        self.coef = coef
        self.parametric_distribution_kwargs = parametric_distribution_kwargs
        self.centered = centered
        self.scaled = scaled

        if basis_dot_and_deriv_fn is None:
            bspline = ExtrapBSplineApprox(knots=knots, order=3)
            basis_dot_and_deriv_fn = bspline.get_extrap_basis_dot_and_deriv_fn(
                target_slope=1.0
            )

        self.bdot_and_deriv_fn = basis_dot_and_deriv_fn

        if reference_distribution is None:
            self.reference_distribution = tfd.Normal(loc=0.0, scale=1.0)
        else:
            self.reference_distribution = reference_distribution

        if parametric_distribution is None and parametric_distribution_kwargs:
            raise ValueError(
                "Provided 'parametric_distribution_kwargs', but no value for"
                " 'parametric_distribution'."
            )

        if parametric_distribution is None:
            self.parametric_distribution = None
        else:
            self.parametric_distribution = parametric_distribution(
                **parametric_distribution_kwargs
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
        if self.parametric_distribution is None:
            parametric_mean = 0.0
        else:
            parametric_mean = self.parametric_distribution._mean()

        if self.centered:
            return parametric_mean

        return parametric_mean + self.transformation_spline_mean()

    def _stddev(self) -> Array:
        if self.parametric_distribution is None:
            parametric_stddev: float | Array = 1.0
        else:
            try:
                parametric_stddev = self.parametric_distribution._stddev()
            except NotImplementedError:
                parametric_stddev = jnp.sqrt(self.parametric_distribution._variance())

        if self.scaled:
            return parametric_stddev

        mean = self.transformation_spline_mean()
        return parametric_stddev * jnp.sqrt(self.transformation_spline_variance(mean))

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
        return jnp.moveaxis(self._quantile(jnp.moveaxis(u, 0, -1)), -1, 0)

    def _quantile(self, value: Array) -> Array:
        z = self.reference_distribution.quantile(value)
        y = self.inverse_transformation(z)
        return y

    def quantile_spline(self, value: Array) -> Array:
        z = self.reference_distribution.quantile(value)
        return self.inverse_transformation_spline(z)

    def _event_shape(self):
        return tf.TensorShape([])

    def _event_shape_tensor(self):
        return jnp.array([], dtype=jnp.int32)

    def _batch_shape(self):
        kwargs_shapes = [
            jnp.shape(param) for param in self.parametric_distribution_kwargs.values()
        ]
        parametric_shape = tuple()
        for shape in kwargs_shapes:
            parametric_shape = tf.broadcast_static_shape(parametric_shape, shape)

        coef_shape = self.coef.shape[:-1]

        if len(parametric_shape):
            coef_shape = coef_shape + (1,)

        coef_shape = tf.TensorShape(coef_shape)

        return tf.broadcast_static_shape(coef_shape, parametric_shape)

    def _batch_shape_tensor(self):
        kwargs_shapes = [
            jnp.shape(param) for param in self.parametric_distribution_kwargs.values()
        ]
        parametric_shape = tuple()
        for shape in kwargs_shapes:
            parametric_shape = tf.broadcast_dynamic_shape(parametric_shape, shape)

        coef_shape = self.coef.shape[:-1]

        if len(parametric_shape):
            coef_shape = coef_shape + (1,)

        coef_shape = tf.TensorShape(coef_shape)

        return tf.broadcast_dynamic_shape(coef_shape, parametric_shape)

    def log_prob_spline(self, value: Array):
        z, logdet = self.transformation_and_logdet_spline(value)
        return self.reference_distribution.log_prob(z) + logdet

    def prob_spline(self, value: Array):
        return jnp.exp(self.log_prob_spline(value))

    def cdf_spline(self, value: Array):
        z, _ = self.transformation_and_logdet_spline(value)
        return self.reference_distribution.cdf(z)

    def transformation_and_logdet_parametric(self, value: Array) -> tuple[Array, Array]:
        if self.parametric_distribution is None:
            return value, jnp.zeros_like(value)

        F_apriori = self.parametric_distribution
        Fz = self.reference_distribution

        # Use jnp.finfo to get machine epsilon and min/max float values
        eps = jnp.finfo(value.dtype).eps
        tiny = jnp.finfo(value.dtype).tiny
        max_float = 1.0 - eps

        u = F_apriori.cdf(value)
        u = jnp.where(u >= 1.0, max_float, u)  # safeguard using max float
        u = jnp.where(u <= 0.0, tiny, u)  # safeguard using smallest positive float

        transf = Fz.quantile(u)
        logdet = F_apriori.log_prob(value) - Fz.log_prob(transf)

        return transf, logdet

    def _transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        nan_mask = jnp.isnan(value)
        transf, deriv = self.bdot_and_deriv_fn(value, self.coef)
        transf = jnp.where(nan_mask, jnp.nan, transf)
        deriv = jnp.where(nan_mask, jnp.nan, deriv)
        tiny = jnp.finfo(value.dtype).tiny
        deriv = jnp.clip(deriv, min=tiny)  # safeguard against numerical issues
        return transf, jnp.log(deriv)

    def transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        if self.centered:
            ymean = self.transformation_spline_mean()  # intercept / expected val.
        else:
            ymean = jnp.zeros(1)

        if self.scaled and not self.centered:
            _ymean = self.transformation_spline_mean()  # intercept / expected val.
            ystd = jnp.sqrt(self.transformation_spline_variance(_ymean))
        elif self.scaled and self.centered:
            ystd = jnp.sqrt(self.transformation_spline_variance(ymean))
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

    def transformation_spline_mean(self) -> Array:
        def fn(x):
            z, logdet = self._transformation_and_logdet_spline(x)
            return x * self.reference_distribution.prob(z) * jnp.exp(logdet)

        mom = integrate_simpson(fn, a=self.knots[0], b=self.knots[-1], N=1024)

        return mom

    def transformation_spline_variance(self, mean: Array | None = None) -> Array:
        if mean is None:
            mean = self.transformation_spline_mean()

        def fn(x):
            z, logdet = self._transformation_and_logdet_spline(x)
            return (
                (x - mean) ** 2 * self.reference_distribution.prob(z) * jnp.exp(logdet)
            )

        var = integrate_simpson(fn, a=self.knots[0], b=self.knots[-1], N=1024)

        return var

    def _inverse_fn_newton(self, z, initial_guess, tol, max_iter):
        def newton_step(y, z):
            h, logdet = self.transformation_and_logdet_spline(y)
            h_deriv = jnp.exp(logdet)
            h_deriv = jnp.clip(h_deriv, min=1e-30)  # safeguard against numerical issues
            return y - (h - z) / h_deriv

        def cond_fn(state):
            y, iter_count, z = state
            h, _ = self.transformation_and_logdet_spline(y)
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

    def _find_y_lo(
        self,
        min_z: float | Array,
    ) -> Array:
        y_lo = self.knots[0]
        left_shift = jnp.array(0.1)
        min_zgrid = self.transformation_and_logdet_spline(y_lo - left_shift)[0]
        min_zgrid = jnp.min(min_zgrid)

        def _cond_fun_l(val):
            _, min_zgrid = val
            return jnp.squeeze(min_zgrid >= min_z)

        def _body_fun_l(val):
            left_shift, min_zgrid = val
            left_shift = left_shift + 0.5
            min_zgrid = self.transformation_and_logdet_spline(y_lo - left_shift)[0]
            min_zgrid = jnp.min(min_zgrid)
            return left_shift, min_zgrid

        # while _cond_fun_l((left_shift, min_zgrid)):
        #     left_shift, min_zgrid = _body_fun_l((left_shift, min_zgrid))

        left_shift, _ = jax.lax.while_loop(
            _cond_fun_l, _body_fun_l, (left_shift, min_zgrid)
        )
        return y_lo - left_shift

    def _find_y_hi(
        self,
        max_z: float | Array,
    ) -> Array:
        y_hi = self.knots[-1]
        right_shift = jnp.array(0.1)
        max_zgrid = self.transformation_and_logdet_spline(y_hi + right_shift)[0]
        max_zgrid = jnp.max(max_zgrid)

        def _cond_fun_r(val):
            _, max_zgrid = val
            return jnp.squeeze(max_zgrid <= max_z)

        def _body_fun_r(val):
            right_shift, max_zgrid = val
            right_shift = right_shift + 0.5
            max_zgrid = self.transformation_and_logdet_spline(y_hi + right_shift)[0]
            max_zgrid = jnp.max(max_zgrid)
            return right_shift, max_zgrid

        # while _cond_fun_r((right_shift, max_zgrid)):
        #     right_shift, max_zgrid = _body_fun_r((right_shift, max_zgrid))
        right_shift, _ = jax.lax.while_loop(
            _cond_fun_r, _body_fun_r, (right_shift, max_zgrid)
        )
        return y_hi + right_shift

    def _find_grid(
        self,
        y_lo: float | Array,
        y_hi: float | Array,
        ngrid: int = 200,
    ) -> tuple[Array, Array]:
        """
        Finds a grid of y values such that h(y) covers the range of z values.
        """

        ygrid = jnp.linspace(y_lo, y_hi, ngrid)
        zgrid = self.transformation_and_logdet_spline(ygrid)[0]

        return ygrid, zgrid

    def _inverse_transformation_initial_guess(self, value: Array) -> Array:
        if self.centered:
            ymean = self.transformation_spline_mean()  # intercept / expected val.
        else:
            ymean = jnp.zeros(1)
        if self.scaled and not self.centered:
            _ymean = self.transformation_spline_mean()  # intercept / expected val.
            ystd = jnp.sqrt(self.transformation_spline_variance(_ymean))
        elif self.scaled and self.centered:
            ystd = jnp.sqrt(self.transformation_spline_variance(ymean))
        else:
            ystd = jnp.ones(1)

        scaled_value = (value - ymean) / ystd

        y_lo = self._find_y_lo(jnp.min(scaled_value))
        y_hi = self._find_y_hi(jnp.max(scaled_value))
        # ngrid = jnp.array((y_hi - y_lo + 1) * 100, int)
        y_tilde_grid, zgrid = self._find_grid(y_lo, y_hi, ngrid=1000)

        initial_guess = approximate_inverse(y_tilde_grid, zgrid, value)

        return initial_guess

    def inverse_transformation_spline(
        self, value: Array, tol: float = 1e-5, max_iter: int = 100
    ) -> Array:
        initial_guess = self._inverse_transformation_initial_guess(value)

        y_tilde = self._inverse_fn_newton(
            value,
            initial_guess,
            tol,
            max_iter,
        )
        return y_tilde

    def inverse_transformation_parametric(self, value: Array) -> Array:
        if self.parametric_distribution is None:
            return value

        # Use jnp.finfo to get machine epsilon and min/max float values
        eps = jnp.finfo(value.dtype).eps
        tiny = jnp.finfo(value.dtype).tiny
        max_float = 1.0 - eps

        u = self.reference_distribution.cdf(value)
        u = jnp.where(u >= 1.0, max_float, u)  # safeguard against numerical issues
        u = jnp.where(u <= 0.0, tiny, u)  # safeguard against numerical issues
        y = self.parametric_distribution.quantile(u)

        return y

    def inverse_transformation(
        self, value: Array, tol: float = 1e-6, max_iter: int = 100
    ) -> Array:
        y_tilde = self.inverse_transformation_spline(value, tol, max_iter)
        y = self.inverse_transformation_parametric(y_tilde)

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
        centered: bool = False,
        scaled: bool = False,
    ) -> None:
        super().__init__(
            knots=knots,
            coef=coef,
            parametric_distribution=tfd.Normal,
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
        if self.parametric_distribution is None:
            raise RuntimeError

        sd = self.parametric_distribution.stddev()
        transf = (value - self.parametric_distribution.mean()) / sd

        logdet = -jnp.log(sd)

        return transf, logdet

    def inverse_transformation_parametric(self, value: Array) -> Array:
        if self.parametric_distribution is None:
            raise RuntimeError

        sd = self.parametric_distribution.stddev()
        m = self.parametric_distribution.mean()
        y = value * sd + m

        return y
