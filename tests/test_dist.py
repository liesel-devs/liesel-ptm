import jax
import jax.numpy as jnp
import jax.random as jrd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from liesel_ptm import bsplines
from liesel_ptm.bsplines import BSpline, ExtrapBSplineApprox
from liesel_ptm.dist import LocScaleTransformationDist, TransformationDist
from liesel_ptm.nodes import OnionCoefParam, VarWeibull

key = jrd.PRNGKey(42)


class TestTransformationDist:
    def test_init(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        assert dist is not None

    def test_batch_shape(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        log_increments = jrd.uniform(key, shape=(30, 4, knots.nparam_full_domain))
        coef = jnp.cumsum(jnp.exp(log_increments), axis=0)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef, basis_dot_and_deriv_fn=fn
        )

        assert dist.batch_shape == tf.TensorShape([30, 4])

    def test_batch_shape2(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        log_increments = jrd.uniform(key, shape=(30, 4, knots.nparam_full_domain))
        coef = jnp.cumsum(jnp.exp(log_increments), axis=0)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots,
            coef=coef,
            basis_dot_and_deriv_fn=fn,
            parametric_distribution=tfd.Normal,
            loc=0.0,
            scale=1.0,
        )

        assert dist.batch_shape == tf.TensorShape([30, 4])

    def test_transformation_and_logdet_spline(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        y = jnp.linspace(-2.0, 2.0, 50)
        z, logdet = dist.transformation_and_logdet_spline(y)

        assert jnp.allclose(y, z, atol=1e-5)
        assert jnp.allclose(logdet, 0.0, atol=1e-6)

    def test_transformation_and_logdet_parametric(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        y = jnp.linspace(-2.0, 2.0, 50)
        z, logdet = dist.transformation_and_logdet_parametric(y)

        assert jnp.allclose(y, z, atol=1e-5)
        assert jnp.allclose(logdet, 0.0, atol=1e-6)

        dist = TransformationDist(
            knots=knots.knots,
            coef=coef.value,
            parametric_distribution=tfd.Normal,
            loc=2.0,
            scale=2.0,
            basis_dot_and_deriv_fn=fn,
        )

        y = jnp.linspace(-2.0, 2.0, 50)
        z, logdet = dist.transformation_and_logdet_parametric(y)

        assert jnp.allclose((y - 2.0) / 2.0, z, atol=1e-5)
        assert jnp.allclose(logdet, -jnp.log(2.0), atol=1e-6)

    def test_transformation_and_logdet(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        y = jnp.linspace(-2.0, 2.0, 50)
        z, logdet = dist.transformation_and_logdet(y)

        assert jnp.allclose(y, z, atol=1e-5)
        assert jnp.allclose(logdet, 0.0, atol=1e-5)

    def test_log_prob(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        y = jnp.linspace(-2.0, 2.0, 50)
        lp = dist.log_prob(y)
        lp2 = tfd.Normal(loc=0.0, scale=1.0).log_prob(y)

        assert jnp.allclose(lp, lp2, atol=1e-5)

    def test_cdf(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        y = jnp.linspace(-2.0, 2.0, 50)
        p = dist.cdf(y)
        p2 = tfd.Normal(loc=0.0, scale=1.0).cdf(y)

        assert jnp.allclose(p, p2, atol=1e-5)

    def test_quantile(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        u = jnp.linspace(0.01, 0.99, 10)

        y = dist.quantile(u)
        y2 = tfd.Normal(loc=0.0, scale=1.0).quantile(u)

        assert jnp.allclose(y, y2, atol=1e-5)

    def test_sample(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        y = dist.sample(10, seed=key)
        y2 = tfd.Normal(loc=0.0, scale=1.0).sample(10, seed=key)

        assert jnp.allclose(y, y2, atol=1e-5)

    def test_sample_batched(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        log_increments = 0.1 * jrd.uniform(key, shape=(4, 30, knots.nparam_full_domain))
        coef = jnp.cumsum(jnp.exp(log_increments), axis=0)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef, basis_dot_and_deriv_fn=fn
        )
        with jax.disable_jit(disable=False):
            y = dist.sample(12, seed=key)

        assert y.shape == (12, 4, 30)

    def test_mean(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )
        assert dist.mean() == pytest.approx(0.0, abs=1e-5)

    def test_variance(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )
        assert dist.variance() == pytest.approx(1.0, abs=1e-5)

    def test_jit(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)
        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        def log_prob(y, coef):
            dist = TransformationDist(
                knots=knots.knots, coef=coef, basis_dot_and_deriv_fn=fn
            )
            return dist.log_prob(y)

        y = jnp.linspace(-2.0, 2.0, 50)
        lp = jax.jit(log_prob)(y, coef.value)
        assert lp is not None

    def test_jit_grad(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        def log_prob(y, coef):
            dist = TransformationDist(
                knots=jnp.asarray(knots.knots), coef=coef, basis_dot_and_deriv_fn=fn
            )
            return dist.log_prob(y).sum()

        y = jnp.linspace(-2.0, 2.0, 50)
        lp = jax.jit(jax.grad(log_prob, argnums=1))(y, coef.value)
        assert lp is not None

    @pytest.mark.parametrize("seed", (1, 2, 3, 4, 5, 6))
    def test_inverse_transformation(self, seed) -> None:
        knots = bsplines.OnionKnots(-1.0, 1.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)
        shape = coef.log_increments.transformed.value.shape
        coef.log_increments.transformed.value = jrd.normal(jrd.PRNGKey(seed), shape)
        coef.log_increments.update()
        coef.update()

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        ygrid = jnp.linspace(-5.0, 5.0, 300)
        zgrid, _ = dist.transformation_and_logdet(ygrid)

        ygrid_initial_guess = dist._inverse_transformation_initial_guess(zgrid)
        ygrid_inverted = dist.inverse_transformation(zgrid)

        assert not jnp.allclose(ygrid, zgrid, atol=0.2)

        initial_deviations = jnp.abs(ygrid_initial_guess - ygrid)
        deviations = jnp.abs(ygrid_inverted - ygrid)

        assert initial_deviations.sum() > deviations.sum()
        assert (initial_deviations >= 1e-5).sum() > (deviations >= 1e-5).sum()
        assert jnp.allclose(ygrid, ygrid_inverted, atol=1e-2)

    def test_inverse_transformation_jit(self) -> None:
        seed = 1
        knots = bsplines.OnionKnots(-1.0, 1.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)
        shape = coef.log_increments.transformed.value.shape
        coef.log_increments.transformed.value = jrd.normal(jrd.PRNGKey(seed), shape)
        coef.log_increments.update()
        coef.update()

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        ygrid = jnp.linspace(-5.0, 5.0, 300)
        zgrid, _ = dist.transformation_and_logdet(ygrid)

        ygrid_inverted = jax.jit(dist.inverse_transformation)(zgrid)

        assert jnp.allclose(ygrid, ygrid_inverted, atol=1e-7)

    def test_inverse_transformation_initial_guess(self) -> None:
        knots = bsplines.OnionKnots(-1.0, 1.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)
        shape = coef.log_increments.transformed.value.shape
        coef.log_increments.transformed.value = jrd.normal(key, shape)
        coef.log_increments.update()
        coef.update()

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        ygrid = jnp.linspace(-5, 5, 300)
        zgrid, _ = dist.transformation_and_logdet(ygrid)

        ygrid_inverted = dist._inverse_transformation_initial_guess(zgrid)

        assert not jnp.allclose(ygrid, zgrid, atol=0.2)
        assert jnp.allclose(ygrid, ygrid_inverted, atol=1e-2)

    def test_transformation_stability_no_parametric(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)
        shape = coef.log_increments.transformed.value.shape
        coef.log_increments.transformed.value = jrd.normal(key, shape)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        ygrid = jnp.linspace(-20, 20, 300)

        zgrid, logdet = dist.transformation_and_logdet(ygrid)

        assert not jnp.any(jnp.isinf(zgrid))
        assert not jnp.any(jnp.isinf(logdet))
        assert not jnp.any(jnp.isnan(zgrid))
        assert not jnp.any(jnp.isnan(logdet))

    def test_transformation_stability_normal(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)
        shape = coef.log_increments.transformed.value.shape
        coef.log_increments.transformed.value = jrd.normal(key, shape)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots,
            coef=coef.value,
            basis_dot_and_deriv_fn=fn,
            parametric_distribution=tfd.Normal,
            loc=0.0,
            scale=1.0,
        )

        ygrid = jnp.linspace(-20, 20, 300)

        zgrid, logdet = dist.transformation_and_logdet(ygrid)

        assert not jnp.any(jnp.isinf(zgrid))
        assert not jnp.any(jnp.isinf(logdet))
        assert not jnp.any(jnp.isnan(zgrid))
        assert not jnp.any(jnp.isnan(logdet))

    def test_transformation_inverse_stability(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)
        shape = coef.log_increments.transformed.value.shape
        coef.log_increments.transformed.value = jrd.normal(key, shape)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = TransformationDist(
            knots=knots.knots, coef=coef.value, basis_dot_and_deriv_fn=fn
        )

        zgrid = jnp.linspace(-20, 20, 300)

        ygrid = dist.inverse_transformation(zgrid)

        assert not jnp.any(jnp.isinf(ygrid))
        assert not jnp.any(jnp.isnan(ygrid))


class TestLocScaleTransformationDist:
    def test_quantile_batched(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        bspline = ExtrapBSplineApprox(knots=knots.knots, order=3)
        fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        dist = LocScaleTransformationDist(
            knots=knots.knots,
            coef=coef.value,
            basis_dot_and_deriv_fn=fn,
            loc=jax.random.normal(key, shape=(4, 8, 32)),
            scale=jnp.exp(jax.random.normal(key, shape=(4, 8, 32))),
        )

        u = jnp.linspace(0.01, 0.99, 10)
        u = jnp.reshape(u, shape=(10, 1, 1, 1))

        y = dist.quantile(u)
        assert y.shape == (10, 4, 8, 32)

    def test_transformation_sample_shape(self):
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        tau2 = VarWeibull(1.0, scale=0.05, name="tau2")
        coef = OnionCoefParam(knots, tau2=tau2)

        samples = {}
        samples[coef.log_increments.transformed.name] = jax.random.normal(
            key, shape=(4, 100, knots.nparam)
        )
        samples["tau2_transformed"] = jnp.ones((4, 100))
        coef_samples = coef.predict(samples)

        loc = jax.random.normal(key, shape=(4, 100, 382))
        scale = jnp.log(1 + jnp.exp(jax.random.normal(key, shape=(4, 100, 382))))

        bspline = BSpline(knots=knots.knots, target_slope=1.0)

        dist = LocScaleTransformationDist(
            knots=knots.knots,
            basis_dot_and_deriv_fn=bspline.dot_and_deriv,
            coef=coef_samples,
            loc=loc,
            scale=scale,
        )

        assert dist.batch_shape is not None

        z, _ = dist.transformation_and_logdet(jnp.linspace(-5.0, 5.0, 382))
        assert z.shape == (4, 100, 382)

        z, _ = dist.transformation_and_logdet(0.0)
        assert z.shape == (4, 100, 382)

        dist = LocScaleTransformationDist(
            knots=knots.knots,
            basis_dot_and_deriv_fn=bspline.dot_and_deriv,
            coef=coef_samples[0, ...],
            loc=loc[0, ...],
            scale=scale[0, ...],
        )

        assert dist.batch_shape is not None

        z, _ = dist.transformation_and_logdet(jnp.linspace(-5.0, 5.0, 382))
        assert z.shape == (100, 382)

        z, _ = dist.transformation_and_logdet(0.0)
        assert z.shape == (100, 382)

        dist = LocScaleTransformationDist(
            knots=knots.knots,
            basis_dot_and_deriv_fn=bspline.dot_and_deriv,
            coef=coef_samples[0, 0, ...],
            loc=loc[0, 0, ...],
            scale=scale[0, 0, ...],
        )

        assert dist.batch_shape is not None

        z, _ = dist.transformation_and_logdet(jnp.linspace(-5.0, 5.0, 382))
        assert z.shape == (382,)

        z, _ = dist.transformation_and_logdet(0.0)
        assert z.shape == (382,)

        dist = LocScaleTransformationDist(
            knots=knots.knots,
            basis_dot_and_deriv_fn=bspline.dot_and_deriv,
            coef=coef_samples[0, 0, ...],
            loc=0.0,
            scale=0.0,
        )

        assert dist.batch_shape is not None

        z, _ = dist.transformation_and_logdet(jnp.linspace(-5.0, 5.0, 382))
        assert z.shape == (382,)

        z, _ = dist.transformation_and_logdet(0.0)
        assert z.shape == (1,)
