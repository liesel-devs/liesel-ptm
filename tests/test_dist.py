import jax
import jax.numpy as jnp
import jax.random as jrd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from liesel_ptm import bsplines
from liesel_ptm.dist import TransformationDist
from liesel_ptm.nodes import OnionCoefParam

key = jrd.PRNGKey(42)


class TestTransformationDist:
    def test_init(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        coef = OnionCoefParam(knots)

        dist = TransformationDist(knots=knots.knots, coef=coef.value)

        assert dist is not None

    def test_batch_shape(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        log_increments = jrd.uniform(key, shape=(30, 4, knots.nparam_full_domain))
        coef = jnp.cumsum(jnp.exp(log_increments), axis=0)

        dist = TransformationDist(knots=knots.knots, coef=coef)

        assert dist.batch_shape == tf.TensorShape([30, 4])

    def test_transformation_and_logdet_spline(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        coef = OnionCoefParam(knots)

        dist = TransformationDist(knots=knots.knots, coef=coef.value)

        y = jnp.linspace(-2.0, 2.0, 50)
        z, logdet = dist.transformation_and_logdet_spline(y)

        assert jnp.allclose(y, z, atol=1e-5)
        assert jnp.allclose(logdet, 0.0, atol=1e-6)

    def test_transformation_and_logdet_parametric(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        coef = OnionCoefParam(knots)

        dist = TransformationDist(knots=knots.knots, coef=coef.value)

        y = jnp.linspace(-2.0, 2.0, 50)
        z, logdet = dist.transformation_and_logdet_parametric(y)

        assert jnp.allclose(y, z, atol=1e-5)
        assert jnp.allclose(logdet, 0.0, atol=1e-6)

        dist = TransformationDist(
            knots=knots.knots,
            coef=coef.value,
            apriori_distribution=tfd.Normal,
            loc=2.0,
            scale=2.0,
        )

        y = jnp.linspace(-2.0, 2.0, 50)
        z, logdet = dist.transformation_and_logdet_parametric(y)

        assert jnp.allclose((y - 2.0) / 2.0, z, atol=1e-5)
        assert jnp.allclose(logdet, -jnp.log(2.0), atol=1e-6)

    def test_transformation_and_logdet(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        coef = OnionCoefParam(knots)

        dist = TransformationDist(knots=knots.knots, coef=coef.value)

        y = jnp.linspace(-2.0, 2.0, 50)
        z, logdet = dist.transformation_and_logdet(y)

        assert jnp.allclose(y, z, atol=1e-5)
        assert jnp.allclose(logdet, 0.0, atol=1e-5)

    def test_log_prob(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        coef = OnionCoefParam(knots)

        dist = TransformationDist(knots=knots.knots, coef=coef.value)

        y = jnp.linspace(-2.0, 2.0, 50)
        lp = dist.log_prob(y)
        lp2 = tfd.Normal(loc=0.0, scale=1.0).log_prob(y)

        assert jnp.allclose(lp, lp2, atol=1e-5)

    def test_cdf(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        coef = OnionCoefParam(knots)

        dist = TransformationDist(knots=knots.knots, coef=coef.value)

        y = jnp.linspace(-2.0, 2.0, 50)
        p = dist.cdf(y)
        p2 = tfd.Normal(loc=0.0, scale=1.0).cdf(y)

        assert jnp.allclose(p, p2, atol=1e-5)

    def test_quantile(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        coef = OnionCoefParam(knots)

        dist = TransformationDist(knots=knots.knots, coef=coef.value)

        u = jnp.linspace(0.01, 0.99, 10)

        y = dist.quantile(u)
        y2 = tfd.Normal(loc=0.0, scale=1.0).quantile(u)

        assert jnp.allclose(y, y2, atol=1e-5)

    def test_sample(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        coef = OnionCoefParam(knots)

        dist = TransformationDist(knots=knots.knots, coef=coef.value)

        y = dist.sample(10, seed=key)
        y2 = tfd.Normal(loc=0.0, scale=1.0).sample(10, seed=key)

        assert jnp.allclose(y, y2, atol=1e-5)

    def test_sample_batched(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        log_increments = 0.1 * jrd.uniform(key, shape=(4, 30, knots.nparam_full_domain))
        coef = jnp.cumsum(jnp.exp(log_increments), axis=0)
        dist = TransformationDist(knots=knots.knots, coef=coef)
        with jax.disable_jit(disable=False):
            y = dist.sample(12, seed=key)

        assert y.shape == (12, 4, 30)

    def test_mean(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        coef = OnionCoefParam(knots)

        dist = TransformationDist(knots=knots.knots, coef=coef.value)
        assert dist.mean() == pytest.approx(0.0)

    def test_variance(self) -> None:
        knots = bsplines.OnionKnots(-3.0, 3.0, nparam=10)
        coef = OnionCoefParam(knots)

        dist = TransformationDist(knots=knots.knots, coef=coef.value)
        assert dist.variance() == pytest.approx(1.0)
