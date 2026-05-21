import pytest

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm.bspline.onion import OnionKnots, OnionSpline
from liesel_ptm.bspline.ptm import PTMKnots, PTMSpline


knots = PTMKnots(-4.0, 4.0, nparam=10)
coef = jax.random.normal(jax.random.key(1), (1, knots.nparam))
bs = PTMSpline(knots.knots)


def normal_with_batch(batch_shape):
    return tfd.Normal(loc=jnp.zeros(batch_shape), scale=jnp.ones(batch_shape))


def assert_same_shape_as_normal(dist, normal, method_name: str, value):
    dist_value = getattr(dist, method_name)(value)
    normal_value = getattr(normal, method_name)(value)
    assert dist_value.shape == normal_value.shape
    assert jnp.all(jnp.isfinite(dist_value))


def assert_both_raise_for_shape(dist, normal, method_name: str, value):
    with pytest.raises((TypeError, ValueError)):
        getattr(normal, method_name)(value)

    with pytest.raises((TypeError, ValueError)):
        getattr(dist, method_name)(value)


class TestDistOneCoef:
    def test_log_prob_sample_layout(self):
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )
        normal = normal_with_batch(())

        values = (
            1.0,
            jax.random.normal(jax.random.key(0), (30,)),
            jax.random.normal(jax.random.key(1), (4, 30)),
        )

        for value in values:
            assert_same_shape_as_normal(dist, normal, "log_prob", value)

    def test_quantile_and_sample_layout(self):
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )

        assert dist.quantile(0.5).shape == ()
        assert dist.quantile(jnp.full((30,), 0.5)).shape == (30,)
        assert dist.sample(seed=jax.random.key(1)).shape == ()
        assert dist.sample(2, seed=jax.random.key(1)).shape == (2,)
        assert dist.sample((2, 3), seed=jax.random.key(1)).shape == (2, 3)


class TestDistBatchedCoef:
    def test_log_prob_matches_normal_broadcasting(self):
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )
        normal = normal_with_batch((2,))

        values = (
            1.0,
            jnp.ones((2,)),
            jnp.ones((5, 1)),
            jnp.ones((5, 2)),
        )

        for value in values:
            assert_same_shape_as_normal(dist, normal, "log_prob", value)

        assert_both_raise_for_shape(dist, normal, "log_prob", jnp.ones((5,)))

    def test_quantile_matches_normal_broadcasting(self):
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )
        normal = normal_with_batch((2,))

        probs = (
            0.5,
            jnp.full((2,), 0.5),
            jnp.full((5, 1), 0.5),
            jnp.full((5, 2), 0.5),
        )

        for prob in probs:
            assert_same_shape_as_normal(dist, normal, "quantile", prob)

        assert_both_raise_for_shape(dist, normal, "quantile", jnp.full((5,), 0.5))

    def test_cdf_quantile_roundtrip_broadcasting(self):
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )

        probs = (
            jnp.array(0.5),
            jnp.full((5, 1), 0.2),
            jnp.linspace(0.1, 0.9, 10).reshape(5, 2),
        )

        for prob in probs:
            q = dist.quantile(prob)
            cdf = dist.cdf(q)
            assert cdf.shape == q.shape
            assert jnp.allclose(cdf, jnp.broadcast_to(prob, q.shape), atol=1e-4)

    def test_sample_and_moment_shapes(self):
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )

        assert dist.sample(seed=jax.random.key(1)).shape == (2,)
        assert dist.sample(2, seed=jax.random.key(1)).shape == (2, 2)
        assert dist.sample((2, 3), seed=jax.random.key(1)).shape == (2, 3, 2)
        assert dist.mean().shape == (2,)
        assert dist.stddev().shape == (2,)

    def test_parametric_and_coef_batch_broadcasting(self):
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef,
            loc=jnp.array([0.0, 1.0]),
            scale=jnp.array([1.0, 2.0]),
            bspline=bs,
            centered=True,
            scaled=True,
        )

        assert dist.batch_shape == (2,)
        assert dist.log_prob(jnp.ones((5, 1))).shape == (5, 2)
        assert dist.mean().shape == (2,)
        assert dist.stddev().shape == (2,)

    def test_singleton_batch_axis_expands_like_normal(self):
        coef = jax.random.normal(jax.random.key(1), (2, 1, 1, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef,
            loc=jnp.zeros((2, 1)),
            scale=jnp.ones((2, 1)),
            bspline=bs,
        )
        normal = normal_with_batch((2, 1))

        assert_same_shape_as_normal(dist, normal, "log_prob", jnp.ones((5,)))
        assert_same_shape_as_normal(dist, normal, "quantile", jnp.full((5,), 0.5))
        assert dist.sample(3, seed=jax.random.key(1)).shape == (3, 2, 1)
        assert dist.mean().shape == (2, 1)
        assert dist.stddev().shape == (2, 1)

    def test_two_dimensional_batch_shape(self):
        coef = jax.random.normal(jax.random.key(1), (3, 2, 1, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )
        normal = normal_with_batch((3, 2))

        assert dist.batch_shape == (3, 2)
        assert_same_shape_as_normal(dist, normal, "log_prob", 1.0)
        assert_same_shape_as_normal(dist, normal, "log_prob", jnp.ones((5, 1, 1)))
        assert dist.quantile(0.5).shape == (3, 2)
        assert dist.sample((2, 3), seed=jax.random.key(1)).shape == (2, 3, 3, 2)
        assert dist.mean().shape == (3, 2)
        assert dist.stddev().shape == (3, 2)


class TestDistGPTM:
    def test_rowwise_batching_matches_tfp_layout(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots, subscripts="...nj,...nj->...n")
        n = 17
        k = jax.random.key(1)

        coef = jax.random.normal(k, (n, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=False
        )

        x = jax.random.normal(k, (n,))
        assert dist.batch_shape == (n,)
        assert dist.log_prob(x).shape == (n,)
        assert dist.log_prob(jnp.ones((5, 1))).shape == (5, n)
        assert dist.inverse_transformation(x).shape == (n,)

        b = 13
        coef = jax.random.normal(k, (b, n, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=False
        )

        assert dist.batch_shape == (b, n)
        assert dist.log_prob(x).shape == (b, n)
        assert dist.log_prob(jnp.ones((5, 1, 1))).shape == (5, b, n)
        assert dist.inverse_transformation(x).shape == (b, n)
        assert dist.sample(seed=jax.random.key(1)).shape == (b, n)
        assert dist.sample(2, seed=jax.random.key(1)).shape == (2, b, n)
        assert dist.sample((2, 3), seed=jax.random.key(1)).shape == (2, 3, b, n)
        assert dist.quantile(0.1).shape == (b, n)
        assert dist.quantile(jnp.full((2, b, n), fill_value=0.1)).shape == (2, b, n)

    def test_rowwise_cdf_quantile_roundtrip(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots, subscripts="...nj,...nj->...n")
        n = 7
        b = 3
        coef = jax.random.normal(jax.random.key(1), (b, n, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=False
        )

        prob = jnp.linspace(0.1, 0.9, b * n).reshape(b, n)
        q = dist.quantile(prob)
        cdf = dist.cdf(q)

        assert q.shape == (b, n)
        assert cdf.shape == (b, n)
        assert jnp.allclose(cdf, prob, atol=1e-4)
