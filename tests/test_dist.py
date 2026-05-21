import pytest

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm.bspline.onion import OnionKnots, OnionSpline
from liesel_ptm.bspline.ptm import PTMKnots, PTMSpline
from liesel_ptm.dist import (
    GaussianPseudoTransformationDist,
    LocScalePseudoTransformationDist,
    PseudoTransformationDist,
)


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


class TestBaseTransformationDist:
    def test_base_distribution_without_parametric_layer(self):
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
        dist = ptm.TransformationDist(
            coef=coef, bspline=bs, simpson_integration_n=8
        )

        assert dist.batch_shape == (2,)
        assert dist.event_shape == ()
        assert dist.log_prob(1.0).shape == (2,)
        assert dist.cdf(jnp.ones((5, 1))).shape == (5, 2)
        assert dist.quantile(0.5).shape == (2,)
        assert dist.sample(seed=jax.random.key(1)).shape == (2,)
        assert dist.sample((2, 3), seed=jax.random.key(1)).shape == (2, 3, 2)
        assert dist.mean().shape == (2,)
        assert dist.stddev().shape == (2,)

    def test_spline_only_public_methods_use_tfp_layout(self):
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
        dist = ptm.TransformationDist(coef=coef, bspline=bs)

        value = jnp.ones((5, 1))
        log_prob = dist.log_prob_spline(value)
        prob = dist.prob_spline(value)
        cdf = dist.cdf_spline(value)

        assert log_prob.shape == (5, 2)
        assert prob.shape == (5, 2)
        assert cdf.shape == (5, 2)
        assert jnp.allclose(prob, jnp.exp(log_prob))

        probs = jnp.linspace(0.1, 0.9, 10).reshape(5, 2)
        quantiles = dist.quantile_spline(probs)
        assert quantiles.shape == (5, 2)
        assert jnp.allclose(dist.cdf_spline(quantiles), probs, atol=1e-4)

    def test_coefficient_validation(self):
        coef_list = [[0.0] * knots.nparam]
        dist = ptm.TransformationDist(coef=coef_list, bspline=bs)

        assert jnp.issubdtype(dist.coef.dtype, jnp.floating)
        assert dist.log_prob(0.0).shape == ()

        with pytest.raises(TypeError, match="floating-point"):
            ptm.TransformationDist(
                coef=jnp.ones((1, knots.nparam), dtype=jnp.int32), bspline=bs
            )

    def test_simpson_integration_n_api(self):
        default_dist = ptm.TransformationDist(coef=coef, bspline=bs)
        custom_dist = ptm.TransformationDist(
            coef=coef, bspline=bs, simpson_integration_n=8
        )

        assert default_dist.simpson_integration_n == 32
        assert custom_dist.simpson_integration_n == 8
        assert jnp.isfinite(custom_dist.mean())

        for invalid in (0, -2, 3):
            with pytest.raises(ValueError, match="positive even"):
                ptm.TransformationDist(
                    coef=coef, bspline=bs, simpson_integration_n=invalid
                )

    def test_extreme_probabilities_remain_finite(self):
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs
        )
        probs = jnp.array([1e-6, 1.0 - 1e-6]).reshape(2, 1)

        quantiles = dist.quantile(probs)
        samples = dist.sample(4, seed=jax.random.key(1))

        assert jnp.all(jnp.isfinite(quantiles))
        assert jnp.all(jnp.isfinite(dist.cdf(quantiles)))
        assert jnp.all(jnp.isfinite(dist.log_prob(quantiles)))
        assert jnp.all(jnp.isfinite(samples))

    def test_log_prob_autodiff_wrt_coefficients(self):
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))

        def objective(coef):
            dist = ptm.LocScaleTransformationDist(
                coef=coef,
                loc=0.0,
                scale=1.0,
                bspline=bs,
                simpson_integration_n=8,
            )
            return jnp.sum(dist.log_prob(jnp.ones((5, 1))))

        grad = jax.grad(objective)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))


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
            simpson_integration_n=8,
        )

        assert dist.simpson_integration_n == 8
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


class TestPseudoDistributions:
    def test_gaussian_pseudo_matches_normal(self):
        loc = jnp.array([0.0, 1.0])
        scale = jnp.array([1.0, 2.0])
        dist = GaussianPseudoTransformationDist(
            coef=jnp.array([[0.0]]),
            loc=loc,
            scale=scale,
            simpson_integration_n=8,
        )
        normal = tfd.Normal(loc=loc, scale=scale)

        assert dist.simpson_integration_n == 8
        value = jnp.ones((5, 1))
        probs = jnp.full((5, 1), 0.25)

        assert jnp.allclose(dist.log_prob(value), normal.log_prob(value))
        assert jnp.allclose(dist.cdf(value), normal.cdf(value))
        assert jnp.allclose(dist.quantile(probs), normal.quantile(probs))
        assert jnp.allclose(dist.mean(), normal.mean())
        assert jnp.allclose(dist.stddev(), normal.stddev())
        assert dist.sample(3, seed=jax.random.key(1)).shape == (3, 2)

    def test_pseudo_transformation_dist_matches_parametric_distribution(self):
        rate = jnp.array([1.0, 2.0])
        dist = PseudoTransformationDist(
            coef=jnp.array([[0.0]]),
            parametric_distribution=tfd.Exponential,
            rate=rate,
        )
        expected = tfd.Exponential(rate=rate)

        value = jnp.full((5, 1), 0.75)
        probs = jnp.full((5, 1), 0.25)

        assert jnp.allclose(dist.log_prob(value), expected.log_prob(value))
        assert jnp.allclose(dist.cdf(value), expected.cdf(value))
        assert jnp.allclose(dist.quantile(probs), expected.quantile(probs))

    def test_loc_scale_pseudo_matches_normal_when_parametric_is_normal(self):
        loc = jnp.array([0.0, 1.0])
        scale = jnp.array([1.0, 2.0])
        dist = LocScalePseudoTransformationDist(
            coef=jnp.array([[0.0]]),
            loc=loc,
            scale=scale,
            parametric_distribution=tfd.Normal,
        )
        normal = tfd.Normal(loc=loc, scale=scale)

        value = jnp.ones((5, 1))
        probs = jnp.full((5, 1), 0.75)

        assert jnp.allclose(dist.log_prob(value), normal.log_prob(value))
        assert jnp.allclose(dist.cdf(value), normal.cdf(value))
        assert jnp.allclose(dist.quantile(probs), normal.quantile(probs))
        assert jnp.allclose(dist.mean(), normal.mean())
        assert jnp.allclose(dist.stddev(), normal.stddev())


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
