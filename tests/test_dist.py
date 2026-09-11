import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm.bspline.onion import OnionKnots, OnionSpline
from liesel_ptm.bspline.ptm import PTMKnots, PTMSpline
from liesel_ptm.dist import (
    GaussianPseudoTransformationDist,
    LocScalePseudoTransformationDist,
    PseudoTransformationDist,
    _gauss_legendre_nodes_and_weights,
    integrate_piecewise_gauss_legendre,
)

knots = PTMKnots(-4.0, 4.0, nparam=10)
coef = jax.random.normal(jax.random.key(1), (knots.nparam,))
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


def random_walk_coef(key, batch_shape: tuple[int, ...], nparam: int, scale=0.35):
    innovations = jax.random.normal(key, batch_shape + (nparam,)) * scale
    coef = jnp.cumsum(innovations, axis=-1)
    coef = coef - jnp.mean(coef, axis=-1, keepdims=True)
    return coef


class TestPiecewiseGaussLegendreIntegration:
    def test_integrates_polynomial_over_multiple_pieces_with_batching(self):
        nodes, weights = _gauss_legendre_nodes_and_weights(3, jnp.float32)
        breaks = jnp.array([-2.0, -1.0, 0.5, 3.0])
        scales = jnp.array([1.0, 2.0])

        def antiderivative(x):
            return x**5 / 5 + 2 * x**3 / 3 + x

        def fn(x):
            return (x**4 + 2 * x**2 + 1) * scales

        value = integrate_piecewise_gauss_legendre(
            fn, breaks=breaks, nodes=nodes, weights=weights, batch_ndims=1
        )
        expected = (antiderivative(breaks[-1]) - antiderivative(breaks[0])) * scales

        assert value.shape == (2,)
        assert jnp.allclose(value, expected, atol=1e-5)

    def test_distribution_builds_default_and_custom_breaks_from_knots(self):
        default_dist = ptm.TransformationDist(coef=coef, bspline=bs)
        inner_dist = ptm.TransformationDist(
            coef=coef, bspline=bs, integration_bounds=(-2.5, 2.5)
        )
        outer_dist = ptm.TransformationDist(
            coef=coef, bspline=bs, integration_bounds=(-8.0, 8.0)
        )

        assert default_dist.integration_bounds == (
            float(knots.knots[0]),
            float(knots.knots[-1]),
        )
        assert jnp.allclose(default_dist.integration_breaks, knots.knots)
        assert float(inner_dist.integration_breaks[0]) == pytest.approx(-2.5)
        assert float(inner_dist.integration_breaks[-1]) == pytest.approx(2.5)
        assert float(outer_dist.integration_breaks[0]) == pytest.approx(-8.0)
        assert float(outer_dist.integration_breaks[-1]) == pytest.approx(8.0)

        assert jnp.isfinite(inner_dist.mean())
        assert jnp.isfinite(outer_dist.stddev())


class TestOnionDistFactory:
    def test_loc_scale_factory_builds_and_reuses_spline(self):
        Dist = ptm.onion_dist(
            a=-4.0,
            b=4.0,
            nparam=11,
            centered=True,
            scaled=True,
            gauss_legendre_order=4,
        )
        bspline = getattr(Dist, "keywords")["bspline"]
        coef = jax.random.normal(jax.random.key(1), (11,))

        dist1 = Dist(coef=coef, loc=0.0, scale=1.0)
        dist2 = Dist(coef=coef, loc=1.0, scale=2.0)

        assert isinstance(bspline, OnionSpline)
        assert isinstance(dist1, ptm.LocScaleTransformationDist)
        assert dist1.bspline is bspline
        assert dist2.bspline is bspline
        assert dist1.centered
        assert dist1.scaled
        assert dist1.gauss_legendre_order == 4
        assert dist1.log_prob(0.0).shape == ()
        assert jnp.isfinite(dist1.log_prob(0.0))

    def test_factory_can_build_base_transformation_dist_with_custom_knots(self):
        onion_knots = OnionKnots(-3.0, 3.0, nparam=11)
        Dist = ptm.onion_dist(
            knots=onion_knots,
            loc_scale=False,
            parametric_distribution=tfd.Exponential,
        )
        coef = jax.random.normal(jax.random.key(2), (onion_knots.nparam,))
        dist = Dist(coef=coef, rate=jnp.array([1.0, 2.0]))

        assert isinstance(dist, ptm.TransformationDist)
        assert not isinstance(dist, ptm.LocScaleTransformationDist)
        assert dist.bspline is getattr(Dist, "keywords")["bspline"]
        assert jnp.allclose(dist.bspline.knots, onion_knots.knots)
        assert dist.log_prob(jnp.ones((5, 1))).shape == (5, 2)

    def test_factory_supports_rowwise_onion_batching(self):
        n = 7
        Dist = ptm.onion_dist(a=-4.0, b=4.0, nparam=11, gauss_legendre_order=4)
        coef = jax.random.normal(jax.random.key(3), (n, 11))
        dist = Dist(coef=coef, loc=0.0, scale=1.0)

        assert dist.bspline is getattr(Dist, "keywords")["bspline"]
        assert dist.batch_shape == (n,)
        assert dist.log_prob(jnp.ones((5, 1))).shape == (5, n)
        assert dist.sample(2, seed=jax.random.key(1)).shape == (2, n)


class TestBaseTransformationDist:
    def test_base_distribution_without_parametric_layer(self):
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
        dist = ptm.TransformationDist(coef=coef, bspline=bs, gauss_legendre_order=8)

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
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
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
        coef_list = [0.0] * knots.nparam
        dist = ptm.TransformationDist(coef=coef_list, bspline=bs)

        assert jnp.issubdtype(dist.coef.dtype, jnp.floating)
        assert dist.log_prob(0.0).shape == ()

        with pytest.raises(TypeError, match="floating-point"):
            ptm.TransformationDist(
                coef=jnp.ones((knots.nparam,), dtype=jnp.int32), bspline=bs
            )

    def test_gauss_legendre_integration_api(self):
        default_dist = ptm.TransformationDist(coef=coef, bspline=bs)
        custom_dist = ptm.TransformationDist(
            coef=coef, bspline=bs, gauss_legendre_order=12
        )

        assert default_dist.gauss_legendre_order == 8
        assert custom_dist.gauss_legendre_order == 12
        assert jnp.isfinite(custom_dist.mean())

        for invalid in (0, -2):
            with pytest.raises(ValueError, match="positive integer"):
                ptm.TransformationDist(
                    coef=coef, bspline=bs, gauss_legendre_order=invalid
                )

        with pytest.raises(TypeError, match="positive integer"):
            invalid_order: Any = 2.5
            ptm.TransformationDist(
                coef=coef, bspline=bs, gauss_legendre_order=invalid_order
            )

    def test_invalid_integration_bounds_raise(self):
        invalid_bounds: Any = (
            (1.0, 1.0),
            (2.0, 1.0),
            (jnp.nan, 1.0),
            (0.0, 1.0, 2.0),
        )
        for invalid in invalid_bounds:
            with pytest.raises(ValueError, match="integration"):
                ptm.TransformationDist(
                    coef=coef, bspline=bs, integration_bounds=invalid
                )

    def test_extreme_probabilities_remain_finite(self):
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
        dist = ptm.LocScaleTransformationDist(coef=coef, loc=0.0, scale=1.0, bspline=bs)
        probs = jnp.array([1e-6, 1.0 - 1e-6]).reshape(2, 1)

        quantiles = dist.quantile(probs)
        samples = dist.sample(4, seed=jax.random.key(1))

        assert jnp.all(jnp.isfinite(quantiles))
        assert jnp.all(jnp.isfinite(dist.cdf(quantiles)))
        assert jnp.all(jnp.isfinite(dist.log_prob(quantiles)))
        assert jnp.all(jnp.isfinite(samples))

    def test_log_cdf_and_log_survival_function_are_stable_in_tails(self):
        dist = ptm.LocScaleTransformationDist(coef=coef, loc=0.0, scale=1.0, bspline=bs)
        values = jnp.array([-20.0, 20.0])
        z, _ = dist.transformation_and_logdet(values)

        assert dist.cdf(values[1]) == 1.0
        assert dist.survival_function(values[1]) == 0.0
        assert jnp.allclose(
            dist.log_cdf(values),
            dist.reference_distribution.log_cdf(z),
            rtol=1e-5,
            atol=1e-5,
        )
        assert jnp.allclose(
            dist.log_survival_function(values),
            dist.reference_distribution.log_survival_function(z),
            rtol=1e-5,
            atol=1e-5,
        )
        assert jnp.all(jnp.isfinite(dist.log_cdf(values)))
        assert jnp.all(jnp.isfinite(dist.log_survival_function(values)))

    def test_cdf_and_survival_match_transformed_value_reference(self):
        coef = random_walk_coef(jax.random.key(2), (2, 1), knots.nparam)
        loc = jnp.linspace(-0.5, 0.5, 8).reshape((2, 4))
        scale = jnp.ones((2, 1))
        dist = ptm.LocScaleTransformationDist(
            coef=coef,
            loc=loc,
            scale=scale,
            bspline=bs,
            centered=True,
            scaled=True,
            gauss_legendre_order=4,
        )
        value = jnp.linspace(-1.5, 1.5, 4).reshape((1, 4))
        z, _ = dist.transformation_and_logdet(value)

        assert jnp.allclose(
            dist.cdf(value),
            dist.reference_distribution.cdf(z),
            rtol=2e-5,
            atol=2e-5,
        )
        assert jnp.allclose(
            dist.log_cdf(value),
            dist.reference_distribution.log_cdf(z),
            rtol=2e-5,
            atol=2e-5,
        )
        assert jnp.allclose(
            dist.survival_function(value),
            dist.reference_distribution.survival_function(z),
            rtol=2e-5,
            atol=2e-5,
        )
        assert jnp.allclose(
            dist.log_survival_function(value),
            dist.reference_distribution.log_survival_function(z),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_log_prob_autodiff_wrt_coefficients(self):
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))

        def objective(coef):
            dist = ptm.LocScaleTransformationDist(
                coef=coef,
                loc=0.0,
                scale=1.0,
                bspline=bs,
                gauss_legendre_order=8,
            )
            return jnp.sum(dist.log_prob(jnp.ones((5, 1))))

        grad = jax.grad(objective)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))


class TestSplineMomentQuadrature:
    def test_random_walk_coefficients_match_higher_order_reference(self):
        knots = PTMKnots(-4.0, 4.0, nparam=20)
        bs = PTMSpline(knots.knots)
        coef = random_walk_coef(jax.random.key(11), (16,), knots.nparam)

        dist = ptm.TransformationDist(coef=coef, bspline=bs, gauss_legendre_order=8)
        reference = ptm.TransformationDist(
            coef=coef, bspline=bs, gauss_legendre_order=32
        )

        mean = dist.transformation_spline_mean()
        reference_mean = reference.transformation_spline_mean()
        variance = dist.transformation_spline_variance(mean=mean)
        reference_variance = reference.transformation_spline_variance(
            mean=reference_mean
        )

        mean_abs = jnp.max(jnp.abs(mean - reference_mean))
        variance_abs = jnp.max(jnp.abs(variance - reference_variance))
        mean_rel = jnp.max(
            jnp.abs(mean - reference_mean) / jnp.maximum(jnp.abs(reference_mean), 1e-5)
        )
        variance_rel = jnp.max(
            jnp.abs(variance - reference_variance)
            / jnp.maximum(jnp.abs(reference_variance), 1e-5)
        )

        assert mean.shape == (16,)
        assert variance.shape == (16,)
        assert mean_abs < 5e-4
        assert variance_abs < 5e-4
        assert mean_rel < 5e-4
        assert variance_rel < 5e-4

    def test_moment_quadrature_diagnostic_reports_batch_errors(self):
        knots = PTMKnots(-4.0, 4.0, nparam=20)
        bs = PTMSpline(knots.knots)
        coef = random_walk_coef(jax.random.key(12), (8,), knots.nparam)
        dist = ptm.TransformationDist(coef=coef, bspline=bs, gauss_legendre_order=8)

        diagnostic = dist.moment_quadrature_diagnostic(
            reference_order=24, rtol=5e-4, atol=5e-5
        )

        assert set(diagnostic) == {
            "mean",
            "variance",
            "mean_reference",
            "variance_reference",
            "mean_abs_error",
            "variance_abs_error",
            "mean_rel_error",
            "variance_rel_error",
            "ok",
        }
        assert diagnostic["mean_abs_error"].shape == (8,)
        assert diagnostic["variance_abs_error"].shape == (8,)
        assert bool(diagnostic["ok"])

    def test_pseudo_diagnostic_is_exact_identity(self):
        dist = GaussianPseudoTransformationDist(
            coef=jnp.array([0.0]), loc=0.0, scale=1.0
        )

        diagnostic = dist.moment_quadrature_diagnostic()

        assert diagnostic["mean"] == 0.0
        assert diagnostic["variance"] == 1.0
        assert bool(diagnostic["ok"])

    def test_autodiff_through_centered_scaled_moments(self):
        knots = PTMKnots(-4.0, 4.0, nparam=14)
        bs = PTMSpline(knots.knots)
        coef = random_walk_coef(jax.random.key(13), (3,), knots.nparam)

        def objective(coef):
            dist = ptm.LocScaleTransformationDist(
                coef=coef,
                loc=0.0,
                scale=1.0,
                bspline=bs,
                centered=True,
                scaled=True,
                gauss_legendre_order=8,
            )
            return jnp.sum(dist.log_prob(jnp.ones((5, 1))))

        grad = jax.grad(objective)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_piecewise_gl_moment_speed_smoke(self):
        knots = PTMKnots(-4.0, 4.0, nparam=20)
        bs = PTMSpline(knots.knots)
        coef = random_walk_coef(jax.random.key(14), (16,), knots.nparam)
        dist = ptm.TransformationDist(coef=coef, bspline=bs, gauss_legendre_order=8)

        n_eval = (dist.integration_breaks.size - 1) * dist.gauss_legendre_order
        assert n_eval <= 200

        @jax.jit
        def moments(coef):
            dist = ptm.TransformationDist(coef=coef, bspline=bs, gauss_legendre_order=8)
            mean = dist.transformation_spline_mean()
            variance = dist.transformation_spline_variance(mean=mean)
            return mean, variance

        warm_mean, warm_variance = moments(coef)
        warm_mean.block_until_ready()
        warm_variance.block_until_ready()

        start = time.perf_counter()
        mean, variance = moments(coef)
        mean.block_until_ready()
        variance.block_until_ready()
        elapsed = time.perf_counter() - start

        assert mean.shape == (16,)
        assert variance.shape == (16,)
        assert elapsed < 2.0

    def test_centered_scaled_moments_use_coef_batch_shape(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        bs = PTMSpline(knots.knots)
        coef = random_walk_coef(jax.random.key(15), (2, 3, 1), knots.nparam)
        loc = jnp.zeros((2, 3, 5))
        scale = jnp.ones((2, 3, 1))
        dist = ptm.LocScaleTransformationDist(
            coef=coef,
            loc=loc,
            scale=scale,
            bspline=bs,
            centered=True,
            scaled=True,
            gauss_legendre_order=4,
        )

        mean = dist.transformation_spline_mean()
        variance = dist.transformation_spline_variance(mean=mean)
        log_survival = dist.log_survival_function(jnp.ones((1, 1, 5)))

        assert dist.batch_shape == (2, 3, 5)
        assert mean.shape == (2, 3, 1)
        assert variance.shape == (2, 3, 1)
        assert dist.mean().shape == (2, 3, 5)
        assert dist.stddev().shape == (2, 3, 5)
        assert log_survival.shape == (2, 3, 5)
        assert jnp.all(jnp.isfinite(log_survival))


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
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
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
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
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
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
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
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=True
        )

        assert dist.sample(seed=jax.random.key(1)).shape == (2,)
        assert dist.sample(2, seed=jax.random.key(1)).shape == (2, 2)
        assert dist.sample((2, 3), seed=jax.random.key(1)).shape == (2, 3, 2)
        assert dist.mean().shape == (2,)
        assert dist.stddev().shape == (2,)

    def test_parametric_and_coef_batch_broadcasting(self):
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef,
            loc=jnp.array([0.0, 1.0]),
            scale=jnp.array([1.0, 2.0]),
            bspline=bs,
            centered=True,
            scaled=True,
            gauss_legendre_order=8,
        )

        assert dist.gauss_legendre_order == 8
        assert dist.batch_shape == (2,)
        assert dist.log_prob(jnp.ones((5, 1))).shape == (5, 2)
        assert dist.mean().shape == (2,)
        assert dist.stddev().shape == (2,)

    def test_unbatched_coef_broadcasts_over_parametric_batch(self):
        loc = jnp.array([0.0, 1.0])
        scale = jnp.array(1.0)
        dist = ptm.LocScaleTransformationDist(
            coef=coef,
            loc=loc,
            scale=scale,
            bspline=bs,
            gauss_legendre_order=8,
        )
        normal = tfd.Normal(loc=loc, scale=scale)

        assert dist.batch_shape == normal.batch_shape
        assert_same_shape_as_normal(dist, normal, "log_prob", 1.0)
        assert_same_shape_as_normal(dist, normal, "log_prob", jnp.ones((5, 1)))
        assert dist.sample(3, seed=jax.random.key(1)).shape == (3, 2)

    def test_singleton_batch_axis_expands_like_normal(self):
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
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
        coef = jax.random.normal(jax.random.key(1), (3, 2, knots.nparam))
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
    def test_spline_only_identity_methods_match_tfp_batch_broadcasting(self):
        batch_shape = (2, 3)
        loc = jnp.zeros(batch_shape)
        scale = jnp.ones(batch_shape)
        rate = jnp.ones(batch_shape)
        distributions = (
            (
                GaussianPseudoTransformationDist(
                    coef=jnp.array([0.0]), loc=loc, scale=scale
                ),
                tfd.Normal(loc=loc, scale=scale),
            ),
            (
                PseudoTransformationDist(
                    coef=jnp.array([0.0]),
                    parametric_distribution=tfd.Exponential,
                    rate=rate,
                ),
                tfd.Exponential(rate=rate),
            ),
            (
                LocScalePseudoTransformationDist(
                    coef=jnp.array([0.0]),
                    loc=loc,
                    scale=scale,
                    parametric_distribution=tfd.Normal,
                ),
                tfd.Normal(loc=loc, scale=scale),
            ),
        )

        for dist, reference in distributions:
            for value in (jnp.array(0.75), jnp.full((5, 1, 1), 0.75)):
                expected_shape = reference.log_prob(value).shape
                expected_value = jnp.broadcast_to(value, expected_shape)

                transformed, logdet = dist.transformation_and_logdet_spline(value)
                assert transformed.shape == expected_shape
                assert logdet.shape == expected_shape
                assert jnp.array_equal(transformed, expected_value)
                assert jnp.array_equal(logdet, jnp.zeros(expected_shape))
                assert jnp.array_equal(
                    dist._transformation_spline(value), expected_value
                )
                assert jnp.array_equal(
                    dist.inverse_transformation_spline(value), expected_value
                )

            bad_value = jnp.ones((5, 2))
            with pytest.raises((TypeError, ValueError)):
                reference.log_prob(bad_value)
            with pytest.raises((TypeError, ValueError)):
                dist.transformation_and_logdet_spline(bad_value)

    def test_gaussian_pseudo_matches_normal(self):
        loc = jnp.array([0.0, 1.0])
        scale = jnp.array([1.0, 2.0])
        dist = GaussianPseudoTransformationDist(
            coef=jnp.array([0.0]),
            loc=loc,
            scale=scale,
            gauss_legendre_order=8,
        )
        normal = tfd.Normal(loc=loc, scale=scale)

        assert dist.gauss_legendre_order == 8
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
            coef=jnp.array([0.0]),
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
            coef=jnp.array([0.0]),
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
    def test_shared_onion_batching_matches_tfp_layout(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        b = 3
        coef = jax.random.normal(jax.random.key(1), (b, knots.nparam))
        dist = ptm.LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bs, batched=False
        )

        assert dist.batch_shape == (b,)
        assert dist.log_prob(1.0).shape == (b,)
        assert dist.log_prob(jnp.ones((5, 1))).shape == (5, b)
        assert dist.quantile(jnp.full((5, 1), fill_value=0.1)).shape == (5, b)
        assert dist.sample(2, seed=jax.random.key(1)).shape == (2, b)

    def test_rowwise_batching_matches_tfp_layout(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
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
        bs = OnionSpline(knots.knots)
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


class TestResponseMoments:
    @staticmethod
    def skewt(**kwargs):
        options = {
            "coef": jnp.linspace(-0.2, 0.2, 10),
            "loc": 1.0,
            "scale": 2.0,
            "skewness": 0.7,
            "df": 20.0,
        }
        options.update(kwargs)
        return ptm.onion_dist(
            a=-6.0,
            b=6.0,
            nparam=10,
            loc_scale=False,
            parametric_distribution=tfd.TwoPieceStudentT,
        )(**options)

    @staticmethod
    def reference_moments(dist):
        # Independent NumPy reduction of higher-resolution public quantiles.
        from scipy.special import roots_legendre

        nodes, weights = roots_legendre(2048)
        probabilities = jnp.asarray((nodes + 1) / 2, dtype=dist.dtype)
        probabilities = probabilities.reshape((-1,) + (1,) * len(dist.batch_shape))
        values = np.asarray(dist.quantile(probabilities))
        weights = (weights / 2).reshape((-1,) + (1,) * len(dist.batch_shape))
        mean = np.sum(weights * values, axis=0)
        variance = np.sum(weights * (values - mean) ** 2, axis=0)
        return mean, variance

    @pytest.mark.parametrize(
        "centered,scaled", [(False, False), (True, False), (False, True), (True, True)]
    )
    def test_skewt_moments_match_complete_quantiles(self, centered, scaled):
        dist = self.skewt(
            centered=centered, scaled=scaled, integration_bounds=(-3.0, 3.0)
        )
        mean, variance = self.reference_moments(dist)
        np.testing.assert_allclose(dist.mean(), mean, rtol=5e-4, atol=1e-4)
        np.testing.assert_allclose(dist.variance(), variance, rtol=5e-4, atol=1e-4)
        np.testing.assert_allclose(dist.stddev(), np.sqrt(variance), rtol=5e-4)
        np.testing.assert_allclose(dist.stddev() ** 2, dist.variance(), rtol=1e-6)

    def test_affine_mean_includes_scale_and_full_tails(self):
        Dist = ptm.onion_dist(a=-6.0, b=6.0, nparam=10)
        options = {
            "coef": jnp.linspace(-0.3, 0.3, 10),
            "integration_bounds": (-1.0, 1.0),
        }
        standardized = Dist(loc=0.0, scale=1.0, **options)
        shifted = Dist(loc=3.0, scale=2.5, **options)
        expected_mean, expected_variance = self.reference_moments(standardized)
        np.testing.assert_allclose(shifted.mean(), 3 + 2.5 * expected_mean, atol=3e-4)
        np.testing.assert_allclose(
            shifted.variance(), 2.5**2 * expected_variance, rtol=3e-4
        )
        assert (
            abs(float(standardized.mean() - standardized.transformation_spline_mean()))
            > 0.1
        )

    def test_finite_bounds_only_affect_normalization(self):
        narrow = self.skewt(integration_bounds=(-1.0, 1.0))
        wide = self.skewt(integration_bounds=(-6.0, 6.0))
        np.testing.assert_array_equal(narrow.mean(), wide.mean())
        np.testing.assert_array_equal(narrow.variance(), wide.variance())
        assert not np.isclose(
            narrow.transformation_spline_mean(), wide.transformation_spline_mean()
        )

    def test_batched_moments_and_jitted_gradients(self):
        coef = jnp.broadcast_to(jnp.linspace(-0.2, 0.2, 10), (2, 1, 10))
        loc = jnp.array([[0.0, 1.0, 2.0]])

        Dist = ptm.onion_dist(
            a=-6.0,
            b=6.0,
            nparam=10,
            loc_scale=False,
            parametric_distribution=tfd.TwoPieceStudentT,
        )

        @jax.jit
        def objective(coef, loc, scale, skewness, df):
            dist = Dist(coef=coef, loc=loc, scale=scale, skewness=skewness, df=df)
            return jnp.sum(dist.mean() + dist.variance())

        parameters = (coef, loc, jnp.array(2.0), jnp.array(0.7), jnp.array(20.0))
        grads = jax.grad(objective, argnums=(0, 1, 2, 3, 4))(*parameters)
        for gradient, parameter in zip(grads, parameters, strict=True):
            assert gradient.shape == parameter.shape
            assert jnp.all(jnp.isfinite(gradient))
            assert jnp.linalg.norm(gradient) > 0
        dist = self.skewt(coef=coef, loc=loc)
        assert (
            dist.mean().shape == dist.variance().shape == dist.stddev().shape == (2, 3)
        )
        expected_mean, expected_variance = self.reference_moments(dist)
        np.testing.assert_allclose(dist.mean(), expected_mean, atol=1e-4)
        np.testing.assert_allclose(dist.variance(), expected_variance, rtol=5e-4)

    def test_response_diagnostic_detects_heavy_tail_error(self):
        dist = self.skewt(coef=jnp.zeros(10), df=3.0, response_moment_order=128)
        diagnostic = dist.moment_quadrature_diagnostic(target="response")
        np.testing.assert_allclose(diagnostic["mean"], dist.mean())
        np.testing.assert_allclose(diagnostic["variance"], dist.variance())
        assert not bool(diagnostic["ok"])
        assert diagnostic["variance_abs_error"] > 0.01
        with pytest.raises(ValueError, match="target"):
            dist.moment_quadrature_diagnostic(target="invalid")
        with pytest.raises(ValueError, match="reference_order"):
            dist.moment_quadrature_diagnostic(target="response", reference_order=128)

    @pytest.mark.parametrize(
        "order,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError)]
    )
    def test_invalid_response_order(self, order, error):
        with pytest.raises(error, match="response_moment_order"):
            self.skewt(response_moment_order=order)

    def test_response_rule_is_lazy_and_does_not_change_model(self, monkeypatch):
        from liesel_ptm import dist as dist_module

        orders = []
        original = dist_module._gauss_legendre_nodes_and_weights

        def record(order, dtype):
            orders.append(order)
            return original(order, dtype)

        monkeypatch.setattr(dist_module, "_gauss_legendre_nodes_and_weights", record)
        low = self.skewt(centered=True, scaled=True, response_moment_order=32)
        high = self.skewt(centered=True, scaled=True, response_moment_order=64)
        for method, argument in [("log_prob", 0.25), ("cdf", 0.25), ("quantile", 0.25)]:
            np.testing.assert_array_equal(
                getattr(low, method)(argument), getattr(high, method)(argument)
            )
        np.testing.assert_array_equal(
            low.transformation_spline_mean(), high.transformation_spline_mean()
        )
        np.testing.assert_array_equal(
            low.transformation_spline_variance(), high.transformation_spline_variance()
        )
        assert 32 not in orders and 64 not in orders
        assert np.isfinite(low.mean())
        assert 32 in orders

    def test_exact_pseudo_moments_and_nonfinite_diagnostic(self):
        distributions = [
            (
                GaussianPseudoTransformationDist(
                    coef=jnp.zeros(1), loc=3.0, scale=2.0, response_moment_order=16
                ),
                tfd.Normal(3.0, 2.0),
            ),
            (
                PseudoTransformationDist(
                    coef=jnp.zeros(1),
                    parametric_distribution=tfd.Exponential,
                    rate=jnp.array([1.0, 2.0]),
                    response_moment_order=16,
                ),
                tfd.Exponential(jnp.array([1.0, 2.0])),
            ),
            (
                PseudoTransformationDist(
                    coef=jnp.zeros(1),
                    reference_distribution=tfd.Normal(3.0, 2.0),
                    centered=True,
                    scaled=True,
                    response_moment_order=16,
                ),
                tfd.Normal(3.0, 2.0),
            ),
            (
                LocScalePseudoTransformationDist(
                    coef=jnp.zeros(1),
                    loc=3.0,
                    scale=2.0,
                    parametric_distribution=tfd.Normal,
                    response_moment_order=16,
                ),
                tfd.Normal(3.0, 2.0),
            ),
        ]
        for dist, expected in distributions:
            np.testing.assert_allclose(
                dist.mean(), jnp.broadcast_to(expected.mean(), dist.batch_shape)
            )
            np.testing.assert_allclose(
                dist.variance(), jnp.broadcast_to(expected.variance(), dist.batch_shape)
            )
            diagnostic = dist.moment_quadrature_diagnostic(target="response")
            np.testing.assert_allclose(diagnostic["mean"], dist.mean())
            assert bool(diagnostic["ok"])
            assert dist.response_moment_order == 16
        undefined = PseudoTransformationDist(
            coef=jnp.zeros(1),
            parametric_distribution=tfd.StudentT,
            df=0.5,
            loc=0.0,
            scale=1.0,
        )
        assert np.isnan(undefined.mean())
        assert np.isnan(undefined.variance())
        assert not bool(undefined.moment_quadrature_diagnostic(target="response")["ok"])

    def test_pseudo_preserves_underlying_stats_errors(self):
        class NoMean(tfd.Normal):
            def _mean(self):
                raise NotImplementedError("no mean")

        dist = PseudoTransformationDist(
            coef=jnp.zeros(1), parametric_distribution=NoMean, loc=0.0, scale=1.0
        )
        with pytest.raises(NotImplementedError, match="no mean"):
            dist.mean()

        np.testing.assert_allclose(dist.variance(), 1.0)

        class NoVariance(tfd.Normal):
            def _variance(self):
                raise NotImplementedError("no variance")

            def _stddev(self):
                raise NotImplementedError("no variance")

        dist = PseudoTransformationDist(
            coef=jnp.zeros(1), parametric_distribution=NoVariance, loc=3.0, scale=1.0
        )
        np.testing.assert_allclose(dist.mean(), 3.0)
        with pytest.raises(NotImplementedError, match="no variance"):
            dist.variance()
