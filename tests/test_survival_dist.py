import jax
import jax.numpy as jnp
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_ptm.bspline.ptm import PTMKnots, PTMSpline
from liesel_ptm.dist import LocScaleTransformationDist
from liesel_ptm.survival_dist import (
    CensoredDistribution,
    CensoredPTMDist,
    interval_censored,
    left_censored,
    right_censored,
    uncensored,
)


def assert_no_nan(value):
    assert not jnp.any(jnp.isnan(value))


class TestCensoredDistributionApi:
    def test_import_and_distribution_shapes(self):
        dist = CensoredDistribution(
            tfd.Normal, loc=jnp.zeros((2,)), scale=jnp.ones((2,))
        )

        assert dist.event_shape == (3,)
        assert dist.batch_shape == (2,)
        assert isinstance(dist.base_distribution, tfd.Normal)

    def test_record_helpers_broadcast_and_stack_event_axis(self):
        assert uncensored(jnp.ones((5, 1))).shape == (5, 1, 3)
        assert left_censored(jnp.ones((5, 1))).shape == (5, 1, 3)
        assert right_censored(jnp.ones((5, 1))).shape == (5, 1, 3)

        records = interval_censored(jnp.zeros((5, 1)), jnp.ones((1, 2)))

        assert records.shape == (5, 2, 3)
        assert jnp.all(jnp.isnan(records[..., 0]))
        assert jnp.all(records[..., 1] == 0.0)
        assert jnp.all(records[..., 2] == 1.0)

    def test_sample_returns_uncensored_records(self):
        dist = CensoredDistribution(
            tfd.Normal, loc=jnp.zeros((2,)), scale=jnp.ones((2,))
        )

        sample = dist.sample(seed=jax.random.key(1))
        sample_2 = dist.sample(2, seed=jax.random.key(2))
        sample_23 = dist.sample((2, 3), seed=jax.random.key(3))

        assert sample.shape == (2, 3)
        assert sample_2.shape == (2, 2, 3)
        assert sample_23.shape == (2, 3, 2, 3)
        assert jnp.all(jnp.isnan(sample[..., 1:]))
        assert jnp.all(jnp.isfinite(sample[..., 0]))

    def test_non_scalar_event_base_distribution_raises(self):
        with pytest.raises(ValueError, match="scalar-event"):
            CensoredDistribution(
                tfd.MultivariateNormalDiag,
                loc=jnp.zeros(2),
                scale_diag=jnp.ones(2),
            )

    def test_record_level_cdf_and_quantile_are_not_defined(self):
        dist = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)

        with pytest.raises(NotImplementedError, match="base_distribution"):
            dist.cdf(uncensored(0.0))

        with pytest.raises(NotImplementedError, match="base_distribution"):
            dist.quantile(0.5)


class TestCensoredDistributionCorrectness:
    def test_uncensored_records_match_base_log_prob(self):
        base = tfd.Normal(loc=0.0, scale=1.0)
        dist = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        value = jnp.array([-1.0, 0.0, 1.0])

        assert jnp.allclose(dist.log_prob(uncensored(value)), base.log_prob(value))

    def test_left_censored_records_match_base_log_cdf(self):
        base = tfd.Normal(loc=0.0, scale=1.0)
        dist = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        upper = jnp.array([-1.0, 0.0, 1.0])

        assert jnp.allclose(dist.log_prob(left_censored(upper)), base.log_cdf(upper))

    def test_right_censored_records_match_base_log_survival(self):
        base = tfd.Normal(loc=0.0, scale=1.0)
        dist = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        lower = jnp.array([-1.0, 0.0, 1.0])

        assert jnp.allclose(
            dist.log_prob(right_censored(lower)),
            base.log_survival_function(lower),
        )

    def test_interval_records_match_base_cdf_difference(self):
        base = tfd.Normal(loc=0.0, scale=1.0)
        dist = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        lower = jnp.array([-1.0, -0.5, 0.0])
        upper = jnp.array([0.0, 0.5, 1.0])
        expected = jnp.log(base.cdf(upper) - base.cdf(lower))

        assert jnp.allclose(dist.log_prob(interval_censored(lower, upper)), expected)

    def test_mixed_records_match_manual_formulas(self):
        base = tfd.Normal(loc=0.0, scale=1.0)
        dist = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )
        expected = jnp.array(
            [
                base.log_prob(0.0),
                base.log_cdf(-1.0),
                base.log_survival_function(1.0),
                jnp.log(base.cdf(1.0) - base.cdf(-1.0)),
            ]
        )

        assert jnp.allclose(dist.log_prob(records), expected)

    def test_batched_base_distribution_follows_tfp_broadcasting(self):
        loc = jnp.array([0.0, 1.0])
        scale = jnp.array([1.0, 2.0])
        base = tfd.Normal(loc=loc, scale=scale)
        dist = CensoredDistribution(tfd.Normal, loc=loc, scale=scale)
        records = uncensored(jnp.ones((5, 1)))

        log_prob = dist.log_prob(records)

        assert log_prob.shape == (5, 2)
        assert jnp.allclose(log_prob, base.log_prob(jnp.ones((5, 1))))

    def test_wrapping_transformation_distribution_matches_base(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        bspline = PTMSpline(knots.knots)
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
        base = LocScaleTransformationDist(
            coef=coef, loc=0.0, scale=1.0, bspline=bspline
        )
        dist = CensoredDistribution(
            LocScaleTransformationDist,
            coef=coef,
            loc=0.0,
            scale=1.0,
            bspline=bspline,
        )
        value = jnp.ones((5, 1))

        assert dist.batch_shape == base.batch_shape
        assert jnp.allclose(dist.log_prob(uncensored(value)), base.log_prob(value))
        assert jnp.allclose(dist.log_prob(left_censored(value)), base.log_cdf(value))
        assert jnp.allclose(
            dist.log_prob(right_censored(value)), base.log_survival_function(value)
        )


class TestCensoredDistributionJaxCompatibility:
    def test_jit_mixed_log_prob(self):
        dist = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )

        log_prob = jax.jit(dist.log_prob)(records)

        assert log_prob.shape == (4,)
        assert_no_nan(log_prob)

    def test_grad_through_normal_location(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )

        def objective(loc):
            dist = CensoredDistribution(tfd.Normal, loc=loc, scale=1.0)
            return jnp.sum(dist.log_prob(records))

        grad = jax.grad(objective)(jnp.array(0.0))

        assert jnp.isfinite(grad)

    def test_grad_through_ptm_coefficients(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        bspline = PTMSpline(knots.knots)
        coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        ).reshape(4, 1, 3)

        def objective(coef):
            dist = CensoredDistribution(
                LocScaleTransformationDist,
                coef=coef,
                loc=0.0,
                scale=1.0,
                bspline=bspline,
                gauss_legendre_order=4,
            )
            return jnp.sum(dist.log_prob(records))

        grad = jax.grad(objective)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))


class TestCensoredDistributionNumericalStability:
    def test_extreme_one_sided_censoring_has_no_nan(self):
        dist = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        log_prob = jnp.asarray(
            [
                dist.log_prob(left_censored(-10.0)),
                dist.log_prob(right_censored(10.0)),
            ]
        )

        assert_no_nan(log_prob)
        assert jnp.all(jnp.isfinite(log_prob))

    def test_far_tail_interval_is_finite_or_negative_infinity(self):
        dist = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        log_prob = dist.log_prob(interval_censored(8.0, 8.1))

        assert_no_nan(log_prob)
        assert bool(jnp.isfinite(log_prob) | jnp.isneginf(log_prob))

    def test_very_narrow_interval_has_no_nan(self):
        dist = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        log_prob = dist.log_prob(interval_censored(0.0, 1e-5))

        assert_no_nan(log_prob)
        assert bool(jnp.isfinite(log_prob) | jnp.isneginf(log_prob))

    def test_invalid_intervals_return_negative_infinity(self):
        dist = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        records = interval_censored(jnp.array([1.0, 2.0]), jnp.array([1.0, 1.0]))

        log_prob = dist.log_prob(records)

        assert jnp.all(jnp.isneginf(log_prob))


class TestCensoredPTMDist:
    def test_liesel_helper_builds_censored_distribution(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        coef = jax.random.normal(jax.random.key(1), (1, knots.nparam))
        loc = lsl.Var.new_param(0.0, name="loc")
        scale = lsl.Var.new_param(1.0, name="scale")
        shape = lsl.Var.new_param(coef, name="shape")
        dist_node = CensoredPTMDist(
            knots=knots.knots,
            loc=loc,
            scale=scale,
            shape=shape,
            gauss_legendre_order=4,
        )

        dist = dist_node.init_dist()

        assert isinstance(dist, CensoredDistribution)
        assert isinstance(dist.base_distribution, LocScaleTransformationDist)
        assert dist.log_prob(uncensored(0.0)).shape == ()
