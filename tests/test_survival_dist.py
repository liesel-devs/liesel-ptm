from functools import partial

import jax
import jax.numpy as jnp
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_ptm.bspline.ptm import PTMKnots, PTMSpline
from liesel_ptm.dist import LocScaleTransformationDist
from liesel_ptm.survival_dist import (
    CensoredDistribution,
    IntervalCensoredDistribution,
    LeftCensoredDistribution,
    RightCensoredDistribution,
    interval_censored,
    left_censored,
    right_censored,
    setup_censored_vars,
    uncensored,
)


def assert_no_nan(value):
    assert not jnp.any(jnp.isnan(value))


class BranchSentinelDistribution(tfd.Distribution):
    def __init__(self, validate_args=False, allow_nan_stats=True, name="sentinel"):
        super().__init__(
            dtype=jnp.float32,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=dict(locals()),
            name=name,
        )

    def _event_shape(self):
        return ()

    def _event_shape_tensor(self):
        return jnp.array([], dtype=jnp.int32)

    def _batch_shape(self):
        return ()

    def _batch_shape_tensor(self, **kwargs):
        return jnp.array([], dtype=jnp.int32)

    def _log_prob(self, value):
        raise AssertionError("log_prob branch should not be evaluated")

    def _log_cdf(self, value):
        return -jax.nn.softplus(-value)

    def _log_survival_function(self, value, **kwargs):
        return -jax.nn.softplus(value)

    def _sample_n(self, n, seed=None, **kwargs):
        return jnp.zeros((n,), dtype=jnp.float32)


class ProbabilityFallbackDistribution(tfd.Distribution):
    def __init__(self, validate_args=False, allow_nan_stats=True, name="fallback"):
        super().__init__(
            dtype=jnp.float32,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=dict(locals()),
            name=name,
        )

    def _event_shape(self):
        return ()

    def _event_shape_tensor(self):
        return jnp.array([], dtype=jnp.int32)

    def _batch_shape(self):
        return ()

    def _batch_shape_tensor(self, **kwargs):
        return jnp.array([], dtype=jnp.int32)

    def _log_prob(self, value):
        return jnp.zeros_like(value)

    def _cdf(self, value):
        return jnp.where(value < 0.0, 0.0, 0.25)

    def _log_cdf(self, value):
        raise NotImplementedError

    def _survival_function(self, value, **kwargs):
        return jnp.where(value > 0.0, 0.0, 0.25)

    def _log_survival_function(self, value, **kwargs):
        raise NotImplementedError

    def _sample_n(self, n, seed=None, **kwargs):
        return jnp.zeros((n,), dtype=jnp.float32)


class UncensoredOnlyDistribution(tfd.Distribution):
    def __init__(self, validate_args=False, allow_nan_stats=True, name="uncensored"):
        super().__init__(
            dtype=jnp.float32,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=dict(locals()),
            name=name,
        )

    def _event_shape(self):
        return ()

    def _event_shape_tensor(self):
        return jnp.array([], dtype=jnp.int32)

    def _batch_shape(self):
        return ()

    def _batch_shape_tensor(self, **kwargs):
        return jnp.array([], dtype=jnp.int32)

    def _log_prob(self, value):
        return jnp.zeros_like(value)

    def _log_cdf(self, value):
        raise AssertionError("log_cdf branch should not be evaluated")

    def _log_survival_function(self, value, **kwargs):
        raise AssertionError("log_survival branch should not be evaluated")

    def _sample_n(self, n, seed=None, **kwargs):
        return jnp.zeros((n,), dtype=jnp.float32)


class TestCensoredDistributionApi:
    def test_import_and_distribution_shapes(self):
        dist = CensoredDistribution(
            tfd.Normal, loc=jnp.zeros((2,)), scale=jnp.ones((2,))
        )

        assert dist.event_shape == (3,)
        assert dist.batch_shape == (2,)
        assert isinstance(dist.base_distribution, tfd.Normal)

    def test_cached_records_initialize_and_preserve_event_shape(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, 1.0, jnp.nan],
            ]
        )
        dist = CensoredDistribution(
            tfd.Normal, censoring_records=records, loc=0.0, scale=1.0
        )

        assert dist.event_shape == (3,)
        assert dist.batch_shape == ()
        assert dist.censoring_records is not None
        assert dist.censoring_records.shape == records.shape
        assert dist.log_prob(records).shape == (2,)

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

    def test_cached_records_require_trailing_event_shape_and_observation_axis(self):
        with pytest.raises(ValueError, match="trailing event shape"):
            CensoredDistribution(
                tfd.Normal,
                censoring_records=jnp.ones((2,)),
                loc=0.0,
                scale=1.0,
            )

        with pytest.raises(ValueError, match="observation"):
            CensoredDistribution(
                tfd.Normal,
                censoring_records=uncensored(0.0),
                loc=0.0,
                scale=1.0,
            )

    def test_specialized_distribution_shapes(self):
        left = LeftCensoredDistribution(
            tfd.Normal, loc=jnp.zeros((2,)), scale=jnp.ones((2,))
        )
        right = RightCensoredDistribution(
            tfd.Normal, loc=jnp.zeros((2,)), scale=jnp.ones((2,))
        )
        interval = IntervalCensoredDistribution(
            tfd.Normal, loc=jnp.zeros((2,)), scale=jnp.ones((2,))
        )

        assert left.event_shape == ()
        assert right.event_shape == ()
        assert interval.event_shape == (2,)
        assert left.batch_shape == (2,)
        assert right.batch_shape == (2,)
        assert interval.batch_shape == (2,)

    def test_specialized_sample_is_not_defined_for_bounds(self):
        dist = RightCensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)

        with pytest.raises(NotImplementedError, match="base_distribution"):
            dist.sample(seed=jax.random.key(1))


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

    def test_cached_records_match_uncached_mixed_records(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )
        uncached = CensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        cached = CensoredDistribution(
            tfd.Normal, censoring_records=records, loc=0.0, scale=1.0
        )

        assert jnp.allclose(cached.log_prob(records), uncached.log_prob(records))

    def test_cached_value_shape_mismatch_raises(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, 1.0, jnp.nan],
            ]
        )
        dist = CensoredDistribution(
            tfd.Normal, censoring_records=records, loc=0.0, scale=1.0
        )

        with pytest.raises(ValueError, match="broadcast"):
            dist.log_prob(jnp.concatenate([records, records[:1]], axis=0))

        with pytest.raises(ValueError, match="additional sample"):
            dist.log_prob(records.reshape(2, 1, 3))

        assert dist.log_prob(records[0]).shape == (2,)

    def test_cached_invalid_records_return_negative_infinity(self):
        records = jnp.array(
            [
                [jnp.nan, jnp.nan, jnp.nan],
                [0.0, jnp.nan, jnp.nan],
            ]
        )
        dist = CensoredDistribution(
            tfd.Normal, censoring_records=records, loc=0.0, scale=1.0
        )

        log_prob = dist.log_prob(records)

        assert jnp.isneginf(log_prob[0])
        assert jnp.isfinite(log_prob[1])

    def test_batched_base_distribution_follows_tfp_broadcasting(self):
        loc = jnp.array([0.0, 1.0])
        scale = jnp.array([1.0, 2.0])
        base = tfd.Normal(loc=loc, scale=scale)
        dist = CensoredDistribution(tfd.Normal, loc=loc, scale=scale)
        records = uncensored(jnp.ones((5, 1)))

        log_prob = dist.log_prob(records)

        assert log_prob.shape == (5, 2)
        assert jnp.allclose(log_prob, base.log_prob(jnp.ones((5, 1))))

    def test_cached_observation_batched_normal_matches_manual_formulas(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )
        loc = jnp.array([0.0, 1.0, 2.0, 3.0])
        scale = jnp.ones(4)
        dist = CensoredDistribution(
            tfd.Normal,
            censoring_records=records,
            loc=loc,
            scale=scale,
        )

        expected = jnp.array(
            [
                tfd.Normal(loc=loc[0], scale=scale[0]).log_prob(0.0),
                tfd.Normal(loc=loc[1], scale=scale[1]).log_cdf(-1.0),
                tfd.Normal(loc=loc[2], scale=scale[2]).log_survival_function(1.0),
                jnp.log(
                    tfd.Normal(loc=loc[3], scale=scale[3]).cdf(1.0)
                    - tfd.Normal(loc=loc[3], scale=scale[3]).cdf(-1.0)
                ),
            ]
        )

        assert jnp.allclose(dist.log_prob(records), expected)

    def test_cached_matrix_records_broadcast_over_posterior_batch(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )
        loc = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 10.0
        scale = jnp.ones((1, 3, 1), dtype=jnp.float32)
        uncached = CensoredDistribution(tfd.Normal, loc=loc, scale=scale)
        cached = CensoredDistribution(
            tfd.Normal,
            censoring_records=records,
            loc=loc,
            scale=scale,
        )
        value = records.reshape(1, 1, 4, 3)

        log_prob = cached.log_prob(value)

        assert log_prob.shape == (2, 3, 4)
        assert jnp.allclose(log_prob, uncached.log_prob(value))

    def test_cached_explicit_singleton_record_axes_match_uncached(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        ).reshape(1, 1, 4, 3)
        loc = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 10.0
        uncached = CensoredDistribution(tfd.Normal, loc=loc, scale=1.0)
        cached = CensoredDistribution(
            tfd.Normal,
            censoring_records=records,
            loc=loc,
            scale=1.0,
        )

        assert cached.censoring_records is not None
        assert cached.censoring_records.shape == records.shape
        assert cached.log_prob(records).shape == (2, 3, 4)
        assert jnp.allclose(cached.log_prob(records), uncached.log_prob(records))
        assert jnp.allclose(
            cached.log_prob(records.reshape(4, 3)),
            uncached.log_prob(records.reshape(4, 3)),
        )

    def test_wrapping_transformation_distribution_matches_base(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        bspline = PTMSpline(knots.knots)
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
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

    def test_cached_transformation_distribution_matches_uncached_mixed_records(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        bspline = PTMSpline(knots.knots)
        coef = jax.random.normal(jax.random.key(1), (knots.nparam,))
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )
        loc = jnp.array([0.0, 0.2, -0.1, 0.4])
        scale = jnp.ones(4)
        uncached = CensoredDistribution(
            LocScaleTransformationDist,
            coef=coef,
            loc=loc,
            scale=scale,
            bspline=bspline,
            gauss_legendre_order=4,
        )
        cached = CensoredDistribution(
            LocScaleTransformationDist,
            censoring_records=records,
            coef=coef,
            loc=loc,
            scale=scale,
            bspline=bspline,
            gauss_legendre_order=4,
        )

        assert jnp.allclose(
            cached.log_prob(records), uncached.log_prob(records), atol=1e-4
        )

    def test_cached_transformation_distribution_broadcasted_coef_layouts(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        bspline = PTMSpline(knots.knots)
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        ).reshape(1, 1, 4, 3)
        loc = jnp.array(
            [
                [[0.0, 0.2, -0.1, 0.4], [0.1, 0.3, -0.2, 0.5]],
                [[-0.1, 0.0, 0.1, 0.2], [0.2, -0.3, 0.4, -0.5]],
            ],
            dtype=jnp.float32,
        )
        scale = jnp.ones((2, 1, 1), dtype=jnp.float32)
        key = jax.random.key(2)
        shared_coef = jax.random.normal(key, (knots.nparam,))
        posterior_shared_coef = jax.random.normal(
            jax.random.key(3), (2, 2, 1, knots.nparam)
        )
        observation_coef = jax.random.normal(jax.random.key(4), (2, 2, 4, knots.nparam))

        for coef in (shared_coef, posterior_shared_coef, observation_coef):
            uncached = CensoredDistribution(
                LocScaleTransformationDist,
                coef=coef,
                loc=loc,
                scale=scale,
                bspline=bspline,
                gauss_legendre_order=4,
            )
            cached = CensoredDistribution(
                LocScaleTransformationDist,
                censoring_records=records,
                coef=coef,
                loc=loc,
                scale=scale,
                bspline=bspline,
                gauss_legendre_order=4,
            )

            assert cached.log_prob(records).shape == (2, 2, 4)
            assert jnp.allclose(
                cached.log_prob(records),
                uncached.log_prob(records),
                atol=5e-4,
                rtol=5e-4,
            )

    def test_cached_centered_scaled_transformation_distribution_shared_observation_coef(
        self,
    ):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        bspline = PTMSpline(knots.knots)
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ],
            dtype=jnp.float32,
        ).reshape(1, 1, 4, 3)
        loc = jnp.array(
            [
                [[0.0, 0.2, -0.1, 0.4], [0.1, 0.3, -0.2, 0.5]],
                [[-0.1, 0.0, 0.1, 0.2], [0.2, -0.3, 0.4, -0.5]],
            ],
            dtype=jnp.float32,
        )
        scale = jnp.ones((2, 1, 1), dtype=jnp.float32)
        coef = 0.2 * jax.random.normal(
            jax.random.key(5), (2, 2, 1, knots.nparam), dtype=jnp.float32
        )

        uncached = CensoredDistribution(
            LocScaleTransformationDist,
            coef=coef,
            loc=loc,
            scale=scale,
            bspline=bspline,
            centered=True,
            scaled=True,
            gauss_legendre_order=4,
        )
        cached = CensoredDistribution(
            LocScaleTransformationDist,
            censoring_records=records,
            coef=coef,
            loc=loc,
            scale=scale,
            bspline=bspline,
            centered=True,
            scaled=True,
            gauss_legendre_order=4,
        )

        assert isinstance(cached.base_distribution, LocScaleTransformationDist)
        assert cached.base_distribution.transformation_spline_mean().shape == (
            2,
            2,
            1,
        )
        assert cached.log_prob(records).shape == (2, 2, 4)
        assert jnp.allclose(
            cached.log_prob(records),
            uncached.log_prob(records),
            atol=5e-4,
            rtol=5e-4,
        )

    def test_specialized_distributions_match_base_formulas(self):
        base = tfd.Normal(loc=0.0, scale=1.0)
        left = LeftCensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        right = RightCensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        interval = IntervalCensoredDistribution(tfd.Normal, loc=0.0, scale=1.0)
        value = jnp.array([-1.0, 0.0, 1.0])
        interval_value = jnp.stack((value - 0.5, value + 0.5), axis=-1)

        assert jnp.allclose(left.log_prob(value), base.log_cdf(value))
        assert jnp.allclose(right.log_prob(value), base.log_survival_function(value))
        assert jnp.allclose(
            interval.log_prob(interval_value),
            jnp.log(base.cdf(value + 0.5) - base.cdf(value - 0.5)),
        )

    def test_cached_mode_skips_unused_branches(self):
        left_records = left_censored(jnp.array([-1.0, 0.0]))
        right_records = right_censored(jnp.array([0.0, 1.0]))
        uncensored_records = uncensored(jnp.array([0.0, 1.0]))

        left = CensoredDistribution(
            BranchSentinelDistribution, censoring_records=left_records
        )
        right = CensoredDistribution(
            BranchSentinelDistribution, censoring_records=right_records
        )
        uncensored_dist = CensoredDistribution(
            UncensoredOnlyDistribution, censoring_records=uncensored_records
        )

        assert jnp.all(jnp.isfinite(left.log_prob(left_records)))
        assert jnp.all(jnp.isfinite(right.log_prob(right_records)))
        assert jnp.all(jnp.isfinite(uncensored_dist.log_prob(uncensored_records)))

    def test_specialized_distributions_do_not_evaluate_uncensored_branch(self):
        left = LeftCensoredDistribution(BranchSentinelDistribution)
        right = RightCensoredDistribution(BranchSentinelDistribution)
        interval = IntervalCensoredDistribution(BranchSentinelDistribution)

        assert jnp.isfinite(left.log_prob(0.0))
        assert jnp.isfinite(right.log_prob(0.0))
        assert jnp.isfinite(interval.log_prob(jnp.array([-1.0, 1.0])))

    def test_log_tail_fallbacks_preserve_zero_probability(self):
        left = LeftCensoredDistribution(ProbabilityFallbackDistribution)
        right = RightCensoredDistribution(ProbabilityFallbackDistribution)

        left_log_prob = left.log_prob(jnp.array([-1.0, 1.0]))
        right_log_prob = right.log_prob(jnp.array([-1.0, 1.0]))

        assert jnp.isneginf(left_log_prob[0])
        assert jnp.allclose(left_log_prob[1], jnp.log(0.25))
        assert jnp.allclose(right_log_prob[0], jnp.log(0.25))
        assert jnp.isneginf(right_log_prob[1])


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

    def test_jit_cached_log_prob(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )
        dist = CensoredDistribution(
            tfd.Normal, censoring_records=records, loc=0.0, scale=1.0
        )

        log_prob = jax.jit(dist.log_prob)(records)

        assert log_prob.shape == (4,)
        assert_no_nan(log_prob)

    def test_jit_cached_log_prob_with_explicit_singleton_record_axes(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        ).reshape(1, 1, 4, 3)
        loc = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 10.0
        dist = CensoredDistribution(
            tfd.Normal,
            censoring_records=records,
            loc=loc,
            scale=1.0,
        )

        log_prob = jax.jit(dist.log_prob)(records)

        assert log_prob.shape == (2, 3, 4)
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

    def test_grad_through_cached_normal_location(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )

        def objective(loc):
            dist = CensoredDistribution(
                tfd.Normal, censoring_records=records, loc=loc, scale=1.0
            )
            return jnp.sum(dist.log_prob(records))

        grad = jax.grad(objective)(jnp.array(0.0))

        assert jnp.isfinite(grad)

    def test_grad_through_cached_posterior_batched_normal_location(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        ).reshape(1, 1, 4, 3)

        def objective(loc):
            dist = CensoredDistribution(
                tfd.Normal,
                censoring_records=records,
                loc=loc,
                scale=1.0,
            )
            return jnp.sum(dist.log_prob(records))

        loc = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 10.0
        grad = jax.grad(objective)(loc)

        assert grad.shape == loc.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_ptm_coefficients(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        bspline = PTMSpline(knots.knots)
        coef = jax.random.normal(jax.random.key(1), (2, knots.nparam))
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

    def test_grad_through_cached_ptm_coefficients(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        bspline = PTMSpline(knots.knots)
        coef = jax.random.normal(jax.random.key(1), (knots.nparam,))
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )

        def objective(coef):
            dist = CensoredDistribution(
                LocScaleTransformationDist,
                censoring_records=records,
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

    def test_grad_through_cached_observation_batched_ptm_coefficients(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        bspline = PTMSpline(knots.knots)
        coef = jax.random.normal(jax.random.key(1), (2, 3, 4, knots.nparam))
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        ).reshape(1, 1, 4, 3)
        loc = jnp.zeros((2, 3, 4), dtype=jnp.float32)

        def objective(coef):
            dist = CensoredDistribution(
                LocScaleTransformationDist,
                censoring_records=records,
                coef=coef,
                loc=loc,
                scale=1.0,
                bspline=bspline,
                gauss_legendre_order=4,
            )
            return jnp.sum(dist.log_prob(records))

        grad = jax.grad(objective)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_and_grad_specialized_right_censoring(self):
        values = jnp.array([-1.0, 0.0, 1.0])

        def objective(loc):
            dist = RightCensoredDistribution(tfd.Normal, loc=loc, scale=1.0)
            return jnp.sum(jax.jit(dist.log_prob)(values))

        grad = jax.grad(objective)(jnp.array(0.0))

        assert jnp.isfinite(grad)


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

    def test_right_censored_transformation_distribution_tail_is_finite(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        bspline = PTMSpline(knots.knots)
        coef = jax.random.normal(jax.random.key(1), (knots.nparam,))
        dist = RightCensoredDistribution(
            LocScaleTransformationDist,
            coef=coef,
            loc=0.0,
            scale=1.0,
            bspline=bspline,
        )
        values = jnp.array([8.0, 10.0, 20.0])

        log_prob = dist.log_prob(values)

        assert_no_nan(log_prob)
        assert jnp.all(jnp.isfinite(log_prob))
        assert jnp.allclose(
            log_prob,
            dist.base_distribution.log_survival_function(values),
            rtol=1e-5,
            atol=1e-5,
        )

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


class TestCensoredTransformationDistNode:
    def test_liesel_dist_builds_censored_transformation_distribution(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        coef = jax.random.normal(jax.random.key(1), (knots.nparam,))
        bspline = PTMSpline(knots.knots)
        loc = lsl.Var.new_param(0.0, name="loc")
        scale = lsl.Var.new_param(1.0, name="scale")
        shape = lsl.Var.new_param(coef, name="shape")
        bspline_value = lsl.Value(bspline, convert=lambda value: value)
        dist_node = lsl.Dist(
            partial(CensoredDistribution, distribution=LocScaleTransformationDist),
            loc=loc,
            scale=scale,
            coef=shape,
            bspline=bspline_value,
            gauss_legendre_order=4,
        )

        dist = dist_node.init_dist()

        assert isinstance(dist, CensoredDistribution)
        assert isinstance(dist.base_distribution, LocScaleTransformationDist)
        assert dist.log_prob(uncensored(0.0)).shape == ()


class TestSetupCensoredVars:
    def test_splits_record_values_and_specializes_likelihoods(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )
        loc = lsl.Var.new_obs(jnp.array([0.0, 1.0, 2.0, 3.0]), name="loc")
        scale = lsl.Var.new_obs(jnp.ones(4), name="scale")
        dist = lsl.Dist(
            partial(CensoredDistribution, distribution=tfd.Normal),
            loc=loc,
            scale=scale,
        )
        response = lsl.Var.new_obs(records, dist, name="response")

        split = setup_censored_vars(response)
        assert split.uncensored.dist_node is not None
        assert split.left_censored.dist_node is not None
        assert split.right_censored.dist_node is not None
        assert split.interval_censored.dist_node is not None

        assert jnp.allclose(split.uncensored.value, jnp.array([0.0]))
        assert jnp.allclose(split.left_censored.value, jnp.array([-1.0]))
        assert jnp.allclose(split.right_censored.value, jnp.array([1.0]))
        assert jnp.allclose(split.interval_censored.value, jnp.array([[-1.0, 1.0]]))

        assert isinstance(split.uncensored.dist_node.init_dist(), tfd.Normal)
        assert isinstance(
            split.left_censored.dist_node.init_dist(), LeftCensoredDistribution
        )
        assert isinstance(
            split.right_censored.dist_node.init_dist(), RightCensoredDistribution
        )
        assert isinstance(
            split.interval_censored.dist_node.init_dist(),
            IntervalCensoredDistribution,
        )

        assert jnp.allclose(
            split.uncensored.dist_node.kwinputs["loc"].value, jnp.array([0.0])
        )
        assert jnp.allclose(
            split.left_censored.dist_node.kwinputs["loc"].value, jnp.array([1.0])
        )
        assert jnp.allclose(
            split.right_censored.dist_node.kwinputs["loc"].value, jnp.array([2.0])
        )
        assert jnp.allclose(
            split.interval_censored.dist_node.kwinputs["loc"].value, jnp.array([3.0])
        )

    def test_split_likelihood_matches_mixed_likelihood_sum(self):
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )
        dist = lsl.Dist(
            partial(CensoredDistribution, distribution=tfd.Normal),
            loc=0.0,
            scale=1.0,
        )
        response = lsl.Var.new_obs(records, dist, name="response")
        split = setup_censored_vars(response)
        assert response.dist_node is not None
        assert split.uncensored.dist_node is not None
        assert split.left_censored.dist_node is not None
        assert split.right_censored.dist_node is not None
        assert split.interval_censored.dist_node is not None

        mixed_log_prob = response.dist_node.init_dist().log_prob(response.value)
        split_log_prob = (
            jnp.sum(
                split.uncensored.dist_node.init_dist().log_prob(split.uncensored.value)
            )
            + jnp.sum(
                split.left_censored.dist_node.init_dist().log_prob(
                    split.left_censored.value
                )
            )
            + jnp.sum(
                split.right_censored.dist_node.init_dist().log_prob(
                    split.right_censored.value
                )
            )
            + jnp.sum(
                split.interval_censored.dist_node.init_dist().log_prob(
                    split.interval_censored.value
                )
            )
        )

        assert jnp.allclose(split_log_prob, jnp.sum(mixed_log_prob))

    def test_splits_censored_ptm_dist_and_preserves_base_kwargs(self):
        knots = PTMKnots(-4.0, 4.0, nparam=10)
        coef = jax.random.normal(jax.random.key(1), (knots.nparam,))
        bspline = PTMSpline(knots.knots)
        loc = lsl.Var.new_param(0.0, name="loc")
        scale = lsl.Var.new_param(1.0, name="scale")
        shape = lsl.Var.new_param(coef, name="shape")
        bspline_value = lsl.Value(bspline, convert=lambda value: value)
        records = jnp.array(
            [
                [0.0, jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan, -1.0],
                [jnp.nan, 1.0, jnp.nan],
                [jnp.nan, -1.0, 1.0],
            ]
        )
        dist = lsl.Dist(
            partial(CensoredDistribution, distribution=LocScaleTransformationDist),
            loc=loc,
            scale=scale,
            coef=shape,
            bspline=bspline_value,
            gauss_legendre_order=4,
        )
        response = lsl.Var.new_obs(records, dist, name="response")

        split = setup_censored_vars(response)
        assert response.dist_node is not None
        assert split.uncensored.dist_node is not None
        assert split.left_censored.dist_node is not None
        assert split.right_censored.dist_node is not None
        assert split.interval_censored.dist_node is not None

        assert isinstance(
            split.uncensored.dist_node.init_dist(), LocScaleTransformationDist
        )
        left_dist = split.left_censored.dist_node.init_dist()
        assert isinstance(left_dist, LeftCensoredDistribution)
        assert isinstance(left_dist.base_distribution, LocScaleTransformationDist)
        assert split.uncensored.dist_node.kwinputs["bspline"].value is bspline
        assert split.left_censored.dist_node.kwinputs["bspline"].value is bspline

        mixed_log_prob = response.dist_node.init_dist().log_prob(response.value)
        split_log_prob = (
            jnp.sum(
                split.uncensored.dist_node.init_dist().log_prob(split.uncensored.value)
            )
            + jnp.sum(
                split.left_censored.dist_node.init_dist().log_prob(
                    split.left_censored.value
                )
            )
            + jnp.sum(
                split.right_censored.dist_node.init_dist().log_prob(
                    split.right_censored.value
                )
            )
            + jnp.sum(
                split.interval_censored.dist_node.init_dist().log_prob(
                    split.interval_censored.value
                )
            )
        )

        assert jnp.allclose(split_log_prob, jnp.sum(mixed_log_prob), atol=1e-4)

    def test_empty_censoring_type_returns_none_var(self):
        records = uncensored(jnp.array([0.0, 1.0]))
        dist = lsl.Dist(
            partial(CensoredDistribution, distribution=tfd.Normal),
            loc=0.0,
            scale=1.0,
        )
        response = lsl.Var.new_obs(records, dist, name="response")

        split = setup_censored_vars(response)

        assert jnp.allclose(split.uncensored.value, jnp.array([0.0, 1.0]))
        assert split.left_censored.value is None
        assert split.right_censored.value is None
        assert split.interval_censored.value is None

    def test_setup_censored_vars_rejects_non_matrix_records(self):
        response = lsl.Var.new_obs(jnp.ones((2, 2, 3)), name="response")

        with pytest.raises(ValueError, match="shape \\(n, 3\\)"):
            setup_censored_vars(response)
