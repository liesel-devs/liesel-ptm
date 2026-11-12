"""Censoring encoders record the intended observation type in [time, lower, upper]."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_ptm import survival_dist as surv

NAN = np.nan
INF = np.inf


def assert_records(actual, expected):
    actual = np.asarray(actual)
    assert actual.shape == np.shape(expected)
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=actual.dtype))


class TestCensored:
    def test_infers_every_observation_type_from_bounds(self):
        records = surv.censored(
            jnp.array([1.0, -INF, 2.0, 3.0]), jnp.array([1.0, 4.0, INF, 5.0])
        )
        assert_records(
            records,
            [
                [1.0, NAN, NAN],
                [NAN, NAN, 4.0],
                [NAN, 2.0, NAN],
                [NAN, 3.0, 5.0],
            ],
        )
        assert isinstance(records, jax.Array)
        assert list(surv.censoring_status(records)) == [
            "uncensored",
            "left_censored",
            "right_censored",
            "interval_censored",
        ]

    def test_explicit_event_flag_encodes_exact_observations(self):
        assert_records(surv.censored(2.0, 2.0, event_observed=True), [2.0, NAN, NAN])

    @pytest.mark.parametrize("upper", [3.0, INF])
    def test_event_flag_requires_matching_finite_bounds(self, upper):
        with pytest.raises(ValueError, match="matching finite bounds"):
            surv.censored(2.0, upper, event_observed=True)
        with pytest.raises(ValueError, match="matching finite bounds"):
            surv.censored(INF, INF, event_observed=True)

    def test_censoring_flag_never_promotes_equal_bounds(self):
        with pytest.raises(ValueError, match="lower < upper"):
            surv.censored(2.0, 2.0, event_observed=False)
        assert_records(
            surv.censored([1.0, -INF], [2.0, 3.0], event_observed=False),
            [[NAN, 1.0, 2.0], [NAN, NAN, 3.0]],
        )

    @pytest.mark.parametrize(
        "lower,upper",
        [
            (2.0, 1.0),
            (NAN, 1.0),
            (1.0, NAN),
            (-INF, INF),
            (INF, INF),
            (-INF, -INF),
            (INF, 5.0),
            (0.0, -INF),
        ],
    )
    def test_rejects_invalid_bounds(self, lower, upper):
        with pytest.raises(ValueError, match="lower < upper"):
            surv.censored(lower, upper)

    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, False, False]),
            np.array([1, 0, 0]),
            np.array([1.0, 0.0, 0.0]),
            [True, False, False],
        ],
    )
    def test_accepts_boolean_and_zero_one_masks(self, mask):
        records = surv.censored([1.0, 1.0, -INF], [1.0, 2.0, 0.5], event_observed=mask)
        assert_records(records, [[1.0, NAN, NAN], [NAN, 1.0, 2.0], [NAN, NAN, 0.5]])

    @pytest.mark.parametrize("mask", [np.array([2, 0]), np.array([0.5, 1.0]), "yes"])
    def test_rejects_non_binary_masks(self, mask):
        with pytest.raises(ValueError, match="event_observed"):
            surv.censored([1.0, 1.0], [1.0, 2.0], event_observed=mask)

    def test_broadcasts_bounds_and_mask(self):
        records = surv.censored(
            jnp.array([[0.0], [1.0]]),
            jnp.array([1.0, 1.0]),
            event_observed=np.array([[False], [True]]),
        )
        assert records.shape == (2, 2, 3)
        assert surv.censoring_status(records).tolist() == [
            ["interval_censored", "interval_censored"],
            ["uncensored", "uncensored"],
        ]

    def test_broadcast_status_matches_bounds(self):
        records = surv.censored(jnp.array([[0.0], [1.0]]), jnp.array([1.0, INF]))
        assert records.shape == (2, 2, 3)
        assert surv.censoring_status(records).tolist() == [
            ["interval_censored", "right_censored"],
            ["uncensored", "right_censored"],
        ]

    def test_uses_exact_equality_not_isclose(self):
        records = surv.censored(1.0, 1.0 + 1e-12)
        assert surv.censoring_status(records) == "interval_censored"

    def test_status_survives_conversion_that_collapses_bounds(self):
        # float32 cannot represent 1 + 1e-9, so the bounds collapse on conversion.
        lower = np.float64(1.0)
        upper = np.float64(1.0 + 1e-9)
        records = surv.censored(lower, upper)
        assert records.dtype == jnp.float32
        assert records[1] == records[2]
        assert surv.censoring_status(records) == "interval_censored"
        assert jnp.isnan(records[0])

    def test_integer_bounds_become_floating_point(self):
        records = surv.censored(np.array([1, 2]), np.array([1, 3]))
        assert jnp.issubdtype(records.dtype, jnp.floating)
        assert surv.censoring_status(records).tolist() == [
            "uncensored",
            "interval_censored",
        ]

    def test_infers_types_under_jit(self):
        records = jax.jit(surv.censored)(
            jnp.array([1.0, -INF, 2.0, 3.0]), jnp.array([1.0, 4.0, INF, 5.0])
        )
        assert surv.censoring_status(records).tolist() == [
            "uncensored",
            "left_censored",
            "right_censored",
            "interval_censored",
        ]


class TestDedicatedHelpers:
    def test_defaults_keep_existing_encodings(self):
        assert_records(
            surv.right_censored([1.0, 2.0]), [[NAN, 1.0, NAN], [NAN, 2.0, NAN]]
        )
        assert_records(surv.left_censored(1.0), [NAN, NAN, 1.0])
        assert_records(
            surv.interval_censored([0.0, 1.0], 2.0),
            [[NAN, 0.0, 2.0], [NAN, 1.0, 2.0]],
        )

    def test_defaults_remain_jit_compatible(self):
        value = jnp.array([1.0, 2.0])
        assert_records(jax.jit(surv.right_censored)(value), surv.right_censored(value))
        assert_records(jax.jit(surv.left_censored)(value), surv.left_censored(value))
        assert_records(
            jax.jit(surv.interval_censored)(value, value + 1.0),
            surv.interval_censored(value, value + 1.0),
        )

    @pytest.mark.parametrize(
        "helper,censored_label",
        [
            (surv.right_censored, "right_censored"),
            (surv.left_censored, "left_censored"),
        ],
    )
    def test_observed_rows_use_value_as_exact_observation(self, helper, censored_label):
        records = helper(jnp.array([1.0, 2.0, 3.0]), event_observed=[1, 0, 1])
        assert_records(records[0], [1.0, NAN, NAN])
        assert_records(records[2], [3.0, NAN, NAN])
        assert surv.censoring_status(records).tolist() == [
            "uncensored",
            censored_label,
            "uncensored",
        ]

    def test_scalar_flag_broadcasts_over_values(self):
        assert surv.censoring_status(
            surv.right_censored([1.0, 2.0], event_observed=True)
        ).tolist() == ["uncensored", "uncensored"]

    def test_mask_broadcasts_with_values(self):
        records = surv.right_censored(
            jnp.array([1.0, 2.0]), event_observed=np.array([[True], [False]])
        )
        assert surv.censoring_status(records).tolist() == [
            ["uncensored", "uncensored"],
            ["right_censored", "right_censored"],
        ]

    def test_traced_mask_is_supported(self):
        records = jax.jit(surv.right_censored)(
            jnp.array([1.0, 2.0]), event_observed=jnp.array([True, False])
        )
        assert surv.censoring_status(records).tolist() == [
            "uncensored",
            "right_censored",
        ]

    @pytest.mark.parametrize("helper", [surv.right_censored, surv.left_censored])
    def test_observed_rows_must_be_finite(self, helper):
        with pytest.raises(ValueError, match="finite"):
            helper([1.0, INF], event_observed=[False, True])

    def test_interval_helper_requires_matching_bounds_for_observed_rows(self):
        records = surv.interval_censored(
            [1.0, 0.0], [1.0, 2.0], event_observed=[True, False]
        )
        assert_records(records, [[1.0, NAN, NAN], [NAN, 0.0, 2.0]])
        with pytest.raises(ValueError, match="matching finite bounds"):
            surv.interval_censored([1.0], [2.0], event_observed=True)

    @pytest.mark.parametrize("lower,upper", [(1.0, 1.0), (2.0, 1.0), (NAN, 1.0)])
    def test_interval_helper_requires_positive_width(self, lower, upper):
        with pytest.raises(ValueError, match="positive-width"):
            surv.interval_censored(lower, upper)

    def test_interval_helper_keeps_infinite_endpoints(self):
        records = surv.interval_censored(-INF, 1.0)
        assert_records(records, [NAN, -INF, 1.0])
        assert surv.censoring_status(records) == "interval_censored"

    def test_interval_helper_keeps_status_after_collapse(self):
        records = surv.interval_censored(np.float64(1.0), np.float64(1.0 + 1e-9))
        assert surv.censoring_status(records) == "interval_censored"

    @pytest.mark.parametrize(
        "call",
        [
            lambda: surv.right_censored([1.0], event_observed=[2]),
            lambda: surv.left_censored(
                [1.0],
                event_observed=None,  # ty: ignore[invalid-argument-type]
            ),
            lambda: surv.interval_censored([1.0], [2.0], event_observed=[0.3]),
        ],
    )
    def test_rejects_invalid_flags(self, call):
        with pytest.raises(ValueError, match="event_observed"):
            call()


class TestCensoringStatus:
    def test_labels_every_record_pattern(self):
        records = np.array(
            [
                [0.0, NAN, NAN],
                [NAN, NAN, 1.0],
                [NAN, 1.0, NAN],
                [NAN, 1.0, 2.0],
                [NAN, NAN, NAN],
                [0.0, 1.0, NAN],
                [0.0, NAN, 1.0],
                [0.0, 1.0, 2.0],
            ]
        )
        status = surv.censoring_status(records)
        assert isinstance(status, np.ndarray)
        assert status.dtype.kind == "U"
        assert status.tolist() == [
            "uncensored",
            "left_censored",
            "right_censored",
            "interval_censored",
            "invalid",
            "invalid",
            "invalid",
            "invalid",
        ]

    def test_reports_encoded_type_without_validating_width(self):
        assert surv.censoring_status([NAN, 2.0, 1.0]) == "interval_censored"

    def test_keeps_batch_shape(self):
        records = surv.right_censored(jnp.ones((2, 3)))
        assert surv.censoring_status(records).shape == (2, 3)
        assert surv.censoring_status(records[0, 0]).shape == ()

    @pytest.mark.parametrize("records", [1.0, np.ones(2), np.ones((4, 2))])
    def test_rejects_wrong_record_layout(self, records):
        with pytest.raises(ValueError, match="trailing"):
            surv.censoring_status(records)

    def test_agrees_with_likelihood_branches(self):
        records = jnp.array(
            [
                [0.5, NAN, NAN],
                [NAN, NAN, 0.5],
                [NAN, 0.5, NAN],
                [NAN, -0.5, 0.5],
                [NAN, NAN, NAN],
            ]
        )
        normal = surv.CensoredDistribution(distribution=tfd.Normal, loc=0.0, scale=1.0)
        log_prob = np.asarray(normal.log_prob(records))
        base = normal.base_distribution
        expected = {
            "uncensored": base.log_prob(0.5),
            "left_censored": base.log_cdf(0.5),
            "right_censored": base.log_survival_function(0.5),
            "interval_censored": jnp.log(base.cdf(0.5) - base.cdf(-0.5)),
            "invalid": -INF,
        }
        for value, label in zip(log_prob, surv.censoring_status(records), strict=True):
            np.testing.assert_allclose(value, expected[str(label)], rtol=1e-5)
