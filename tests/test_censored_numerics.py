"""Censored log-probabilities and their gradients stay finite and accurate."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm import survival_dist as surv

NAN = np.nan
INF = np.inf
COEF = jnp.linspace(-0.3, 0.3, 8)
SPLINE = ptm.OnionSpline(ptm.OnionKnots(-4.0, 4.0, nparam=8).knots)


def normal(loc, scale, **kwargs):
    return tfd.Normal(loc, scale, **kwargs)


def ptm_dist(loc, scale, **kwargs):
    return ptm.LocScaleTransformationDist(
        coef=COEF,
        loc=loc,
        scale=scale,
        bspline=SPLINE,
        centered=True,
        scaled=True,
        **kwargs,
    )


BASES = {"normal": normal, "ptm": ptm_dist}


def mixed_records(shift, scale):
    """One row of each type around ``shift``, in units of ``scale``."""
    return jnp.array(
        [
            [shift + 0.5 * scale, NAN, NAN],
            [NAN, NAN, shift - 0.5 * scale],
            [NAN, shift + 0.5 * scale, NAN],
            [NAN, shift - 0.5 * scale, shift + scale],
        ]
    )


def value_and_grad(base, records, loc, scale, *, cache):
    def log_prob(loc):
        dist = surv.CensoredDistribution(
            base(loc=jnp.full(records.shape[:1], loc), scale=scale),
            censoring_records=records if cache else None,
        )
        return dist.log_prob(records)

    value = log_prob(loc)
    grad = jax.grad(lambda loc: log_prob(loc).sum())(jnp.float32(loc))
    return value, grad


def manual_log_prob(dist, records):
    time, lower, upper = (records[:, i] for i in range(3))
    return jnp.array(
        [
            dist.log_prob(time[0]),
            dist.log_cdf(upper[1]),
            dist.log_survival_function(lower[2]),
            jnp.log(dist.cdf(upper[3]) - dist.cdf(lower[3])),
        ]
    )


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("base", BASES.values(), ids=BASES.keys())
@pytest.mark.parametrize(
    "shift,scale", [(0.0, 1.0), (7.0, 0.3), (-7.0, 0.3), (40.0, 1.0)]
)
def test_data_far_from_zero_have_finite_gradients(base, shift, scale, cache):
    records = mixed_records(shift, scale)
    value, grad = value_and_grad(base, records, shift, scale, cache=cache)
    expected = manual_log_prob(base(loc=shift, scale=scale), records)
    np.testing.assert_allclose(value, expected, rtol=1e-4, atol=1e-5)
    assert np.isfinite(grad)


def normal_interval_log_prob(lower, upper):
    """Reference in float64, accurate in the tails."""
    if lower >= 0:
        prob = 0.5 * (math.erfc(lower / math.sqrt(2)) - math.erfc(upper / math.sqrt(2)))
    else:
        prob = 0.5 * (
            math.erfc(-upper / math.sqrt(2)) - math.erfc(-lower / math.sqrt(2))
        )
    return math.log(prob)


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("lower,upper", [(8.0, 9.0), (-9.0, -8.0), (1.0, 1.001)])
def test_tail_and_narrow_intervals_are_accurate_with_finite_gradients(
    lower, upper, cache
):
    records = jnp.array([[NAN, lower, upper]] * 2)
    value, grad = value_and_grad(normal, records, 0.0, 1.0, cache=cache)
    np.testing.assert_allclose(value, normal_interval_log_prob(lower, upper), rtol=1e-3)
    assert np.isfinite(grad)


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("base", BASES.values(), ids=BASES.keys())
def test_infinite_interval_endpoints_are_one_sided(base, cache):
    records = jnp.array(
        [[NAN, -INF, 0.3], [NAN, 0.3, INF], [NAN, -INF, INF], [NAN, -INF, 0.3]]
    )
    value, grad = value_and_grad(base, records, 0.1, 1.0, cache=cache)
    dist = base(loc=0.1, scale=1.0)
    expected = [
        dist.log_cdf(0.3),
        dist.log_survival_function(0.3),
        0.0,
        dist.log_cdf(0.3),
    ]
    np.testing.assert_allclose(value, expected, rtol=1e-5, atol=1e-6)
    assert np.isfinite(grad)


def test_positive_support_base_has_finite_gradients_uncached():
    records = surv.right_censored([1.0, 2.0, 3.0], censored=[False, True, False])

    def log_lik(loc):
        dist = surv.CensoredDistribution(tfd.LogNormal(loc=loc, scale=0.5))
        return dist.log_prob(records).sum()

    assert np.isfinite(jax.grad(log_lik)(jnp.float32(0.3)))


def test_rejects_integer_base_distribution():
    with pytest.raises(ValueError, match="floating-point"):
        surv.CensoredDistribution(tfd.Categorical(logits=jnp.zeros(3)))


@pytest.mark.parametrize(
    "records,match",
    [
        (jnp.array([[NAN, NAN, NAN]]), "invalid"),
        (jnp.array([[NAN, 1.0, 1.0]]), "positive width"),
    ],
)
def test_validate_args_rejects_malformed_records(records, match):
    dist = surv.CensoredDistribution(tfd.Normal(loc=0.0, scale=1.0), validate_args=True)
    with pytest.raises(ValueError, match=match):
        dist.log_prob(records)
    unchecked = surv.CensoredDistribution(tfd.Normal(loc=0.0, scale=1.0))
    assert np.isneginf(unchecked.log_prob(records)).all()
