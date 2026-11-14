"""The censoring wrapper selects rows with TFP batch slicing."""

from functools import partial

import jax
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm import survival_dist as surv

NAN = np.nan
RECORDS = jnp.array(
    [
        [0.3, NAN, NAN],
        [NAN, NAN, -0.2],
        [NAN, 0.4, NAN],
        [NAN, -0.6, 0.9],
    ]
)
N = RECORDS.shape[0]
SPLINE = ptm.OnionSpline(ptm.OnionKnots(-4.0, 4.0, nparam=6).knots)


def ptm_dist(n=N):
    return ptm.LocScaleTransformationDist(
        coef=jnp.linspace(-0.3, 0.4, 6) + jnp.arange(n)[:, None] * 0.05,
        loc=jnp.linspace(-0.2, 0.3, n),
        scale=1.1,
        bspline=SPLINE,
        centered=True,
        scaled=True,
    )


def assert_cache_matches(base, records):
    uncached = surv.CensoredDistribution(base)
    cached = surv.CensoredDistribution(base, censoring_records=records)
    np.testing.assert_allclose(
        cached.log_prob(records), uncached.log_prob(records), rtol=1e-5, atol=1e-6
    )


def test_rejects_a_constructor():
    with pytest.raises(TypeError, match="built distribution"):
        surv.CensoredDistribution(tfd.Normal)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "base",
    [tfd.Normal(0.1, 1.2), tfd.Normal(jnp.zeros(1), 1.0), ptm_dist(1)],
    ids=["no observation axis", "size-one axis", "ptm"],
)
def test_single_observation_can_be_cached(base):
    assert_cache_matches(base, RECORDS[:1])


def test_vector_parameter_with_as_many_components_as_rows():
    # The former shape guessing confused the component axis with the rows here.
    logits = jnp.linspace(-1.0, 1.0, N * N).reshape(N, N)
    base = tfd.MixtureSameFamily(
        tfd.Categorical(logits=logits),
        tfd.Normal(jnp.linspace(-1.0, 1.0, N * N).reshape(N, N), 1.0),
    )
    assert_cache_matches(base, RECORDS)


def test_ptm_rows_are_selected_by_slicing():
    assert_cache_matches(ptm_dist(), RECORDS)


def test_posterior_batch_rows_are_selected_by_slicing():
    base = tfd.Normal(jnp.linspace(-1.0, 1.0, 2 * 3 * N).reshape(2, 3, N), 1.0)
    assert_cache_matches(base, RECORDS)


class RenamedNormal(tfd.Normal):
    """Parameter names that TFP's parameter properties do not know."""

    def __init__(self, mu, sigma):
        super().__init__(loc=mu, scale=sigma)


def test_cache_rejects_a_base_that_cannot_be_sliced():
    base = ptm.TransformationDist(
        coef=jnp.zeros(6),
        bspline=SPLINE,
        parametric_distribution=RenamedNormal,
        mu=jnp.zeros(N),
        sigma=1.0,
    )
    surv.CensoredDistribution(base).log_prob(RECORDS)
    with pytest.raises(ValueError, match="batch slicing.*without the cache"):
        surv.CensoredDistribution(base, censoring_records=RECORDS)


def test_cached_node_reports_a_base_that_cannot_be_sliced_at_model_build():
    dist = surv.CensoredDist(
        partial(
            ptm.TransformationDist,
            coef=jnp.zeros(6),
            bspline=SPLINE,
            parametric_distribution=RenamedNormal,
        ),
        cache=True,
        mu=jnp.zeros(N),
        sigma=1.0,
    )
    with pytest.raises(ValueError, match="batch slicing"):
        lsl.Model([lsl.Var.new_obs(RECORDS, dist, name="y")])


@pytest.mark.parametrize("cached", [False, True])
def test_wrapper_supports_batch_slicing(cached):
    base = ptm_dist()
    dist = surv.CensoredDistribution(
        base, censoring_records=RECORDS if cached else None
    )
    rows = jnp.array([3, 1])
    sliced = dist[rows]
    assert sliced.batch_shape == (2,)
    np.testing.assert_allclose(
        sliced.log_prob(RECORDS[rows]), dist.log_prob(RECORDS)[rows], rtol=1e-6
    )


def test_uncached_wrapper_passes_into_jit():
    dist = surv.CensoredDistribution(ptm_dist())
    log_prob = jax.jit(lambda dist, records: dist.log_prob(records))
    np.testing.assert_allclose(
        log_prob(dist, RECORDS), dist.log_prob(RECORDS), rtol=1e-5
    )


def test_cached_wrapper_cannot_pass_into_jit():
    dist = surv.CensoredDistribution(ptm_dist(), censoring_records=RECORDS)
    log_prob = jax.jit(lambda dist, records: dist.log_prob(records))
    with pytest.raises(TypeError, match="inside the function"):
        log_prob(dist, RECORDS)
