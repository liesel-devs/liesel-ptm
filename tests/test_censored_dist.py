"""CensoredDist wraps any keyword-input Liesel distribution in censoring."""

from functools import partial

import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm import survival_dist as surv
from tests.test_censored_ptm import (
    RECORDS,
    X,
    add_terms,
    jit_log_prob_grad,
    likelihood_node,
    nonzero_position,
)


def normal_model(records, **kwargs):
    loc = lsl.Var.new_param(0.2, name="loc")
    dist = surv.CensoredDist(tfd.Normal, loc=loc, scale=1.5, **kwargs)
    return lsl.Model([lsl.Var.new_obs(records, dist, name="y")])


def cached_ptm(records, *, conditional_shape=False, x=X):
    """Censored PTM built by hand from a PTMDist with the record cache on."""
    template = add_terms(
        ptm.PTM(jnp.zeros(records.shape[0]), nshape=4),
        x,
        conditional_shape=conditional_shape,
    )
    ptm_dist = likelihood_node(template)
    template.pop_nodes_and_vars()
    template.response.dist_node = None
    dist = surv.CensoredDist.from_dist(ptm_dist, censoring_records=records)
    return lsl.Model([lsl.Var.new_obs(records, dist, name="y")])


@pytest.mark.parametrize("cached", [False, True])
def test_matches_censored_distribution(cached):
    model = normal_model(RECORDS, censoring_records=RECORDS if cached else None)
    expected = surv.CensoredDistribution(tfd.Normal, loc=0.2, scale=1.5)
    np.testing.assert_allclose(
        model.vars["y"].log_prob, expected.log_prob(RECORDS), rtol=1e-6
    )
    node = likelihood_node_of(model)
    assert isinstance(node, surv.CensoredDist)
    assert node.base_distribution is tfd.Normal
    if cached:
        np.testing.assert_array_equal(node.censoring_records, RECORDS)
    else:
        assert node.censoring_records is None
        assert isinstance(node.distribution, partial)
        assert "censoring_records" not in node.distribution.keywords


def likelihood_node_of(model) -> surv.CensoredDist:
    node = model.vars["y"].dist_node
    assert isinstance(node, surv.CensoredDist)
    return node


def test_from_dist_matches_manual_recipe_and_leaves_original_unchanged():
    loc = lsl.Var.new_param(0.2, name="loc")
    original = lsl.Dist(tfd.Normal, loc=loc, scale=1.5)
    original.per_obs = False
    constructor = original.distribution
    censored = surv.CensoredDist.from_dist(original)
    assert original.distribution is constructor
    assert original.var is None and original.at is None
    assert censored.per_obs is False
    assert censored.kwinputs["loc"] is original.kwinputs["loc"]
    model = lsl.Model([lsl.Var.new_obs(RECORDS, censored, name="y")])
    expected = surv.CensoredDistribution(tfd.Normal, loc=0.2, scale=1.5)
    np.testing.assert_allclose(model.log_lik, expected.log_prob(RECORDS).sum())


def test_rejects_positional_attached_and_censored_distributions():
    with pytest.raises(ValueError, match="keyword"):
        surv.CensoredDist.from_dist(lsl.Dist(tfd.Normal, 0.0, 1.0))
    attached = lsl.Dist(tfd.Normal, loc=0.0, scale=1.0)
    lsl.Var.new_obs(jnp.zeros(2), attached, name="z")
    with pytest.raises(ValueError, match="already belongs"):
        surv.CensoredDist.from_dist(attached)
    censored = surv.CensoredDist(tfd.Normal, loc=0.0, scale=1.0)
    with pytest.raises(ValueError, match="already censored"):
        surv.CensoredDist.from_dist(censored)
    with pytest.raises(ValueError, match="already censored"):
        surv.CensoredDist(
            partial(surv.CensoredDistribution, distribution=tfd.Normal),
            loc=0.0,
            scale=1.0,
        )


@pytest.mark.parametrize("records", [jnp.zeros(3), jnp.zeros((4, 2))])
def test_rejects_cache_with_wrong_layout(records):
    with pytest.raises(ValueError, match="trailing"):
        surv.CensoredDist(tfd.Normal, censoring_records=records, loc=0.0, scale=1.0)


@pytest.mark.parametrize("conditional_shape", [False, True])
def test_cache_matches_uncached_ptm_at_nonzero_coefficients(conditional_shape):
    uncached = add_terms(
        ptm.PTM.from_censored(RECORDS, nshape=4),
        conditional_shape=conditional_shape,
    )
    cached = cached_ptm(RECORDS, conditional_shape=conditional_shape)
    assert cached.parameters.keys() == uncached.parameters.keys()
    position = nonzero_position(uncached)
    for graph in (cached, uncached):
        graph.state = graph.update_state(position)
    np.testing.assert_allclose(cached.log_lik, uncached.log_lik, rtol=1e-6)
    np.testing.assert_allclose(cached.log_prior, uncached.log_prior, rtol=1e-6)
    actual = jit_log_prob_grad(cached, position)
    expected = jit_log_prob_grad(uncached, position)
    for name in position:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-5)


def test_cache_follows_values_while_row_types_stay_fixed():
    model = normal_model(RECORDS, censoring_records=RECORDS)
    shifted = RECORDS + 0.3
    model.vars["y"].value = shifted
    expected = surv.CensoredDistribution(tfd.Normal, loc=0.2, scale=1.5)
    np.testing.assert_allclose(
        model.vars["y"].log_prob, expected.log_prob(shifted), rtol=1e-6
    )


@pytest.mark.parametrize(
    "changed",
    [
        RECORDS.at[2].set(jnp.array([0.4, jnp.nan, jnp.nan])),
        RECORDS[jnp.array([1, 0, 2, 3, 4, 5])],
    ],
)
def test_cache_rejects_changed_row_types(changed):
    model = normal_model(RECORDS, censoring_records=RECORDS)
    with pytest.raises(ValueError, match="row types"):
        model.vars["y"].value = changed


def test_copy_keeps_cache():
    model = normal_model(RECORDS, censoring_records=RECORDS)
    copied = model.copy()
    node = likelihood_node_of(copied)
    np.testing.assert_array_equal(node.censoring_records, RECORDS)
    np.testing.assert_allclose(copied.log_lik, model.log_lik)


def test_summaries_and_plots_accept_cached_response():
    cached = cached_ptm(RECORDS)
    ordinary = add_terms(ptm.PTM(jnp.zeros(6), nshape=4))
    position = nonzero_position(ordinary)
    samples = {
        name: jnp.stack([value, 1.1 * value])[None] for name, value in position.items()
    }
    intercept = ordinary.pred.shape.intercept
    assert intercept is not None
    actual = ptm.summarise_intercept_dist(cached.vars["y"], intercept, samples, rgrid=5)
    expected = ptm.summarise_intercept_dist(
        ordinary.response, intercept, samples, rgrid=5
    )
    np.testing.assert_allclose(actual["mean"], expected["mean"], rtol=1e-6)
    ptm.plot_intercept_dist(cached.vars["y"], intercept, samples, rgrid=5)
    summary = ptm.summarise_conditional_dist(
        cached.vars["y"], samples, newdata={"x": [-0.5, 0.5]}, rgrid=5
    )
    assert np.isfinite(summary["mean"]).all()


def test_base_distribution_strips_fixed_keywords():
    node = surv.CensoredDist(partial(tfd.Normal, scale=2.0), loc=0.0)
    normal = node.base_distribution(loc=0.0)
    assert isinstance(normal, tfd.Normal)
    assert normal.scale == 2.0
