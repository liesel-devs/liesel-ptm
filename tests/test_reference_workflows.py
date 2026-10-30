"""Behavior that the cleanup must preserve; see GitHub issue #2."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from demo import reference_modern, reference_original


@pytest.fixture(scope="module")
def modern():
    return reference_modern.build_model()


def test_modern_identity_shape_recovers_gaussian_likelihood(modern):
    response = modern.vars["y"]
    dist = response.dist_node.init_dist()
    expected = tfd.Normal(modern.vars["loc"].value, modern.vars["scale"].value)
    np.testing.assert_allclose(
        dist.log_prob(response.value), expected.log_prob(response.value), atol=2e-5
    )
    assert dist.batch_shape == (24,)


def test_modern_shape_covariates_and_distribution_roundtrip():
    model = reference_modern.build_model()
    shape = model.vars["delta"]
    term = shape.terms["h.ps(x)"]
    coef = jnp.linspace(-0.15, 0.15, term.coef.value.size).reshape(
        term.coef.value.shape
    )
    model.state = model.update_state({term.coef.name: coef})
    np.testing.assert_allclose(shape.value.sum(axis=-1), 0.0, atol=1e-6)
    assert not np.allclose(shape.value[0], shape.value[-1])
    dist = model.vars["y"].dist_node.init_dist()
    probabilities = jnp.full(24, 0.7)
    np.testing.assert_allclose(dist.cdf(dist.quantile(probabilities)), 0.7, atol=2e-4)
    assert np.isfinite(model.log_prob)
    np.testing.assert_allclose(dist.mean(), model.vars["loc"].value, atol=2e-4)
    np.testing.assert_allclose(dist.stddev(), model.vars["scale"].value, atol=2e-4)


def test_modern_parameter_blocks_cover_the_model(modern):
    loc = set(modern.parental_submodel(modern.vars["loc"]).parameters)
    scale = set(modern.parental_submodel(modern.vars["scale"]).parameters)
    shape = set(modern.parental_submodel(modern.vars["delta"]).parameters)
    assert not (loc & scale or loc & shape or scale & shape)
    assert loc | scale | shape == set(modern.parameters)
    assert len(shape) > 3  # Includes the shape intercept's smoothing parameter.


def test_modern_staged_optimization_vi_prediction_and_effects():
    model = reference_modern.build_model()
    initial_log_prob = float(model.log_prob)
    point, samples = reference_modern.fit(model)
    assert set(point) == set(samples) == set(model.parameters)
    assert float(model.log_prob) > initial_log_prob
    for name, values in samples.items():
        assert values.shape == (1, 3) + model.parameters[name].value.shape
        assert np.isfinite(values).all()

    shape = model.vars["delta"]
    prediction = model.predict(
        samples,
        predict=["loc", "scale", "delta"],
        newdata={"x": jnp.array([-0.5, 0.5]), "group": jnp.array([0, 1])},
    )
    assert prediction["delta"].shape == (1, 3, 2, 4)
    assert prediction["loc"].shape == prediction["scale"].shape == (1, 3, 2)
    np.testing.assert_allclose(prediction["delta"].sum(axis=-1), 0.0, atol=1e-6)
    assert np.all(prediction["scale"] > 0.0)

    response = model.vars["y"]
    for values, sample_size in ((point, 1), (samples, 3)):
        summary = ptm.summarise_1d_smooth_dist(
            response,
            shape.terms["h.ps(x)"],
            values,
            intercept=shape.intercept,
            ngrid=3,
            rgrid=9,
        )
        assert summary["sample_size"].eq(sample_size).all()
        assert set(summary["quantity"]) >= {"density", "cdf", "transformation"}
        assert np.isfinite(summary["mean"]).all()
    plot = ptm.plot_cluster_dist(
        response,
        shape.terms["h.group"],
        samples,
        intercept=shape.intercept,
        rgrid=9,
    )
    figure = plot.draw()
    assert len(figure.axes) == 1
    plt.close(figure)


def test_original_computed_intercepts_update_and_standardize_residuals():
    model = reference_original.build_model()
    original_intercepts = [
        float(model.vars[name].value) for name in ("beta0", "gamma0")
    ]
    loc_term = model.vars["loc_terms"].terms["m.x"]
    scale_term = model.vars["log_scale_terms"].terms["s.x"]
    model.state = model.update_state(
        {
            loc_term.coef.name: jnp.full_like(loc_term.coef.value, 0.4),
            scale_term.coef.name: jnp.full_like(scale_term.coef.value, 0.2),
        }
    )
    intercepts = [float(model.vars[name].value) for name in ("beta0", "gamma0")]
    assert not np.allclose(intercepts, original_intercepts)
    assert not {"beta0", "gamma0"} & model.parameters.keys()
    residual = (model.vars["y"].value - model.vars["loc"].value) / model.vars[
        "scale"
    ].value
    np.testing.assert_allclose(residual.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(residual.std(), 1.0, atol=1e-6)
    assert np.isfinite(model.log_prob)


def test_original_has_shared_modern_deltas_and_the_ptm_spline():
    model = reference_original.build_model()
    shape = model.vars["delta"]
    assert not shape.terms
    assert shape.value.shape == (6,)
    assert shape.latent_ndim == 5
    assert isinstance(
        model.vars["y"].dist_node.init_dist().bspline, ptm.bspline.PTMSpline
    )
    intercept = shape.intercept
    model.state = model.update_state(
        {intercept.coef.name: jnp.linspace(-0.1, 0.1, intercept.coef.value.size)}
    )
    np.testing.assert_allclose(shape.value.sum(), 0.0, atol=1e-6)
    assert np.isfinite(model.log_prob)
