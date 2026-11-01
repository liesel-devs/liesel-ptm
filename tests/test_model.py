"""Public PTM construction must match the underlying transformation likelihood."""

import jax
import jax.numpy as jnp
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
import pytest

import liesel_ptm as ptm
from liesel_ptm.bspline import PTMSpline


@pytest.mark.parametrize(
    "spline_kind,rowwise", [("ptm", False), ("onion", False), ("onion", True)]
)
@pytest.mark.parametrize("standardized", [False, True])
def test_ptm_dist_matches_transformation_likelihood(spline_kind, rowwise, standardized):
    knots_class = ptm.PTMKnots if spline_kind == "ptm" else ptm.OnionKnots
    knots = knots_class(-4.0, 4.0, nparam=6).knots
    coef = jnp.linspace(-0.2, 0.2, 6)
    if rowwise:
        coef = jnp.array([0.5, 1.0, 1.5])[:, None] * coef
    loc = lsl.Var.new_value(jnp.array([-0.2, 0.0, 0.2]), name="loc")
    scale = lsl.Var.new_value(jnp.array([0.8, 1.0, 1.2]), name="scale")
    shape = lsl.Var.new_param(coef, name="delta")
    node = ptm.PTMDist(
        knots,
        loc,
        scale,
        shape,
        bspline=spline_kind,
        centered=standardized,
        scaled=standardized,
    )
    response = lsl.Var.new_obs(jnp.array([-0.5, 0.1, 0.6]), dist=node, name="y")
    model = lsl.Model(response)
    spline_class = PTMSpline if spline_kind == "ptm" else ptm.OnionSpline
    expected = ptm.LocScaleTransformationDist(
        coef=coef,
        loc=loc.value,
        scale=scale.value,
        bspline=spline_class(knots),
        centered=standardized,
        scaled=standardized,
    )
    actual = node.init_dist()
    assert isinstance(actual, ptm.LocScaleTransformationDist)
    np.testing.assert_allclose(
        actual.log_prob(response.value), expected.log_prob(response.value), atol=1e-6
    )
    np.testing.assert_allclose(
        actual.cdf(response.value), expected.cdf(response.value), atol=1e-6
    )
    assert np.isfinite(model.log_prob)
    repeated = node.init_dist()
    assert isinstance(repeated, ptm.LocScaleTransformationDist)
    assert repeated.bspline is actual.bspline
    updated = model.update_state({shape.name: coef * 1.2})
    assert not np.allclose(updated["y_log_prob"].value, model.state["y_log_prob"].value)


@pytest.mark.parametrize("variant", ["identity", "ptm_nj", "onion_nj", "unknown"])
def test_ptm_dist_rejects_removed_and_unknown_spline_variants(variant):
    knots = ptm.PTMKnots(-4.0, 4.0, nparam=6).knots
    with pytest.raises(ValueError, match="Unsupported spline"):
        ptm.PTMDist(knots, 0.0, 1.0, jnp.zeros(6), bspline=variant)


@pytest.mark.parametrize("invalid_slope", ["invalid"])
def test_ptm_dist_preserves_ptm_tail_configuration(invalid_slope):
    knots = ptm.PTMKnots(-4.0, 4.0, nparam=6).knots
    node = ptm.PTMDist(
        knots,
        0.0,
        1.0,
        jnp.linspace(-0.4, 0.4, 6),
        trafo_lambda=0.2,
        trafo_target_slope="continue_linearly",
    )
    actual = node.init_dist()
    expected = ptm.LocScaleTransformationDist(
        coef=jnp.linspace(-0.4, 0.4, 6),
        loc=0.0,
        scale=1.0,
        bspline=PTMSpline(knots, eps=0.2, continue_linearly=True),
    )
    np.testing.assert_allclose(
        actual.log_prob(jnp.array([-10.0, 10.0])),
        expected.log_prob(jnp.array([-10.0, 10.0])),
    )
    with pytest.raises(ValueError, match="Unsupported tail slope"):
        ptm.PTMDist(knots, 0.0, 1.0, jnp.zeros(6), trafo_target_slope=invalid_slope)


@pytest.mark.parametrize("legacy", [False, True])
def test_convenience_matches_manual_likelihood_and_prior_at_nonzero_coefficients(
    legacy,
):
    y = jnp.array([-0.8, 0.2, 1.1])
    bounds = (-3.0, 5.0)
    # Independent reference construction, including the explicit shape-intercept
    # recipe used before the convenience constructors were introduced.
    loc = gam.AdditivePredictor("loc_terms" if legacy else "loc", intercept=not legacy)
    scale = gam.AdditivePredictor(
        "scale_terms" if legacy else "scale", inv_link=jnp.exp, intercept=not legacy
    )
    shape = gam.MVAdditivePredictor.from_random_walk(
        "delta", ndim=4, intercept=False
    ).constrain("sumzero_coef")
    shape_builder = gam.MVTermBuilder.from_predictor(
        shape, term_builder=gam.TermBuilder.from_dict({})
    )
    shape.intercept = shape_builder.intercept(scale=gam.scale_wb(0.1, scale=0.05))
    loc_full, scale_full = loc, scale
    if legacy:
        training_y = lsl.Data(y)
        beta = ptm.LocationIntercept(
            training_y, loc, scale.linear_predictor, name="beta0"
        )
        loc_full = gam.AdditivePredictor("loc", intercept=beta)
        loc_full += loc
        gamma = ptm.LogScaleIntercept(
            training_y, loc_full, scale.linear_predictor, name="gamma0"
        )
        scale_full = gam.AdditivePredictor("scale", inv_link=jnp.exp, intercept=gamma)
        scale_full += scale.linear_predictor
    knots = (ptm.PTMKnots if legacy else ptm.OnionKnots)(*bounds, nparam=4)
    manual = lsl.Model(
        lsl.Var.new_obs(
            y,
            name="y",
            dist=ptm.PTMDist(
                knots.knots,
                loc_full,
                scale_full,
                shape,
                bspline="ptm" if legacy else "onion",
                centered=not legacy,
                scaled=not legacy,
            ),
        )
    )
    model = (ptm.LegacyPTM if legacy else ptm.PTM)(y, nshape=4, bounds=bounds)
    # Upstream automatic and explicit intercepts use different parameter names.
    manual_intercept = shape.intercept
    auto_intercept = model.pred.shape.intercept
    assert isinstance(manual_intercept, gam.MultivariateIntercept)
    assert isinstance(auto_intercept, gam.MultivariateIntercept)
    assert manual_intercept.scale is not None and auto_intercept.scale is not None
    manual_intercept.coef.name = auto_intercept.coef.name
    manual_variance = manual_intercept.scale.value_node[0]
    auto_variance = auto_intercept.scale.value_node[0]
    assert isinstance(manual_variance, lsl.Var) and isinstance(auto_variance, lsl.Var)
    manual_variance.bijected_var.name = auto_variance.bijected_var.name
    for predictors in ((loc, scale), (model.pred.loc, model.pred.scale)):
        builder = gam.TermBuilder.from_dict({"x": jnp.array([-1.0, 0.0, 1.0])})
        predictors[0].append(builder.slin("x", scale=1.0, name="m.x"))
        predictors[1].extend([builder.slin("x", scale=0.5, name="s.x")])
    assert model.parameters.keys() == manual.parameters.keys()
    position = {
        name: var.value
        + jnp.linspace(0.03, 0.09, var.value.size).reshape(var.value.shape)
        for name, var in model.parameters.items()
    }
    for graph in (manual, model):
        graph.state = graph.update_state(position)
    np.testing.assert_allclose(model.log_prior, manual.log_prior, atol=2e-6)
    np.testing.assert_allclose(model.log_lik, manual.log_lik, atol=2e-6)
    for name in ("loc", "scale", "delta"):
        np.testing.assert_allclose(
            model.vars[name].value, manual.vars[name].value, atol=2e-6
        )
    np.testing.assert_allclose(model.pred.shape.value.sum(), 0.0, atol=1e-6)
    response = model.response if isinstance(model, ptm.LegacyPTM) else model.vars["y"]
    node = response.dist_node
    assert isinstance(node, ptm.PTMDist)
    transformation = node.init_dist()
    assert isinstance(transformation, ptm.LocScaleTransformationDist)
    np.testing.assert_array_equal(transformation.bspline.knots, knots.knots)


@pytest.mark.parametrize("cls,minimum", [(ptm.PTM, 2), (ptm.LegacyPTM, 3)])
def test_minimum_shape_dimension_has_finite_density(cls, minimum):
    model = cls(jnp.array([-0.5, 0.1, 0.8]), nshape=minimum)
    assert isinstance(model, lsl.Model)
    assert np.isfinite(model.log_prob)
    assert model.pred.shape.ndim == minimum
    assert model.pred.shape.latent_ndim == minimum - 1


@pytest.mark.parametrize("cls", [ptm.PTM, ptm.LegacyPTM])
def test_copy_keeps_namespace_linked_to_copied_graph(cls):
    model = cls(jnp.array([-0.5, 0.1, 0.8]), nshape=4)
    copied = model.copy()
    assert isinstance(copied, cls)
    builder = gam.TermBuilder.from_dict({"x": jnp.array([-1.0, 0.0, 1.0])})
    copied.pred.loc += builder.slin("x", scale=1.0)
    for name in ("loc", "scale", "shape"):
        predictor = getattr(copied.pred, name)
        assert predictor is copied.vars[predictor.name]
        assert predictor is not getattr(model.pred, name)
    assert set(model.parameters) < set(copied.parameters)
    assert not model.pred.loc.terms


def test_native_predictors_accept_late_shape_terms_and_honor_graph_lock():
    model = ptm.PTM(jnp.array([-0.5, 0.1, 0.8]), nshape=4)
    builder = gam.TermBuilder.from_dict({"x": jnp.array([-1.0, 0.0, 1.0])})
    shape_builder = gam.MVTermBuilder.from_predictor(
        model.pred.shape, term_builder=builder
    )
    term = shape_builder.slin("x", scale=1.0)
    model.pred.shape.extend([term])
    assert term.coef.name in model.parameters
    assert model.pred.shape.value.shape == (3, 4)
    loc_term = builder.slin("x", scale=1.0, name="m.x")
    model.locked = True
    with pytest.raises(RuntimeError, match="locked"):
        model.pred.loc.append(loc_term)


def test_supplied_predictors_and_shape_scale_use_native_inputs():
    loc = gam.AdditivePredictor("mu")
    scale = gam.AdditivePredictor("sigma", inv_link=jnp.exp)
    shape = gam.MVAdditivePredictor.from_random_walk(
        "d", ndim=4, intercept_scale=0.3
    ).constrain("sumzero_coef")
    response = lsl.Var.new_obs(jnp.array([-0.5, 0.1, 0.8]), name="response")
    model = ptm.PTM(response, nshape=4, loc=loc, scale=scale, shape=shape)
    assert (
        model.pred.loc is loc
        and model.pred.scale is scale
        and model.pred.shape is shape
    )
    assert model.vars["response"] is response
    assert isinstance(shape.intercept, gam.MultivariateIntercept)
    assert shape.intercept.scale is not None
    np.testing.assert_allclose(shape.intercept.scale.value, 0.3)
    fixed = ptm.PTM(response.value, nshape=4, shape_scale=0.25)
    assert isinstance(fixed.pred.shape.intercept, gam.MultivariateIntercept)
    assert fixed.pred.shape.intercept.scale is not None
    np.testing.assert_allclose(fixed.pred.shape.intercept.scale.value, 0.25)
    assert (
        len(fixed.parameters) == 3
    )  # Two intercepts and the shape vector; no scale parameter.
    with pytest.raises(ValueError, match="shape_scale"):
        ptm.PTM(response.value, nshape=4, shape=shape, shape_scale=0.25)


@pytest.mark.parametrize(
    "cls,nshape",
    [
        (ptm.PTM, 1),
        (ptm.LegacyPTM, 2),
        (ptm.PTM, 4.5),
        (ptm.PTM, True),
    ],
)
def test_invalid_shape_dimensions(cls, nshape):
    with pytest.raises(ValueError, match="nshape"):
        cls([0.0, 1.0], nshape=nshape)


@pytest.mark.parametrize(
    "bounds", [(1.0, -1.0), (1.0, 1.0), (0.0, np.inf), (np.nan, 1.0), (-1.0, 0.0, 1.0)]
)
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="bounds"):
        ptm.PTM([0.0, 1.0], nshape=4, bounds=bounds)


def test_invalid_supplied_predictors():
    shape = gam.MVAdditivePredictor.from_random_walk("delta", ndim=3).constrain(
        "sumzero_coef"
    )
    with pytest.raises(ValueError, match="nshape dimensions"):
        ptm.PTM([0.0, 1.0], nshape=4, shape=shape)
    with pytest.raises(ValueError, match="intercept-free"):
        ptm.LegacyPTM([0.0, 1.0], nshape=4, loc=gam.AdditivePredictor("m"))


def test_supplied_modern_scale_must_be_positive_but_need_not_use_exp():
    with pytest.raises(ValueError, match="positive"):
        ptm.PTM([0.0, 1.0], nshape=4, scale=gam.AdditivePredictor("sigma"))
    scale = gam.AdditivePredictor("sigma", inv_link=jax.nn.softplus)
    model = ptm.PTM([0.0, 1.0], nshape=4, scale=scale)
    assert model.pred.scale is scale
    assert np.isfinite(model.log_prob)


def test_modern_scale_rejects_identity_link_even_with_positive_initial_value():
    scale = gam.AdditivePredictor("sigma", intercept=lsl.Var.new_param(1.0, name="s0"))
    with pytest.raises(ValueError, match="positive inverse link"):
        ptm.PTM([0.0, 1.0], nshape=4, scale=scale)


@pytest.mark.parametrize("cls", [ptm.PTM, ptm.LegacyPTM])
def test_common_response_handle_supports_distribution_summaries(cls):
    model = cls([-0.5, 0.1, 0.8], nshape=4)
    samples = {
        name: jnp.broadcast_to(var.value, (2, 2) + var.value.shape)
        for name, var in model.parameters.items()
    }
    summary = ptm.summarise_intercept_dist(
        model.response, model.pred.shape.intercept, samples, rgrid=5
    )
    assert {"density", "cdf", "transformation"} <= set(summary["quantity"])
    assert np.isfinite(summary["mean"]).all()


@pytest.mark.parametrize("link", [None, jax.nn.softplus, lambda x: jnp.exp(2 * x)])
def test_supplied_legacy_scale_requires_exponential_link(link):
    scale = gam.AdditivePredictor("scale_terms", inv_link=link, intercept=False)
    with pytest.raises(ValueError, match="inv_link=jnp.exp"):
        ptm.LegacyPTM([0.0, 1.0], nshape=4, scale=scale)


@pytest.mark.parametrize("cls", [ptm.PTM, ptm.LegacyPTM])
def test_failed_constructor_leaves_response_reusable(cls):
    response = lsl.Var.new_obs(jnp.array([-0.5, 0.1, 0.8]), name="loc")
    with pytest.raises(RuntimeError, match="Duplicate"):
        cls(response, nshape=4)
    assert response.dist_node is None
    assert response.model is None
    response.name = "y"
    model = cls(response, nshape=4)
    assert model.vars["y"] is response


def test_failed_constructor_releases_nodes_after_late_failure(monkeypatch):
    response = lsl.Var.new_obs(jnp.array([-0.5, 0.1, 0.8]), name="y")
    initialize = lsl.Model.__init__

    def fail_after_initialization(*args, **kwargs):
        initialize(*args, **kwargs)
        raise RuntimeError("late failure")

    with monkeypatch.context() as patch:
        patch.setattr(lsl.Model, "__init__", fail_after_initialization)
        with pytest.raises(RuntimeError, match="late failure"):
            ptm.PTM(response, nshape=4)
    assert response.dist_node is None and response.model is None
    assert np.isfinite(ptm.PTM(response, nshape=4).log_prob)


@pytest.mark.parametrize("argument", ["loc", "scale"])
def test_legacy_constructor_rejects_reserved_predictor_names(argument):
    predictor = gam.AdditivePredictor(argument, inv_link=jnp.exp, intercept=False)
    with pytest.raises(ValueError, match="reserved"):
        ptm.LegacyPTM(
            [0.0, 1.0],
            nshape=4,
            loc=predictor if argument == "loc" else None,
            scale=predictor if argument == "scale" else None,
        )


def test_fixed_shape_scale_accepts_integer_and_rejects_bool():
    model = ptm.PTM([0.0, 1.0], nshape=4, shape_scale=1)
    intercept = model.pred.shape.intercept
    assert isinstance(intercept, gam.MultivariateIntercept)
    assert intercept.scale is not None
    np.testing.assert_allclose(intercept.scale.value, 1.0)
    assert len(model.parameters) == 3
    with pytest.raises(TypeError, match="shape_scale"):
        ptm.PTM([0.0, 1.0], nshape=4, shape_scale=True)
