"""Public PTM construction must match the underlying transformation likelihood."""

import jax.numpy as jnp
import liesel.model as lsl
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
