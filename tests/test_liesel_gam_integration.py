"""PTM contracts at the liesel-gam boundary."""

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
import pytest
from liesel_gam.iwls_proposals import (
    GaussianLocIWLSProposal,
    GaussianScaleIWLSProposal,
    apply_gaussian_iwls_spec_loc,
    gaussian_iwls_spec_loc,
)

import liesel_ptm as ptm


def structured_model():
    x = jnp.linspace(-1.0, 1.0, 12)
    registry = gam.DictRegistry({"x": x, "x2": x**2})
    model = ptm.LocScalePTM(
        x + jnp.sin(5 * x),
        ptm.PTMKnots(-4.0, 4.0, nparam=6).knots,
        intercepts="compute",
        bspline="identity",
    )
    for predictor, name in ((model.loc, "m"), (model.scale, "s")):
        predictor += gam.TermBuilder(registry, prefix_names_by=name).slin(
            "x + x2",
            name=name,
            scale=gam.ScaleIG(0.6, 2.0, 0.3),
            inference=None,
        )
    return model.build()


def test_gaussian_proposals_use_current_scale_and_prior_precision():
    model = structured_model()
    model.setup_default_mcmc_kernels(strategy="iwls-nuts")
    loc, scale = model.loc.terms["m"], model.scale.terms["s"]
    state = model.graph.update_state(
        {
            loc.scale.value_node[0].name: 0.9**2,
            scale.coef.name: jnp.array([0.2, -0.1]),
        }
    )
    for term, cls in (
        (loc, GaussianLocIWLSProposal),
        (scale, GaussianScaleIWLSProposal),
    ):
        callback = term.coef.inference.kernel_kwargs["chol_info_fn"]
        assert isinstance(callback.__self__, cls)
        basis = term.basis.value
        tau = model.graph.extract_position([term.scale.name], state)[term.scale.name]
        weights = (
            jnp.asarray(2.0)
            if term is scale
            else 1
            / model.graph.extract_position([model.scale.name], state)[model.scale.name]
            ** 2
        )
        weighted = basis * (weights[:, None] if jnp.ndim(weights) else weights)
        expected = basis.T @ weighted + jnp.eye(basis.shape[-1]) / tau**2
        expected += 1e-6 * jnp.diag(expected).mean() * jnp.eye(basis.shape[-1])
        chol = jax.jit(callback)(state)
        np.testing.assert_allclose(chol @ chol.T, expected, rtol=2e-6)
        assert term.coef.name in model.graph.parameters
        assert term.scale.value_node[0].name in model.graph.parameters
    results = model.run_mcmc(
        seed=42,
        warmup=50,
        posterior=4,
        num_chains=1,
        warm_start=False,
        strategy="manual",
    )
    samples = results.get_posterior_samples()
    assert set(samples) == set(model.graph.parameters)
    assert all(np.isfinite(value).all() for value in samples.values())


def test_ptm_inverse_gamma_gibbs_uses_latent_coefficients_and_transformed_penalty():
    knots = ptm.PTMKnots(-4.0, 4.0, nparam=6).knots
    coef = ptm.PTMCoef.new_rw1_sumzero_ig(knots, 2.0, 0.3, name="delta")
    model = lsl.Model(coef)
    variance = coef.scale.value_node[0]
    assert isinstance(variance, lsl.Var)
    spec = variance.inference
    assert isinstance(spec, gs.MCMCSpec)
    kernel = spec.kernel([variance.name], **spec.kernel_kwargs)
    values = jnp.linspace(-0.3, 0.3, coef.latent_coef.value.size)
    state = model.update_state({coef.latent_coef.name: values})
    key = jax.random.key(42)
    epoch = gs.EpochConfig(gs.EpochType.POSTERIOR, 1, 1, None).to_state(0, 0)
    kernel.set_model(gs.LieselInterface(model))
    outcome = kernel.transition(key, kernel.init_state(key, state), state, epoch)
    shape = 2.0 + jnp.linalg.matrix_rank(coef.penalty) / 2
    rate = 0.3 + values @ coef.penalty @ values / 2
    expected = rate / jax.random.gamma(key, shape)
    draw = model.extract_position([variance.name], outcome.model_state)[variance.name]
    np.testing.assert_allclose(draw, expected)
    constrained = model.extract_position([coef.name], state)[coef.name]
    np.testing.assert_allclose(constrained.sum(), 0.0, atol=1e-6)


def test_legacy_noncentered_inverse_gamma_rejects_invalid_gibbs_update():
    knots = ptm.PTMKnots(-4.0, 4.0, nparam=6).knots
    with pytest.raises(ValueError, match="require centered coefficients"):
        ptm.PTMCoef.new_rw1_fromzero_ig(knots, 2.0, 0.3, noncentered=True)


@pytest.mark.parametrize("kind", ["plain", "anisotropic", "factored"])
def test_upstream_gaussian_iwls_limits_are_explicit(kind):
    x = jnp.linspace(-1.0, 1.0, 12)
    builder = gam.TermBuilder(gam.DictRegistry({"x": x, "x2": x**2}))
    if kind == "plain":
        term = builder.lin("x", inference=gs.MCMCSpec(gs.NUTSKernel))
    elif kind == "factored":
        term = builder.slin("x", scale=1.0, factor_scale=True)
    else:
        term = gam.StrctInteractionTerm(
            builder.slin("x", scale=1.0),
            builder.slin("x2", scale=1.0),
            name="interaction",
            inference=gs.MCMCSpec(gs.NUTSKernel),
        )
    predictor = gam.AdditivePredictor("loc", intercept=False)
    predictor += term
    previous = term.coef.inference
    if kind == "factored":
        assert isinstance(term, gam.StrctLinTerm)
        with pytest.raises(ValueError, match="scale-factored"):
            gaussian_iwls_spec_loc(term)
    else:
        apply_gaussian_iwls_spec_loc(predictor)
        assert term.coef.inference is previous
        if kind == "anisotropic":
            assert isinstance(term, gam.StrctInteractionTerm)
            with pytest.raises(NotImplementedError, match="isotropic"):
                gaussian_iwls_spec_loc(term)
