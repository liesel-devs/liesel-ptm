"""Contracts relied on when delegating inference to Liesel."""

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.flatten_util import ravel_pytree

import liesel_ptm as ptm
from demo import reference_original
from liesel_ptm.gam.var import LinearTerm
from liesel_ptm.iwls_proposals import CholInfo, PTMCholInfoFixed


@pytest.mark.parametrize("provider", [CholInfo, PTMCholInfoFixed])
def test_fixed_curvature_uses_current_state_and_conditional_posterior(provider):
    design = jnp.array([[1.0, -1.0], [1.0, 0.5], [1.0, 2.0]])
    beta = lsl.Var.new_param(
        jnp.zeros(2), lsl.Dist(tfd.Normal, loc=0.0, scale=2.0), name="beta"
    )
    sigma = lsl.Var.new_param(1.0, lsl.Dist(tfd.HalfNormal, scale=1.0), name="sigma")
    loc = lsl.Var(lsl.Calc(lambda b: design @ b, beta), name="loc")
    y = lsl.Var.new_obs(
        jnp.array([0.0, 1.0, 3.0]), lsl.Dist(tfd.Normal, loc, sigma), name="y"
    )
    model = lsl.Model(y)
    interface = gs.LieselInterface(model)
    kwargs = {"coef_name": "beta", "model": interface, "model_state": model.state}
    if provider is PTMCholInfoFixed:
        kwargs.update(coef_scale_name="sigma", penalty=jnp.eye(2))
    info = provider(**kwargs)
    state = model.update_state({"sigma": 1.8, "beta": jnp.array([0.2, -0.3])})
    raw, regularized = jax.jit(info.current_fisher_info)(state)
    expected = design.T @ design / 1.8**2 + jnp.eye(2) / 2.0**2
    if provider is CholInfo:
        expected += 1e-6 * jnp.eye(2)
    np.testing.assert_allclose(raw, expected, rtol=2e-6)
    np.testing.assert_allclose(regularized, expected, rtol=2e-6)
    assert not np.allclose(raw, info.fisher_info)
    np.testing.assert_array_equal(beta.value, jnp.zeros(2))
    assert float(sigma.value) == 1.0


def legacy_linear_model():
    x = jnp.linspace(-1.0, 1.0, 12)
    model = ptm.LocScalePTM(
        x + jnp.sin(5 * x),
        ptm.PTMKnots(-4.0, 4.0, nparam=6).knots,
        intercepts="compute",
        bspline="identity",
    )
    for predictor, name in ((model.loc, "m"), (model.scale, "s")):
        predictor += LinearTerm(
            jnp.c_[x, x**2],
            name=name,
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        )
    return model.build()


def test_optimization_checks_curvature_without_changing_initial_state():
    model = legacy_linear_model()
    before = model.graph.extract_position(model.graph.parameters)
    with pytest.warns(FutureWarning, match="liesel.goose.optim_flat is deprecated"):
        result = model.optim(
            stopper=gs.Stopper(max_iter=3, patience=2),
            test_for_positive_definiteness=True,
        )
    assert set(result.position) == set(before)
    assert all(np.isfinite(value).all() for value in result.position.values())
    for name, value in before.items():
        np.testing.assert_array_equal(model.graph.parameters[name].value, value)


def test_observed_iwls_blocks_and_cached_sampling(tmp_path, monkeypatch):
    model = legacy_linear_model()
    model.setup_default_mcmc_kernels(strategy="iwls-nuts")
    for param in model.graph.parameters.values():
        spec = param.inference
        assert spec.kernel is gs.IWLSKernel
        assert spec.kernel_kwargs["chol_info_fn"] is None
        assert spec.kernel_kwargs["fallback_chol_info"] == "identity"
    path = tmp_path / "samples.pkl"
    results = model.run_mcmc(
        seed=42,
        warmup=50,
        posterior=4,
        num_chains=1,
        thinning_posterior=2,
        warm_start=False,
        strategy="manual",
        cache_path=path,
    )
    samples = results.get_posterior_samples()
    assert set(samples) == set(model.graph.parameters)
    assert all(value.shape == (1, 2, 2) for value in samples.values())
    assert all(np.isfinite(value).all() for value in samples.values())

    def unexpected_build(*args, **kwargs):
        pytest.fail("Cached results must load before sampling or initialization")

    monkeypatch.setattr(model, "initialize", unexpected_build)
    monkeypatch.setattr(gs.EngineBuilder, "build", unexpected_build)
    cached = model.run_mcmc(seed=99, warmup=50, posterior=4, cache_path=path)
    for name, value in cached.get_posterior_samples().items():
        np.testing.assert_array_equal(value, samples[name])


def test_gaussian_iwls_geometry_still_targets_the_ptm_posterior():
    model = reference_original.build_model()
    shape = model.vars["delta"].intercept.coef
    model.state = model.update_state(
        {shape.name: jnp.linspace(-0.6, 0.6, shape.value.size)}
    )
    term = model.vars["loc_terms"].terms["m.x"]
    name = term.coef.name
    interface = gs.LieselInterface(model)
    flat, unravel = ravel_pytree(model.extract_position([name]))
    # Gaussian likelihood information plus the standard-normal prior precision.
    design = term.basis.value
    precision = design.T @ (design / model.vars["scale"].value[:, None] ** 2)
    precision += jnp.eye(flat.size)
    chol = jnp.linalg.cholesky(precision)
    kernel = gs.IWLSKernel(
        [name], chol_info_fn=lambda state: chol, initial_step_size=0.7
    )
    kernel.set_model(interface)
    logprob = gs.FlatInterfaceLogProb(interface, model.state, [name])
    key = jax.random.key(3)
    proposal_key, _ = jax.random.split(key)

    def proposal_mean(value):
        return value + 0.7**2 / 2 * jnp.linalg.solve(precision, logprob.grad(value))

    proposed = proposal_mean(flat) + 0.7 * jnp.linalg.solve(
        chol.T, jax.random.normal(proposal_key, flat.shape)
    )
    covariance_chol = jnp.linalg.cholesky(0.7**2 * jnp.linalg.inv(precision))
    forward = tfd.MultivariateNormalTriL(proposal_mean(flat), covariance_chol)
    backward = tfd.MultivariateNormalTriL(proposal_mean(proposed), covariance_chol)
    expected = jnp.minimum(
        1.0,
        jnp.exp(
            logprob(proposed)
            - logprob(flat)
            + backward.log_prob(flat)
            - forward.log_prob(proposed)
        ),
    )
    assert 0.0 < expected < 0.99  # Keep the target-density check nontrivial.
    epoch = gs.EpochConfig(gs.EpochType.POSTERIOR, 1, 1, None).to_state(0, 0)
    outcome = jax.jit(kernel.transition)(
        key, kernel.init_state(key, model.state), model.state, epoch
    )
    np.testing.assert_allclose(outcome.info.acceptance_prob, expected, atol=2e-5)
    retained = interface.extract_position([name], outcome.model_state)[name]
    np.testing.assert_allclose(
        retained,
        unravel(proposed)[name] if outcome.info.position_moved else term.coef.value,
    )
