from collections import Counter

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import liesel.optim as opt
import liesel_gam as gam
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from demo import reference_legacy


def setup(model, **kwargs):
    return model.setup_mcmc(seed=42, num_chains=2, **kwargs)


def rebuild(variables):
    return ptm.LegacyPTM(
        variables["y"].value,
        nshape=6,
        loc=variables["loc_terms"],
        scale=variables["scale_terms"],
        shape=variables["delta"],
    )


def test_intercept_formulas_match_empirical_moment_identification():
    y = jnp.array([1.0, 4.0, 8.0])
    m = jnp.array([0.3, 0.7, 1.1])
    g = jnp.array([-0.1, 0.2, 0.4])
    beta = jnp.sum((y - m) * jnp.exp(-g)) / jnp.sum(jnp.exp(-g))
    gamma = jnp.log(jnp.std((y - beta - m) * jnp.exp(-g)))
    np.testing.assert_allclose(ptm.LocationIntercept.compute_intercept(y, m, g), beta)
    np.testing.assert_allclose(
        ptm.LogScaleIntercept.compute_intercept(y, beta + m, g), gamma
    )


def test_shared_response_updates_and_full_data_shuffling_preserve_identification():
    model = reference_legacy.build_model()
    model.state = model.update_state(
        {
            model.vars["loc_terms"].terms["m.x"].coef.name: jnp.array([0.4]),
            model.vars["scale_terms"].terms["s.x"].coef.name: jnp.array([0.2]),
        }
    )
    position = model.extract_position(["beta0", "gamma0"])
    log_lik = model.log_lik
    data = model.extract_position(["x", "y"])
    batches = opt.Batches.from_model(model, batch_size=24, shuffle=True)
    assert set(batches.position_keys) == {"x", "y"}
    batches = batches.start_epoch(jax.random.key(13))
    shuffled = batches.get_batched_position(data, batch_index=0)
    assert not np.array_equal(shuffled["x"], data["x"])
    model.state = jax.jit(model.update_state)(shuffled)
    for name, value in position.items():
        np.testing.assert_allclose(model.vars[name].value, value, atol=2e-6)
    np.testing.assert_allclose(model.log_lik, log_lik, atol=2e-5)
    assert model.vars["y"].strong and model.response.weak
    model.vars["y"].value = shuffled["y"] + 5.0
    np.testing.assert_allclose(
        model.vars["beta0"].value, position["beta0"] + 5, atol=2e-6
    )
    np.testing.assert_allclose(
        model.vars["gamma0"].value, position["gamma0"], atol=2e-6
    )
    residual = (model.response.value - model.vars["loc"].value) / model.vars[
        "scale"
    ].value
    np.testing.assert_allclose(residual.mean(), 0.0, atol=2e-6)
    np.testing.assert_allclose(residual.std(), 1.0, atol=1e-6)


@pytest.mark.parametrize("keys", [("x", "y"), ("x",), ("y",)])
def test_legacy_rejects_joint_and_partial_minibatches(keys):
    model = reference_legacy.build_model()
    batch = {name: model.vars[name].value[:4] for name in keys}
    with pytest.raises((ValueError, RuntimeError)) as caught:
        jax.jit(model.update_state)(batch)
    messages = []
    error = caught.value
    while error is not None:
        messages.append(str(error))
        error = error.__cause__
    assert "LegacyPTM requires full-data fitting" in " ".join(messages)
    assert model.vars["x"].value.shape == model.vars["y"].value.shape == (24,)


@pytest.mark.parametrize("minibatch", [False, True])
@pytest.mark.parametrize("factory", [opt.LieselOptim, opt.LieselVI])
def test_legacy_optimization_and_vi_require_full_data(minibatch, factory):
    model = ptm.LegacyPTM([-0.5, 0.1, 0.8, 1.2], nshape=3)
    builder = gam.TermBuilder.from_dict({"x": jnp.linspace(-1.0, 1.0, 4)})
    model.pred.loc += builder.slin("x", scale=1.0)
    optimizer = factory(
        model,
        loss_monitor="train_full_data",
        optimizers=optax.adam(0.001),
        batch_size=2 if minibatch else None,
        stopper=opt.Stopper(epochs=1, patience=1),
        show_progress=False,
    )
    if minibatch:
        with pytest.raises((ValueError, RuntimeError)) as caught:
            optimizer.fit()
        assert "full-data fitting" in str(caught.value) + str(caught.value.__cause__)
    else:
        result = optimizer.fit()
        assert all(np.isfinite(value).all() for value in result.position_final.values())


@pytest.mark.parametrize("kernel_class", [gs.NUTSKernel, gs.HMCKernel])
def test_kernel_coverage_gaussian_information_and_shape_block(kernel_class):
    model = reference_legacy.build_model()
    shape_spec = gs.MCMCSpec(kernel_class, kernel_kwargs={"da_target_accept": 0.85})
    builder = setup(model, shape_spec=shape_spec)
    assert builder.positions_included == ["beta0", "gamma0"]
    assert Counter(k for ker in builder.kernels for k in ker.position_keys) == Counter(
        model.parameters.keys()
    )
    shape_keys = set(model.parental_submodel(model.vars["delta"]).parameters)
    shape_kernel = next(ker for ker in builder.kernels if isinstance(ker, kernel_class))
    assert set(shape_kernel.position_keys) == shape_keys
    assert shape_spec.kernel_group is None  # The caller's specification is unchanged.
    assert any(isinstance(ker, gs.GibbsKernel) for ker in builder.kernels)
    state = model.update_state(
        {model.vars["scale_terms"].terms["s.x"].coef.name: jnp.array([0.3])}
    )
    sigma = model.extract_position(["scale"], state)["scale"]
    for predictor_name in ("loc_terms", "scale_terms"):
        for term in model.vars[predictor_name].terms.values():
            ker = next(
                ker for ker in builder.kernels if term.coef.name in ker.position_keys
            )
            assert isinstance(ker, gs.IWLSKernel)
            basis = term.basis.value
            weights = (
                1 / sigma**2 if predictor_name == "loc_terms" else jnp.asarray(2.0)
            )
            tau = model.extract_position([term.scale.name], state)[term.scale.name]
            weighted = basis * (weights[:, None] if weights.ndim else weights)
            precision = basis.T @ weighted + term.basis.penalty.value / tau**2
            precision += 1e-6 * jnp.diag(precision).mean() * jnp.eye(precision.shape[0])
            chol = jax.jit(ker.chol_info_fn)(state)
            np.testing.assert_allclose(chol @ chol.T, precision, rtol=3e-6, atol=1e-6)


def test_shape_prior_precision_constraints_and_unclipped_variance():
    model = reference_legacy.build_model()
    shape = model.vars["delta"]
    intercept = shape.intercept
    reparam = intercept.dimension_reparam.value
    differences = jnp.diff(jnp.eye(6), axis=0)
    penalty = differences.T @ differences
    penalty /= jnp.linalg.norm(penalty, ord=jnp.inf)
    latent_penalty = intercept.dimension_penalty.value
    np.testing.assert_allclose(latent_penalty, reparam.T @ penalty @ reparam, atol=1e-6)
    covariance = reparam @ jnp.linalg.inv(latent_penalty) @ reparam.T
    np.testing.assert_allclose(covariance, jnp.linalg.pinv(penalty), atol=1e-5)
    np.testing.assert_allclose(reparam.sum(axis=0), 0.0, atol=1e-6)
    q = jnp.linspace(-0.2, 0.2, intercept.coef.value.size)
    prior = intercept.coef.dist_node.init_dist()
    expected = -0.5 * q @ latent_penalty @ q / intercept.scale.value**2
    np.testing.assert_allclose(
        prior.log_prob(q) - prior.log_prob(jnp.zeros_like(q)), expected, rtol=1e-5
    )
    variance = intercept.scale.value_node[0]
    unconstrained = variance.bijected_var
    state = model.update_state({unconstrained.name: jnp.array(-20.0)})
    value = model.extract_position([variance.name], state)[variance.name]
    np.testing.assert_allclose(value, jnp.exp(-20.0))
    assert np.isfinite(state[unconstrained.dist_node.name].value)


def test_legacy_transformation_has_gaussian_identity_and_unit_tail_slopes():
    model = reference_legacy.build_model()
    node = model.response.dist_node
    assert isinstance(node, ptm.PTMDist)
    transformation = node.init_dist()
    assert isinstance(transformation, ptm.LocScaleTransformationDist)
    spline = transformation.bspline
    tail = jnp.array([-20.0, 20.0])
    coef = jnp.linspace(-0.4, 0.4, 6)
    _, derivative = spline.dot_and_deriv(tail, coef[None, :])
    np.testing.assert_allclose(derivative, 1.0, atol=1e-6)
    dist = ptm.LocScaleTransformationDist(
        coef=jnp.zeros(6),
        bspline=spline,
        loc=0.0,
        scale=1.0,
        centered=False,
        scaled=False,
    )
    grid = jnp.linspace(-7.0, 7.0, 17)
    np.testing.assert_allclose(
        dist.log_prob(grid), tfd.Normal(0.0, 1.0).log_prob(grid), atol=2e-5
    )


@pytest.fixture(scope="module")
def sampled():
    model = reference_legacy.build_model()
    results = reference_legacy.fit(model)
    return model, results


def test_short_sampler_records_end_of_draw_intercepts_and_updates_all_blocks(sampled):
    model, results = sampled
    samples = results.get_posterior_samples()
    assert set(samples) == set(model.parameters) | {"beta0", "gamma0"}
    assert samples["beta0"].shape == samples["gamma0"].shape == (2, 12)
    assert all(np.isfinite(v).all() for v in samples.values())
    params = {name: samples[name] for name in model.parameters}

    def recompute(position):
        state = model.update_state(position, model.state)
        return model.extract_position(["beta0", "gamma0"], state)

    expected = jax.jit(jax.vmap(jax.vmap(recompute)))(params)
    for name in expected:
        np.testing.assert_allclose(samples[name], expected[name], atol=2e-6)
    for name in model.parameters:
        assert np.any(np.diff(samples[name], axis=1) != 0), name
    for values in results.get_posterior_acceptance_probabilities().values():
        assert np.isfinite(values).all()
    prediction = model.predict(
        samples,
        predict=["loc", "scale", "delta"],
        newdata={"x": jnp.array([-0.5, 0.5])},
    )
    assert prediction["loc"].shape == prediction["scale"].shape == (2, 12, 2)
    assert prediction["delta"].shape == (2, 12, 6)


def example_samples(model):
    samples = {
        name: jnp.broadcast_to(var.value, (2, 3) + var.value.shape)
        for name, var in model.parameters.items()
    }
    samples["beta0"] = jnp.arange(6, dtype=float).reshape(2, 3) + 10
    samples["gamma0"] = jnp.log(jnp.arange(6, dtype=float).reshape(2, 3) + 2)
    m = model.vars["loc_terms"].terms["m.x"].coef.name
    s = model.vars["scale_terms"].terms["s.x"].coef.name
    samples[m] = jnp.full((2, 3, 1), 0.4)
    samples[s] = jnp.full((2, 3, 1), 0.2)
    return samples


@pytest.mark.parametrize("nobs", [4, 24])
def test_prediction_reuses_draws_without_training_response_and_preserves_fit(
    monkeypatch,
    nobs,
):
    model = reference_legacy.build_model()
    samples = example_samples(model)
    before = [np.array(value) for value in jax.tree.leaves(model.state)]
    training_node = model.vars["beta0"].value_node[0].name
    original_submodel = lsl.Model.parental_submodel

    def inspect_graph(prediction_model, *args, **kwargs):
        submodel = original_submodel(prediction_model, *args, **kwargs)
        assert training_node not in submodel.nodes
        assert "y" not in submodel.vars
        for target in args:
            intercept = {"loc": "beta0", "scale": "gamma0"}[target.name]
            assert submodel.vars[intercept].strong
        return submodel

    monkeypatch.setattr(lsl.Model, "parental_submodel", inspect_graph)
    x = jnp.linspace(-0.8, 0.9, nobs)
    for chunk_size in (1, None):
        prediction = model.predict(
            samples,
            ["loc", "scale"],
            newdata={"x": x},
            chunk_size=chunk_size,
        )
        np.testing.assert_allclose(
            prediction["loc"], samples["beta0"][..., None] + 0.4 * x, atol=2e-6
        )
        np.testing.assert_allclose(
            prediction["scale"],
            jnp.exp(samples["gamma0"][..., None] + 0.2 * x),
            rtol=2e-6,
        )
        for name in ("loc", "scale"):
            variable_prediction = model.vars[name].predict(
                samples, newdata={"x": x}, chunk_size=chunk_size
            )
            np.testing.assert_allclose(variable_prediction, prediction[name], rtol=2e-6)
    assert model.vars["beta0"].weak and model.vars["gamma0"].weak
    for previous, current in zip(before, jax.tree.leaves(model.state), strict=True):
        np.testing.assert_array_equal(previous, current)


def test_prediction_defaults_and_model_log_likelihood_follow_liesel_interface():
    model = reference_legacy.build_model()
    samples = example_samples(model)
    prediction = model.predict(samples)
    assert prediction.keys() == model.vars.keys()
    np.testing.assert_array_equal(prediction["beta0"], samples["beta0"])
    target = "_model_log_lik"
    result = model.predict(samples, [target])
    nodes, variables = model.copy_nodes_and_vars()
    for name in ("beta0", "gamma0"):
        variables[name].value_node = lsl.Value(samples[name][0, 0])
    manual = lsl.Model(variables[model.response.name])
    manual.state = manual.update_state(
        {name: value[0, 0] for name, value in samples.items()}
    )
    np.testing.assert_allclose(result[target][0, 0], manual.log_lik, atol=2e-5)


def test_default_prediction_with_new_covariates_omits_unreplaced_observations():
    model = reference_legacy.build_model()
    samples = example_samples(model)
    x = jnp.array([-0.5, 0.5])
    result = model.predict(samples, newdata={"x": x})
    assert "y" not in result
    np.testing.assert_allclose(
        result["loc"], samples["beta0"][..., None] + 0.4 * x, atol=2e-6
    )
    assert result["scale"].shape == (2, 3, 2)
    result = model.predict(samples, newdata={"x": x, "y": jnp.zeros(2)})
    assert result["y"].shape == (2, 3, 2)
    assert model.response.name in result
    np.testing.assert_array_equal(result[model.response.name], result["y"])
    node = model.response.dist_node
    assert node is not None
    prediction = model.predict(samples, [node.name], {"x": x, "y": jnp.zeros(2)})
    assert prediction[node.name].shape == (2, 3, 2)
    assert np.isfinite(prediction[node.name]).all()
    assert model.vars["y"].value.shape == (24,)


def test_prediction_accepts_known_unused_covariates_but_rejects_unknown_data():
    model = reference_legacy.build_model()
    builder = gam.TermBuilder.from_dict({"z": jnp.linspace(-1.0, 1.0, 24)})
    model.pred.scale += builder.slin("z", scale=1.0, name="s.z")
    samples = example_samples(model)
    x = jnp.array([-0.5, 0.5])
    result = model.predict(samples, ["loc"], {"x": x, "z": jnp.ones(2)})
    np.testing.assert_allclose(
        result["loc"], samples["beta0"][..., None] + 0.4 * x, atol=2e-6
    )
    with pytest.raises(KeyError, match="not part of the model"):
        model.predict(samples, ["loc"], {"unknown": x})


@pytest.mark.parametrize("missing", ["beta0", "gamma0"])
def test_prediction_requires_recorded_intercepts(missing):
    model = reference_legacy.build_model()
    samples = example_samples(model)
    samples.pop(missing)
    with pytest.raises(ValueError, match="Missing recorded intercept"):
        model.predict(samples, predict=["loc"])
    with pytest.raises(ValueError, match="Missing recorded intercept"):
        model.vars["loc"].predict(samples)


def test_prediction_checks_intercept_axes_and_supports_custom_names():
    model = reference_legacy.build_model()
    samples = example_samples(model)
    samples["gamma0"] = samples["gamma0"][:1]
    with pytest.raises(ValueError, match="matching sample axes"):
        model.predict(samples, predict=["loc"])
    samples = example_samples(model)
    for old, new in (("beta0", "offset"), ("gamma0", "log_spread")):
        model.vars[old].name = new
        samples[new] = samples.pop(old)
    builder = setup(model)
    assert builder.positions_included == ["offset", "log_spread"]
    result = model.predict(samples, predict=["offset", "log_spread"])
    np.testing.assert_array_equal(result["offset"], samples["offset"])
    np.testing.assert_array_equal(result["log_spread"], samples["log_spread"])
    node_name = model.vars["offset"].value_node.name
    result = model.predict(samples, predict=[node_name])
    np.testing.assert_array_equal(result[node_name], samples["offset"])


@pytest.mark.parametrize("kind", ["plain", "anisotropic", "factored"])
def test_setup_rejects_unsupported_location_terms(kind):
    model = reference_legacy.build_model()
    variables = model.copy_vars()
    predictor = variables["loc_terms"]
    builder = gam.TermBuilder.from_dict(
        {"z": jnp.linspace(-1, 1, 24), "w": jnp.linspace(0, 1, 24)}
    )
    if kind == "plain":
        term = builder.lin("z")
    elif kind == "factored":
        term = builder.slin("z", scale=1.0, factor_scale=True)
    else:
        term = gam.StrctInteractionTerm(
            builder.slin("z", scale=1.0),
            builder.slin("w", scale=1.0),
            name="interaction",
        )
    predictor += term
    changed = rebuild(variables)
    with pytest.raises(ValueError, match="structured term|scale-factored"):
        setup(changed)


def test_setup_rejects_overlap_and_unexplained_parameters():
    model = reference_legacy.build_model()
    model.pred.scale += model.pred.loc.terms["m.x"]
    with pytest.raises(ValueError, match="overlap"):
        setup(model)
    model = reference_legacy.build_model()
    extra = lsl.Var.new_param(0.0, dist=lsl.Dist(tfd.Normal, 0.0, 1.0), name="extra")
    model.add(extra)
    with pytest.raises(ValueError, match="outside .pred"):
        setup(model)


def test_shape_gibbs_and_missing_scale_inference_are_covered():
    variables = reference_legacy.build_model().copy_vars()
    shape = variables["delta"]
    builder = gam.MVTermBuilder.from_predictor(
        shape, term_builder=gam.TermBuilder.from_dict({})
    )
    shape_scale = gam.ScaleIG(0.1, 1.0, 0.001)
    shape.intercept = builder.intercept(scale=shape_scale)
    model = rebuild(variables)
    missing = model.vars["loc_terms"].terms["m.ps(x)"].scale.value_node[0]
    missing.inference = None
    engine_builder = setup(model)
    shape_variance = shape_scale.value_node[0]
    assert any(
        isinstance(ker, gs.GibbsKernel) and shape_variance.name in ker.position_keys
        for ker in engine_builder.kernels
    )
    assert any(
        isinstance(ker, gs.NUTSKernel) and ker.position_keys == (missing.name,)
        for ker in engine_builder.kernels
    )


def test_setup_rejects_unconstrained_shape_and_invalid_shape_sampler():
    model = reference_legacy.build_model()
    with pytest.raises(ValueError, match="NUTSKernel or HMCKernel"):
        setup(model, shape_spec=gs.MCMCSpec(gs.IWLSKernel))
    variables = model.copy_vars()
    variables["delta"] = gam.MVAdditivePredictor.from_random_walk("delta", ndim=6)
    with pytest.raises(ValueError, match="sumzero_coef"):
        rebuild(variables)
    # The native predictor remains mutable after construction; re-check its matrix.
    model.vars["delta"].intercept.dimension_reparam.value = jnp.eye(6)[:, :5]
    with pytest.raises(ValueError, match="sumzero_coef"):
        setup(model)
