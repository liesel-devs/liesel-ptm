import logging
import time
import timeit
from collections.abc import Iterator

import jax
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from plotnine import aes, geom_point, ggplot  # noqa
from scipy import stats

import liesel_ptm as ptm
import liesel_ptm.nodes as nd
from liesel_ptm import ptm_ls
from liesel_ptm.datagen import sample_shape
from liesel_ptm.liesel_internal import splines

kn = splines.create_equidistant_knots

nparam = 7
x = np.random.uniform(size=20)
knots = kn(x, order=3, n_params=20)
X = lsl.obs(np.random.uniform(-2, 2, size=(10, 3)), name="x1")
coef = lsl.param(np.random.uniform(0, 1, size=(3,)), name="b")


@pytest.fixture(scope="module")
def samples():
    path = "tests/resources/nonnormal_linear/results.pickle"
    results = gs.engine.SamplingResults.pkl_load(path)
    yield results.get_posterior_samples()


@pytest.fixture(scope="module")
def model() -> Iterator[ptm_ls.PTMLocScale]:
    df = pd.read_csv("tests/resources/nonnormal_linear/data.csv")

    x = df.x.to_numpy()
    y = df.y.to_numpy()

    path = "tests/resources/nonnormal_linear/knots.pickle"
    model = ptm_ls.PTMLocScale._from_knots(path, y)
    model.loc_model += ptm.LinearTerm(x, name="x_loc")
    model.log_scale_model += ptm.LinearTerm(x, name="x_scale")
    yield model


@pytest.fixture(scope="module")
def pred(model, samples) -> Iterator[ptm_ls.PTMLocScalePredictions]:
    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=None, x_scale=None)
    yield pred


@pytest.fixture(scope="module")
def samples_lin3():
    path = "tests/resources/normal_linear3/results.pickle"
    results = gs.engine.SamplingResults.pkl_load(path)
    yield results.get_posterior_samples()


@pytest.fixture(scope="module")
def model_lin3() -> Iterator[ptm_ls.PTMLocScale]:
    df = pd.read_csv("tests/resources/normal_linear3/data.csv")

    x = df[["x0", "x1", "x2"]].to_numpy()
    y = df.y.to_numpy()

    path = "tests/resources/normal_linear3/knots.pickle"
    model = ptm_ls.PTMLocScale._from_knots(path, y)
    model.loc_model += ptm.LinearTerm(x, name="x_loc")
    model.log_scale_model += ptm.LinearTerm(x, name="x_scale")
    yield model


@pytest.fixture(scope="module")
def pred_lin3(model_lin3, samples_lin3) -> Iterator[ptm_ls.PTMLocScalePredictions]:
    pred = ptm_ls.PTMLocScalePredictions(
        samples_lin3, model_lin3, x_loc=None, x_scale=None
    )
    yield pred


@pytest.mark.skip("I only test this manually at the moment.")
def test_run_mcmc_nuts_linear():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)
    model.loc_model += ptm.LinearTerm(x, name="x_loc")
    model.log_scale_model += ptm.LinearTerm(x, name="x_scale")

    graph = model.build_graph(maxiter=10)

    eb = gs.EngineBuilder(seed=192023, num_chains=4)
    eb.set_model(gs.LieselInterface(graph))
    eb.set_initial_values(graph.state)
    eb.set_duration(warmup_duration=1000, posterior_duration=1000)
    eb.positions_included = model.intercept_names
    eb.positions_included += model.normalization.mean_and_sd_names

    # eb.add_kernel(gs.NUTSKernel(["loc0"]))  # , "scale0"
    # eb.add_kernel(gs.NUTSKernel(["loc0", "scale0"]))  # , "scale0"
    eb.add_kernel(gs.NUTSKernel(model.loc_model.nuts_params))
    eb.add_kernel(gs.NUTSKernel(model.log_scale_model.nuts_params))
    eb.add_kernel(gs.NUTSKernel(model.normalization.nuts_params))
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]

    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

    z = pred.predict_z().mean(axis=(0, 1))
    _, pval = stats.normaltest(z)
    assert pval > 0.05

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc_coef = samples["x_loc_coef"].mean()
    assert loc_coef == pytest.approx(0.4, abs=0.2)

    loc_intercept = pred.loc_intercept.mean()
    assert loc_intercept == pytest.approx(1.2, abs=0.2)

    scale_intercept = np.log(pred.scale_intercept).mean()
    assert scale_intercept == pytest.approx(-0.3, abs=0.2)

    scale_coef = samples["x_scale_coef"].mean()
    assert scale_coef == pytest.approx(0.2, abs=0.2)


@pytest.mark.skip("I only test this manually at the moment.")
def test_run_mcmc_iwls_linear():
    """
    With only linear effects on loc and scale, IWLS works fine.
    But as soon as we add a nonlinear effect, it fails.
    """
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)
    model.loc_model += ptm.LinearTerm(x, name="x_loc")
    model.loc_model += ptm.NonlinearPSpline.from_nparam(
        x,
        nparam=10,
        scale=ptm.ExpParam(10.0, lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)),
        name="x_loc_nonlin",
    )
    model.log_scale_model += ptm.LinearTerm(x, name="x_scale")
    model.log_scale_model += ptm.NonlinearPSpline.from_nparam(
        x,
        nparam=10,
        scale=ptm.ExpParam(10.0, lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)),
        name="x_scale_nonlin",
    )

    graph = model.build_graph(maxiter=10)

    eb = gs.EngineBuilder(seed=192023, num_chains=4)
    eb.set_model(gs.LieselInterface(graph))
    eb.set_initial_values(graph.state)
    eb.set_duration(warmup_duration=1000, posterior_duration=1000)
    eb.positions_included = model.intercept_names
    eb.positions_included += model.normalization.mean_and_sd_names

    # eb.add_kernel(gs.NUTSKernel(["loc0"]))  # , "scale0"
    # eb.add_kernel(gs.NUTSKernel(["loc0", "scale0"]))  # , "scale0"
    eb.add_kernel(gs.IWLSKernel(model.loc_model.nuts_params))
    eb.add_kernel(gs.IWLSKernel(model.log_scale_model.nuts_params))
    eb.add_kernel(gs.NUTSKernel(model.normalization.nuts_params))
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]

    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

    z = pred.predict_z().mean(axis=(0, 1))
    _, pval = stats.normaltest(z)
    assert pval > 0.05

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc_coef = samples["x_loc_coef"].mean()
    assert loc_coef == pytest.approx(0.4, abs=0.2)

    loc_intercept = pred.loc_intercept.mean()
    assert loc_intercept == pytest.approx(1.2, abs=0.2)

    scale_intercept = np.log(pred.scale_intercept).mean()
    assert scale_intercept == pytest.approx(-0.3, abs=0.2)

    scale_coef = samples["x_scale_coef"].mean()
    assert scale_coef == pytest.approx(0.2, abs=0.2)


@pytest.mark.skip("I only test this manually at the moment.")
def test_run_mcmc_nuts_linear_autosetup():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)
    model.loc_model += ptm.LinearTerm(x, name="x_loc")
    model.log_scale_model += ptm.LinearTerm(x, name="x_scale")

    graph = model.build_graph(maxiter=100)

    eb = gs.EngineBuilder(seed=192023, num_chains=4)
    eb = model.setup_engine_builder(eb, graph)
    eb.set_duration(warmup_duration=1000, posterior_duration=1000)
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]
    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

    z = pred.predict_z().mean(axis=(0, 1))
    _, pval = stats.normaltest(z)
    assert pval > 0.05

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc_coef = samples["x_loc_coef"].mean()
    assert loc_coef == pytest.approx(0.4, abs=0.2)

    loc_intercept = pred.loc_intercept.mean()
    assert loc_intercept == pytest.approx(1.2, abs=0.2)

    scale_intercept = np.log(pred.scale_intercept).mean()
    assert scale_intercept == pytest.approx(-0.3, abs=0.2)

    scale_coef = samples["x_scale_coef"].mean()
    assert scale_coef == pytest.approx(0.2, abs=0.2)


@pytest.mark.skip("I only test this manually at the moment.")
def test_run_mcmc_nuts_nonlinear_autosetup():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)

    tau2_loc = ptm.VarHalfCauchy(10.0, scale=25.0, name="tau2_loc")
    model.loc_model += ptm.S.pspline(x, nparam=20, tau2=tau2_loc, name="x_loc")

    tau2_scale = ptm.VarHalfCauchy(10.0, scale=25.0, name="tau2_scale")
    model.log_scale_model += ptm.S.pspline(
        x, nparam=20, tau2=tau2_scale, name="x_scale"
    )

    graph = model.build_graph(maxiter=1000)

    eb = gs.EngineBuilder(seed=192023, num_chains=4)
    eb = model.setup_engine_builder(eb, graph)
    eb.set_duration(warmup_duration=1000, posterior_duration=1000)
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]
    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

    z = pred.predict_z().mean(axis=(0, 1))
    _, pval = stats.normaltest(z)
    assert pval > 0.05

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc_intercept = pred.loc_intercept.mean()
    assert loc_intercept == pytest.approx(1.2, abs=0.2)

    scale_intercept = np.log(pred.scale_intercept_exp).mean()
    assert scale_intercept == pytest.approx(-0.3, abs=0.2)


@pytest.mark.skip("I only test this manually at the moment.")
def test_run_mcmc_IWLS_normal():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)

    tau2_loc = ptm.VarHalfCauchy(10.0, scale=25.0, name="tau2_loc")
    model.loc_model += ptm.S.pspline(x, nparam=20, tau2=tau2_loc, name="x_loc")

    tau2_scale = ptm.VarHalfCauchy(10.0, scale=25.0, name="tau2_scale")
    model.log_scale_model += ptm.S.pspline(
        x, nparam=20, tau2=tau2_scale, name="x_scale"
    )

    graph = model.build_graph(maxiter=1000, optimize_normalization=False)

    model.loc_model.x_loc.mcmc_kernels[0] = gs.IWLSKernel(["x_loc_coef"])
    model.log_scale_model.x_scale.mcmc_kernels[0] = gs.IWLSKernel(["x_scale_coef"])

    eb = gs.EngineBuilder(seed=192023, num_chains=4)
    eb = model.setup_engine_builder(eb, graph, sample_normalization=False)

    eb.set_duration(warmup_duration=1000, posterior_duration=1000)
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]
    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

    z = pred.predict_z().mean(axis=(0, 1))
    _, pval = stats.normaltest(z)
    assert pval > 0.05

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc_intercept = pred.loc_intercept.mean()
    assert loc_intercept == pytest.approx(1.2, abs=0.2)

    scale_intercept = np.log(pred.scale_intercept_exp).mean()
    assert scale_intercept == pytest.approx(-0.3, abs=0.2)


@pytest.mark.skip("I only test this manually at the moment.")
def test_run_mcmc_IWLS_normal_both():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)

    tau2_loc = ptm.VarHalfCauchy(10.0, scale=25.0, name="tau2_loc")
    model.loc_model += ptm.S.pspline(x, nparam=20, tau2=tau2_loc, name="x_loc")

    tau2_scale = ptm.VarHalfCauchy(10.0, scale=25.0, name="tau2_scale")
    model.log_scale_model += ptm.S.pspline(
        x, nparam=20, tau2=tau2_scale, name="x_scale"
    )

    graph = model.build_graph(maxiter=1000, optimize_normalization=False)

    model.loc_model.x_loc.mcmc_kernels[0] = gs.IWLSKernel(["x_loc_coef"])
    model.loc_model.x_loc.mcmc_kernels[1] = gs.IWLSKernel(["tau2_loc_root_transformed"])
    model.log_scale_model.x_scale.mcmc_kernels[0] = gs.IWLSKernel(["x_scale_coef"])
    model.log_scale_model.x_scale.mcmc_kernels[1] = gs.IWLSKernel(
        ["tau2_scale_root_transformed"]
    )

    eb = gs.EngineBuilder(seed=192023, num_chains=4)
    eb = model.setup_engine_builder(eb, graph, sample_normalization=False)

    eb.set_duration(warmup_duration=1000, posterior_duration=1000)
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]
    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

    z = pred.predict_z().mean(axis=(0, 1))
    _, pval = stats.normaltest(z)
    assert pval > 0.05

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc_intercept = pred.loc_intercept.mean()
    assert loc_intercept == pytest.approx(1.2, abs=0.2)

    scale_intercept = np.log(pred.scale_intercept_exp).mean()
    assert scale_intercept == pytest.approx(-0.3, abs=0.2)


@pytest.mark.skip("I only test this manually at the moment.")
def test_run_mcmc_IWLS_normal_igvar():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)

    tau2_loc = ptm.VarInverseGamma(
        10.0, concentration=0.01, scale=0.01, name="tau2_loc"
    )
    model.loc_model += ptm.S.pspline(x, nparam=20, tau2=tau2_loc, name="x_loc")

    tau2_scale = ptm.VarInverseGamma(
        10.0, concentration=0.01, scale=0.01, name="tau2_scale"
    )
    model.log_scale_model += ptm.S.pspline(
        x, nparam=20, tau2=tau2_scale, name="x_scale"
    )

    graph = model.build_graph(maxiter=1000, optimize_normalization=False)

    model.loc_model.x_loc.mcmc_kernels[0] = gs.IWLSKernel(["x_loc_coef"])
    model.log_scale_model.x_scale.mcmc_kernels[0] = gs.IWLSKernel(["x_scale_coef"])

    eb = gs.EngineBuilder(seed=192023, num_chains=4)
    eb = model.setup_engine_builder(eb, graph, sample_normalization=False)

    eb.set_duration(warmup_duration=1000, posterior_duration=1000)
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]
    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

    z = pred.predict_z().mean(axis=(0, 1))
    _, pval = stats.normaltest(z)
    assert pval > 0.05

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc_intercept = pred.loc_intercept.mean()
    assert loc_intercept == pytest.approx(1.2, abs=0.2)

    scale_intercept = np.log(pred.scale_intercept_exp).mean()
    assert scale_intercept == pytest.approx(-0.3, abs=0.2)


@pytest.mark.skip("I only test this manually at the moment.")
def test_run_mcmc_nuts_exp_hc():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.exponential(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)

    tau2_loc = ptm.VarHalfCauchy(10.0, scale=25.0, name="tau2_loc")
    model.loc_model += ptm.S.pspline(x, nparam=20, tau2=tau2_loc, name="x_loc")

    tau2_scale = ptm.VarHalfCauchy(10.0, scale=25.0, name="tau2_scale")
    model.log_scale_model += ptm.S.pspline(
        x, nparam=20, tau2=tau2_scale, name="x_scale"
    )

    graph = model.build_graph(maxiter=1000)

    eb = gs.EngineBuilder(seed=192023, num_chains=4)
    eb = model.setup_engine_builder(eb, graph)

    eb.set_duration(warmup_duration=1000, posterior_duration=1000)
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]
    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

    z = pred.predict_z().mean(axis=(0, 1))
    _, pval = stats.normaltest(z)
    assert pval > 0.05

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc_intercept = pred.loc_intercept.mean()
    assert loc_intercept == pytest.approx(1.2, abs=0.2)

    scale_intercept = np.log(pred.scale_intercept_exp).mean()
    assert scale_intercept == pytest.approx(-0.3, abs=0.2)


@pytest.mark.skip("I only test this manually at the moment.")
def test_run_mcmc_IWLS_exp_igvar():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.exponential(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)

    tau2_loc = ptm.VarInverseGamma(
        10.0, concentration=0.01, scale=0.01, name="tau2_loc"
    )
    model.loc_model += ptm.S.pspline(x, nparam=20, tau2=tau2_loc, name="x_loc")

    tau2_scale = ptm.VarInverseGamma(
        10.0, concentration=0.01, scale=0.01, name="tau2_scale"
    )
    model.log_scale_model += ptm.S.pspline(
        x, nparam=20, tau2=tau2_scale, name="x_scale"
    )

    graph = model.build_graph(maxiter=1000)

    model.loc_model.x_loc.mcmc_kernels[0] = gs.IWLSKernel(["x_loc_coef"])
    model.log_scale_model.x_scale.mcmc_kernels[0] = gs.IWLSKernel(["x_scale_coef"])

    norm_params = model.normalization.parameters
    norm_hyper = model.normalization.hyper_parameters
    model.normalization.mcmc_kernels[0] = gs.NUTSKernel(norm_params)
    model.normalization.mcmc_kernels.append(gs.IWLSKernel(norm_hyper))

    eb = gs.EngineBuilder(seed=192023, num_chains=4)
    eb = model.setup_engine_builder(eb, graph)

    eb.set_duration(warmup_duration=1000, posterior_duration=1000)
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]
    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

    z = pred.predict_z().mean(axis=(0, 1))
    _, pval = stats.normaltest(z)
    assert pval > 0.05

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc_intercept = pred.loc_intercept.mean()
    assert loc_intercept == pytest.approx(1.2, abs=0.2)

    scale_intercept = np.log(pred.scale_intercept_exp).mean()
    assert scale_intercept == pytest.approx(-0.3, abs=0.2)


@pytest.mark.skip("I only test this manually at the moment.")
def test_run_mcmc_nuts_nonlinear_combined():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)

    tau2_loc = ptm.VarHalfCauchy(10.0, scale=25.0, name="tau2_loc")
    model.loc_model += ptm.S.pspline(x, nparam=20, tau2=tau2_loc, name="x_loc")

    params = model.loc_model.x_loc.parameters
    hyper = model.loc_model.x_loc.hyper_parameters
    model.loc_model.x_loc.mcmc_kernels = [gs.NUTSKernel(params + hyper)]

    tau2_scale = ptm.VarHalfCauchy(10.0, scale=25.0, name="tau2_scale")
    model.log_scale_model += ptm.S.pspline(
        x, nparam=20, tau2=tau2_scale, name="x_scale"
    )

    params = model.log_scale_model.x_scale.parameters
    hyper = model.log_scale_model.x_scale.hyper_parameters
    model.log_scale_model.x_scale.mcmc_kernels = [gs.NUTSKernel(params + hyper)]

    graph = model.build_graph(maxiter=1000)

    eb = gs.EngineBuilder(seed=192023, num_chains=4)
    eb = model.setup_engine_builder(eb, graph)
    eb.set_duration(warmup_duration=1000, posterior_duration=1000)
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]
    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

    z = pred.predict_z().mean(axis=(0, 1))
    _, pval = stats.normaltest(z)
    assert pval > 0.05

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc_intercept = pred.loc_intercept.mean()
    assert loc_intercept == pytest.approx(1.2, abs=0.2)

    scale_intercept = np.log(pred.scale_intercept).mean()
    assert scale_intercept == pytest.approx(-0.3, abs=0.2)


@pytest.mark.skip("I only test this manually at the moment.")
def test_run_mcmc_nuts_nonlinear_igvar():
    np.random.seed(2812)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)

    tau2_loc = ptm.VarInverseGamma(
        10.0, concentration=0.01, scale=0.01, name="tau2_loc"
    )
    model.loc_model += ptm.S.pspline(x, nparam=20, tau2=tau2_loc, name="x_loc")

    tau2_scale = ptm.VarInverseGamma(
        10.0, concentration=0.01, scale=0.01, name="tau2_scale"
    )
    model.log_scale_model += ptm.S.pspline(
        x, nparam=20, tau2=tau2_scale, name="x_scale"
    )

    graph = model.build_graph(
        [["x_loc_coef", "x_scale_coef"]], maxiter=1000, optimize_normalization=False
    )

    eb = gs.EngineBuilder(seed=123, num_chains=4)
    eb = model.setup_engine_builder(eb, graph, sample_normalization=False)
    eb.set_duration(warmup_duration=1000, posterior_duration=1000)
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]
    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

    z = pred.predict_z().mean(axis=(0, 1))
    _, pval = stats.normaltest(z)
    assert pval > 0.05

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc_intercept = pred.loc_intercept.mean()
    assert loc_intercept == pytest.approx(1.2, abs=0.2)

    scale_intercept = np.log(pred.scale_intercept).mean()
    assert scale_intercept == pytest.approx(-0.3, abs=0.2)


def test_log_prob():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)
    # model.loc += ptm.Intercept("loc0")
    model.loc_model += ptm.LinearTerm(x, name="x_loc")
    # model.log_scale += ptm.Intercept("scale0")
    model.log_scale_model += ptm.LinearTerm(x, name="x_scale")

    gb = lsl.GraphBuilder().add(model.response)
    lmod = gb.build_model()
    goose_model = gs.LieselInterface(lmod)

    position = goose_model.extract_position(
        ["normalization_tau2_root_transformed"], lmod.state
    )

    def log_prob_fn(position):
        new_model_state = goose_model.update_state(position, lmod.state)
        return goose_model.log_prob(new_model_state)

    log_prob_fn(position)

    key = jax.random.PRNGKey(4)
    shape_sample = jax.random.normal(key, shape=(28,))

    pos = {"normalization_shape_transformed": shape_sample}
    log_prob_fn(pos)

    lmod.state["response_log_prob"]
    lmod.state["scaled_normalization"]
    model.normalization.shape_param.shape_reparam.value = shape_sample
    lmod.update()
    model.residuals.value
    model.normalization.shape_param.value.mean()
    model.normalization.unscaled_normalization_and_deriv.value[0].mean()
    model.normalization.unscaled_normalization_and_deriv.value[0].std()
    model.normalization.coef.value
    model.normalization.unscaled_normalization_and_deriv.value
    lmod.state["scaled_normalization"]
    lmod.state["normalization_shape_transformed_value"]


class TestLocScaleTM:
    def test_runs(self) -> None:
        y = np.random.normal(size=13)
        model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)
        assert model is not None

    def test_build_model(self) -> None:
        y = np.random.normal(size=13)
        model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)
        gb = lsl.GraphBuilder().add(model.response)
        lmod = gb.build_model()
        assert lmod is not None

    def test_optimize_knots(self) -> None:
        df = pd.read_csv("tests/resources/nonnormal_linear/data.csv")
        x = df.x.to_numpy()
        y = df.y.to_numpy()
        model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)
        model.loc_model += ptm.LinearTerm(x, name="x_loc")
        model.log_scale_model += ptm.LinearTerm(x, name="x_scale")

        new_model, _ = model.optimize_knots()

        # assert that knots have changed
        assert not np.allclose(model.knots, new_model.knots)
        assert model.knots.shape == new_model.knots.shape

        # assert that the new model has the old terms
        loc_terms = list(model.loc_model.terms.values())
        scale_terms = list(model.log_scale_model.terms.values())

        assert len(loc_terms) == len(new_model.loc_model.terms)
        assert len(scale_terms) == len(new_model.log_scale_model.terms)

        for term in new_model.loc_model.terms.values():
            assert term in loc_terms

        for term in new_model.log_scale_model.terms.values():
            assert term in scale_terms

        # assert that the new model can be built into a graph
        gb = lsl.GraphBuilder().add(new_model.response)
        graph = gb.build_model()
        assert graph is not None

    def test_init_graph_and_interface(self, caplog) -> None:
        logger = logging.getLogger("liesel_ptm")
        logger.propagate = True
        caplog.set_level(logging.INFO)
        df = pd.read_csv("tests/resources/normal_nonlinear/data.csv")
        x = df.x.to_numpy()
        y = df.y.to_numpy()
        model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)
        model.loc_model += ptm.LinearTerm(x, name="x_loc")
        model.log_scale_model += ptm.LinearTerm(x, name="x_scale")

        graph = model.build_graph(maxiter=3)

        assert graph is not None

    def test_optimize_knots_and_graph(self, caplog) -> None:
        caplog.set_level(logging.INFO)
        df = pd.read_csv("tests/resources/normal_nonlinear/data.csv")
        x = df.x.to_numpy()
        y = df.y.to_numpy()
        model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)
        model.loc_model += ptm.LinearTerm(x, name="x_loc")
        model.log_scale_model += ptm.LinearTerm(x, name="x_scale")
        model, position = model.optimize_knots()

        graph = model.build_graph(position=position, maxiter=3)

        assert graph is not None


class TestLocScaleTMPredictions:
    def test_quantile_shape(self, model, samples) -> None:
        pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=0.0, x_scale=0.0)
        y = pred.predict_quantile(np.array([0.4, 0.6]))
        chains, iterations = samples["loc_intercept"].shape
        assert y.shape == (chains, iterations, 1, 2)

        y = pred.predict_quantile(np.linspace(0.1, 0.9, 5))
        assert y.shape == (chains, iterations, 1, 5)

    def test_predict_transformation_shape(self, model, samples) -> None:
        pred = ptm_ls.PTMLocScalePredictions(
            samples, model, y=0.0, x_loc=0.0, x_scale=0.0
        )
        z = pred.predict_transformation()
        chains, iterations = samples["loc_intercept"].shape
        assert z.shape == (chains, iterations, 1)

    def test_inverse_transformation(self, pred) -> None:
        z = pred.predict_transformation()
        yhat = pred.predict_transformation_inverse(z)

        n = np.array(range(pred.y.shape[0]))
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                np.allclose(yhat[i, j, n, n], pred.y, atol=0.1)

    def test_predict_residuals(self, pred, samples) -> None:
        zt = pred.predict_residuals()
        assert zt.mean() == pytest.approx(0.0, abs=0.1)
        assert zt.std() == pytest.approx(1.0, abs=0.1)

        zt_samples = samples["residuals"]
        assert np.allclose(zt_samples, zt, atol=0.1)

    def test_predict_z(self, pred, samples) -> None:
        z = pred.predict_z()

        z_samples = samples["scaled_normalization"]

        assert z_samples.mean() == pytest.approx(0.0, abs=0.1)
        assert z_samples.std() == pytest.approx(1.0, abs=0.1)

        assert np.allclose(z_samples, z, atol=0.1)

    def test_predict_z_deriv(self, pred, samples) -> None:
        z_deriv = pred.predict_z_deriv()

        z_deriv_samples = samples["z_deriv"]

        assert np.allclose(z_deriv, z_deriv_samples, atol=0.1)

    def test_predict_coef(self, pred, samples) -> None:
        shape = pred.model.normalization.shape_param.shape_calc.predict(samples)
        dknots = np.diff(pred.model.knots).mean()
        coef = nd.normalization_coef(shape, dknots)
        coef_samples = samples["normalization_coef"]
        assert np.allclose(coef_samples, coef, atol=0.1)

    def test_sample(self, samples, model) -> None:
        pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=0.0, x_scale=0.0)
        key = jax.random.PRNGKey(4)
        y = pred.sample(key, shape=(30,))

        chains, iterations = pred.samples["loc_intercept"].shape

        assert y.shape == (chains, iterations, 1, 30)

    def test_summarise_density(self, pred, samples, model) -> None:
        summary = pred.summarise_density_by_quantiles()
        assert summary.shape == (100, 18)
        assert not summary.isnull().values.any()

        pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=0.0, x_scale=0.0)
        summary = pred.summarise_density_by_quantiles()
        assert summary.shape == (100, 18)
        assert not summary.isnull().values.any()

    def test_summarise_density_sample(self, pred, samples, model) -> None:
        key = jax.random.PRNGKey(4)
        summary = pred.summarise_density_by_samples(key, n=10)
        assert summary.shape == (1000, 10)
        assert not summary.isnull().values.any()

        pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=0.0, x_scale=0.0)

        summary = pred.summarise_density_by_samples(key, n=10)
        assert summary.shape == (1000, 10)
        assert not summary.isnull().values.any()

    def test_summarise_density_lin3(self, samples_lin3, model_lin3) -> None:
        x = np.zeros((1, 3))
        pred = ptm_ls.PTMLocScalePredictions(
            samples_lin3, model_lin3, x_loc=x, x_scale=x
        )
        summary = pred.summarise_density_by_quantiles()
        assert summary.shape == (300, 22)
        assert not summary.isnull().values.any()


class TestNormalizationFn:
    def test_normalize(self) -> None:
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=10)
        dknots = np.diff(knots).mean()
        key = jax.random.PRNGKey(4)
        key, subkey = jax.random.split(key)
        shape = sample_shape(key, 9)
        coef = nd.normalization_coef(shape.sample, dknots)

        y = jax.random.normal(subkey, shape=(20,))

        normalization = ptm_ls.NormalizationFn(knots)

        normalized = normalization(y, coef, np.zeros(1), np.ones(1))

        assert not np.allclose(normalized, y)

    def test_normalize_identity(self) -> None:
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=10)
        dknots = np.diff(knots).mean()
        key = jax.random.PRNGKey(4)
        key, subkey = jax.random.split(key)
        shape = np.zeros(9)
        coef = nd.normalization_coef(shape, dknots)

        y = jax.random.normal(subkey, shape=(20,))
        y = y - y.mean()

        normalization = ptm_ls.NormalizationFn(knots)

        normalized = normalization(y, coef, np.zeros(1), np.ones(1))
        normalized = normalized - normalized.mean()

        assert np.allclose(normalized, y, atol=1e-4)

    def test_normalize_batched(self) -> None:
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=10)
        dknots = np.diff(knots).mean()
        key = jax.random.PRNGKey(4)
        key, subkey = jax.random.split(key)
        shape = np.zeros((4, 6, 9))
        coef = nd.normalization_coef(shape, dknots)

        y = jax.random.normal(subkey, shape=(20,))
        y = y - y.mean()

        normalization = ptm_ls.NormalizationFn(knots)

        normalized = normalization(y, coef, np.zeros(1), np.ones(1))
        assert normalized.shape == (4, 6, 20)

        normalized = normalized - normalized.mean(axis=-1, keepdims=True)
        assert normalized.shape == (4, 6, 20)

        for i in range(4):
            for j in range(6):
                assert np.allclose(normalized[i, j, :], y, atol=1e-4)

    def test_inverse_identity(self) -> None:
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=10)
        dknots = np.diff(knots).mean()
        key = jax.random.PRNGKey(4)
        key, subkey = jax.random.split(key)
        shape = np.zeros(9)
        coef = nd.normalization_coef(shape, dknots)

        z = jax.random.normal(subkey, shape=(20,))
        z = z - z.mean()

        normalization = ptm_ls.NormalizationFn(knots)

        zt = normalization.inverse(z, coef, np.zeros(1), np.ones(1))
        zt = zt - zt.mean()

        assert np.allclose(z, zt, atol=1e-4)

    def test_inverse_identity_newton(self) -> None:
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=10)
        dknots = np.diff(knots).mean()
        key = jax.random.PRNGKey(4)
        key, subkey = jax.random.split(key)
        shape = np.zeros(9)
        coef = nd.normalization_coef(shape, dknots)

        z = jax.random.normal(subkey, shape=(20,))
        z = z - z.mean()

        normalization = ptm_ls.NormalizationFn(knots)

        zt = normalization.inverse_newton(z, coef, np.zeros(1), np.ones(1), max_iter=30)
        zt = zt - zt.mean()

        assert np.allclose(z, zt, atol=1e-4)

    def test_inverse(self) -> None:
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=10)
        dknots = np.diff(knots).mean()
        key = jax.random.PRNGKey(4)
        key, subkey = jax.random.split(key)
        shape = sample_shape(key, 9)
        coef = nd.normalization_coef(shape.sample, dknots)

        y = np.linspace(-2.0, 2.0, 20)

        normalization = ptm_ls.NormalizationFn(knots)

        z = normalization(y, coef, np.zeros(1), np.ones(1))

        zt = normalization.inverse(z, coef, np.zeros(1), np.ones(1))

        # ggplot() + aes(y, z) + geom_point()
        # ggplot() + geom_point(aes(zt, z)) + geom_point(aes(y, z), color="red")

        assert not np.allclose(z, zt, atol=1e-4)
        assert np.allclose(y, zt, atol=1e-4)

    def test_inverse_newton(self) -> None:
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=10)
        dknots = np.diff(knots).mean()
        key = jax.random.PRNGKey(4)
        key, subkey = jax.random.split(key)
        shape = sample_shape(key, 9)
        coef = nd.normalization_coef(shape.sample, dknots)

        y = np.linspace(-2.0, 2.0, 20)

        normalization = ptm_ls.NormalizationFn(knots)

        z = normalization(y, coef, np.zeros(1), np.ones(1))

        zt = normalization.inverse_newton(z, coef, np.zeros(1), np.ones(1), max_iter=30)

        # ggplot() + aes(y, z) + geom_point()
        # ggplot() + geom_point(aes(zt, z)) + geom_point(aes(y, z), color="red")

        assert not np.allclose(z, zt, atol=1e-4)
        assert np.allclose(y, zt, atol=1e-4)

    def test_inverse_batched(self) -> None:
        knots = kn(np.array([-2.0, 2.0]), order=3, n_params=10)
        dknots = np.diff(knots).mean()
        key = jax.random.PRNGKey(4)
        key, subkey = jax.random.split(key)
        shape = np.zeros((4, 6, 9))
        coef = nd.normalization_coef(shape, dknots)

        z = jax.random.normal(subkey, shape=(20,))
        z = z - z.mean()

        normalization = ptm_ls.NormalizationFn(knots)

        zt = normalization.inverse(z, coef, np.zeros(1), np.ones(1))
        zt = zt - zt.mean(axis=-1, keepdims=True)

        assert zt.shape == (4, 6, 20)

        assert np.allclose(z, zt, atol=1e-4)


def test_state_to_samples() -> None:
    df = pd.read_csv("tests/resources/normal_nonlinear/data.csv")
    x = df.x.to_numpy()
    y = df.y.to_numpy()
    model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)
    model.loc_model += ptm.LinearTerm(x, name="x_loc")
    model.log_scale_model += ptm.LinearTerm(x, name="x_scale")
    model, position = model.optimize_knots()

    graph = model.build_graph(position=position, maxiter=3)

    position_keys = model.all_parameter_names()
    samples = ptm.state_to_samples(position_keys, graph)

    pred = ptm.PTMLocScalePredictions(samples, model)

    summary_df = pred.summarise_density_by_quantiles()

    assert not summary_df.isna().sum().sum()


class TestCache:
    def test_cache_model(self, tmp_path):
        cache = ptm.cache(tmp_path)

        @cache
        def setup_model():
            df = pd.read_csv("tests/resources/nonnormal_linear/data.csv")

            x = df.x.to_numpy()
            y = df.y.to_numpy()
            time.sleep(1)

            path = "tests/resources/nonnormal_linear/knots.pickle"
            model = ptm_ls.PTMLocScale._from_knots(path, y)
            model.loc_model += ptm.LinearTerm(x, name="x_loc")
            model.log_scale_model += ptm.LinearTerm(x, name="x_scale")
            return model

        time1 = timeit.timeit(setup_model, number=1)
        time2 = timeit.timeit(setup_model, number=1)

        assert time1 > time2
        assert time1 > 1.0
        assert time2 < 0.01

    def test_cache_predictions(self, tmp_path, model, samples):
        cache = ptm.cache(tmp_path)

        @cache
        def setup_predictions():
            time.sleep(1)
            pred = ptm_ls.PTMLocScalePredictions(
                samples, model, x_loc=None, x_scale=None
            )
            return pred

        time1 = timeit.timeit(setup_predictions, number=1)
        time2 = timeit.timeit(setup_predictions, number=1)

        assert time1 > time2
        assert time1 > 1.0
        assert time2 < 0.01
