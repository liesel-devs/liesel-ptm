from collections.abc import Iterator

import jax
import liesel
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm import Var, kwargs_full
from liesel_ptm import nodes as nd
from liesel_ptm import ptm_ls

nparam = 7
x = np.random.uniform(size=20)

X = lsl.obs(np.random.uniform(-2, 2, size=(10, 3)), name="x1")
coef = lsl.param(np.random.uniform(0, 1, size=(3,)), name="b")


@pytest.fixture
def basis() -> Iterator[nd.BSplineBasis]:
    yield nd.BSplineBasis.auto(x, nparam=nparam)


class TestDot:
    def test_dot(self) -> None:
        smooth = nd.Dot(X, coef).update()

        assert smooth.value.shape == (10,)
        assert np.allclose(X.value @ coef.value, smooth.value)

    @pytest.mark.skipif(
        liesel.__version_info__ < (0, 2, 5),
        reason="Test depends on a new feature of liesel 0.2.5",
    )
    def test_dot_error(self) -> None:
        coef = lsl.param(np.random.uniform(0, 1, size=(4,)), name="b")
        with pytest.raises(RuntimeError):
            nd.Dot(X, coef).update()

    @pytest.mark.skipif(
        liesel.__version_info__ < (0, 2, 5),
        reason="Test depends on a new feature of liesel 0.2.5",
    )
    def test_dot_error_lambda(self) -> None:
        coef = lsl.param(np.random.uniform(0, 1, size=(4,)), name="b")
        with pytest.raises(RuntimeError):
            lsl.Calc(lambda x, b: x @ b, X, coef).update()

    def test_dot_predict_multiple_chains(self) -> None:
        smooth = nd.Dot(X, coef).update()
        # 4 chains, 7 samples, 3 coefs
        samples = {coef.name: np.random.uniform(0, 1, size=(4, 7, 3))}

        prediction = smooth.predict(samples, x=np.array([[1.0, 1.0, 1.0]]))

        assert prediction.shape == (4, 7, 1)

    def test_dot_predict_one_chain(self) -> None:
        smooth = nd.Dot(X, coef).update()
        # 1 chain, 7 samples, 3 coefs
        samples = {coef.name: np.random.uniform(0, 1, size=(1, 7, 3))}

        prediction = smooth.predict(samples, x=np.array([[1.0, 1.0, 1.0]]))

        assert prediction.shape == (1, 7, 1)

    def test_dot_predict_multiple_x(self) -> None:
        smooth = nd.Dot(X, coef).update()
        # 4 chains, 7 samples, 3 coefs
        samples = {coef.name: np.random.uniform(0, 1, size=(4, 7, 3))}

        prediction = smooth.predict(
            samples, x=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        )

        assert prediction.shape == (4, 7, 2)

    def test_basis_dot(self) -> None:
        x = np.linspace(-2, 2, 10)
        basis = nd.BSplineBasis.auto(x, nparam=8)
        coef = lsl.param(np.random.uniform(0, 1, size=(basis.nparam,)))

        smooth = nd.BasisDot(basis, coef).update()

        assert smooth.value.shape == (10,)
        assert np.allclose(basis.value @ coef.value, smooth.value)

    def test_basis_dot_predict(self) -> None:
        x = np.linspace(-2, 2, 10)
        basis = nd.Var(nd.BSplineBasis.auto(x, nparam=8))
        coef = lsl.param(np.random.uniform(0, 1, size=(basis.nparam,)), name="b")

        # 4 chains, 7 samples, nparam coefs
        samples = {coef.name: np.random.uniform(0, 1, size=(4, 7, basis.nparam))}

        smooth = nd.BasisDot(basis, coef).update()

        prediction = smooth.predict(samples)
        assert prediction.shape == (4, 7, 10)

        prediction = smooth.predict(samples, x=1.0)
        assert prediction.shape == (4, 7, 1)

        prediction = smooth.predict(samples, x=[1.0, 1.3])
        assert prediction.shape == (4, 7, 2)

        prediction = smooth.predict(samples, x=np.array([1.0, 1.3]))
        assert prediction.shape == (4, 7, 2)


class TestCalcVar:
    def test_predict(self) -> None:
        smooth = nd.Var(nd.Dot(X, coef).update())

        # 4 chains, 7 samples, 3 coefs
        samples = {coef.name: np.random.uniform(0, 1, size=(4, 7, 3))}

        prediction = smooth.predict(samples, x=np.array([[1.0, 1.0, 1.0]]))

        assert prediction.shape == (4, 7, 1)


class TestScaledBasisDot:
    def test_prediction_shape(self) -> None:
        x = np.linspace(-2, 2, 10)
        basis = nd.Var(nd.BSplineBasis.auto(x, nparam=8))
        coef = lsl.param(np.random.uniform(0, 1, size=(basis.nparam,)), name="b")

        prior = lsl.Dist(tfd.Exponential, rate=lsl.Var(1.0, name="rate"))
        scale = nd.ExpParam(1.0, prior, name="var").update()

        smooth = nd.ScaledBasisDot(basis, coef, scale)

        samples = dict()
        samples[coef.name] = np.random.uniform(0, 1, size=(4, 7, basis.nparam))
        samples[scale.log_var.name] = np.random.uniform(0, 1, size=(4, 7, 1))

        prediction = smooth.predict(samples, x=np.array([1.0, 1.0, 1.0]))
        assert prediction.shape == (4, 7, 3)

        prediction = smooth.predict(samples, x=1.0)
        assert prediction.shape == (4, 7, 1)

    def test_prediction_value(self) -> None:
        x = np.linspace(-2, 2, 10)
        basis = nd.Var(nd.BSplineBasis.auto(x, nparam=8))
        coef = lsl.param(np.random.uniform(0, 1, size=(basis.nparam,)), name="b")

        prior = lsl.Dist(tfd.Exponential, rate=lsl.Var(1.0, name="rate"))
        scale = nd.ExpParam(1.0, prior, name="var").update()

        smooth = nd.ScaledBasisDot(basis, coef, scale)

        samples = dict()
        samples[coef.name] = np.random.uniform(0, 1, size=(4, 7, basis.nparam))
        samples[scale.log_var.name] = np.random.uniform(-1, 1, size=(4, 7, 1))

        scaled_coef = np.exp(samples[scale.log_var.name]) * samples[coef.name]
        manual_prediction = np.tensordot(
            basis.evaluate(1.0), scaled_coef, axes=([1], [-1])
        )
        manual_prediction = np.moveaxis(manual_prediction, 0, -1)

        prediction = smooth.predict(samples, x=1.0)

        assert np.allclose(prediction, manual_prediction)


class TestBSplineBasis:
    def test_init(self, basis: nd.BSplineBasis) -> None:
        assert np.all(basis.observed_value == x)

    def test_nparam(self, basis: nd.BSplineBasis) -> None:
        assert basis.value.shape[-1] == nparam

    def test_rowsum(self, basis: nd.BSplineBasis) -> None:
        assert np.allclose(basis.value.sum(axis=1), 1.0)

    def test_evaluate(self, basis: nd.BSplineBasis) -> None:
        assert basis.evaluate(0.5).shape == (1, 7)
        assert np.allclose(basis.evaluate(x[10]), basis.value[10, :])
        assert np.allclose(basis.evaluate(x), basis.value)

    def test_deriv(self, basis: nd.BSplineBasis) -> None:
        assert basis.deriv(x).shape == basis.value.shape

    def test_centered(self) -> None:
        basis = nd.BSplineBasis.auto(x, nparam=nparam, centered=True)
        assert np.allclose(basis.value.mean(axis=0), 0.0, atol=1e-6)

    def test_evaluate_centered(self, basis: nd.BSplineBasis) -> None:
        basis_centered = nd.BSplineBasis.auto(x, nparam=nparam, centered=True)
        basis_centered_eval = basis_centered.evaluate(0.5)
        basis_eval = basis.evaluate(0.5)

        assert not np.allclose(basis_eval, basis_centered_eval)
        assert np.allclose(basis_centered_eval, basis_eval - basis.colmeans)

    def test_reparam(self) -> None:
        basis = nd.BSplineBasis.auto(x, nparam=nparam)
        pen = nd.diffpen(basis.value.shape[-1], diff=2)
        Z = nd.nullspace_remover(pen)

        basis = basis.reparam(Z)
        assert basis.value.shape == (basis.value.shape[0], pen.shape[1] - 2)

        basis_eval = basis.evaluate(0.5)
        assert basis_eval.shape == (1, pen.shape[1] - 2)

    def test_reparam_twice(self) -> None:
        basis = nd.BSplineBasis.auto(x, nparam=nparam)
        pen = nd.diffpen(basis.value.shape[-1], diff=2)
        Z = nd.nullspace_remover(pen)
        Ltinv = nd.cholesky_ltinv(Z.T @ pen @ Z)

        basis = basis.reparam(Z)
        assert basis.value.shape == (basis.value.shape[0], pen.shape[1] - 2)

        basis = basis.reparam(Ltinv)
        assert np.allclose(basis.reparam_matrix, Z @ Ltinv)
        assert basis.value.shape == (basis.value.shape[0], pen.shape[1] - 2)


def test_nullspace_remover(basis: nd.BSplineBasis) -> None:
    pen = nd.diffpen(basis.value.shape[-1], diff=2)
    Z = nd.nullspace_remover(pen)
    assert Z.shape == (pen.shape[0], pen.shape[1] - 2)


class TestExpParam:
    def test_log(self) -> None:
        prior = lsl.Dist(tfd.Exponential, rate=lsl.Var(1.0, name="rate"))
        var = nd.ExpParam(1.0, prior, name="var").update()
        assert var.log_var.value == pytest.approx(np.log(var.value))

    def test_exp_param(self):
        prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
        tau = nd.ExpParam(10.0, prior, name="tau")
        log_tau = tau.log_var

        assert tau.weak
        assert not tau.log_var.weak

        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau.value == pytest.approx(np.log(10.0))

        prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
        tau = nd.Var(10.0, prior, name="tau")
        log_tau_gb = lsl.GraphBuilder().transform(tau, tfb.Exp)

        assert tau.weak
        assert not log_tau.weak

        tau.update()
        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau_gb.value == pytest.approx(np.log(10.0))

        log_tau.dist_node.update()
        log_tau_gb.dist_node.update()
        assert log_tau.log_prob == pytest.approx(log_tau_gb.log_prob)


@pytest.fixture(scope="module")
def samples_nonlin():
    path = "tests/resources/normal_nonlinear/results.pickle"
    results = gs.engine.SamplingResults.pkl_load(path)
    yield results.get_posterior_samples()


@pytest.fixture(scope="module")
def model_nonlin() -> Iterator[ptm_ls.PTMLocScale]:
    df = pd.read_csv("tests/resources/normal_nonlinear/data.csv")

    x = df.x.to_numpy()
    y = df.y.to_numpy()

    path = "tests/resources/normal_nonlinear/knots.pickle"

    model = ptm_ls.PTMLocScale._from_knots(path, y)
    model.loc_model += ptm.LinearTerm(x, name="x_loc_lin")
    model.log_scale_model += ptm.LinearTerm(x, name="x_scale_lin")

    scale = ptm.ScaleHalfCauchy(10.0, scale=25.0, name="loc_tau")

    model.loc_model += ptm.NonlinearPSpline.from_nparam(
        x, nparam=20, scale=scale, name="x_loc_nonlin"
    )

    scale = ptm.ScaleHalfCauchy(10.0, scale=25.0, name="scale_tau")
    model.log_scale_model += ptm.NonlinearPSpline.from_nparam(
        x, nparam=20, scale=scale, name="x_scale_nonlin"
    )

    yield model


@pytest.fixture(scope="module")
def pred_nonlin(
    model_nonlin, samples_nonlin
) -> Iterator[ptm_ls.PTMLocScalePredictions]:
    pred = ptm_ls.PTMLocScalePredictions(
        samples_nonlin, model_nonlin, **kwargs_full("x")
    )
    yield pred


class TestNonlinearPSpline:
    def test_init(self) -> None:
        prior = lsl.Dist(tfd.Exponential, rate=1.0)
        scale = nd.ExpParam(1.0, prior).update()
        x = np.linspace(-2, 2, 10)
        term = nd.NonlinearPSpline.from_nparam(x, nparam=9, scale=scale, name="x1")

        assert term.smooth.value is not None

    # UNCOMMENT TO VIEW THE GRAPH
    # def test_plot_graph(self) -> None:
    #     prior = lsl.Dist(tfd.Exponential, rate=1.0)
    #     scale = ps.ExpParam(1.0, prior).update()
    #     x = np.linspace(-2, 2, 10)
    #     term = ps.NonlinearPSpline.auto(x, nparam=9, scale=scale, name="x1")

    #     gb = lsl.GraphBuilder().add(term.smooth)
    #     gb.plot_vars()

    def test_summarise_by_quantiles(self, pred_nonlin) -> None:
        pred = pred_nonlin
        term = pred.model.loc_model.terms["x_loc_nonlin"]
        df = term.summarise_by_quantiles(pred.samples)
        assert df.shape == (300, 7)
        assert not df.isnull().values.any()

    def test_samples(self, pred_nonlin) -> None:
        key = jax.random.PRNGKey(4)
        pred = pred_nonlin
        term = pred.model.loc_model.terms["x_loc_nonlin"]
        df = term.summarise_by_samples(key, pred.samples, n=10)
        assert df.shape == (3000, 6)
        assert not df.isnull().values.any()

    def test_summarise_by_samples_fixed_val(self, samples_nonlin, model_nonlin) -> None:
        key = jax.random.PRNGKey(4)
        term = model_nonlin.loc_model.terms["x_loc_nonlin"]
        df = term.summarise_by_samples(key, samples_nonlin, x=1.0, n=10)
        assert df.shape == (10, 6)
        assert not df.isnull().values.any()


class TestPredictor:
    def test_init(self) -> None:
        prior = lsl.Dist(tfd.Exponential, rate=1.0)
        scale = nd.ExpParam(1.0, prior).update()
        x = np.linspace(-2, 2, 10)
        term = nd.NonlinearPSpline.from_nparam(x, nparam=9, scale=scale, name="x1")

        loc = nd.Predictor("loc")
        loc += term

        assert loc.parameters == term.parameters
        assert np.allclose(loc.value, term.value.value)

    def test_add_predictor(self) -> None:
        prior = lsl.Dist(tfd.Exponential, rate=1.0)
        scale = nd.ExpParam(1.0, prior).update()
        x = np.linspace(-2, 2, 10)
        term1 = nd.NonlinearPSpline.from_nparam(
            x, nparam=9, scale=scale, name="x1nonlin"
        )
        term2 = nd.LinearTerm(x, name="x1")

        loc = nd.Predictor("loc")
        loc += term1
        loc += term2

        assert loc.parameters == (term1.parameters + term2.parameters)

        terms_sum = term1.value.value + term2.value.value
        assert np.allclose(loc.value, terms_sum)

    def test_getattr_access(self) -> None:
        prior = lsl.Dist(tfd.Exponential, rate=1.0)
        scale = nd.ExpParam(1.0, prior).update()
        x = np.linspace(-2, 2, 10)
        term1 = nd.NonlinearPSpline.from_nparam(
            x, nparam=9, scale=scale, name="x1nonlin"
        )
        term2 = nd.LinearTerm(x, name="x1")

        loc = nd.Predictor("loc")
        loc += term1
        loc += term2

        assert loc.x1nonlin is term1
        assert loc.x1 is term2

    def test_getitem_access(self) -> None:
        prior = lsl.Dist(tfd.Exponential, rate=1.0)
        scale = nd.ExpParam(1.0, prior).update()
        x = np.linspace(-2, 2, 10)
        term1 = nd.NonlinearPSpline.from_nparam(
            x, nparam=9, scale=scale, name="x1nonlin"
        )
        term2 = nd.LinearTerm(x, name="x1")

        loc = nd.Predictor("loc")
        loc += term1
        loc += term2

        assert loc["x1nonlin"] is term1
        assert loc["x1"] is term2

    def test_predict_without_intercept(self) -> None:
        scale = ptm.ScaleHalfCauchy(10.0, scale=25.0, name="loc_tau")
        x = np.linspace(-2, 2, 10)
        term1 = nd.NonlinearPSpline.from_nparam(
            x, nparam=9, scale=scale, name="x1nonlin"
        )
        term2 = nd.LinearTerm(x, name="x1")

        loc = nd.Predictor("loc") + term1 + term2

        samples = dict()
        samples[term1.coef.name] = np.random.uniform(-2, 2, size=(4, 7, term1.nparam))
        samples[term1.scale.transformed.name] = np.random.uniform(
            0.1, 1, size=(4, 7, 1)
        )
        samples[term2.coef.name] = np.random.uniform(-2, 2, size=(4, 7, term2.nparam))

        prediction = loc.predict(samples, x1nonlin=1.0, x1=1.0)
        assert prediction.shape == (4, 7, 1)

        manual_prediction = term1.predict(samples, x=1.0) + term2.predict(
            samples, x=1.0
        )
        assert np.allclose(prediction, manual_prediction)

        # you can supply unused arguments. they will have no effect.
        loc.predict(samples, x1nonlin=1.0, x1=1.0, x2=1.0)

        with pytest.raises(KeyError):
            loc.predict(samples, x1nonlin=1.0)

    def test_predict_with_intercept(self) -> None:
        prior = lsl.Dist(tfd.Exponential, rate=1.0)
        scale = nd.ExpParam(1.0, prior).update()
        x = np.linspace(-2, 2, 10)
        term1 = nd.NonlinearPSpline.from_nparam(
            x, nparam=9, scale=scale, name="x1nonlin"
        )
        term2 = nd.LinearTerm(x, name="x1")

        loc = nd.Predictor.with_intercept("loc") + term1 + term2

        samples = dict()
        samples[loc.intercept.name] = np.random.uniform(-2, 2, size=(4, 7, 1))
        samples[term1.coef.name] = np.random.uniform(-2, 2, size=(4, 7, term1.nparam))
        samples[term1.scale.log_var.name] = np.random.uniform(0.1, 1, size=(4, 7, 1))
        samples[term2.coef.name] = np.random.uniform(-2, 2, size=(4, 7, term2.nparam))

        prediction = loc.predict(samples, x1nonlin=1.0, x1=1.0)
        assert prediction.shape == (4, 7, 1)

        manual_prediction = (
            samples[loc.intercept.name]
            + term1.predict(samples, x=1.0)
            + term2.predict(samples, x=1.0)
        )
        assert np.allclose(prediction, manual_prediction)

    def test_predict_broadcasting(self) -> None:
        prior = lsl.Dist(tfd.Exponential, rate=1.0)
        scale = nd.ExpParam(1.0, prior).update()
        x = np.linspace(-2, 2, 10)
        term1 = nd.NonlinearPSpline.from_nparam(
            x, nparam=9, scale=scale, name="x1nonlin"
        )
        term2 = nd.LinearTerm(x, name="x1")

        loc = nd.Predictor("loc") + term1 + term2

        samples = dict()
        samples[term1.coef.name] = np.random.uniform(-2, 2, size=(4, 7, term1.nparam))
        samples[term1.scale.log_var.name] = np.random.uniform(0.1, 1, size=(4, 7, 1))
        samples[term2.coef.name] = np.random.uniform(-2, 2, size=(4, 7, term2.nparam))

        prediction = loc.predict(samples, x1nonlin=1.0, x1=np.array([1.0, 2.0]))
        assert prediction.shape == (4, 7, 2)

        prediction = loc.predict(samples, x1nonlin=np.array([1.0, 1.5]), x1=1.0)
        assert prediction.shape == (4, 7, 2)


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


class TestLinearSmooth:
    def test_summarise(self, pred) -> None:
        df = pred.model.loc_model.terms["x_loc"].summarise_by_quantiles(pred.samples)
        assert df.shape == (100, 9)
        assert not df.isnull().values.any()

    def test_summarise_lin3(self, pred_lin3) -> None:
        df = pred_lin3.model.loc_model.terms["x_loc"].summarise_by_quantiles(
            pred_lin3.samples
        )
        assert df.shape == (900, 9)
        assert not df.isnull().values.any()

    def test_samples(self, pred) -> None:
        key = jax.random.PRNGKey(4)
        term = pred.model.loc_model.terms["x_loc"]
        df = term.summarise_by_samples(key, pred.samples, n=10)
        assert df.shape == (1000, 9)
        assert not df.isnull().values.any()

    def test_samples_lin3(self, pred_lin3) -> None:
        key = jax.random.PRNGKey(4)
        term = pred_lin3.model.loc_model.terms["x_loc"]
        df = term.summarise_by_samples(key, pred_lin3.samples, n=10)
        assert df.shape == (9000, 9)
        assert not df.isnull().values.any()

    def test_summarise_lin3_one_index_int(self, pred_lin3) -> None:
        term = pred_lin3.model.loc_model.terms["x_loc"]
        with pytest.raises(TypeError):
            term.summarise_by_quantiles(pred_lin3.samples, indices=0)

    def test_summarise_lin3_one_index(self, pred_lin3) -> None:
        term = pred_lin3.model.loc_model.terms["x_loc"]
        df = term.summarise_by_quantiles(pred_lin3.samples, indices=[0])
        assert df.shape == (300, 9)
        assert not df.isnull().values.any()

    def test_summarise_lin3_two_indices(self, pred_lin3) -> None:
        term = pred_lin3.model.loc_model.terms["x_loc"]
        df = term.summarise_by_quantiles(pred_lin3.samples, indices=[0, 1])
        assert df.shape == (600, 9)
        assert not df.isnull().values.any()

    def test_samples_lin3_one_index(self, pred_lin3) -> None:
        key = jax.random.PRNGKey(4)
        term = pred_lin3.model.loc_model.terms["x_loc"]
        df = term.summarise_by_samples(key, pred_lin3.samples, n=10, indices=[0])
        assert df.shape == (3000, 9)
        assert not df.isnull().values.any()

    def test_samples_lin3_two_indices(self, pred_lin3) -> None:
        key = jax.random.PRNGKey(4)
        term = pred_lin3.model.loc_model.terms["x_loc"]
        df = term.summarise_by_samples(key, pred_lin3.samples, n=10, indices=[0, 1])
        assert df.shape == (6000, 9)
        assert not df.isnull().values.any()


class TestFindParam:
    def test_var_is_param(self):
        tau2 = lsl.param(1.0, name="tau2")
        assert nd.find_param(tau2) == tau2

    def test_var_is_not_param(self):
        tau2 = lsl.Var(1.0, name="tau2")
        assert nd.find_param(tau2) is None

    def test_var_is_not_param_but_has_param(self):
        prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
        tau2 = Var(1.0, prior, name="tau2")
        tau2.parameter = True
        tau2_log = tau2.transform(None)
        assert nd.find_param(tau2) is tau2_log

    def test_two_calcs_until_param(self):
        tau2 = nd.VarHalfCauchy(1.0, scale=25.0, name="tau2")
        tau_transformed = nd.find_param(tau2)
        assert tau_transformed is not None


class TestIGGibbsTransition:
    def test_returns_transition(self):
        prior = lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.01)
        tau2 = lsl.param(1.0, prior, name="tau2")
        s = nd.S.pspline(x, nparam=10, tau2=tau2, name="x1")

        transition_fn = nd.ig_gibbs_transition_fn(s)
        assert transition_fn is not None

    def test_transition_runs(self):
        prior = lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.01)
        tau2 = lsl.param(1.0, prior, name="tau2")
        s = nd.S.pspline(x, nparam=10, tau2=tau2, name="x1")

        transition_fn = nd.ig_gibbs_transition_fn(s)

        key = jax.random.PRNGKey(4)

        model = lsl.GraphBuilder().add(s.value).build_model()
        result = transition_fn(key, model.state)
        assert result is not None
        assert result[tau2.name] == pytest.approx(0.0027, abs=1e-3)


class TestS:
    def test_tau2_ig(self):
        tau2 = nd.VarInverseGamma(
            1.0, concentration=1.0, scale=0.01, bijector=None, name="tau2"
        )
        s = nd.S.pspline(x, nparam=10, tau2=tau2, name="x1")

        assert isinstance(s.mcmc_kernels[0], gs.NUTSKernel)
        assert isinstance(s.mcmc_kernels[1], gs.GibbsKernel)

    def test_tau_hc(self):
        tau2 = nd.VarHalfCauchy(1.0, scale=25.0, name="tau2")
        s = nd.S.pspline(x, nparam=10, tau2=tau2, name="x1")

        assert isinstance(s.mcmc_kernels[0], gs.NUTSKernel)
        assert isinstance(s.mcmc_kernels[1], gs.NUTSKernel)


class TestVarParams:
    def test_var_ig(self):
        ptm.VarInverseGamma(1.0, 1.0, 0.5, name="tau2")

    def test_scale_ig(self):
        ptm.ScaleInverseGamma(1.0, 1.0, 0.5, name="tau")

    def test_var_hc(self):
        ptm.VarHalfCauchy(1.0, scale=25.0, name="tau2")

    def test_scale_hc(self):
        ptm.ScaleHalfCauchy(1.0, scale=25.0, name="tau")

    def test_var_weibull(self):
        ptm.VarWeibull(1.0, scale=0.05, name="tau2")

    def test_scale_weibull(self):
        ptm.ScaleWeibull(1.0, scale=0.05, name="tau")

    def test_var_weibull01(self):
        ptm.VarWeibull(1.0, scale=0.01, name="tau2")

    def test_scale_weibull01(self):
        ptm.ScaleWeibull(1.0, scale=0.01, name="tau")

    def test_var_weibull003(self):
        with pytest.raises(AssertionError):
            ptm.VarWeibull(1.0, scale=0.003, name="tau2")

        ptm.VarWeibull(0.5, scale=0.003, name="tau2")

    def test_scale_weibull003(self):
        with pytest.raises(AssertionError):
            ptm.ScaleWeibull(1.0, scale=0.003, name="tau")

        ptm.ScaleWeibull(0.5, scale=0.003, name="tau")


class TestTransformedVar:
    def test_value_no_dist(self):
        var = ptm.TransformedVar(value=1.0, name="test")
        assert var.value == pytest.approx(1.0)


class TestOnionCoefParam:
    def test_nparam(self):
        knots = nd.OnionKnots(a=-3.0, b=3.0, nparam=10)
        coef = nd.OnionCoefParam(knots=knots, name="coef")

        assert coef.value.shape == (knots.nparam + 7,)

    def test_values(self):
        knots = nd.OnionKnots(a=-3.0, b=3.0, nparam=10)
        coef = nd.OnionCoefParam(knots=knots, name="coef")

        assert np.allclose(knots.knots[2:-2], coef.value)
