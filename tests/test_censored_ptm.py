"""PTM.from_censored must match an independent censored likelihood."""

from functools import partial

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm import survival_dist as surv

NAN = np.nan
X = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0, 0.25])
RECORDS = jnp.array(
    [
        [0.3, NAN, NAN],
        [NAN, NAN, -0.2],
        [NAN, 0.4, NAN],
        [NAN, -0.6, 0.9],
        [-1.1, NAN, NAN],
        [NAN, 1.2, NAN],
    ]
)


def likelihood_node(model) -> lsl.Dist:
    node = model.response.dist_node
    assert node is not None
    return node


def plot_mean(plot) -> np.ndarray:
    assert isinstance(plot.data, pd.DataFrame)
    return plot.data["mean"].to_numpy()


def add_terms(model, x=X, *, conditional_shape=False):
    builder = gam.TermBuilder.from_dict({"x": x})
    model.pred.loc += builder.slin("x", scale=1.0, name="m.x")
    model.pred.scale += builder.slin("x", scale=0.5, name="s.x")
    if conditional_shape:
        shape_builder = gam.MVTermBuilder.from_predictor(
            model.pred.shape, term_builder=builder
        )
        model.pred.shape.extend([shape_builder.slin("x", scale=1.0)])
    return model


def nonzero_position(model):
    return {
        name: var.value
        + jnp.linspace(0.03, 0.09, var.value.size).reshape(var.value.shape)
        for name, var in model.parameters.items()
    }


def manual_censored_model(records, *, conditional_shape=False, x=X):
    """Previous guide recipe: replace the likelihood of an ordinary PTM."""
    model = add_terms(
        ptm.PTM(jnp.zeros(records.shape[0]), nshape=4),
        x,
        conditional_shape=conditional_shape,
    )
    likelihood = likelihood_node(model)
    model.auto_update = False
    base = likelihood.distribution

    def censored(**inputs):
        dist = base(**inputs)
        assert isinstance(dist, tfd.Distribution)
        return surv.CensoredDistribution(dist)

    likelihood.distribution = censored
    model.response.value = records
    model.auto_update = True
    model.update()
    return model


def rowwise_log_lik(ordinary, records):
    """Censored log likelihood from the ordinary model's event-time distribution."""
    dist = likelihood_node(ordinary).init_dist()
    status = surv.censoring_status(records)
    time, lower, upper = (np.asarray(records[:, i]) for i in range(3))
    safe = np.nan_to_num
    contributions = {
        "uncensored": dist.log_prob(safe(time)),
        "left_censored": dist.log_cdf(safe(upper)),
        "right_censored": dist.log_survival_function(safe(lower)),
        "interval_censored": jnp.log(dist.cdf(safe(upper)) - dist.cdf(safe(lower))),
    }
    return sum(
        jnp.where(status == label, value, 0.0).sum()
        for label, value in contributions.items()
    )


def jit_log_prob_grad(model, position):
    interface = gs.LieselInterface(model)
    state = model.state

    @jax.jit
    def grad(position):
        return jax.grad(lambda p: interface.log_prob(interface.update_state(p, state)))(
            position
        )

    return grad(position)


class TestConstruction:
    def test_array_records_become_observed_response_named_y(self):
        model = ptm.PTM.from_censored(RECORDS, nshape=4)
        assert type(model) is ptm.PTM
        assert model.response is model.vars["y"]
        assert model.response.observed
        assert model.response.value.shape == (6, 3)
        node = likelihood_node(model)
        assert isinstance(node, surv.CensoredDist)
        assert not node.cache
        assert node.init_dist().censoring_records is None
        assert isinstance(node.init_dist(), surv.CensoredDistribution)
        assert np.isfinite(model.log_prob)

    def test_supplied_response_and_predictors_are_retained(self):
        response = lsl.Var.new_obs(RECORDS, name="log_time")
        loc = gam.AdditivePredictor("mu")
        scale = gam.AdditivePredictor("sigma", inv_link=jnp.exp)
        shape = gam.MVAdditivePredictor.from_random_walk(
            "d", ndim=4, intercept_scale=0.3
        ).constrain("sumzero_coef")
        model = ptm.PTM.from_censored(
            response, nshape=4, loc=loc, scale=scale, shape=shape
        )
        assert model.response is response is model.vars["log_time"]
        assert model.pred.loc is loc and model.pred.scale is scale
        assert model.pred.shape is shape

    def test_options_match_ordinary_constructor(self):
        censored = ptm.PTM.from_censored(
            RECORDS, nshape=5, bounds=(-3.0, 5.0), shape_scale=0.25
        )
        ordinary = ptm.PTM(X, nshape=5, bounds=(-3.0, 5.0), shape_scale=0.25)
        assert censored.parameters.keys() == ordinary.parameters.keys()
        np.testing.assert_allclose(censored.log_prior, ordinary.log_prior)
        censored_dist = likelihood_node(censored).init_dist()
        ordinary_dist = likelihood_node(ordinary).init_dist()
        assert isinstance(censored_dist, surv.CensoredDistribution)
        assert isinstance(ordinary_dist, ptm.LocScaleTransformationDist)
        base = censored_dist.distribution
        assert isinstance(base, ptm.LocScaleTransformationDist)
        knots = base.bspline.knots
        expected = ordinary_dist.bspline.knots
        np.testing.assert_array_equal(knots, expected)

    def test_ordinary_constructor_is_unchanged(self):
        distribution = likelihood_node(ptm.PTM(X, nshape=4)).distribution
        assert isinstance(distribution, partial)
        assert distribution.func is ptm.LocScaleTransformationDist

    def test_copy_keeps_censored_likelihood(self):
        model = add_terms(ptm.PTM.from_censored(RECORDS, nshape=4))
        copied = model.copy()
        assert isinstance(copied, ptm.PTM)
        assert copied.response is not model.response
        node = likelihood_node(copied)
        assert isinstance(node, surv.CensoredDist)
        assert isinstance(node.init_dist(), surv.CensoredDistribution)
        np.testing.assert_allclose(copied.log_lik, model.log_lik)


@pytest.mark.parametrize("conditional_shape", [False, True])
class TestLikelihood:
    def test_matches_manual_constructions_at_nonzero_coefficients(
        self, conditional_shape
    ):
        model = add_terms(
            ptm.PTM.from_censored(RECORDS, nshape=4),
            conditional_shape=conditional_shape,
        )
        manual = manual_censored_model(RECORDS, conditional_shape=conditional_shape)
        ordinary = add_terms(
            ptm.PTM(jnp.zeros(6), nshape=4), conditional_shape=conditional_shape
        )
        assert model.parameters.keys() == manual.parameters.keys()
        position = nonzero_position(model)
        for graph in (model, manual, ordinary):
            graph.state = graph.update_state(position)

        np.testing.assert_allclose(model.log_prior, manual.log_prior, rtol=1e-6)
        np.testing.assert_allclose(model.log_prior, ordinary.log_prior, rtol=1e-6)
        np.testing.assert_allclose(model.log_lik, manual.log_lik, rtol=1e-6)
        np.testing.assert_allclose(
            model.log_lik, rowwise_log_lik(ordinary, RECORDS), rtol=1e-5
        )
        actual = jit_log_prob_grad(model, position)
        expected = jit_log_prob_grad(manual, position)
        for name in position:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-5)
            assert np.all(np.isfinite(actual[name]))

    def test_record_updates_change_statuses(self, conditional_shape):
        model = add_terms(
            ptm.PTM.from_censored(RECORDS, nshape=4),
            conditional_shape=conditional_shape,
        )
        updated = surv.censored(
            jnp.array([0.3, -jnp.inf, 0.3, 0.1, -1.1, 0.5]),
            jnp.array([jnp.inf, 0.4, 0.3, 0.2, -1.1, jnp.inf]),
        )
        fresh = add_terms(
            ptm.PTM.from_censored(updated, nshape=4),
            conditional_shape=conditional_shape,
        )
        position = nonzero_position(model)
        model.state = model.update_state(position)
        model.response.value = updated
        model.update()
        fresh.state = fresh.update_state(position)
        assert not (
            surv.censoring_status(updated) == surv.censoring_status(RECORDS)
        ).all()
        np.testing.assert_allclose(model.log_lik, fresh.log_lik, rtol=1e-6)

    def test_row_shuffling_preserves_log_likelihood(self, conditional_shape):
        order = jnp.array([3, 0, 5, 1, 4, 2])
        model = add_terms(
            ptm.PTM.from_censored(RECORDS, nshape=4),
            conditional_shape=conditional_shape,
        )
        shuffled = add_terms(
            ptm.PTM.from_censored(RECORDS[order], nshape=4),
            X[order],
            conditional_shape=conditional_shape,
        )
        position = nonzero_position(model)
        model.state = model.update_state(position)
        shuffled.state = shuffled.update_state(position)
        np.testing.assert_allclose(model.log_lik, shuffled.log_lik, rtol=1e-6)

    def test_predicts_at_new_covariates_like_ordinary_model(self, conditional_shape):
        model = add_terms(
            ptm.PTM.from_censored(RECORDS, nshape=4),
            conditional_shape=conditional_shape,
        )
        ordinary = add_terms(
            ptm.PTM(jnp.zeros(6), nshape=4), conditional_shape=conditional_shape
        )
        samples = {
            name: jnp.broadcast_to(value, (1, 2) + value.shape)
            for name, value in nonzero_position(model).items()
        }
        newdata = {"x": jnp.array([-0.3, 0.7])}
        names = ["loc", "scale", "shape"]
        actual = model.predict(samples, predict=names, newdata=newdata)
        expected = ordinary.predict(samples, predict=names, newdata=newdata)
        for name in names:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6)


class TestRecordValidation:
    @pytest.mark.parametrize(
        "records",
        [
            jnp.zeros((0, 3)),
            jnp.zeros((4, 2)),
            jnp.zeros(3),
            jnp.zeros((2, 2, 3)),
        ],
    )
    def test_rejects_record_shapes(self, records):
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            ptm.PTM.from_censored(records, nshape=4)

    def test_rejects_invalid_record_patterns(self):
        records = RECORDS.at[2].set(jnp.array([NAN, NAN, NAN]))
        with pytest.raises(ValueError, match="invalid.*row indices 2"):
            ptm.PTM.from_censored(records, nshape=4)

    def test_rejects_nonpositive_interval_width(self):
        records = RECORDS.at[3].set(jnp.array([NAN, 0.5, 0.5]))
        with pytest.raises(ValueError, match="positive width"):
            ptm.PTM.from_censored(records, nshape=4)

    def test_collapsed_interval_never_becomes_uncensored(self):
        records = surv.censored(np.array([0.3, 1.0]), np.array([0.3, 1.0 + 1e-9]))
        assert surv.censoring_status(records).tolist() == [
            "uncensored",
            "interval_censored",
        ]
        response = lsl.Var.new_obs(records, name="log_time")
        with pytest.raises(ValueError, match="positive width.*float32"):
            ptm.PTM.from_censored(response, nshape=4)
        assert response.dist_node is None and response.model is None

    def test_rejects_owned_or_distributed_response(self):
        response = lsl.Var.new_obs(RECORDS, name="y")
        ptm.PTM.from_censored(response, nshape=4)
        with pytest.raises(ValueError, match="unowned"):
            ptm.PTM.from_censored(response, nshape=4)
        unobserved = lsl.Var.new_value(RECORDS, name="y")
        with pytest.raises(ValueError, match="unowned"):
            ptm.PTM.from_censored(unobserved, nshape=4)

    @pytest.mark.parametrize(
        "kwargs", [{"nshape": 1}, {"nshape": 4, "bounds": (1.0, -1.0)}]
    )
    def test_validates_geometry(self, kwargs):
        with pytest.raises(ValueError, match="nshape|bounds"):
            ptm.PTM.from_censored(RECORDS, **kwargs)

    def test_failed_construction_leaves_response_reusable(self):
        response = lsl.Var.new_obs(RECORDS, name="loc")
        with pytest.raises(RuntimeError, match="Duplicate"):
            ptm.PTM.from_censored(response, nshape=4)
        assert response.dist_node is None and response.model is None
        response.name = "y"
        assert ptm.PTM.from_censored(response, nshape=4).vars["y"] is response

    def test_late_failure_releases_nodes(self, monkeypatch):
        response = lsl.Var.new_obs(RECORDS, name="y")
        initialize = lsl.Model.__init__

        def fail_after_initialization(*args, **kwargs):
            initialize(*args, **kwargs)
            raise RuntimeError("late failure")

        with monkeypatch.context() as patch:
            patch.setattr(lsl.Model, "__init__", fail_after_initialization)
            with pytest.raises(RuntimeError, match="late failure"):
                ptm.PTM.from_censored(response, nshape=4)
        assert response.dist_node is None and response.model is None
        assert np.isfinite(ptm.PTM.from_censored(response, nshape=4).log_prob)


class TestSummariesAndPlots:
    @pytest.fixture
    def models(self):
        censored = add_terms(ptm.PTM.from_censored(RECORDS, nshape=4))
        ordinary = add_terms(ptm.PTM(jnp.zeros(6), nshape=4))
        position = nonzero_position(censored)
        samples = {
            name: jnp.stack([value, 1.1 * value])[None]
            for name, value in position.items()
        }
        return censored, ordinary, samples

    def test_intercept_summary_and_plot_use_uncensored_distribution(self, models):
        censored, ordinary, samples = models
        intercept = censored.pred.shape.intercept
        actual = ptm.summarise_intercept_dist(
            censored.response, intercept, samples, rgrid=7
        )
        expected = ptm.summarise_intercept_dist(
            ordinary.response, ordinary.pred.shape.intercept, samples, rgrid=7
        )
        np.testing.assert_allclose(actual["mean"], expected["mean"], rtol=1e-6)
        plot = ptm.plot_intercept_dist(
            censored.response, intercept, samples, rgrid=7, show_n_samples=2
        )
        reference = ptm.plot_intercept_dist(
            ordinary.response,
            ordinary.pred.shape.intercept,
            samples,
            rgrid=7,
            show_n_samples=2,
        )
        np.testing.assert_allclose(plot_mean(plot), plot_mean(reference))

    def test_conditional_survival_from_exponential_cdf_summary(self, models):
        censored, ordinary, samples = models
        time_grid = jnp.array([0.5, 1.0, 2.0, 4.0])
        newdata = {"x": [-0.5, 0.5]}
        summary, reference = (
            ptm.summarise_conditional_dist(
                response,
                samples,
                newdata=newdata,
                rgrid=time_grid,
                response_bijector=tfb.Exp(),
                include_loc=True,
                include_scale=True,
            )
            for response in (censored.response, ordinary.response)
        )
        np.testing.assert_allclose(summary["mean"], reference["mean"], rtol=1e-6)

        cdf = summary.loc[summary["quantity"] == "cdf"]
        mean_s = 1.0 - cdf["mean"].to_numpy()
        lower_s = 1.0 - cdf["q_0.95"].to_numpy()
        upper_s = 1.0 - cdf["q_0.05"].to_numpy()

        prediction = censored.predict(
            samples, predict=["loc", "scale", "shape"], newdata=newdata
        )
        node = likelihood_node(censored)
        assert isinstance(node, surv.CensoredDist)
        base = node.base_distribution
        event_times = tfd.TransformedDistribution(
            base(
                coef=prediction["shape"][..., None, :],
                loc=prediction["loc"],
                scale=prediction["scale"],
            ),
            bijector=tfb.Exp(),
        )
        survival = event_times.survival_function(time_grid[:, None, None, None])
        # (time, chain, draw, condition) -> (draws, condition-major rows)
        draws = np.moveaxis(np.asarray(survival), 0, -1).reshape(2, -1)
        np.testing.assert_allclose(mean_s, draws.mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(
            lower_s, np.quantile(draws, 0.05, axis=0), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            upper_s, np.quantile(draws, 0.95, axis=0), rtol=1e-5, atol=1e-6
        )
        assert np.all(lower_s <= mean_s) and np.all(mean_s <= upper_s)

    def test_conditional_plot_accepts_censored_response(self, models):
        censored, ordinary, samples = models
        plot, reference = (
            ptm.plot_conditional_dist(
                response, samples, newdata={"x": [-0.5, 0.5]}, rgrid=5
            )
            for response in (censored.response, ordinary.response)
        )
        np.testing.assert_allclose(plot_mean(plot), plot_mean(reference))
