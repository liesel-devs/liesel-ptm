"""THROWAWAY: two subclasses versus one mode-selecting Model subclass.

Question: can native .pred predictors accept terms after Model initialization,
without a second builder lifecycle, while preserving legacy intercept prediction?

Run: uv run --locked python demo/prototype_issue28.py

Variant A: PTM(y, knots=...) / LegacyPTM(y, knots=...).
Variant B: ModePTM(y, knots=..., legacy=False/True).
Both use the same numerical construction and the same inputs/terms below.
Legacy predictor attributes currently hold intercept-free effects; computed
intercepts enter the likelihood separately, as agreed for the planned interface.
Verdict: native graph growth works; two dedicated subclasses keep the two
formulations explicit. Copying, Gibbs coverage and recorded-intercept prediction
work without a finalize step. Shape-prior shorthand matches the reference recipe.
This is exploratory code, not the proposed production implementation.
"""

import json
import signal
from collections import Counter
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import liesel_gam as gam
import numpy as np

import liesel_ptm as ptm


@dataclass
class Predictors:
    loc: gam.AdditivePredictor
    scale: gam.AdditivePredictor
    shape: gam.MVAdditivePredictor


class _PrototypeModel(lsl.Model):
    def __init__(self, y, *, knots, legacy: bool):
        self.legacy = legacy
        loc = gam.AdditivePredictor(
            "loc_terms" if legacy else "loc", intercept=not legacy
        )
        scale = gam.AdditivePredictor(
            "scale_terms" if legacy else "scale", inv_link=jnp.exp, intercept=not legacy
        )
        shape = gam.MVAdditivePredictor.from_random_walk(
            "delta",
            ndim=knots.nparam,
            intercept_scale=gam.scale_wb(0.1, scale=0.05),
        ).constrain("sumzero_coef")
        self.pred = Predictors(loc, scale, shape)
        response = lsl.Var.new_obs(y, name="y")
        if legacy:
            # Match the existing legacy construction: the observation variable's
            # own value node would introduce a cycle in Liesel's variable graph.
            training_y = lsl.Data(y)
            beta = ptm.LocationIntercept(
                training_y, loc, scale.linear_predictor, name="beta0"
            )
            full_loc = lsl.Var.new_calc(lambda b, m: b + m, beta, loc, name="loc")
            gamma = ptm.LogScaleIntercept(
                training_y, full_loc, scale.linear_predictor, name="gamma0"
            )
            full_scale = lsl.Var.new_calc(
                lambda g, relative: jnp.exp(g) * relative, gamma, scale, name="scale"
            )
        else:
            full_loc, full_scale = loc, scale
        response.dist_node = ptm.PTMDist(
            knots.knots,
            loc=full_loc,
            scale=full_scale,
            shape=shape,
            bspline="ptm" if legacy else "onion",
            centered=not legacy,
            scaled=not legacy,
        )
        super().__init__(response)

    def setup_mcmc(self, seed: int = 42, num_chains: int = 2) -> gs.EngineBuilder:
        assert self.legacy, "Only legacy setup is exercised in this prototype."
        return ptm.LegacyPTM(self).setup_mcmc(
            loc_terms=self.pred.loc,
            log_scale_terms=self.pred.scale,
            shape=self.pred.shape,
            seed=seed,
            num_chains=num_chains,
        )

    def predict(self, samples, predict=None, newdata=None, chunk_size=64):
        if not self.legacy:
            return super().predict(samples, predict, newdata, chunk_size)
        return ptm.LegacyPTM(self).predict(
            samples,
            predict=list(self.vars) if predict is None else predict,
            newdata=newdata,
            chunk_size=chunk_size,
        )


class PTM(_PrototypeModel):
    def __init__(self, y, *, knots):
        super().__init__(y, knots=knots, legacy=False)


class LegacyPTM(_PrototypeModel):
    def __init__(self, y, *, knots):
        super().__init__(y, knots=knots, legacy=True)


class ModePTM(_PrototypeModel):
    """Variant B: inherited constructor selects the formulation with a flag."""


def add_terms(model: _PrototypeModel, registry: gam.DictRegistry):
    """Identical post-initialization calls for both variants."""
    before = set(model.parameters)
    loc_builder = gam.TermBuilder(registry, prefix_names_by="m.")
    scale_builder = gam.TermBuilder(registry, prefix_names_by="s.")
    loc_term = loc_builder.slin("x", scale=1.0, name="m.x")
    scale_term = scale_builder.slin("x", scale=1.0, name="s.x")
    smooth = loc_builder.ps("x", k=5, scale=gam.VarIGPrior(1.0, 0.001), name="m.ps(x)")
    model.pred.loc += loc_term
    model.pred.loc += smooth
    model.pred.scale += scale_term
    assert loc_term.coef.model is model and scale_term.coef.model is model
    assert smooth.coef.model is model
    expected_new = {loc_term.coef.name, scale_term.coef.name}
    expected_new |= set(model.parental_submodel(smooth).parameters)
    assert set(model.parameters) - before == expected_new
    position = {
        loc_term.coef.name: jnp.full_like(loc_term.coef.value, 0.4),
        scale_term.coef.name: jnp.full_like(scale_term.coef.value, 0.2),
    }
    if not model.legacy:
        shape_builder = gam.MVTermBuilder.from_predictor(
            model.pred.shape, gam.TermBuilder(registry, prefix_names_by="h.")
        )
        shape_term = shape_builder.lin("x", dimension_scale=0.3, name="h.x")
        model.pred.shape += shape_term
        assert shape_term.coef.model is model
        position[shape_term.coef.name] = jnp.linspace(
            -0.1, 0.1, shape_term.coef.value.size
        ).reshape(shape_term.coef.value.shape)
    model.state = model.update_state(position)
    assert not model.graph_outdated
    return len(before), len(model.parameters)


def inspect_case(label: str, model: _PrototypeModel, registry: gam.DictRegistry):
    before, after = add_terms(model, registry)
    assert isinstance(model, lsl.Model)
    assert np.isfinite(model.log_prob)
    np.testing.assert_allclose(model.pred.shape.value.sum(axis=-1), 0.0, atol=1e-6)
    copied = model.copy()
    assert isinstance(copied, _PrototypeModel)
    for key in ("loc", "scale", "shape"):
        predictor = getattr(copied.pred, key)
        assert predictor is copied.vars[predictor.name]
        assert predictor is not getattr(model.pred, key)
    samples = {
        name: jnp.stack((var.value, var.value))[None, ...]
        for name, var in model.parameters.items()
    }
    if model.legacy:
        residual = (model.vars["y"].value - model.vars["loc"].value) / model.vars[
            "scale"
        ].value
        np.testing.assert_allclose(residual.mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(residual.std(), 1.0, atol=1e-6)
        builder = model.setup_mcmc()
        keys = Counter(k for kernel in builder.kernels for k in kernel.position_keys)
        assert keys == Counter(model.parameters.keys())
        assert builder.positions_included == ["beta0", "gamma0"]
        assert any(isinstance(kernel, gs.GibbsKernel) for kernel in builder.kernels)
        samples["beta0"] = jnp.array([[10.0, 20.0]])
        samples["gamma0"] = jnp.log(jnp.array([[2.0, 3.0]]))
    x_new = jnp.array([-0.5, 0.5, 0.9])
    prediction = model.predict(
        samples, predict=["loc", "scale", "delta"], newdata={"x": x_new}
    )
    assert prediction["loc"].shape == prediction["scale"].shape == (1, 2, 3)
    if model.legacy:
        np.testing.assert_allclose(
            prediction["loc"], samples["beta0"][..., None] + 0.4 * x_new, atol=2e-6
        )
        np.testing.assert_allclose(
            prediction["scale"],
            jnp.exp(samples["gamma0"][..., None] + 0.2 * x_new),
            atol=2e-6,
        )
    result = {
        "variant": label,
        "formulation": "legacy" if model.legacy else "modern",
        "parameter_counts_before_after": [before, after],
        "training_shape": list(model.pred.shape.value.shape),
        "prediction_shapes": {
            key: list(value.shape) for key, value in prediction.items()
        },
        "log_prob": float(model.log_prob),
        "native_graph_growth_copy_and_prediction": "PASS",
    }
    print(json.dumps(result, indent=2), flush=True)
    return float(model.log_prob)


def main():
    signal.alarm(120)
    x = np.linspace(-1.0, 1.0, 8)
    y = 1.0 + 0.5 * x + np.array([0.2, -0.8, 0.3, 1.0, -0.4, 0.6, -0.9, 0.5])
    print(
        "Inputs: 8 observations; post-init location/scale slopes 0.4/0.2.", flush=True
    )
    scores: dict[str, dict[str, Any]] = {}
    for label in ("A: two subclasses", "B: one class with mode"):
        scores[label] = {}
        for legacy in (False, True):
            knots = (
                ptm.PTMKnots(-4.0, 4.0, nparam=6)
                if legacy
                else ptm.OnionKnots(-4.0, 4.0, nparam=6)
            )
            registry = gam.DictRegistry({"x": x})
            if label.startswith("A"):
                model = LegacyPTM(y, knots=knots) if legacy else PTM(y, knots=knots)
            else:
                model = ModePTM(y, knots=knots, legacy=legacy)
            scores[label][str(legacy)] = inspect_case(label, model, registry)
    np.testing.assert_allclose(
        list(scores["A: two subclasses"].values()),
        list(scores["B: one class with mode"].values()),
        atol=1e-5,
    )
    print("VERDICT: entry-point variants agree; no finalize hook or graph observer.")
    print("Recommendation: two subclasses; inference remains in Liesel/liesel-gam.")


if __name__ == "__main__":
    main()
