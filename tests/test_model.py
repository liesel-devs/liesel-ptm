from typing import Any

import jax
import jax.numpy as jnp
import liesel.goose as gs
import tensorflow_probability.substrates.jax.bijectors as tfb

import liesel_ptm as ptm
from liesel_ptm.util.testing import mock_samples

Array = Any

nobs = 100
nparam = 10
knots = ptm.LogIncKnots(a=-4.0, b=4.0, nparam=nparam)
y = jax.random.normal(jax.random.key(1), (nobs,))


model = ptm.LocScalePTM(y, knots.knots)
model.trafo += ptm.PTMCoef.new_rw1_sumzero(
    knots=knots.knots,
    scale=ptm.ScaleWeibull(
        1.0,
        scale=0.02,
        name="tau",
        inference=gs.MCMCSpec(gs.HMCKernel),
        bijector=tfb.Exp(),
    ),
    name="trafo0",
    inference=gs.MCMCSpec(gs.HMCKernel),
)
model.build()


samples = mock_samples(model.graph, (2, 3))


class TestEvaluatePTM:
    def test_waic(self) -> None:
        meval = ptm.EvaluatePTM(model, samples)
        assert meval.waic() is not None

    def test_log_prob(self) -> None:
        meval = ptm.EvaluatePTM(model, samples)
        lp = meval.log_prob(newdata={"response": model.response.value})
        dist = model.init_dist(samples)
        assert jnp.allclose(lp, dist.log_prob(model.response.value))

    def test_lppdi(self) -> None:
        meval = ptm.EvaluatePTM(model, samples)
        newdata = {"response": y}
        lppdi = meval.lppdi(newdata)
        assert lppdi is not None

    def test_quantile_score(self) -> None:
        meval = ptm.EvaluatePTM(model, samples)
        probs = jnp.linspace(0.1, 0.9, 5)
        qs = meval.quantile_score(probs)
        assert qs is not None

    def test_crps(self) -> None:
        meval = ptm.EvaluatePTM(model, samples)
        probs = jnp.linspace(0.1, 0.9, 5)
        crps = meval.crps(probs)

        assert crps is not None

    def test_w1(self) -> None:
        meval = ptm.EvaluatePTM(model, samples)

        mad = meval.cdf_mad(
            true_cdf=jnp.linspace(0.01, 0.99, 100), newdata={"response": y}
        )

        assert mad is not None

    def test_w2(self) -> None:
        meval = ptm.EvaluatePTM(model, samples)

        mad = meval.quantile_mse(
            true_cdf=jnp.linspace(0.01, 0.99, 100), newdata={"response": y}
        )

        assert mad is not None
