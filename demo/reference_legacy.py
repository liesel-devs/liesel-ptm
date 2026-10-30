"""Legacy PTM with Gaussian IWLS, shape NUTS and recorded-intercept prediction.

Run with ``uv run python -m demo.reference_legacy``. The short fit checks integration;
use longer chains and inspect diagnostics for statistical inference.
"""

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
import pandas as pd

import liesel_ptm as ptm


def build_model() -> lsl.Model:
    """Construct the shared-shape PTM without the legacy model wrapper."""
    x = np.linspace(-1.0, 1.0, 24)
    y = 1.0 + 0.5 * x + np.exp(0.15 * x) * np.random.default_rng(42).normal(size=x.size)
    registry = gam.PandasRegistry(pd.DataFrame({"x": x}))
    response_value = lsl.Data(y)
    loc_terms = gam.AdditivePredictor("loc_terms", intercept=False)
    log_scale_terms = gam.AdditivePredictor("log_scale_terms", intercept=False)
    loc_builder = gam.TermBuilder(registry, prefix_names_by="m.")
    loc_terms += loc_builder.slin("x", scale=1.0, name="m.x")
    # A centered smooth adds a conjugate smoothing-variance Gibbs update.
    loc_terms += loc_builder.ps(
        "x", k=5, scale=gam.VarIGPrior(1.0, 0.001), name="m.ps(x)"
    )
    log_scale_terms += gam.TermBuilder(registry, prefix_names_by="s.").slin(
        "x", scale=1.0, name="s.x"
    )

    beta0 = ptm.LocationIntercept(
        response_value, loc_terms, log_scale_terms, name="beta0"
    )
    loc = gam.AdditivePredictor("loc", intercept=beta0)
    loc += loc_terms
    gamma0 = ptm.LogScaleIntercept(response_value, loc, log_scale_terms, name="gamma0")
    scale = gam.AdditivePredictor("scale", inv_link=jnp.exp, intercept=gamma0)
    scale += log_scale_terms

    knots = ptm.PTMKnots(-4.0, 4.0, nparam=6)
    shape = gam.MVAdditivePredictor.from_random_walk(
        "delta", ndim=knots.nparam, intercept=False
    ).constrain("sumzero_coef")
    builder = gam.MVTermBuilder.from_predictor(
        shape,
        term_builder=gam.TermBuilder(registry, prefix_names_by="h."),
    )
    shape.intercept = builder.intercept(scale=gam.scale_wb(0.1, scale=0.05))

    # The modern Weibull variance prior is intentionally unclipped.
    response = lsl.Var.new_obs(
        y,
        dist=ptm.PTMDist(
            knots.knots,
            loc=loc,
            scale=scale,
            shape=shape,
            bspline="ptm",
            centered=False,
            scaled=False,
        ),
        name="y",
    )
    return lsl.Model(response)


def fit(
    model: lsl.Model, adaptation: int = 80, posterior: int = 12, num_chains: int = 2
) -> gs.SamplingResults:
    """Sample all parameters and record both computed intercepts at every draw."""
    legacy = ptm.LegacyPTM(model)
    builder = legacy.setup_mcmc(
        loc_terms=model.vars["loc_terms"],
        log_scale_terms=model.vars["log_scale_terms"],
        shape=model.vars["delta"],
        seed=43,
        num_chains=num_chains,
    )
    builder.show_progress = False
    builder.add_adaptation(adaptation)
    builder.add_posterior(posterior)
    engine = builder.build()
    engine.sample_all_epochs()
    return engine.get_results()


if __name__ == "__main__":
    model = build_model()
    legacy = ptm.LegacyPTM(model)
    samples = fit(model).get_posterior_samples()
    prediction = legacy.predict(
        samples,
        predict=["loc", "scale", "delta"],
        newdata={"x": jnp.array([-0.5, 0.5])},
    )
    print(
        "Recorded intercept shapes:",
        {name: samples[name].shape for name in ("beta0", "gamma0")},
    )
    print(
        "Prediction shapes:", {name: value.shape for name, value in prediction.items()}
    )
