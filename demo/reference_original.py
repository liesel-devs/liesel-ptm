"""Original PTM construction with computed intercepts and modern deltas.

Run with ``uv run python -m demo.reference_original``. Convenient Gaussian-proposal
IWLS plus NUTS/HMC setup on these modern terms is the remaining work in issue #7.
"""

import jax.numpy as jnp
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm.model.model import PTMDist
from liesel_ptm.predictor import LocationIntercept, LogScaleIntercept


def build_model() -> lsl.Model:
    """Construct the shared-shape PTM without the legacy model wrapper."""
    x = np.linspace(-1.0, 1.0, 24)
    y = 1.0 + 0.5 * x + np.exp(0.15 * x) * np.random.default_rng(42).normal(size=x.size)
    registry = gam.PandasRegistry(pd.DataFrame({"x": x}))
    response_value = lsl.Data(y)
    loc_terms = gam.AdditivePredictor("loc_terms", intercept=False)
    log_scale_terms = gam.AdditivePredictor("log_scale_terms", intercept=False)
    prior = lsl.Dist(tfd.Normal, loc=0.0, scale=1.0)
    loc_terms += gam.TermBuilder(registry, prefix_names_by="m.").lin(
        "x", prior=prior, name="m.x"
    )
    log_scale_terms += gam.TermBuilder(registry, prefix_names_by="s.").lin(
        "x", prior=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="s.x"
    )

    beta0 = LocationIntercept(response_value, loc_terms, log_scale_terms, name="beta0")
    loc = gam.AdditivePredictor("loc", intercept=beta0)
    loc += loc_terms
    gamma0 = LogScaleIntercept(response_value, loc, log_scale_terms, name="gamma0")
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

    # Keep the intercepts tied to training responses when predicting at new data.
    response = lsl.Var.new_obs(
        y,
        dist=PTMDist(
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


if __name__ == "__main__":
    model = build_model()
    print({name: model.vars[name].value for name in ("beta0", "gamma0")})
    print("Log probability:", model.log_prob)
