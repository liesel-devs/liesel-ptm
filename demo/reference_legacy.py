"""Legacy PTM with Gaussian IWLS, shape NUTS and recorded-intercept prediction.

Run with ``uv run python -m demo.reference_legacy``. The response has a nonlinear
mean, changing scale, and a shared right-skewed shape.
"""

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel_gam as gam
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm


def build_model() -> ptm.LegacyPTM:
    """Initialize the shared-shape PTM, then add location and log-scale terms."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-1.0, 1.0, 500)
    error_dist = tfd.TwoPieceNormal(loc=0.0, scale=1.0, skewness=2.0)
    errors = error_dist.sample(x.size, seed=jax.random.key(43))
    residual = np.asarray((errors - error_dist.mean()) / error_dist.stddev())
    y = 1.0 + 0.5 * x + 0.7 * np.sin(np.pi * x) + np.exp(0.3 * x) * residual
    registry = gam.PandasRegistry(pd.DataFrame({"x": x}))
    model = ptm.LegacyPTM(y, nshape=8)
    tb_loc = gam.TermBuilder(registry, prefix_names_by="m.")
    tb_scale = gam.TermBuilder(registry, prefix_names_by="s.")
    smooth_scale = gam.VarIGPrior(1.0, 0.001)

    model.pred.loc += tb_loc.slin("x", scale=1.0, name="m.x")
    # A centered smooth adds a conjugate smoothing-variance Gibbs update.
    model.pred.loc += tb_loc.ps("x", k=5, scale=smooth_scale, name="m.ps(x)")
    model.pred.scale += tb_scale.slin("x", scale=1.0, name="s.x")

    return model


def fit(
    model: ptm.LegacyPTM,
    adaptation: int = 500,
    posterior: int = 200,
    num_chains: int = 2,
) -> gs.SamplingResults:
    """Sample all parameters and record both computed intercepts at every draw."""
    builder = model.setup_mcmc(seed=43, num_chains=num_chains)
    builder.show_progress = False
    builder.add_adaptation(adaptation)
    builder.add_posterior(posterior)
    engine = builder.build()
    engine.sample_all_epochs()
    return engine.get_results()


if __name__ == "__main__":
    model = build_model()
    samples = fit(model).get_posterior_samples()
    prediction = model.predict(
        samples,
        predict=["loc", "scale", "shape"],
        newdata={"x": jnp.array([-0.5, 0.5])},
    )
    print(
        "Recorded intercept shapes:",
        {name: samples[name].shape for name in ("beta0", "gamma0")},
    )
    print(
        "Prediction shapes:", {name: value.shape for name, value in prediction.items()}
    )
