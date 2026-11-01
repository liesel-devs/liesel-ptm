"""Small, deterministic legacy model for inference and prediction contracts."""

import liesel_gam as gam
import numpy as np
import pandas as pd

import liesel_ptm as ptm


def build_legacy_model() -> ptm.LegacyPTM:
    # Keep this fixture independent of the documentation demo's data and defaults.
    x = np.linspace(-1.0, 1.0, 24)
    y = 1.0 + 0.5 * x + np.exp(0.15 * x) * np.random.default_rng(42).normal(size=x.size)
    model = ptm.LegacyPTM(y, nshape=6)
    registry = gam.PandasRegistry(pd.DataFrame({"x": x}))
    tb_loc = gam.TermBuilder(registry, prefix_names_by="m.")
    tb_scale = gam.TermBuilder(registry, prefix_names_by="s.")
    model.pred.loc += tb_loc.slin("x", scale=1.0, name="m.x")
    model.pred.loc += tb_loc.ps(
        "x", k=5, scale=gam.VarIGPrior(1.0, 0.001), name="m.ps(x)"
    )
    model.pred.scale += tb_scale.slin("x", scale=1.0, name="s.x")
    return model
