"""Small conditional-shape PTM: construction, staged fitting, and prediction.

Run from the repository root with ``uv run python -m demo.reference_modern``.
The synthetic response changes from left-skewed to right-skewed across x.
"""

import jax
import liesel.model as lsl
import liesel.optim as opt
import liesel_gam as gam
import numpy as np
import optax
import pandas as pd

import liesel_ptm as ptm


def build_model() -> ptm.PTM:
    """Construct a synthetic model with smooth and categorical shape effects."""
    rng = np.random.default_rng(42)
    x = np.tile(np.linspace(-1.0, 1.0, 100), 3)
    group = pd.Categorical(np.repeat(["a", "b", "c"], 100))
    weight = 1.0 / (1.0 + np.exp(-3.0 * x - 0.6 * (group.codes - 1)))
    right = rng.exponential(size=x.size) - 1.0
    left = rng.exponential(size=x.size) - 1.0
    residual = (weight * right - (1.0 - weight) * left) / np.sqrt(
        weight**2 + (1.0 - weight) ** 2
    )
    y = 0.8 * np.sin(np.pi * x) + np.exp(0.3 * x) * residual
    data = pd.DataFrame({"x": x, "group": group, "y": y})
    registry = gam.PandasRegistry(data)
    model = ptm.PTM(registry.get_obs("y"), nshape=8)
    tb_loc = gam.TermBuilder(registry, prefix_names_by="m.")
    tb_scale = gam.TermBuilder(registry, prefix_names_by="s.")

    model.pred.loc += tb_loc.ps("x", k=8, scale=1.0)
    model.pred.scale += tb_scale.lin("x", name="s.x")
    shape = model.pred.shape
    tb_shape_base = gam.TermBuilder(registry, prefix_names_by="h.")
    tb_shape = gam.MVTermBuilder.from_predictor(shape, term_builder=tb_shape_base)
    shape += tb_shape.ps("x", k=5, scale=0.5, dimension_scale=0.7)
    shape += tb_shape.lin("group", dimension_scale=0.7, name="h.group")

    return model


def fit(model: lsl.Model, epochs: int = 3000, draws: int = 1000):
    """Initialize with MAP, then fit VI with early stopping up to ``epochs``."""
    loc_scale_parameters = list(model.parental_submodel(model.vars["loc"]).parameters)
    loc_scale_parameters += list(
        model.parental_submodel(model.vars["scale"]).parameters
    )
    shape_parameters = list(model.parental_submodel(model.vars["shape"]).parameters)

    for parameters in (loc_scale_parameters, shape_parameters):
        result = opt.LieselOptim(
            model,
            optimizers=[opt.Optimizer(parameters, optax.adam(0.01))],
            loss_monitor="train_full_data",
            stopper=opt.Stopper(epochs=600, patience=600),
            seed=43,
            show_progress=False,
        ).fit()
        model.state = model.update_state(result.position_min_monitor)
    point = {name: var.value for name, var in model.parameters.items()}

    loc_scale_block = opt.VDist(loc_scale_parameters, model).mvn_tril()
    shape_block = opt.VDist(shape_parameters, model).mvn_diag()
    family = opt.CompositeVDist(loc_scale_block, shape_block).build()
    loss = opt.NegElboLoss.from_vdist(family, nsamples=8)
    for parameters in (
        loc_scale_block.parameters,
        shape_block.parameters,
        family.parameters,
    ):
        result = opt.LieselVI(
            model,
            loss=loss,
            optimizers=[opt.Optimizer(parameters, optax.adam(0.003))],
            loss_monitor=opt.EmaTrainLossMonitor(effective_window=20),
            stopper=opt.Stopper(epochs=epochs, patience=100, min_epochs=300, rtol=1e-4),
            seed=44,
            show_progress=False,
        ).fit()
        family.q.state = family.q.update_state(result.position_final)
    posterior = loss.approximate_joint_posterior(result)
    samples = posterior.sample((1, draws), seed=jax.random.key(45))
    return point, samples


if __name__ == "__main__":
    model = build_model()
    point, samples = fit(model)
    shape = model.vars["shape"]
    smooth = shape.terms["h.ps(x)"]
    summary = ptm.summarise_1d_smooth_dist(
        model.response, smooth, samples, intercept=shape.intercept, ngrid=3, rgrid=9
    )
    print(summary.head())
