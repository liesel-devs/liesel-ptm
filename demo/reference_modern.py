"""Small conditional-shape PTM: construction, staged fitting, and prediction.

Run from the repository root with ``uv run python -m demo.reference_modern``.
The short fit checks integration; it is not suitable for statistical inference.
"""

import jax
import jax.numpy as jnp
import liesel.model as lsl
import liesel.optim as opt
import liesel_gam as gam
import numpy as np
import optax
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm.model.model import PTMDist


def build_model() -> lsl.Model:
    """Construct a synthetic model with smooth and categorical shape effects."""
    x = np.linspace(-1.0, 1.0, 24)
    y = 0.5 * x + np.exp(0.15 * x) * np.random.default_rng(42).normal(size=x.size)
    data = pd.DataFrame(
        {"x": x, "group": pd.Categorical(np.tile(["a", "b", "c"], 8)), "y": y}
    )
    registry = gam.PandasRegistry(data)
    loc = gam.AdditivePredictor("loc")
    scale = gam.AdditivePredictor("scale", inv_link=jnp.exp)
    loc += gam.TermBuilder(registry, prefix_names_by="m.").ps("x", k=5, scale=0.5)
    scale += gam.TermBuilder(registry, prefix_names_by="s.").lin(
        "x", prior=lsl.Dist(tfd.Normal, loc=0.0, scale=0.5)
    )

    knots = ptm.OnionKnots(-4.0, 4.0, nparam=4)
    shape = gam.MVAdditivePredictor.from_random_walk(
        "delta", ndim=knots.nparam, intercept=False
    ).constrain("sumzero_coef")
    builder = gam.MVTermBuilder.from_predictor(
        shape,
        term_builder=gam.TermBuilder(registry, prefix_names_by="h."),
    )
    shape.intercept = builder.intercept(scale=gam.scale_wb(0.1, scale=0.05))
    shape += builder.ps("x", k=5, scale=0.5, dimension_scale=0.3)
    shape += builder.lin("group", dimension_scale=0.3, name="h.group")

    response = registry.get_obs("y")
    response.dist_node = PTMDist(
        knots.knots,
        loc=loc,
        scale=scale,
        shape=shape,
        bspline="onion",
        centered=True,
        scaled=True,
    )
    return lsl.Model(response)


def fit(model: lsl.Model, epochs: int = 2):
    """Pre-optimize location/scale then shape, and fit a blocked Gaussian family."""
    loc_scale_parameters = list(model.parental_submodel(model.vars["loc"]).parameters)
    loc_scale_parameters += list(
        model.parental_submodel(model.vars["scale"]).parameters
    )
    shape_parameters = list(model.parental_submodel(model.vars["delta"]).parameters)

    for parameters in (loc_scale_parameters, shape_parameters):
        result = opt.LieselOptim(
            model,
            optimizers=[opt.Optimizer(parameters, optax.adam(0.001))],
            loss_monitor="train_full_data",
            stopper=opt.Stopper(epochs=epochs, patience=epochs),
            seed=43,
            show_progress=False,
        ).fit()
        model.state = model.update_state(result.position_min_monitor)
    point = {name: var.value for name, var in model.parameters.items()}

    loc_scale_block = opt.VDist(loc_scale_parameters, model).mvn_tril()
    shape_block = opt.VDist(shape_parameters, model).mvn_diag()
    family = opt.CompositeVDist(loc_scale_block, shape_block).build()
    loss = opt.NegElboLoss.from_vdist(family, nsamples=2)
    for parameters in (
        loc_scale_block.parameters,
        shape_block.parameters,
        family.parameters,
    ):
        result = opt.LieselVI(
            model,
            loss=loss,
            optimizers=[opt.Optimizer(parameters, optax.adam(0.001))],
            loss_monitor="train_full_data",
            stopper=opt.Stopper(epochs=epochs, patience=epochs),
            seed=44,
            show_progress=False,
        ).fit()
        family.q.state = family.q.update_state(result.position_final)
    posterior = loss.approximate_joint_posterior(result)
    samples = posterior.sample((1, 3), seed=jax.random.key(45))
    return point, samples


if __name__ == "__main__":
    model = build_model()
    point, samples = fit(model)
    shape = model.vars["delta"]
    smooth = shape.terms["h.ps(x)"]
    summary = ptm.summarise_1d_smooth_dist(
        model.vars["y"], smooth, samples, intercept=shape.intercept, ngrid=3, rgrid=9
    )
    print(summary.head())
