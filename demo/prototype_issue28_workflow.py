"""THROWAWAY issue #28 workflow; PTM's import is the only planned-API difference.

Run: uv run --locked python demo/prototype_issue28_workflow.py
"""

import jax
import liesel.optim as opt
import liesel_gam as gam
import numpy as np
import optax

import liesel_ptm as ptm
from demo.prototype_issue28 import PTM

random = np.random.default_rng(42)
x = np.linspace(-1.0, 1.0, 40)
y = 0.5 * x + np.exp(0.2 * x) * random.normal(size=x.size)
registry = gam.DictRegistry({"x": x})
loc_terms = gam.TermBuilder(registry, prefix_names_by="m.")
scale_terms = gam.TermBuilder(registry, prefix_names_by="s.")

model = PTM(y, knots=ptm.OnionKnots(-4.0, 4.0, nparam=6))
model.pred.loc += loc_terms.slin("x", scale=1.0)
model.pred.scale += scale_terms.slin("x", scale=0.5)
# model.pred.shape already contains a zero-initialized, constrained intercept.

# Fit a diagonal-Gaussian variational approximation with Liesel.
family = opt.VDist(list(model.parameters), model).mvn_diag().build()
loss = opt.NegElboLoss.from_vdist(family, nsamples=8)
result = opt.LieselVI(
    model,
    loss=loss,
    optimizers=[opt.Optimizer(family.parameters, optax.adam(0.01))],
    loss_monitor="train_full_data",
    stopper=opt.Stopper(epochs=500, patience=100),
    seed=43,
    show_progress=False,
).fit()
posterior = loss.approximate_joint_posterior(result)
samples = posterior.sample((1, 500), seed=jax.random.key(44))

# Evaluate parameter functions at three new covariate values.
x_new = np.array([-0.5, 0.0, 0.5])
prediction = model.predict(
    samples, predict=["loc", "scale", "delta"], newdata={"x": x_new}
)
loc_interval = np.quantile(prediction["loc"], [0.05, 0.5, 0.95], axis=(0, 1))
print("New x:", x_new)
print("Location 5%, median, 95% (rows):\n", loc_interval)
print("Mean scale:", np.mean(prediction["scale"], axis=(0, 1)))

assert set(samples) == set(model.parameters)
assert all(np.isfinite(value).all() for value in samples.values())
assert prediction["loc"].shape == prediction["scale"].shape == (1, 500, 3)
assert prediction["delta"].shape == (1, 500, 6)
assert all(np.isfinite(value).all() for value in prediction.values())
assert np.all(prediction["scale"] > 0.0)
