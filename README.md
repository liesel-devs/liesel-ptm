# Penalized Transformation Models in Liesel

[![pre-commit](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pre-commit.yml)
[![pytest](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pytest.yml/badge.svg)](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pytest.yml)


Penalized Transformation Models are a class of flexible structured additive
distributional regression models. This is a Python library for estimating these models
with optimization, variational inference, and Markov Chain Monte Carlo (MCMC).
Liesel provides inference and model graphs; liesel-gam provides predictors, terms,
priors, and Gaussian IWLS proposals. This library provides PTM transformations,
distributions, computed intercepts, and distribution summaries and plots.

Learn more in the paper:

Brachem, J., Wiemann, P. F. V., & Kneib, T. (2026). [Bayesian Penalized Transformation Models: Structured Additive Location-Scale Regression for Arbitrary Conditional Distributions](https://doi.org/10.1080/10618600.2026.2703269). *Journal of Computational and Graphical Statistics*.

Application examples are included in the supplementary materials for the paper, which
can be found here: https://github.com/liesel-devs/ptm-supplement

## Installation

Requires Python 3.13. Until the required optimization/VI APIs are released, install
Liesel's `vi-modernization` branch alongside this library:

```bash
pip install "liesel @ git+https://github.com/liesel-devs/liesel.git@vi-modernization" \
  "git+https://github.com/liesel-devs/liesel-ptm.git@codex/cleanup"
```

For a checkout, `uv sync --locked --group dev` installs the tested environment.
Run `uv run --locked pytest`, `uv run --locked pre-commit run -a`, and
`uv run --locked sphinx-build -W --keep-going -b html docs/source docs/build/html`.


## Getting Started


Construct predictors and terms with [liesel-gam](https://github.com/liesel-devs/liesel-gam).
This minimal model has a smooth location effect, a constant scale, and a shared
transformation shape. The data are synthetic.

```python
import jax
import jax.numpy as jnp
import liesel.model as lsl
import liesel_gam as gam

import liesel_ptm as ptm

x = jnp.linspace(-2.0, 2.0, 100)
y = jnp.sin(x) + jax.random.normal(jax.random.key(42), x.shape)
builder = gam.TermBuilder(gam.DictRegistry({"x": x}))

loc = gam.AdditivePredictor("loc")
loc += builder.ps("x", k=10)
scale = gam.AdditivePredictor("scale", inv_link=jnp.exp)

knots = ptm.PTMKnots(-4.0, 4.0, nparam=6)
shape = gam.MVAdditivePredictor.from_random_walk(
    "delta",
    ndim=knots.nparam,
    intercept=False,
).constrain("sumzero_coef")
shape_builder = gam.MVTermBuilder.from_predictor(shape, term_builder=builder)
shape.intercept = shape_builder.intercept(scale=gam.scale_wb(0.1, scale=0.05))

response = lsl.Var.new_obs(
    y,
    dist=ptm.PTMDist(
        knots.knots,
        loc=loc,
        scale=scale,
        shape=shape,
        bspline="ptm",
        centered=True,
        scaled=True,
    ),
    name="y",
)
model = lsl.Model(response)
```

For fitting, prediction, and more model components, see the reference examples:

- [Modern GPTM](demo/reference_modern.py): location, scale, and shape predictors; optimization and VI.
- [Legacy PTM](demo/reference_legacy.py): shared shape and computed intercepts.

For a shared-shape model built with `LocationIntercept` and `LogScaleIntercept`,
`LegacyPTM` sets up Gaussian IWLS proposals and joint shape NUTS, recording both
intercepts at every posterior draw:

```python
legacy = ptm.LegacyPTM(model)
engine_builder = legacy.setup_mcmc(
    loc_terms=model.vars["loc_terms"],
    log_scale_terms=model.vars["log_scale_terms"],
    shape=model.vars["delta"],
    seed=43,
    num_chains=2,
)
```

Use standard Liesel scheduling and sampling, then `legacy.predict()` for prediction
with the recorded intercepts. The linked legacy example contains the complete model.
Its Weibull variance prior is unclipped.

See the [plotting notebook](demo/term-distribution-plots.ipynb) for distribution effects.
It illustrates plots with simulated coefficient draws.
