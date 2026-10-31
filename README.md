# Penalized Transformation Models in Liesel

[![pre-commit](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pre-commit.yml)
[![pytest](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pytest.yml/badge.svg)](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pytest.yml)


Penalized Transformation Models are a class of flexible structured additive
distributional regression models. This is a Python library for estimating these models
with optimization, variational inference, and Markov Chain Monte Carlo (MCMC).
Liesel provides inference and model graphs; liesel-gam provides predictors, terms,
priors, and Gaussian IWLS proposals. This library provides PTM transformations,
distributions, computed intercepts, and distribution summaries and plots.

The current PTM formulation uses an onion spline and affine standardization, with
two variants: **conditional-shape PTMs** let standardized shape vary with
covariates; **fixed-shape PTMs** estimate one shared shape. Both allow location and
scale effects and follow the formulation described in:

Brachem, J., & Kneib, T. (2026). [Reconciling Interpretability with Covariate-Dependent Shape Flexibility in Penalized Transformation Models for Distributional Regression](https://arxiv.org/abs/2609.02662). *arXiv preprint*.

The **legacy PTM** retains the PTM spline transformation and computed-intercept
handling from the first PTM paper:

Brachem, J., Wiemann, P. F. V., & Kneib, T. (2026). [Bayesian Penalized Transformation Models: Structured Additive Location-Scale Regression for Arbitrary Conditional Distributions](https://doi.org/10.1080/10618600.2026.2703269). *Journal of Computational and Graphical Statistics*.

Application examples are included in the supplementary materials for the first paper, which
can be found here: https://github.com/liesel-devs/ptm-supplement

## Installation

Requires Python 3.13. Until the required optimization/VI APIs are released, install
Liesel's `vi-modernization` branch alongside this library:

```bash
pip install "liesel @ git+https://github.com/liesel-devs/liesel.git@vi-modernization" \
  "git+https://github.com/liesel-devs/liesel-ptm.git@main"
```

For a checkout, `uv sync --locked --group dev` installs the tested environment.
Run `uv run --locked pytest` and `uv run --locked pre-commit run -a`.

## Documentation

Read the [documentation](https://liesel-devs.github.io/liesel-ptm/) for executable
guides and the API reference. To build it locally, install Quarto 1.8.27 and
Graphviz, then run:

```bash
uv sync --locked --group dev --group docs
uv run --locked --group docs python scripts/docs.py render
uv run --locked --group docs zensical build --clean --strict
```

Preview with `uv run --locked --group docs zensical serve`. After changing a guide,
library code, a shared guide fragment, or dependencies, force fresh execution with
`uv run --locked --group docs python scripts/docs.py render --fresh`. Commit the
Quarto sources, generated Markdown and figures, `_freeze` results, and output
manifest together. See the [contributor instructions](docs/contributing.md) for
setup, validation, and GitHub Pages deployment.


## Getting Started


Construct predictors and terms with [liesel-gam](https://github.com/liesel-devs/liesel-gam).
This fixed-shape PTM uses the current formulation, with a smooth location effect,
a constant scale, and a shared transformation shape. The data are synthetic.

```python
import jax
import jax.numpy as jnp
import liesel_gam as gam

import liesel_ptm as ptm

x = jnp.linspace(-2.0, 2.0, 100)
y = jnp.sin(x) + jax.random.normal(jax.random.key(42), x.shape)
builder = gam.TermBuilder(gam.DictRegistry({"x": x}))

model = ptm.PTM(y, nshape=6, bounds=(-4.0, 4.0))
model.pred.loc += builder.ps("x", k=10)
# model.pred.scale and model.pred.shape already have intercepts.
```

For fitting, prediction, and more model components, see the guides:

- [Conditional-shape PTM](https://liesel-devs.github.io/liesel-ptm/guides/conditional-shape/): location, scale, and shape predictors; optimization and VI.
- [Fixed-shape PTM](https://liesel-devs.github.io/liesel-ptm/guides/fixed-shape/): shared shape with the current transformation and standardization.
- [Legacy PTM](https://liesel-devs.github.io/liesel-ptm/guides/legacy/): the legacy transformation and computed intercepts.

`PTM` is a Liesel `Model`; add native liesel-gam terms through `.pred.loc`,
`.pred.scale`, and `.pred.shape`, then use ordinary Liesel inference. `nshape`
counts shape coefficients before their sum-to-zero constraint. `bounds` describes
the flexible interval on the standardized residual scale and defaults to `(-4, 4)`.
Manual `Model`/`PTMDist` construction remains available.
Both model classes expose `model.response` for distribution summaries and plots.

For the legacy shared-shape model, location and scale terms exclude the computed
intercepts. `LegacyPTM` sets up Gaussian IWLS proposals and joint shape NUTS,
recording both intercepts at every posterior draw:

```python
legacy = ptm.LegacyPTM(y, nshape=6)
legacy_terms = gam.TermBuilder.from_dict({"x": x})
legacy.pred.loc += legacy_terms.slin("x", scale=1.0, name="m.x")
legacy.pred.scale += legacy_terms.slin("x", scale=0.5, name="s.x")
engine_builder = legacy.setup_mcmc(seed=43, num_chains=2)
```

Legacy fitting requires full data; joint shuffling is supported, minibatches are
rejected. `legacy.vars["y"]` is the shared response input; `legacy.response` carries
the likelihood.

Use standard Liesel scheduling and sampling, then `legacy.predict()` for prediction
with the recorded intercepts. The linked legacy example contains the complete workflow,
including a smoothing update. Its Weibull variance prior is unclipped.

See the [plotting notebook](demo/term-distribution-plots.ipynb) for distribution effects.
It illustrates plots with simulated coefficient draws.
