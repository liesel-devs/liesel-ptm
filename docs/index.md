# Penalized Transformation Models in Liesel

Liesel-PTM fits structured additive distributional regression models. You can let
covariates change the location, scale, and shape of a response distribution, then
estimate the model with optimization, variational inference, or MCMC.
[Liesel](https://liesel-project.org/) provides inference and model graphs;
[liesel-gam](https://github.com/liesel-devs/liesel-gam) provides predictors, terms,
priors, and Gaussian IWLS proposals. Liesel-PTM provides transformations,
distributions, computed intercepts, and distribution summaries and plots.

## Model variants

The current formulation, described in the
[conditional-shape PTM paper](https://arxiv.org/abs/2609.02662), uses the onion
spline and affine standardization of the transformation. It has two variants:

- **Conditional-shape PTM:** the standardized distributional shape can vary with
  covariates, alongside location and scale.
- **Fixed-shape PTM:** one standardized shape is shared across covariate values,
  while location and scale can vary. The shared shape is estimated from the data.

Here, shape describes the distribution after location and scale have been
separated. Both variants use the same transformation construction and
standardization, so location and scale represent the conditional mean and standard
deviation.

The **legacy PTM** retains the PTM spline transformation and computed-intercept
handling from the [first PTM paper](https://doi.org/10.1080/10618600.2026.2703269).
It also has a shared shape. See the [legacy guide](guides/legacy.md) for that
formulation and its MCMC workflow.

## Install

Requires Python 3.13. Until Liesel releases the required optimization and VI APIs,
install its development branch alongside this library:

```bash
pip install "liesel @ git+https://github.com/liesel-devs/liesel.git@vi-modernization" \
  "git+https://github.com/liesel-devs/liesel-ptm.git@main"
```

For the tested checkout environment, follow the [contributor setup](contributing.md).

## Build a model

This fixed-shape PTM has a smooth location effect, constant scale, and a shared
transformation shape. It uses the current formulation. The data are synthetic;
predictors and terms come from liesel-gam.

```python
import jax
import jax.numpy as jnp
import liesel_gam as gam

import liesel_ptm as ptm

x = jnp.linspace(-2.0, 2.0, 100)
y = jnp.sin(x) + jax.random.normal(jax.random.key(42), x.shape)
tb_loc = gam.TermBuilder(gam.DictRegistry({"x": x}))

model = ptm.PTM(y, nshape=6, bounds=(-4.0, 4.0))
model.pred.loc += tb_loc.ps("x", k=10)
```

[PTM][liesel_ptm.PTM] creates the onion spline, standardization corrections,
and intercepts. Add native liesel-gam terms through `model.pred.loc`,
`model.pred.scale`, and `model.pred.shape` before creating inference objects.
Leaving shape with only its intercept gives this fixed-shape model; adding shape
terms gives a conditional-shape PTM.

`nshape` counts shape coefficients before the sum-to-zero constraint. `bounds`
is the flexible interval on the standardized residual scale. The constructor's
standardization makes location and scale the conditional mean and standard
deviation. Use `model.response` for distribution summaries and plots, and ordinary
Liesel methods for fitting and prediction.

For the earlier transformation and computed intercepts, use
[LegacyPTM][liesel_ptm.LegacyPTM] as shown in the [legacy guide](guides/legacy.md).
Both classes inherit from Liesel's `Model`. For manual likelihood construction,
see [PTMDist][liesel_ptm.PTMDist].

## Fit, predict, and inspect results

- [Fit a fixed-shape PTM](guides/fixed-shape.md): fit a shared shape, plot location
  and scale effects, and compare fitted distributions with their generating truth.
- [Fit a censored PTM](guides/censored.md): fit right-censored log event times
  and compare predicted survival curves with the generating truth.
- [Fit a conditional-shape PTM](guides/conditional-shape.md): extend the fixed-shape
  workflow with shape effects and staged, blocked VI.
- [Fit a legacy PTM](guides/legacy.md): use the legacy transformation and computed
  intercepts with IWLS and NUTS sampling.
- [Inspect distribution effects](guides/term-distributions.md): choose summaries
  and plots for individual terms or full conditional distributions.
- [Look up the API](api/models.md): check arguments, defaults, and return values.

## Papers and application examples

The conditional-shape extension is described in:

Brachem, J., & Kneib, T. (2026).
[Reconciling Interpretability with Covariate-Dependent Shape Flexibility in Penalized Transformation Models for Distributional Regression](https://arxiv.org/abs/2609.02662).
*arXiv preprint*.

The legacy transformation and intercept construction are described in the first
PTM paper:

Brachem, J., Wiemann, P. F. V., & Kneib, T. (2026).
[Bayesian Penalized Transformation Models: Structured Additive Location-Scale Regression for Arbitrary Conditional Distributions](https://doi.org/10.1080/10618600.2026.2703269).
*Journal of Computational and Graphical Statistics*.

Application examples for the first paper are included in its
[supplementary materials](https://github.com/liesel-devs/ptm-supplement).

## Acknowledgements and funding

Liesel-PTM is developed by Johannes Brachem with support from Paul Wiemann and
Thomas Kneib at the [University of Göttingen](https://www.uni-goettingen.de/en).
As a specialized extension, Liesel-PTM belongs to the Liesel project. We are grateful
to the [German Research Foundation (DFG)](https://www.dfg.de/en) for funding the
development through grant 443179956.

![University of Göttingen](https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/uni-goe.svg)

![Funded by DFG](https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/funded-by-dfg.svg)
