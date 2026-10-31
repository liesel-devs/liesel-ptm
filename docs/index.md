# Penalized Transformation Models in Liesel

Liesel-PTM fits structured additive distributional regression models. You can let
covariates change the location, scale, and shape of a response distribution, then
estimate the model with optimization, variational inference, or MCMC.
[Liesel](https://liesel-project.org/) provides inference and model graphs;
[liesel-gam](https://github.com/liesel-devs/liesel-gam) provides predictors, terms,
priors, and Gaussian IWLS proposals. Liesel-PTM provides transformations,
distributions, computed intercepts, and distribution summaries and plots.

## Install

Requires Python 3.13. Until Liesel releases the required optimization and VI APIs,
install its development branch alongside this library:

```bash
pip install "liesel @ git+https://github.com/liesel-devs/liesel.git@vi-modernization" \
  "git+https://github.com/liesel-devs/liesel-ptm.git@main"
```

For the tested checkout environment, follow the [contributor setup](contributing.md).

## Build a model

This small model has a smooth location effect, constant scale, and a shared
transformation shape. The data are synthetic. Predictors and terms come from
liesel-gam.

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

With both corrections enabled, location and scale are the conditional mean and
standard deviation. Shape effects that depend on covariates require the onion
spline. The [PTMDist reference][liesel_ptm.PTMDist] describes the available options.

## Fit, predict, and inspect results

- [Fit a modern GPTM](guides/modern.md): define covariate effects on location, scale,
  and shape; fit with staged optimization and VI; predict at new inputs.
- [Fit the original PTM](guides/original.md): use shared shape and computed
  intercepts with IWLS and NUTS sampling.
- [Inspect distribution effects](guides/term-distributions.md): choose summaries
  and plots for individual terms or full conditional distributions.
- [Look up the API](api/models.md): check arguments, defaults, and return values.

The [plotting notebook](https://github.com/liesel-devs/liesel-ptm/blob/main/demo/term-distribution-plots.ipynb)
also illustrates smooth, categorical, tensor, and spatial effects with simulated
coefficient draws.

## Paper and application examples

Brachem, J., Wiemann, P. F. V., & Kneib, T. (2026).
[Bayesian Penalized Transformation Models: Structured Additive Location-Scale Regression for Arbitrary Conditional Distributions](https://doi.org/10.1080/10618600.2026.2703269).
*Journal of Computational and Graphical Statistics*.

Application examples are included in the
[supplementary materials](https://github.com/liesel-devs/ptm-supplement).

## Acknowledgements and funding

Liesel-PTM is developed by Johannes Brachem with support from Paul Wiemann and
Thomas Kneib at the [University of Göttingen](https://www.uni-goettingen.de/en).
As a specialized extension, Liesel-PTM belongs to the Liesel project. We are grateful
to the [German Research Foundation (DFG)](https://www.dfg.de/en) for funding the
development through grant 443179956.

![University of Göttingen](https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/uni-goe.svg)

![Funded by DFG](https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/funded-by-dfg.svg)
