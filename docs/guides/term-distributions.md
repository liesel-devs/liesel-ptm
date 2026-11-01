# Inspect distribution effects


Use distribution plots to inspect how a transformation-predictor term changes
standardized shape, or how the complete response distribution changes across
covariate values. Each example below produces a plot.

The examples use small synthetic inputs and simulated coefficient draws; no model
is fitted. The ribbons illustrate variation across those draws, rather than
uncertainty learned from observations. For a fitted model, supply its posterior
sample dictionary in the same form. The
[conditional-shape PTM guide](conditional-shape.md) shows that fitting workflow.

Run this guide from a [repository checkout](../contributing.md#build-the-documentation).

## Prepare the examples

The distribution constructor uses the onion spline with centering and scaling
enabled. Term-induced distributions therefore have mean zero and standard
deviation one; their differences describe standardized shape.

``` python
import jax
import jax.numpy as jnp
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
import pandas as pd
import plotnine as p9
import tensorflow_probability.substrates.jax.bijectors as tfb
from IPython.display import Markdown

import liesel_ptm as ptm

dist = ptm.onion_dist(nparam=4, centered=True, scaled=True)
dimension_scale = jnp.eye(4)
keys = jax.random.split(jax.random.key(42), 3)
```

This helper creates 40 coefficient draws around a seeded synthetic center.
The first two array axes are `(sample group, draw)`, here `(1, 40)`; the remaining
axes match the term’s coefficient array. These axes also match the dictionaries
used by the plotting helpers after fitting.

``` python
def simulated_samples(term, key, scale=0.08, center=None):
    center_key, draw_key = jax.random.split(key)
    if center is None:
        center = 0.5 * jax.random.normal(center_key, term.coef.value.shape)
    sample_shape = (1, 40) + term.coef.value.shape
    values = center + scale * jax.random.normal(draw_key, sample_shape)
    return {term.coef.name: values}
```

## Plot a smooth shape effect

Create a small P-spline term and attach it to a Liesel model so it can be evaluated
at new covariate values. The distribution plots isolate this term: sibling terms,
location, scale, and the shape intercept are excluded unless requested explicitly.

``` python
smooth_data = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
tb_smooth = gam.MVTermBuilder.from_df(smooth_data, dimension_scale)
smooth = tb_smooth.ps("x", k=5, scale=1.0, dimension_scale=1.0)
smooth_model = lsl.Model(smooth)
smooth_samples = simulated_samples(smooth, keys[1])
```

[plot_1d_smooth_dist](../api/plots.md#liesel_ptm.plot_1d_smooth_dist) places density
curves on covariate baselines. Increasing values of `x` appear higher in the stack
and use the high end of the color scale.

``` python
ptm.plot_1d_smooth_dist(
    dist,
    smooth,
    smooth_samples,
    rgrid=100,
    ngrid=5,
    ridge_spacing=0.3,
)
```

<img src="term-distributions_files/figure-commonmark/cell-5-output-1.png" width="480" height="300" alt="Five standardized density ridges showing the smooth shape effect across x." />

Compare the curves’ skewness and concentration rather than their vertical position:
the offsets separate the ridges and are not density values. Gray dotted curves
show the reference distribution. Ribbons contain pointwise 90% intervals
over the supplied draws. With fitted posterior samples, these ribbons
would describe posterior variation.

See the [plot reference](../api/plots.md#liesel_ptm.plot_1d_smooth_dist) for grids and ridge layout options.

### Compare CDFs and transformations

Use `quantity="cdf"` to inspect cumulative probabilities on the standardized scale.
Here three covariate values keep the overlapping curves easy to compare.

``` python
ptm.plot_1d_smooth_dist(
    dist,
    smooth,
    smooth_samples,
    quantity="cdf",
    rgrid=100,
    ngrid=3,
    ci_quantiles=None,
)
```

<img src="term-distributions_files/figure-commonmark/cell-6-output-1.png" width="480" height="300" alt="CDFs at three x values compared with the reference CDF." />

CDF differences show where probability accumulates more quickly. The transformation
plot instead shows the monotone map from the standardized response to the normal
reference scale. Its gray dotted identity line represents the reference shape.

``` python
ptm.plot_1d_smooth_dist(
    dist,
    smooth,
    smooth_samples,
    quantity="transformation",
    rgrid=100,
    ngrid=3,
    ci_quantiles=None,
)
```

<img src="term-distributions_files/figure-commonmark/cell-7-output-1.png" width="480" height="300" alt="Transformation curves at three x values and the identity line." />

Use `quantity="transformation_raw"` for the spline transformation before its
centering and scaling corrections. These two transformations can differ even
when they describe the same standardized shape.

### Inspect numerical summaries

[summarise_1d_smooth_dist](../api/summaries.md#liesel_ptm.summarise_1d_smooth_dist)
returns the numerical values behind the display. Compare densities near a
standardized response of zero at five values of `x`.

``` python
summary = ptm.summarise_1d_smooth_dist(
    dist,
    smooth,
    smooth_samples,
    rgrid=jnp.linspace(-3.0, 3.0, 61),
    ngrid=5,
)
density_at_zero = summary.loc[
    (summary["quantity"] == "density") & (summary["r"].abs() < 0.1),
    ["x", "r", "mean", "q_0.05", "q_0.95", "sample_size"],
].round(3)
```

``` python
Markdown(density_at_zero.to_markdown(index=False))
```

|    x |    r |  mean | q_0.05 | q_0.95 | sample_size |
|-----:|-----:|------:|-------:|-------:|------------:|
|   -1 | -0.1 | 0.418 |  0.398 |  0.435 |          40 |
|   -1 |    0 | 0.404 |  0.386 |  0.423 |          40 |
| -0.5 | -0.1 | 0.429 |  0.421 |  0.438 |          40 |
| -0.5 |    0 | 0.431 |  0.424 |  0.438 |          40 |
|    0 | -0.1 | 0.434 |  0.429 |  0.439 |          40 |
|    0 |    0 | 0.436 |  0.432 |   0.44 |          40 |
|  0.5 | -0.1 | 0.364 |  0.356 |  0.374 |          40 |
|  0.5 |    0 | 0.379 |   0.37 |  0.389 |          40 |
|    1 | -0.1 | 0.297 |  0.255 |  0.346 |          40 |
|    1 |    0 | 0.352 |  0.301 |  0.414 |          40 |

`mean` averages density values over draws. It is not the density evaluated at an
average coefficient vector. The `quantity` column also contains CDF, transformation,
and raw-transformation rows; further columns include `sd`, `var`, `hdi_low`, and
`hdi_high`. `sample_size` counts the supplied draws.

## Plot the shape intercept

The shape intercept supplies a common standardized distribution. It is excluded
from the smooth-term plot above. Create an intercept example to inspect that
shared contribution by itself.

``` python
tb_intercept = gam.MVTermBuilder.from_df(
    pd.DataFrame({"x": [0.0, 1.0]}), dimension_scale
)
intercept = tb_intercept.intercept(scale=0.1)
intercept_model = lsl.Model(intercept)
intercept_samples = simulated_samples(intercept, keys[0], scale=0.12)
```

``` python
ptm.plot_intercept_dist(
    dist,
    intercept,
    intercept_samples,
    rgrid=100,
    show_n_samples=4,
    seed=4,
)
```

<img src="term-distributions_files/figure-commonmark/cell-11-output-1.png" width="480" height="300" alt="Intercept density with a quantile band and four simulated draws." />

The solid density summarizes the draws; transparent trajectories show four
reproducibly selected draws. Use `show_n_samples` and `seed` for this display.
Pass `hdi_prob` to add HDI lines, or `show_reference_dist=False` to omit the
reference distribution.

To combine a smooth term with an intercept, pass the concrete intercept variable
as `intercept=...`.

## Plot conditional response distributions

A term-induced display answers a different question from a complete conditional
response distribution. The next example defines location and scale predictors
with chosen coefficient values and a shared normal shape. It is a plotting
example, not a fit to the two placeholder observations.

``` python
model = ptm.PTM(jnp.zeros(2), nshape=4, shape_scale=0.1)
tb_conditional = gam.TermBuilder.from_dict(
    {"x": jnp.array([0.0, 1.0]), "z": jnp.array([0.0, 1.0])}
)
loc_effect = tb_conditional.lin("x", name="location_effect")
scale_effect = tb_conditional.lin("z", name="scale_effect")
model.pred.loc += loc_effect
model.pred.scale += scale_effect

samples = {
    name: jnp.zeros((1, 1) + jnp.shape(var.value))
    for name, var in model.parameters.items()
}
samples[loc_effect.coef.name] = jnp.ones((1, 1, 1))
samples[scale_effect.coef.name] = jnp.full((1, 1, 1), jnp.log(2.0))
```

[PTM](../api/models.md#liesel_ptm.PTM) supplies the standardized transformation
and intercepts. The sample dictionary has one chosen value for every parameter:
zero intercepts and shape coefficients, a location slope of one, and a log-scale
slope of `log(2)`. These values give a normal distribution with mean `x` and
standard deviation `2**z`; they are not fitted estimates.

Each `newdata` row specifies one condition. Enable `include_loc` and
`include_scale` to include both predictors, and supply `rgrid` in response units.
The default conditional plot isolates standardized shape by fixing location to
zero and scale to one.

``` python
cases = [
    {"x": 0.0, "z": 0.0},
    {"x": 0.5, "z": 0.5},
    {"x": 1.0, "z": 1.0},
]
```

``` python
ptm.plot_conditional_dist(
    model.response,
    samples,
    newdata=cases,
    condition_labels=["A", "B", "C"],
    rgrid=jnp.linspace(-7.0, 9.0, 160),
    include_loc=True,
    include_scale=True,
    ci_quantiles=None,
)
```

<img src="term-distributions_files/figure-commonmark/cell-14-output-1.png" width="480" height="300" alt="Three response densities with varying location and scale." />

The curves move and widen because location and scale change; their standardized
shape is the same. Conditional plots show no reference curve. This example has
only one draw, so ribbons are disabled.

Use [summarise_conditional_dist](../api/summaries.md#liesel_ptm.summarise_conditional_dist)
for a numerical summary and the input-row rules.

### Show a transformed response scale

If the modeled response is a logarithm, `tfb.Exp()` maps it back to positive
response units. Supply the grid on that reported scale.

``` python
ptm.plot_conditional_dist(
    model.response,
    samples,
    newdata=cases,
    condition_labels=["A", "B", "C"],
    rgrid=jnp.geomspace(0.01, 25.0, 240),
    include_loc=True,
    include_scale=True,
    ci_quantiles=None,
    response_bijector=tfb.Exp(),
) + p9.labs(x="Response on original scale")
```

<img src="term-distributions_files/figure-commonmark/cell-15-output-1.png" width="480" height="300" alt="Positive densities after the exponential response transformation." />

The transformed densities are skewed, and the chosen range shows only part of
their right tails. Density and CDF values follow the transformed distribution;
transformation curves compose the fitted transformation with the bijector’s
inverse. Plotnine labels can replace the default axis title, as shown here.

## Compare categorical effects

Create a categorical term with three observed groups and one unobserved level.
Each group has its own term-induced standardized distribution.

``` python
groups = pd.Categorical(["a", "b", "a", "c", "b"], categories=["a", "b", "c", "d"])
tb_cluster = gam.MVTermBuilder.from_df(pd.DataFrame({"group": groups}), dimension_scale)
cluster = tb_cluster.ri("group", scale=1.0, dimension_scale=1.0)
cluster_model = lsl.Model(cluster)
cluster_samples = simulated_samples(cluster, keys[2], scale=0.10)
```

``` python
ptm.plot_cluster_dist(
    dist,
    cluster,
    cluster_samples,
    rgrid=100,
    show_unobserved=True,
    ridge_spacing=0.3,
)
```

<img src="term-distributions_files/figure-commonmark/cell-17-output-1.png" width="480" height="300" alt="Group densities: observed groups solid and unobserved d dashed." />

Group `d` is in the category mapping but absent from the observations. Its dashed
line distinguishes it from observed groups. All coefficients here are simulated;
this does not demonstrate what an unobserved group’s posterior would look like.

For multidimensional and spatial displays, see the existing
[plot reference](../api/plots.md) and [plotting notebook](https://github.com/liesel-devs/liesel-ptm/blob/main/demo/term-distribution-plots.ipynb).
