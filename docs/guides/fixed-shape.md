# Fit a fixed-shape PTM

A fixed-shape PTM is a variant of the current PTM formulation. It estimates one
shared standardized shape, while location and scale can vary with covariates.
It uses the same onion spline and affine standardization as a conditional-shape
PTM. The [legacy PTM](legacy.md) has its own transformation construction and
intercept handling.

## Use data with a shared shape

Start with the imports from the [conditional-shape example](conditional-shape.md),
then replace its data setup with 240 observations whose standardized shape stays
the same across `x`:

```python
rng = np.random.default_rng(42)
x = np.linspace(-1.0, 1.0, 240)
residual = (rng.gamma(shape=2.0, size=x.size) - 2.0) / np.sqrt(2.0)
y = 0.8 * np.sin(np.pi * x) + np.exp(0.3 * x) * residual
registry = gam.PandasRegistry(pd.DataFrame({"x": x, "y": y}))
```

The centered gamma residuals have a long right tail, mean zero, and variance one.
Only the mean and standard deviation change with `x`. This matches the fixed-shape
assumption; the conditional-shape guide's skewness-changing data would not.

## Keep only the shape intercept

Initialize the model with [PTM][liesel_ptm.PTM] as in the conditional-shape guide,
add its location and scale effects, then omit both
shape-term additions in [Define shape effects](conditional-shape.md#define-shape-effects):
`tb_shape.ps()` and `tb_shape.lin()`.

The constructor already creates the shape intercept and its smoothing prior.
Leaving `model.pred.shape` with only this intercept gives a fixed-shape PTM.
It uses the onion spline with both standardization corrections enabled, so
location and scale represent the conditional mean and standard deviation.

The [getting-started example](../index.md#build-a-model) shows a complete model
construction with this shared-shape structure.

## Fit and predict

Follow the same [starting-value optimization](conditional-shape.md#find-starting-values)
and [variational fit](conditional-shape.md#fit-a-variational-family). The shape
parameter block now contains only parameters belonging to the shape intercept
and its prior.

Prediction uses `model.predict()` as in the conditional-shape example. Supply
`x` in `newdata`; the `group` input is no longer needed after removing the
categorical shape term. The shared
shape has axes `(sample group, draw, shape coefficient)`, without a
new-observation axis. Location and scale retain their new-observation axis.

Skip the smooth-term and group-specific displays at the end of the
conditional-shape example. To display the shared shape, use
[plot_intercept_dist][liesel_ptm.plot_intercept_dist] with the shape intercept:

```python
ptm.plot_intercept_dist(model.response, model.pred.shape.intercept, samples)
```
