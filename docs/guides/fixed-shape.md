# Fit a fixed-shape PTM

A fixed-shape PTM is a variant of the current PTM formulation. It estimates one
shared standardized shape, while location and scale can vary with covariates.
It uses the same onion spline and affine standardization as a conditional-shape
PTM. The [legacy PTM](legacy.md) has its own transformation construction and
intercept handling.

## Keep only the shape intercept

Start with the complete [conditional-shape example](conditional-shape.md). In
[Define shape and the response](conditional-shape.md#define-shape-and-the-response),
retain the shape intercept and omit both additions to `shape`: the smooth term
from `builder.ps()` and the categorical term from `builder.lin()`.

Keep [OnionKnots][liesel_ptm.OnionKnots], `bspline="onion"`, and
`centered=True, scaled=True`. These
choices standardize the transformation so the location and scale predictors
represent the conditional mean and standard deviation. The shape intercept
remains a vector of estimated coefficients with a smoothing prior.

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

Skip the smooth-term display at the end of the conditional-shape example, which
requires a covariate-dependent shape term. To display the shared shape, use
[plot_intercept_dist][liesel_ptm.plot_intercept_dist] with the shape intercept.
