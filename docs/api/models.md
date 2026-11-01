# Models and transformations

Start with [PTM][liesel_ptm.PTM] for conditional-shape or fixed-shape PTMs using
the current formulation. Add native liesel-gam terms through
[PTMPredictors][liesel_ptm.PTMPredictors], available as `model.pred`.
[LegacyPTM][liesel_ptm.LegacyPTM] constructs the legacy transformation and computed
intercepts, with helpers for MCMC setup and prediction. Both classes are Liesel
models and expose `model.response` for distribution summaries and plots.

Use [PTMDist][liesel_ptm.PTMDist] to construct a likelihood manually. The
transformation distributions also support direct evaluation outside a model graph.

See the [conditional-shape](../guides/conditional-shape.md) and
[fixed-shape](../guides/fixed-shape.md) guides for the current formulation, or the
[legacy guide](../guides/legacy.md) for the earlier transformation, computed
intercepts, and MCMC workflow.

::: liesel_ptm.PTM

::: liesel_ptm.LegacyPTM

::: liesel_ptm.PTMPredictors

::: liesel_ptm.PTMDist

::: liesel_ptm.PTMKnots

::: liesel_ptm.OnionKnots

::: liesel_ptm.OnionSpline

::: liesel_ptm.onion_dist

::: liesel_ptm.TransformationDist

::: liesel_ptm.LocScaleTransformationDist

::: liesel_ptm.LocationIntercept

::: liesel_ptm.LogScaleIntercept
