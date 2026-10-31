# Models and transformations

Connect Liesel predictors to a response distribution with
[PTMDist][liesel_ptm.PTMDist]. The current conditional-shape and fixed-shape PTMs
use the onion spline and affine standardization. Use
[LegacyPTM][liesel_ptm.LegacyPTM] for the legacy computed-intercept workflow.
The transformation distributions also support direct evaluation outside a model graph.

See the [conditional-shape](../guides/conditional-shape.md) and
[fixed-shape](../guides/fixed-shape.md) guides for the current formulation, or the
[legacy guide](../guides/legacy.md) for the earlier transformation, computed
intercepts, and MCMC workflow.

::: liesel_ptm.PTMDist

::: liesel_ptm.PTMKnots

::: liesel_ptm.OnionKnots

::: liesel_ptm.OnionSpline

::: liesel_ptm.onion_dist

::: liesel_ptm.TransformationDist

::: liesel_ptm.LocScaleTransformationDist

::: liesel_ptm.LocationIntercept

::: liesel_ptm.LogScaleIntercept

::: liesel_ptm.LegacyPTM
