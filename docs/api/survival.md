# Censoring and survival

The `liesel_ptm.survival_dist` module encodes censored observations as records
with columns `[time, lower, upper]` and supplies the censoring distributions that
evaluate them. Encode records with [censored][liesel_ptm.survival_dist.censored]
or a dedicated helper, check them with
[censoring_status][liesel_ptm.survival_dist.censoring_status], and fit them with
[PTM.from_censored][liesel_ptm.PTM.from_censored]. To censor another Liesel
likelihood, wrap it in a [CensoredDist][liesel_ptm.survival_dist.CensoredDist];
both can cache the row types for fixed data, which is several times faster.
See [Fit a censored PTM](../guides/censored.md) for an executable fixed-shape
example with right censoring and predicted survival curves.

Censored records contribute `P(Y <= upper)` (left), `P(Y > lower)` (right) and
`P(lower < Y <= upper)` (interval). The inequalities only matter for discrete
distributions. Infinite interval endpoints give the corresponding one-sided
probability.

::: liesel_ptm.survival_dist.censored

::: liesel_ptm.survival_dist.censoring_status

::: liesel_ptm.survival_dist.CensoredDist

::: liesel_ptm.survival_dist.uncensored

::: liesel_ptm.survival_dist.left_censored

::: liesel_ptm.survival_dist.right_censored

::: liesel_ptm.survival_dist.interval_censored

::: liesel_ptm.survival_dist.CensoredDistribution
