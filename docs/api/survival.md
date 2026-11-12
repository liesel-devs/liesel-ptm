# Censoring and survival

The `liesel_ptm.survival_dist` module encodes censored observations as records
with columns `[time, lower, upper]` and supplies the censoring distributions that
evaluate them. Encode records with [censored][liesel_ptm.survival_dist.censored]
or a dedicated helper, check them with
[censoring_status][liesel_ptm.survival_dist.censoring_status], and fit them with
[PTM.from_censored][liesel_ptm.PTM.from_censored]. The distributions wrap any
scalar-event response distribution for manual model construction.
See [Fit a censored PTM](../guides/censored.md) for an executable fixed-shape
example with right censoring and predicted survival curves.

::: liesel_ptm.survival_dist.censored

::: liesel_ptm.survival_dist.censoring_status

::: liesel_ptm.survival_dist.uncensored

::: liesel_ptm.survival_dist.left_censored

::: liesel_ptm.survival_dist.right_censored

::: liesel_ptm.survival_dist.interval_censored

::: liesel_ptm.survival_dist.CensoredDistribution

::: liesel_ptm.survival_dist.LeftCensoredDistribution

::: liesel_ptm.survival_dist.RightCensoredDistribution

::: liesel_ptm.survival_dist.IntervalCensoredDistribution

::: liesel_ptm.survival_dist.setup_censored_vars

::: liesel_ptm.survival_dist.subset_var
