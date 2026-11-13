# Distribution summaries

These functions return tidy dataframes for transformation-predictor effects or
conditional response distributions. See [distribution effects](../guides/term-distributions.md)
for the scales, grids, and terms included in each summary.

For a model from [PTM.from_censored][liesel_ptm.PTM.from_censored], pass
`model.response` as usual; the functions use the uncensored response
distribution, not the censoring-record likelihood.

::: liesel_ptm.summarise_intercept_dist

::: liesel_ptm.summarise_1d_smooth_dist

::: liesel_ptm.summarise_nd_smooth_dist

::: liesel_ptm.summarise_cluster_dist

::: liesel_ptm.summarise_conditional_dist
