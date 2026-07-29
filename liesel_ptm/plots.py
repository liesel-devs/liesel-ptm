from collections.abc import Mapping, Sequence
from typing import Any, cast, overload

import jax.numpy as jnp
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
import pandas as pd
import plotnine as p9
from jax import Array
from jax.typing import ArrayLike

from .summary import (
    ClusterNewData,
    DistInput,
    Intercept,
    Labels,
    MarginalTerm,
    NewData,
    _make_dist,
    _normalise_sample_dims,
    _predict,
    summarise_1d_smooth_dist,
    summarise_cluster_dist,
    summarise_intercept_dist,
    summarise_nd_smooth_dist,
)
from .util.plots import plot_loss, plot_param_history

_QUANTITY_LABELS = {
    "density": "Density",
    "cdf": "CDF",
    "transformation": "Transformation",
    "transformation_raw": "Raw transformation",
}
_QUANTITIES = set(_QUANTITY_LABELS)


def _no_panel_grid() -> p9.theme:
    return p9.theme_light() + p9.theme(panel_grid=p9.element_blank())


def _no_y_axis() -> p9.theme:
    return p9.theme(
        axis_text_y=p9.element_blank(),
        axis_ticks_major_y=p9.element_blank(),
        axis_title_y=p9.element_blank(),
    )


def _ordered_groups(values: pd.Series) -> list[Any]:
    if isinstance(values.dtype, pd.CategoricalDtype):
        present = set(values.dropna())
        return [value for value in values.cat.categories if value in present]
    try:
        return sorted(pd.unique(values))
    except TypeError:
        return list(pd.unique(values))


def _reference_data(
    dist: DistInput, term: lsl.Var, r: ArrayLike, quantity: str
) -> pd.DataFrame:
    r = np.asarray(r)
    fitted = _make_dist(dist, jnp.zeros(term.value.shape[-1]))
    if quantity == "density":
        value = fitted.reference_distribution.prob(r)
    elif quantity == "cdf":
        value = fitted.reference_distribution.cdf(r)
    else:
        value = r
    return pd.DataFrame({"r": r, "reference": np.asarray(value)})


def _plot_curves(
    summary: pd.DataFrame,
    *,
    reference: pd.DataFrame,
    quantity: str,
    ci_quantiles: tuple[float, float] | None,
    hdi_prob: float | None,
    trajectories: pd.DataFrame | None = None,
) -> p9.ggplot:
    plot = p9.ggplot(summary, p9.aes("r", "mean"))
    if ci_quantiles is not None:
        plot += p9.geom_ribbon(
            p9.aes(
                ymin=f"q_{ci_quantiles[0]}",
                ymax=f"q_{ci_quantiles[1]}",
            ),
            alpha=0.25,
        )
    if hdi_prob is not None:
        plot += p9.geom_line(p9.aes(y="hdi_low"), linetype="dashed")
        plot += p9.geom_line(p9.aes(y="hdi_high"), linetype="dashed")
    if trajectories is not None:
        plot += p9.geom_line(
            p9.aes(y="value", group="sample"),
            data=trajectories,
            alpha=0.25,
        )
    return (
        plot
        + p9.geom_line()
        + p9.geom_line(
            p9.aes("r", "reference"),
            data=reference,
            inherit_aes=False,
            linetype="dotted",
            color="gray",
        )
        + p9.labs(x="r", y=_QUANTITY_LABELS[quantity])
        + _no_panel_grid()
    )


def _plot_density_ridges(
    summary: pd.DataFrame,
    *,
    reference: pd.DataFrame,
    ridge_by: str,
    ridge_spacing: float | None,
    ci_quantiles: tuple[float, float] | None,
    hdi_prob: float | None,
    trajectories: pd.DataFrame | None = None,
) -> p9.ggplot:
    groups = _ordered_groups(summary[ridge_by])
    if ridge_spacing is None:
        ridge_spacing = 1.15 * float(summary["mean"].max())
    baselines = np.arange(len(groups)) * ridge_spacing
    baseline_for = dict(zip(groups, baselines))
    summary["baseline"] = summary[ridge_by].map(baseline_for)
    summary["plot_mean"] = summary["mean"] + summary["baseline"]

    plot = p9.ggplot(
        summary,
        p9.aes("r", "plot_mean", group=ridge_by, color=ridge_by),
    )
    if ci_quantiles is not None:
        summary["plot_low"] = summary[f"q_{ci_quantiles[0]}"] + summary["baseline"]
        summary["plot_high"] = summary[f"q_{ci_quantiles[1]}"] + summary["baseline"]
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_low", ymax="plot_high", fill=ridge_by),
            alpha=0.25,
        )
    if hdi_prob is not None:
        summary["plot_hdi_low"] = summary["hdi_low"] + summary["baseline"]
        summary["plot_hdi_high"] = summary["hdi_high"] + summary["baseline"]
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_hdi_low", ymax="plot_hdi_high", fill=ridge_by),
            alpha=0.25,
        )
    if trajectories is not None:
        trajectories["plot_value"] = trajectories["value"] + trajectories[ridge_by].map(
            baseline_for
        )
        plot += p9.geom_line(
            p9.aes(y="plot_value", group="trajectory"),
            data=trajectories,
            alpha=0.25,
        )
    reference["plot_reference"] = reference["reference"] + reference[ridge_by].map(
        baseline_for
    )
    return (
        plot
        + p9.geom_hline(yintercept=baselines, linetype="dotted", alpha=0.35)
        + p9.geom_line(
            p9.aes("r", "plot_reference", group=ridge_by),
            data=reference,
            inherit_aes=False,
            linetype="dotted",
            color="gray",
            alpha=0.5,
        )
        + p9.geom_line()
        + p9.scale_y_continuous(breaks=[])
        + p9.labs(x="r", y="Density", color=ridge_by, fill=ridge_by)
        + _no_panel_grid()
    )


def _trajectory_data(
    dist: DistInput,
    coef: Array,
    r: ArrayLike,
    *,
    quantity: str,
    n: int,
    seed: int,
) -> pd.DataFrame:
    r = np.asarray(r)
    r_batched = jnp.asarray(r).reshape((-1,) + (1,) * (coef.ndim - 1))
    fitted = _make_dist(dist, coef, raw=quantity == "transformation_raw")
    if quantity == "density":
        values = fitted.prob(r_batched)
    elif quantity == "cdf":
        values = fitted.cdf(r_batched)
    elif quantity == "transformation":
        values = fitted.transformation_and_logdet(r_batched)[0]
    else:
        values = fitted.transformation_and_logdet_spline(r_batched)[0]
    values = np.asarray(jnp.moveaxis(values, 0, -1)).reshape((-1, len(r)))
    n = min(n, len(values))
    selected = np.random.default_rng(seed).choice(len(values), n, replace=False)
    return pd.DataFrame(
        {
            "r": np.tile(r, n),
            "value": values[selected].ravel(),
            "sample": np.repeat(np.arange(n), len(r)),
        }
    )


def _trajectory_data_grid(
    dist: DistInput,
    coef: Array,
    r: ArrayLike,
    *,
    quantity: str,
    n: int,
    seed: int,
    covariates: Mapping[str, ArrayLike | Sequence[Any]],
) -> pd.DataFrame:
    r = np.asarray(r)
    r_batched = jnp.asarray(r).reshape((-1,) + (1,) * (coef.ndim - 1))
    fitted = _make_dist(dist, coef, raw=quantity == "transformation_raw")
    if quantity == "density":
        values = fitted.prob(r_batched)
    elif quantity == "cdf":
        values = fitted.cdf(r_batched)
    elif quantity == "transformation":
        values = fitted.transformation_and_logdet(r_batched)[0]
    else:
        values = fitted.transformation_and_logdet_spline(r_batched)[0]
    values = np.asarray(jnp.moveaxis(values, 0, -1))
    values = values.reshape((-1,) + values.shape[-2:])
    n = min(n, len(values))
    selected = np.random.default_rng(seed).choice(len(values), n, replace=False)
    ngrid = values.shape[-2]
    data = pd.DataFrame(
        {
            "r": np.tile(r, n * ngrid),
            "value": values[selected].ravel(),
            "sample": np.repeat(np.arange(n), ngrid * len(r)),
        }
    )
    for name, covariate in covariates.items():
        data[name] = np.tile(np.repeat(np.asarray(covariate), len(r)), n)
    return data


def _nd_trajectory_data(
    dist: DistInput,
    term: MarginalTerm,
    samples: Mapping[str, ArrayLike],
    summary: pd.DataFrame,
    *,
    inputs: Sequence[str],
    quantity: str,
    marginals: Sequence[MarginalTerm],
    intercept: Intercept,
    n: int | None,
    seed: int,
) -> pd.DataFrame | None:
    if n is None or n <= 0:
        return None
    combinations = summary[list(inputs)].drop_duplicates()
    grid = {name: combinations[name].to_numpy() for name in inputs}
    coef = _normalise_sample_dims(_predict(term, samples, grid), term.value.ndim)
    for marginal in marginals:
        marginal_grid = {
            name: value for name, value in grid.items() if name in marginal.input_obs
        }
        coef = coef + _normalise_sample_dims(
            _predict(marginal, samples, marginal_grid),
            marginal.value.ndim,
        )
    if intercept is not None:
        intercept_coef = _normalise_sample_dims(
            _predict(intercept, samples), intercept.value.ndim
        )
        coef = coef + intercept_coef[..., None, :]
    trajectories = _trajectory_data_grid(
        dist,
        coef,
        summary["r"].drop_duplicates().to_numpy(),
        quantity=quantity,
        n=n,
        seed=seed,
        covariates=grid,
    )
    trajectories["trajectory"] = trajectories["sample"].astype(str)
    for name in inputs:
        trajectories["trajectory"] += ":" + trajectories[name].astype(str)
    return trajectories


@overload
def plot_intercept_dist(
    dist: DistInput,
    term: gam.MultivariateIntercept,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot: ...


@overload
def plot_intercept_dist(  # type: ignore[overload-cannot-match]
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot: ...


def plot_intercept_dist(
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot:
    """Plot a standardized PTM distribution induced by an intercept term."""
    if quantity not in _QUANTITIES:
        raise ValueError(f"Unknown quantity {quantity!r}.")
    quantiles = (0.05, 0.5, 0.95) if ci_quantiles is None else ci_quantiles
    summary = summarise_intercept_dist(
        dist,
        term,
        samples,
        rgrid=rgrid,
        quantiles=quantiles,
        hdi_prob=0.9 if hdi_prob is None else hdi_prob,
    )
    summary = summary.loc[summary["quantity"] == quantity].copy()
    r = summary["r"].drop_duplicates().to_numpy()
    trajectories = None
    if show_n_samples is not None and show_n_samples > 0:
        coef = _normalise_sample_dims(_predict(term, samples), term.value.ndim)
        trajectories = _trajectory_data(
            dist,
            coef,
            r,
            quantity=quantity,
            n=show_n_samples,
            seed=seed,
        )
    return _plot_curves(
        summary,
        reference=_reference_data(dist, term, r, quantity),
        quantity=quantity,
        ci_quantiles=ci_quantiles,
        hdi_prob=hdi_prob,
        trajectories=trajectories,
    )


@overload
def plot_1d_smooth_dist(
    dist: DistInput,
    term: gam.MultivariateStrctTerm,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: NewData = None,
    ngrid: int = 5,
    intercept: Intercept = None,
    ridge_spacing: float | None = None,
    show_y_axis: bool = False,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot: ...


@overload
def plot_1d_smooth_dist(  # type: ignore[overload-cannot-match]
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: NewData = None,
    ngrid: int = 5,
    intercept: Intercept = None,
    ridge_spacing: float | None = None,
    show_y_axis: bool = False,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot: ...


def plot_1d_smooth_dist(
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: NewData = None,
    ngrid: int = 5,
    intercept: Intercept = None,
    ridge_spacing: float | None = None,
    show_y_axis: bool = False,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot:
    """Plot standardized PTM distributions induced by a one-dimensional smooth."""
    term = cast(MarginalTerm, term)
    if quantity not in _QUANTITIES:
        raise ValueError(f"Unknown quantity {quantity!r}.")
    quantiles = (0.05, 0.5, 0.95) if ci_quantiles is None else ci_quantiles
    summary = summarise_1d_smooth_dist(
        dist,
        term,
        samples,
        rgrid=rgrid,
        newdata=newdata,
        ngrid=ngrid,
        intercept=intercept,
        quantiles=quantiles,
        hdi_prob=0.9 if hdi_prob is None else hdi_prob,
    )
    summary = summary.loc[summary["quantity"] == quantity].copy()
    covariate = next(iter(term.input_obs))
    groups = _ordered_groups(summary[covariate])
    trajectories = None
    if show_n_samples is not None and show_n_samples > 0:
        grid = {covariate: np.asarray(groups)}
        coef = _normalise_sample_dims(_predict(term, samples, grid), term.value.ndim)
        if intercept is not None:
            intercept_coef = _normalise_sample_dims(
                _predict(intercept, samples), intercept.value.ndim
            )
            coef = coef + intercept_coef[..., None, :]
        trajectories = _trajectory_data_grid(
            dist,
            coef,
            summary["r"].drop_duplicates().to_numpy(),
            quantity=quantity,
            n=show_n_samples,
            seed=seed,
            covariates=grid,
        )
        trajectories["trajectory"] = (
            trajectories["sample"].astype(str)
            + ":"
            + trajectories[covariate].astype(str)
        )

    if quantity != "density":
        plot = p9.ggplot(
            summary,
            p9.aes("r", "mean", group=covariate, color=covariate),
        )
        if ci_quantiles is not None:
            plot += p9.geom_ribbon(
                p9.aes(
                    ymin=f"q_{ci_quantiles[0]}",
                    ymax=f"q_{ci_quantiles[1]}",
                    fill=covariate,
                ),
                alpha=0.25,
            )
        if hdi_prob is not None:
            plot += p9.geom_line(p9.aes(y="hdi_low"), linetype="dashed")
            plot += p9.geom_line(p9.aes(y="hdi_high"), linetype="dashed")
        if trajectories is not None:
            plot += p9.geom_line(
                p9.aes(y="value", group="trajectory"),
                data=trajectories,
                alpha=0.25,
            )
        reference = _reference_data(
            dist, term, summary["r"].drop_duplicates().to_numpy(), quantity
        )
        return (
            plot
            + p9.geom_line()
            + p9.geom_line(
                p9.aes("r", "reference"),
                data=reference,
                inherit_aes=False,
                linetype="dotted",
                color="gray",
            )
            + p9.labs(
                x="r",
                y=_QUANTITY_LABELS[quantity],
                color=covariate,
                fill=covariate,
            )
            + _no_panel_grid()
        )

    if ridge_spacing is None:
        ridge_spacing = 1.15 * float(summary["mean"].max())
    baselines = np.arange(len(groups)) * ridge_spacing
    baseline_for = dict(zip(groups, baselines))
    summary["baseline"] = summary[covariate].map(baseline_for)
    summary["plot_mean"] = summary["mean"] + summary["baseline"]
    summary["plot_low"] = (
        summary[f"q_{ci_quantiles[0]}"] + summary["baseline"]
        if ci_quantiles is not None
        else summary["mean"]
    )
    summary["plot_high"] = (
        summary[f"q_{ci_quantiles[1]}"] + summary["baseline"]
        if ci_quantiles is not None
        else summary["mean"]
    )

    plot = p9.ggplot(
        summary,
        p9.aes("r", "plot_mean", group=covariate, color=covariate),
    )
    if ci_quantiles is not None:
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_low", ymax="plot_high", fill=covariate),
            alpha=0.25,
        )
    if hdi_prob is not None:
        summary["plot_hdi_low"] = summary["hdi_low"] + summary["baseline"]
        summary["plot_hdi_high"] = summary["hdi_high"] + summary["baseline"]
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_hdi_low", ymax="plot_hdi_high", fill=covariate),
            alpha=0.25,
        )
    if trajectories is not None:
        trajectories["plot_value"] = trajectories["value"] + trajectories[
            covariate
        ].map(baseline_for)
        plot += p9.geom_line(
            p9.aes(y="plot_value", group="trajectory"),
            data=trajectories,
            alpha=0.25,
        )
    reference = pd.concat(
        [
            _reference_data(
                dist,
                term,
                summary["r"].drop_duplicates().to_numpy(),
                quantity,
            ).assign(
                **{
                    covariate: group,
                    "plot_reference": lambda data, baseline=baseline: (
                        data["reference"] + baseline
                    ),
                }
            )
            for group, baseline in zip(groups, baselines)
        ],
        ignore_index=True,
    )
    plot = (
        plot
        + p9.geom_hline(yintercept=baselines, linetype="dotted", alpha=0.35)
        + p9.geom_line(
            p9.aes("r", "plot_reference", group=covariate),
            data=reference,
            inherit_aes=False,
            linetype="dotted",
            color="gray",
            alpha=0.5,
        )
        + p9.geom_line()
        + p9.scale_y_continuous(
            breaks=baselines.tolist(), labels=[str(v) for v in groups]
        )
        + p9.labs(x="r", y=covariate, color=covariate, fill=covariate)
        + _no_panel_grid()
    )
    if not show_y_axis:
        plot += _no_y_axis()
    return plot


@overload
def plot_2d_smooth_dist(
    dist: DistInput,
    term: (
        gam.MultivariateStrctTerm
        | gam.MultivariateStrctInteractionTerm
        | gam.MultivariateTPTerm
    ),
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: NewData = None,
    ngrid: int = 5,
    newdata_meshgrid: bool = False,
    marginals: Sequence[MarginalTerm] = (),
    intercept: Intercept = None,
    facet_by: str | None = None,
    ridge_spacing: float | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot: ...


@overload
def plot_2d_smooth_dist(  # type: ignore[overload-cannot-match]
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: NewData = None,
    ngrid: int = 5,
    newdata_meshgrid: bool = False,
    marginals: Sequence[MarginalTerm] = (),
    intercept: Intercept = None,
    facet_by: str | None = None,
    ridge_spacing: float | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot: ...


def plot_2d_smooth_dist(
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: NewData = None,
    ngrid: int = 5,
    newdata_meshgrid: bool = False,
    marginals: Sequence[MarginalTerm] = (),
    intercept: Intercept = None,
    facet_by: str | None = None,
    ridge_spacing: float | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot:
    """Plot standardized PTM distributions induced by a two-dimensional smooth."""
    term = cast(MarginalTerm, term)
    if quantity not in _QUANTITIES:
        raise ValueError(f"Unknown quantity {quantity!r}.")
    inputs = list(term.input_obs)
    if len(inputs) != 2:
        raise ValueError(f"Expected two input covariates, got {len(inputs)}.")
    facet_by = inputs[1] if facet_by is None else facet_by
    if facet_by not in inputs:
        raise ValueError(f"facet_by must be one of {inputs}.")
    ridge_by = next(name for name in inputs if name != facet_by)
    quantiles = (0.05, 0.5, 0.95) if ci_quantiles is None else ci_quantiles
    summary = summarise_nd_smooth_dist(
        dist,
        term,
        samples,
        rgrid=rgrid,
        newdata=newdata,
        ngrid=ngrid,
        newdata_meshgrid=newdata_meshgrid,
        marginals=marginals,
        intercept=intercept,
        quantiles=quantiles,
        hdi_prob=0.9 if hdi_prob is None else hdi_prob,
    )
    summary = summary.loc[summary["quantity"] == quantity].copy()
    trajectories = _nd_trajectory_data(
        dist,
        term,
        samples,
        summary,
        inputs=inputs,
        quantity=quantity,
        marginals=marginals,
        intercept=intercept,
        n=show_n_samples,
        seed=seed,
    )

    if quantity != "density":
        plot = p9.ggplot(
            summary,
            p9.aes("r", "mean", group=ridge_by, color=ridge_by),
        )
        if ci_quantiles is not None:
            plot += p9.geom_ribbon(
                p9.aes(
                    ymin=f"q_{ci_quantiles[0]}",
                    ymax=f"q_{ci_quantiles[1]}",
                    fill=ridge_by,
                ),
                alpha=0.2,
            )
        if hdi_prob is not None:
            plot += p9.geom_line(p9.aes(y="hdi_low"), linetype="dashed")
            plot += p9.geom_line(p9.aes(y="hdi_high"), linetype="dashed")
        if trajectories is not None:
            plot += p9.geom_line(
                p9.aes(y="value", group="trajectory"),
                data=trajectories,
                alpha=0.25,
            )
        facet_values = summary[[facet_by]].drop_duplicates()
        reference = facet_values.merge(
            _reference_data(
                dist,
                term,
                summary["r"].drop_duplicates().to_numpy(),
                quantity,
            ),
            how="cross",
        )
        return (
            plot
            + p9.geom_line()
            + p9.geom_line(
                p9.aes("r", "reference"),
                data=reference,
                inherit_aes=False,
                linetype="dotted",
                color="gray",
            )
            + p9.facet_wrap(f"~{facet_by}")
            + p9.labs(
                x="r",
                y=_QUANTITY_LABELS[quantity],
                color=ridge_by,
                fill=ridge_by,
            )
            + _no_panel_grid()
        )

    reference = (
        summary[[facet_by, ridge_by]]
        .drop_duplicates()
        .merge(
            _reference_data(
                dist,
                term,
                summary["r"].drop_duplicates().to_numpy(),
                quantity,
            ),
            how="cross",
        )
    )
    return _plot_density_ridges(
        summary,
        reference=reference,
        ridge_by=ridge_by,
        ridge_spacing=ridge_spacing,
        ci_quantiles=ci_quantiles,
        hdi_prob=hdi_prob,
        trajectories=trajectories,
    ) + p9.facet_wrap(f"~{facet_by}", labeller="label_both")


@overload
def plot_3d_smooth_dist(
    dist: DistInput,
    term: (
        gam.MultivariateStrctTerm
        | gam.MultivariateStrctInteractionTerm
        | gam.MultivariateTPTerm
    ),
    samples: Mapping[str, ArrayLike],
    *,
    x: str,
    y: str,
    ridge_by: str,
    rgrid: int | ArrayLike = 150,
    newdata: NewData = None,
    ngrid: int = 5,
    newdata_meshgrid: bool = False,
    marginals: Sequence[MarginalTerm] = (),
    intercept: Intercept = None,
    ridge_spacing: float | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot: ...


@overload
def plot_3d_smooth_dist(  # type: ignore[overload-cannot-match]
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    x: str,
    y: str,
    ridge_by: str,
    rgrid: int | ArrayLike = 150,
    newdata: NewData = None,
    ngrid: int = 5,
    newdata_meshgrid: bool = False,
    marginals: Sequence[MarginalTerm] = (),
    intercept: Intercept = None,
    ridge_spacing: float | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot: ...


def plot_3d_smooth_dist(
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    x: str,
    y: str,
    ridge_by: str,
    rgrid: int | ArrayLike = 150,
    newdata: NewData = None,
    ngrid: int = 5,
    newdata_meshgrid: bool = False,
    marginals: Sequence[MarginalTerm] = (),
    intercept: Intercept = None,
    ridge_spacing: float | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot:
    """Plot faceted density ridges for a three-dimensional smooth."""
    term = cast(MarginalTerm, term)
    inputs = list(term.input_obs)
    if len(inputs) != 3 or set((x, y, ridge_by)) != set(inputs):
        raise ValueError("x, y, and ridge_by must name the term's three inputs.")

    quantiles = (0.05, 0.5, 0.95) if ci_quantiles is None else ci_quantiles
    summary = summarise_nd_smooth_dist(
        dist,
        term,
        samples,
        rgrid=rgrid,
        newdata=newdata,
        ngrid=ngrid,
        newdata_meshgrid=newdata_meshgrid,
        marginals=marginals,
        intercept=intercept,
        quantiles=quantiles,
        hdi_prob=0.9 if hdi_prob is None else hdi_prob,
    )
    summary = summary.loc[summary["quantity"] == "density"].copy()
    trajectories = _nd_trajectory_data(
        dist,
        term,
        samples,
        summary,
        inputs=inputs,
        quantity="density",
        marginals=marginals,
        intercept=intercept,
        n=show_n_samples,
        seed=seed,
    )
    reference = (
        summary[[x, y, ridge_by]]
        .drop_duplicates()
        .merge(
            _reference_data(
                dist,
                term,
                summary["r"].drop_duplicates().to_numpy(),
                "density",
            ),
            how="cross",
        )
    )
    return _plot_density_ridges(
        summary,
        reference=reference,
        ridge_by=ridge_by,
        ridge_spacing=ridge_spacing,
        ci_quantiles=ci_quantiles,
        hdi_prob=hdi_prob,
        trajectories=trajectories,
    ) + p9.facet_grid(rows=x, cols=y, labeller="label_both")


@overload
def plot_3d_smooth_dist_stacked(
    dist: DistInput,
    term: (
        gam.MultivariateStrctTerm
        | gam.MultivariateStrctInteractionTerm
        | gam.MultivariateTPTerm
    ),
    samples: Mapping[str, ArrayLike],
    *,
    x: str,
    y: str,
    ridge_by: str,
    points: Mapping[str, ArrayLike],
    ridge_values: Sequence[Any],
    rgrid: int | ArrayLike = 150,
    marginals: Sequence[MarginalTerm] = (),
    intercept: Intercept = None,
    glyph_width: float | None = None,
    glyph_height: float | None = None,
    ridge_spacing: float | None = None,
    point_size: float = 2.5,
    point_shape: str | int = "x",
    point_color: str = "black",
    ci_quantiles: tuple[float, float] | None = None,
    hdi_prob: float | None = None,
) -> p9.ggplot: ...


@overload
def plot_3d_smooth_dist_stacked(  # type: ignore[overload-cannot-match]
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    x: str,
    y: str,
    ridge_by: str,
    points: Mapping[str, ArrayLike],
    ridge_values: Sequence[Any],
    rgrid: int | ArrayLike = 150,
    marginals: Sequence[MarginalTerm] = (),
    intercept: Intercept = None,
    glyph_width: float | None = None,
    glyph_height: float | None = None,
    ridge_spacing: float | None = None,
    point_size: float = 2.5,
    point_shape: str | int = "x",
    point_color: str = "black",
    ci_quantiles: tuple[float, float] | None = None,
    hdi_prob: float | None = None,
) -> p9.ggplot: ...


def plot_3d_smooth_dist_stacked(
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    x: str,
    y: str,
    ridge_by: str,
    points: Mapping[str, ArrayLike],
    ridge_values: Sequence[Any],
    rgrid: int | ArrayLike = 150,
    marginals: Sequence[MarginalTerm] = (),
    intercept: Intercept = None,
    glyph_width: float | None = None,
    glyph_height: float | None = None,
    ridge_spacing: float | None = None,
    point_size: float = 2.5,
    point_shape: str | int = "x",
    point_color: str = "black",
    ci_quantiles: tuple[float, float] | None = None,
    hdi_prob: float | None = None,
) -> p9.ggplot:
    """Plot stacked local density glyphs for a three-dimensional smooth."""
    term = cast(MarginalTerm, term)
    inputs = list(term.input_obs)
    if len(inputs) != 3 or set((x, y, ridge_by)) != set(inputs):
        raise ValueError("x, y, and ridge_by must name the term's three inputs.")
    if set(points) != {x, y}:
        raise ValueError(f"points must contain exactly {x!r} and {y!r}.")
    xpoints = np.asarray(points[x])
    ypoints = np.asarray(points[y])
    ridge_values = _ordered_groups(pd.Series(ridge_values))
    if len(xpoints) != len(ypoints) or not len(xpoints):
        raise ValueError("Point coordinate arrays must have the same nonzero length.")
    if not ridge_values:
        raise ValueError("ridge_values must not be empty.")

    npoints = len(xpoints)
    newdata = {
        x: np.tile(xpoints, len(ridge_values)),
        y: np.tile(ypoints, len(ridge_values)),
        ridge_by: np.repeat(ridge_values, npoints),
    }
    quantiles = (0.05, 0.5, 0.95) if ci_quantiles is None else ci_quantiles
    summary = summarise_nd_smooth_dist(
        dist,
        term,
        samples,
        rgrid=rgrid,
        newdata=newdata,
        marginals=marginals,
        intercept=intercept,
        quantiles=quantiles,
        hdi_prob=0.9 if hdi_prob is None else hdi_prob,
    )
    summary = summary.loc[summary["quantity"] == "density"].copy()
    if isinstance(summary[ridge_by].dtype, pd.CategoricalDtype):
        summary[ridge_by] = summary[ridge_by].cat.set_categories(
            ridge_values, ordered=True
        )

    xspan = float(np.ptp(xpoints)) or 1.0
    yspan = float(np.ptp(ypoints)) or 1.0
    glyph_width = 0.12 * xspan if glyph_width is None else glyph_width
    glyph_height = 0.08 * yspan if glyph_height is None else glyph_height
    ridge_spacing = 1.2 * glyph_height if ridge_spacing is None else ridge_spacing
    rspan = float(summary["r"].max() - summary["r"].min()) or 1.0
    rmid = 0.5 * float(summary["r"].max() + summary["r"].min())
    density_scale = float(summary["mean"].max()) or 1.0
    ridge_index = np.asarray(
        summary[ridge_by].map(dict(zip(ridge_values, range(len(ridge_values))))),
        dtype=float,
    )
    summary["glyph_x"] = summary[x] + (summary["r"] - rmid) / rspan * glyph_width
    summary["baseline"] = summary[y] + ridge_index * ridge_spacing
    summary["glyph_y"] = (
        summary["baseline"] + summary["mean"] / density_scale * glyph_height
    )
    summary["glyph_id"] = summary.groupby(
        [x, y, ridge_by], observed=False, sort=False
    ).ngroup()

    plot = p9.ggplot(
        summary,
        p9.aes(
            "glyph_x",
            "glyph_y",
            group="glyph_id",
            color=ridge_by,
        ),
    )
    if ci_quantiles is not None:
        summary["glyph_low"] = (
            summary["baseline"]
            + summary[f"q_{ci_quantiles[0]}"] / density_scale * glyph_height
        )
        summary["glyph_high"] = (
            summary["baseline"]
            + summary[f"q_{ci_quantiles[1]}"] / density_scale * glyph_height
        )
        plot += p9.geom_ribbon(
            p9.aes(ymin="glyph_low", ymax="glyph_high", fill=ridge_by),
            alpha=0.2,
        )
    if hdi_prob is not None:
        summary["glyph_hdi_low"] = (
            summary["baseline"] + summary["hdi_low"] / density_scale * glyph_height
        )
        summary["glyph_hdi_high"] = (
            summary["baseline"] + summary["hdi_high"] / density_scale * glyph_height
        )
        plot += p9.geom_ribbon(
            p9.aes(ymin="glyph_hdi_low", ymax="glyph_hdi_high", fill=ridge_by),
            alpha=0.2,
        )
    plot += p9.geom_line()

    anchors = pd.DataFrame({x: xpoints, y: ypoints})
    return (
        plot
        + p9.geom_point(
            p9.aes(x=x, y=y),
            data=anchors,
            inherit_aes=False,
            size=point_size,
            shape=point_shape,
            color=point_color,
        )
        + p9.coord_equal()
        + p9.labs(x=x, y=y, color=ridge_by, fill=ridge_by)
        + _no_panel_grid()
    )


@overload
def plot_cluster_dist(
    dist: DistInput,
    term: gam.MultivariateStrctTerm,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: ClusterNewData = None,
    labels: Labels = None,
    intercept: Intercept = None,
    show_unobserved: bool = True,
    ridge_spacing: float | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot: ...


@overload
def plot_cluster_dist(  # type: ignore[overload-cannot-match]
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: ClusterNewData = None,
    labels: Labels = None,
    intercept: Intercept = None,
    show_unobserved: bool = True,
    ridge_spacing: float | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot: ...


def plot_cluster_dist(
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: ClusterNewData = None,
    labels: Labels = None,
    intercept: Intercept = None,
    show_unobserved: bool = True,
    ridge_spacing: float | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot:
    """Plot standardized PTM distributions induced by a categorical term."""
    term = cast(MarginalTerm, term)
    if quantity not in _QUANTITIES:
        raise ValueError(f"Unknown quantity {quantity!r}.")
    quantiles = (0.05, 0.5, 0.95) if ci_quantiles is None else ci_quantiles
    summary = summarise_cluster_dist(
        dist,
        term,
        samples,
        rgrid=rgrid,
        newdata=newdata,
        labels=labels,
        intercept=intercept,
        quantiles=quantiles,
        hdi_prob=0.9 if hdi_prob is None else hdi_prob,
    )
    summary = summary.loc[summary["quantity"] == quantity].copy()
    if not show_unobserved:
        summary = summary.loc[summary["observed"]].copy()
    category = next(iter(term.input_obs))
    groups = _ordered_groups(summary[category])
    trajectories = None
    if show_n_samples is not None and show_n_samples > 0:
        mapping = getattr(term.marginal_terms[0], "mapping", None)
        group_array = np.asarray(groups)
        if np.issubdtype(group_array.dtype, np.integer):
            codes = group_array
        elif mapping is not None:
            codes = np.asarray(mapping.labels_to_integers(group_array))
        else:
            label_values = (
                labels
                if labels is not None and not isinstance(labels, gam.CategoryMapping)
                else groups
            )
            label_codes = {label: code for code, label in enumerate(label_values)}
            codes = np.asarray([label_codes[label] for label in groups])
        coef = _normalise_sample_dims(
            _predict(term, samples, {category: codes}), term.value.ndim
        )
        if intercept is not None:
            intercept_coef = _normalise_sample_dims(
                _predict(intercept, samples), intercept.value.ndim
            )
            coef = coef + intercept_coef[..., None, :]
        trajectories = _trajectory_data_grid(
            dist,
            coef,
            summary["r"].drop_duplicates().to_numpy(),
            quantity=quantity,
            n=show_n_samples,
            seed=seed,
            covariates={category: groups},
        )
        trajectories["trajectory"] = (
            trajectories["sample"].astype(str)
            + ":"
            + trajectories[category].astype(str)
        )
        observed_for = (
            summary[[category, "observed"]]
            .drop_duplicates()
            .set_index(category)["observed"]
            .to_dict()
        )
        trajectories["observed"] = trajectories[category].map(observed_for)

    if quantity != "density":
        plot = p9.ggplot(
            summary,
            p9.aes(
                "r",
                "mean",
                group=category,
                linetype="observed",
                color=category,
            ),
        )
        if trajectories is not None:
            plot += p9.geom_line(
                p9.aes(y="value", group="trajectory"),
                data=trajectories,
                alpha=0.25,
            )
        if ci_quantiles is not None:
            plot += p9.geom_ribbon(
                p9.aes(
                    ymin=f"q_{ci_quantiles[0]}",
                    ymax=f"q_{ci_quantiles[1]}",
                    fill=category,
                ),
                alpha=0.25,
            )
        if hdi_prob is not None:
            plot += p9.geom_line(p9.aes(y="hdi_low"), linetype="dashed")
            plot += p9.geom_line(p9.aes(y="hdi_high"), linetype="dashed")
        reference = _reference_data(
            dist, term, summary["r"].drop_duplicates().to_numpy(), quantity
        )
        return (
            plot
            + p9.geom_line()
            + p9.geom_line(
                p9.aes("r", "reference"),
                data=reference,
                inherit_aes=False,
                linetype="dotted",
                color="gray",
            )
            + p9.labs(
                x="r",
                y=_QUANTITY_LABELS[quantity],
                color=category,
                fill=category,
            )
            + _no_panel_grid()
        )

    if ridge_spacing is None:
        ridge_spacing = 1.15 * float(summary["mean"].max())
    baselines = np.arange(len(groups)) * ridge_spacing
    summary["baseline"] = np.asarray(
        summary[category].map(dict(zip(groups, baselines))), dtype=float
    )
    summary["plot_mean"] = summary["mean"] + summary["baseline"]
    plot = p9.ggplot(
        summary,
        p9.aes(
            "r",
            "plot_mean",
            group=category,
            linetype="observed",
            color=category,
        ),
    )
    if ci_quantiles is not None:
        summary["plot_low"] = summary[f"q_{ci_quantiles[0]}"] + summary["baseline"]
        summary["plot_high"] = summary[f"q_{ci_quantiles[1]}"] + summary["baseline"]
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_low", ymax="plot_high", fill=category),
            alpha=0.25,
        )
    if hdi_prob is not None:
        summary["plot_hdi_low"] = summary["hdi_low"] + summary["baseline"]
        summary["plot_hdi_high"] = summary["hdi_high"] + summary["baseline"]
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_hdi_low", ymax="plot_hdi_high", fill=category),
            alpha=0.25,
        )
    if trajectories is not None:
        trajectories["plot_value"] = trajectories["value"] + trajectories[category].map(
            dict(zip(groups, baselines))
        )
        plot += p9.geom_line(
            p9.aes(y="plot_value", group="trajectory"),
            data=trajectories,
            alpha=0.25,
        )
    reference = pd.concat(
        [
            _reference_data(
                dist,
                term,
                summary["r"].drop_duplicates().to_numpy(),
                quantity,
            ).assign(
                **{
                    category: group,
                    "plot_reference": lambda data, baseline=baseline: (
                        data["reference"] + baseline
                    ),
                }
            )
            for group, baseline in zip(groups, baselines)
        ],
        ignore_index=True,
    )
    return (
        plot
        + p9.geom_hline(yintercept=baselines, linetype="dotted", alpha=0.35)
        + p9.geom_line(
            p9.aes("r", "plot_reference", group=category),
            data=reference,
            inherit_aes=False,
            linetype="dotted",
            color="gray",
            alpha=0.5,
        )
        + p9.geom_line()
        + p9.scale_y_continuous(
            breaks=baselines.tolist(), labels=[str(v) for v in groups]
        )
        + p9.labs(
            x="r",
            y=category,
            color=category,
            fill=category,
            linetype="Observed",
        )
        + _no_panel_grid()
    )


def _polygon_centroid(vertices: ArrayLike) -> tuple[float, float]:
    xy = np.asarray(vertices, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Each polygon must be an array with shape (n, 2), n >= 3.")
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 3:
        raise ValueError("Each polygon must be an array with shape (n, 2), n >= 3.")
    if not np.array_equal(xy[0], xy[-1]):
        xy = np.vstack((xy, xy[0]))
    cross = xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1]
    area2 = cross.sum()
    if np.isclose(area2, 0.0):
        return tuple(np.mean(xy[:-1], axis=0))
    centroid = np.array(
        [
            np.sum((xy[:-1, 0] + xy[1:, 0]) * cross),
            np.sum((xy[:-1, 1] + xy[1:, 1]) * cross),
        ]
    ) / (3.0 * area2)
    return float(centroid[0]), float(centroid[1])


@overload
def plot_regions_dist(
    dist: DistInput,
    term: gam.MultivariateStrctTerm,
    samples: Mapping[str, ArrayLike],
    *,
    polys: Mapping[Any, ArrayLike] | None = None,
    locations: Mapping[Any, tuple[float, float]] | None = None,
    rgrid: int | ArrayLike = 150,
    newdata: ClusterNewData = None,
    labels: Labels = None,
    intercept: Intercept = None,
    glyph_width: float | None = None,
    glyph_height: float | None = None,
    show_density_fill: bool = True,
    ci_quantiles: tuple[float, float] | None = None,
    hdi_prob: float | None = None,
) -> p9.ggplot: ...


@overload
def plot_regions_dist(  # type: ignore[overload-cannot-match]
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    polys: Mapping[Any, ArrayLike] | None = None,
    locations: Mapping[Any, tuple[float, float]] | None = None,
    rgrid: int | ArrayLike = 150,
    newdata: ClusterNewData = None,
    labels: Labels = None,
    intercept: Intercept = None,
    glyph_width: float | None = None,
    glyph_height: float | None = None,
    show_density_fill: bool = True,
    ci_quantiles: tuple[float, float] | None = None,
    hdi_prob: float | None = None,
) -> p9.ggplot: ...


def plot_regions_dist(
    dist: DistInput,
    term: lsl.Var,
    samples: Mapping[str, ArrayLike],
    *,
    polys: Mapping[Any, ArrayLike] | None = None,
    locations: Mapping[Any, tuple[float, float]] | None = None,
    rgrid: int | ArrayLike = 150,
    newdata: ClusterNewData = None,
    labels: Labels = None,
    intercept: Intercept = None,
    glyph_width: float | None = None,
    glyph_height: float | None = None,
    show_density_fill: bool = True,
    ci_quantiles: tuple[float, float] | None = None,
    hdi_prob: float | None = None,
) -> p9.ggplot:
    """Plot local density glyphs for a spatial categorical term."""
    term = cast(MarginalTerm, term)
    if polys is None:
        polys = next(
            (
                getattr(candidate, "polys")
                for candidate in (term, *getattr(term, "marginal_terms", ()))
                if getattr(candidate, "polys", None) is not None
            ),
            None,
        )
    if polys is None:
        polys = next(
            (
                getattr(candidate, "polygons")
                for candidate in (term, *getattr(term, "marginal_terms", ()))
                if getattr(candidate, "polygons", None) is not None
            ),
            None,
        )
    locations = {} if locations is None else dict(locations)
    centers = (
        {}
        if polys is None
        else {label: _polygon_centroid(xy) for label, xy in polys.items()}
    )
    centers.update(locations)

    quantiles = (0.05, 0.5, 0.95) if ci_quantiles is None else ci_quantiles
    summary = summarise_cluster_dist(
        dist,
        term,
        samples,
        rgrid=rgrid,
        newdata=newdata,
        labels=labels,
        intercept=intercept,
        quantiles=quantiles,
        hdi_prob=0.9 if hdi_prob is None else hdi_prob,
    )
    summary = summary.loc[summary["quantity"] == "density"].copy()
    category = next(iter(term.input_obs))
    shown = list(pd.unique(summary[category]))
    missing = [label for label in shown if label not in centers]
    if missing:
        raise ValueError(f"Missing polygons or locations for categories: {missing}.")
    summary["anchor_x"] = np.asarray(
        summary[category].map({label: centers[label][0] for label in shown}),
        dtype=float,
    )
    summary["anchor_y"] = np.asarray(
        summary[category].map({label: centers[label][1] for label in shown}),
        dtype=float,
    )
    summary["group"] = summary[category]

    center_array = np.asarray([centers[label] for label in shown])
    if polys is not None:
        all_vertices = np.concatenate([np.asarray(xy) for xy in polys.values()])
        all_vertices = all_vertices[np.isfinite(all_vertices).all(axis=1)]
        xspan = float(np.ptp(all_vertices[:, 0])) or 1.0
        yspan = float(np.ptp(all_vertices[:, 1])) or 1.0
    else:
        xspan = float(np.ptp(center_array[:, 0])) or 1.0
        yspan = float(np.ptp(center_array[:, 1])) or 1.0
    glyph_width = 0.12 * xspan if glyph_width is None else glyph_width
    glyph_height = 0.08 * yspan if glyph_height is None else glyph_height
    rspan = float(summary["r"].max() - summary["r"].min()) or 1.0
    rmid = 0.5 * float(summary["r"].max() + summary["r"].min())
    density_scale = float(summary["mean"].max()) or 1.0
    summary["glyph_x"] = (
        summary["anchor_x"] + (summary["r"] - rmid) / rspan * glyph_width
    )
    summary["glyph_y"] = (
        summary["anchor_y"] + summary["mean"] / density_scale * glyph_height
    )

    plot = p9.ggplot(
        summary,
        p9.aes("glyph_x", "glyph_y", group="group"),
    )
    if polys is not None:
        polygons = pd.concat(
            [
                pd.DataFrame(
                    np.asarray(xy), columns=pd.Index(["map_x", "map_y"])
                ).assign(region=str(label))
                for label, xy in polys.items()
            ],
            ignore_index=True,
        )
        plot += p9.geom_polygon(
            p9.aes("map_x", "map_y", group="region"),
            data=polygons,
            inherit_aes=False,
            fill=None,
            color="grey",
        )
    if show_density_fill:
        plot += p9.geom_ribbon(
            p9.aes(ymin="anchor_y", ymax="glyph_y"),
            alpha=0.3,
        )
    if ci_quantiles is not None:
        summary["glyph_low"] = (
            summary["anchor_y"]
            + summary[f"q_{ci_quantiles[0]}"] / density_scale * glyph_height
        )
        summary["glyph_high"] = (
            summary["anchor_y"]
            + summary[f"q_{ci_quantiles[1]}"] / density_scale * glyph_height
        )
        plot += p9.geom_ribbon(
            p9.aes(ymin="glyph_low", ymax="glyph_high"),
            alpha=0.2,
        )
    if hdi_prob is not None:
        summary["glyph_hdi_low"] = (
            summary["anchor_y"] + summary["hdi_low"] / density_scale * glyph_height
        )
        summary["glyph_hdi_high"] = (
            summary["anchor_y"] + summary["hdi_high"] / density_scale * glyph_height
        )
        plot += p9.geom_ribbon(
            p9.aes(ymin="glyph_hdi_low", ymax="glyph_hdi_high"),
            alpha=0.2,
        )
    return (
        plot
        + p9.geom_line()
        + p9.coord_equal()
        + p9.labs(x="x", y="y")
        + _no_panel_grid()
    )


__all__ = [
    "plot_1d_smooth_dist",
    "plot_2d_smooth_dist",
    "plot_3d_smooth_dist",
    "plot_3d_smooth_dist_stacked",
    "plot_cluster_dist",
    "plot_regions_dist",
    "plot_intercept_dist",
    "plot_loss",
    "plot_param_history",
]
