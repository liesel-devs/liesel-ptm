from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotnine as p9
from jax.typing import ArrayLike

from .summary import (
    _make_dist,
    _normalise_sample_dims,
    _predict,
    summarise_1d_smooth_dist,
    summarise_cluster_dist,
    summarise_intercept_dist,
    summarise_nd_smooth_dist,
)
from .util.plots import plot_loss, plot_param_history

_QUANTITIES = {"density", "cdf", "transformation", "transformation_raw"}
_NO_PANEL_GRID = p9.theme(panel_grid=p9.element_blank())
_NO_Y_AXIS = p9.theme(
    axis_text_y=p9.element_blank(),
    axis_ticks_major_y=p9.element_blank(),
    axis_title_y=p9.element_blank(),
)


def _reference_data(dist, term, r: np.ndarray, quantity: str) -> pd.DataFrame:
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
        )
        + p9.labs(x="r", y=quantity)
        + _NO_PANEL_GRID
    )


def _trajectory_data(
    dist,
    coef: jnp.ndarray,
    r: np.ndarray,
    *,
    quantity: str,
    n: int,
    seed: int,
) -> pd.DataFrame:
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
    dist,
    coef: jnp.ndarray,
    r: np.ndarray,
    *,
    quantity: str,
    n: int,
    seed: int,
    covariates: Mapping[str, Any],
) -> pd.DataFrame:
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


def plot_intercept_dist(
    dist,
    term,
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


def plot_1d_smooth_dist(
    dist,
    term,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: Mapping[str, ArrayLike] | None = None,
    ngrid: int = 5,
    intercept=None,
    ridge_spacing: float | None = None,
    show_y_axis: bool = False,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot:
    """Plot standardized PTM distributions induced by a one-dimensional smooth."""
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
    groups = list(pd.unique(summary[covariate]))
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
        plot = p9.ggplot(summary, p9.aes("r", "mean", group=covariate))
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
            )
            + p9.labs(x="r", y=quantity)
            + _NO_PANEL_GRID
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

    plot = p9.ggplot(summary, p9.aes("r", "plot_mean", group=covariate))
    if ci_quantiles is not None:
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_low", ymax="plot_high"),
            alpha=0.25,
        )
    if hdi_prob is not None:
        summary["plot_hdi_low"] = summary["hdi_low"] + summary["baseline"]
        summary["plot_hdi_high"] = summary["hdi_high"] + summary["baseline"]
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_hdi_low", ymax="plot_hdi_high"),
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
            alpha=0.5,
        )
        + p9.geom_line()
        + p9.scale_y_continuous(
            breaks=baselines.tolist(), labels=[str(v) for v in groups]
        )
        + p9.labs(x="r", y=covariate)
        + _NO_PANEL_GRID
    )
    if not show_y_axis:
        plot += _NO_Y_AXIS
    return plot


def plot_2d_smooth_dist(
    dist,
    term,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: Mapping[str, ArrayLike] | None = None,
    ngrid: int = 5,
    newdata_meshgrid: bool = False,
    marginals: Sequence = (),
    intercept=None,
    facet_by: str | None = None,
    ridge_spacing: float | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot:
    """Plot standardized PTM distributions induced by a two-dimensional smooth."""
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
    trajectories = None
    if show_n_samples is not None and show_n_samples > 0:
        combinations = summary[inputs].drop_duplicates()
        grid = {name: combinations[name].to_numpy() for name in inputs}
        coef = _normalise_sample_dims(_predict(term, samples, grid), term.value.ndim)
        for marginal in marginals:
            marginal_grid = {
                name: value
                for name, value in grid.items()
                if name in marginal.input_obs
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
            n=show_n_samples,
            seed=seed,
            covariates=grid,
        )
        trajectories["trajectory"] = trajectories["sample"].astype(str)
        for name in inputs:
            trajectories["trajectory"] += ":" + trajectories[name].astype(str)

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
            )
            + p9.facet_wrap(f"~{facet_by}")
            + p9.labs(x="r", y=quantity)
            + _NO_PANEL_GRID
        )

    groups = list(pd.unique(summary[ridge_by]))
    if ridge_spacing is None:
        ridge_spacing = 1.15 * float(summary["mean"].max())
    baselines = np.arange(len(groups)) * ridge_spacing
    summary["baseline"] = summary[ridge_by].map(dict(zip(groups, baselines)))
    summary["plot_mean"] = summary["mean"] + summary["baseline"]
    plot = p9.ggplot(summary, p9.aes("r", "plot_mean", group=ridge_by))
    if ci_quantiles is not None:
        summary["plot_low"] = summary[f"q_{ci_quantiles[0]}"] + summary["baseline"]
        summary["plot_high"] = summary[f"q_{ci_quantiles[1]}"] + summary["baseline"]
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_low", ymax="plot_high"),
            alpha=0.25,
        )
    if hdi_prob is not None:
        summary["plot_hdi_low"] = summary["hdi_low"] + summary["baseline"]
        summary["plot_hdi_high"] = summary["hdi_high"] + summary["baseline"]
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_hdi_low", ymax="plot_hdi_high"),
            alpha=0.25,
        )
    if trajectories is not None:
        trajectories["plot_value"] = trajectories["value"] + trajectories[ridge_by].map(
            dict(zip(groups, baselines))
        )
        plot += p9.geom_line(
            p9.aes(y="plot_value", group="trajectory"),
            data=trajectories,
            alpha=0.25,
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
    reference["plot_reference"] = reference["reference"] + reference[ridge_by].map(
        dict(zip(groups, baselines))
    )
    return (
        plot
        + p9.geom_hline(yintercept=baselines, linetype="dotted", alpha=0.35)
        + p9.geom_line(
            p9.aes("r", "plot_reference", group=ridge_by),
            data=reference,
            inherit_aes=False,
            linetype="dotted",
            alpha=0.5,
        )
        + p9.geom_line()
        + p9.facet_wrap(f"~{facet_by}")
        + p9.scale_y_continuous(
            breaks=baselines.tolist(), labels=[str(v) for v in groups]
        )
        + p9.labs(x="r", y=ridge_by)
        + _NO_PANEL_GRID
    )


def plot_3d_smooth_dist(
    dist,
    term,
    samples: Mapping[str, ArrayLike],
    *,
    x: str,
    y: str,
    ridge_by: str,
    points: Mapping[str, ArrayLike],
    ridge_values: Sequence,
    rgrid: int | ArrayLike = 150,
    layout: str = "stack",
    marginals: Sequence = (),
    intercept=None,
    glyph_width: float | None = None,
    glyph_height: float | None = None,
    ridge_spacing: float | None = None,
    point_size: float = 2.5,
    point_shape: str | int = "x",
    point_color: str = "black",
    ci_quantiles: tuple[float, float] | None = None,
    hdi_prob: float | None = None,
) -> p9.ggplot:
    """Plot local density glyphs for a three-dimensional smooth."""
    inputs = list(term.input_obs)
    if len(inputs) != 3 or set((x, y, ridge_by)) != set(inputs):
        raise ValueError("x, y, and ridge_by must name the term's three inputs.")
    if layout not in {"stack", "facet"}:
        raise ValueError("layout must be 'stack' or 'facet'.")
    if set(points) != {x, y}:
        raise ValueError(f"points must contain exactly {x!r} and {y!r}.")
    xpoints = np.asarray(points[x])
    ypoints = np.asarray(points[y])
    ridge_values = list(ridge_values)
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
    summary[ridge_by] = pd.Categorical(
        summary[ridge_by], categories=ridge_values, ordered=True
    )

    xspan = float(np.ptp(xpoints)) or 1.0
    yspan = float(np.ptp(ypoints)) or 1.0
    glyph_width = 0.12 * xspan if glyph_width is None else glyph_width
    glyph_height = 0.08 * yspan if glyph_height is None else glyph_height
    ridge_spacing = 1.2 * glyph_height if ridge_spacing is None else ridge_spacing
    rspan = float(summary["r"].max() - summary["r"].min()) or 1.0
    rmid = 0.5 * float(summary["r"].max() + summary["r"].min())
    density_scale = float(summary["mean"].max()) or 1.0
    ridge_index = summary[ridge_by].cat.codes.to_numpy()
    displacement = ridge_index * ridge_spacing if layout == "stack" else 0.0
    summary["glyph_x"] = summary[x] + (summary["r"] - rmid) / rspan * glyph_width
    summary["baseline"] = summary[y] + displacement
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
    if layout == "facet":
        anchors = pd.concat(
            [anchors.assign(**{ridge_by: value}) for value in ridge_values],
            ignore_index=True,
        )
        anchors[ridge_by] = pd.Categorical(
            anchors[ridge_by], categories=ridge_values, ordered=True
        )
        plot += p9.facet_wrap(f"~{ridge_by}")
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
        + _NO_PANEL_GRID
    )


def plot_cluster_dist(
    dist,
    term,
    samples: Mapping[str, ArrayLike],
    *,
    quantity: str = "density",
    rgrid: int | ArrayLike = 150,
    newdata: Mapping[str, ArrayLike] | None = None,
    labels=None,
    intercept=None,
    show_unobserved: bool = True,
    ridge_spacing: float | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = None,
    seed: int = 1,
) -> p9.ggplot:
    """Plot standardized PTM distributions induced by a categorical term."""
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
    groups = list(pd.unique(summary[category]))
    trajectories = None
    if show_n_samples is not None and show_n_samples > 0:
        mapping = getattr(term.marginal_terms[0], "mapping", None)
        group_array = np.asarray(groups)
        if np.issubdtype(group_array.dtype, np.integer):
            codes = group_array
        elif mapping is not None:
            codes = np.asarray(mapping.labels_to_integers(group_array))
        else:
            label_codes = {label: code for code, label in enumerate(labels or groups)}
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
            )
            + p9.labs(x="r", y=quantity)
            + _NO_PANEL_GRID
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
        ),
    )
    if ci_quantiles is not None:
        summary["plot_low"] = summary[f"q_{ci_quantiles[0]}"] + summary["baseline"]
        summary["plot_high"] = summary[f"q_{ci_quantiles[1]}"] + summary["baseline"]
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_low", ymax="plot_high"),
            alpha=0.25,
        )
    if hdi_prob is not None:
        summary["plot_hdi_low"] = summary["hdi_low"] + summary["baseline"]
        summary["plot_hdi_high"] = summary["hdi_high"] + summary["baseline"]
        plot += p9.geom_ribbon(
            p9.aes(ymin="plot_hdi_low", ymax="plot_hdi_high"),
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
            alpha=0.5,
        )
        + p9.geom_line()
        + p9.scale_y_continuous(
            breaks=baselines.tolist(), labels=[str(v) for v in groups]
        )
        + p9.labs(x="r", y=category, linetype="Observed")
        + _NO_PANEL_GRID
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


def plot_regions_dist(
    dist,
    term,
    samples: Mapping[str, ArrayLike],
    *,
    polys: Mapping[Any, ArrayLike] | None = None,
    locations: Mapping[Any, tuple[float, float]] | None = None,
    rgrid: int | ArrayLike = 150,
    newdata: Mapping[str, ArrayLike] | None = None,
    labels=None,
    intercept=None,
    glyph_width: float | None = None,
    glyph_height: float | None = None,
    show_density_fill: bool = True,
    ci_quantiles: tuple[float, float] | None = None,
    hdi_prob: float | None = None,
) -> p9.ggplot:
    """Plot local density glyphs for a spatial categorical term."""
    if polys is None:
        polys = next(
            (
                candidate.polys
                for candidate in (term, *getattr(term, "marginal_terms", ()))
                if getattr(candidate, "polys", None) is not None
            ),
            None,
        )
    if polys is None:
        polys = next(
            (
                candidate.polygons
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
        + _NO_PANEL_GRID
    )


__all__ = [
    "plot_1d_smooth_dist",
    "plot_2d_smooth_dist",
    "plot_3d_smooth_dist",
    "plot_cluster_dist",
    "plot_regions_dist",
    "plot_intercept_dist",
    "plot_loss",
    "plot_param_history",
]
