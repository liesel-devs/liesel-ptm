from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
from jax.typing import ArrayLike

from .dist import LocScaleTransformationDist, TransformationDist


def _normalise_sample_dims(a: ArrayLike, value_ndim: int) -> jnp.ndarray:
    a = jnp.asarray(a)
    sample_ndim = a.ndim - value_ndim
    if sample_ndim == 0:
        return a[None, None, ...]
    if sample_ndim == 1:
        return a[None, ...]
    if sample_ndim == 2:
        return a
    raise ValueError("Expected zero, one, or two leading sample dimensions.")


def _make_dist(constructor, coef, *, raw: bool = False):
    if isinstance(constructor, lsl.Var):
        if constructor.dist_node is None:
            raise TypeError("The supplied variable has no distribution.")
        constructor = constructor.dist_node
    if isinstance(constructor, lsl.Dist):
        constructor = constructor.distribution

    kwargs = {"coef": coef}
    dist_class = getattr(constructor, "func", constructor)
    if not (
        isinstance(dist_class, type) and issubclass(dist_class, TransformationDist)
    ):
        raise TypeError("dist must construct a PTM transformation distribution.")
    if issubclass(dist_class, LocScaleTransformationDist):
        kwargs.update(loc=0.0, scale=1.0)
    if raw:
        kwargs.update(centered=False, scaled=False)
    return constructor(**kwargs)


def _summarise_array(
    values: ArrayLike,
    *,
    quantity: str,
    rgrid: jnp.ndarray,
    quantiles: Sequence[float],
    hdi_prob: float,
) -> pd.DataFrame:
    summary = (
        gs.SamplesSummary.from_array(
            jnp.asarray(values),
            name=quantity,
            quantiles=quantiles,
            hdi_prob=hdi_prob,
            which=["mean", "sd", "var", "hdi", "quantiles"],
        )
        .to_dataframe()
        .reset_index()
    )
    summary["r"] = np.tile(np.asarray(rgrid), len(summary) // len(rgrid))
    summary = summary.drop(columns=["variable", "var_fqn", "var_index"])
    summary.insert(0, "quantity", quantity)
    return summary


def _summarise_coef(
    dist,
    coef: jnp.ndarray,
    *,
    rgrid: int | ArrayLike,
    quantiles: Sequence[float],
    hdi_prob: float,
    covariates: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if isinstance(rgrid, int):
        if rgrid <= 0:
            raise ValueError("Integer rgrid must be positive.")
        r = jnp.linspace(-5.0, 5.0, rgrid)
    else:
        r = jnp.asarray(rgrid)
        if r.ndim != 1 or not len(r):
            raise ValueError("Array rgrid must be a nonempty one-dimensional vector.")
    r_batched = r.reshape((-1,) + (1,) * (coef.ndim - 1))
    fitted = _make_dist(dist, coef)
    raw = _make_dist(dist, coef, raw=True)
    quantities = {
        "density": fitted.prob(r_batched),
        "cdf": fitted.cdf(r_batched),
        "transformation": fitted.transformation_and_logdet(r_batched)[0],
        "transformation_raw": raw.transformation_and_logdet_spline(r_batched)[0],
    }
    summary = pd.concat(
        [
            _summarise_array(
                jnp.moveaxis(values, 0, -1),
                quantity=quantity,
                rgrid=r,
                quantiles=quantiles,
                hdi_prob=hdi_prob,
            )
            for quantity, values in quantities.items()
        ],
        ignore_index=True,
    )
    for name, values in (covariates or {}).items():
        summary[name] = np.tile(np.repeat(np.asarray(values), len(r)), len(quantities))
    return summary


def _predict(term, samples, newdata=None) -> jnp.ndarray:
    return term.predict(dict(samples), newdata=newdata)


def _category_mapping(term, input_name: str):
    for marginal in getattr(term, "marginal_terms", ()):
        mapping = getattr(marginal, "mapping", None)
        basis = getattr(marginal, "basis", None)
        if mapping is not None and getattr(basis, "input_name", None) == input_name:
            return mapping
        nested = _category_mapping(marginal, input_name)
        if nested is not None:
            return nested
    return None


def _predict_intercept(intercept, samples) -> jnp.ndarray | None:
    if intercept is None:
        return None
    if not isinstance(intercept, lsl.Var):
        raise TypeError("intercept must be a Liesel variable or None.")
    return _normalise_sample_dims(
        _predict(intercept, samples), value_ndim=intercept.value.ndim
    )


def summarise_intercept_dist(
    dist,
    term,
    samples: Mapping[str, ArrayLike],
    *,
    rgrid: int | ArrayLike = 150,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
) -> pd.DataFrame:
    """Summarise the standardized PTM distribution induced by an intercept term."""
    coef = _normalise_sample_dims(_predict(term, samples), term.value.ndim)
    return _summarise_coef(
        dist,
        coef,
        rgrid=rgrid,
        quantiles=quantiles,
        hdi_prob=hdi_prob,
    )


def summarise_1d_smooth_dist(
    dist,
    term,
    samples: Mapping[str, ArrayLike],
    *,
    rgrid: int | ArrayLike = 150,
    newdata: Mapping[str, ArrayLike] | None = None,
    ngrid: int = 5,
    intercept=None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
) -> pd.DataFrame:
    """Summarise the standardized PTM distribution induced by a 1D smooth."""
    inputs = term.input_obs
    if len(inputs) != 1:
        raise ValueError(f"Expected one input covariate, got {len(inputs)}.")
    name, observed = next(iter(inputs.items()))
    grid: dict[str, Any] = (
        {name: jnp.linspace(jnp.min(observed.value), jnp.max(observed.value), ngrid)}
        if newdata is None
        else {name: value for name, value in newdata.items()}
    )
    coef = _normalise_sample_dims(
        _predict(term, samples, grid), value_ndim=term.value.ndim
    )
    intercept_coef = _predict_intercept(intercept, samples)
    if intercept_coef is not None:
        coef = coef + intercept_coef[..., None, :]
    return _summarise_coef(
        dist,
        coef,
        rgrid=rgrid,
        quantiles=quantiles,
        hdi_prob=hdi_prob,
        covariates=grid,
    )


def summarise_nd_smooth_dist(
    dist,
    term,
    samples: Mapping[str, ArrayLike],
    *,
    rgrid: int | ArrayLike = 150,
    newdata: Mapping[str, ArrayLike] | None = None,
    ngrid: int = 5,
    newdata_meshgrid: bool = False,
    marginals: Sequence = (),
    intercept=None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
) -> pd.DataFrame:
    """Summarise a standardized PTM distribution induced by an nD smooth."""
    inputs = term.input_obs
    mappings = {name: _category_mapping(term, name) for name in inputs}
    grid: dict[str, Any]
    if newdata is None:
        axes = {
            name: (
                np.arange(len(mappings[name].labels_to_integers_map))
                if mappings[name] is not None
                else np.linspace(np.min(obs.value), np.max(obs.value), ngrid)
            )
            for name, obs in inputs.items()
        }
        arrays = np.meshgrid(*axes.values(), indexing="ij")
        grid = {name: value.ravel() for name, value in zip(axes, arrays)}
    elif newdata_meshgrid:
        axes = {}
        for name, values in newdata.items():
            values = np.asarray(values)
            if mappings[name] is not None and not np.issubdtype(
                values.dtype, np.integer
            ):
                values = np.asarray(mappings[name].labels_to_integers(values))
            axes[name] = values
        arrays = np.meshgrid(*axes.values(), indexing="ij")
        grid = {name: value.ravel() for name, value in zip(newdata, arrays)}
    else:
        grid = {}
        for name, values in newdata.items():
            values = np.asarray(values)
            if mappings[name] is not None and not np.issubdtype(
                values.dtype, np.integer
            ):
                values = np.asarray(mappings[name].labels_to_integers(values))
            grid[name] = values
        lengths = {len(np.asarray(value)) for value in grid.values()}
        if len(lengths) != 1:
            raise ValueError("All newdata arrays must have the same length.")

    display_grid = {
        name: (
            list(mappings[name].integers_to_labels(values))
            if mappings[name] is not None
            else values
        )
        for name, values in grid.items()
    }

    coef = _normalise_sample_dims(
        _predict(term, samples, grid), value_ndim=term.value.ndim
    )
    for marginal in marginals:
        marginal_grid = {
            name: value for name, value in grid.items() if name in marginal.input_obs
        }
        marginal_coef = _normalise_sample_dims(
            _predict(marginal, samples, marginal_grid),
            value_ndim=marginal.value.ndim,
        )
        coef = coef + marginal_coef
    intercept_coef = _predict_intercept(intercept, samples)
    if intercept_coef is not None:
        coef = coef + intercept_coef[..., None, :]

    summary = _summarise_coef(
        dist,
        coef,
        rgrid=rgrid,
        quantiles=quantiles,
        hdi_prob=hdi_prob,
        covariates=display_grid,
    )
    for name, mapping in mappings.items():
        if mapping is not None:
            summary[name] = pd.Categorical(
                summary[name], categories=list(mapping.labels_to_integers_map)
            )
    return summary


def summarise_cluster_dist(
    dist,
    term,
    samples: Mapping[str, ArrayLike],
    *,
    rgrid: int | ArrayLike = 150,
    newdata: Mapping[str, ArrayLike] | None = None,
    labels=None,
    intercept=None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
) -> pd.DataFrame:
    """Summarise a standardized PTM distribution induced by a categorical term."""
    inputs = term.input_obs
    if len(inputs) != 1:
        raise ValueError(f"Expected one categorical input, got {len(inputs)}.")
    name, observed_var = next(iter(inputs.items()))
    mapping = getattr(term.marginal_terms[0], "mapping", None)
    explicit_mapping = labels if hasattr(labels, "labels_to_integers_map") else None
    active_mapping = explicit_mapping or (mapping if labels is None else None)

    if active_mapping is not None:
        categories = list(active_mapping.labels_to_integers_map)
        codes = (
            np.arange(len(categories)) if newdata is None else np.asarray(newdata[name])
        )
        if not np.issubdtype(codes.dtype, np.integer):
            codes = np.asarray(active_mapping.labels_to_integers(codes))
        display = list(active_mapping.integers_to_labels(codes))
    else:
        categories = list(labels) if labels is not None else []
        values = (
            np.arange(len(categories)) if newdata is None else np.asarray(newdata[name])
        )
        if np.issubdtype(values.dtype, np.integer):
            codes = values
        else:
            label_codes = {label: code for code, label in enumerate(categories)}
            codes = np.asarray([label_codes[value] for value in values])
        display = [categories[code] for code in codes]

    observed_codes = set(np.asarray(observed_var.value).tolist())
    observed = np.asarray([code in observed_codes for code in codes])
    prediction_grid = {name: jnp.asarray(codes)}
    coef = _normalise_sample_dims(
        _predict(term, samples, prediction_grid), value_ndim=term.value.ndim
    )
    intercept_coef = _predict_intercept(intercept, samples)
    if intercept_coef is not None:
        coef = coef + intercept_coef[..., None, :]

    summary = _summarise_coef(
        dist,
        coef,
        rgrid=rgrid,
        quantiles=quantiles,
        hdi_prob=hdi_prob,
        covariates={name: display},
    )
    nr = len(summary) // (4 * len(codes))
    summary[name] = pd.Categorical(summary[name], categories=categories)
    summary["observed"] = np.tile(np.repeat(observed, nr), 4)
    return summary
