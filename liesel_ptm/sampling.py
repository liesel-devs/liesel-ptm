from collections.abc import Callable, Sequence
from pathlib import Path

import jax
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import optax
import pandas as pd

from .custom_types import Array, KeyArray
from .optim import OptimResult, Stopper, optim_flat


def cache_results(
    engine_builder: gs.EngineBuilder, filename: str | Path, use_cache: bool = True
) -> gs.engine.SamplingResults:
    if use_cache:
        fp = Path(filename)
        if fp.exists():
            return gs.engine.SamplingResults.pkl_load(fp)

    engine = engine_builder.build()
    engine.sample_all_epochs()
    results = engine.get_results()

    if use_cache:
        fp = Path(filename)
        results.pkl_save(fp)
    return results


def summarise_samples(
    samples: dict[str, Array],
    fn: Callable,
    suffix: str = "",
    **kwargs,
) -> dict[str, Array]:
    """Apply summary function to all elements of the given samples."""
    return {key + suffix: fn(val, **kwargs) for key, val in samples.items()}


def sample_quantiles(
    samples: dict[str, Array],
    q: float,
    axis: tuple = (0, 1),
    suffix: str = "",
    keepdims: bool = False,
) -> dict[str, Array]:
    """Calculate quantiles of the given samples."""
    return summarise_samples(
        samples, fn=np.quantile, axis=axis, q=q, suffix=suffix, keepdims=keepdims
    )


def sample_means(
    samples: dict[str, Array],
    axis: tuple = (0, 1),
    suffix: str = "",
    keepdims: bool = False,
) -> dict[str, Array]:
    """Calculate mean of the given samples."""
    return summarise_samples(
        samples, fn=np.mean, axis=axis, suffix=suffix, keepdims=keepdims
    )


def get_log_prob_fn(model: lsl.Model):
    """
    Returns the log-probability function with the position as the only argument.
    """

    model_state = model.state
    model = gs.LieselInterface(model)

    def log_prob_fn(position) -> float:
        new_model_state = model.update_state(position, model_state)
        return new_model_state["_model_log_prob"].value

    return log_prob_fn


def get_log_lik_fn(model: lsl.Model):
    """
    Returns the log-probability function with the position as the only argument.
    """

    model_state = model.state
    model = gs.LieselInterface(model)

    def log_lik_fn(position) -> float:
        new_model_state = model.update_state(position, model_state)
        return new_model_state["_model_log_lik"].value

    return log_lik_fn


def optimize_parameters(
    graph: lsl.Model,
    params: Sequence[str],
    optimizer: optax.GradientTransformation | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-6,
    maxiter: int = 10_000,
    batch_size: int | None = None,
) -> OptimResult:
    """
    Optimize the parameters of a goose model interface.

    This is a thin wrapper around :func:`.run_optimization`, kept only to keep
    the syntax backwards-compatible.

    See Also
    --------
    run_optimization : Use this function instead.
    """

    patience = min(maxiter, 100)

    stopper = Stopper(max_iter=maxiter, patience=patience, atol=atol, rtol=rtol)

    return optim_flat(
        model=graph,
        params=params,
        optimizer=optimizer,
        stopper=stopper,
        batch_size=batch_size,
    )


def summarise_by_quantiles(
    a: Array,
    names_prefix: str = "",
    axis: Sequence[int] | int = (0, 1),
    lo: float = 0.1,
    hi: float = 0.9,
) -> pd.DataFrame:
    mean = np.asarray(a.mean(axis=axis))
    sd = np.asarray(a.std(axis=axis))
    median = np.asarray(np.quantile(a, q=0.5, axis=axis))
    q_lo = np.asarray(np.quantile(a, q=lo, axis=axis))
    q_hi = np.asarray(np.quantile(a, q=hi, axis=axis))

    df = dict()
    sep = "_" if names_prefix else ""
    df[f"{names_prefix}{sep}mean"] = mean
    df[f"{names_prefix}{sep}sd"] = sd
    df[f"{names_prefix}{sep}median"] = median
    df[f"{names_prefix}{sep}lo"] = q_lo
    df[f"{names_prefix}{sep}hi"] = q_hi
    df = pd.DataFrame(df)

    return df


def summarise_by_samples(
    key: KeyArray, a: Array, name: str, n: int = 100
) -> pd.DataFrame:
    """
    - index: index of the flattened array
    - sample: sample number
    - obs: observation number (enumerates response values)
    - chain: chain number
    """

    _, iterations, _ = a.shape

    a = np.concatenate(a, axis=0)
    idx = jax.random.choice(key, a.shape[0], shape=(n,), replace=True)

    a_column = a[idx, :].ravel()
    sample_column = np.repeat(np.arange(n), a.shape[-1])
    index_column = np.repeat(idx, a.shape[-1])
    obs_column = np.tile(np.arange(a.shape[-1]), n)

    data = {name: a_column, "sample": sample_column}
    data["index"] = index_column
    data["obs"] = obs_column
    df = pd.DataFrame(data)

    df["chain"] = df["index"] // iterations

    return df


def kwargs_full(name: str, value: Array | None = None) -> dict[str, Array | None]:
    kwargs = dict()
    kwargs[f"{name}_loc_lin"] = value
    kwargs[f"{name}_loc_nonlin"] = value
    kwargs[f"{name}_scale_lin"] = value
    kwargs[f"{name}_scale_nonlin"] = value
    return kwargs


def kwargs_loc(name: str, value: Array | None = None) -> dict[str, Array | None]:
    kwargs = dict()
    kwargs[f"{name}_loc_lin"] = value
    kwargs[f"{name}_loc_nonlin"] = value
    return kwargs


def kwargs_scale(name: str, value: Array | None = None) -> dict[str, Array | None]:
    kwargs = dict()
    kwargs[f"{name}_scale_lin"] = value
    kwargs[f"{name}_scale_nonlin"] = value
    return kwargs


def kwargs_lin(name: str, value: Array | None = None) -> dict[str, Array | None]:
    kwargs = dict()
    kwargs[f"{name}_loc_lin"] = value
    kwargs[f"{name}_scale_lin"] = value
    return kwargs


def kwargs_loc_lin(name: str, value: Array | None = None) -> dict[str, Array | None]:
    kwargs = dict()
    kwargs[f"{name}_loc_lin"] = value
    return kwargs


def kwargs_scale_lin(name: str, value: Array | None = None) -> dict[str, Array | None]:
    kwargs = dict()
    kwargs[f"{name}_scale_lin"] = value
    return kwargs
