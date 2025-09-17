from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import liesel.goose as gs
import numpy as np
import pandas as pd

Array = Any
KeyArray = Any


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


def grouped_summary(
    a: Array,
    names_prefix: str = "",
    axis: Sequence[int] = (0, 1),
    lo: float = 0.1,
    hi: float = 0.9,
    groups: dict[str, Array] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Summarise an array, assuming that the last axis is some grouping axis.
    """

    if a.ndim <= len(axis):
        raise ValueError("a.ndim <= len(axis)")

    summaries = {
        "mean": np.mean(a, axis=axis),
        "std": np.std(a, axis=axis),
        "median": np.median(a, axis=axis),
        "lo": np.quantile(a, lo, axis=axis),
        "hi": np.quantile(a, hi, axis=axis),
    }

    dfs = []

    sep = "_" if names_prefix else ""

    for name, value in summaries.items():
        df = pd.DataFrame(value).reset_index(names="i")
        df["summary"] = f"{names_prefix}{sep}{name}"
        for k, v in kwargs.items():
            df[k] = v

        dfs.append(df)

    df = (
        pd.concat(dfs)
        .melt(id_vars=["summary", "i"] + list(kwargs), var_name="group")
        .pivot(index=["i", "group"] + list(kwargs), columns="summary", values="value")
        .reset_index()
    )

    if groups is not None:
        for name, value in groups.items():
            df[name] = value[df["group"].values]

    return df


def subsample_array(x, indices):
    # Flatten (a, b, ...) to (a*b, ...)
    x_flat = x.reshape(-1, *x.shape[2:])
    # Subsample using provided indices
    return x_flat[indices]


def subsample_tree(key, tree, num_samples=1000):
    # Get one array to infer the total number of entries
    example_array = jax.tree_util.tree_leaves(tree)[0]
    total_entries = example_array.shape[0] * example_array.shape[1]

    # Create random indices for subsampling
    indices = jax.random.choice(key, total_entries, shape=(num_samples,), replace=False)

    # Apply subsampling to each array in the tree
    return jax.tree.map(
        lambda x: jnp.expand_dims(subsample_array(x, indices), axis=0), tree
    )
