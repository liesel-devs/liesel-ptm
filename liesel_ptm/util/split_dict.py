from __future__ import annotations

import jax
import jax.numpy as jnp

PyTreeDict = dict[str, jax.Array]


def _check_shapes(d: PyTreeDict, axis: int = 1) -> tuple[int, int]:
    """Return (C, S) and sanity-check same S across keys."""
    keys = list(d.keys())
    if not keys:
        raise ValueError("Input dict is empty.")
    first = d[keys[0]]
    if first.ndim < axis + 1:
        raise ValueError(f"Arrays must have ndim >= {axis + 1}, got {first.ndim}.")
    S = first.shape[axis]
    C = first.shape[0]
    for k in keys[1:]:
        a = d[k]
        if a.shape[axis] != S:
            raise ValueError(f"Key {k!r} has S={a.shape[axis]} but expected {S}.")
        if a.shape[0] != C:
            raise ValueError(f"Key {k!r} has C={a.shape[0]} but expected {C}.")
    return C, S


def split_dict_exact(d: PyTreeDict, k: int, axis: int = 1) -> list[PyTreeDict]:
    """
    Split dict of arrays (C,S,...) into k equal blocks along `axis`.
    Requires S % k == 0. Returns list of k dicts with arrays (C, M, ...).
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    _, S = _check_shapes(d, axis=axis)
    if S % k != 0:
        raise ValueError(f"Exact split requires S % k == 0, got S={S}, k={k}.")
    M = S // k

    # reshape (C, S, ...) -> (C, k, M, ...) and then select each block i
    def reshape_one(a):
        # Move split axis to position 1 if needed (here default axis=1 already).
        if axis != 1:
            a = jnp.moveaxis(a, axis, 1)
        new_shape = (a.shape[0], k, M) + a.shape[2:]
        a = a.reshape(new_shape)
        return a  # (C, k, M, ...)

    reshaped = {key: reshape_one(val) for key, val in d.items()}

    blocks: list[PyTreeDict] = []
    for i in range(k):
        block = {key: val[:, i, ...] for key, val in reshaped.items()}  # (C, M, ...)
        if axis != 1:
            # move the M axis back to original `axis`
            block = {key: jnp.moveaxis(val, 1, axis) for key, val in block.items()}
        blocks.append(block)
    return blocks


def split_dict_rough(d: PyTreeDict, k: int, axis: int = 1) -> list[PyTreeDict]:
    """
    Split dict of arrays (C,S,...) into k blocks along `axis`, with the last
    block possibly smaller. Returns list length k; block i has size (C, Mi, ...),
    where Mi is either ceil(S/k) or smaller for the last.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    _, S = _check_shapes(d, axis=axis)
    M = (S + k - 1) // k  # ceil(S/k)

    def take_block(a, i: int):
        if axis != 1:
            a = jnp.moveaxis(a, axis, 1)
        start = i * M
        stop = jnp.minimum((i + 1) * M, S)
        # dynamic_slice wants static slice sizes; easiest is slice then trim in Python.
        # Here we slice with normal indexing (works fine outside jit).
        blk = a[:, start:stop, ...]
        if axis != 1:
            blk = jnp.moveaxis(blk, 1, axis)
        return blk

    return [{key: take_block(val, i) for key, val in d.items()} for i in range(k)]
