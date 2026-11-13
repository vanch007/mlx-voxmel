"""Wavelet-Factorized Haar decomposition and interleave for VoxMel in MLX."""
from __future__ import annotations

import mlx.core as mx


def even_odd(x: mx.array, axis: str) -> tuple[mx.array, mx.array]:
    if axis == "t":
        # x is (B, T, M)
        e = x[:, 0::2, :]
        o = x[:, 1::2, :]
        L = min(e.shape[1], o.shape[1])
        return e[:, :L, :], o[:, :L, :]
    elif axis == "f":
        # x is (B, T, M)
        e = x[..., 0::2]
        o = x[..., 1::2]
        L = min(e.shape[-1], o.shape[-1])
        return e[..., :L], o[..., :L]
    raise ValueError(f"axis must be 't' or 'f', got {axis!r}")


def split(x: mx.array, axis: str) -> tuple[mx.array, mx.array]:
    e, o = even_odd(x, axis)
    return (e + o) / 2.0, (e - o) / 2.0


def interleave(e: mx.array, o: mx.array, axis: str) -> mx.array:
    first, second = e + o, e - o
    if axis == "t":
        B, L, M = e.shape
        stacked = mx.stack([first, second], axis=2)
        return mx.reshape(stacked, (B, 2 * L, M))
    elif axis == "f":
        B, T, L = e.shape
        stacked = mx.stack([first, second], axis=3)
        return mx.reshape(stacked, (B, T, 2 * L))
    raise ValueError(f"axis must be 't' or 'f', got {axis!r}")


def pyramid(x: mx.array, split_axes: tuple[str, ...]) -> tuple[mx.array, list[tuple[str, mx.array, mx.array]]]:
    details_rev = []
    cur = x
    for axis in reversed(split_axes):
        e, o = split(cur, axis)
        details_rev.append((axis, e, o))
        cur = e
    return cur, list(reversed(details_rev))


def tier_dims(n_mels: int, split_axes: tuple[str, ...]) -> tuple[int, list[int]]:
    cur = n_mels
    dims_rev = []
    for axis in reversed(split_axes):
        if axis == "f":
            cur = cur // 2
        dims_rev.append(cur)
    return cur, list(reversed(dims_rev))
