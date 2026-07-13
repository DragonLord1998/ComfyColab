"""Deterministic random-generator contract for UltraShape preprocessing."""

from __future__ import annotations


MAX_SEED = 0xFFFFFFFFFFFFFFFF


def normalize_seed(seed: int) -> int:
    value = int(seed)
    if value < 0 or value > MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")
    return value


def make_numpy_rng(seed: int):
    import numpy as np

    return np.random.default_rng(normalize_seed(seed))
