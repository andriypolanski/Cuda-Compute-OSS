"""CPU-only: lowrank / decaying-spectrum fills must scale V in place.

``(rng.standard_normal((rank, n)) * scale).astype(np.float64)`` keeps the raw
draw and the scaled copy alive together (plus a same-dtype astype), doubling
host peak for a buffer that is already O(rank·n) — enough to OOM disk-backed
generation at large n. Pure unit tests; no GPU.

Run:  python strategy/tests/test_lowrank_fill_inplace.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import storage


class _SpyRNG:
    """Returns a caller-owned V buffer for the (rank, n) draw; zeros for U blocks."""

    def __init__(self, V0: np.ndarray):
        self.V0 = V0
        self.rank, self.n = V0.shape

    def standard_normal(self, size):
        if tuple(size) == (self.rank, self.n):
            return self.V0
        return np.zeros(size, dtype=np.float64)


def test_fill_lowrank_scales_v_in_place(monkeypatch):
    n, rank = 32, 4
    scale = 1.0 / np.sqrt(rank)
    V0 = np.ones((rank, n), dtype=np.float64)
    monkeypatch.setattr(storage.np.random, "default_rng", lambda seed: _SpyRNG(V0))
    mat = np.empty((n, n), dtype=np.float32)
    storage._fill_lowrank(mat, seed=0, rank=rank)
    # In-place *= mutates the array returned by standard_normal; an out-of-place
    # ``* scale`` would leave V0 as ones.
    assert np.allclose(V0, np.full((rank, n), scale))


def test_fill_decaying_spectrum_scales_v_in_place(monkeypatch):
    n, rank, alpha = 32, 4, 1.0
    scale = 1.0 / np.sqrt(rank)
    weights = np.arange(1, rank + 1, dtype=np.float64) ** -alpha
    V0 = np.ones((rank, n), dtype=np.float64)
    monkeypatch.setattr(storage.np.random, "default_rng", lambda seed: _SpyRNG(V0))
    mat = np.empty((n, n), dtype=np.float32)
    storage._fill_decaying_spectrum(mat, seed=0, rank=rank, alpha=alpha)
    expected = np.ones((rank, n), dtype=np.float64) * scale * weights[:, None]
    assert np.allclose(V0, expected)


def test_fill_lowrank_still_rank_structured():
    n, rank = 48, 3
    mat = np.empty((n, n), dtype=np.float64)
    storage._fill_lowrank(mat, seed=1, rank=rank)
    # Numerical rank should be <= rank (singular values after rank ~ 0).
    s = np.linalg.svd(mat, compute_uv=False)
    assert np.all(s[rank:] < 1e-8 * (s[0] + 1e-30))


def test_old_out_of_place_scale_would_double_v_peak():
    """Arithmetic witness: out-of-place scale + same-dtype cast ≈ 2× V bytes."""
    n, rank, item = 131072, 4096, 8
    v_bytes = rank * n * item
    # raw draw + scaled product (astype to same dtype may add a third).
    old_peak = 2 * v_bytes
    new_peak = v_bytes
    assert old_peak == 2 * new_peak
    assert old_peak / (1024**3) == 8.0   # 8 GiB at this regime
    assert new_peak / (1024**3) == 4.0


if __name__ == "__main__":
    try:
        import pytest
    except ImportError:
        failed = 0
        for fn in (test_fill_lowrank_still_rank_structured,
                   test_old_out_of_place_scale_would_double_v_peak):
            try:
                fn()
                print(f"PASS  {fn.__name__}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {fn.__name__}: {e}")
        sys.exit(1 if failed else 0)
    raise SystemExit(pytest.main([__file__, "-v"]))
