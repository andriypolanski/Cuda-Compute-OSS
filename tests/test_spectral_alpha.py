"""Tests for the decaying-spectrum exponent exposed by the strategy CLI."""
from __future__ import annotations

import numpy as np
import pytest

from strategy import cli, storage


def test_spectral_alpha_is_parsed_and_defaults_to_one():
    assert cli.build_parser().parse_args(["--spectral-alpha", "2.5"]).spectral_alpha == 2.5
    assert cli.build_parser().parse_args([]).spectral_alpha == 1.0


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), float("-inf"), -0.1])
def test_generate_rejects_invalid_spectral_alpha(alpha):
    with pytest.raises(ValueError, match="spectral_alpha must be a finite number >= 0"):
        storage.generate(8, np.float32, False, None, seed=0,
                         fill="decaying-spectrum", data_rank=4,
                         spectral_alpha=alpha)


def test_larger_alpha_steepens_the_spectrum():
    def leading_energy_fraction(mat, k):
        s = np.linalg.svd(np.asarray(mat, dtype=np.float64), compute_uv=False)
        return float((s[:k] ** 2).sum() / (s ** 2).sum())

    gentle = storage.generate(96, np.float64, False, None, seed=0,
                              fill="decaying-spectrum", data_rank=32,
                              spectral_alpha=0.5)
    steep = storage.generate(96, np.float64, False, None, seed=0,
                             fill="decaying-spectrum", data_rank=32,
                             spectral_alpha=3.0)
    assert leading_energy_fraction(steep, 4) > leading_energy_fraction(gentle, 4)


@pytest.mark.parametrize("compare", [False, True])
def test_cli_threads_spectral_alpha_to_runner(monkeypatch, compare):
    seen = {}

    def fake_run(n, cfg, **kwargs):
        seen.update(kwargs)
        return {}

    def fake_compare(n, cfg, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(cli.runner, "run", fake_run)
    monkeypatch.setattr(cli.runner, "compare", fake_compare)
    argv = ["--n", "8", "--spectral-alpha", "2.5"]
    if compare:
        argv.append("--compare")
    assert cli.main(argv) == 0
    assert seen["spectral_alpha"] == 2.5
