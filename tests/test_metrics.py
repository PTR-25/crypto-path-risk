import numpy as np
import pandas as pd
import pytest
from statistics import NormalDist

from pathrisk.metrics import (
    brownian_breach_ratio_discrete, brownian_quantile_multiple, breach_counts, daily_rates, summarise,
)
from pathrisk.windows import build_windows


def test_random_walk_reproduces_reflection_principle(rw_bars):
    """On a driftless random walk the pipeline must find P(path) / P(close) close to 2,
    less the known discrete-monitoring shortfall. This checks the whole metric chain
    against theory before any market data is involved."""
    sigma_1m, h = 0.001, 240
    bars = rw_bars(1440 * 1000, sigma_1m, seed=1)
    w = build_windows(bars, horizon_min=h, step_min=h)  # non-overlapping
    z = 1.645
    b = z * sigma_1m * np.sqrt(h)  # barrier in log space, so long and short are symmetric
    grid = pd.DataFrame({"tier": "t", "leverage": 1.0, "side": ["long", "short"],
                         "d": [1 - np.exp(-b), np.exp(b) - 1]})
    w = w.assign(date=w["start"].dt.normalize(), symbol="RW", tier="t", horizon=h)
    s = summarise(daily_rates(breach_counts(w, grid))).set_index("side")

    expected = brownian_breach_ratio_discrete(z, steps=h)
    assert 1.8 < expected < 2.0
    for side in ("long", "short"):
        assert s.loc[side, "p_close"] == pytest.approx(NormalDist().cdf(-z), rel=0.1)
        assert s.loc[side, "breach_ratio"] == pytest.approx(expected, rel=0.08)
        assert s.loc[side, "hidden_share"] == pytest.approx(1 - 1 / expected, abs=0.03)


def test_close_breach_implies_path_breach(rw_bars):
    w = build_windows(rw_bars(1440 * 20, 0.002, seed=2), 60, 15)
    w = w.assign(date=w["start"].dt.normalize(), symbol="X", tier="t", horizon=60)
    grid = pd.DataFrame({"tier": "t", "leverage": 1.0, "side": ["long", "short"], "d": [0.01, 0.01]})
    c = breach_counts(w, grid)
    assert (c["n_close"] <= c["n_path"]).all() and c["n_close"].sum() > 0


def test_equal_weight_per_symbol_within_a_date():
    counts = pd.DataFrame({
        "date": pd.Timestamp("2026-01-01", tz="UTC"), "symbol": ["A", "B"], "tier": "t", "horizon": 60,
        "leverage": 1.0, "side": "long", "n": [1000, 10], "n_path": [0, 10], "n_close": [0, 5],
    })
    d = daily_rates(counts)
    assert d["p_path"].item() == pytest.approx(0.5)  # not 10 / 1010: B counts as much as A


def test_brownian_quantile_multiple():
    assert brownian_quantile_multiple(0.99) == pytest.approx(2.5758 / 2.3263, rel=1e-3)
