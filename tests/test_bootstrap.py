import numpy as np
import pandas as pd

from pathrisk.bootstrap import block_indices, bootstrap_ci
from pathrisk.metrics import summarise


def test_block_indices_are_contiguous_runs():
    idx = block_indices(100, 7, np.random.default_rng(0))
    assert len(idx) == 100
    steps = np.diff(idx[:7])
    assert ((steps == 1) | (steps == -99)).all()


def test_ci_brackets_point_estimate():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2026-01-01", periods=200, freq="D", tz="UTC")
    p_close = rng.uniform(0.02, 0.06, len(dates))
    daily = pd.DataFrame({"date": dates, "tier": "t", "horizon": 60, "leverage": 10.0, "side": "long",
                          "p_close": p_close, "p_path": p_close * rng.uniform(1.8, 2.6, len(dates)),
                          "n_symbols": 5, "n_windows": 120})
    s = summarise(daily).merge(bootstrap_ci(daily, n_boot=500), on=["tier", "horizon", "leverage", "side"])
    row = s.iloc[0]
    assert row["breach_ratio_lo"] < row["breach_ratio"] < row["breach_ratio_hi"]
    assert row["hidden_share_lo"] < row["hidden_share"] < row["hidden_share_hi"]
