import numpy as np
import pandas as pd
import pytest

T0 = int(pd.Timestamp("2026-01-01", tz="UTC").value // 1_000_000)  # ms


def random_walk_bars(n_minutes: int, sigma_1m: float, seed: int, start_ms: int = T0) -> pd.DataFrame:
    """Driftless log random walk sampled each minute. High/low are the bar's endpoints,
    so the path is monitored once a minute (the discrete-monitoring case)."""
    rng = np.random.default_rng(seed)
    logp = np.log(100.0) + np.concatenate([[0.0], np.cumsum(rng.normal(0, sigma_1m, n_minutes))])
    p = np.exp(logp)
    o, c = p[:-1], p[1:]
    return pd.DataFrame(
        {
            "open_time": start_ms + 60_000 * np.arange(n_minutes, dtype=np.int64),
            "open": o,
            "high": np.maximum(o, c),
            "low": np.minimum(o, c),
            "close": c,
            "quote_volume": 1e6,
        }
    )


@pytest.fixture
def rw_bars():
    return random_walk_bars
