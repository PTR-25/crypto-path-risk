import numpy as np
import pandas as pd
import pytest

from conftest import T0
from pathrisk.windows import build_windows


def bars(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "open_time", T0 + 60_000 * np.arange(len(df), dtype=np.int64))
    return df


def test_window_extremes_cover_every_bar():
    b = bars([(100, 101, 99, 100), (100, 100, 90, 98), (98, 104, 97, 102), (102, 103, 101, 102)])
    w = build_windows(b, horizon_min=3, step_min=1)
    assert len(w) == 2
    first = w.iloc[0]
    assert first["r"] == pytest.approx(0.02)          # 102 / 100 - 1
    assert first["mae_long"] == pytest.approx(-0.10)  # the wick to 90 that the close never shows
    assert first["mae_short"] == pytest.approx(0.04)


def test_window_with_missing_minute_is_dropped():
    b = bars([(100, 100, 100, 100)] * 6).drop(index=2).reset_index(drop=True)
    w = build_windows(b, horizon_min=2, step_min=1)
    starts = (w["start"] - pd.Timestamp(T0, unit="ms", tz="UTC")) // pd.Timedelta(minutes=1)
    assert list(starts) == [0, 3, 4]  # windows touching minute 2 are gone


def test_windows_start_on_step_boundaries():
    b = bars([(100, 100, 100, 100)] * 300)
    w = build_windows(b, horizon_min=60, step_min=60)
    assert (w["start"].dt.minute == 0).all()
    assert len(w) == 5
