"""Rolling holding windows built from 1-minute bars.

A 24/7 market has no natural close. The exchange's 00:00 UTC daily candle is an
arbitrary cut, so instead of daily candles we open a hypothetical position every
`step` minutes and hold it for `horizon` minutes. For each window we keep:

    r          = C / O - 1           the close-to-close return a daily risk model sees
    mae_long   = L / O - 1  (<= 0)   worst excursion against a long
    mae_short  = H / O - 1  (>= 0)   worst excursion against a short

where O is the first bar's open, C the last bar's close, and H / L the extreme
high / low over every bar in the window. A window with any missing minute is
dropped, never filled: a gap is exactly where an unseen wick could hide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MINUTE_MS = 60_000


def complete_minutes(bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex bars onto a gap-free minute grid; missing minutes become NaN rows."""
    t = bars["open_time"].to_numpy(np.int64)
    if len(t) == 0:
        return bars.iloc[0:0]
    if np.any(t % MINUTE_MS):
        raise ValueError("open_time is not aligned to whole minutes")
    grid = np.arange(t[0], t[-1] + MINUTE_MS, MINUTE_MS, dtype=np.int64)
    return (
        bars.set_index("open_time")[["open", "high", "low", "close"]]
        .reindex(grid)
        .rename_axis("open_time")
        .reset_index()
    )


def build_windows(bars: pd.DataFrame, horizon_min: int, step_min: int) -> pd.DataFrame:
    """One row per complete window starting on a `step_min` boundary (UTC).

    Returns columns: start (UTC timestamp), r, mae_long, mae_short.
    """
    if horizon_min < 1 or step_min < 1:
        raise ValueError("horizon and step must be positive")
    g = complete_minutes(bars)
    if len(g) < horizon_min:
        return pd.DataFrame(columns=["start", "r", "mae_long", "mae_short"])

    h = horizon_min
    # Each rolling value at row j covers rows j-h+1 .. j; shift back so row s is the window's start.
    low = g["low"].rolling(h, min_periods=h).min().shift(-(h - 1))
    high = g["high"].rolling(h, min_periods=h).max().shift(-(h - 1))
    complete = g["open"].notna().astype(np.int8).rolling(h).sum().shift(-(h - 1)) == h
    close = g["close"].shift(-(h - 1))
    o = g["open"]

    on_step = (g["open_time"] // MINUTE_MS) % step_min == 0
    keep = complete & on_step & (o > 0)

    out = pd.DataFrame(
        {
            "start": pd.to_datetime(g.loc[keep, "open_time"], unit="ms", utc=True),
            "r": (close[keep] / o[keep] - 1.0),
            "mae_long": (low[keep] / o[keep] - 1.0),
            "mae_short": (high[keep] / o[keep] - 1.0),
        }
    )
    return out.astype({"r": np.float32, "mae_long": np.float32, "mae_short": np.float32}).reset_index(
        drop=True
    )
