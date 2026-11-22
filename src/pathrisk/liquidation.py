"""Liquidation thresholds: how far the mark price can move before a position is liquidated.

Model (isolated margin, one position, fees and funding ignored):

    entry notional   N = q * P0
    initial margin   IM = N / L
    price move       x = P / P0 - 1        (fraction, signed)
    PnL              long:  +N * x         short:  -N * x      (linear in x: a perpetual is delta-one)
    equity           E(x) = IM + PnL(x)
    maintenance      MM(x) = m * (notional at the current mark)   Binance computes MM on mark notional
    liquidation      E(x) = MM(x)

`liquidation_distance` solves E(x) = MM(x) for |x|. The analysis then compares
that distance with two quantities per window:

    close-to-close breach   long: r <= -d_long      short: r >= d_short
    path breach             long: mae_long <= -d    short: mae_short >= d

A path breach without a close breach is a liquidation that a close-to-close risk
model never sees.

Binance's maintenance margin is tiered by position notional (`/fapi/v1/leverageBracket`)
and has a per-bracket "maintenance amount" deduction. The study uses one
representative rate per liquidity tier instead; those rates are assumptions,
stated in the results.
"""

from __future__ import annotations

import warnings
from typing import Literal

import pandas as pd

Side = Literal["long", "short"]


def liquidation_distance(leverage: float, mm_rate: float, side: Side) -> float:
    """Fractional adverse move from entry at which equity falls to maintenance margin.

    Returns a positive number (e.g. 0.095 means a 9.5% adverse move), or a
    value <= 0 when the position would already be under maintenance at entry
    (1 / leverage <= mm_rate).
    """
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    if not 0 <= mm_rate < 1:
        raise ValueError("mm_rate must be in [0, 1)")
    # Per unit of entry notional, with x the signed move and MM on mark notional m * (1 + x):
    #   long:  1/L + x = m (1 + x)  ->  x = -(1/L - m) / (1 - m)
    #   short: 1/L - x = m (1 + x)  ->  x = +(1/L - m) / (1 + m)
    # The long's maintenance shrinks as it loses, so it survives slightly further than a short.
    buffer = 1.0 / leverage - mm_rate
    if side == "long":
        return buffer / (1.0 - mm_rate)
    if side == "short":
        return buffer / (1.0 + mm_rate)
    raise ValueError(f"side must be 'long' or 'short', not {side!r}")


def threshold_grid(
    leverages: list[float], mm_by_tier: dict[str, float]
) -> pd.DataFrame:
    """One row per (tier, leverage, side) with its liquidation distance `d`.

    Combinations already under maintenance at entry are dropped with a warning:
    a venue would not let you open them.
    """
    rows = []
    for tier, m in mm_by_tier.items():
        for lev in leverages:
            for side in ("long", "short"):
                d = liquidation_distance(lev, m, side)
                if d <= 0:
                    warnings.warn(f"{tier} {lev}x {side}: under maintenance at entry (m={m}); skipped")
                    continue
                rows.append({"tier": tier, "leverage": float(lev), "side": side, "d": float(d)})
    return pd.DataFrame(rows)
