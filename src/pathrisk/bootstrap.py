"""Moving-block bootstrap over dates.

Windows are not independent: rolling windows overlap, and a crash day hits every
symbol at once. So the resampling unit is the calendar day (all symbols together,
which keeps the cross-sectional correlation), drawn in contiguous blocks (which
keeps the serial correlation). The effective sample size is about the number of
days, not the number of instrument-windows.

Every bootstrap draw uses the same resampled dates for every group, so a
comparison between tiers or horizons is made on the same days.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .metrics import GROUP


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block resample of range(n)."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]


def bootstrap_ci(
    daily: pd.DataFrame,
    n_boot: int = 2000,
    block: int = 7,
    level: float = 0.95,
    seed: int = 7,
) -> pd.DataFrame:
    """Confidence intervals for breach_ratio and hidden_share per group."""
    p_path = daily.pivot_table(index="date", columns=GROUP, values="p_path")
    p_close = daily.pivot_table(index="date", columns=GROUP, values="p_close")
    p_close = p_close.reindex(index=p_path.index, columns=p_path.columns)
    a, b = p_path.to_numpy(), p_close.to_numpy()

    rng = np.random.default_rng(seed)
    ratios = np.empty((n_boot, a.shape[1]))
    hidden = np.empty((n_boot, a.shape[1]))
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns (a tier absent on the drawn days)
        for i in range(n_boot):
            idx = block_indices(len(a), block, rng)
            mp, mc = np.nanmean(a[idx], axis=0), np.nanmean(b[idx], axis=0)
            ratios[i] = np.where(mc > 0, mp / mc, np.nan)
            hidden[i] = np.where(mp > 0, 1 - mc / mp, np.nan)

    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out = p_path.columns.to_frame(index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # groups with no close breach in any draw
        out["breach_ratio_lo"] = np.nanquantile(ratios, lo, axis=0)
        out["breach_ratio_hi"] = np.nanquantile(ratios, hi, axis=0)
        out["hidden_share_lo"] = np.nanquantile(hidden, lo, axis=0)
        out["hidden_share_hi"] = np.nanquantile(hidden, hi, axis=0)
    return out
