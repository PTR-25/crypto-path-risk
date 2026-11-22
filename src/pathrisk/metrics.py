"""Breach counting, cross-sectional aggregation, and the Brownian baselines.

Weighting: each date counts once, and within a date each symbol counts once.
Rates are computed per (symbol, date), averaged across symbols for each date,
then averaged across dates. Without this, months with many new listings (bull
markets) would dominate the pooled result.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

GROUP = ["tier", "horizon", "leverage", "side"]

# Driftless Brownian motion, continuous monitoring (reflection principle):
# P(min over [0, T] <= -d) = 2 * P(B_T <= -d). So even in a Gaussian world a
# close-to-close model misses half of all barrier touches.
BROWNIAN_BREACH_RATIO = 2.0
BROWNIAN_HIDDEN_SHARE = 0.5


def brownian_quantile_multiple(alpha: float) -> float:
    """Ratio of the alpha-quantile of the running-minimum loss to that of the terminal loss."""
    n = NormalDist()
    return n.inv_cdf((1 + alpha) / 2) / n.inv_cdf(alpha)


def brownian_breach_ratio_discrete(z: float, steps: int) -> float:
    """Expected breach ratio when the path is only seen at `steps` points.

    Broadie-Glasserman-Kou continuity correction: a discretely monitored barrier
    behaves like a continuous one shifted outward by 0.5826 * sigma * sqrt(dt).
    `z` is d / sigma_T.
    """
    n = NormalDist()
    shift = 0.5826 / np.sqrt(steps)
    return 2 * n.cdf(-(z + shift)) / n.cdf(-z)


def breach_counts(windows: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Count breaches per (date, symbol, tier, horizon, leverage, side).

    `windows` needs: date, symbol, tier, horizon, r, mae_long, mae_short.
    `grid` needs: tier, leverage, side, d (from `liquidation.threshold_grid`).
    """
    out = []
    for (lev, side), g in grid.groupby(["leverage", "side"]):
        d = windows["tier"].map(g.set_index("tier")["d"])
        w = windows.loc[d.notna()]
        d = d[d.notna()].to_numpy()
        if side == "long":
            path = w["mae_long"].to_numpy() <= -d
            close = w["r"].to_numpy() <= -d
        else:
            path = w["mae_short"].to_numpy() >= d
            close = w["r"].to_numpy() >= d
        c = w[["date", "symbol", "tier", "horizon"]].assign(
            n=1, n_path=path.astype(np.int32), n_close=close.astype(np.int32)
        )
        c = c.groupby(["date", "symbol", "tier", "horizon"], as_index=False)[["n", "n_path", "n_close"]].sum()
        out.append(c.assign(leverage=lev, side=side))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def daily_rates(counts: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight cross-sectional breach rates for each date and group."""
    c = counts.assign(p_path=counts["n_path"] / counts["n"], p_close=counts["n_close"] / counts["n"])
    return c.groupby(["date", *GROUP], as_index=False).agg(
        p_path=("p_path", "mean"),
        p_close=("p_close", "mean"),
        n_symbols=("symbol", "nunique"),
        n_windows=("n", "sum"),
    )


def summarise(daily: pd.DataFrame) -> pd.DataFrame:
    s = daily.groupby(GROUP, as_index=False).agg(
        p_path=("p_path", "mean"),
        p_close=("p_close", "mean"),
        n_dates=("date", "nunique"),
        mean_symbols=("n_symbols", "mean"),
        n_windows=("n_windows", "sum"),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        s["breach_ratio"] = np.where(s["p_close"] > 0, s["p_path"] / s["p_close"], np.nan)
        s["hidden_share"] = np.where(s["p_path"] > 0, 1 - s["p_close"] / s["p_path"], np.nan)
    return s


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return float(np.interp(q * cw[-1], cw, v))


def quantile_multiples(windows: pd.DataFrame, alphas=(0.95, 0.99, 0.999)) -> pd.DataFrame:
    """Loss quantiles of the path (MAE) against close-to-close, per tier, horizon and side,
    weighted so every date and every symbol within a date counts once."""
    w = windows.copy()
    per_symdate = w.groupby(["tier", "horizon", "date", "symbol"])["r"].transform("size")
    syms_per_date = w.groupby(["tier", "horizon", "date"])["symbol"].transform("nunique")
    w["weight"] = 1.0 / (per_symdate * syms_per_date)
    rows = []
    for (tier, horizon), g in w.groupby(["tier", "horizon"]):
        wt = g["weight"].to_numpy()
        losses = {
            "long": (-g["r"].to_numpy(float), -g["mae_long"].to_numpy(float)),
            "short": (g["r"].to_numpy(float), g["mae_short"].to_numpy(float)),
        }
        for side, (close_loss, path_loss) in losses.items():
            for a in alphas:
                qc = weighted_quantile(close_loss, wt, a)
                qp = weighted_quantile(path_loss, wt, a)
                rows.append(
                    {
                        "tier": tier, "horizon": horizon, "side": side, "alpha": a,
                        "q_close": qc, "q_path": qp,
                        "multiple": qp / qc if qc > 0 else np.nan,
                        "brownian_multiple": brownian_quantile_multiple(a),
                    }
                )
    return pd.DataFrame(rows)
