"""Point-in-time universe and liquidity tiers.

The pool of perpetuals changes every month: new listings, delistings, renames.
Rather than fix a list today (survivorship bias), every date t gets its own
universe built only from what was knowable before t:

    1. listed: the symbol's first archive bar is on or before t;
    2. age < burn_in_days            -> tier "new" (launch regime, studied on its own);
    3. symbol in majors              -> tier "majors";
    4. trailing median daily quote volume over [t - trailing_days, t - 1]
       below min_quote_volume        -> excluded (too illiquid to be a realistic book);
    5. rank by that trailing median  -> "large" (rank <= large_n) or "tail".

The tier *definitions* stay fixed while their membership moves, like an index
reconstitution. That is what makes periods with 150 symbols and periods with
250 comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TIERS = ("majors", "large", "tail", "new")


@dataclass(frozen=True)
class UniverseConfig:
    burn_in_days: int = 30
    trailing_days: int = 30
    min_trailing_obs: int = 20
    large_n: int = 30
    min_quote_volume: float = 5e6  # USDT per day, trailing median
    majors: tuple[str, ...] = field(default=("BTCUSDT", "ETHUSDT"))


def daily_quote_volume(daily_bars: pd.DataFrame) -> pd.Series:
    """1d klines -> quote volume indexed by UTC date (midnight timestamps)."""
    idx = pd.to_datetime(daily_bars["open_time"], unit="ms", utc=True).dt.normalize()
    return pd.Series(daily_bars["quote_volume"].to_numpy(float), index=idx).groupby(level=0).sum()


def build_universe(
    volumes: dict[str, pd.Series],
    listing: dict[str, pd.Timestamp],
    dates: pd.DatetimeIndex,
    cfg: UniverseConfig = UniverseConfig(),
) -> pd.DataFrame:
    """Return one row per (date, symbol) in the universe: date, symbol, tier, age_days, trailing_qv."""
    rows = []
    for sym, qv in volumes.items():
        listed = listing.get(sym)
        if listed is None:
            continue
        full = qv.reindex(pd.date_range(qv.index.min(), dates.max(), freq="D", tz="UTC"))
        # shift(1): the median for day t uses days t-trailing .. t-1, never t itself.
        trailing = full.rolling(cfg.trailing_days, min_periods=cfg.min_trailing_obs).median().shift(1)
        d = dates[dates >= listed.normalize()]
        if len(d) == 0:
            continue
        frame = pd.DataFrame(
            {
                "date": d,
                "symbol": sym,
                "age_days": (d - listed.normalize()).days.astype(int),
                "trailing_qv": trailing.reindex(d).to_numpy(),
            }
        )
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "tier", "age_days", "trailing_qv"])
    u = pd.concat(rows, ignore_index=True)

    new = u["age_days"] < cfg.burn_in_days
    major = u["symbol"].isin(cfg.majors) & ~new
    liquid = (u["trailing_qv"] >= cfg.min_quote_volume) & ~new & ~major
    ranked = u.loc[liquid].groupby("date")["trailing_qv"].rank(ascending=False, method="first")

    u["tier"] = None
    u.loc[new, "tier"] = "new"
    u.loc[major, "tier"] = "majors"
    u.loc[ranked.index, "tier"] = np.where(ranked <= cfg.large_n, "large", "tail")
    return u.dropna(subset=["tier"]).reset_index(drop=True)
