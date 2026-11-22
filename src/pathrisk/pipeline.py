"""End-to-end study: universe -> windows -> breaches -> summary with bootstrap CIs.

Memory stays bounded by processing one symbol at a time: its windows are reduced
to per-(date, symbol) breach counts before the next symbol loads. Only the
float32 window table needed for the quantile multiples is kept across symbols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import archive
from .bootstrap import bootstrap_ci
from .liquidation import threshold_grid
from .metrics import breach_counts, daily_rates, quantile_multiples, summarise
from .universe import UniverseConfig, build_universe, daily_quote_volume
from .windows import build_windows


@dataclass(frozen=True)
class StudyConfig:
    start: str  # YYYY-MM-DD, first window start date
    end: str  # YYYY-MM-DD, last window start date
    horizons: tuple[int, ...] = (60, 240, 480, 1440)  # minutes: 1h, 4h, the 8h funding interval, 24h
    step: int = 60  # a new window every hour
    leverages: tuple[float, ...] = (2, 3, 5, 10, 20, 50)
    # Representative maintenance-margin rates per tier. ASSUMPTIONS, not venue data:
    # replace with Binance's first-bracket rates from /fapi/v1/leverageBracket if you have a key.
    mm_by_tier: dict[str, float] = field(
        default_factory=lambda: {"majors": 0.004, "large": 0.005, "tail": 0.01, "new": 0.01}
    )
    universe: UniverseConfig = UniverseConfig()
    n_boot: int = 2000
    block_days: int = 7
    symbols: tuple[str, ...] | None = None  # restrict the universe, e.g. to the balanced cohort


def balanced_cohort(
    listing: pd.DataFrame, last_trade: pd.DataFrame, start: str, end: str, burn_in_days: int
) -> list[str]:
    """Symbols past their burn-in on `start` and still trading at the end of `end`.

    These survived the whole study, so the cohort is biased towards calmer assets: if the
    result holds here, survivorship is not what drives it.
    """
    lst = pd.to_datetime(listing.set_index("symbol")["listing_date"], utc=True)
    until = pd.to_datetime(last_trade.set_index("symbol")["trading_until"], utc=True)
    first_ok = pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=burn_in_days)
    last_ok = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    both = pd.concat([lst.rename("listed"), until.rename("until")], axis=1).dropna()
    return sorted(both.index[(both["listed"] <= first_ok) & (both["until"] >= last_ok)])


LAST_TRADE_CSV = "last_trade.csv"


def compute_last_trades(data_dir: Path) -> pd.DataFrame:
    """End of each symbol's last minute with traded volume, from last-price 1m klines.

    The archive keeps publishing flat, zero-volume bars for a symbol after it is delisted.
    Those windows show no risk at all, so the study stops each symbol here. NaT means the
    symbol never traded inside the downloaded range.
    """
    rows = []
    for p in sorted((data_dir / "parquet" / archive.LAST_1M.name / archive.LAST_1M.interval).glob("*.parquet")):
        d = pd.read_parquet(p, columns=["open_time", "quote_volume"])
        traded = d.loc[d["quote_volume"] > 0, "open_time"]
        end = pd.to_datetime(int(traded.max()) + 60_000, unit="ms", utc=True) if len(traded) else pd.NaT
        rows.append({"symbol": p.stem, "trading_until": end})
    return pd.DataFrame(rows, columns=["symbol", "trading_until"])


def load_universe(data_dir: Path, cfg: StudyConfig) -> pd.DataFrame:
    listing = pd.read_csv(data_dir / "listing.csv", parse_dates=["listing_date"])
    listing_map = {
        r.symbol: pd.Timestamp(r.listing_date).tz_localize("UTC")
        if pd.Timestamp(r.listing_date).tzinfo is None
        else pd.Timestamp(r.listing_date)
        for r in listing.dropna().itertuples()
    }
    vol_dir = data_dir / "parquet" / archive.LAST_1D.name / archive.LAST_1D.interval
    volumes = {
        p.stem: daily_quote_volume(pd.read_parquet(p)) for p in sorted(vol_dir.glob("*.parquet"))
    }
    dates = pd.date_range(cfg.start, cfg.end, freq="D", tz="UTC")
    universe = build_universe(volumes, listing_map, dates, cfg.universe)

    lt_path = data_dir / LAST_TRADE_CSV
    if not lt_path.exists():
        raise FileNotFoundError(f"{lt_path} missing: run `pathrisk delistings` first")
    last = pd.read_csv(lt_path)
    last["trading_until"] = pd.to_datetime(last["trading_until"], utc=True)
    # Inner join: a symbol with no last-price data, or no trade at all, cannot be shown to be live.
    universe = universe.merge(last.dropna(), on="symbol", how="inner")
    # Keep the delisting day itself: windows that close before the final trade still count.
    return universe[universe["date"] <= universe["trading_until"].dt.normalize()].reset_index(drop=True)


def run(
    dataset: archive.Dataset, data_dir: Path, out_dir: Path, cfg: StudyConfig
) -> dict[str, pd.DataFrame]:
    universe = load_universe(data_dir, cfg)
    if cfg.symbols is not None:
        universe = universe[universe["symbol"].isin(cfg.symbols)].reset_index(drop=True)
    grid = threshold_grid(list(cfg.leverages), cfg.mm_by_tier)
    bar_dir = data_dir / "parquet" / dataset.name / dataset.interval
    lo, hi = pd.Timestamp(cfg.start, tz="UTC"), pd.Timestamp(cfg.end, tz="UTC")

    counts, kept = [], []
    n_syms = universe["symbol"].nunique()
    for i, (sym, tiers) in enumerate(universe.groupby("symbol"), 1):
        if i % 50 == 0 or i == n_syms:
            print(f"  {i}/{n_syms} symbols", flush=True)
        path = bar_dir / f"{sym}.parquet"
        if not path.exists():
            continue
        bars = pd.read_parquet(path)
        tier_by_date = tiers.set_index("date")["tier"]
        trading_until = tiers["trading_until"].iloc[0]
        for h in cfg.horizons:
            w = build_windows(bars, h, cfg.step)
            w["date"] = w["start"].dt.normalize()
            ends = w["start"] + pd.Timedelta(minutes=h)
            w = w[(w["date"] >= lo) & (w["date"] <= hi) & (ends <= trading_until)]
            w["tier"] = w["date"].map(tier_by_date)
            w = w.dropna(subset=["tier"]).assign(symbol=sym, horizon=h)
            if w.empty:
                continue
            counts.append(breach_counts(w, grid))
            kept.append(w[["date", "symbol", "tier", "horizon", "r", "mae_long", "mae_short"]])

    if not counts:
        raise RuntimeError("no windows: check that bars, listing.csv and volumes cover the dates")
    daily = daily_rates(pd.concat(counts, ignore_index=True))
    summary = summarise(daily).merge(
        bootstrap_ci(daily, n_boot=cfg.n_boot, block=cfg.block_days), on=["tier", "horizon", "leverage", "side"]
    )
    windows = pd.concat(kept, ignore_index=True).astype({"symbol": "category", "tier": "category"})
    quantiles = quantile_multiples(windows)
    coverage = universe.groupby(["date", "tier"]).size().unstack(fill_value=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    quantiles.to_csv(out_dir / "quantiles.csv", index=False)
    daily.to_csv(out_dir / "daily_rates.csv", index=False)
    coverage.to_csv(out_dir / "universe_coverage.csv")
    grid.to_csv(out_dir / "liquidation_grid.csv", index=False)
    return {"summary": summary, "quantiles": quantiles, "daily": daily, "coverage": coverage}
