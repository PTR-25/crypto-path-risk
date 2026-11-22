import pandas as pd

from conftest import random_walk_bars
from pathrisk import archive, pipeline


def _run_synthetic(tmp_path, monkeypatch):
    data = tmp_path / "data"
    mark_dir = data / "parquet" / "markPriceKlines" / "1m"
    last_dir = data / "parquet" / "klines" / "1m"
    vol_dir = data / "parquet" / "klines" / "1d"
    for d in (mark_dir, last_dir, vol_dir):
        d.mkdir(parents=True)
    start_ms = int(pd.Timestamp("2025-12-01", tz="UTC").value // 1_000_000)
    days = 60
    # CCC is delisted at 09:00 on 15 Jan; the archive then keeps flat, zero-volume bars, as Binance's does.
    cut = pd.Timestamp("2026-01-15 09:00", tz="UTC")
    symbols = ["BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT"]
    for i, sym in enumerate(symbols):
        bars = random_walk_bars(1440 * days, 0.0008 * (i + 1), seed=i, start_ms=start_ms)
        vol = pd.Series(1e8 / (i + 1), index=range(days))
        if sym == "CCCUSDT":
            dead = pd.to_datetime(bars["open_time"], unit="ms", utc=True) >= cut
            final = bars.loc[~dead, "close"].iloc[-1]
            bars.loc[dead, ["open", "high", "low", "close"]] = final
            bars.loc[dead, "quote_volume"] = 0.0
            vol[vol.index >= (cut.normalize() - pd.Timestamp("2025-12-01", tz="UTC")).days + 1] = 0.0
        bars.to_parquet(mark_dir / f"{sym}.parquet")
        bars.to_parquet(last_dir / f"{sym}.parquet")
        pd.DataFrame({"open_time": start_ms + 86_400_000 * pd.RangeIndex(days).to_numpy(),
                      "quote_volume": vol.to_numpy()}).to_parquet(vol_dir / f"{sym}.parquet")
    pd.DataFrame({"symbol": symbols,
                  "listing_date": ["2025-12-01", "2025-12-01", "2026-01-10", "2025-12-01"]}).to_csv(
        data / "listing.csv", index=False)
    lt = pipeline.compute_last_trades(data)
    assert lt.set_index("symbol").loc["CCCUSDT", "trading_until"] == cut
    lt.to_csv(data / pipeline.LAST_TRADE_CSV, index=False)

    # Fixed thresholds, so this test does not depend on liquidation_distance.
    def fixed_grid(leverages, mm_by_tier):
        return pd.DataFrame([{"tier": t, "leverage": L, "side": s, "d": 0.9 / L}
                             for t in mm_by_tier for L in leverages for s in ("long", "short")])
    monkeypatch.setattr(pipeline, "threshold_grid", fixed_grid)

    cfg = pipeline.StudyConfig(start="2026-01-01", end="2026-01-29", horizons=(60, 1440),
                               leverages=(10, 20), n_boot=50)
    return pipeline.run(archive.MARK_1M, data, tmp_path / "out", cfg)


def test_pipeline_end_to_end_on_synthetic_archive(tmp_path, monkeypatch):
    res = _run_synthetic(tmp_path, monkeypatch)
    s = res["summary"]
    assert set(s["tier"]) == {"majors", "large", "new"}
    assert (s["p_path"] >= s["p_close"]).all()
    assert (tmp_path / "out" / "summary.csv").exists()
    cov = res["coverage"]
    assert cov.loc[pd.Timestamp("2026-01-05", tz="UTC"), "new"] == 0      # BBB not listed yet
    assert cov.loc[pd.Timestamp("2026-01-15", tz="UTC"), "new"] == 1


def test_regression_stale_bars_after_delisting_are_cut(tmp_path, monkeypatch):
    """Found in the 2026-11-22 full pull: 139 delisted symbols kept flat zero-volume bars."""
    res = _run_synthetic(tmp_path, monkeypatch)
    cov, daily = res["coverage"], res["daily"]
    assert cov.loc[pd.Timestamp("2026-01-15", tz="UTC"), "large"] == 2   # AAA and CCC on its last day
    assert cov.loc[pd.Timestamp("2026-01-16", tz="UTC"), "large"] == 1   # CCC gone after delisting
    day = daily[(daily["date"] == pd.Timestamp("2026-01-15", tz="UTC")) & (daily["tier"] == "large")
                & (daily["horizon"] == 60)]
    assert set(day["n_windows"]) == {24 + 9}  # CCC keeps only the 9 hourly windows closing by 09:00


def test_balanced_cohort_needs_burn_in_before_start_and_trading_at_end():
    listing = pd.DataFrame({"symbol": ["OLD", "YOUNG", "GONE"],
                            "listing_date": ["2025-01-01", "2025-10-20", "2025-01-01"]})
    last = pd.DataFrame({"symbol": ["OLD", "YOUNG", "GONE"],
                         "trading_until": ["2026-11-01 00:00:00+00:00", "2026-11-01 00:00:00+00:00",
                                           "2026-06-01 09:00:00+00:00"]})
    assert pipeline.balanced_cohort(listing, last, "2025-11-01", "2026-10-31", 30) == ["OLD"]


def test_per_contract_margins_skip_unoffered_leverage():
    terms = {"tail": (0.025, 20.0)}  # a typical small contract: 2.5% maintenance, 20x max
    g = pipeline.symbol_grid((10, 20, 25, 50), terms)
    assert set(g["leverage"]) == {10.0, 20.0}
    d10 = g[(g.leverage == 10) & (g.side == "long")]["d"].item()
    assert abs(d10 - (0.1 - 0.025) / (1 - 0.025)) < 1e-12


def test_tier_fallback_is_median_of_known_contracts():
    u = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "tier": ["tail"] * 4})
    fb = pipeline.tier_fallback_margins(u, {"A": (0.01, 50.0), "B": (0.02, 25.0), "C": (0.03, 20.0)})
    assert fb == {"tail": (0.02, 25.0)}
