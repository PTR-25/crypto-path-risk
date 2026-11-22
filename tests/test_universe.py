import pandas as pd

from pathrisk.universe import UniverseConfig, build_universe

CFG = UniverseConfig(burn_in_days=5, trailing_days=5, min_trailing_obs=3, large_n=1, min_quote_volume=100)
DAYS = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")


def flat(v, days=DAYS):
    return pd.Series(float(v), index=days)


def tiers(u, date):
    return u[u["date"] == pd.Timestamp(date, tz="UTC")].set_index("symbol")["tier"].to_dict()


def test_tiers_are_point_in_time():
    a, b = flat(1000), flat(500)
    b[b.index >= pd.Timestamp("2026-01-20", tz="UTC")] = 10**9  # B's volume steps up on the 20th
    listing = {"A": DAYS[0], "B": DAYS[0]}
    u = build_universe({"A": a, "B": b}, listing, DAYS[10:], CFG)
    assert tiers(u, "2026-01-20") == {"A": "large", "B": "tail"}  # the step day itself is unseen
    assert tiers(u, "2026-01-22")["B"] == "tail"   # 2 of 5 trailing days high: median unmoved
    assert tiers(u, "2026-01-23")["B"] == "large"  # 3 of 5: the median, and the rank, move


def test_one_day_spike_does_not_change_the_tier():
    a, b = flat(1000), flat(500)
    b[pd.Timestamp("2026-01-20", tz="UTC")] = 10**9
    u = build_universe({"A": a, "B": b}, {"A": DAYS[0], "B": DAYS[0]}, DAYS[10:], CFG)
    assert set(u.loc[u["symbol"] == "B", "tier"]) == {"tail"}


def test_new_listings_majors_and_illiquid():
    vols = {"BTCUSDT": flat(1), "A": flat(1000), "DUST": flat(10), "NEW": flat(5000)}
    listing = {"BTCUSDT": DAYS[0], "A": DAYS[0], "DUST": DAYS[0], "NEW": DAYS[18]}
    cfg = UniverseConfig(**{**CFG.__dict__, "majors": ("BTCUSDT",)})
    u = build_universe(vols, listing, DAYS[10:], cfg)
    assert tiers(u, "2026-01-20") == {"BTCUSDT": "majors", "A": "large", "NEW": "new"}
    assert "NEW" not in tiers(u, "2026-01-15")  # not listed yet
    assert tiers(u, "2026-01-29")["NEW"] == "large"  # past burn-in, outranks A
