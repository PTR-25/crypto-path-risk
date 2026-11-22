"""Snapshot of maximum leverage on linear perpetuals at five venues, from their public APIs.

Writes paper/data/venue_leverage.csv (one row per contract, with the retrieval time) and
paper/tables/tab_venues.tex. Leverage limits change; the committed snapshot is what the
paper cites. Re-running replaces it with today's values.

    uv run python -I scripts/venue_leverage.py

Sources (no API key needed; read-only market metadata):
  Binance  https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets
           (the feed behind Binance's Leverage & Margin page; first bracket of each USDT perpetual)
  Bybit    https://api.bybit.com/v5/market/instruments-info?category=linear
  OKX      https://www.okx.com/api/v5/public/instruments?instType=SWAP
  Deribit  https://www.deribit.com/api/v2/public/get_instruments?currency=USDC&kind=future
  Hyperliquid  POST https://api.hyperliquid.xyz/info {"type": "meta"}
"""

from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "pathrisk-research/0.1", "Content-Type": "application/json"}


def get(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=UA, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def binance() -> list[tuple]:
    rows = []
    for s in get("https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets")["data"]["brackets"]:
        if s["symbol"].endswith("USDT"):
            b = min(s["riskBrackets"], key=lambda x: x["bracketSeq"])
            rows.append(("Binance", s["symbol"], float(b["maxOpenPosLeverage"]), float(b["bracketMaintenanceMarginRate"])))
    return rows


def bybit() -> list[tuple]:
    rows, cursor = [], ""
    while True:
        r = get(f"https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000&cursor={cursor}")["result"]
        rows += [("Bybit", x["symbol"], float(x["leverageFilter"]["maxLeverage"]), None) for x in r["list"]
                 if x.get("contractType") == "LinearPerpetual" and x["symbol"].endswith("USDT")]
        cursor = r.get("nextPageCursor") or ""
        if not cursor:
            return rows


def okx() -> list[tuple]:
    return [("OKX", x["instId"], float(x["lever"]), None)
            for x in get("https://www.okx.com/api/v5/public/instruments?instType=SWAP")["data"]
            if x["instId"].endswith("-USDT-SWAP") and x["lever"]]


def deribit() -> list[tuple]:
    r = get("https://www.deribit.com/api/v2/public/get_instruments?currency=USDC&kind=future")["result"]
    return [("Deribit", x["instrument_name"], float(x["max_leverage"]), None) for x in r
            if x.get("settlement_period") == "perpetual"]


def hyperliquid() -> list[tuple]:
    u = get("https://api.hyperliquid.xyz/info", {"type": "meta"})["universe"]
    return [("Hyperliquid", a["name"], float(a["maxLeverage"]), None) for a in u if not a.get("isDelisted")]


BTC = {"Binance": "BTCUSDT", "Bybit": "BTCUSDT", "OKX": "BTC-USDT-SWAP", "Deribit": "BTC_USDC-PERPETUAL",
       "Hyperliquid": "BTC"}


def main() -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rows = binance() + bybit() + okx() + deribit() + hyperliquid()
    df = pd.DataFrame(rows, columns=["venue", "contract", "max_leverage", "first_bracket_mmr"]).assign(retrieved=now)
    (ROOT / "paper" / "data").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "paper" / "data" / "venue_leverage.csv", index=False)

    lines = []
    for venue in BTC:
        v = df[df.venue == venue]
        lev = v.max_leverage
        btc = v.loc[v.contract == BTC[venue], "max_leverage"]
        cells = [venue, f"{len(v)}", f"{btc.iloc[0]:.0f}x" if len(btc) else "--", f"{lev.max():.0f}x",
                 f"{lev.median():.0f}x", f"{(lev >= 50).mean():.0%}".replace("%", "\\%"),
                 f"{(lev <= 10).mean():.0%}".replace("%", "\\%")]
        lines.append(" & ".join(cells) + " \\\\")
    (ROOT / "paper" / "tables" / "tab_venues.tex").write_text(
        "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
        "Venue & Contracts & BTC & Highest & Median & $\\geq$ 50x & $\\leq$ 10x \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    b = df[df.venue == "Binance"]
    print(f"retrieved {now}: {len(df)} contracts")
    print(f"Binance first-bracket MMR: median {b.first_bracket_mmr.median():.4f}; "
          f"BTC {b.loc[b.contract == 'BTCUSDT', 'first_bracket_mmr'].iloc[0]}")
    print((ROOT / "paper" / "tables" / "tab_venues.tex").read_text())


if __name__ == "__main__":
    main()
