"""Command line: symbols -> listing -> fetch -> analyze.

    pathrisk symbols
    pathrisk listing
    pathrisk fetch --dataset volume --start 2025-09 --end 2026-10
    pathrisk fetch --dataset mark   --start 2025-11 --end 2026-10
    pathrisk fetch --dataset last   --start 2025-11 --end 2026-10
    pathrisk delistings
    pathrisk analyze --dataset mark --start 2025-11-01 --end 2026-10-31

Every command takes --data (default ./data). Only `symbols`, `listing` and
`fetch` touch the network, and only the public data.binance.vision archive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import archive
from .pipeline import LAST_TRADE_CSV, StudyConfig, balanced_cohort, compute_last_trades, run
from .universe import UniverseConfig


def _symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return args.symbols.split(",")
    f = args.data / "symbols.txt"
    if not f.exists():
        sys.exit(f"{f} not found: run `pathrisk symbols` first or pass --symbols")
    return f.read_text().split()


def cmd_symbols(args: argparse.Namespace) -> None:
    syms = archive.list_symbols(archive.MARK_1M, quote=args.quote)
    args.data.mkdir(parents=True, exist_ok=True)
    (args.data / "symbols.txt").write_text("\n".join(syms) + "\n")
    print(f"{len(syms)} {args.quote} perpetuals (delisted included) -> {args.data / 'symbols.txt'}")


def cmd_listing(args: argparse.Namespace) -> None:
    rows = []
    for i, sym in enumerate(_symbols(args), 1):
        try:
            rows.append({"symbol": sym, "listing_date": archive.listing_date(sym, args.data)})
        except Exception as exc:
            print(f"  FAILED {sym}: {exc!r}", flush=True)
            rows.append({"symbol": sym, "listing_date": None})
        if i % 50 == 0:
            print(f"  {i} symbols", flush=True)
    df = pd.DataFrame(rows)
    out = args.data / "listing.csv"
    if out.exists():  # merge, so re-running a few symbols keeps everyone else's dates
        old = pd.read_csv(out)
        df = pd.concat([old[~old["symbol"].isin(df["symbol"])], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"{df['listing_date'].notna().sum()} listing dates -> {args.data / 'listing.csv'}")


def cmd_fetch(args: argparse.Namespace) -> None:
    ds = archive.DATASETS[args.dataset]
    syms = _symbols(args)
    listing_csv = args.data / "listing.csv"
    if listing_csv.exists() and not args.symbols:
        listed = pd.read_csv(listing_csv).dropna()
        cutoff = pd.Period(args.end, "M").end_time.strftime("%Y-%m-%d")
        syms = sorted(set(syms) & set(listed.loc[listed["listing_date"] <= cutoff, "symbol"]))
    print(f"fetching {ds.name}/{ds.interval} for {len(syms)} symbols, {args.start}..{args.end}")
    res = archive.fetch_many(ds, syms, args.start, args.end, args.data, workers=args.workers)
    failed = sorted(s for s, p in res.items() if isinstance(p, Exception))
    empty = sum(p is None for p in res.values())
    print(f"{len(res) - empty - len(failed)} symbols written, {empty} with no files in range, "
          f"{len(failed)} failed{': ' + ','.join(failed) if failed else ''}")
    if failed:
        sys.exit(1)


def cmd_delistings(args: argparse.Namespace) -> None:
    df = compute_last_trades(args.data)
    df.to_csv(args.data / LAST_TRADE_CSV, index=False)
    end = df["trading_until"].max()
    stopped = df[df["trading_until"] < end - pd.Timedelta(days=1)]
    print(f"{len(df)} symbols: {len(stopped)} stopped trading before {end:%Y-%m-%d}, "
          f"{df['trading_until'].isna().sum()} never traded in range -> {args.data / LAST_TRADE_CSV}")


def cmd_cohort(args: argparse.Namespace) -> None:
    syms = balanced_cohort(pd.read_csv(args.data / "listing.csv"), pd.read_csv(args.data / LAST_TRADE_CSV),
                           args.start, args.end, UniverseConfig().burn_in_days)
    (args.data / "cohort.txt").write_text("\n".join(syms) + "\n")
    print(f"{len(syms)} symbols listed before {args.start} (plus burn-in) and trading through {args.end} "
          f"-> {args.data / 'cohort.txt'}")


def cmd_analyze(args: argparse.Namespace) -> None:
    symbols = tuple(args.symbols_file.read_text().split()) if args.symbols_file else None
    cfg = StudyConfig(
        symbols=symbols,
        start=args.start,
        end=args.end,
        horizons=tuple(int(h) for h in args.horizons.split(",")),
        step=args.step,
        leverages=tuple(float(x) for x in args.leverages.split(",")),
        n_boot=args.n_boot,
    )
    out = args.out / (args.tag or args.dataset)
    res = run(archive.DATASETS[args.dataset], args.data, out, cfg)
    s = res["summary"]
    cols = ["tier", "horizon", "leverage", "side", "p_close", "p_path", "breach_ratio",
            "breach_ratio_lo", "breach_ratio_hi", "hidden_share"]
    with pd.option_context("display.width", 160, "display.max_rows", 80):
        print(s.loc[s["horizon"] == max(cfg.horizons), cols].round(4).to_string(index=False))
    print(f"\nresults -> {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pathrisk")
    p.add_argument("--data", type=Path, default=Path("data"))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("symbols", help="list every USD-M perpetual in the archive")
    s.add_argument("--quote", default="USDT")
    s.set_defaults(fn=cmd_symbols)

    s = sub.add_parser("listing", help="find each symbol's listing date")
    s.add_argument("--symbols", help="comma-separated; default data/symbols.txt")
    s.set_defaults(fn=cmd_listing)

    s = sub.add_parser("fetch", help="download klines for a month range")
    s.add_argument("--dataset", choices=sorted(archive.DATASETS), required=True)
    s.add_argument("--start", required=True, help="YYYY-MM")
    s.add_argument("--end", required=True, help="YYYY-MM")
    s.add_argument("--symbols", help="comma-separated; default: listed symbols in data/symbols.txt")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("delistings", help="find each symbol's last real trade (needs last-price 1m data)")
    s.set_defaults(fn=cmd_delistings)

    s = sub.add_parser("cohort", help="write the balanced cohort: symbols live for the whole study")
    s.add_argument("--start", required=True, help="YYYY-MM-DD")
    s.add_argument("--end", required=True, help="YYYY-MM-DD")
    s.set_defaults(fn=cmd_cohort)

    s = sub.add_parser("analyze", help="run the study on downloaded data")
    s.add_argument("--dataset", choices=["mark", "last"], default="mark")
    s.add_argument("--start", required=True, help="YYYY-MM-DD")
    s.add_argument("--end", required=True, help="YYYY-MM-DD")
    s.add_argument("--horizons", default="60,240,480,1440", help="minutes")
    s.add_argument("--step", type=int, default=60, help="minutes between window starts")
    s.add_argument("--leverages", default="2,3,5,10,20,50")
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--out", type=Path, default=Path("results"))
    s.add_argument("--symbols-file", type=Path, help="restrict the universe to these symbols")
    s.add_argument("--tag", help="results subfolder (default: the dataset name)")
    s.set_defaults(fn=cmd_analyze)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
