"""Download and parse Binance USD-M futures klines from the public archive.

Source: https://data.binance.vision (an S3 bucket). Unlike the live REST API's
`exchangeInfo`, the archive keeps files for delisted symbols, so a universe
built from it is not survivorship-biased.

Layout used here (monthly files; the current month only exists as daily files):

    data/futures/um/monthly/{dataset}/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY-MM}.zip
    data/futures/um/daily/{dataset}/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY-MM-DD}.zip

Each zip has a sibling `.zip.CHECKSUM` holding its SHA-256.

Downloaded files are untrusted data: they are only ever parsed as CSV, never
executed, and they live under `data/raw/`, apart from the code.
"""

from __future__ import annotations

import hashlib
import io
import time
import urllib.parse
import urllib.request
import defusedxml.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LISTING_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DOWNLOAD_URL = "https://data.binance.vision/"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
USER_AGENT = "pathrisk-research/0.1"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
KEEP_COLUMNS = ["open_time", "open", "high", "low", "close", "quote_volume"]


@dataclass(frozen=True)
class Dataset:
    """One archive series: `markPriceKlines` (liquidation price) or `klines` (last price)."""

    name: str
    interval: str

    def monthly_prefix(self, symbol: str | None = None) -> str:
        base = f"data/futures/um/monthly/{self.name}/"
        return base if symbol is None else f"{base}{symbol}/{self.interval}/"

    def daily_prefix(self, symbol: str) -> str:
        return f"data/futures/um/daily/{self.name}/{symbol}/{self.interval}/"


MARK_1M = Dataset("markPriceKlines", "1m")
LAST_1M = Dataset("klines", "1m")
LAST_1D = Dataset("klines", "1d")  # daily quote volume, for the point-in-time liquidity tiers
DATASETS = {"mark": MARK_1M, "last": LAST_1M, "volume": LAST_1D}


def _get(url: str, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _list(prefix: str) -> tuple[list[str], list[str]]:
    """Return (sub-prefixes, keys) under `prefix`, following S3 pagination."""
    prefixes: list[str] = []
    keys: list[str] = []
    marker = ""
    while True:
        query = {"delimiter": "/", "prefix": prefix}
        if marker:
            query["marker"] = marker
        root = ET.fromstring(_get(f"{LISTING_URL}?{urllib.parse.urlencode(query)}"))
        page_prefixes = [p.text for p in root.findall("s3:CommonPrefixes/s3:Prefix", S3_NS)]
        page_keys = [k.text for k in root.findall("s3:Contents/s3:Key", S3_NS)]
        prefixes += [p for p in page_prefixes if p]
        keys += [k for k in page_keys if k]
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS)
        if truncated.lower() != "true":
            return prefixes, keys
        marker = root.findtext("s3:NextMarker", default="", namespaces=S3_NS) or max(
            page_prefixes + page_keys
        )


def list_symbols(dataset: Dataset = MARK_1M, quote: str = "USDT") -> list[str]:
    """Every perpetual symbol the archive has ever held for `dataset`, delisted included.

    Dated delivery contracts (`BTCUSDT_250328`) are dropped, and only one quote
    asset is kept so the same underlying is not counted twice (BTCUSDT and BTCUSDC).
    """
    prefixes, _ = _list(dataset.monthly_prefix())
    symbols = [p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes]
    return sorted(s for s in symbols if s.endswith(quote) and "_" not in s)


def list_files(dataset: Dataset, symbol: str, start: str, end: str) -> list[str]:
    """Zip keys for `symbol` with month in [start, end] (YYYY-MM), plus daily files
    for months the archive has no monthly file for yet (the current month)."""
    _, keys = _list(dataset.monthly_prefix(symbol))
    monthly = [k for k in keys if k.endswith(".zip") and start <= _period(k)[:7] <= end]
    have = {_period(k)[:7] for k in monthly}
    months = pd.period_range(start, end, freq="M").strftime("%Y-%m")
    missing = [m for m in months if m not in have]
    daily: list[str] = []
    if missing:
        _, dkeys = _list(dataset.daily_prefix(symbol))
        daily = [k for k in dkeys if k.endswith(".zip") and _period(k)[:7] in missing]
    return sorted(monthly + daily)


def _period(key: str) -> str:
    """`.../BTCUSDT-1m-2026-03.zip` -> `2026-03`; daily keys give `2026-03-04`."""
    stem = key.rsplit("/", 1)[-1].removesuffix(".zip")
    parts = stem.split("-")
    return "-".join(parts[2:])


def download(key: str, raw_dir: Path) -> Path:
    """Fetch one zip into `raw_dir`, verify its SHA-256, and return the local path."""
    dest = raw_dir / key
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = DOWNLOAD_URL + urllib.parse.quote(key)  # some symbols are not ASCII (e.g. 龙虾USDT)
    body = _get(url)
    try:
        expected = _get(url + ".CHECKSUM").decode().split()[0]
    except urllib.error.HTTPError:
        expected = None
    actual = hashlib.sha256(body).hexdigest()
    if expected is not None and expected != actual:
        raise ValueError(f"checksum mismatch for {key}: {actual} != {expected}")
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return dest


def read_zip(path: Path) -> pd.DataFrame:
    """Parse one archive zip into KEEP_COLUMNS, tolerating files with or without a header."""
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected one CSV, found {names}")
        raw = zf.read(names[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLUMNS, dtype=str)
    df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]  # drop a header row
    out = pd.DataFrame(
        {c: pd.to_numeric(df[c], errors="coerce") for c in KEEP_COLUMNS}
    ).dropna(subset=["open_time", "open", "high", "low", "close"])
    t = out["open_time"].astype(np.int64)
    # Binance moved some archives to microsecond timestamps; normalise to milliseconds.
    out["open_time"] = np.where(t > 10**14, t // 1000, t)
    return out


def fetch_symbol(
    dataset: Dataset, symbol: str, start: str, end: str, data_dir: Path
) -> Path | None:
    """Download and consolidate one symbol into `parquet/{name}/{interval}/{SYMBOL}.parquet`."""
    keys = list_files(dataset, symbol, start, end)
    if not keys:
        return None
    raw_dir = data_dir / "raw"
    frames = [read_zip(download(k, raw_dir)) for k in keys]
    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("open_time")
        .sort_values("open_time", ignore_index=True)
    )
    out = data_dir / "parquet" / dataset.name / dataset.interval / f"{symbol}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def fetch_many(
    dataset: Dataset,
    symbols: list[str],
    start: str,
    end: str,
    data_dir: Path,
    workers: int = 4,
) -> dict[str, Path | Exception | None]:
    """Fetch several symbols with a small thread pool (the archive is a CDN, not the API,
    so there is no request weight, but stay polite)."""
    def one(sym: str) -> tuple[str, Path | Exception | None]:
        try:
            return sym, fetch_symbol(dataset, sym, start, end, data_dir)
        except Exception as exc:  # one bad symbol must not abort a multi-hour pull
            return sym, exc

    results: dict[str, Path | Exception | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (sym, res) in enumerate(pool.map(one, symbols), 1):
            results[sym] = res
            if isinstance(res, Exception):
                print(f"  FAILED {sym}: {res!r}", flush=True)
            if i % 25 == 0 or i == len(symbols):
                print(f"  {i}/{len(symbols)} symbols", flush=True)
    return results


def load(dataset: Dataset, symbol: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(
        data_dir / "parquet" / dataset.name / dataset.interval / f"{symbol}.parquet"
    )


def listing_date(symbol: str, data_dir: Path) -> pd.Timestamp | None:
    """First day the archive has a 1d bar for `symbol`: its listing day.

    Taken from the earliest monthly 1d file, so it is the true listing date even when
    the study window starts later. A symbol delisted and relisted under the same name
    keeps its first listing date (its age is overstated after the relisting).
    """
    _, keys = _list(LAST_1D.monthly_prefix(symbol))
    zips = sorted((k for k in keys if k.endswith(".zip")), key=_period)
    if not zips:
        return None
    first = read_zip(download(zips[0], data_dir / "raw"))
    if first.empty:
        return None
    return pd.to_datetime(int(first["open_time"].min()), unit="ms", utc=True).normalize()
