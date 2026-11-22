import zipfile

import pytest

from pathrisk.archive import _period, read_zip

ROW = "1767225600000,100,101,99,100.5,10,1767225659999,1005,3,5,502,0\n"
HEADER = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
          "taker_buy_volume,taker_buy_quote_volume,ignore\n")


def make_zip(tmp_path, body, name="X-1m-2026-01.csv"):
    p = tmp_path / "x.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr(name, body)
    return p


@pytest.mark.parametrize("body", [ROW, HEADER + ROW])
def test_read_zip_with_or_without_header(tmp_path, body):
    df = read_zip(make_zip(tmp_path, body))
    assert len(df) == 1
    assert df["open_time"].item() == 1767225600000
    assert df["low"].item() == 99


def test_read_zip_normalises_microseconds(tmp_path):
    df = read_zip(make_zip(tmp_path, ROW.replace("1767225600000", "1767225600000000", 1)))
    assert df["open_time"].item() == 1767225600000


def test_period_parsing():
    assert _period("data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-03.zip") == "2026-03"
    assert _period("data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-03-04.zip") == "2026-03-04"
