"""Properties any correct liquidation_distance must have. They skip until the TODO(human)
in liquidation.py is filled in."""

import pytest

from pathrisk.liquidation import liquidation_distance


@pytest.fixture(autouse=True)
def require_implementation():
    try:
        liquidation_distance(10, 0.005, "long")
    except NotImplementedError:
        pytest.skip("liquidation_distance is still TODO(human)")


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("lev,m", [(2, 0.004), (10, 0.005), (20, 0.01), (50, 0.004)])
def test_equity_equals_maintenance_at_the_threshold(side, lev, m):
    d = liquidation_distance(lev, m, side)
    x = -d if side == "long" else d          # the adverse move
    pnl = x if side == "long" else -x        # per unit of entry notional
    equity = 1 / lev + pnl
    maintenance = m * (1 + x)                # m times notional at the current mark
    assert equity == pytest.approx(maintenance, abs=1e-12)


@pytest.mark.parametrize("side", ["long", "short"])
def test_zero_maintenance_gives_one_over_leverage(side):
    assert liquidation_distance(10, 0.0, side) == pytest.approx(0.1)


@pytest.mark.parametrize("side", ["long", "short"])
def test_more_leverage_means_closer_liquidation(side):
    ds = [liquidation_distance(L, 0.005, side) for L in (2, 5, 10, 20, 50)]
    assert ds == sorted(ds, reverse=True)
    assert all(0 < d < 1 / L for d, L in zip(ds, (2, 5, 10, 20, 50)))


def test_under_maintenance_at_entry_is_not_positive():
    assert liquidation_distance(200, 0.01, "long") <= 0
