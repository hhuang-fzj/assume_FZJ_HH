# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test of replay_soc() in storage_two_market.py: the SOC planned one day ahead must equal
the SOC that ASSUME reaches when it executes the committed volumes
(storage.py, execute_current_dispatch).

Battery: Battery_Bus14 of the scenario (13.7 kW, 27.4 kWh, efficiency 0.95,
SOC 10-90 %, initial 50 %).
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from assume.common.forecaster import UnitForecaster
from assume.strategies.storage_two_market import StorageTwoMarketStrategy, replay_soc
from assume.units.storage import Storage

DAY1 = pd.Timestamp("2016-06-21")
DAY2 = pd.Timestamp("2016-06-22")


INDEX = pd.date_range(DAY1, "2016-06-22 23:45", freq="15min")


def make_battery(prices=None):
    if prices is None:
        prices = np.full(len(INDEX), 30.0)
    forecaster = UnitForecaster(INDEX, market_prices={"LEM_DA": prices})
    strategy = StorageTwoMarketStrategy()
    return Storage(
        id="Battery_Bus14",
        unit_operator="op",
        technology="battery",
        bidding_strategies={"LEM_DA": strategy, "WM_DA": strategy},
        forecaster=forecaster,
        max_power_charge=-0.0137,
        max_power_discharge=0.0137,
        capacity=0.0274,
        min_soc=0.1,
        max_soc=0.9,
        initial_soc=0.5,
        efficiency_charge=0.95,
        efficiency_discharge=0.95,
    )


@pytest.mark.parametrize(
    "plan",
    [
        "charge_all_day",  # SOC limit cuts the charge after a few quarter hours
        "discharge_all_day",  # SOC limit cuts the discharge
        "mixed",  # charge in the morning, discharge in the evening
    ],
)
def test_replay_equals_assume_dispatch(plan):
    unit = make_battery()
    day1 = unit.index[DAY1 : DAY2 - timedelta(minutes=15)]
    for i, t in enumerate(day1):
        if plan == "charge_all_day":
            power = -0.0137
        elif plan == "discharge_all_day":
            power = 0.0137
        else:
            power = -0.0137 if i < 40 else (0.0137 if i > 70 else 0.0)
        unit.outputs["energy"].at[t] = power  # committed volume (set_dispatch_plan)

    planned = replay_soc(unit, DAY2)  # what the strategy sees one day ahead
    assert unit.outputs["soc"].at[DAY2] == pytest.approx(0.5)  # stale value

    unit.execute_current_dispatch(DAY1, DAY2 - timedelta(minutes=15))  # what ASSUME does
    assert planned == pytest.approx(unit.outputs["soc"].at[DAY2], abs=1e-12)


def test_day2_bids_use_replayed_soc():
    """After a full charge on day 1 the battery is at 90 % and cannot charge at 00:00."""
    # forecast below the daily mean at 00:00 of day 2 -> the rule wants to charge
    prices = np.full(len(INDEX), 30.0)
    prices[INDEX.get_loc(DAY2)] = 10.0
    unit = make_battery(prices)
    for t in unit.index[DAY1 : DAY2 - timedelta(minutes=15)]:
        unit.outputs["energy"].at[t] = -0.0137

    day2 = pd.date_range(DAY2, "2016-06-22 23:45", freq="15min")
    products = [(t, t + timedelta(minutes=15), None) for t in day2]
    from assume.common.market_objects import MarketConfig, MarketProduct

    config = MarketConfig(
        market_id="LEM_DA",
        market_products=[MarketProduct(timedelta(minutes=15), 96, timedelta(0))],
    )
    bids = unit.bidding_strategies["LEM_DA"].calculate_bids(unit, config, products)
    first = {b["start_time"]: b for b in bids}.get(DAY2)
    assert replay_soc(unit, DAY2) == pytest.approx(0.9)
    assert first is None or first["volume"] >= 0  # no charge bid at full SOC

    # control: without the committed charge (SOC 50 %) the same rule bids a charge
    fresh = make_battery(prices)
    bids = fresh.bidding_strategies["LEM_DA"].calculate_bids(fresh, config, products)
    assert {b["start_time"]: b for b in bids}[DAY2]["volume"] < 0


def _commit(unit, market_id, orders):
    """Accept orders as ASSUME does after clearing (units_operator.set_unit_dispatch)."""
    from assume.common.market_objects import MarketConfig, MarketProduct

    config = MarketConfig(
        market_id=market_id,
        market_products=[MarketProduct(timedelta(minutes=15), 96, timedelta(0))],
    )
    book = [
        {
            "start_time": t,
            "end_time": t + timedelta(minutes=15),
            "only_hours": None,
            "price": 0.0,
            "volume": v,
            "accepted_volume": v,
            "accepted_price": 30.0,
        }
        for t, v in orders
    ]
    unit.set_dispatch_plan(config, book)


def test_acceptance_uses_true_soc():
    """
    The case found in the WP15 runs: day 1 charges in the night, day 2 has its first
    order at 05:00. ASSUME's set_dispatch_plan starts its SOC update at that order and
    reads the stored SOC there. Without the SOC written by calculate_bids it reads the
    old 50 % and cuts the discharge; with it, the full accepted volume is kept.
    """
    from assume.common.market_objects import MarketConfig, MarketProduct

    night = [(DAY1 + timedelta(minutes=15 * i), -0.0137) for i in range(4)]
    morning = [(DAY2 + timedelta(hours=5, minutes=15 * i), 0.0137) for i in range(6)]
    day2 = pd.date_range(DAY2, "2016-06-22 23:45", freq="15min")
    products = [(t, t + timedelta(minutes=15), None) for t in day2]
    config = MarketConfig(
        market_id="LEM_DA",
        market_products=[MarketProduct(timedelta(minutes=15), 96, timedelta(0))],
    )

    # without the workaround: ASSUME alone
    plain = make_battery()
    _commit(plain, "WM_DA", night)
    _commit(plain, "WM_DA", morning)
    cut = sum(plain.outputs["energy"].at[t] for t, _ in morning)

    # with the workaround: the strategy bids for day 2 before the acceptance
    fixed = make_battery()
    _commit(fixed, "WM_DA", night)
    soc_day2 = replay_soc(fixed, DAY2)
    fixed.bidding_strategies["LEM_DA"].calculate_bids(fixed, config, products)
    assert fixed.outputs["soc"].at[DAY2 + timedelta(hours=5)] == pytest.approx(soc_day2)
    _commit(fixed, "WM_DA", morning)
    kept = sum(fixed.outputs["energy"].at[t] for t, _ in morning)

    # the battery is at 90 % after the night; 6 x 13.7 kW discharge fit:
    # (0.9 - 0.1) * 27.4 kWh * 0.95 = 20.8 kWh >= 6 * 13.7 kW * 0.25 h = 20.6 kWh
    assert soc_day2 == pytest.approx(0.9)
    assert kept == pytest.approx(6 * 0.0137)
    assert cut < kept - 1e-6  # ASSUME alone cuts it (old SOC 50 %)
