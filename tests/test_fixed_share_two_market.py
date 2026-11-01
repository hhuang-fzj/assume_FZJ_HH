# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests of FixedShareTwoMarketStrategy (round 2, rule R8).

The numbers are example values of the scenario LocalRetailer_test (model document,
section 6.1 and section 5.9; read at 2016-06-21 12:00 in an earlier version of the
scenario, before the time fixes of 2026-10-27 and 2026-10-29). The tests use them as
constant inputs, so the time base does not matter here:
    Demand_Bus2:       demand_df.csv = 0.007139 MW
    CommunityPV_Bus41: max_power 0.1288 MW, availability_df.csv = 0.595951
    price_LEM_DA:      naive forecast = 25.24 EUR/MWh (WM price of 2016-06-20 12:00)
"""

from datetime import timedelta

import pandas as pd
import pytest

import assume.strategies.fixed_share_two_market as fixed_share
from assume.common.forecaster import DemandForecaster, PowerplantForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies.fixed_share_two_market import FixedShareTwoMarketStrategy
from assume.units.demand import Demand
from assume.units.powerplant import PowerPlant

NOON = pd.Timestamp("2016-06-21 12:00")
NIGHT = pd.Timestamp("2016-06-21 00:00")
DEMAND_MW = 0.007139
PV_MAX_POWER_MW = 0.1288
PV_AVAILABILITY = 0.595951
FORECAST = 25.24


@pytest.fixture
def index():
    return pd.date_range("2016-06-21", "2016-06-22 23:45", freq="15min")


@pytest.fixture
def products():
    day = pd.date_range("2016-06-21", "2016-06-21 23:45", freq="15min")
    return [(t, t + timedelta(minutes=15), None) for t in day]


def market(market_id):
    return MarketConfig(
        market_id=market_id,
        market_products=[MarketProduct(timedelta(minutes=15), 96, timedelta(0))],
    )


@pytest.fixture
def demand_unit(index):
    strategy = FixedShareTwoMarketStrategy()
    forecaster = DemandForecaster(
        index,
        demand=pd.Series(-DEMAND_MW, index=index),  # loader_csv.py: -|demand_df|
        market_prices={"LEM_DA": pd.Series(FORECAST, index=index)},
    )
    return Demand(
        id="Demand_Bus2",
        unit_operator="independent_operator",
        technology="inflex_demand",
        bidding_strategies={"LEM_DA": strategy, "WM_DA": strategy},
        forecaster=forecaster,
        max_power=-0.0079,
        min_power=0,
    )


@pytest.fixture
def pv_unit(index):
    strategy = FixedShareTwoMarketStrategy()
    availability = pd.Series(PV_AVAILABILITY, index=index)
    availability[availability.index.hour < 5] = 0.0  # night: no PV
    forecaster = PowerplantForecaster(
        index,
        availability=availability,
        market_prices={"LEM_DA": pd.Series(FORECAST, index=index)},
    )
    return PowerPlant(
        id="CommunityPV_Bus41",
        unit_operator="LEC_operator",
        technology="solar",
        bidding_strategies={"LEM_DA": strategy, "WM_DA": strategy},
        forecaster=forecaster,
        max_power=PV_MAX_POWER_MW,
        fuel_type="renewable",
        additional_cost=123.1,
    )


def bids_at(unit, market_id, products, time):
    strategy = unit.bidding_strategies[market_id]
    bids = strategy.calculate_bids(unit, market(market_id), products)
    return {b["start_time"]: b for b in bids}.get(time)


def test_demand_split_and_prices(demand_unit, products):
    lem = bids_at(demand_unit, "LEM_DA", products, NOON)
    wm = bids_at(demand_unit, "WM_DA", products, NOON)
    assert lem["volume"] == pytest.approx(-fixed_share.BETA * DEMAND_MW)
    assert lem["price"] == pytest.approx(FORECAST)
    assert wm["volume"] == pytest.approx(-(1 - fixed_share.BETA) * DEMAND_MW)
    assert wm["price"] == fixed_share.WM_BUY_PRICE  # 3000, always accepted
    # the two markets together carry the full volume
    assert lem["volume"] + wm["volume"] == pytest.approx(-DEMAND_MW)


def test_pv_split_and_prices(pv_unit, products):
    full = PV_AVAILABILITY * PV_MAX_POWER_MW  # 0.076758 MW
    lem = bids_at(pv_unit, "LEM_DA", products, NOON)
    wm = bids_at(pv_unit, "WM_DA", products, NOON)
    assert lem["volume"] == pytest.approx(fixed_share.BETA * full)
    assert lem["price"] == pytest.approx(FORECAST)  # not the 123.1 of round 1
    assert wm["volume"] == pytest.approx((1 - fixed_share.BETA) * full)
    assert wm["price"] == fixed_share.WM_SELL_PRICE  # -500, always accepted


def test_pv_no_bid_at_night(pv_unit, products):
    assert bids_at(pv_unit, "LEM_DA", products, NIGHT) is None
    assert bids_at(pv_unit, "WM_DA", products, NIGHT) is None


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_extreme_shares(demand_unit, products, monkeypatch, beta):
    monkeypatch.setattr(fixed_share, "BETA", beta)
    lem = bids_at(demand_unit, "LEM_DA", products, NOON)
    wm = bids_at(demand_unit, "WM_DA", products, NOON)
    if beta == 0.0:
        assert lem is None
        assert wm["volume"] == pytest.approx(-DEMAND_MW)
    else:
        assert lem["volume"] == pytest.approx(-DEMAND_MW)
        assert wm is None


def test_other_market_rejected(demand_unit, products):
    with pytest.raises(ValueError):
        demand_unit.bidding_strategies["LEM_DA"].calculate_bids(
            demand_unit, market("EOM"), products
        )
