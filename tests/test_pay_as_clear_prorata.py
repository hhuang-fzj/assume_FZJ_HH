# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the pay-as-clear market with pro-rata allocation at rationed price levels.

The tests compare :class:`PayAsClearProRataRole` with the standard
:class:`PayAsClearRole` on small hand-made orderbooks, so the expected allocation can
be checked by hand.
"""

import copy
import random
from datetime import datetime, timedelta

import pytest
from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms import (
    PayAsClearProRataRole,
    PayAsClearRole,
    clearing_mechanisms,
)

from .utils import extend_orderbook

config = MarketConfig(
    market_id="lem",
    market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    additional_fields=["node"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2016, 6, 1),
        until=datetime(2016, 6, 2),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    volume_unit="MWh",
    price_unit="EUR/MWh",
    market_mechanism="pay_as_clear_prorata",
)


@pytest.fixture
def products():
    """One hourly product of the test market."""
    next_opening = config.opening_hours.after(datetime(2016, 6, 1))
    return get_available_products(config.market_products, next_opening)


def clear(role_class, orderbook, products):
    """Clear a deep copy of the orderbook, so both roles see the same input."""
    return role_class(config).clear(copy.deepcopy(orderbook), products)


def accepted_by_bid(accepted):
    """Map each bid id to its accepted volume."""
    return {order["bid_id"]: order["accepted_volume"] for order in accepted}


def label(orderbook):
    """Give every order a readable bid id, e.g. 'o0', 'o1', ..."""
    for number, order in enumerate(orderbook):
        order["bid_id"] = f"o{number}"
    return orderbook


def test_registered():
    """The role is available under its market mechanism name."""
    assert clearing_mechanisms["pay_as_clear_prorata"] is PayAsClearProRataRole


def test_return_format(products):
    """The role returns the same four values as the parent role."""
    orderbook = extend_orderbook(products, -10, 288)
    orderbook = extend_orderbook(products, 10, 123, orderbook)
    result = clear(PayAsClearProRataRole, label(orderbook), products)
    assert len(result) == 4


def test_no_tie_equals_parent(products):
    """Without equal prices at a rationed level, the result equals the parent role."""
    orderbook = extend_orderbook(products, -30, 288)
    orderbook = extend_orderbook(products, -20, 250, orderbook)
    orderbook = extend_orderbook(products, 40, 123, orderbook)
    orderbook = label(orderbook)

    parent = clear(PayAsClearRole, orderbook, products)
    prorata = clear(PayAsClearProRataRole, orderbook, products)

    assert accepted_by_bid(prorata[0]) == pytest.approx(accepted_by_bid(parent[0]))
    assert prorata[2][0]["price"] == parent[2][0]["price"]


def test_demand_tie_shared_pro_rata(products):
    """
    Two buyers at the same price share scarce supply in proportion to their volume.

    Supply 10, buyers want 20 and 30 at the same price: they receive 4 and 6.
    """
    orderbook = extend_orderbook(products, -20, 288)
    orderbook = extend_orderbook(products, -30, 288, orderbook)
    orderbook = extend_orderbook(products, 10, 123, orderbook)
    accepted, rejected, meta, _ = clear(
        PayAsClearProRataRole, label(orderbook), products
    )

    volumes = accepted_by_bid(accepted)
    assert volumes["o0"] == pytest.approx(-4)
    assert volumes["o1"] == pytest.approx(-6)
    assert volumes["o2"] == pytest.approx(10)
    assert rejected == []
    assert meta[0]["price"] == 123


def test_supply_tie_shared_pro_rata(products):
    """
    Two sellers at the same price share limited demand in proportion to their volume.

    Demand 10, sellers offer 10 and 30 at the same price: they sell 2.5 and 7.5.
    """
    orderbook = extend_orderbook(products, -10, 288)
    orderbook = extend_orderbook(products, 10, 123, orderbook)
    orderbook = extend_orderbook(products, 30, 123, orderbook)
    accepted, _, _, _ = clear(PayAsClearProRataRole, label(orderbook), products)

    volumes = accepted_by_bid(accepted)
    assert volumes["o1"] == pytest.approx(2.5)
    assert volumes["o2"] == pytest.approx(7.5)


def test_totals_and_price_equal_parent(products):
    """Pro-rata allocation changes the split, never the total volume or the price."""
    orderbook = extend_orderbook(products, -20, 288)
    orderbook = extend_orderbook(products, -30, 288, orderbook)
    orderbook = extend_orderbook(products, -5, 100, orderbook)
    orderbook = extend_orderbook(products, 10, 123, orderbook)
    orderbook = label(orderbook)

    parent = clear(PayAsClearRole, orderbook, products)
    prorata = clear(PayAsClearProRataRole, orderbook, products)

    def total(accepted, sign):
        return sum(o["accepted_volume"] for o in accepted if o["volume"] * sign > 0)

    assert total(prorata[0], -1) == pytest.approx(total(parent[0], -1))
    assert total(prorata[0], 1) == pytest.approx(total(parent[0], 1))
    assert prorata[2][0]["price"] == parent[2][0]["price"]


def test_rejected_order_moves_to_accepted(products):
    """
    A buyer rejected by the random tie-break is accepted after pro-rata allocation.

    Supply 10 exactly covers the first buyer served by the parent role, so the other
    buyer at the same price is rejected there. With pro-rata both receive a share and
    both carry the clearing price.
    """
    orderbook = extend_orderbook(products, -10, 288)
    orderbook = extend_orderbook(products, -10, 288, orderbook)
    orderbook = extend_orderbook(products, 10, 123, orderbook)
    accepted, rejected, _, _ = clear(PayAsClearProRataRole, label(orderbook), products)

    buyers = [o for o in accepted if o["volume"] < 0]
    assert len(buyers) == 2
    assert all(o["accepted_volume"] == pytest.approx(-5) for o in buyers)
    assert all(o["accepted_price"] == 123 for o in buyers)
    assert rejected == []


def test_independent_of_order_and_seed(products):
    """The allocation does not depend on the order of bids or on the random seed."""
    orderbook = extend_orderbook(products, -7, 288)
    orderbook = extend_orderbook(products, -11, 288, orderbook)
    orderbook = extend_orderbook(products, -13, 288, orderbook)
    orderbook = extend_orderbook(products, 9, 123, orderbook)
    orderbook = label(orderbook)

    results = []
    for seed in range(5):
        random.seed(seed)
        shuffled = copy.deepcopy(orderbook)
        random.shuffle(shuffled)
        accepted, _, _, _ = clear(PayAsClearProRataRole, shuffled, products)
        results.append(accepted_by_bid(accepted))

    for result in results[1:]:
        assert result == pytest.approx(results[0])
    share = 9 / 31
    assert results[0]["o0"] == pytest.approx(-7 * share)
    assert results[0]["o1"] == pytest.approx(-11 * share)
    assert results[0]["o2"] == pytest.approx(-13 * share)
