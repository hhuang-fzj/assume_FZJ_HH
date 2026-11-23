# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms import PayAsClearRole, clearing_mechanisms

from .utils import create_orderbook, extend_orderbook

simple_dayahead_auction_config = MarketConfig(
    market_id="simple_dayahead_auction",
    market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    additional_fields=["node"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2005, 6, 1),
        until=datetime(2005, 6, 2),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    volume_unit="MW",
    volume_tick=0.1,
    price_unit="€/MW",
    market_mechanism="pay_as_clear",
)


def eligible_test_market(entries, rules=None, **config_values):
    config = copy.copy(simple_dayahead_auction_config)
    config.market_id = "LocalElectricityMarket"
    config.product_type = "energy"
    config.price_unit = "EUR/MWh"
    config.volume_tick = None
    config.param_dict = {"forbidden_trades": rules if rules is not None else [
        {"seller": "Power Grid", "buyer": "CellB"}]}
    for key, value in config_values.items():
        setattr(config, key, value)
    role = PayAsClearEligibleRole(config)
    units = {"Power Grid", "CellB", "Energyhub", "CellA"} | {row[0] for row in entries}
    role.registered_agents = {"operator": [{"id": unit} for unit in units]}
    start = datetime(2024, 1, 1)
    product = (start, start + timedelta(minutes=15), None)
    orders = [{"unit_id": unit, "bid_id": f"bid_{i}", "agent_addr": "operator",
               "start_time": product[0], "end_time": product[1], "only_hours": None,
               "volume": volume, "price": price} for i, (unit, volume, price) in enumerate(entries)]
    return role, orders, product


@pytest.mark.parametrize("entries,expected_volume,expected_price", [
    ([("Power Grid", 1, 50), ("CellB", -1, 180)], 0, 0),
    ([("Power Grid", 1, 50), ("CellA", -1, 3000)], 1, 50),
    # First assigning local supply to CellA must be reversible to serve CellB too.
    ([("Energyhub", 1, 10), ("Power Grid", 1, 100), ("CellA", -1, 3000), ("CellB", -1, 180)], 2, 100),
    ([("Energyhub", 0.5, 10), ("Power Grid", 1, 100), ("CellA", -2, 3000), ("CellB", -1, 180)], 1.5, 100),
    ([("Energyhub", 0.75, 180), ("CellB", -1, 180)], 0.75, 180),
    ([("Energyhub", 1, 181), ("CellB", -1, 180)], 0, 0),
])
def test_eligible_clearing_allocations_and_match_accounting(entries, expected_volume, expected_price):
    role, orders, product = eligible_test_market(entries)
    originals = copy.deepcopy(orders)
    accepted, rejected, meta, flows = role.clear(orders, [product])
    assert len(accepted) + len(rejected) == len(orders)
    assert not flows
    assert meta[0]["supply_volume"] == pytest.approx(expected_volume)
    assert meta[0]["demand_volume_energy"] == pytest.approx(expected_volume * 0.25)
    assert meta[0]["price"] == pytest.approx(expected_price)
    assert sum(o["accepted_volume"] for o in accepted) == pytest.approx(0)
    for match in role.matched_trades:
        assert (match["seller_unit_id"], match["buyer_unit_id"]) != ("Power Grid", "CellB")
        assert match["energy_mwh"] == pytest.approx(match["volume_mw"] * 0.25)
        assert match["price_eur_per_mwh"] == expected_price
        assert match["start_time"] == product[0] and match["end_time"] == product[1]
    for original, order in zip(originals, orders):
        assert order["volume"] == original["volume"] and order["price"] == original["price"]
        side = "seller" if order["volume"] > 0 else "buyer"
        matched = sum(m["volume_mw"] for m in role.matched_trades if m[f"{side}_bid_id"] == order["bid_id"])
        assert abs(order["accepted_volume"]) == pytest.approx(matched)
        if order["accepted_volume"]:
            assert order["price"] <= expected_price if side == "seller" else order["price"] >= expected_price
    # Uniform-price transfers balance between both sides (MW -> quarter-hour MWh).
    assert sum(o["accepted_volume"] * o["accepted_price"] * 0.25 for o in accepted) == pytest.approx(0)


def test_eligible_clearing_reconsiders_incompatible_uniform_prices():
    rules = [{"seller": "Power Grid", "buyer": "CellB"}, {"seller": "Energyhub", "buyer": "CellA"}]
    role, orders, product = eligible_test_market([
        ("Energyhub", 1, 10), ("Power Grid", 1, 100), ("CellA", -1, 3000), ("CellB", -1, 50)], rules)
    accepted, rejected, meta, _ = role.clear(orders, [product])
    assert {o["unit_id"] for o in accepted} == {"Power Grid", "CellA"}
    assert meta[0]["price"] == 100
    assert all(o["accepted_volume"] == 0 for o in rejected)


def test_eligible_clearing_ticks_and_unsupported_product():
    role, orders, product = eligible_test_market([("Energyhub", 3, 20), ("CellB", -5, 40)],
                                                volume_tick=0.1, price_tick=0.5)
    role.clear(orders, [product])
    match = role.matched_trades[0]
    assert match["volume_mw"] == pytest.approx(0.3)
    assert match["energy_mwh"] == pytest.approx(0.075)
    assert match["price_eur_per_mwh"] == 10
    accepted, rejected, meta, _ = role.clear(orders, [])
    assert not accepted and not meta and not role.matched_trades
    assert len(rejected) == 2 and all(o["accepted_volume"] == 0 for o in rejected)


@pytest.mark.parametrize("rules", [None, {}, [{"seller": "Power Grid"}],
                                   [{"seller": "", "buyer": "CellB"}],
                                   [{"seller": "Power Grid", "buyer": "CellB"}] * 2])
def test_eligible_clearing_rejects_malformed_policy(rules):
    config = copy.copy(simple_dayahead_auction_config)
    config.param_dict = {"forbidden_trades": rules}
    with pytest.raises(ValueError, match="forbidden|duplicate"):
        PayAsClearEligibleRole(config)


def test_eligible_clearing_rejects_unknown_identity_and_unsupported_orders():
    role, orders, product = eligible_test_market([("Energyhub", 1, 10), ("CellB", -1, 180)])
    role.registered_agents["operator"] = [{"id": "Energyhub"}, {"id": "CellB"}]
    with pytest.raises(ValueError, match="unregistered.*Power Grid"):
        role.clear(orders, [product])
    role.registered_agents["operator"].append({"id": "Power Grid"})
    for change, message in [({"only_hours": (8, 16)}, "continuous"), ({"volume": {product[0]: 1}}, "scalar"),
                             ({"price": float("nan")}, "finite"), ({"unit_id": "unknown"}, "registered"),
                             ({"agent_addr": "other"}, "submitting agent"), ({"bid_type": "BB"}, "simple")]:
        bad = copy.deepcopy(orders)
        bad[0].update(change)
        with pytest.raises(ValueError, match=message):
            role.clear(bad, [product])
    with pytest.raises(ValueError, match="duplicate order"):
        role.clear(orders + [orders[0].copy()], [product])


def test_eligible_clearing_empty_policy_preserves_standard_behavior():
    role, orders, product = eligible_test_market([("Energyhub", 2, 10), ("CellB", -1, 180)], rules=[])
    ordinary = PayAsClearRole(role.marketconfig)
    assert role.clear(copy.deepcopy(orders), [product]) == ordinary.clear(copy.deepcopy(orders), [product])


def test_eligible_clearing_keeps_intervals_separate_and_clears_round_state():
    role, orders, first = eligible_test_market([("Energyhub", 1, 10), ("CellB", -2, 180)])
    second = (first[1], first[1] + timedelta(minutes=15), None)
    later = copy.deepcopy(orders)
    for order in later:
        order.update(start_time=second[0], end_time=second[1])
    later[0]["price"] = 50
    accepted, _, meta, _ = role.clear(orders + later, [first, second])
    assert [row["price"] for row in meta] == [10, 50]
    assert len(role.matched_trades) == 2
    assert {m["start_time"] for m in role.matched_trades} == {first[0], second[0]}
    assert sum(o["accepted_volume"] for o in accepted) == 0
    role.clear([], [first])
    assert role.matched_trades == []


async def test_eligible_clearing_routes_matches_to_existing_output_agent():
    from types import SimpleNamespace
    from unittest.mock import patch

    role, orders, product = eligible_test_market([("Energyhub", 1, 10), ("CellB", -1, 180)])
    role.clear(orders, [product])
    messages = []

    async def send_message(**kwargs):
        messages.append(kwargs)

    context = SimpleNamespace(data={"output_agent_addr": "local_writer"}, send_message=send_message)
    with patch.object(PayAsClearEligibleRole, "context", property(lambda self: context), create=True):
        await role.store_order_book(orders)
    assert [m["content"]["type"] for m in messages] == ["market_orders", "market_matches"]
    assert messages[1]["receiver_addr"] == "local_writer"
    assert messages[1]["content"]["data"] == role.matched_trades


def test_market():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    print(products)

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 900, price = 50
    """
    orderbook = extend_orderbook(products, -1000, 3000)
    orderbook = extend_orderbook(products, 1000, 100, orderbook)
    orderbook = extend_orderbook(products, 900, 50, orderbook)

    mr = PayAsClearRole(simple_dayahead_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    import pandas as pd

    print(pd.DataFrame(mr.all_orders))
    print(pd.DataFrame(accepted))
    print(meta)


async def test_simple_market_mechanism():
    for name, role in clearing_mechanisms.items():
        skip = False
        for skip_name in ["complex", "nodal", "redispatch", "contract"]:
            if skip_name in name:
                skip = True
        if skip:
            continue

        print(name)
        market_config = copy.copy(simple_dayahead_auction_config)
        market_config.market_mechanism = name
        next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
        products = get_available_products(market_config.market_products, next_opening)
        assert len(products) == 1
        order = {
            "start_time": products[0][0],
            "end_time": products[0][1],
            "only_hours": products[0][2],
        }

        orderbook = create_orderbook(order, node_ids=[0, 1, 2])
        mr = role(simple_dayahead_auction_config)
        accepted, rejected, meta, flows = mr.clear(orderbook, products)
        assert meta[0]["supply_volume"] > 0
        assert meta[0]["price"] > 0
        # import pandas as pd
        # print(pd.DataFrame(mr.all_orders))
        # print(pd.DataFrame(clearing_result))
        # print(meta)

    # return mr.all_orders, meta


def test_market_pay_as_clear():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 900, price = 50
    """
    orderbook = extend_orderbook(products, -400, 3000)
    orderbook = extend_orderbook(products, -100, 3000, orderbook)
    orderbook = extend_orderbook(products, 300, 100, orderbook)
    orderbook = extend_orderbook(products, 200, 50, orderbook)

    mr = PayAsClearRole(simple_dayahead_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 4
    assert len(rejected) == 0
    assert meta[0]["supply_volume"] == 500
    assert meta[0]["demand_volume"] == 500
    assert meta[0]["price"] == 100
    for bid in accepted:
        assert bid["volume"] == bid["accepted_volume"]


def test_market_pay_as_clears_single_demand():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 900, price = 50
    """
    orderbook = extend_orderbook(products, -700, 3000)
    orderbook = extend_orderbook(products, 300, 100, orderbook)
    orderbook = extend_orderbook(products, 200, 50, orderbook)

    mr = PayAsClearRole(simple_dayahead_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 3
    assert len(rejected) == 0
    assert meta[0]["supply_volume"] == 500
    assert meta[0]["demand_volume"] == 500
    assert meta[0]["price"] == 100
    assert accepted[0]["volume"] == -700
    assert accepted[0]["accepted_volume"] == -500


def test_market_pay_as_clears_single_demand_more_generation():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 900, price = 50
    """
    orderbook = extend_orderbook(products, -400, 3000)
    orderbook = extend_orderbook(products, 300, 100, orderbook)
    orderbook = extend_orderbook(products, 200, 50, orderbook)
    orderbook = extend_orderbook(products, 230, 60, orderbook)

    mr = PayAsClearRole(simple_dayahead_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 3
    assert len(rejected) == 1
    assert meta[0]["supply_volume"] == 400
    assert meta[0]["demand_volume"] == 400
    assert meta[0]["price"] == 60
    assert accepted[0]["volume"] == -400
    assert accepted[0]["accepted_volume"] == -400
