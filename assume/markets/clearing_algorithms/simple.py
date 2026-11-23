# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import random
from datetime import timedelta
from itertools import groupby
from math import isclose, isfinite
from numbers import Real
from operator import itemgetter

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)


def calculate_meta(accepted_supply_orders, accepted_demand_orders, product):
    supply_volume = sum(map(itemgetter("accepted_volume"), accepted_supply_orders))
    demand_volume = -sum(map(itemgetter("accepted_volume"), accepted_demand_orders))
    prices = list(map(itemgetter("accepted_price"), accepted_supply_orders)) or [0]
    # can also be self.marketconfig.maximum_bid..?
    duration_hours = (product[1] - product[0]) / timedelta(hours=1)
    avg_price = 0
    if supply_volume:
        weighted_price = [
            order["accepted_volume"] * order["accepted_price"]
            for order in accepted_supply_orders
        ]
        avg_price = sum(weighted_price) / supply_volume
    return {
        "supply_volume": supply_volume,
        "demand_volume": demand_volume,
        "demand_volume_energy": demand_volume * duration_hours,
        "supply_volume_energy": supply_volume * duration_hours,
        "price": avg_price,
        "max_price": max(prices),
        "min_price": min(prices),
        "node": None,
        "product_start": product[0],
        "product_end": product[1],
        "only_hours": product[2],
    }


class PayAsClearRole(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        """
        Performs electricity market clearing using a pay-as-clear mechanism. This means that the clearing price is the
        highest price that is still accepted. The clearing price is the same for all accepted orders.

        Args:
            orderbook (Orderbook): the orders to be cleared as an orderbook
            market_products (list[MarketProduct]): the list of products which are cleared in this clearing

        Returns:
            tuple: accepted orderbook, rejected orderbook and clearing meta data
        """
        market_getter = itemgetter("start_time", "end_time", "only_hours")
        accepted_orders: Orderbook = []
        rejected_orders: Orderbook = []
        clear_price = 0
        meta = []
        orderbook.sort(key=market_getter)
        for product, product_orders in groupby(orderbook, market_getter):
            accepted_demand_orders: Orderbook = []
            accepted_supply_orders: Orderbook = []
            rejected_product_orders: Orderbook = []
            product_orders = list(product_orders)
            if product not in market_products:
                rejected_product_orders.extend(product_orders)
                # logger.debug(f'found unwanted bids for {product} should be {market_products}')
                continue

            supply_orders = [x for x in product_orders if x["volume"] > 0]
            demand_orders = [x for x in product_orders if x["volume"] < 0]
            # volume 0 is ignored/invalid

            # Sort supply orders by price with randomness for tie-breaking
            supply_orders.sort(key=lambda x: (x["price"], random.random()))

            # Sort demand orders by price in descending order with randomness for tie-breaking
            demand_orders.sort(
                key=lambda x: (x["price"], random.random()), reverse=True
            )

            dem_vol, gen_vol = 0, 0
            # the following algorithm is inspired by one bar for generation and one for demand
            # add generation for currents demand price, until it matches demand
            # generation above it has to be sold for the lower price (or not at all)
            for demand_order in demand_orders:
                if not supply_orders:
                    # if no more generation - continue
                    # reject left over demand at the end
                    continue

                # assert dem_vol == gen_vol
                # now add the next demand order
                dem_vol += -demand_order["volume"]
                demand_order["accepted_volume"] = demand_order["volume"]
                # and add supply until the demand order is matched
                while supply_orders and gen_vol < dem_vol:
                    supply_order = supply_orders.pop(0)
                    if supply_order["price"] <= demand_order["price"]:
                        added = supply_order["volume"] - supply_order.get(
                            "accepted_volume", 0
                        )
                        should_insert = not supply_order.get("accepted_volume")#the order should be inserted into orderbook only if it is not accepted before
                        supply_order["accepted_volume"] = supply_order["volume"]#Update the
                        if should_insert:
                            accepted_supply_orders.append(supply_order)
                        gen_vol += added
                    # if supply is not partially accepted before, reject it
                    elif not supply_order.get("accepted_volume"):
                        rejected_product_orders.append(supply_order)
                # now we know which orders we need
                # we only need to see how to arrange it.

                diff = gen_vol - dem_vol

                if diff < 0:
                    # gen < dem
                    # generation is not enough - accept partially
                    demand_order["accepted_volume"] = demand_order["volume"] - diff
                elif diff > 0:
                    # generation left over - accept generation bid partially
                    supply_order = accepted_supply_orders[-1]
                    supply_order["accepted_volume"] = supply_order["volume"] - diff

                    # changed supply_order is still part of to_commit and will be added
                    # only volume-diff can be sold for current price
                    gen_vol -= diff

                    # add left over to supply_orders again
                    supply_orders.insert(0, supply_order)
                    demand_order["accepted_volume"] = demand_order["volume"]
                else:
                    demand_order["accepted_volume"] = demand_order["volume"]

                if demand_order["accepted_volume"]:
                    accepted_demand_orders.append(demand_order)

            # if demand is fulfilled, we do have some additional supply orders
            # these will be rejected
            for order in product_orders:
                # if the order was not accepted partially, it is rejected
                if (
                    not order.get("accepted_volume")
                    and order not in rejected_product_orders
                ):
                    rejected_product_orders.append(order)

            # set clearing price - merit order - uniform pricing
            if accepted_supply_orders:
                clear_price = float(
                    max(map(itemgetter("price"), accepted_supply_orders))
                )
            else:
                clear_price = 0

            accepted_product_orders = accepted_demand_orders + accepted_supply_orders
            for order in accepted_product_orders:
                order["accepted_price"] = clear_price
            accepted_orders.extend(accepted_product_orders)

            # set accepted volume to 0 and price to clear price for rejected orders
            for order in rejected_product_orders:
                order["accepted_volume"] = 0
                order["accepted_price"] = clear_price
            rejected_orders.extend(rejected_product_orders)

            meta.append(
                calculate_meta(
                    accepted_supply_orders,
                    accepted_demand_orders,
                    product,
                )
            )

        # write network flows here if applicable
        flows = []

        return accepted_orders, rejected_orders, meta, flows


class PayAsClearEligibleRole(PayAsClearRole):
    """Uniform-price clearing with explicit direct seller/buyer exclusions.

    At each candidate supply price, find the greatest bid surplus, then traded
    volume. Residual paths can reassign earlier matches instead of starving an
    eligible buyer. Compare candidates to enforce one price within every
    accepted order's limit. Ties retain the lowest candidate price and input
    order. This is counterparty eligibility, not provenance through resellers.

    Empty policies delegate to ordinary pay_as_clear, including its tie policy.
    Restricted clearing supports scalar, continuous-delivery MW orders only.
    """

    _epsilon = 1e-9

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)
        rules = marketconfig.param_dict.get("forbidden_trades", [])
        if not isinstance(rules, list):
            raise ValueError(f"{marketconfig.market_id}: forbidden_trades must be a list")
        self.forbidden_trades = set()
        for rule in rules:
            if (
                not isinstance(rule, dict)
                or set(rule) != {"seller", "buyer"}
                or any(not isinstance(v, str) or not v.strip() for v in rule.values())
            ):
                raise ValueError(
                    f"{marketconfig.market_id}: invalid forbidden_trades entry "
                    f"{rule!r}; use seller/buyer unit IDs"
                )
            pair = (rule["seller"], rule["buyer"])
            if pair in self.forbidden_trades:
                raise ValueError(f"{marketconfig.market_id}: duplicate forbidden trade {pair!r}")
            self.forbidden_trades.add(pair)
        self.matched_trades = []

        # Match records assume MW, MWh and EUR/MWh. Check the declared units;
        # this cannot detect incorrectly scaled numerical inputs.
        # Electricity-only support is an implementation scope, not a flow constraint.
        if self.forbidden_trades and (
            marketconfig.volume_unit != "MW" or marketconfig.product_type != "energy"
        ):
            raise ValueError(
                f"{marketconfig.market_id}: eligible clearing requires energy products in MW"
            )
        if self.forbidden_trades and marketconfig.price_unit not in {"EUR/MWh", "€/MWh"}:
            raise ValueError(f"{marketconfig.market_id}: eligible clearing requires prices in EUR/MWh")

    def _validate_participants(self, orderbook):
        # Currently every unit named in a restriction must remain registered,
        # even if it submits no bid. Unit identity is separate from agent identity.
        # TODO: Distinguish invalid configured unit IDs from temporarily absent
        # participants (e.g. an EV that deregisters and later returns). Validate
        # rule names against the scenario inventory where available, while allowing
        # valid restrictions to remain inactive during absence. Keep rejecting
        # submitted orders from unregistered units. Defer this lifecycle change;
        # the strict missing-unit error below remains in effect for now.
        by_agent = {
            agent: {unit["id"] for unit in units}
            for agent, units in self.registered_agents.items()
        }
        known = {unit for units in by_agent.values() for unit in units}
        required = {unit for pair in self.forbidden_trades for unit in pair}
        missing = required - known
        if missing:
            raise ValueError(
                f"{self.marketconfig.market_id}: forbidden_trades references "
                f"unregistered unit IDs {sorted(missing)!r}"
            )
        identities = set()
        for order in orderbook:
            # MarketRole checks the submitting agent; additionally verify the
            # unit's registration, ownership and bid identity for pair restrictions.
            unit, bid = order.get("unit_id"), order.get("bid_id")
            if unit not in known or not isinstance(bid, str) or not bid:
                raise ValueError(
                    f"{self.marketconfig.market_id}: orders require registered unit_id "
                    f"and nonempty bid_id; got {unit!r}, {bid!r}"
                )
            if "agent_addr" in order and unit not in by_agent.get(order["agent_addr"], set()):
                raise ValueError(
                    f"{self.marketconfig.market_id}: unit {unit!r} is not "
                    "registered to the submitting agent"
                )
            identity = (unit, bid, order["start_time"], order["end_time"])
            if identity in identities:
                raise ValueError(f"{self.marketconfig.market_id}: duplicate order identity {identity!r}")
            identities.add(identity)
            if (
                order.get("only_hours") is not None
                or order["end_time"] <= order["start_time"]
                # Advanced-order dictionary keys used by advanced_orders.py and
                # complex_clearing.py: SB = simple, BB = block, LB = linked bid.
                # This matcher cannot enforce minimum acceptance or parent links.
                or order.get("bid_type", "SB") != "SB"
                or order.get("min_acceptance_ratio") not in (None, 0)
                or order.get("parent_bid_id") is not None
            ):
                raise ValueError(
                    f"{self.marketconfig.market_id}: eligible clearing supports "
                    f"simple continuous-delivery orders only ({identity!r})"
                )
            for key in ("volume", "price"):
                if (
                    isinstance(order[key], bool)
                    or not isinstance(order[key], Real)
                    or not isfinite(order[key])
                ):
                    raise ValueError(
                        f"{self.marketconfig.market_id}: {identity!r}: "
                        f"{key} must be a finite scalar"
                    )

    def _match_at_price(self, sellers, buyers, price):
        """Successive shortest residual paths; no optimization by solver.

        Path cost is ask minus bid. Negative paths increase surplus; zero-cost
        paths increase volume without sacrificing surplus. Reverse edges allow
        reassignment. Stop before any path would reduce total surplus.
        """
        source, sink = len(sellers) + len(buyers), len(sellers) + len(buyers) + 1
        graph = [[] for _ in range(sink + 1)]

        def edge(a, b, capacity, cost):
            forward = [b, len(graph[b]), capacity, cost]
            reverse = [a, len(graph[a]), 0.0, -cost]
            graph[a].append(forward)
            graph[b].append(reverse)
            return forward, reverse

        for i, seller in enumerate(sellers):
            if seller["price"] <= price:
                edge(source, i, seller["volume"], seller["price"])
        for j, buyer in enumerate(buyers):
            if buyer["price"] >= price:
                edge(len(sellers) + j, sink, -buyer["volume"], -buyer["price"])
        pairs = []
        for i, seller in enumerate(sellers):
            for j, buyer in enumerate(buyers):
                if (
                    seller["price"] <= price <= buyer["price"]
                    and (seller["unit_id"], buyer["unit_id"]) not in self.forbidden_trades
                ):
                    _, reverse = edge(
                        i, len(sellers) + j,
                        min(seller["volume"], -buyer["volume"]), 0.0,
                    )
                    pairs.append((i, j, reverse))
        while True:
            distance = [float("inf")] * len(graph)
            previous = [None] * len(graph)
            distance[source] = 0.0
            # Bellman-Ford permits negative bid-surplus edges and reverse paths.
            for _ in range(len(graph) - 1):
                changed = False
                for a, edges in enumerate(graph):
                    for k, (b, _, capacity, cost) in enumerate(edges):
                        if (
                            capacity > self._epsilon
                            and distance[a] + cost < distance[b] - self._epsilon
                        ):
                            distance[b] = distance[a] + cost
                            previous[b] = (a, k)
                            changed = True
                if not changed:
                    break
            if previous[sink] is None or distance[sink] > self._epsilon:
                break
            path, node = [], sink
            while node != source:
                a, k = previous[node]
                path.append((a, k))
                node = a
                if len(path) >= len(graph):
                    raise RuntimeError("Eligible clearing encountered a cyclic residual path")
            amount = min(graph[a][k][2] for a, k in path)
            for a, k in path:
                b, reverse, _, _ = graph[a][k]
                graph[a][k][2] -= amount
                graph[b][reverse][2] += amount
        return [(i, j, reverse[2]) for i, j, reverse in pairs if reverse[2] > self._epsilon]

    def clear(self, orderbook: Orderbook, market_products):
        self.matched_trades = []
        if not self.forbidden_trades:
            return super().clear(orderbook, market_products)
        self._validate_participants(orderbook)
        accepted, rejected, meta = [], [], []
        getter = itemgetter("start_time", "end_time", "only_hours")
        for product, orders in groupby(sorted(orderbook, key=getter), getter):
            orders = list(orders)
            if product not in market_products:
                for order in orders:
                    order.update(accepted_volume=0.0, accepted_price=0.0)
                rejected.extend(orders)
                continue
            sellers = [o for o in orders if o["volume"] > 0]
            buyers = [o for o in orders if o["volume"] < 0]
            best, best_surplus, best_volume = [], 0.0, 0.0
            for price in sorted({o["price"] for o in sellers}):
                matches = self._match_at_price(sellers, buyers, price)
                surplus = sum(
                    v * (buyers[j]["price"] - sellers[i]["price"])
                    for i, j, v in matches
                )
                volume = sum(v for _, _, v in matches)
                if (
                    surplus > best_surplus + self._epsilon
                    or (
                        isclose(surplus, best_surplus, rel_tol=0, abs_tol=self._epsilon)
                        and volume > best_volume + self._epsilon
                    )
                ):
                    best, best_surplus, best_volume = matches, surplus, volume
            clearing_price = max((sellers[i]["price"] for i, _, _ in best), default=0.0)
            for order in orders:
                order.update(accepted_volume=0.0, accepted_price=clearing_price)
            duration = (product[1] - product[0]) / timedelta(hours=1)
            volume_scale = self.marketconfig.volume_tick or 1.0
            price_scale = self.marketconfig.price_tick or 1.0
            for i, j, volume in best:
                seller, buyer = sellers[i], buyers[j]
                seller["accepted_volume"] += volume
                buyer["accepted_volume"] -= volume
                self.matched_trades.append(
                    {
                        "market_id": self.marketconfig.market_id,
                        "start_time": product[0], "end_time": product[1],
                        "seller_unit_id": seller["unit_id"],
                        "buyer_unit_id": buyer["unit_id"],
                        "seller_bid_id": seller["bid_id"],
                        "buyer_bid_id": buyer["bid_id"],
                        "volume_mw": volume * volume_scale,
                        "energy_mwh": volume * volume_scale * duration,
                        "duration_hours": duration,
                        "price_eur_per_mwh": clearing_price * price_scale,
                    }
                )
            supply = [o for o in sellers if o["accepted_volume"] > self._epsilon]
            demand = [o for o in buyers if o["accepted_volume"] < -self._epsilon]
            accepted.extend(supply + demand)
            rejected.extend(o for o in orders if abs(o["accepted_volume"]) <= self._epsilon)
            meta.append(calculate_meta(supply, demand, product))
        return accepted, rejected, meta, []

    async def store_order_book(self, orderbook: Orderbook):
        # Other clearing roles inherit market_orders storage from MarketRole.
        # Preserve it and additionally send bilateral market_matches records.
        await super().store_order_book(orderbook)
        db_addr = self.context.data.get("output_agent_addr")
        if db_addr and self.matched_trades:
            await self.context.send_message(
                receiver_addr=db_addr,
                content={
                    "context": "write_results", "type": "market_matches",
                    "market_id": self.marketconfig.market_id,
                    "data": self.matched_trades,
                },
            )


class PayAsBidRole(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        """
        Simulates electricity market clearing using a pay-as-bid mechanism.

        Args:
            orderbook (Orderbook): the orders to be cleared as an orderbook
            market_products (list[MarketProduct]): the list of products which are cleared in this clearing

        Returns:
            tuple[Orderbook, Orderbook, list[dict]]: accepted orderbook, rejected orderbook and clearing meta data
        """

        market_getter = itemgetter("start_time", "end_time", "only_hours")
        accepted_orders: Orderbook = []
        rejected_orders: Orderbook = []
        meta = []
        orderbook.sort(key=market_getter)
        for product, product_orders in groupby(orderbook, market_getter):
            accepted_demand_orders: Orderbook = []
            accepted_supply_orders: Orderbook = []
            if product not in market_products:
                rejected_orders.extend(product_orders)
                # logger.debug(f'found unwanted bids for {product} should be {market_products}')
                continue

            product_orders = list(product_orders)
            supply_orders = [x for x in product_orders if x["volume"] > 0]
            demand_orders = [x for x in product_orders if x["volume"] < 0]
            # volume 0 is ignored/invalid

            # Sort supply orders by price with randomness for tie-breaking
            supply_orders.sort(key=lambda i: (i["price"], random.random()))
            # Sort demand orders by price in descending order with randomness for tie-breaking
            demand_orders.sort(
                key=lambda i: (i["price"], random.random()), reverse=True
            )

            dem_vol, gen_vol = 0, 0
            # the following algorithm is inspired by one bar for generation and one for demand
            # add generation for currents demand price, until it matches demand
            # generation above it has to be sold for the lower price (or not at all)
            for demand_order in demand_orders:
                if not supply_orders:
                    # if no more generation - continue
                    # reject left over demand at the end
                    continue

                dem_vol += -demand_order["volume"]
                to_commit: Orderbook = []

                while supply_orders and gen_vol < dem_vol:
                    supply_order = supply_orders.pop(0)
                    if supply_order["price"] <= demand_order["price"]:
                        supply_order["accepted_volume"] = supply_order["volume"]
                        to_commit.append(supply_order)
                        gen_vol += supply_order["volume"]
                    # if supply is not partially accepted before, reject it
                    elif not supply_order.get("accepted_volume"):
                        rejected_orders.append(supply_order)
                # now we know which orders we need
                # we only need to see how to arrange it.

                diff = gen_vol - dem_vol

                if diff < 0:
                    # gen < dem
                    demand_order["accepted_volume"] = demand_order["volume"] - diff
                elif diff > 0:
                    # generation left over - split generation
                    supply_order = to_commit[-1]
                    split_supply_order = supply_order.copy()
                    split_supply_order["volume"] = diff
                    supply_order["accepted_volume"] = supply_order["volume"] - diff
                    # only volume-diff can be sold for current price
                    # add left over to supply_orders again
                    gen_vol -= diff

                    supply_orders.insert(0, split_supply_order)
                    demand_order["accepted_volume"] = demand_order["volume"]
                else:
                    # diff == 0 perfect match
                    demand_order["accepted_volume"] = demand_order["volume"]

                if demand_order["accepted_volume"]:
                    accepted_demand_orders.append(demand_order)
                # pay as bid
                for supply_order in to_commit:
                    supply_order["accepted_price"] = supply_order["price"]

                    demand_order["accepted_price"] = supply_order["price"]
                accepted_supply_orders.extend(to_commit)

            # if demand is fulfilled, we do have some additional supply orders
            # these will be rejected
            for order in product_orders:
                # if the order was not accepted partially, it is rejected
                if not order.get("accepted_volume") and order not in rejected_orders:
                    rejected_orders.append(order)

            accepted_product_orders = accepted_demand_orders + accepted_supply_orders

            accepted_orders.extend(accepted_product_orders)
            meta.append(
                calculate_meta(
                    accepted_supply_orders,
                    accepted_demand_orders,
                    product,
                )
            )

        # write network flows here if applicable
        flows = []

        return accepted_orders, rejected_orders, meta, flows
