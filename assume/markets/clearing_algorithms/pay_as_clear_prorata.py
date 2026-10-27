# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Pay-as-clear with pro-rata allocation of rationed volume at equal prices.

The standard pay-as-clear role of ASSUME breaks ties between orders of equal price with
random numbers. When supply is scarce and several buyers bid the same price, the random
order decides who is served, and the result depends on the order in which the bids
reached the market. This role clears the market in the same way but shares the volume
of a rationed price level proportionally among all orders at that level.

The proportional sharing is adopted by analogy to the curtailment rule of the Single
Day-Ahead Coupling (SDAC): when a bidding zone is curtailed, the curtailed volume is
shared proportionally by all price-taking orders of that zone. SDAC states this rule
for curtailment at the price limits; applying it to any rationed price level is a
modelling choice of this role.

Reference:
    SDAC (2023): SDAC measures in cases of short supply: Curtailment management,
    Second auction, and Peak Load Capacity in SDAC. January 2023, p. 3.
    https://www.epexspot.com/sites/default/files/2025-02/sdac-publication-sdac-measures-in-cases-of-short-supply.pdf
"""

from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketProduct, Orderbook
from assume.markets.clearing_algorithms.simple import PayAsClearRole

# Two volumes (power in MW) closer than this are treated as equal: 1e-6 MW = 1 W
VOLUME_TOLERANCE = 1e-6


def share_level_pro_rata(level: Orderbook) -> None:
    """
    Share the accepted volume of one price level proportionally among its orders.

    Args:
        level (Orderbook): All orders of one product on one side of the market with
            the same price. Their ``accepted_volume`` is changed in place.

    Note:
        The total accepted volume of the level is kept, so the clearing price and the
        volume of the other side of the market do not change. Only the split among
        orders of equal price changes.
    """
    total_volume = sum(order["volume"] for order in level)
    total_accepted = sum(order.get("accepted_volume", 0.0) for order in level)
    if total_volume == 0:
        return
    share = total_accepted / total_volume
    for order in level:
        order["accepted_volume"] = order["volume"] * share


def is_rationed(level: Orderbook) -> bool:
    """
    Tell whether a price level is only partly accepted.

    Args:
        level (Orderbook): All orders of one product and one side with the same price.

    Returns:
        bool: True if some but not all of the volume of the level is accepted, i.e.
        the random tie-breaking of the parent role decided who was served.
    """
    total_volume = abs(sum(order["volume"] for order in level))
    total_accepted = abs(sum(order.get("accepted_volume", 0.0) for order in level))
    return VOLUME_TOLERANCE < total_accepted < total_volume - VOLUME_TOLERANCE


class PayAsClearProRataRole(PayAsClearRole):
    """
    Pay-as-clear market with pro-rata allocation at a rationed price level.

    The clearing price and the accepted volume of each price level are those of the
    parent :class:`PayAsClearRole`. Within a price level that is only partly
    accepted, the accepted volume is shared in proportion to the bid volumes, which
    makes the result independent of the order in which bids arrive.
    """

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> tuple[Orderbook, Orderbook, list[dict], list]:
        """
        Clear the market and share rationed price levels proportionally.

        Args:
            orderbook (Orderbook): Orders of all products of this market.
            market_products (list[MarketProduct]): Products to clear.

        Returns:
            tuple: Accepted orders, rejected orders, clearing meta data and network
            flows, in the same format as :meth:`PayAsClearRole.clear`.
        """
        accepted, rejected, meta, flows = super().clear(orderbook, market_products)

        product_key = itemgetter("start_time", "end_time", "only_hours")
        all_orders = sorted(accepted + rejected, key=product_key)

        for _, product_orders in groupby(all_orders, product_key):
            product_orders = list(product_orders)
            clear_price = next(  
                (o["accepted_price"] for o in product_orders if "accepted_price" in o),
                0.0,
            )  #Get the clearing price of this product. All accepted orders of the product share this price
            for is_demand in (True, False): #Treat demand orders and supply orders separately
                side = [o for o in product_orders if (o["volume"] < 0) == is_demand]
                side.sort(key=itemgetter("price")) #Sort the orders by price and group orders with the same price into one price level
                for _, level in groupby(side, itemgetter("price")):
                    level = list(level)
                    if len(level) > 1 and is_rationed(level): #Act only if the level holds more than one order and is only partly accepted (is_rationed)
                        share_level_pro_rata(level)  #Share the accepted volume of the level among its orders in proportion to their bid volumes
                        for order in level:
                            order["accepted_price"] = clear_price #Set the clearing price on all orders of the level

        # Orders may have moved between accepted and rejected
        accepted = [o for o in all_orders if abs(o["accepted_volume"]) > 0]
        rejected = [o for o in all_orders if abs(o["accepted_volume"]) == 0]
        return accepted, rejected, meta, flows
