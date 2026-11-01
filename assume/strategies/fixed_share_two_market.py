# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Fixed-share strategy for two parallel day-ahead markets (round 2 of the thesis).

Used by the participants that are not optimized: the independent demand nodes
(``Demand``) and the community PV plant (``PowerPlant``). Each of them splits its
volume with a fixed share BETA between the two markets (rule R8, decision D1):

    LEM_DA: volume = BETA * full volume,      price = naive forecast (rule R6, R7)
    WM_DA:  volume = (1 - BETA) * full volume, price = WM price limit, so the order
            is always accepted (decisions D3, D4)

        buyer  (Demand, volume < 0): WM price = 3000 (WM price cap)
        seller (PowerPlant, volume > 0): WM price = -500 (WM price floor)

Full volume of one product = the same volume the naive strategy of round 1 bids
(``EnergyNaiveStrategy``, naive_strategies.py lines 43-45, 60-62):
    Demand:     forecaster.demand - already committed volume
                (demand.py, calculate_min_max_power, lines 151-156)
    PowerPlant: availability * max_power - already committed volume
                (powerplant.py, calculate_min_max_power, lines 318-330)
Both markets open at the same time (config.yaml, opening_frequency 24h, same
start_date), so nothing is committed yet when the bids of both markets are made and
LEM volume + WM volume = full volume.

Volumes follow the ASSUME convention: sell > 0, buy < 0, in MW.

Data path (where every input of this strategy comes from):
    Line numbers refer to branch Yihan_local_aggregator (checked 2026-10-26); search
    the quoted code if the lines have moved.

    Which units use it:
        round2_two_market_participants.py writes the strategy name into the columns
        bidding_LEM_DA and bidding_WM_DA of
            demand_units.csv      (rows with unit_operator = independent_operator)
            powerplant_units.csv  (rows with unit_operator = LEC_operator)
        -> loader_csv.py, read_units, lines 478-484: column bidding_<market> ->
           strategy per market
        -> assume/strategies/__init__.py: names "demand_two_market_fixed_share" and
           "powerplant_two_market_fixed_share" -> this class
        -> units_operator.py, participate, lines 154-178: because the unit now has a
           WM_DA strategy, its operator also registers at WM_DA
        -> the operators have no portfolio strategy, so UnitsOperatorDirectStrategy
           (portfolio_strategies.py, lines 73-111) calls this class for each unit.

    Full volume:
        Demand_BusX: Load.csv + LoadProfile.csv -> simbench_to_assume.py
            build_node_load (173-198) -> demand_df.csv -> loader_csv.py 717-726
            -> forecaster.demand (negative)                   (doc 5.9, chain 5)
        CommunityPV_Bus41: RES.csv pRES -> powerplant_units.csv max_power;
            RESProfile.csv PV5 -> availability_df.csv -> loader_csv.py 707-716
            -> forecaster.availability                         (doc 5.9, chain 6)

    LEM price (naive similar-day forecast of the WM price):
        SMARD WM price 2016 -> round2_add_batteries.py, naive_forecast()
        -> column price_LEM_DA of forecasts_df.csv
        -> loader_csv.py 651 -> world.py init_forecasts 942-957
        -> forecaster.py line 71: column price_LEM_DA = price forecast of LEM_DA
        -> here: unit.forecaster.price["LEM_DA"].
        Without this column ASSUME computes its own LEM forecast instead
        (price_naive_forecast), so round2_two_market_participants.py checks that the
        column exists.

    BETA, WM_SELL_PRICE, WM_BUY_PRICE: imported from storage_two_market.py, so all
    participants other than the aggregator use the same share (decision D1).

Sources:
    Fixed share per market: todo 4 agreed with the supervisor; value of BETA [OWN]
        (decision D1, scenarios 0 / 0.5 / 1).
    Bid at own opportunity cost: Mello, J., de Lorenzo, C., Campos, F. A., Villar, J.
        (2023). Pricing and Simulating Energy Transactions in Energy Communities.
        Energies 16(4), 1949, Sect. 3.1, p. 8 (rule R6).
    Naive forecast: Weron, R. (2014). Electricity price forecasting: A review of the
        state-of-the-art with a look into the future. International Journal of
        Forecasting 30(4), 1030-1081, Sect. 3.8.1, p. 1049-1050 (rule R7).
    WM sell at -500: Qussous, R., Harder, N., Weidlich, A. (2022). Understanding Power
        Market Dynamics by Reflecting Market Interrelations and Flexibility-Oriented
        Bidding Strategies. Energies 15(2), 494, Sect. 3.2.1, p. 9 (decision D4).
    Not included: a market premium of the community PV under direct marketing
        (EEG 2014 clauses not yet checked).

Known limit: while the LEM price limits are 123.1 / 288 (config.yaml, round 1), the
LEM market clips these bids (base_market.py, lines 374-383): a forecast of 30 EUR/MWh
becomes 123.1. The LEM results are only meaningful after decision F4.
"""

import numpy as np

from assume.common.base import MinMaxStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.strategies.storage_two_market import BETA, WM_BUY_PRICE, WM_SELL_PRICE

LEM_MARKET_ID = "LEM_DA"
WM_MARKET_ID = "WM_DA"
# Market whose forecast column holds the naive WM forecast (price_LEM_DA)
FORECAST_MARKET_ID = "LEM_DA"


class FixedShareTwoMarketStrategy(MinMaxStrategy):
    """
    Bid the share BETA of the volume in the LEM at the naive forecast and the share
    1 - BETA in the WM at the WM price limit.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Bids of one unit for all products of one market opening (one day).

        Args:
            unit (SupportsMinMax): A Demand or PowerPlant unit.
            market_config (MarketConfig): LEM_DA or WM_DA.
            product_tuples (list[Product]): The 96 products of the delivery day.

        Returns:
            Orderbook: One bid per product with price and volume (MW).
        """
        market_id = market_config.market_id
        if market_id == LEM_MARKET_ID:
            share = BETA
        elif market_id == WM_MARKET_ID:
            share = 1 - BETA
        else:
            raise ValueError(f"{type(self).__name__} only bids on LEM_DA and WM_DA")
        if share == 0:
            return []

        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        # Full volume per product, same function as the naive strategy of round 1
        # (Demand: buy < 0; PowerPlant: sell > 0); see "Full volume" above
        _, full_volumes = unit.calculate_min_max_power(start, end_all)
        full_volumes = np.asarray(full_volumes, dtype=float)

        if market_id == LEM_MARKET_ID:
            # Naive WM forecast, column price_LEM_DA of forecasts_df.csv
            forecast = unit.forecaster.price[FORECAST_MARKET_ID]

        bids = []
        for product, full_volume in zip(product_tuples, full_volumes):
            if market_id == LEM_MARKET_ID:
                price = float(forecast[product[0]])
            else:
                price = WM_BUY_PRICE if full_volume < 0 else WM_SELL_PRICE
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": price,
                    "volume": share * full_volume,
                    "node": unit.node,  # default "node0" (base.py, line 47), no grid model
                }
            )

        # orders with volume 0 (e.g. PV at night) are not sent
        return self.remove_empty_bids(bids)
