# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Battery strategy for two parallel day-ahead markets (round 2 of the thesis).

The strategy is the flexABLE storage rule of ASSUME
(:class:`StorageEnergyHeuristicFlexableStrategy`) with three changes:

1. Price forecast: the naive similar-day forecast of the WM price, read from the
   column ``price_LEM_DA`` of ``forecasts_df.csv`` (written by
   ``round2_add_batteries.py``). The flexABLE rule uses the forecast of the market
   that is bid on; here both markets use the same forecast (rule R7).
2. Reference price: mean forecast price of the delivery day (the 96 products of one
   opening) instead of a window of +/- 12 h around each product (decision E4).
3. Two markets: the battery bids the share BETA of its volume in the LEM and the
   share 1 - BETA in the WM (rule R8). The WM part is bid at the WM price limits, so
   it is always accepted (research design). The planned state of charge follows
   the full volume of both markets.

Volumes follow the ASSUME convention: discharge (sell) > 0, charge (buy) < 0, in MW.

Data path (where every input of this strategy comes from):
    Line numbers refer to branch Yihan_local_aggregator (checked 2026-10-26); search
    the quoted code if the lines have moved.

    Battery parameters, one row per battery:
        SimBench simbench_raw/Storage.csv (columns sR, eStore, etaStore)
        -> round2_add_batteries.py writes storage_units.csv (columns max_power_charge,
           max_power_discharge, capacity, efficiency_charge, efficiency_discharge,
           min_soc, max_soc, initial_soc, bidding_LEM_DA, bidding_WM_DA)
        -> assume/scenario/loader_csv.py, line 620: storage_units.csv is read;
           lines 636-640: max_power_charge is made negative (charging < 0)
        -> loader_csv.py, read_units, lines 478-505: each row becomes a dict
           ``unit_params``; the columns bidding_<market> select this strategy
        -> assume/world.py, lines 524-528: ``unit_class(**unit_params)`` calls the
           constructor of assume/units/storage.py (class Storage, lines 56-75)
        -> storage.py stores the values as unit attributes:
           unit.capacity, unit.max_soc, unit.min_soc (lines 125-127),
           unit.initial_soc (line 136), unit.efficiency_charge,
           unit.efficiency_discharge (lines 201-202).
        So no battery parameter is set in this file; change them in the input data.

    Price forecast:
        SMARD WM price 2016 (file of simbench_to_assume.py, WM_PRICE_FILE)
        -> round2_add_batteries.py writes the naive forecast into the column
           price_LEM_DA of forecasts_df.csv
        -> loader_csv.py, lines 651-652: forecasts_df.csv is read
        -> world.py, init_forecasts, lines 942-957: every unit forecaster gets it
        -> assume/common/forecaster.py, calculate_base_forecasts, line 71:
           ``forecast_df.get(f"{prefix}_{market_id}")`` takes the column
           price_LEM_DA as the price forecast of market LEM_DA
        -> here: unit.forecaster.price["LEM_DA"].

    Volume limits (unchanged ASSUME functions, called below):
        unit.calculate_min_max_charge / _discharge (storage.py, lines 411-497):
            power limit minus volume already sold for this period, then limited by
            the SOC through calculate_soc_max_charge / _discharge (lines 369-409):
            discharge <= (SOC - min_soc) * capacity * efficiency_discharge / dt,
            charge    <= (max_soc - SOC) * capacity / efficiency_charge / dt
            (Qussous et al. 2022, Eq. (20)-(21)).
        unit.calculate_ramp_charge / _discharge (storage.py, lines 499-569): ramp
            limits; no ramp columns are given, so they do not restrict here.

    Settings of this file: BETA, WM_SELL_PRICE, WM_BUY_PRICE (see below, each with
    its source).

Sources:
    Rule: Qussous, R., Harder, N., Weidlich, A. (2022). Understanding Power Market
        Dynamics by Reflecting Market Interrelations and Flexibility-Oriented Bidding
        Strategies. Energies 15(2), 494, Sect. 3.2.3, Eq. (20)-(24), p. 12-13.
    Price with efficiency: flexable_storage.py, lines 134-138 (decision E1).
    Naive forecast: Weron, R. (2014). Electricity price forecasting: A review of the
        state-of-the-art with a look into the future. International Journal of
        Forecasting 30(4), 1030-1081, Sect. 3.8.1, p. 1049-1050.
    Mean over the delivery day, fixed share BETA: [OWN] (decisions E4, D1).
    Hand calculation and learning script: course document WP13,
        examples/inputs/LocalRetailer_test/course/wp13_battery_bid.py.
"""

from datetime import timedelta

import numpy as np

from assume.common.base import SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.strategies.flexable_storage import StorageEnergyHeuristicFlexableStrategy

# Market whose forecast column holds the naive WM forecast (price_LEM_DA)
FORECAST_MARKET_ID = "LEM_DA"
LEM_MARKET_ID = "LEM_DA"
WM_MARKET_ID = "WM_DA"
# Share of the volume bid in the LEM (decision D1: scenarios 0, 0.5, 1)
BETA = 0.5
# WM price limits (config.yaml): orders at these prices are always accepted
WM_SELL_PRICE = -500.0
WM_BUY_PRICE = 3000.0


class StorageTwoMarketStrategy(StorageEnergyHeuristicFlexableStrategy):
    """
    flexABLE storage rule with a naive forecast, a daily reference price and a fixed
    split of the volume between LEM and WM.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Bids of the battery for all products of one market opening (one day).

        Args:
            unit (SupportsMinMaxCharge): The battery.
            market_config (MarketConfig): LEM_DA or WM_DA.
            product_tuples (list[Product]): The 96 products of the delivery day.

        Returns:
            Orderbook: One bid per product with price and volume (MW).
        """
        market_id = market_config.market_id
        share = BETA if market_id == LEM_MARKET_ID else 1 - BETA
        if share == 0:
            return []

        # Naive WM forecast, column price_LEM_DA of forecasts_df.csv (see "Data path")
        forecast = unit.forecaster.price[FORECAST_MARKET_ID]
        # Change 2: reference price = mean forecast of the delivery day
        reference_price = float(np.mean([forecast[p[0]] for p in product_tuples]))

        start = product_tuples[0][0]
        # SOC at the start of the day: initial_soc of storage_units.csv on day 1,
        # afterwards the SOC that ASSUME computed from the accepted volumes
        # (storage.py, line 177 and execute_current_dispatch, lines 277-340)
        theoretic_soc = unit.outputs["soc"].at[start]
        # Power of the last period before the day (assume/common/base.py, line 229)
        previous_power = unit.get_output_before(start)

        bids = []
        for product in product_tuples:
            start, end = product[0], product[1]
            current_power = unit.outputs["energy"].at[start]
            current_power_discharge = max(current_power, 0)
            current_power_charge = min(current_power, 0)

            # Volume limits from SOC, power and ramping (unchanged ASSUME logic)
            min_charge, max_charge = unit.calculate_min_max_charge(
                start, end, soc=theoretic_soc
            )
            min_discharge, max_discharge = unit.calculate_min_max_discharge(
                start, end, soc=theoretic_soc
            )
            max_discharge = unit.calculate_ramp_discharge(
                theoretic_soc, previous_power, max_discharge[0],
                current_power_discharge, min_discharge[0],
            )
            max_charge = unit.calculate_ramp_charge(
                theoretic_soc, previous_power, max_charge[0],
                current_power_charge, min_charge[0],
            )

            # Eq. (23): discharge if forecast >= reference price, else charge
            if forecast[start] >= reference_price and max_discharge:
                full_volume = max_discharge  # > 0
                # efficiency_discharge: storage_units.csv <- SimBench etaStore
                lem_price = reference_price / unit.efficiency_discharge
                wm_price = WM_SELL_PRICE
            else:
                full_volume = max_charge  # < 0
                # efficiency_charge: storage_units.csv <- SimBench etaStore
                lem_price = reference_price * unit.efficiency_charge
                wm_price = WM_BUY_PRICE

            # Change 3: share of the volume and price for this market
            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": lem_price if market_id == LEM_MARKET_ID else wm_price,
                    "volume": share * full_volume,
                    "node": unit.node,  # default "node0" (base.py, line 47), no grid model
                }
            )

            # Planned SOC follows the full volume of both markets (as in the parent)
            hours = (end - start) / timedelta(hours=1)
            power = full_volume + current_power
            # Same SOC update as storage.py, execute_current_dispatch, lines 312-333 (Wagner et al. 2026, Eq. (3));
            # capacity: storage_units.csv <- SimBench eStore (MWh)
            if power > 0:
                delta_soc = -(power * hours / unit.efficiency_discharge) / unit.capacity
            elif power < 0:
                delta_soc = -(power * hours * unit.efficiency_charge) / unit.capacity
            else:
                delta_soc = 0
            theoretic_soc += delta_soc
            previous_power = power

        return self.remove_empty_bids(bids)
