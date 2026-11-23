# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from math import ceil, floor, isfinite

from assume.common.base import MinMaxStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Order, Orderbook, Product


class EnergyNaiveStrategy(MinMaxStrategy):
    """
    A naive strategy that bids the marginal cost of the unit on the market.

    Methods
    -------
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        start = product_tuples[0][0]  # start time of the first product
        end_all = product_tuples[-1][1]  # end time of the last product
        previous_power = unit.get_output_before(
            start
        )  # power output of the unit before the start time of the first product
        op_time = unit.get_operation_time(start)
        min_power_values, max_power_values = unit.calculate_min_max_power(
            start, end_all
        )  # minimum and maximum power output of the unit between the start time of the first product and the end time of the last product

        bids = []
        for product, min_power, max_power in zip(
            product_tuples, min_power_values, max_power_values
        ):
            # for each product, calculate the marginal cost of the unit at the start time of the product
            # and the volume of the product. Dispatch the order to the market.
            start = product[0]
            current_power = unit.outputs["energy"].at[
                start
            ]  # power output of the unit at the start time of the current product
            marginal_cost = unit.calculate_marginal_cost(
                start, previous_power
            )  # calculation of the marginal costs
            volume = unit.calculate_ramp(
                op_time, previous_power, max_power, current_power
            )
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": marginal_cost,
                    "volume": volume,
                    "node": unit.node,
                }
            )

            if "node" in market_config.additional_fields:
                bids[-1]["max_power"] = unit.max_power if volume > 0 else unit.min_power
                bids[-1]["min_power"] = min_power if volume > 0 else unit.max_power

            previous_power = volume + current_power
            if previous_power > 0:
                op_time = max(op_time, 0) + 1
            else:
                op_time = min(op_time, 0) - 1

        if "node" in market_config.additional_fields:
            return bids
        else:
            return self.remove_empty_bids(bids)


class EnergyNaiveProfileStrategy(MinMaxStrategy):
    """
    A naive strategy that bids the marginal cost of the unit as block bids over 24 hours on the day ahead market.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """

        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        op_time = unit.get_operation_time(start)
        _, max_power = unit.calculate_min_max_power(start, end_all)

        current_power = unit.outputs["energy"].at[start]
        marginal_cost = unit.calculate_marginal_cost(start, previous_power)

        # calculate the ramp up volume using the initial maximum power
        volume = unit.calculate_ramp(
            op_time, previous_power, max_power[0], current_power
        )

        profile = {product[0]: volume for product in product_tuples}
        order: Order = {
            "start_time": start,
            "end_time": product_tuples[-1][1],
            "only_hours": product_tuples[0][2],
            "price": marginal_cost,
            "volume": profile,
            "bid_type": "BB",
            "node": unit.node,
        }

        bids = [order]

        bids = self.remove_empty_bids(bids)
        return bids


class DsmEnergyOptimizationStrategy(MinMaxStrategy):
    """
    A naive strategy of a Demand Side Management (DSM) unit. The bid volume is the optimal power requirement of
    the unit at the start time of the product. The bid price is the marginal cost of the unit at the start time of the product.

    For rolling-horizon configured units, this strategy triggers re-optimization at each market clearing round,
    ensuring the unit optimizes for the next window using updated state and remaining demand.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """

        if unit.horizon_mode == "rolling_horizon":
            current_market_time = product_tuples[0][0]
            # Hook to refresh runtime forecasts before re-optimising the next window.
            # Currently the configured update algorithms default to no-ops; the hook is
            # kept so price/forecast learning can be plugged in without touching strategies.
            unit.forecaster.update(unit=unit)
            did_reoptimize = unit._check_and_reoptimize_rolling_window(
                current_market_time
            )
            if not did_reoptimize and unit.optimisation_counter == 0:
                unit.determine_optimal_operation_with_flex()
                unit.optimisation_counter = 1
        elif unit.optimisation_counter == 0:
            unit.determine_optimal_operation_with_flex()
            unit.optimisation_counter = 1

        bids = []
        for product in product_tuples:
            """
            for each product, calculate the marginal cost of the unit at the start time of the product
            and the volume of the product. Dispatch the order to the market.
            """
            start = product[0]

            volume = unit.opt_power_requirement.at[start]

            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": 3000,
                    "volume": -volume,
                }
            )

        return bids


class IntentBasedBiddingStrategy(MinMaxStrategy):
    """Translate an optimized multi-energy plan into bids with distinct intentions.

    Use configured directional prices or optional component-based cost-recovery
    selling prices and surcharges. Units describe bid blocks and their pricing
    intentions through market_bid_blocks(); this strategy resolves configured or
    market-maximum prices, preserving the planned total quantity.

    Reuse a full-horizon plan or refresh the unit's rolling optimization window.
    Map positive internal imports to negative market orders and record submitted
    positions and pricing audits. Price bounds and ticks apply to cost-recovery
    offers and blocks requesting price limits.

    The unit optimizes internal operation; this strategy translates its results
    into bids without optimizing bid prices or jointly clearing energy products.
    """

    def __init__(self, *args, cost_recovery=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Existing bidding_strategy_params transport; opt-in per ASSUME unit ID.
        self.cost_recovery = cost_recovery or {}
        if not isinstance(self.cost_recovery, dict):
            raise ValueError("cost_recovery must map unit IDs to pricing assumptions")

    @staticmethod
    def _cost_recovery_price_limits(price, market):
        """Validate the price and align market bounds to the permitted tick grid."""
        raw = float(price)
        low, high = market.minimum_bid_price, market.maximum_bid_price
        low = float(low) if low is not None else float("-inf")
        high = float(high) if high is not None else float("inf")
        tick = market.price_tick
        if not isfinite(raw) or low > high:
            raise ValueError(f"{market.market_id}: invalid cost-recovery price or market bounds")
        if tick is not None:
            tick = float(tick)
            if not isfinite(tick) or tick <= 0:
                raise ValueError(f"{market.market_id}: price_tick must be positive")
            low = ceil(low / tick - 1e-12) * tick if isfinite(low) else low
            high = floor(high / tick + 1e-12) * tick if isfinite(high) else high
            if low > high:
                raise ValueError(f"{market.market_id}: no price tick lies within market bounds")
        return raw, low, high, tick

    @classmethod
    def bounded_cost_recovery_price(cls, price, market):
        """Return EUR/MWh on the permitted tick grid; portfolio converts to ticks once."""
        raw, low, high, tick = cls._cost_recovery_price_limits(price, market)
        bounded = min(max(raw, low), high)
        final = min(max(round(bounded / tick) * tick, low), high) if tick else bounded
        return final, {"calculated_price": raw, "submitted_price": final,
                       "price_clipped": bounded != raw, "price_tick_rounded": final != bounded,
                       "submitted_tick_count": int(round(final / tick)) if tick else None}

    @staticmethod
    def _market_interface(unit, market_config):
        """Resolve the solved position and verify the market's energy product."""
        interfaces = {"LLEC": ("opt_power_volume", "energy"),
                      "HTH": ("opt_HT_heat_volume", "ht_heat"),
                      "LTH": ("opt_LT_heat_volume", "lt_heat")}
        market_id = market_config.market_id
        if market_id not in interfaces:
            raise ValueError(f"Unit {unit.id}: unsupported DSM market {market_id!r}")
        attribute, expected_product = interfaces[market_id]
        if market_config.product_type != expected_product:
            raise ValueError(f"Unit {unit.id}: market {market_id} requires product {expected_product!r}")
        return attribute, expected_product

    @staticmethod
    def _validate_products(unit, market_config, product_tuples):
        """Reject incompatible delivery intervals before starting optimization."""
        # Validate every product before solving or returning a partial bid list.
        for start, end, only_hours in product_tuples:
            if (start not in unit.index or end - start != unit.index.freq
                    or only_hours is not None):
                raise ValueError(f"Unit {unit.id}, market {market_config.market_id}: product {start} to {end} is outside or incompatible with the optimization horizon")

    @staticmethod
    def _ensure_operating_plan(unit, first_delivery):
        """Refresh a rolling window or reuse the existing full-horizon solution."""
        if unit.horizon_mode == "rolling_horizon":
            unit.forecaster.update(unit=unit)
            changed = unit._check_and_reoptimize_rolling_window(first_delivery)
            if not changed and unit.optimisation_counter == 0:
                unit.determine_optimal_operation_with_flex()
                unit.optimisation_counter = 1
        elif unit.optimisation_counter == 0:
            unit.determine_optimal_operation_with_flex()
            unit.optimisation_counter = 1

    @staticmethod
    def _bid_prices(unit, market_config):
        """Read directional prices, retaining the existing legacy DSM defaults."""
        market_id = market_config.market_id
        prices = market_config.param_dict.get("bid_prices", {}).get(unit.id)
        if prices is None:
            # Preserve the existing electricity/HT policy for legacy DSM units.
            # BPJ and LT require explicit scenario inputs, never PPA prices.
            legacy = {"LLEC": 158.5, "HTH": 93.7}
            if market_id not in legacy or getattr(unit, "technology", None) == "energy_cell":
                raise ValueError(f"Unit {unit.id}, market {market_id}: configure param_dict.bid_prices for buy/sell directions")
            prices = {"buy": legacy[market_id], "sell": legacy[market_id]}
        if not isinstance(prices, dict):
            raise ValueError(f"Unit {unit.id}, market {market_id}: bid_prices must map buy/sell directions to prices")
        return prices

    def _cost_recovery_settings(self, unit):
        """Validate optional component pricing and product-specific surcharges."""
        rules = self.cost_recovery.get(unit.id)
        if rules is not None and not isinstance(rules, dict):
            raise ValueError(f"{unit.id}: cost_recovery assumptions must be a mapping")
        adders = (rules or {}).get("selling_price_adders", {})
        if not isinstance(adders, dict) or set(adders) - {"energy", "ht_heat"}:
            raise ValueError(f"{unit.id}: selling_price_adders must map energy/ht_heat to EUR/MWh surcharges")
        for product, adder in adders.items():
            if (isinstance(adder, bool) or not isinstance(adder, (int, float))
                    or not isfinite(adder) or adder < 0):
                raise ValueError(f"{unit.id}: selling_price_adders.{product} must be finite nonnegative EUR/MWh")
        return rules, adders

    @staticmethod
    def _solved_net_import(unit, market_config, attribute, start):
        """Read one finite interface position in MW; imports are positive."""
        market_id = market_config.market_id
        position = getattr(unit, attribute)
        if start not in position.index:
            raise ValueError(f"Unit {unit.id}, market {market_id}: solved interface does not cover {start}")
        net_import = float(position.at[start])
        if not isfinite(net_import):
            raise ValueError(f"Unit {unit.id}, market {market_id}: non-finite position at {start}")
        return net_import

    @staticmethod
    def _directional_price(unit, market_config, prices, direction, start):
        """Resolve a configured scalar or timestamped purchase/sale price."""
        market_id = market_config.market_id
        if direction not in prices:
            raise ValueError(f"Unit {unit.id}, market {market_id}: unsupported {direction} direction; no bid price configured")
        price = prices[direction]
        if hasattr(price, "at"):
            try:
                price = price.at[start]
            except (IndexError, KeyError) as exc:
                raise ValueError(f"Unit {unit.id}, market {market_id}: no {direction} bid price at {start}") from exc
        price = float(price)
        if not isfinite(price):
            raise ValueError(f"Unit {unit.id}, market {market_id}: invalid {direction} price")
        return price

    def _validate_bid_blocks(self, unit, market, blocks, net_import):
        """Check the block contract before constructing any orders for an interval."""
        context = f"Unit {unit.id}, market {market.market_id}"
        if not isinstance(blocks, list) or not blocks:
            raise ValueError(f"{context}: market_bid_blocks must return a nonempty list")
        labels = set()
        total = 0.0
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError(f"{context}: each bid block must be a mapping")
            quantity, label = block.get("quantity_mw"), block.get("label")
            if (isinstance(quantity, bool) or not isinstance(quantity, (int, float))
                    or not isfinite(quantity) or quantity < 0):
                raise ValueError(f"{context}: block quantity_mw must be finite and nonnegative")
            if label is not None and (not isinstance(label, str) or not label):
                raise ValueError(f"{context}: bid block label must be a nonempty string")
            if label in labels:
                raise ValueError(f"{context}: duplicate bid block label {label!r}")
            labels.add(label)
            if (not isinstance(block.get("price_rule"), str)
                    or block["price_rule"] not in {"configured", "market_max"}):
                raise ValueError(f"{context}: unsupported bid block price_rule {block.get('price_rule')!r}")
            total += quantity
        if abs(total - abs(net_import)) > 1e-5:
            raise ValueError(f"{context}: bid block quantities do not match the planned net position")

    def _bid_block_orders(self, unit, market, product, net_import, configured_price, rules):
        """Price unit-described intentions and build orders without equipment knowledge."""
        start, end, only_hours = product
        describe = getattr(unit, "market_bid_blocks", None)
        # Preserve unsplit bids for units without the optional COMANDO hook.
        blocks = describe(start, market, net_import, rules) if describe else [
            {"quantity_mw": abs(net_import), "price_rule": "configured"}]
        self._validate_bid_blocks(unit, market, blocks, net_import)
        context = f"Unit {unit.id}, market {market.market_id}"
        bids, audits = [], []
        for block in blocks:
            quantity = block["quantity_mw"]
            if quantity <= 1e-9:
                continue
            raw = configured_price
            if block["price_rule"] == "market_max":
                raw = market.maximum_bid_price
                if raw is None or not isfinite(float(raw)):
                    raise ValueError(f"{context}: market_max block requires a configured maximum_bid_price")
            adjustment = None
            if block.get("apply_price_limits") or block["price_rule"] == "market_max":
                price, adjustment = self.bounded_cost_recovery_price(raw, market)
            else:
                price = configured_price
            bid = {"start_time": start, "end_time": end, "only_hours": only_hours,
                   "price": price, "volume": -quantity if net_import > 0 else quantity}
            if block.get("label") is not None:
                bid["bid_id"] = f"{unit.id}_{start.isoformat()}_{block['label']}"
            bids.append(bid)
            if adjustment is not None:
                audits.append({"datetime": start, "market_id": market.market_id,
                               "product_type": market.product_type,
                               "pricing_rule": block.get("pricing_rule", block["price_rule"]),
                               "quantity_mw": quantity, **adjustment})
        return bids, audits

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """Validate delivery products, obtain a plan, price its positions and record bids."""
        attribute, expected_product = self._market_interface(unit, market_config)
        if not product_tuples:
            return []
        self._validate_products(unit, market_config, product_tuples)
        self._ensure_operating_plan(unit, product_tuples[0][0])
        market_id = market_config.market_id
        if not hasattr(unit, attribute):
            raise ValueError(f"Unit {unit.id}, market {market_id}: missing solved interface {attribute}")
        prices = self._bid_prices(unit, market_config)
        rules, adders = self._cost_recovery_settings(unit)
        bids = []
        audits = []
        for start, end, only_hours in product_tuples:
            net_import = self._solved_net_import(unit, market_config, attribute, start)
            if abs(net_import) <= 1e-9:  # Ignore numerical solver noise in MW.
                continue
            direction = "buy" if net_import > 0 else "sell"
            technology = getattr(unit, "technology", None)
            selling_rule = rules is not None and direction == "sell" and (
                technology == "energy_hub" and expected_product in {"energy", "ht_heat"}
                or technology == "waste_heat_recovery" and expected_product == "ht_heat"
            )
            if selling_rule:
                base, components = unit.cost_recovery_selling_price(start, expected_product, -net_import, rules)
                # A bidding surcharge, applied once after component aggregation;
                # it does not enter the COMANDO dispatch objective or quantities.
                adder = float(adders.get(expected_product, 0.0))
                raw = base + adder
                price, adjustment = self.bounded_cost_recovery_price(raw, market_config)
                audits.append({"datetime": start, "market_id": market_id, "product_type": expected_product,
                               "pricing_rule": "component_cost_recovery", "components": components,
                               "base_cost_recovery_price": base, "selling_price_adder": adder, **adjustment})
                bids.append({"start_time": start, "end_time": end, "only_hours": only_hours,
                             "price": price, "volume": -net_import})
                continue
            price = self._directional_price(unit, market_config, prices, direction, start)
            block_bids, block_audits = self._bid_block_orders(
                unit, market_config, (start, end, only_hours), net_import, price, rules)
            bids.extend(block_bids)
            audits.extend(block_audits)
        if hasattr(unit, "submitted_orders"):
            planned = {}
            for bid in bids:
                unit.submitted_orders.append({**bid, "market_id": market_id,
                                              "product_type": market_config.product_type})
                planned[bid["start_time"]] = planned.get(bid["start_time"], 0.0) + bid["volume"]
            for start, volume in planned.items():
                unit.outputs[f"{market_config.product_type}_planned"].at[start] = volume
        if audits:
            unit.cost_recovery_audit.extend(audits)
        return bids


class DsmEnergyNaiveRedispatchStrategy(MinMaxStrategy):
    """
    A naive strategy of a Demand Side Management (DSM) unit that bids the available flexibility of the unit on the redispatch market.
    The bid volume is the flexible power requirement of the unit at the start time of the product. The bid price is the marginal cost of the unit at the start time of the product.

    For rolling-horizon configured units, this strategy triggers re-optimization at each market clearing round.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        if unit.horizon_mode == "rolling_horizon":
            current_market_time = product_tuples[0][0]
            # Hook to refresh runtime forecasts before re-optimising the next window.
            # Currently the configured update algorithms default to no-ops; the hook is
            # kept so price/forecast learning can be plugged in without touching strategies.
            unit.forecaster.update(unit=unit)
            unit._check_and_reoptimize_rolling_window(current_market_time)

        # calculate the optimal operation of the unit according to the objective function
        unit.determine_optimal_operation_with_flex()

        bids = []
        for product in product_tuples:
            """
            for each product, calculate the marginal cost of the unit at the start time of the product
            and the volume of the product. Dispatch the order to the market.
            """
            start = product[0]
            volume = unit.flex_power_requirement.at[start]
            marginal_price = unit.calculate_marginal_cost(start, volume)
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": marginal_price,
                    "volume": -volume,
                }
            )

        return bids


class EnergyNaiveRedispatchStrategy(MinMaxStrategy):
    """
    A naive strategy that simply submits all information about the unit and
    currently dispatched power for the following hours to the redispatch market.
    Information includes the marginal cost, the ramp up and down values, and the dispatch.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        :param unit: the unit to be dispatched
        :type unit: SupportsMinMax
        :param market_config: the market configuration
        :type market_config: MarketConfig
        :param product_tuples: list of all products the unit can offer
        :type product_tuples: list[Product]
        :return: the bids consisting of the start time, end time, only hours, price and volume.
        :rtype: Orderbook
        """
        start = product_tuples[0][0]
        previous_power = unit.get_output_before(start)
        p_nom = unit.max_power

        bids = []
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]
            marginal_cost = unit.calculate_marginal_cost(
                start, previous_power
            )  # calculation of the marginal costs
            available_power = unit.forecaster.availability.at[start] * unit.max_power
            # if additional max_power is below the technical minimum, the unit cannot run
            # set to current power to avoid supplying redispatch potential (although current power should generally be 0 in that case)
            if available_power < unit.min_power:
                min_power = current_power
                max_power = current_power
            else:
                min_power = unit.min_power
                max_power = available_power

            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": marginal_cost,
                    "volume": current_power,
                    "max_power": max_power,
                    "min_power": min_power,
                    "p_nom": p_nom,
                    "node": unit.node,
                }
            )

            # update previous power for the next iteration
            previous_power = current_power

        return bids


class ExchangeEnergyNaiveStrategy(MinMaxStrategy):
    """
    A naive strategy for an exchange unit that bids the defined import and export prices on the market.
    It submits two bids, one for import and one for export, with the respective prices and volumes.
    Export bids have negative volumes and are treated as demand on the market.
    Import bids have positive volumes and are treated as supply on the market.

    Methods
    -------
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """

        bids = []
        for product in product_tuples:
            # for each product, calculate the marginal cost of the unit at the start time of the product
            # and the volume of the product. Dispatch the order to the market.
            start = product[0]

            # append import bid
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": unit.price_import,
                    "volume": unit.volume_import.at[start],
                    "node": unit.node,
                }
            )

            # append export bid
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": unit.price_export,
                    "volume": unit.volume_export.at[start],
                    "node": unit.node,
                }
            )

        # clean up empty bids
        bids = self.remove_empty_bids(bids)

        return bids


class EnergyHeuristicElasticStrategy(MinMaxStrategy):
    """
    A bidding strategy for a demand unit that submits multiple bids to approximate
    a marginal utility curve, based on linear or isoelastic demand theory.
    P = Price, Q = Quantity, E = Elasticity.

    - Linear model: P = P_max + slope * Q (slope is only defined by P_max and Q_max, negative value)
    - Isoelastic model: P = (Q/Q_max) ** (1/E) (E is negative)
      (derived from log-log price elasticity of demand)

    See:
    - https://en.wikipedia.org/wiki/Price_elasticity_of_demand
    - Arnold, Fabian. 2023. https://hdl.handle.net/10419/286380
    - Hirth, Lion et al. 2024. https://doi.org/10.1016/j.eneco.2024.107652.

    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        bids = []

        for product in product_tuples:
            start, end, only_hours = product
            max_abs_power = max(abs(unit.min_power), abs(unit.max_power))

            if unit.elasticity_model == "isoelastic":
                # ISOELASTIC model (constant elasticity of demand):
                # P = (Q / Q_max)^(1 / E)
                # Derived from:
                #   E = (dQ/dP) * (P / Q(P)), E == constant
                #   E = Q'(P) * (P / Q(P))
                #   Q'(P) = E * 1/P * Q(P)
                #   dQ/dP = E * (1 / P) * Q
                #   integrate:
                #   \int 1/Q dQ = E * \int 1/P dP
                #   ln(Q) = E * ln(P) + C
                #   exp(ln(Q)) = exp(E * ln(p) + C)
                #   Q(p) = (exp(ln(p))^E) * exp(C)
                #   Q(p) = P^E * exp(C)
                #   possibly C = 0, C = 1 or C >= 1. We assume C >= 1.
                #   C shifts the demand curve (demand vs. price) up / down and
                #   can be interpreted as the demand that will always be served
                #   we set C = ln(Q_max) and derive the demand curve from there:
                #   P = Q^(1/E) * exp(-C/E) and C = ln(Q_max)
                #   => P = Q^(1/E) * exp(-ln(Q_max)/E) and because exp(-ln(Q_max)/E) = Q_max^(-1/E)
                #   => P = Q^(1/E) * Q_max^(-1/E)
                #   finally
                #   => P = (Q / Q_max)^(1/E)
                # This yields decreasing price as volume increases

                # calculate first bid in isoelastic model (the volume that is bid at max price)
                first_bid_volume = self.find_first_block_bid(
                    unit.elasticity, unit.max_price, max_abs_power
                )
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": unit.max_price,
                        "volume": -first_bid_volume,
                        "node": unit.node,
                    }
                )

                remaining_volume = max_abs_power - first_bid_volume
                if remaining_volume < 0:
                    raise ValueError(
                        "Negative remaining volume for bidding. Check max_power and elasticity."
                    )

                bid_volume = remaining_volume / (unit.num_bids - 1)
                # calculate the remaining bids in isoelastic model
                # P = (Q/Q_max) ** (1/E)
                for i in range(1, unit.num_bids):
                    ratio = (first_bid_volume + i * bid_volume) / max_abs_power
                    if ratio <= 0:
                        continue
                    bid_price = ratio ** (1 / unit.elasticity)

                    bids.append(
                        {
                            "start_time": start,
                            "end_time": end,
                            "only_hours": only_hours,
                            "price": bid_price,
                            "volume": -bid_volume,
                            "node": unit.node,
                        }
                    )

            elif unit.elasticity_model == "linear":
                # LINEAR model: P = P_max - slope * Q
                # where slope = -(P_max / Q_max)
                # This ensures price drops linearly with volume
                bid_volume = max_abs_power / unit.num_bids
                slope = -(unit.max_price / max_abs_power)
                for i in range(unit.num_bids):
                    bid_price = unit.max_price + (slope * i * bid_volume)

                    bids.append(
                        {
                            "start_time": start,
                            "end_time": end,
                            "only_hours": only_hours,
                            "price": bid_price,
                            "volume": -bid_volume,
                            "node": unit.node,
                        }
                    )

        return self.remove_empty_bids(bids)

    def find_first_block_bid(
        self, elasticity: float, max_price: float, max_power: float
    ) -> float:
        """
        Calculate the first block bid volume at max_price. P = Price, Q = Quantity, E = Elasticity.
        Assumes isoelastic demand:

        .. math::
            Q = Q_{max} * P^E

        The first block bid is the volume that is always bid at maximum price, because the
        willingness to pay for it is higher than the markets maximal price.
        The first block bid volume is calculated by finding the intersection of the isoelastic demand
        curve and the maximum price in the marginal utility plot. All demand left of the intersection
        is always bought at maximum price and is called $Q_{first}$.

        .. math::
            Q_{first} = Q
            Q_{first} = Q_{max} * P^E

        Therefore:

        .. math::
            Q_{first} = power_{max} * (price_{max} ^ E)

        Returns:
            float: Volume > 0, demand that is always bought at max willingness to pay
        """
        volume = max_power * max_price**elasticity

        if abs(volume) > abs(max_power):
            raise ValueError(
                f"Calculated first block bid volume ({volume}) exceeds max power ({max_power})."
            )
        return volume


class DsmCapacityHeuristicBalancingStrategy(MinMaxStrategy):
    """
    Strategy for Positive/Negative CRM Reserve (Demand Side, i.e., up & down, symmetric).
    """

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        bids = []
        max_power = unit.max_plant_capacity
        min_power = unit.min_plant_capacity

        for product in product_tuples:
            start, end, only_hours = product
            block_times = [dt for dt in unit.index.get_date_list() if start <= dt < end]

            # For all time steps in block, calculate possible symmetric bid
            up_caps = []
            down_caps = []
            for t in block_times:
                flex = unit.flex_power_requirement.at[t]
                up_caps.append(max_power - flex)
                down_caps.append(flex - min_power)
            # The symmetric bid is the minimum capacity that is possible in *all* timesteps in the block
            symmetric_capacity = min(min(up_caps), min(down_caps))
            if symmetric_capacity > 0:
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": 0,  # or unit.calculate_marginal_cost(...)
                        "volume": symmetric_capacity,
                        "unit_id": unit.id,
                    }
                )
        return self.remove_empty_bids(bids)
