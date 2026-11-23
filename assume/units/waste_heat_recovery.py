# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo
import pandas as pd

from assume.common.base import SupportsMinMax
from assume.common.forecaster import WasteHeatRecoveryForecaster
from assume.units.comando_facade import ComandoFacade

from comando.utility import make_tac_objective
from comando.interfaces.gurobi import to_gurobi

logger = logging.getLogger(__name__)

class WasteHeatRecovery(ComandoFacade, SupportsMinMax):

    # Required and optional technologies for the steel plant
    required_technologies = []
    optional_technologies = ["hpc", "heat_pump"]

    def __init__(
            self,
            id: str,
            unit_operator: str,
            bidding_strategies: dict,
            forecaster: WasteHeatRecoveryForecaster,
            components: dict[str, dict] = None,
            technology: str = "waste_heat_recovery",
            objective: str = "min_variable_cost",
            **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            components=components,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            # node=node,
            # location=location,
            **kwargs,
        )
        # check if the required components are present in the components dictionary
        for component in self.required_technologies:
            if component not in components.keys():
                raise ValueError(
                    f"Component {component} is required for the unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                    component not in self.required_technologies
                    and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the unit."
                )

        self.ht_heating_price = self.forecaster.fuel_prices['HT_heat']
        self.lt_heating_price = self.forecaster.fuel_prices['HT_heat']#ToDo: Change this to Low Temp Price

        self.objective = objective

        # WasteHeatRecoveryForecaster builds the model after initializing market prices.
    def define_constraints(self):
        # add expressions to energy system
        for expre in ['investment_costs', 'fixed_costs', 'variable_costs', 'emissions']:
            self.comando_system.add_expression(expre, self.comando_system.aggregate_component_expressions(expre))

    def create_problem(self):
        params = dict()
        index_pd = self.index.as_datetimeindex()
        index_pd_extent = index_pd.append(pd.DatetimeIndex([index_pd[-1] + self.index.freq]))
        self.bind_series(params, 'grid_Electricity', 'price', self.electricity_price)
        self.bind_series(params, 'grid_HT_Heat', 'price', self.ht_heating_price)
        params = {name: series.reset_index(drop=True) for name, series in params.items()}

        ts = list((index_pd_extent[1:] - index_pd_extent[:-1]).seconds / 3600)
        ts = {i: time_step for i, time_step in enumerate(ts)}
        P = self.comando_system.create_problem(
            *make_tac_objective(self.comando_system, n=20, i=0.012),
            data=params,
            name='HPC',
            timesteps= ts
        )

        self.comando_problem = P
        return to_gurobi(P)

    def cost_recovery_selling_price(self, start, product_type, export_volume, config):
        """HP-only HT cost recovery; LT source heat and additional variable O&M are free.

        COP uses modeled HP heat/electricity, excluding mandatory HPC electricity.
        Fixed O&M excludes investment annuities, HPC/IT and separate infrastructure.
        This forecast/pre-clearing calculation does not secure delivery after clearing.
        """
        if export_volume <= 1e-9:
            return None, []
        if product_type != "ht_heat":
            raise ValueError(f"{self.id}: cost recovery is defined only for HT heat")
        step = self.cost_recovery_timestep(start)
        price = float(self.electricity_price.at[start])
        if not isfinite(price):
            raise ValueError(f"{self.id}: non-finite electricity forecast at {start}")
        hours = self.required_cost_recovery_input(config, "full_load_hours")
        variable = self.required_cost_recovery_input(config, "variable_nonfuel_cost_heat")
        records = []
        for key, hp in self.components.items():
            if not key.startswith("heat_pump_"):
                continue
            heat = float(evaluate(hp.get_expression("output"), idx=step))
            if not isfinite(heat) or heat < -1e-8:
                raise ValueError(f"{self.id}: invalid {key} HT generation at {start}")
            if heat <= 1e-9:
                continue
            electricity = float(evaluate(hp.get_expression("input"), idx=step))
            if not isfinite(electricity) or electricity <= 0:
                raise ValueError(f"{self.id}: operating {key} needs positive HP electricity for COP")
            cop = heat / electricity
            fixed = self.annual_fixed_cost_surcharge(key, float(hp["Qdot_out_nom"].value), hours)
            electric_cost = price / cop
            records.append({"component": key, "output_mw": heat, "hp_electricity_mw": electricity,
                            "forecast_input_price": price, "efficiency_or_cop": cop,
                            "variable_nonfuel_cost": variable, "fuel_or_electricity_cost": electric_cost,
                            **fixed, "component_price": electric_cost + variable + fixed["fixed_surcharge"]})
        total = sum(row["output_mw"] for row in records)
        if total <= 1e-9 or export_volume > total + 1e-5:
            raise ValueError(f"{self.id}: HT export cannot be priced from HP generation; inspect HT grid imports")
        for row in records:
            row["aggregation_weight"] = row["output_mw"] / total
        return sum(row["component_price"] * row["aggregation_weight"] for row in records), records

    def market_bid_blocks(self, start, market_config, net_import, pricing_config):
        """Describe mandatory HPC and optional HP purchases under the existing policy."""
        if not (
            pricing_config and pricing_config.get("mandatory_hpc_at_market_max")
            and market_config.market_id == "LLEC" and net_import > 0
        ):
            return super().market_bid_blocks(start, market_config, net_import, pricing_config)
        hpc, hp = self.electricity_purchase_split(start, net_import)
        return [
            {"label": "hpc", "quantity_mw": hpc, "price_rule": "market_max",
             "apply_price_limits": True, "pricing_rule": "mandatory_hpc_market_max"},
            {"label": "hp", "quantity_mw": hp, "price_rule": "configured",
             "apply_price_limits": True, "pricing_rule": "existing_hp_purchase_policy"},
        ]

    def electricity_purchase_split(self, start, net_import):
        """Separate mandatory HPC electricity from the unchanged pre-clearing HP load."""
        step = self.cost_recovery_timestep(start)
        mandatory = sum(float(c["Electric_consumption"].value)
                        for key, c in self.components.items() if key.startswith("hpc_"))
        hp_power = sum(float(evaluate(c.get_expression("input"), idx=step))
                       for key, c in self.components.items() if key.startswith("heat_pump_"))
        if (not all(isfinite(value) for value in [mandatory, hp_power, net_import])
                or mandatory <= 0 or hp_power < -1e-8
                or abs(net_import - mandatory - hp_power) > 1e-5):
            raise ValueError(f"{self.id}: LLEC net import does not match mandatory HPC plus HP electricity")
        mandatory_bid = min(mandatory, net_import)
        return mandatory_bid, max(0.0, net_import - mandatory_bid)
    # intermediate calculation, and handling rejected/partially accepted bids.
