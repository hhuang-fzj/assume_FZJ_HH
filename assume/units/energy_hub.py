# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from math import isfinite

import pyomo.environ as pyo
import pandas as pd

from assume.common.base import SupportsMinMax
from assume.common.forecaster import EnergyHubForecaster
from assume.units.comando_facade import ComandoFacade

from comando.utility import evaluate, make_tac_objective
from comando.interfaces.gurobi import to_gurobi

logger = logging.getLogger(__name__)


class EnergyHub(ComandoFacade, SupportsMinMax):
    """
    The EnergyHub class represents an energy supply unit within an energy system.
    This unit includes various components and can process and convert different energy carriers into various forms of energy,
    such as electricity, heat, and cooling. Currently, no storage is implemented.
    The class model uese the SteelPlant class as template and depends on the Component-Oriented Modeling (COMANDO)
    framework: https://pypi.org/project/comando/

    Args:


    Attributes:

    """

    # Required and optional technologies for the steel plant
    required_technologies = []
    optional_technologies = ["chp", "boiler", "absorption_chiller", "compression_chiller"]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: EnergyHubForecaster,
        components: dict[str, dict] = None,
        technology: str = "energy_hub",
        objective: str = "min_variable_cost",
        # flexibility_measure: str = "cost_based_load_shift",
        # demand: float = 0,
        # cost_tolerance: float = 10,
        # congestion_threshold: float = 0,
        # peak_load_cap: float = 0,
        # node: str = "node0",
        # location: tuple[float, float] = (0.0, 0.0),
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
                    f"Component {component} is required for the Energy Hub."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the Energy Hub."
                )

        self.natural_gas_price = self.forecaster.fuel_prices['natural gas']
        self.ht_heating_price = self.forecaster.fuel_prices['HT_heat']
        #self.cooling_price = self.forecaster['price_COOL']
        self.cooling_demand = self.forecaster.dsm_demands['Cooling Demand [MW]']
        #self.heating_demand = self.forecaster['dsm_Heating Demand [MW]']

        self.objective = objective

        # EnergyHubForecaster builds the model after initializing market prices.

    def initialize_demands(self):
        # WVVZ's existing cooling demand; do not consume other units' columns.
        from assume.units.comando_components import comando_dst

        self.components["demand_Cooling"] = comando_dst["demand"](f"{self.id}_Cooling")

    def define_constraints(self):
        # add expressions to energy system
        for expre in ['investment_costs', 'fixed_costs', 'variable_costs', 'emissions']:
            self.comando_system.add_expression(expre, self.comando_system.aggregate_component_expressions(expre))
        # implement heat guided operation of the energy_hub, comment out for heat market operation
        # sum_heat_output = self.comando_system.aggregate_component_expressions('heat_output')
        # heat_demand = self.comando_system.aggregate_component_expressions("Heating_demand")
        # ac_input = self.comando_system.aggregate_component_expressions("AC_input")
        # self.comando_system.add_le_constraint(sum_heat_output, heat_demand+ac_input, name='heat_guided_operation')

    def create_problem(self):
        params = dict()
        index_pd = self.index.as_datetimeindex()
        index_pd_extent = index_pd.append(pd.DatetimeIndex([index_pd[-1] + self.index.freq]))
        #params['Heating_demand'] = self.heating_demand.as_pd_series()
        self.bind_series(params, 'demand_Cooling', 'demand', self.cooling_demand)
        self.bind_series(params, 'grid_Electricity', 'price', self.electricity_price)
        self.bind_series(params, 'grid_Gas', 'price', self.natural_gas_price)
        self.bind_series(params, 'grid_HT_Heat', 'price', self.ht_heating_price)
        # COMANDO timesteps below use integer labels, not datetime labels.
        params = {name: series.reset_index(drop=True) for name, series in params.items()}

        ts = list((index_pd_extent[1:] - index_pd_extent[:-1]).seconds / 3600)
        ts = {i: time_step for i, time_step in enumerate(ts)}

        if self.objective == "GWI":
            co2_tot = self.comando_system.get_exp0ression('emissions')

            P = self.comando_system.create_problem(
                operational_objective=co2_tot,
                timesteps=ts,
                data=params,
                name='PreJuCells'
            )
        else:
            P = self.comando_system.create_problem(
                *make_tac_objective(self.comando_system, n=20, i=0.012),
                data=params,
                name='WVVZ',
                timesteps= ts
            )

        self.comando_problem = P
        return to_gurobi(P)

    def cost_recovery_selling_price(self, start, product_type, export_volume, config):
        """Reference-efficiency accounting prices, not optimized marginal costs.

        Alpha allocates shared CHP fuel/fixed O&M once across the two outputs.
        Gross generation weights include internally used heat (e.g. the AC).
        Grid imports receive no CHP/boiler production-cost allocation. There is
        no physical source-to-consumer attribution in this weighted average.
        """
        if export_volume <= 1e-9:
            return None, []
        if product_type not in {"energy", "ht_heat"}:
            raise ValueError(f"{self.id}: unsupported cost-recovery product {product_type}")
        if (config.get("gas_price_basis") not in {"LHV", "HHV"}
                or config.get("gas_price_basis") != config.get("efficiency_basis")):
            raise ValueError(f"{self.id}: gas-price and reference-efficiency heating-value bases must match")
        alpha = self.required_cost_recovery_input(config, "alpha")
        hours = self.required_cost_recovery_input(config, "full_load_hours")
        if not 0 <= alpha <= 1:
            raise ValueError(f"{self.id}: CHP alpha must lie between zero and one")
        step = self.cost_recovery_timestep(start)
        gas = float(self.natural_gas_price.at[start])
        if not isfinite(gas):
            raise ValueError(f"{self.id}: non-finite gas forecast at {start}")
        records = []
        for key, component in self.components.items():
            if key.startswith("chp_"):
                if product_type == "energy":
                    efficiency, share, output = component.eff_el_nom, alpha, "output"
                    variable_key = "chp_variable_nonfuel_cost_el"
                else:
                    efficiency, share, output = component.eff_th_nom, 1 - alpha, "heat_output"
                    variable_key = "chp_variable_nonfuel_cost_heat"
                rated_el = float(component["P_el_out_nom"].value)
                rated = rated_el if product_type == "energy" else rated_el * component.eff_th_nom / component.eff_el_nom
            elif key.startswith("boiler_") and product_type == "ht_heat":
                efficiency, share, output = component.kwargs["thermal_efficiency"], 1.0, "output"
                rated = float(component["Qdot_out_nom"].value)
                variable_key = "boiler_variable_nonfuel_cost_heat"
            else:
                continue
            generation = float(evaluate(component.get_expression(output), idx=step))
            if not isfinite(generation) or generation < -1e-8:
                raise ValueError(f"{self.id}: invalid {key} generation at {start}")
            if generation <= 1e-9:
                continue
            if not isfinite(efficiency) or efficiency <= 0:
                raise ValueError(f"{self.id}: {key}: reference efficiency must be positive")
            variable = self.required_cost_recovery_input(config, variable_key)
            fixed = self.annual_fixed_cost_surcharge(key, rated, hours, share)
            fuel = share * gas / efficiency
            records.append({"component": key, "output_mw": generation, "forecast_input_price": gas,
                            "efficiency_or_cop": float(efficiency), "alpha": alpha if key.startswith("chp_") else None,
                            "heating_value_basis": config["gas_price_basis"],
                            "variable_nonfuel_cost": variable, "fuel_or_electricity_cost": fuel,
                            **fixed, "component_price": fuel + variable + fixed["fixed_surcharge"]})
        total = sum(row["output_mw"] for row in records)
        if total <= 1e-9 or export_volume > total + 1e-5:
            raise ValueError(f"{self.id}: {product_type} export cannot be priced from producing CHP/boiler components; inspect grid imports")
        for row in records:
            row["aggregation_weight"] = row["output_mw"] / total
        return sum(row["component_price"] * row["aggregation_weight"] for row in records), records
