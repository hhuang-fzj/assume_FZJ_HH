# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo
import pandas as pd

from assume.common.base import SupportsMinMax
from assume.common.forecaster import EnergyHubForecaster
from assume.units.comando_facade import ComandoFacade

from comando.utility import make_tac_objective
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

        return to_gurobi(P)
