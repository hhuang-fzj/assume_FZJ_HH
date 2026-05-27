# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo
from assume.common import Forecaster
import pandas as pd

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.comando_facade import ComandoFacade

from comando.core import System
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
        forecaster: Forecaster,
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
                    f"Component {component} is required for the steel plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the steel plant unit."
                )

        self.natural_gas_price = self.forecaster['fuel_price_natural gas']
        self.electricity_price = self.forecaster['price_LLEC']
        self.heating_price = self.forecaster['price_HEAT'] #ToDo: add thermal price forecast
        self.cooling_price = self.forecaster['price_COOL']
        self.co2_price = self.forecaster['price_CO2']#ToDo: add Emission price

        self.objective = objective

        # Initialize the model
        self.setup_model()

    def initialize_energy_system(self):
        comps = self.components.values()

        conns = {
            'Power_Bus': [
                self.components['chp_CHP_1'].POWER_OUT,
                self.components['chp_CHP_2'].POWER_OUT,
                self.components['chp_CHP_3'].POWER_OUT,
                self.components['grid_Electricity'].CONSUMPTION,
                self.components['grid_Electricity'].FEEDIN #this is the interface for selling electricity
            ],
            'Heat_Bus': [
                self.components['chp_CHP_1'].HEAT_OUT,
                self.components['chp_CHP_2'].HEAT_OUT,
                self.components['chp_CHP_3'].HEAT_OUT,
                self.components['boiler_B_1'].OUT,
                self.components['boiler_B_2'].OUT,
                self.components['absorption_chiller_AC'].IN,
                self.components['demand_Heating'].IN,
            ],
            'Gas_Bus': [
                self.components['chp_CHP_1'].IN,
                self.components['chp_CHP_2'].IN,
                self.components['chp_CHP_3'].IN,
                self.components['boiler_B_1'].IN,
                self.components['boiler_B_2'].IN,
                self.components['grid_Gas'].CONSUMPTION,
            ],
            'Cooling_Bus': [
                self.components['compression_chiller_CC'].OUT,
                self.components['absorption_chiller_AC'].OUT,
                self.components['demand_Cooling'].IN,
            ]
        }

        self.comando_system = System(label=self.technology, components=comps, connections=conns)

    def define_constraints(self):
        # add expressions to energy system
        for expre in ['investment_costs', 'fixed_costs', 'variable_costs', 'emissions']:
            self.comando_system.add_expression(expre, self.comando_system.aggregate_component_expressions(expre))

    def create_problem(self):
        params = dict()
        index_pd = self.index.as_datetimeindex()
        index_pd_extent = index_pd.append(pd.DatetimeIndex([index_pd[-1] + self.index.freq]))
        #ToDo: Find a proper way to process the Heating/Cooling demand
        params['Heating_demand'] = self.demand['Heating Demand[kW]'][:745]
        params['Cooling_demand'] = self.demand['Cooling Demand [kW]'][:745]
        params['Electricity_price'] = self.electricity_price.as_pd_series()
        params['Gas_price'] = self.natural_gas_price.as_pd_series()

        ts = list((index_pd_extent[1:] - index_pd_extent[:-1]).seconds / 3600)
        ts = {i: time_step for i, time_step in enumerate(ts)}
        P = self.comando_system.create_problem(
            *make_tac_objective(self.comando_system, n=20, i=0.012),
            data=params,
            name='WVVZ',
            timesteps= ts
        )

        return to_gurobi(P)