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

class WasteHeatRecovery(ComandoFacade, SupportsMinMax):

    # Required and optional technologies for the steel plant
    required_technologies = []
    optional_technologies = ["HPC", "heatpump"]

    def __init__(
            self,
            id: str,
            unit_operator: str,
            bidding_strategies: dict,
            forecaster: Forecaster,
            components: dict[str, dict] = None,
            technology: str = "energy_hub",
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

        self.electricity_price = self.forecaster['price_LLEC']
        self.ht_heating_price = self.forecaster['fuel_price_HT_heat']
        self.lt_heating_price = self.forecaster['fuel_price_HT_heat']#ToDo: Change this to Low Temp Price

        self.objective = objective

        # Initialize the model
        self.setup_model()
    def initialize_energy_system(self):
        comps = self.components.values()

        conns = {
            'Power_Bus': [
                self.components['HPC'].POWER_IN,
                self.components['heat_pump'].POWER_IN,
                self.components['grid_Electricity'].CONSUMPTION,
                self.components['grid_Electricity'].FEEDIN #this is the interface for selling electricity
            ],
            'HT_Heat_Bus': [
                self.components['heat_pump'].HEAT_OUT,
                self.components['grid_HT_Heat'].FEEDIN,
                self.components['grid_HT_Heat'].CONSUMPTION,
            ],
            'LT_Heat_Bus': [
                self.components['heat_pump'].Qdot_in,
                self.components['HPC'].HEAT_OUT,
                self.components['grid_HT_Heat'].FEEDIN,#ToDO: Change it to Grid_LT_Heat
                self.components['grid_HT_Heat'].CONSUMPTION,
            ],
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
        params['Electricity_price'] = self.electricity_price.as_pd_series()
        params['HtHeating_price'] = self.ht_heating_price.as_pd_series()

        ts = list((index_pd_extent[1:] - index_pd_extent[:-1]).seconds / 3600)
        ts = {i: time_step for i, time_step in enumerate(ts)}
        P = self.comando_system.create_problem(
            *make_tac_objective(self.comando_system, n=20, i=0.012),
            data=params,
            name='HPC',
            timesteps= ts
        )

        return to_gurobi(P)