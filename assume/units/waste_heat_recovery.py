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

        return to_gurobi(P)
