# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo
from assume.common import Forecaster

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.comando_facade import ComandoFacade

from comando.core import System

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
    optional_technologies = ["chp", "boiler", "absorption_chiller"]

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

        self.natural_gas_price = self.forecaster['fuel_price']
        self.electricity_price = self.forecaster['price_EOM']
        self.heating_price = self.forecaster['price_HEAT']
        self.cooling_price = self.forecaster['price_COOL']
        self.co2_price = self.forecaster['price_CO2']

        self.objective = objective

        # Initialize the model
        self.setup_model()
    def initialize_energy_system(self):
        #ToDO: Think about we if need to generalize this steps & if we can read the connections from config
        comps = self.components.values()

        conns = {
            'Power_Bus': [
                self.components['chp_CHP_1'].POWER_OUT,
                self.components['chp_CHP_2'].POWER_OUT,
                self.components['chp_CHP_3'].POWER_OUT
            ],
            'Heat_Bus': [
                self.components['chp_CHP_1'].HEAT_OUT,
                self.components['chp_CHP_2'].HEAT_OUT,
                self.components['chp_CHP_3'].HEAT_OUT,
                self.components['boiler_B_1'].OUT,
                self.components['boiler_B_2'].OUT,
                self.components['absorption_chiller_AC'].IN
            ],
            'Gas_Bus': [
                self.components['chp_CHP_1'].IN,
                self.components['chp_CHP_2'].IN,
                self.components['chp_CHP_3'].IN,
                self.components['boiler_B_1'].IN,
                self.components['boiler_B_2'].IN]
        }

        self.comando_system = System(label=self.technology, components=comps, connections=conns)

    def define_constraints(self):
        pass