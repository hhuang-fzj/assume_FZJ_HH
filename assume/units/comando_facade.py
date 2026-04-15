import logging

from comando.core import System
import pandas as pd

from comando.utility import make_tac_objective
from comando.interfaces.gurobi import to_gurobi
from datetime import datetime
from assume.units.comando_components import comando_dst

# import pickle
# import os

class ComandoFacade:

    def __init__(self, components, **kwargs):
        super().__init__(**kwargs)
        self.components = components

    def initialize_components(self):
        """
        Initializes the COMANDO components by creating COMANDO energy system and specify the connections.

        This method iterates over the provided components, instantiates their corresponding classes,
        and adds the respective COMANDO.components to the COMANDO energy system.

        Args:
            components (dict[str, dict]): A dictionary where each key is a technology name and
                                        the value is a dictionary of parameters for the respective technology.
                                        Each technology is mapped to a corresponding class in `demand_side_technologies`.

        The method:
        - Looks up the corresponding class for each technology in `demand_side_technologies`.
        - Instantiates the class by passing the required parameters.
        - Adds the resulting block to the model under the `dsm_blocks` attribute.
        """
        components = self.components.copy()#store the components config for the creation of component classes
        self.components.clear()# Clear the space for component classes

        for technology, component_data in components.items():
            if technology in comando_dst:
                # Get the class from the dictionary mapping (adjust `demand_side_technologies` to hold classes)
                component_class = comando_dst[technology]
                if isinstance(component_data, list):#If multiple components using same technology
                    for data in component_data:
                        # Instantiate the component with the required parameters (unpack the component_data dictionary)
                        component_instance = component_class(**data)
                        self.components[technology + "_" + data['label']] = component_instance

                else:
                    # Instantiate the component with the required parameters (unpack the component_data dictionary)
                    component_instance = component_class(**component_data)
                    self.components[technology + "_" + component_data['label']] = component_instance

    def setup_model(self, presolve=True):
        # Initialize the Pyomo model
        # along with optimal and flexibility constraints
        # and the objective functions

        self.optimisation_counter = 0

        self.initialize_components()
        self.initialize_energy_system()#ToDo: tobe defined in the child class WVVZ

        self.define_constraints()# Placeholder for energy system relevant constraints
        # self.define_objective_opt()

        #Fixme: feature deactivate for test purpose, reactivate after implementation of both operation determination

        # # Solve the model to determine the optimal operation without flexibility
        # # and store the results to be used in the flexibility mode later
        # if presolve:
        #     self.determine_optimal_operation_without_flex(switch_flex_off=False)
        #
        # # Modify the model to include the flexibility measure constraints
        # # as well as add a new objective function to the model
        # # to maximize the flexibility measure
        # if self.flexibility_measure in DSMFlex.flexibility_map:
        #     DSMFlex.flexibility_map[self.flexibility_measure](self, self.model)
        # else:
        #     raise ValueError(f"Unknown flexibility measure: {self.flexibility_measure}")
    def determine_optimal_operation_without_flex(self):
        pass
    def determine_optimal_operation_with_flex(self):
        pass
