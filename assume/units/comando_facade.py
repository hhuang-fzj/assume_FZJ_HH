from assume.common.fast_pandas import FastSeries
from assume.units.comando_components import comando_dst
from assume.common.temp_gurobi_visulize import interactive_timeseries_plot# temperary plot function for gurobi model
# results

# import pickle
# import os

class ComandoFacade:

    def __init__(self, components, **kwargs):
        super().__init__(**kwargs)
        self.components = components
        self.demand = kwargs['demand']#ToDo: move this to energy hub, it is not general feature of a COMANDO Unit

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
        feuls = list()# temporal container for all the fuel types of the main component

        #add main components by reading the csv configuration
        for technology, component_data in components.items():
            if technology in comando_dst:
                # Get the class from the dictionary mapping (adjust `demand_side_technologies` to hold classes)
                component_class = comando_dst[technology]
                if isinstance(component_data, list):#If multiple components using same technology
                    for data in component_data:
                        # Instantiate the component with the required parameters (unpack the component_data dictionary)
                        component_instance = component_class(**data)
                        self.components[technology + "_" + data['label']] = component_instance
                        feuls.append(data['fuel_type'])
                else:
                    # Instantiate the component with the required parameters (unpack the component_data dictionary)
                    component_instance = component_class(**component_data)
                    self.components[technology + "_" + component_data['label']] = component_instance
                    feuls.append(component_data['fuel_type'])
        #Add ancillary components: grids, demands
        #Get the electrical bidirectional grid interface
        grid_data = {
            'label': 'Electricity',
            'compensation': 0.02,#ToDo: Update this value with the market clearing result
            'co2_factor': 0,
            'constrain_flow': True,
            'feedin_limit': None,
        }
        component_class = comando_dst['grid']
        component_instance = component_class(**grid_data)
        self.components["grid_Electricity"] = component_instance
        #Get other energy sources for multi-energy unit depending on the fuels they need
        if 'gas' in feuls:
            grid_data = {
                'label': 'Gas',
                'compensation': 0,
                'co2_factor': 0.201,
            }
            component_class = comando_dst['grid']
            component_instance = component_class(**grid_data)
            self.components["grid_Gas"] = component_instance
        # Get demands the multi-energy unit needs to cover
        # for energy in self.demand.columns:
        dsm_demand_forecasts = self.get_dsm_forecasts()

        for forecast_key, forecast_series in dsm_demand_forecasts.items():
            energy_type = forecast_key.removeprefix("dsm_").split("Demand")[0].strip()

            component_class = comando_dst["demand"]
            component_instance = component_class(energy_type)

            self.components[f"demand_{energy_type}"] = component_instance


    def setup_model(self, presolve=True):
        # Initialize the Pyomo model
        # along with optimal and flexibility constraints and the objective functions

        self.optimisation_counter = 0

        self.initialize_components()
        self.initialize_energy_system()#Define the specific connection between components within a unit.

        self.define_constraints()# Define extra constraints for this specific unit(Comando System).

        self.opt_model = self.create_problem()# Read relevant parameter and create comando problem

    def get_dsm_forecasts(self) -> dict:
        """
        Return all DSM-related forecast time series from the forecaster.

        DSM forecasts are identified by the prefix 'dsm_'.
        The comparison is case-insensitive to avoid issues with capitalization.
        """
        return {
            key: value
            for key, value in self.forecaster.forecasts.items()
            if key.lower().startswith("dsm_")
        }
    def determine_optimal_operation_without_flex(self):
        pass
    def determine_optimal_operation_with_flex(self):#ToDo: rename the function in strategy
        # ToDo: Update the electricity price in rolling horizon mode
        # ToDo: ASSUME is working with MWh, check the unit during market clearing

        print("Solving...")
        options = dict(  # Options assuming Gurobi 9.1.1
            Seed=123,
            NonConvex=2,
            MIPGap=0,
            MIPFocus=2,
            OutputFlag=1,
        )
        self.opt_model.solve(**options)
        opt_power_requirement = [
            self.opt_model.getVarByName(f"Electricity_consumption[{t}]").X
            - self.opt_model.getVarByName(f"Electricity_feedin[{t}]").X
            for t in range(len(self.index))
        ]

        self.opt_power_requirement = FastSeries(
            index=self.index,
            value=opt_power_requirement,
        )
        if self.opt_model.SolCount > 0:
            # Plot timeseries of variables from gurobi result as the user choose
            # interactive_timeseries_plot(self.opt_model,self.index)
            # raise SystemExit("Stopping simulation here")#Fixme: Only for review the result
            pass
        else:
            print("No feasible solution found. Status:", self.opt_model.Status)
        print("Solving...")
