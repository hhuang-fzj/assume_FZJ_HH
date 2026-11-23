from copy import deepcopy
from inspect import signature
from math import isfinite

from comando.core import System

from assume.common.fast_pandas import FastSeries
from assume.units.comando_components import comando_dst
from assume.common.temp_gurobi_visulize import interactive_timeseries_plot
from assume.common.gurobi_diagnoser import optimize_and_diagnose
# temperary plot function for gurobi model results

# import pickle
# import os

class ComandoFacade:

    market_interfaces = {
        "LLEC": ("grid_Electricity", "energy"),
        "HTH": ("grid_HT_Heat", "ht_heat"),
        "LTH": ("grid_LT_Heat", "lt_heat"),
    }

    def market_bid_blocks(self, start, market_config, net_import, pricing_config):
        """Describe one unsplit planned position; subclasses may supply intentions.

        Quantities are unsigned MW; the strategy derives direction from net_import.
        price_rule is configured or market_max. Labeled blocks may request price
        limits and a pricing_rule audit label; they also receive distinct bid IDs.
        This hook describes the existing plan without modifying or solving it.
        """
        return [{"quantity_mw": abs(net_import), "price_rule": "configured"}]

    def __init__(self, components, comando_config=None, comando_config_path=None, **kwargs):
        super().__init__(**kwargs)
        self.comando_config_path = comando_config_path or "<unit parameters>"
        self.validate_comando_config(comando_config, self.comando_config_path, self.id)
        self.comando_config = deepcopy(comando_config)
        self.components = deepcopy(components or {})
        self.horizon_mode = 'full_horizon'
        self.submitted_orders = []
        self.accepted_commitments = []
        self._clearing_rounds = {}
        self._accepted_positions = {}
        self._reconciled_values = {}
        self._reconciliation_finalized = False
        self._cost_recovery_cache = {}
        self.cost_recovery_audit = []
        self._dsm_result_buffers = {
            "dsm_component_dispatch": [], "dsm_market_positions": [], "dsm_reconciliation": [],
        }

    @staticmethod
    def validate_comando_config(config, path, unit_id):
        """Validate static YAML settings without constructing components."""
        context = f"{path}: unit {unit_id!r}"
        if not isinstance(config, dict) or set(config) != {"grids", "connections"}:
            raise ValueError(f"{context}: expected grids and connections mappings")
        grids, connections = config["grids"], config["connections"]
        if not isinstance(grids, dict) or not isinstance(connections, dict) or not connections:
            raise ValueError(f"{context}: grids must be a mapping and connections a nonempty mapping")
        allowed = set(signature(comando_dst["grid"]).parameters) - {"label"}
        for key, params in grids.items():
            where = f"{context}: grid {key!r}"
            if not isinstance(key, str) or not key.startswith("grid_"):
                raise ValueError(f"{where}: expected a grid_ component key")
            if not isinstance(params, dict) or set(params) - allowed:
                raise ValueError(f"{where}: invalid Grid constructor parameters (label is generated)")
            for name, value in params.items():
                if name in {"single_connector", "constrain_flow"}:
                    valid = isinstance(value, bool)
                else:
                    valid = (value is None and name in {"consumption_limit", "feedin_limit"}) or (
                        isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
                        and (not name.endswith("_limit") or value >= 0)
                    )
                if not valid:
                    raise ValueError(f"{where}: invalid static setting {name!r}")
        assigned = {}
        for bus, endpoints in connections.items():
            if not isinstance(bus, str) or not bus or not isinstance(endpoints, list) or not endpoints:
                raise ValueError(f"{context}: bus {bus!r}: expected a nonempty endpoint list")
            for endpoint in endpoints:
                where = f"{context}: bus {bus!r}: endpoint {endpoint!r}"
                if (not isinstance(endpoint, (list, tuple)) or len(endpoint) != 2
                        or not all(isinstance(item, str) and item.isidentifier() for item in endpoint)):
                    raise ValueError(f"{where}: expected [component_key, connector_name] identifiers")
                pair = tuple(endpoint)
                if pair in assigned:
                    raise ValueError(f"{where}: duplicate endpoint; already assigned to bus {assigned[pair]!r}")
                assigned[pair] = bus

    def component_label(self, key):
        """Keep familiar grid labels, with independent labels in every unit."""
        suffix = key.removeprefix("grid_")
        return f"{self.id}_{suffix}"

    def initialize_components(self):
        equipment = deepcopy(self.components)
        self.components = {}
        for technology, records in equipment.items():
            if technology not in comando_dst or technology in {"grid", "demand"}:
                raise ValueError(f"{self.comando_config_path}: unit {self.id!r}: unsupported equipment {technology!r}")
            for record in records if isinstance(records, list) else [records]:
                params = deepcopy(record)
                label = params["label"]
                key = f"{technology}_{label}"
                if key in self.components:
                    raise ValueError(f"{self.comando_config_path}: unit {self.id!r}: duplicate component {key!r}")
                # Keep input keys stable; namespace COMANDO labels across units.
                params["label"] = label if label.startswith(f"{self.id}_") else f"{self.id}_{label}"
                self.components[key] = comando_dst[technology](**params)
        for key, record in self.comando_config["grids"].items():
            params = deepcopy(record)
            params["label"] = self.component_label(key)
            self.components[key] = comando_dst["grid"](**params)

    def initialize_demands(self):
        """Unit classes create their own demands after equipment and grids."""

    def initialize_energy_system(self):
        connections = {}
        for bus, endpoints in self.comando_config["connections"].items():
            connections[bus] = []
            for key, port in endpoints:
                context = (f"{self.comando_config_path}: unit {self.id!r}: "
                           f"bus {bus!r}: endpoint {[key, port]!r}")
                if key not in self.components:
                    raise ValueError(f"{context}: unknown component (equipment is not created automatically)")
                component = self.components[key]
                if port not in component.connectors:
                    raise ValueError(f"{context}: unknown or disabled connector")
                connections[bus].append(component.connectors[port])
        self.comando_system = System(
            label=f"{self.id}_{self.technology}",
            components=list(self.components.values()), connections=connections,
        )

    def parameter_name(self, component_key, parameter):
        """Use actual symbol names after per-unit label generation."""
        return self.components[component_key][parameter].name

    def bind_series(self, data, component_key, parameter, series):
        """Bind forecasts to real symbols; disabled grid directions have none."""
        settings = self.comando_config["grids"].get(component_key, {})
        if parameter == "price" and settings.get("consumption_limit") == 0:
            return
        if parameter == "compensation" and settings.get("feedin_limit", 0) == 0:
            return
        data[self.parameter_name(component_key, parameter)] = series.as_pd_series()

    def annual_fixed_cost_surcharge(self, key, rated_output, full_load_hours, allocation=1.0):
        """Annual fixed O&M only; never infer costs from TAC or annualize capex here.

        IESComponent defines fixed_costs = c_m * investment_cost. The existing
        make_tac_objective uses this as its annual fixed-cost term, separately
        from annuitized investment. Cache by expression, parameters and allocation.
        """
        component = self.components[key]
        try:
            expression = component.get_expression("fixed_costs")
        except KeyError as exc:
            raise ValueError(f"{self.id}: {key}: cost-recovery pricing requires fixed_costs [EUR/year]") from exc
        if get_vars(expression):
            raise ValueError(f"{self.id}: {key}: fixed_costs must be independent of operating/design decisions")
        values = [rated_output, full_load_hours, allocation]
        if (not np.isfinite(values).all() or rated_output <= 0 or full_load_hours <= 0
                or not 0 <= allocation <= 1):
            raise ValueError(f"{self.id}: {key}: invalid rated MW, annual full-load hours or cost allocation")
        parameters = tuple(sorted((symbol.name, float(symbol.value)) for symbol in expression.free_symbols))
        signature = (str(expression), parameters, float(rated_output), float(full_load_hours), float(allocation))
        cache = self._cost_recovery_cache
        cache_key = (key, float(rated_output), float(full_load_hours), float(allocation))
        if cache.get(cache_key, {}).get("signature") != signature:
            annual = float(evaluate(expression))
            if not np.isfinite(annual) or annual < 0:
                raise ValueError(f"{self.id}: {key}: fixed_costs must be finite nonnegative EUR/year")
            cache[cache_key] = {"signature": signature, "annual_fixed_cost": annual,
                          "fixed_cost_expression": str(expression),
                          "fixed_cost_source": f"{component.label}.fixed_costs",
                          "fixed_surcharge": allocation * annual / (rated_output * full_load_hours),
                          "rated_output_mw": float(rated_output), "full_load_hours": float(full_load_hours),
                          "allocation": float(allocation)}
        return {name: value for name, value in cache[cache_key].items() if name != "signature"}

    def cost_recovery_timestep(self, start):
        index = self.index.as_datetimeindex()
        step = index.get_indexer([start])[0]
        if step < 0:
            raise ValueError(f"{self.id}: cost-recovery price outside forecast horizon at {start}")
        return int(step)

    def required_cost_recovery_input(self, config, name):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{self.id}: configure explicit finite cost_recovery.{name}; missing costs are not zero")
        return float(value)

    def setup_model(self, presolve=True):
        # Initialize the Pyomo model
        # along with optimal and flexibility constraints and the objective functions

        self.optimisation_counter = 0

        self.initialize_components()
        self.initialize_demands()
        self.initialize_energy_system()#Define the specific connection between components within a unit.

        self.define_constraints()# Define extra constraints for this specific unit(Comando System).

        self.opt_model = self.create_problem()# Read relevant parameter and create comando problem

        #status = optimize_and_diagnose(self.opt_model)
    def get_dsm_forecasts(self) -> dict:
        """Return DSM demand series keyed by their input column names."""
        return self.forecaster.dsm_demands
    def determine_optimal_operation_without_flex(self):
        pass
    def determine_optimal_operation_with_flex(self):
        #ToDo: rename the function in strategy
        #ToDo: Update the price in rolling horizon mode

        print("Solving...")
        options = dict(  # Options assuming Gurobi 9.1.1
            Seed=123,
            NonConvex=2,
            MIPGap=0,
            MIPFocus=2,
            OutputFlag=1,
        )
        self.opt_model.solve(**options)
        #temporary Gurobi evaluation
        if self.opt_model.SolCount > 0:
            # Plot timeseries of variables from gurobi result as the user choose
            #interactive_timeseries_plot(self.opt_model,self.index)
            #raise SystemExit("Stopping simulation here")#Fixme: Only for review the result
            pass
        else:
            print("No feasible solution found. Status:", self.opt_model.Status)
            status = optimize_and_diagnose(
                model= self.opt_model,
                file_prefix="HT_Heat", #Identifier for the scenario
            )
        print("Solving...")

        if self.opt_model.SolCount <= 0:
            raise RuntimeError(f"Unit {self.id}: no solution available; status {self.opt_model.Status}")
        # Resolve symbols from actual configured grids, including disabled ports.
        for key, output in [("grid_Electricity", "opt_power_volume"),
                            ("grid_HT_Heat", "opt_HT_heat_volume"),
                            ("grid_LT_Heat", "opt_LT_heat_volume")]:
            if key not in self.components:
                continue
            grid = self.components[key]
            values = []
            for t in range(len(self.index)):
                value = 0
                for variable, sign in [("consumption", 1), ("feedin", -1)]:
                    try:
                        symbol = grid[variable]
                    except KeyError:  # Direction disabled by a zero limit.
                        continue
                    value += sign * self.opt_model.getVarByName(f"{symbol.name}[{t}]").X
                values.append(value)
            setattr(self, output, FastSeries(index=self.index, value=values))
