# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Scenario generation from a SimBench low-voltage grid for the local retailer study.

The script reads the SimBench CSV export of one low-voltage grid and writes a complete
ASSUME scenario folder. Each SimBench node is one connection point of the district:

- A node with load only becomes a pure demand unit.
- A node with load and a rooftop PV system becomes a prosumer (building unit).
- A PV system above ``COMMUNITY_PV_THRESHOLD_MW`` is not a rooftop system. It is
  modelled as a community PV plant and acts as the supply side of the local energy
  market (LEM), while the load of its node stays a pure demand unit.

Pipeline:
    1. **Read** the SimBench tables.
    2. **Scale** each profile by the installed power of its element.
    3. **Aggregate** loads per node and resample to the simulation resolution.
    4. **Split** the nodes into prosumers, pure demand and community PV.
    5. **Write** the ASSUME scenario folder.

Note:
    Forecast and realised values are identical in this version (perfect foresight).
    Forecast errors are out of scope, so deviations only arise from partial execution
    of LEM bids.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all modelling assumptions are collected here)
# ---------------------------------------------------------------------------
SIMBENCH_PATH = Path(__file__).parent / "simbench_raw"
SCENARIO_PATH = Path(__file__).parent

SIMULATION_START = "2016-01-01 00:00"  # SimBench profiles cover the year 2016
SIMULATION_END = "2016-01-08 00:00"
RESOLUTION = "1h"

# PV systems above this rated power are community plants, not rooftop systems (MW)
COMMUNITY_PV_THRESHOLD_MW = 0.05

# Number of pure demand nodes that belong to the aggregator portfolio. The remaining
# demand nodes bid on the LEM on their own and are the aggregator's competitors.
AGGREGATOR_DEMAND_NODES = 15

# Opportunity cost of local sellers in EUR/MWh: a prosumer does not sell locally below
# the feed-in tariff it would receive from the grid (assumption, source required)
FEED_IN_TARIFF_EUR_MWH = 80.0

# Share of the district that belongs to the aggregator. The remaining nodes bid on
# the LEM themselves and are the aggregator's competitors there.
N_AGGREGATOR_DEMAND_NODES = 15

STUDY_CASE = "local_retailer_demo"
OPERATOR_AGGREGATOR = "local_retailer"
OPERATOR_COMMUNITY = "LEC_operator"
OPERATOR_INDEPENDENT = "independent_operator"
OPERATOR_STORAGE = "storage_operator"
OPERATOR_GRID = "Grid_operator"
OPERATOR_INDEPENDENT = "independent_operator"
OPERATOR_STORAGE = "storage_operator"
PORTFOLIO_STRATEGY = "units_operator_energy_coordinated_local_retailer"

LEM_MARKET_ID = "LEM_DA"
WM_MARKET_ID = "WM_DA"
EOM_MARKET_ID = "EOM"  # placeholder, the building model reads the price of "EOM"

# Opportunity cost of local sellers: they do not sell locally below the feed-in
# tariff they would receive from the grid (assumption, source required)
PV_OPPORTUNITY_COST_EUR_MWH = 80.0

# Willingness to pay of the independent consumers in the LEM, approximately the
# retail price they would otherwise pay (assumption, source required)
INDEPENDENT_DEMAND_PRICE_EUR_MWH = 250.0

# The aggregator bids at the price cap to secure acceptance, following the
# price-taker logic of Haghifam et al. (2022)
AGGREGATOR_BUY_PRICE_EUR_MWH = 3000.0

WM_PRICE_EUR_MWH = 80.0  # placeholder until ENTSO-E day-ahead prices are used
EOM_PRICE_EUR_MWH = 80.0
WM_COUNTERPARTY_VOLUME_MW = 100.0  # infinite counterparty of the wholesale market


def load_simbench_tables(path: Path) -> dict[str, pd.DataFrame]:
    """
    Read the SimBench CSV tables needed for the scenario.

    Args:
        path (Path): Folder with the SimBench CSV export of one grid.

    Returns:
        dict[str, pd.DataFrame]: Tables ``Load``, ``RES``, ``Storage``,
        ``LoadProfile`` and ``RESProfile``. Profile tables are indexed by timestamp.

    Raises:
        FileNotFoundError: If one of the tables is missing.
    """
    tables = {}
    for name in ["Load", "RES", "Storage"]:
        tables[name] = pd.read_csv(path / f"{name}.csv", sep=";")
    for name in ["LoadProfile", "RESProfile", "StorageProfile"]:
        df = pd.read_csv(path / f"{name}.csv", sep=";")
        df["time"] = pd.to_datetime(df["time"], format="%d.%m.%Y %H:%M")
        tables[name] = df.set_index("time")
    return tables


def build_node_load(
    load: pd.DataFrame, load_profile: pd.DataFrame, resolution: str
) -> pd.DataFrame:
    """
    Aggregate the load time series of all elements to one time series per node.

    A node can carry several load elements, for example a household profile, a heat
    pump and an electric vehicle charger. They belong to the same connection point and
    are therefore summed up.

    Args:
        load (pd.DataFrame): SimBench ``Load`` table.
        load_profile (pd.DataFrame): SimBench ``LoadProfile`` table.
        resolution (str): Target resolution, e.g. ``"1h"``.

    Returns:
        pd.DataFrame: Active power in MW, one column per node.
    """
    per_element = {
        row.id: load_profile[f"{row.profile}_pload"] * row.pLoad
        for row in load.itertuples()
    }
    element_power = pd.DataFrame(per_element)
    node_of_element = load.set_index("id")["node"]
    node_power = element_power.T.groupby(node_of_element).sum().T
    return node_power.resample(resolution).mean()


def build_pv_power(
    res: pd.DataFrame, res_profile: pd.DataFrame, resolution: str
) -> pd.DataFrame:
    """
    Build the PV generation time series per node.

    Args:
        res (pd.DataFrame): SimBench ``RES`` table.
        res_profile (pd.DataFrame): SimBench ``RESProfile`` table.
        resolution (str): Target resolution, e.g. ``"1h"``.

    Returns:
        pd.DataFrame: Active power in MW, one column per node with a PV system.
    """
    pv = res[res["type"] == "PV"]
    per_node = {
        row.node: res_profile[row.profile] * row.pRES for row in pv.itertuples()
    }
    return pd.DataFrame(per_node).resample(resolution).mean()


def unit_id_of_node(node: str, prefix: str) -> str:
    """
    Build a short ASSUME unit id from a SimBench node name.

    Args:
        node (str): SimBench node name, e.g. ``"LV4.101 Bus 18"``.
        prefix (str): Prefix of the unit id, e.g. ``"House"``.

    Returns:
        str: Unit id, e.g. ``"House_Bus18"``.
    """
    return f"{prefix}_{node.split(' ', 1)[1].replace(' ', '')}"


def split_nodes(
    node_load: pd.DataFrame, res: pd.DataFrame
) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
    """
    Split the nodes into the aggregator portfolio, independent participants and
    community PV plants.

    The aggregator does not serve the whole district. It serves all prosumers and the
    first ``AGGREGATOR_DEMAND_NODES`` pure demand nodes. The remaining demand nodes bid
    on the LEM themselves and are the aggregator's competitors, following the market
    setup of Haghifam et al. (2022), where the aggregator faces non-strategic rivals.

    Args:
        node_load (pd.DataFrame): Load per node from :func:`build_node_load`.
        res (pd.DataFrame): SimBench ``RES`` table.

    Returns:
        tuple[list[str], list[str], list[str], pd.DataFrame]: Prosumer nodes of the
        aggregator, pure demand nodes of the aggregator, independent demand nodes and
        the rows of ``res`` that are treated as community PV plants.
    """
    pv = res[res["type"] == "PV"]
    community_pv = pv[pv.pRES > COMMUNITY_PV_THRESHOLD_MW]
    rooftop_nodes = set(pv[pv.pRES <= COMMUNITY_PV_THRESHOLD_MW]["node"])

    prosumer_nodes = sorted(n for n in node_load.columns if n in rooftop_nodes)
    demand_nodes = sorted(n for n in node_load.columns if n not in rooftop_nodes)

    aggregator_demand = demand_nodes[:AGGREGATOR_DEMAND_NODES]
    independent_demand = demand_nodes[AGGREGATOR_DEMAND_NODES:]
    return prosumer_nodes, aggregator_demand, independent_demand, community_pv


def write_unit_tables(
    prosumer_nodes: list[str],
    aggregator_demand: list[str],
    independent_demand: list[str],
    community_pv: pd.DataFrame,
    node_load: pd.DataFrame,
    storage: pd.DataFrame,
    res: pd.DataFrame,
) -> None:
    """
    Write the unit definition tables of the scenario.

    Written files:
        - ``demand_units.csv``: pure demand units of the aggregator and of the
          independent participants.
        - ``residential_dsm_units.csv``: one building unit per prosumer node.
        - ``powerplant_units.csv``: the community PV plants, supply side of the LEM.
        - ``storage_units.csv``: the SimBench storage units, trading on the LEM.
        - ``exchange_units.csv``: the wholesale market counterparty.
        - ``unit_operators.csv``: the aggregator and its portfolio strategy.

    Args:
        prosumer_nodes (list[str]): Nodes with load and rooftop PV.
        aggregator_demand (list[str]): Pure demand nodes served by the aggregator.
        independent_demand (list[str]): Pure demand nodes bidding on their own.
        community_pv (pd.DataFrame): Rows of ``RES`` treated as community PV.
        node_load (pd.DataFrame): Load per node in MW.
        storage (pd.DataFrame): SimBench ``Storage`` table.
        res (pd.DataFrame): SimBench ``RES`` table.
    """

    def demand_row(node: str, operator: str) -> dict:
        """Build one row of ``demand_units.csv`` for a node."""
        return {
            "name": unit_id_of_node(node, "Demand"),
            "technology": "inflex_demand",
            f"bidding_{LEM_MARKET_ID}": "demand_energy_naive",
            f"bidding_{WM_MARKET_ID}": "demand_energy_naive"
            if operator == OPERATOR_AGGREGATOR
            else "",
            "max_power": round(node_load[node].max() * 2, 6),
            "min_power": 0,
            "unit_operator": operator,
        }

    # 1. Pure demand: aggregator portfolio and independent competitors
    demand_rows = [demand_row(n, OPERATOR_AGGREGATOR) for n in aggregator_demand]
    demand_rows += [demand_row(n, OPERATOR_INDEPENDENT) for n in independent_demand]
    pd.DataFrame(demand_rows).to_csv(SCENARIO_PATH / "demand_units.csv", index=False)

    # 2. Prosumers as building units with one PV component each
    rated_pv = res.set_index("node")["pRES"]
    prosumer_rows = [
        {
            "name": unit_id_of_node(node, "House"),
            "unit_type": "building",
            "technology": "pv_plant",
            "node": "node0",
            f"bidding_{LEM_MARKET_ID}": "household_energy_optimization",
            f"bidding_{WM_MARKET_ID}": "household_energy_optimization",
            "unit_operator": OPERATOR_AGGREGATOR,
            "objective": "min_variable_cost",
            "flexibility_measure": "cost_based_load_shift",
            "cost_tolerance": 10,
            "is_prosumer": "Yes",
            "max_power": round(float(rated_pv[node]), 6),
            "uses_power_profile": "Yes",
        }
        for node in prosumer_nodes
    ]
    pd.DataFrame(prosumer_rows).to_csv(
        SCENARIO_PATH / "residential_dsm_units.csv", index=False
    )

    # 3. Community PV as the supply side of the LEM. Its bid price is the feed-in
    #    tariff, the opportunity cost of selling to the grid instead of the LEM.
    plant_rows = [
        {
            "name": unit_id_of_node(row.node, "CommunityPV"),
            "technology": "solar",
            f"bidding_{LEM_MARKET_ID}": "powerplant_energy_naive",
            f"bidding_{WM_MARKET_ID}": "",
            f"bidding_{EOM_MARKET_ID}": "",
            "unit_operator": OPERATOR_COMMUNITY,
            "fuel_type": "renewable",
            "emission_factor": 0,
            "max_power": round(float(row.pRES), 6),
            "min_power": 0,
            "efficiency": 1,
            "additional_cost": FEED_IN_TARIFF_EUR_MWH,
        }
        for row in community_pv.itertuples()
    ]
    pd.DataFrame(plant_rows).to_csv(SCENARIO_PATH / "powerplant_units.csv", index=False)

    # 4. Storage units trade on the LEM and make its price time dependent
    storage_rows = [
        {
            "name": f"Storage_{i + 1}",
            "technology": "battery_storage",
            f"bidding_{LEM_MARKET_ID}": "flexable_eom_storage",
            "unit_operator": OPERATOR_STORAGE,
            "max_power_charge": round(abs(float(row.pStor)), 6),
            "max_power_discharge": round(abs(float(row.pStor)), 6),
            "capacity": round(float(row.eStore), 6),
            "efficiency_charge": float(row.etaStore),
            "efficiency_discharge": float(row.etaStore),
            "initial_soc": 0.5,
        }
        for i, row in enumerate(storage.itertuples())
    ]
    pd.DataFrame(storage_rows).to_csv(SCENARIO_PATH / "storage_units.csv", index=False)

    # 5. Wholesale market counterparty (price taker assumption)
    pd.DataFrame(
        [
            {
                "name": "GridConnection_WM",
                f"bidding_{WM_MARKET_ID}": "exchange_energy_naive",
                "price_import": 0,
                "price_export": 2999,
                "unit_operator": OPERATOR_GRID,
            }
        ]
    ).to_csv(SCENARIO_PATH / "exchange_units.csv", index=False)

    # 6. The aggregator bids for its whole portfolio on both markets
    pd.DataFrame(
        [
            {
                "name": OPERATOR_AGGREGATOR,
                f"bidding_{LEM_MARKET_ID}": PORTFOLIO_STRATEGY,
                f"bidding_{WM_MARKET_ID}": PORTFOLIO_STRATEGY,
            }
        ]
    ).to_csv(SCENARIO_PATH / "unit_operators.csv", index=False)


def write_profile_files(
    prosumer_nodes: list[str],
    aggregator_demand: list[str],
    independent_demand: list[str],
    community_pv: pd.DataFrame,
    node_load: pd.DataFrame,
    pv_power: pd.DataFrame,
    res_profile: pd.DataFrame,
) -> None:
    """
    Write the time series files of the scenario.

    Written files:
        - ``demand_df.csv``: load of the pure demand units in MW.
        - ``forecasts_df.csv``: load and PV profiles of the prosumers, and the
          exogenous prices. No LEM columns, so ASSUME calculates the LEM price itself.
        - ``actuals_df.csv``: realised values, identical to the forecast in this version.
        - ``exchanges_df.csv``: available volume of the wholesale counterparty.
        - ``availability_df.csv``: capacity factor of the community PV plants.

    Args:
        prosumer_nodes (list[str]): Nodes with load and rooftop PV.
        aggregator_demand (list[str]): Demand nodes of the aggregator.
        independent_demand (list[str]): Demand nodes bidding on the LEM themselves.
        community_pv (pd.DataFrame): Rows of ``RES`` treated as community PV.
        node_load (pd.DataFrame): Load per node in MW.
        pv_power (pd.DataFrame): PV generation per node in MW.
        res_profile (pd.DataFrame): SimBench ``RESProfile`` table.
    """
    index = node_load.loc[SIMULATION_START:SIMULATION_END].index

    # 1. Demand of the aggregator and of the independent consumers
    demand_series = {
        unit_id_of_node(node, "Demand"): node_load.loc[index, node]
        for node in aggregator_demand
    }
    demand_series.update(
        {
            unit_id_of_node(node, "Demand"): node_load.loc[index, node]
            for node in independent_demand
        }
    )
    demand_df = pd.DataFrame(demand_series)
    demand_df.index.name = "datetime"
    demand_df.round(9).to_csv(SCENARIO_PATH / "demand_df.csv")

    # 2. Prosumers and exogenous prices
    forecasts = {}
    for node in prosumer_nodes:
        house = unit_id_of_node(node, "House")
        forecasts[f"{house}_load_profile"] = node_load.loc[index, node]
        # The building model multiplies the profile by max_power, so it is per unit
        forecasts[f"{house}_pv_profile"] = (
            res_profile.resample(RESOLUTION).mean().loc[index, "PV5"]
        )
    forecasts[f"price_{WM_MARKET_ID}"] = pd.Series(WM_PRICE_EUR_MWH, index=index)
    forecasts[f"residual_load_{WM_MARKET_ID}"] = node_load.loc[
        index, aggregator_demand + prosumer_nodes
    ].sum(axis=1)
    forecasts[f"price_{EOM_MARKET_ID}"] = pd.Series(EOM_PRICE_EUR_MWH, index=index)
    forecasts[f"residual_load_{EOM_MARKET_ID}"] = pd.Series(0.0, index=index)

    forecasts_df = pd.DataFrame(forecasts)
    forecasts_df.index.name = "datetime"
    forecasts_df.round(9).to_csv(SCENARIO_PATH / "forecasts_df.csv")

    # 3. Realised values, equal to the forecast (perfect foresight)
    actuals = {
        f"demand_actual_{unit_id_of_node(n, 'Demand')}": node_load.loc[index, n]
        for n in aggregator_demand
    }
    for node in prosumer_nodes:
        house = unit_id_of_node(node, "House")
        actuals[f"load_actual_{house}"] = node_load.loc[index, node]
        actuals[f"pv_actual_{house}"] = pv_power.loc[index, node]
    actuals_df = pd.DataFrame(actuals)
    actuals_df.index.name = "datetime"
    actuals_df.round(9).to_csv(SCENARIO_PATH / "actuals_df.csv")

    # 4. Counterparty volumes of the wholesale market
    exchanges = pd.DataFrame(
        {
            "GridConnection_WM_export": WM_COUNTERPARTY_VOLUME_MW,
            "GridConnection_WM_import": WM_COUNTERPARTY_VOLUME_MW,
        },
        index=index,
    )
    exchanges.index.name = "datetime"
    exchanges.to_csv(SCENARIO_PATH / "exchanges_df.csv")

    # 5. Availability of the community PV plants
    availability = pd.DataFrame(
        {
            unit_id_of_node(row.node, "CommunityPV"): res_profile.resample(RESOLUTION)
            .mean()
            .loc[index, row.profile]
            for row in community_pv.itertuples()
        }
    )
    availability.index.name = "datetime"
    availability.round(9).to_csv(SCENARIO_PATH / "availability_df.csv")


def write_config() -> None:
    """
    Write ``config.yaml`` with the LEM, the wholesale market and the EOM placeholder.

    The LEM and the wholesale market open in parallel, so the aggregator has to split
    its volume before either market is cleared. No price column is provided for the
    LEM, so ASSUME calculates the LEM price forecast from the merit order.
    """
    market = {
        "operator": "market_operator",
        "product_type": "energy",
        "start_date": SIMULATION_START,
        "products": [{"duration": "1h", "count": 24, "first_delivery": "24h"}],
        "opening_frequency": "24h",
        "opening_duration": "15min",
        "volume_unit": "MWh",
        "maximum_bid_volume": 100000,
        "maximum_bid_price": 3000,
        "minimum_bid_price": -500,
        "price_unit": "EUR/MWh",
        "market_mechanism": "pay_as_clear",
    }
    config = {
        STUDY_CASE: {
            "start_date": SIMULATION_START,
            "end_date": SIMULATION_END,
            "time_step": RESOLUTION,
            "save_frequency_hours": 48,
            "markets_config": {
                LEM_MARKET_ID: dict(market, operator="LEM_operator"),
                WM_MARKET_ID: dict(market, operator="WM_operator"),
                EOM_MARKET_ID: dict(market, operator="EOM_operator"),
            },
        }
    }
    with open(SCENARIO_PATH / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)


def main() -> None:
    """Convert the SimBench grid into a complete ASSUME scenario folder."""
    # 1. Read the SimBench tables
    tables = load_simbench_tables(SIMBENCH_PATH)

    # 2. Build the time series per node
    node_load = build_node_load(tables["Load"], tables["LoadProfile"], RESOLUTION)
    pv_power = build_pv_power(tables["RES"], tables["RESProfile"], RESOLUTION)

    # 3. Split the nodes between the aggregator, independent participants and the LEM
    #    supply side
    prosumer_nodes, aggregator_demand, independent_demand, community_pv = split_nodes(
        node_load, tables["RES"]
    )
    logger.info(
        f"aggregator: {len(prosumer_nodes)} prosumers and "
        f"{len(aggregator_demand)} demand nodes | "
        f"independent: {len(independent_demand)} demand nodes | "
        f"LEM supply: {len(community_pv)} community PV, {len(tables['Storage'])} storage"
    )

    # 4. Write the scenario
    write_unit_tables(
        prosumer_nodes,
        aggregator_demand,
        independent_demand,
        community_pv,
        node_load,
        tables["Storage"],
        tables["RES"],
    )
    write_profile_files(
        prosumer_nodes,
        aggregator_demand,
        independent_demand,
        community_pv,
        node_load,
        pv_power,
        tables["RESProfile"],
    )
    write_config()
    logger.info(f"Scenario written to {SCENARIO_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
