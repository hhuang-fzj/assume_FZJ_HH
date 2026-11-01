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

# All time stamps of the scenario are in UTC (naive, without time zone information).
# ASSUME itself works in UTC: timestamp2datetime() and datetime2timestamp() in
# assume/common/utils.py convert with timezone.utc and calendar.timegm. Every input
# series (SimBench, SMARD, reBAP) is converted from German local time to UTC by
# to_simulation_time(). Days of the simulation are UTC days (00:00-24:00 UTC, i.e.
# 02:00-02:00 German summer time) [OWN].
SIMULATION_START = "2016-06-19 00:00"  # UTC; SimBench profiles cover the year 2016
SIMULATION_END = "2016-06-27 00:00"  # UTC
# 15-minute resolution everywhere (decision of the thesis, 15.10.2026). SimBench profiles
# are 15 min ([SimBenchFmt] slide 3); since 1 Oct 2025 the day-ahead market (SDAC) uses
# 15-minute products; the 2016 hourly prices are held constant within the hour [OWN].
RESOLUTION = "15min"

# PV systems above this rated power are not household rooftop systems but plants that
# market their power directly; here they are the community PV plants on the LEM (MW).
#   Threshold 100 kW: Paragraph 37 Abs. 2 Nr. 2 EEG 2014 (version in force until
#   31.12.2016): the feed-in tariff is only paid to plants commissioned after
#   31.12.2015 with an installed capacity of at most 100 kW [OK, consolidated text
#   buzer.de]; larger plants must market their power directly. For plants commissioned
#   before 2016 the limit was 500 kW (Nr. 1). SimBench gives no commissioning year:
#   all plants are treated as commissioned in 2016 [OWN assumption].
#   b22d8b5 used 0.05 MW without a source. In this grid both values give the same split:
#   rooftop PV 6.5-13 kW (Bus 1, 4, 18, 19, 21), community PV 128.8 kW (Bus 41).
COMMUNITY_PV_THRESHOLD_MW = 0.1

# Which nodes belong to the aggregator portfolio is read from the SimBench profile names.
#   SimBench marks every load element as household or commercial customer: "The
#   consumer profiles consist of five household based time series (H0-A to H0-L) and
#   22 RLM based series. The latter are sorted by matching SLP type (G0- to L2-)"
#   (Spalthoff et al. 2019, Sect. 3.2). One node = one final customer [OWN]. A node
#   with a G or L profile is a commercial customer and bids on the LEM on its own
#   (independent participant); every other node is a household customer of the
#   aggregator. In this grid no node mixes H0 and G profiles.
#   Check with Paragraph 3 Nr. 57 EnWG (household customer = household use, or
#   commercial use of at most 10 000 kWh per year): a commercial node at or below the
#   limit would be a household customer; split_nodes() stops with an error in that
#   case. In this grid the smallest commercial node uses 11.76 MWh (Bus 44, 2016).
#   b22d8b5 used the first 15 pure demand nodes in string order (no justification);
#   that rule was removed on 2026-10-29.
HOUSEHOLD_LIMIT_MWH_PER_YEAR = 10.0  # 10 000 kWh, Paragraph 3 Nr. 57 EnWG
COMMERCIAL_PROFILE_PREFIXES = ("G", "L")  # SimBench: commercial (G) and agricultural (L)

# Opportunity cost of local sellers in EUR/MWh: a prosumer does not sell locally below
# the feed-in tariff it would receive from the grid (assumption, source required)
STUDY_CASE = "local_retailer_demo"
OPERATOR_AGGREGATOR = "local_retailer"
OPERATOR_COMMUNITY = "LEC_operator"
OPERATOR_INDEPENDENT = "independent_operator"
OPERATOR_STORAGE = "storage_operator"
OPERATOR_GRID = "Grid_operator"
PORTFOLIO_STRATEGY = "units_operator_energy_coordinated_local_retailer"

LEM_MARKET_ID = "LEM_DA"
WM_MARKET_ID = "WM_DA"
EOM_MARKET_ID = "EOM"  # placeholder, the building model reads the price of "EOM"

# The aggregator bids at the price cap of each market to secure acceptance, following
# the price-taker logic of Haghifam et al. (2022)
AGGREGATOR_BUY_PRICE_EUR_MWH = 3000.0

# ---------------------------------------------------------------------------
# Price corridor of the local energy market
#
# The LEM clearing price is bounded by the two tariffs a member faces outside the LEM:
# the feed-in tariff a seller would receive from the grid (lower bound) and the retail
# price a buyer would pay to its supplier (upper bound). Within this corridor local
# trade is beneficial for both sides. The bounds follow the tariff corridor used in the
# local market literature.
#
# TODO replace both values by referenced figures for Germany in 2016:
#   - feed-in tariff: Bundesnetzagentur, EEG payment rates for rooftop PV up to 10 kWp
#   - retail price: BDEW Strompreisanalyse or Destatis, household electricity price
# ---------------------------------------------------------------------------
FEED_IN_TARIFF_EUR_MWH = 123.1  # placeholder, lower bound of the corridor
RETAIL_PRICE_EUR_MWH = 288.0  # placeholder, upper bound of the corridor

# Demand bid price of the independent LEM participants. They are price takers and bid
# at the upper bound, so their bids are accepted whenever local supply is available.
INDEPENDENT_DEMAND_PRICE_EUR_MWH = RETAIL_PRICE_EUR_MWH

# ---------------------------------------------------------------------------
# Wholesale market
#
# The wholesale price is exogenous: the aggregator is a price taker and its bid volume
# is always accepted. This is modelled by a supply unit with a very large capacity whose
# marginal cost is the exogenous price. With an efficiency of one, the marginal cost of
# that unit equals the fuel price of the column WM_FUEL_TYPE in fuel_prices_df.csv, so a
# time series of day-ahead prices can be used directly.
#
# TODO replace the flat price by the day-ahead prices of the German bidding zone in
# 2016 (ENTSO-E transparency platform, Day-ahead Prices).
# ---------------------------------------------------------------------------
WM_PRICE_EUR_MWH = 29.0  # fallback only, used when WM_PRICE_FILE does not exist
# Hourly day-ahead prices 2016, exported from SMARD (Bundesnetzagentur):
# Markt > Grosshandelspreise > Marktgebiet DE/AT/LU (bis 30.09.2018) > Stunde.
# In 2016 Germany, Austria and Luxembourg formed one bidding zone.
WM_PRICE_FILE = SCENARIO_PATH / "Gro_handelspreise_201601010000_201701010000_Stunde.csv"
WM_PRICE_COLUMN = "DE/AT/LU [€/MWh] Berechnete Auflösungen"
WM_FUEL_TYPE = "wholesale"
WM_SUPPLY_CAPACITY_MW = 100.0  # large enough to accept every bid of the aggregator
# Bid price of the wholesale demand unit. It is set to the price cap of the wholesale
# market, so the demand unit always buys and the wholesale supply unit is always the
# marginal unit. The wholesale clearing price therefore equals the exogenous price in
# every hour, also when that price changes hour by hour.
WM_DEMAND_PRICE_EUR_MWH = 3000.0

# Price of the placeholder market that the building model reads
EOM_PRICE_EUR_MWH = RETAIL_PRICE_EUR_MWH


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
        # The time column is German local time (27.03 has no 02:00-02:45, 30.10 has
        # it twice), although Spalthoff et al. (2019, p. 6) write "Timestamps for data
        # are in CET (UTC+1)". The labels are converted to UTC like the market data,
        # so all series share one time axis.
        df["time"] = to_simulation_time(pd.DatetimeIndex(df["time"]))
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


def annual_consumption_mwh(node_load: pd.DataFrame) -> pd.Series:
    """
    Annual consumption of every node in MWh.

    Args:
        node_load (pd.DataFrame): Load per node in MW from :func:`build_node_load`,
            covering the whole year 2016.

    Returns:
        pd.Series: MWh per node (MW x length of one time step in hours).
    """
    step_h = (node_load.index[1] - node_load.index[0]) / pd.Timedelta(hours=1)
    return node_load.sum() * step_h


def commercial_nodes(load: pd.DataFrame) -> set[str]:
    """
    Nodes with at least one commercial or agricultural load element.

    Args:
        load (pd.DataFrame): SimBench ``Load`` table.

    Returns:
        set[str]: Node names whose load profile starts with one of
        ``COMMERCIAL_PROFILE_PREFIXES`` (own mapping of SimBench profile names).
    """
    is_commercial = load["profile"].str.startswith(COMMERCIAL_PROFILE_PREFIXES)
    return set(load.loc[is_commercial, "node"])


def split_nodes(
    node_load: pd.DataFrame, res: pd.DataFrame, load: pd.DataFrame
) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
    """
    Split the nodes into the aggregator portfolio, independent participants and
    community PV plants.

    Prosumers are the nodes with a rooftop PV system (rated power up to
    ``COMMUNITY_PV_THRESHOLD_MW``). A larger PV system is a community plant and acts as
    the supply side of the LEM. A pure demand node with a commercial or agricultural
    SimBench profile (``COMMERCIAL_PROFILE_PREFIXES``) is an independent participant;
    every other pure demand node is a household customer of the aggregator.

    Args:
        node_load (pd.DataFrame): Load per node from :func:`build_node_load` (MW, 2016).
        res (pd.DataFrame): SimBench ``RES`` table.
        load (pd.DataFrame): SimBench ``Load`` table.

    Returns:
        tuple[list[str], list[str], list[str], pd.DataFrame]: Prosumer nodes of the
        aggregator, pure demand nodes of the aggregator, independent demand nodes and
        the rows of ``res`` that are treated as community PV plants.

    Raises:
        ValueError: If a commercial node uses at most ``HOUSEHOLD_LIMIT_MWH_PER_YEAR``
            (it would be a household customer under Paragraph 3 Nr. 57 EnWG) or if a
            prosumer node is commercial.
    """
    pv = res[res["type"] == "PV"]
    community_pv = pv[pv.pRES > COMMUNITY_PV_THRESHOLD_MW]
    rooftop_nodes = set(pv[pv.pRES <= COMMUNITY_PV_THRESHOLD_MW]["node"])

    prosumer_nodes = sorted(n for n in node_load.columns if n in rooftop_nodes)
    demand_nodes = sorted(n for n in node_load.columns if n not in rooftop_nodes)

    commercial = commercial_nodes(load)
    energy = annual_consumption_mwh(node_load)
    small = [n for n in commercial if energy[n] <= HOUSEHOLD_LIMIT_MWH_PER_YEAR]
    if small:
        raise ValueError(f"commercial nodes at or below the household limit: {small}")
    commercial_prosumers = [n for n in prosumer_nodes if n in commercial]
    if commercial_prosumers:
        raise ValueError(f"prosumer nodes with a commercial profile: {commercial_prosumers}")

    aggregator_demand = [n for n in demand_nodes if n not in commercial]
    independent_demand = [n for n in demand_nodes if n in commercial]
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
        - ``powerplant_units.csv`` also holds the wholesale supply unit.
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
            "max_power": round(node_load[node].max(), 6),
            "min_power": 0,
            # Bid price of demand_energy_naive (demand.py: unit.price). Independent
            # participants bid the upper LEM limit, so the bid lies within the market
            # limits (rule of the thesis, supervisor 2026-10-30); before, the ASSUME
            # default 3000 was clipped to 288 by base_market.py, lines 374-383. The
            # demand units of the aggregator keep the ASSUME default; their own bids are
            # not used, because the aggregator bids for the whole portfolio.
            "price": INDEPENDENT_DEMAND_PRICE_EUR_MWH
            if operator == OPERATOR_INDEPENDENT
            else AGGREGATOR_BUY_PRICE_EUR_MWH,
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

    # 3. Community PV as the supply side of the LEM (128.8 kW > 100 kW, so no feed-in
    #    tariff, direct marketing under Paragraph 37 Abs. 2 EEG 2014). It bids at the
    #    lower bound of the LEM price corridor: the LEM cannot clear below this bound
    #    (market rule, corridor [Lezama2021] p. 4 Eq. (2c), [Zepter2019] p. 166), and
    #    every price at or above it exceeds every 2016 wholesale price the plant could
    #    get otherwise (2016 maximum 104.96 EUR/MWh, [SMARD]), so bidding at the bound
    #    is a price-taking bid [OWN reasoning].
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
    storage_file = SCENARIO_PATH / "storage_units.csv"
    if storage_rows:
        pd.DataFrame(storage_rows).to_csv(storage_file, index=False)
    else:
        storage_file.unlink(missing_ok=True)

    # 5. Wholesale supply: exogenous price, large enough to accept every bid
    plant_rows.append(
        {
            "name": "WholesaleSupply",
            "technology": "wholesale_supply",
            f"bidding_{LEM_MARKET_ID}": "",
            f"bidding_{WM_MARKET_ID}": "powerplant_energy_naive",
            f"bidding_{EOM_MARKET_ID}": "",
            "unit_operator": OPERATOR_GRID,
            "fuel_type": WM_FUEL_TYPE,
            "emission_factor": 0,
            "max_power": WM_SUPPLY_CAPACITY_MW,
            "min_power": 0,
            "efficiency": 1,
            "additional_cost": 0,
        }
    )
    pd.DataFrame(plant_rows).to_csv(SCENARIO_PATH / "powerplant_units.csv", index=False)

    # 5b. Wholesale demand: buys at the wholesale price, so surplus sold by the
    #     aggregator on the wholesale market is always accepted as well
    pd.DataFrame(
        [
            {
                "name": "WholesaleDemand",
                f"bidding_{WM_MARKET_ID}": "exchange_energy_naive",
                # import volume is zero, so the import price is never used
                "price_import": 0.0,
                "price_export": WM_DEMAND_PRICE_EUR_MWH,
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


def to_simulation_time(local_time: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Convert German local time with daylight saving time to UTC.

    Args:
        local_time (pd.DatetimeIndex): Time stamps in German local time, including the
            missing hour in March and the repeated hour in October.

    Returns:
        pd.DatetimeIndex: Time stamps in UTC without time zone information.

    Note:
        The simulation runs in UTC, the time base of ASSUME (see SIMULATION_START).
        The time column of the SimBench CSV files is written in German local time
        (27.03.2016 has no 02:00-02:45, 30.10.2016 has it twice), and so is the market
        data from SMARD and netztransparenz. All of them are converted with this
        function: local winter time minus one hour, local summer time minus two hours
        (e.g. 14:00 German summer time = 12:00 UTC).
    """
    return (
        local_time.tz_localize("Europe/Berlin", ambiguous="infer")
        .tz_convert("UTC")
        .tz_localize(None)
    )


def read_wm_price(index: pd.DatetimeIndex) -> pd.Series:
    """
    Read the exogenous wholesale price or fall back to a flat placeholder price.

    Args:
        index (pd.DatetimeIndex): Time steps of the simulation.

    Returns:
        pd.Series: Wholesale price in EUR/MWh for every time step.

    Raises:
        ValueError: If the price file does not cover all time steps of the simulation.

    Note:
        ``WM_PRICE_FILE`` is the SMARD export of hourly day-ahead prices (semicolon
        separated, decimal comma, German date format, local time). Without the file a
        flat price is used and a warning is logged, so a placeholder price is never
        mistaken for real data.
    """
    if not WM_PRICE_FILE.exists():
        logger.warning(
            f"{WM_PRICE_FILE.name} not found, using a flat wholesale price of "
            f"{WM_PRICE_EUR_MWH} EUR/MWh"
        )
        return pd.Series(WM_PRICE_EUR_MWH, index=index)

    raw = pd.read_csv(WM_PRICE_FILE, sep=";", decimal=",", na_values="-")
    local_time = pd.DatetimeIndex(
        pd.to_datetime(raw["Datum von"], format="%d.%m.%Y %H:%M")
    )
    prices = pd.Series(
        raw[WM_PRICE_COLUMN].to_numpy(), index=to_simulation_time(local_time)
    )
    # hourly 2016 prices -> simulation resolution: hold the hourly value for every step
    # of the hour (b22d8b5 used resample().mean(), which gives NaN when upsampling)
    prices = prices.sort_index().reindex(index, method="ffill")
    if prices.isna().any():
        raise ValueError(
            f"{WM_PRICE_FILE.name} does not cover all simulation time steps"
        )
    return prices


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
    # Only the wholesale price is given: it is exogenous by assumption. Every other
    # forecast, in particular the LEM price and all residual loads, is calculated
    # by ASSUME from the units of the scenario.
    forecasts[f"price_{WM_MARKET_ID}"] = read_wm_price(index)
    # The EOM is a placeholder market without participants: the building model of ASSUME
    # reads the price of a market named "EOM". Its price and residual load cannot be
    # calculated from a merit order, so both are given and do not affect any trade.
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

    # 4. Fuel price of the wholesale supply unit. With an efficiency of one its marginal
    #    cost equals this price, so the wholesale price is exogenous and time dependent.
    fuel_prices = pd.DataFrame({WM_FUEL_TYPE: read_wm_price(index), "co2": 0.0})
    fuel_prices.index.name = "datetime"
    fuel_prices.round(6).to_csv(SCENARIO_PATH / "fuel_prices_df.csv")

    # Volumes of the wholesale demand unit: it only buys (export), never sells
    exchanges = pd.DataFrame(
        {
            "WholesaleDemand_export": WM_SUPPLY_CAPACITY_MW / 2,
            "WholesaleDemand_import": 0.0,
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
        # one product per time step of the next day (96 quarter hours at 15 min)
        "products": [
            {
                "duration": RESOLUTION,
                "count": int(pd.Timedelta("24h") / pd.Timedelta(RESOLUTION)),
                "first_delivery": "24h",
            }
        ],
        "opening_frequency": "24h",
        "opening_duration": "15min",
        "volume_unit": "MW",  # ASSUME order volume = power (market_objects.py, MarketConfig.volume_unit)
        "maximum_bid_volume": 100000,
        "maximum_bid_price": 3000,
        "minimum_bid_price": -500,
        "price_unit": "EUR/MWh",
        "market_mechanism": "pay_as_clear",
    }
    lem_market = dict(market, operator="LEM_operator")
    # The LEM has no backstop supplier. Without a price cap the clearing price would run
    # into the technical maximum of the market whenever local supply is scarce. The cap
    # is the retail price: nobody buys locally above what its supplier charges.
    lem_market["maximum_bid_price"] = RETAIL_PRICE_EUR_MWH
    # Lower bound of the LEM price corridor: no seller sells locally below the feed-in
    # tariff ([Lezama2021] p. 4 Eq. (2c); [Zepter2019] p. 166). b22d8b5 kept -500.
    lem_market["minimum_bid_price"] = FEED_IN_TARIFF_EUR_MWH
    # Equal prices are common in the LEM (aggregator and community PV both sell at the
    # lower bound). ASSUME's pay_as_clear breaks such ties at random, so results are not
    # reproducible. pay_as_clear_prorata shares a rationed price level proportionally,
    # by analogy to the SDAC curtailment rule: SDAC (2023), p. 3 [OK]; applying it to
    # every rationed price level is a modelling choice [OWN].
    lem_market["market_mechanism"] = "pay_as_clear_prorata"

    config = {
        STUDY_CASE: {
            "start_date": SIMULATION_START,
            "end_date": SIMULATION_END,
            "time_step": RESOLUTION,
            "save_frequency_hours": 48,
            "markets_config": {
                LEM_MARKET_ID: lem_market,
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
        node_load, tables["RES"], tables["Load"]
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
        tables["Storage"].iloc[0:0],
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
