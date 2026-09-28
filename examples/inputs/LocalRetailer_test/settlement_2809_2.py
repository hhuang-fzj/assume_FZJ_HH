# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Settlement of the local retailer after a simulation run.

The script is a post-processing step. It reads the market results of one simulation and
the realised values of the portfolio, and derives the quantities of research question 3:
how much of the LEM bid was not executed, which imbalance follows from it and what it
costs.

Pipeline:
    1. **Read** the market results, the realised values and the portfolio definition.
    2. **Collect** the portfolio orders of both markets per delivery period.
    3. **Derive** the uncovered LEM volume and the realised residual load.
    4. **Settle** the imbalance at the reBAP price and sum up the procurement costs.

Note:
    Forecast and realised values are identical in this version, so the only source of
    imbalance is the partial execution of the LEM bids. The check in
    :func:`log_settlement_summary` uses this property.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIO_PATH = Path(__file__).parent
OUTPUT_PATH = SCENARIO_PATH / "../../outputs/LocalRetailer_test_local_retailer_demo"

PORTFOLIO_UNIT_ID = "__portfolio__"
LEM_MARKET_ID = "LEM_DA"
WM_MARKET_ID = "WM_DA"
AGGREGATOR_OPERATOR = "local_retailer"

# Imbalance price in EUR/MWh (placeholder until reBAP time series is available)
# Imbalance price. The reBAP time series is read from the files matching REBAP_FILES
# in the scenario folder (monthly downloads, 15-minute values). Without such files the
# flat placeholder REBAP_EUR_MWH is used and a warning is logged.
REBAP_FILES = "rebap_*.xls"
REBAP_EUR_MWH = 120.0  # placeholder, only used when no reBAP file is found

# Resolution of the simulation. Must match RESOLUTION in simbench_to_assume.py.
RESOLUTION = "1h"

SETTLEMENT_FILE = SCENARIO_PATH / "settlement.csv"


def read_portfolio_orders(output_path: Path) -> pd.DataFrame:
    """
    Read the portfolio orders of both markets from the simulation output.

    Args:
        output_path (Path): Folder with the CSV output of the simulation.

    Returns:
        pd.DataFrame: One row per delivery period with the columns ``q_lem_bid``,
        ``q_lem_accepted``, ``q_wm_bid``, ``q_wm_accepted``, ``price_lem`` and
        ``price_wm``. All volumes are positive procurement volumes in MWh. Time steps
        without orders on a market are filled with zero.

    Raises:
        FileNotFoundError: If ``market_orders.csv`` does not exist.
    """
    orders = pd.read_csv(output_path / "market_orders.csv", parse_dates=["start_time"])
    portfolio = orders[orders.unit_id == PORTFOLIO_UNIT_ID]

    result = {}
    for market_id, suffix in ((LEM_MARKET_ID, "lem"), (WM_MARKET_ID, "wm")):
        market = portfolio[portfolio.market_id == market_id].set_index("start_time")
        # The aggregator buys, so volumes are negative in the ASSUME convention
        result[f"q_{suffix}_bid"] = -market["volume"]
        result[f"q_{suffix}_accepted"] = -market["accepted_volume"]
        result[f"price_{suffix}"] = market["accepted_price"]

    # A market can receive no orders at all (alpha = 0 or alpha = 1). Its columns are
    # then missing for those time steps, so they are filled with zero instead of NaN.
    return pd.DataFrame(result).fillna(0.0).sort_index()


def read_portfolio_units(scenario_path: Path) -> tuple[list[str], list[str]]:
    """
    Read which units belong to the aggregator portfolio.

    Args:
        scenario_path (Path): Folder with the scenario input files.

    Returns:
        tuple[list[str], list[str]]: Ids of the pure demand units and of the building
        units of the aggregator.
    """
    demand = pd.read_csv(scenario_path / "demand_units.csv")
    demand_units = demand[demand.unit_operator == AGGREGATOR_OPERATOR]["name"].tolist()

    buildings = pd.read_csv(scenario_path / "residential_dsm_units.csv")
    building_units = buildings[buildings.unit_operator == AGGREGATOR_OPERATOR][
        "name"
    ].tolist()
    return demand_units, building_units


def build_realised_residual_load(
    scenario_path: Path, demand_units: list[str], building_units: list[str]
) -> pd.Series:
    """
    Build the realised residual load of the portfolio from the realised values.

    Args:
        scenario_path (Path): Folder with the scenario input files.
        demand_units (list[str]): Ids of the pure demand units of the aggregator.
        building_units (list[str]): Ids of the building units of the aggregator.

    Returns:
        pd.Series: Realised residual load in MW per time step. A positive value is a
        procurement need, a negative value is a surplus.
    """
    actuals = pd.read_csv(
        scenario_path / "actuals_df.csv", parse_dates=["datetime"], index_col="datetime"
    )

    residual_load = pd.Series(0.0, index=actuals.index)
    for unit in demand_units:
        residual_load += actuals[f"demand_actual_{unit}"]
    for unit in building_units:
        residual_load += actuals[f"load_actual_{unit}"] - actuals[f"pv_actual_{unit}"]
    return residual_load


def read_rebap_file(path: Path) -> pd.Series:
    """
    Read one monthly reBAP file as published by the German transmission system operators.

    The file is an Excel 2003 XML spreadsheet (SpreadsheetML) with the columns
    ``Datum Uhrzeit von``, ``Datum Uhrzeit bis``, ``NRV-Saldo (MW)`` and
    ``reBAP (EUR/MWh)``, one row per quarter hour.

    Args:
        path (Path): Path to the file.

    Returns:
        pd.Series: reBAP in EUR/MWh, indexed by the start of each quarter hour.
    """
    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    rows = []
    for row in ET.parse(path).getroot().findall(".//ss:Row", ns):
        cells = [cell.find("ss:Data", ns) for cell in row.findall("ss:Cell", ns)]
        rows.append([cell.text if cell is not None else None for cell in cells])

    # Data rows start with a date such as " 01.06.2016  00:00  "
    data = [row for row in rows if row and row[0] and row[0].strip()[:2].isdigit()]
    start = pd.to_datetime([row[0].strip() for row in data], format="%d.%m.%Y  %H:%M")
    return pd.Series([float(row[3]) for row in data], index=start, name="rebap")


def read_rebap(index: pd.DatetimeIndex) -> pd.Series:
    """
    Read the reBAP for the settlement periods, or fall back to a flat placeholder.

    Args:
        index (pd.DatetimeIndex): Settlement periods.

    Returns:
        pd.Series: reBAP in EUR/MWh per settlement period. Quarter-hourly values are
        averaged to the simulation resolution.

    Raises:
        ValueError: If reBAP files exist but do not cover all settlement periods.

    Note:
        Averaging to hourly values is a simplification: imbalances are settled per
        quarter hour in Germany. It is exact only if the imbalance is constant within
        the hour.
    """
    files = sorted(SCENARIO_PATH.glob(REBAP_FILES))
    if not files:
        logger.warning(
            f"no file matching {REBAP_FILES} found, using a flat reBAP of "
            f"{REBAP_EUR_MWH} EUR/MWh"
        )
        return pd.Series(REBAP_EUR_MWH, index=index)

    quarter_hourly = pd.concat(read_rebap_file(f) for f in files).sort_index()
    rebap = quarter_hourly.resample(RESOLUTION).mean().reindex(index)
    if rebap.isna().any():
        raise ValueError(
            f"reBAP files {[f.name for f in files]} do not cover all settlement periods"
        )
    logger.info(f"reBAP read from {[f.name for f in files]}")
    return rebap


def settle(
    orders: pd.DataFrame, residual_load: pd.Series, rebap: pd.Series | None = None
) -> pd.DataFrame:
    """
    Derive uncovered volume, imbalance and costs per delivery period.

    Args:
        orders (pd.DataFrame): Portfolio orders from :func:`read_portfolio_orders`.
        residual_load (pd.Series): Realised residual load from
            :func:`build_realised_residual_load`.
        rebap (pd.Series | None, optional): Imbalance price in EUR/MWh per period.
            Read with :func:`read_rebap` if not given.

    Returns:
        pd.DataFrame: ``orders`` extended by ``residual_load``, ``uncovered``,
        ``imbalance``, ``cost_lem``, ``cost_wm``, ``cost_imbalance`` and ``cost_total``.
    """
    settlement = orders.copy()
    settlement["residual_load"] = residual_load.reindex(settlement.index)

    # 1. Part of the LEM bid that was not executed
    settlement["uncovered"] = settlement.q_lem_bid - settlement.q_lem_accepted

    # 2. What the portfolio needs minus what was procured on both markets
    settlement["imbalance"] = settlement.residual_load - (
        settlement.q_lem_accepted + settlement.q_wm_accepted
    )

    # 3. Costs; a positive imbalance is a shortage and has to be bought at the
    #    imbalance price
    settlement["cost_lem"] = settlement.q_lem_accepted * settlement.price_lem
    settlement["cost_wm"] = settlement.q_wm_accepted * settlement.price_wm
    if rebap is None:
        rebap = read_rebap(settlement.index)
    settlement["rebap"] = rebap.reindex(settlement.index)
    settlement["cost_imbalance"] = settlement.imbalance * settlement.rebap
    settlement["cost_total"] = (
        settlement.cost_lem + settlement.cost_wm + settlement.cost_imbalance
    )
    return settlement


def log_settlement_summary(settlement: pd.DataFrame) -> None:
    """
    Log the totals of one settlement and check the internal consistency.

    Args:
        settlement (pd.DataFrame): Output of :func:`settle`.

    Note:
        With perfect foresight the imbalance must equal the uncovered LEM volume.
        A deviation indicates an error in the settlement or in the realised values.
    """
    totals = settlement[
        ["q_lem_accepted", "q_wm_accepted", "uncovered", "imbalance"]
    ].sum()
    costs = settlement[["cost_lem", "cost_wm", "cost_imbalance", "cost_total"]].sum()

    logger.info(
        f"LEM procured {totals.q_lem_accepted:.4f} MWh, cost {costs.cost_lem:.2f} EUR"
    )
    logger.info(
        f"WM procured {totals.q_wm_accepted:.4f} MWh, cost {costs.cost_wm:.2f} EUR"
    )
    logger.info(
        f"uncovered LEM volume {totals.uncovered:.4f} MWh, "
        f"imbalance {totals.imbalance:.4f} MWh, cost {costs.cost_imbalance:.2f} EUR"
    )
    logger.info(f"total cost {costs.cost_total:.2f} EUR")

    deviation = (settlement.imbalance - settlement.uncovered).abs().max()
    if deviation > 1e-6:
        logger.warning(
            f"imbalance and uncovered volume differ by up to {deviation:.6f} MWh, "
            "which should not happen with perfect foresight"
        )
    else:
        logger.info("check passed: imbalance equals the uncovered LEM volume")


def main() -> None:
    """Settle one simulation run and write the settlement table."""
    # 1. Read market results and portfolio definition
    orders = read_portfolio_orders(OUTPUT_PATH)
    demand_units, building_units = read_portfolio_units(SCENARIO_PATH)

    # 2. Build the realised residual load of the portfolio
    residual_load = build_realised_residual_load(
        SCENARIO_PATH, demand_units, building_units
    )

    # 3. Settle and report
    settlement = settle(orders, residual_load)
    log_settlement_summary(settlement)

    settlement.round(6).to_csv(SETTLEMENT_FILE)
    logger.info(f"Settlement written to {SETTLEMENT_FILE.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
