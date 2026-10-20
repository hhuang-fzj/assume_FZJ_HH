"""
WP5: build the 15-minute input table of one week for solve_alpha().

Reads only files that are already in LocalRetailer_test (b22d8b5):
    simbench_raw/            SimBench grid, 15-minute profiles of 2016
    Gro_handelspreise_...csv SMARD day-ahead prices 2016, hourly, German local time
    rebap_2016_*.xls         reBAP 2016, 15-minute, German local time

Writes (same folder as this script):
    wp5_inputs_<start>.csv   one row per quarter hour

Columns of the output table
    decision inputs (known when the aggregator decides, used by solve_alpha):
        L_hat   forecast net residual load of the portfolio   MWh   + buy / - sell
        p_lem   forecast LEM price                            EUR/MWh
        p_wm    forecast WM price                             EUR/MWh
        r_hat   expected imbalance price                      EUR/MWh
        S_hat   expected LEM volume the LEM can take           MWh   >= 0
    ex-post values (NOT known when deciding, only for the settlement in WP7):
        L_actual       actual net residual load           MWh
        p_wm_actual    actual WM price (SMARD)            EUR/MWh
        rebap_actual   actual reBAP                       EUR/MWh

Run from the repository root:
    python examples/inputs/LocalRetailer_test/course/wp5_inputs.py
"""

import sys
from pathlib import Path

import pandas as pd

# --- make the scenario folder importable, so the functions of b22d8b5 are reused ---
SCENARIO_PATH = Path(__file__).resolve().parent.parent  # .../LocalRetailer_test
sys.path.insert(0, str(SCENARIO_PATH))

from settlement import read_rebap_file  # noqa: E402
from simbench_to_assume import (  # noqa: E402
    FEED_IN_TARIFF_EUR_MWH,
    RETAIL_PRICE_EUR_MWH,
    SIMBENCH_PATH,
    WM_PRICE_COLUMN,
    WM_PRICE_FILE,
    build_node_load,
    build_pv_power,
    load_simbench_tables,
    split_nodes,
    to_simulation_time,
)

# ---------------------------------------------------------------------------
# Settings of this experiment (every assumption is here)
# ---------------------------------------------------------------------------
WEEK_START = "2016-06-20 00:00"  # Monday; June because a reBAP file exists for June
WEEK_END = "2016-06-27 00:00"  # exclusive
RESOLUTION = "15min"
DT_H = 0.25  # length of one time step in hours, E = P * DT_H

# Expected imbalance price: the aggregator does not speculate on the imbalance price.
# Unsold volume is valued slightly worse than the WM price (see WP1 task 4).
R_HAT_PENALTY_EUR_MWH = 1.0  # [my choice, no literature] -> to be confirmed (WP6a)

# Expected LEM volume S_hat before any LEM feedback exists:
#   "full"  -> S_hat = |L_hat| (the aggregator assumes the LEM takes everything)
S_HAT_RULE = "full"  # [my choice, no literature] -> replaced by feedback in WP6b

OUTPUT_FILE = Path(__file__).resolve().parent / f"wp5_inputs_{WEEK_START[:10]}.csv"


def build_time_index() -> pd.DatetimeIndex:   #生成这周所有 15 min 时间戳。所有其他数据都对齐到这个索引
    """Quarter hours of the week in simulation time (standard time UTC+1, no DST)."""
    return pd.date_range(WEEK_START, WEEK_END, freq=RESOLUTION, inclusive="left")


def build_portfolio_residual_load(index: pd.DatetimeIndex) -> pd.Series:
    """
    Net residual load of the aggregator portfolio in MWh per quarter hour.

    Portfolio = prosumers + aggregator demand nodes, as split by split_nodes() of
    b22d8b5 (old rule AGGREGATOR_DEMAND_NODES = 15, see open items).
    """
    tables = load_simbench_tables(SIMBENCH_PATH)
    node_load = build_node_load(tables["Load"], tables["LoadProfile"], RESOLUTION)  # MW
    pv_power = build_pv_power(tables["RES"], tables["RESProfile"], RESOLUTION)  # MW
    prosumers, aggregator_demand, _, _ = split_nodes(node_load, tables["RES"])

    load_mw = node_load[prosumers + aggregator_demand].sum(axis=1)
    pv_mw = pv_power[prosumers].sum(axis=1)
    residual_mw = (load_mw - pv_mw).reindex(index)  # + buy / - sell
    if residual_mw.isna().any():
        raise ValueError("SimBench profiles do not cover the whole week")
    print(f"portfolio: {len(prosumers)} prosumers, {len(aggregator_demand)} demand nodes")
    return residual_mw * DT_H  # MW -> MWh per quarter hour


def read_wm_price_15min(index: pd.DatetimeIndex) -> pd.Series:
    """SMARD hourly price, constant within the hour, on the 15-minute index."""
    raw = pd.read_csv(WM_PRICE_FILE, sep=";", decimal=",", na_values="-")
    local_time = pd.DatetimeIndex(pd.to_datetime(raw["Datum von"], format="%d.%m.%Y %H:%M"))
    hourly = pd.Series(raw[WM_PRICE_COLUMN].to_numpy(), index=to_simulation_time(local_time))
    price = hourly.sort_index().reindex(index, method="ffill")  # hold the hourly value
    if price.isna().any():
        raise ValueError("SMARD file does not cover the whole week")
    return price


def read_rebap_15min(index: pd.DatetimeIndex) -> pd.Series:
    """reBAP of all rebap_2016_*.xls files on the 15-minute index (no averaging)."""
    files = sorted(SCENARIO_PATH.glob("rebap_2016_*.xls"))
    rebap = pd.concat(read_rebap_file(f) for f in files).sort_index().reindex(index)
    if rebap.isna().any():
        raise ValueError(f"reBAP files {[f.name for f in files]} do not cover the week")
    return rebap


def build_decision_inputs(L_hat: pd.Series, p_wm: pd.Series) -> pd.DataFrame:
    """Forecasts the aggregator uses when it decides (rules of this experiment)."""
    sell = L_hat < 0
    inputs = pd.DataFrame(index=L_hat.index)
    inputs["L_hat"] = L_hat
    # LEM price forecast: selling -> lower bound of the corridor, buying -> upper bound
    inputs["p_lem"] = sell.map({True: FEED_IN_TARIFF_EUR_MWH, False: RETAIL_PRICE_EUR_MWH})
    inputs["p_wm"] = p_wm
    # unsold volume is valued slightly worse than WM in both directions
    inputs["r_hat"] = p_wm - R_HAT_PENALTY_EUR_MWH * sell.map({True: 1.0, False: -1.0})
    if S_HAT_RULE == "full":
        inputs["S_hat"] = L_hat.abs()
    else:
        raise ValueError(f"unknown S_HAT_RULE {S_HAT_RULE}")
    return inputs


def check(table: pd.DataFrame, residual_mwh: pd.Series) -> None:
    """Five checks of WP5; every line must say OK."""
    n_days = (pd.Timestamp(WEEK_END) - pd.Timestamp(WEEK_START)).days
    print(f"1 rows          : {len(table)} (expected {n_days * 96})",
          "OK" if len(table) == n_days * 96 else "FAIL")
    print("2 missing values:", int(table.isna().sum().sum()),
          "OK" if not table.isna().any().any() else "FAIL")
    # independent check: the same week from the hourly scenario files of b22d8b5 (MW, 1 h)
    actuals = pd.read_csv(SCENARIO_PATH / "actuals_df.csv", parse_dates=["datetime"],
                          index_col="datetime")
    demand = pd.read_csv(SCENARIO_PATH / "demand_units.csv")
    houses = pd.read_csv(SCENARIO_PATH / "residential_dsm_units.csv")
    hourly_mw = sum(actuals[f"demand_actual_{u}"]
                    for u in demand[demand.unit_operator == "local_retailer"].name)
    hourly_mw = hourly_mw + sum(actuals[f"load_actual_{u}"] - actuals[f"pv_actual_{u}"]
                                for u in houses[houses.unit_operator == "local_retailer"].name)
    hourly_mwh = hourly_mw.loc[WEEK_START:WEEK_END].iloc[:-1].sum() * 1.0  # MW x 1 h
    ok = abs(hourly_mwh - residual_mwh.sum()) < 1e-6
    print(f"3 energy        : 15 min sum = {residual_mwh.sum():.4f} MWh, "
          f"hourly b22d8b5 = {hourly_mwh:.4f} MWh", "OK" if ok else "FAIL")
    hourly_unique = table["p_wm"].groupby(table.index.floor("h")).nunique().max()
    print(f"4 price per hour: max {hourly_unique} different values in one hour",
          "OK" if hourly_unique == 1 else "FAIL")
    print(f"5 direction     : {int((table.L_hat < 0).sum())} quarter hours sell, "
          f"{int((table.L_hat > 0).sum())} buy, {int((table.L_hat == 0).sum())} zero")


def main() -> None:
    index = build_time_index()
    residual_mwh = build_portfolio_residual_load(index)
    p_wm = read_wm_price_15min(index)
    rebap = read_rebap_15min(index)

    # perfect forecast in this experiment: forecast = actual (labelled, not hidden)
    table = build_decision_inputs(L_hat=residual_mwh, p_wm=p_wm)
    table["L_actual"] = residual_mwh
    table["p_wm_actual"] = p_wm
    table["rebap_actual"] = rebap

    check(table, residual_mwh)
    table.to_csv(OUTPUT_FILE)
    print(f"written: {OUTPUT_FILE.name}")
    print(table.describe().round(4).T[["min", "mean", "max"]])


if __name__ == "__main__":
    main()
