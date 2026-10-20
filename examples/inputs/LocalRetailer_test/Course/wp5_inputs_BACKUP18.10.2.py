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
# Settings of this experiment (every assumption is here, each with its source)
#
# Status of a source:  [OK] read and page checked | [PART] partly checked
#                      [TODO] not yet checked      | [OWN] own choice, no literature
#
# References (full citations)
#   [Ottesen2016] Ottesen, S. O., Tomasgard, A., Fleten, S.-E. (2016). Prosumer bidding
#       and scheduling in electricity markets. Energy 94, 828-843.
#       doi:10.1016/j.energy.2015.11.047
#   [Iria2019]    Iria, J., Soares, F., Matos, M. (2019). Optimal bidding strategy for an
#       aggregator of prosumers in energy and secondary reserve markets.
#       Applied Energy 238, 1361-1372. doi:10.1016/j.apenergy.2019.01.191
#   [Lezama2021]  Lezama, F., Soares, J., Faia, R., Vale, Z., Kilkki, O., Repo, S.,
#       Segerstam, J. (2021). Bidding in local electricity markets with cascading
#       wholesale market integration. International Journal of Electrical Power &
#       Energy Systems 131, 107045. doi:10.1016/j.ijepes.2021.107045
#   [Zepter2019]  Zepter, J. M., Lueth, A., Crespo del Granado, P., Egging, R. (2019).
#       Prosumer integration in wholesale electricity markets: Synergies of peer-to-peer
#       trade and residential storage. Energy and Buildings 184, 163-176.
#       doi:10.1016/j.enbuild.2018.12.003
#   [Koch2021]    Koch, C. (2021). Intraday imbalance optimization: incentives and impact
#       of strategic intraday bidding behavior. Energy Systems 13, 409-435.
#       doi:10.1007/s12667-021-00445-9
#   [EBGL]        Commission Regulation (EU) 2017/2195 establishing a guideline on
#       electricity balancing, Art. 17(1).
#   [BK6-13-104]  Bundesnetzagentur, Beschlusskammer 6 (2013). Positionspapier zur
#       Bilanzkreisbewirtschaftung, BK6-13-104, 16.09.2013 (cites Paragraph 4 Abs. 2
#       StromNZV).
#   [BDEW2020]    BDEW (2020). BDEW-Strompreisanalyse Juli 2020 - Haushalte und
#       Industrie. Berlin, 30.07.2020, S. 7-8.
#   [SimBench]    Meinecke, S., Sarajlic, D., Drauz, S. R., Klettke, A., Lauven, L.-P.,
#       Rehtanz, C., Moser, A., Braun, M. (2020). SimBench - A Benchmark Dataset of
#       Electric Power Systems to Compare Innovative Solutions Based on Power Flow
#       Analysis. Energies 13(12), 3290. doi:10.3390/en13123290
#   [SimBenchFmt] SimBench consortium (e2n Univ. Kassel, IAEW RWTH Aachen, ie3 TU
#       Dortmund, Fraunhofer IEE) (2019). Elektrische Benchmarknetzmodelle
#       (data format slides), 19.06.2019, slides 13, 23-26.
#   [SMARD]       Bundesnetzagentur | SMARD.de, Grosshandelspreise, Marktgebiet
#       DE/AT/LU, Aufloesung Stunde, 01.01.2016-01.01.2017. Download date: TODO.
#   [reBAP]       Uebertragungsnetzbetreiber via netztransparenz.de, reBAP 2016,
#       monthly files rebap_2016_01/04/06.xls, 15 min. Download date: TODO.
#   [Kuehnbach2022] Kuehnbach, M., Bekk, A., Weidlich, A. (2022). Towards improved
#       prosumer participation: Electricity trading in local markets.
#       Energy 239, 122445. doi:10.1016/j.energy.2021.122445
# ---------------------------------------------------------------------------

# Week of the experiment. June because reBAP files exist only for 01, 04, 06 / 2016.
#   [OWN] choice of the week; data availability: [reBAP]
WEEK_START = "2016-06-20 00:00"  # Monday
WEEK_END = "2016-06-27 00:00"  # exclusive

# 15-minute resolution, E = P * DT_H.
#   [OWN] decision of the thesis (15 min everywhere). SimBench profiles are 15 min:
#   [SimBenchFmt] slide 3 [OK]. reBAP is settled per quarter hour: [reBAP].
RESOLUTION = "15min"
DT_H = 0.25  # length of one time step in hours

# Expected imbalance price used by the optimiser: r_hat = p_wm -/+ penalty
# (selling: p_wm - penalty, buying: p_wm + penalty).
#   Form: imbalance modelled as a penalty relative to the spot price, so that the
#   model plans into balance and does not arbitrage between markets:
#   [Ottesen2016] p. 830, p. 839 [OK]. Expected imbalance cost as a term of the
#   day-ahead bidding objective: [Iria2019] Sect. 3.1, Eq. (1) [PART, page TODO].
#   Why no speculation on the imbalance price: BRPs must strive to be balanced,
#   [EBGL] Art. 17(1) [OK]; [BK6-13-104] [OK]; intentional imbalance positions are
#   not allowed in Germany, [Koch2021] [PART, page TODO].
#   Value 1 EUR/MWh: [OWN]. Ottesen2016 uses a percentage of the spot price (base 400 %);
#   a percentage flips sign at negative prices, so a fixed EUR value is used here.
#   TODO: sensitivity analysis (1 / 10 / 100 EUR/MWh) must show that alpha does not
#   change for any penalty > 0 before this claim is written in the thesis.
R_HAT_PENALTY_EUR_MWH = 1.0

# Expected LEM volume S_hat on the first day, before any LEM feedback exists:
#   "full" -> S_hat = |L_hat|, i.e. the aggregator assumes that the LEM accepts
#   everything it offers.
#   Basis: price-taker assumption of full acceptance, [Iria2019] p. 1363 [OK].
#   From day 2 S_hat is updated from the aggregator's own clearing results (WP8);
#   participants only learn whether their own bids were accepted and adapt from
#   their own past results: [Kuehnbach2022] p. 7 [OK]. The concrete update rule is
#   [OWN] and is documented in the WP8 strategy.
S_HAT_RULE = "full"

OUTPUT_FILE = Path(__file__).resolve().parent / f"wp5_inputs_{WEEK_START[:10]}.csv"


def build_time_index() -> pd.DatetimeIndex: #生成这周所有 15 min 时间戳。所有其他数据都对齐到这个索引
    """Quarter hours of the week in simulation time (standard time UTC+1, no DST)."""
    return pd.date_range(WEEK_START, WEEK_END, freq=RESOLUTION, inclusive="left")


def build_portfolio_residual_load(index: pd.DatetimeIndex) -> pd.Series: #算自己的 Aggregator 组合在每个 15 min 的净剩余负荷 L
    """
    Net residual load of the aggregator portfolio in MWh per quarter hour.

    Portfolio = prosumers + aggregator demand nodes, as split by split_nodes() of
    b22d8b5 (old rule AGGREGATOR_DEMAND_NODES = 15, see open items).

    Sources:
      Data: [SimBench] grid LV4.101, files simbench_raw/ (Load, LoadProfile, RES,
        RESProfile). Column meaning and units (pLoad, pRES in MW): [SimBenchFmt]
        slides 24-25 [OK]. Profiles are relative scaling factors, absolute power =
        profile x rated power: [SimBenchFmt] slide 13 [OK].
      Rooftop PV <= 0.05 MW = prosumer, larger PV = community plant: b22d8b5 [OWN].
      First 15 pure demand nodes belong to the aggregator: b22d8b5 [OWN] (old rule;
        the EnWG household rule of the thesis is not implemented yet).
    """
    tables = load_simbench_tables(SIMBENCH_PATH) #读表，profile 表按 time 列做索引（35 136 行 = 366 天 × 96）
    node_load = build_node_load(tables["Load"], tables["LoadProfile"], RESOLUTION)  # MW 每个负荷元件 = 相对曲线 <profile>_pload × 额定功率 pLoad（MW）= 绝对功率 MW；同一节点的元件相加。形状：35 136 行 × 节点数列
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
    """
    SMARD hourly price, constant within the hour, on the 15-minute index.

    Sources:
      Data: [SMARD], column "DE/AT/LU" (one bidding zone until 30.09.2018).
      2016 day-ahead products were hourly; holding the hourly value for the four
        quarter hours is [OWN] (decision of 15.10.2026: WM also with 15-min products).
      Time conversion German local time -> UTC+1 without DST: to_simulation_time()
        of b22d8b5. Assumption that SimBench uses UTC+1: [OWN, TODO] - the SimBench
        format slides do not state the time zone ([SimBenchFmt] slide 23).
    """
    raw = pd.read_csv(WM_PRICE_FILE, sep=";", decimal=",", na_values="-")
    local_time = pd.DatetimeIndex(pd.to_datetime(raw["Datum von"], format="%d.%m.%Y %H:%M"))
    hourly = pd.Series(raw[WM_PRICE_COLUMN].to_numpy(), index=to_simulation_time(local_time))
    price = hourly.sort_index().reindex(index, method="ffill")  # hold the hourly value
    if price.isna().any():
        raise ValueError("SMARD file does not cover the whole week")
    return price


def read_rebap_15min(index: pd.DatetimeIndex) -> pd.Series:
    """
    reBAP of all rebap_2016_*.xls files on the 15-minute index (no averaging).

    Sources:
      Data: [reBAP]. One symmetric price per quarter hour (same price for shortage and
        surplus): netztransparenz.de, Erlaeuterungen zur reBAP-Berechnung
        (10.07.2023), Sect. 2.1, S. 8 [OK; note: 2023 document, data are 2016].
      Reader read_rebap_file(): b22d8b5 settlement.py.
    """
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
    # LEM price forecast: selling -> lower bound of the corridor, buying -> upper bound.
    #   Corridor (feed-in tariff ... retail price): [Lezama2021] p. 3, p. 4 Eq. (1c)(2c)
    #   [OK]; [Zepter2019] p. 166 [OK]. Upper bound 288 EUR/MWh: [BDEW2020] S. 7-8 [OK].
    #   Lower bound 123.1 EUR/MWh (EEG 2016, rooftop PV <= 10 kWp): [TODO] check in the
    #   Bundesnetzagentur table "Degressions- und Verguetungssaetze" 2016.
    #   Taking the bound that is worst for the aggregator in each direction: [OWN].
    inputs["p_lem"] = sell.map({True: FEED_IN_TARIFF_EUR_MWH, False: RETAIL_PRICE_EUR_MWH})
    # WM price forecast = actual SMARD price (perfect price forecast): [OWN] simplification
    inputs["p_wm"] = p_wm
    # unsold volume is valued slightly worse than WM in both directions
    #   (form: [Ottesen2016]; value: [OWN]; see R_HAT_PENALTY_EUR_MWH)
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

    # perfect load forecast in this experiment: forecast = actual (labelled, not hidden).
    #   Same assumption as b22d8b5 (settlement.py header). [OWN] simplification.
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
