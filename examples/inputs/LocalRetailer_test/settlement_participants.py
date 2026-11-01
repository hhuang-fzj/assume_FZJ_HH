# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Round 2: ex-post settlement of all LEM participants (rule R3), not only the aggregator.

settlement.py (round 1) settles the aggregator (unit_id = __portfolio__). This module
settles every other participant with the same formula (Course/wp11_participant_cost.py):

    D = L - A - W
    C = p_LEM * A + p_WM * W + reBAP * D

    L = realised net volume, A = accepted in the LEM, W = accepted in the WM, all in MWh
    per 15 min; sign buy +, sell -; cost + = pays, cost - = receives.

Realised volume L per participant group:

    group                  units                             L
    ---------------------  --------------------------------  ---------------------------------
    independent demand     demand_units.csv,                 + demand_df.csv x 0.25 h
                           unit_operator = independent_op.   (perfect foresight, as the
                                                             aggregator in round 1)
    community PV           powerplant_units.csv,             - availability_df.csv x
                           unit_operator = LEC_operator      max_power x 0.25 h
    battery                storage_units.csv                 - dispatched power x 0.25 h
                                                             (battery_dispatch(): what ASSUME
                                                             runs, see below)

Battery: ASSUME runs a battery with its accepted volume, but cuts it where the state of
charge does not allow it (storage.py execute_current_dispatch, lines 277-340). This
happens when only part of a bid was accepted (e.g. the LEM part of a charge) and a later
bid of the same day relied on the full volume. Then L differs from A + W and D = L - A - W is the
volume sold or bought but not delivered, settled at the reBAP like every other
deviation. Without cuts D = 0 (rule R5).

The dispatch is rebuilt with ASSUME's own Storage.execute_current_dispatch() from the
accepted volumes of both markets, starting at initial_soc at the simulation start. It is
not read from unit_dispatch.csv, because ASSUME writes that file only every
save_frequency_hours (48 h, config.yaml) and the last two days of the week are missing
there (checked 2026-10-26: the file ends at 2016-06-25 00:15). dispatch_check() compares
both where unit_dispatch.csv has values.

The wholesale units (Grid_operator) are not participants: they represent the rest of
the WM and only make the WM price exogenous (model document, section 4).

Data path:
    market_orders.csv (ASSUME output, outputs.py store_dfs 451-516) -> read_unit_orders()
    unit tables + demand_df.csv + availability_df.csv (scenario),
    storage_units.csv + accepted volumes -> battery_dispatch() (ASSUME Storage class)
    -> realised_volumes()
    rebap_2016_*.xls -> settlement.read_rebap() (lines 179-213)
    -> settle_participants() -> participants.csv (written by Course/wp15_run_round2.py)

Check: with one uniform LEM price, the LEM payments of all LEM participants (including
the aggregator) add up to zero in every period: the buyers pay what the sellers
receive (check_lem_balance()).

Sources:
    Formula: settlement.py settle() (round 1); Course/wp11_participant_cost.py.
    reBAP for buying and selling: netztransparenz.de (2023). Erlaeuterungen zur
        reBAP-Berechnung, Stand 10.07.2023, Sect. 2.1, S. 8.
    Battery runs with its accepted volume: ASSUME assume/common/base.py,
        set_dispatch_plan() (from line 121), within its SOC limits: storage.py,
        execute_current_dispatch() (lines 277-340).
    Same settlement for all participants: research design of the thesis (rule R3).
    Intraday auction before the reBAP (R3): not included, data missing (section 7.4).
"""

from pathlib import Path

import pandas as pd
import yaml

from assume.common.forecaster import UnitForecaster
from assume.units.storage import Storage
from settlement import (
    LEM_MARKET_ID,
    STEP_HOURS,
    WM_MARKET_ID,
    read_rebap,
)

INDEPENDENT_OPERATOR = "independent_operator"
COMMUNITY_OPERATOR = "LEC_operator"
GRID_OPERATOR = "Grid_operator"
GROUP_INDEPENDENT = "independent demand"
GROUP_COMMUNITY_PV = "community PV"
GROUP_BATTERY = "battery"


def participant_groups(scenario_path: Path) -> dict[str, str]:
    """
    Unit id -> participant group, from the unit tables of the scenario.

    Returns:
        dict[str, str]: Only units that are settled here (not the aggregator's units,
        not the wholesale units).
    """
    groups = {}
    demand = pd.read_csv(scenario_path / "demand_units.csv")
    for name in demand[demand.unit_operator == INDEPENDENT_OPERATOR]["name"]:
        groups[name] = GROUP_INDEPENDENT
    plants = pd.read_csv(scenario_path / "powerplant_units.csv")
    for name in plants[plants.unit_operator == COMMUNITY_OPERATOR]["name"]:
        groups[name] = GROUP_COMMUNITY_PV
    storage_file = scenario_path / "storage_units.csv"
    if storage_file.exists():
        storage = pd.read_csv(storage_file)
        for name in storage["name"]:
            groups[name] = GROUP_BATTERY
    return groups


def read_unit_orders(output_path: Path, units: list[str]) -> pd.DataFrame:
    """
    Accepted volumes and prices of the given units on both markets.

    Args:
        output_path (Path): Folder with the CSV output of the simulation.
        units (list[str]): Unit ids to read.

    Returns:
        pd.DataFrame: Columns start_time, unit_id, A, W (MWh, buy +), price_lem,
        price_wm (EUR/MWh). One row per unit and period that has an order on at least
        one market; a market without an order has volume 0 and price NaN.
    """
    orders = pd.read_csv(output_path / "market_orders.csv", parse_dates=["start_time"])
    orders = orders[orders.unit_id.isin(units)]
    parts = []
    for market_id, volume, price in ((LEM_MARKET_ID, "A", "price_lem"),
                                     (WM_MARKET_ID, "W", "price_wm")):
        market = orders[orders.market_id == market_id]
        # ASSUME sign (buy < 0) -> settlement sign (buy +); MW -> MWh per period.
        # A unit has at most one order per market and period (checked below).
        if market.duplicated(["start_time", "unit_id"]).any():
            raise ValueError(f"more than one order per unit and period in {market_id}")
        part = market.set_index(["start_time", "unit_id"])
        parts.append(
            pd.DataFrame(
                {
                    volume: -part["accepted_volume"] * STEP_HOURS,
                    price: part["accepted_price"],
                }
            )
        )
    result = pd.concat(parts, axis=1)
    result[["A", "W"]] = result[["A", "W"]].fillna(0.0) + 0.0  # -0.0 -> 0.0
    return result.reset_index()


def simulation_index(scenario_path: Path) -> pd.DatetimeIndex:
    """Time axis of the simulation from config.yaml (same as loader_csv.py 592-616)."""
    with open(scenario_path / "config.yaml") as f:
        config = next(iter(yaml.safe_load(f).values()))
    return pd.date_range(
        config["start_date"], config["end_date"], freq=config["time_step"]
    )


def battery_dispatch(
    output_path: Path, scenario_path: Path, unit: str
) -> pd.Series:
    """
    Power a battery really runs (MW, ASSUME sign: discharge > 0), on the simulation axis.

    Builds the battery with ASSUME's Storage class and the parameters of
    storage_units.csv (charging power negative, as loader_csv.py 636-642), writes the
    accepted volumes of both markets into outputs["energy"] (as set_dispatch_plan(),
    base.py from line 121) and runs execute_current_dispatch() (storage.py 277-340),
    which cuts the power at the SOC limits.
    """
    index = simulation_index(scenario_path)
    row = pd.read_csv(scenario_path / "storage_units.csv", index_col="name").loc[unit]
    battery = Storage(
        id=unit,
        unit_operator="settlement",
        technology="battery",
        bidding_strategies={},
        forecaster=UnitForecaster(index),
        max_power_charge=-abs(float(row.max_power_charge)),
        max_power_discharge=float(row.max_power_discharge),
        capacity=float(row.capacity),
        min_soc=float(row.min_soc),
        max_soc=float(row.max_soc),
        initial_soc=float(row.initial_soc),
        efficiency_charge=float(row.efficiency_charge),
        efficiency_discharge=float(row.efficiency_discharge),
    )
    orders = pd.read_csv(output_path / "market_orders.csv", parse_dates=["start_time"])
    accepted = orders[orders.unit_id == unit].groupby("start_time").accepted_volume.sum()
    for t, volume in accepted.items():
        battery.outputs["energy"].at[t] = float(volume)
    battery.execute_current_dispatch(index[0], index[-1])
    return pd.Series([battery.outputs["energy"].at[t] for t in index], index=index)


def dispatch_check(output_path: Path, unit: str, dispatch: pd.Series) -> float:
    """
    Max |rebuilt dispatch - unit_dispatch.csv| (MW) where ASSUME wrote the dispatch.

    Returns:
        float: about 0 if battery_dispatch() repeats ASSUME correctly.
    """
    written = pd.read_csv(output_path / "unit_dispatch.csv", parse_dates=["time"])
    written = written[written.unit == unit].set_index("time")["power"]
    common = written.index.intersection(dispatch.index)
    return float((written.loc[common] - dispatch.loc[common]).abs().max())


def realised_volumes(
    output_path: Path,
    scenario_path: Path,
    groups: dict[str, str],
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Realised net volume L (MWh per period, buy +) of every settled participant.

    Returns:
        pd.DataFrame: Index = period start, one column per unit.

    """
    columns = {}
    demand = pd.read_csv(scenario_path / "demand_df.csv", index_col=0, parse_dates=True)
    availability = pd.read_csv(
        scenario_path / "availability_df.csv", index_col=0, parse_dates=True
    )
    plants = pd.read_csv(scenario_path / "powerplant_units.csv", index_col="name")
    for unit, group in groups.items():
        if group == GROUP_INDEPENDENT:
            columns[unit] = demand[unit].abs() * STEP_HOURS
        elif group == GROUP_COMMUNITY_PV:
            columns[unit] = -availability[unit] * plants.at[unit, "max_power"] * STEP_HOURS
        elif group == GROUP_BATTERY:
            # ASSUME power: discharge (sell) > 0 -> settlement sign: sell < 0
            power = battery_dispatch(output_path, scenario_path, unit).reindex(index)
            columns[unit] = -power * STEP_HOURS
    return pd.DataFrame(columns).reindex(index)


def settle_participants(
    output_path: Path,
    scenario_path: Path,
    start: str,
    end: str,
    rebap: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Settle every participant except the aggregator for the delivery periods start..end.

    Returns:
        pd.DataFrame: One row per unit and period with group, L, A, W, D, prices and
        the costs cost_lem, cost_wm, cost_imbalance, cost_total (EUR).
    """
    index = pd.date_range(start, end, freq=pd.Timedelta(hours=STEP_HOURS))
    groups = participant_groups(scenario_path)
    orders = read_unit_orders(output_path, list(groups))
    realised = realised_volumes(output_path, scenario_path, groups, index)
    if rebap is None:
        rebap = read_rebap(index)

    rows = []
    for unit, group in groups.items():
        unit_orders = (
            orders[orders.unit_id == unit].set_index("start_time").reindex(index)
        )
        frame = pd.DataFrame(index=index)
        frame["unit_id"] = unit
        frame["group"] = group
        frame["A"] = unit_orders["A"].fillna(0.0)
        frame["W"] = unit_orders["W"].fillna(0.0)
        frame["price_lem"] = unit_orders["price_lem"]
        frame["price_wm"] = unit_orders["price_wm"]
        frame["L"] = realised[unit]
        frame["D"] = frame.L - frame.A - frame.W
        frame["rebap"] = rebap.reindex(index)
        # a market without accepted volume costs nothing, also when its price is NaN
        frame["cost_lem"] = (frame.A * frame.price_lem).where(frame.A != 0, 0.0)
        frame["cost_wm"] = (frame.W * frame.price_wm).where(frame.W != 0, 0.0)
        frame["cost_imbalance"] = frame.D * frame.rebap
        frame["cost_total"] = frame.cost_lem + frame.cost_wm + frame.cost_imbalance
        rows.append(frame)
    result = pd.concat(rows)
    result.index.name = "start_time"
    if result[["L", "A", "W", "D", "cost_total"]].isna().any().any():
        raise ValueError("missing values in the participant settlement")
    return result


def check_lem_balance(output_path: Path, start: str, end: str) -> float:
    """
    Largest absolute sum of LEM payments of all participants in one period (EUR).

    With a uniform price, sum(price x accepted volume) over buyers and sellers is zero.

    Returns:
        float: Maximum over the periods of |sum of LEM payments|; about 0 if correct.
    """
    orders = pd.read_csv(output_path / "market_orders.csv", parse_dates=["start_time"])
    lem = orders[(orders.market_id == LEM_MARKET_ID)
                 & (orders.start_time >= start) & (orders.start_time <= end)]
    payments = -lem.accepted_volume * STEP_HOURS * lem.accepted_price  # buy + pays
    if lem.empty:
        return 0.0
    return float(payments.groupby(lem.start_time).sum().abs().max())


def summarise(participants: pd.DataFrame) -> pd.DataFrame:
    """Weekly totals per participant group (MWh, EUR)."""
    columns = ["L", "A", "W", "D", "cost_lem", "cost_wm", "cost_imbalance", "cost_total"]
    return participants.groupby("group")[columns].sum()

