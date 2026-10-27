# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Round 2: add the SimBench batteries and the naive price forecast to the scenario.

Writes two inputs of the scenario folder:
    1. storage_units.csv: the four batteries of SimBench LV4.101 (scenario 2), each an
       independent participant (decision D5) with the strategy
       ``storage_two_market`` on LEM_DA and WM_DA.
    2. Column ``price_LEM_DA`` of forecasts_df.csv: naive similar-day forecast of the
       WM price. ASSUME passes this column to every unit as its price forecast for
       LEM_DA; the battery strategy uses it for both markets (rule R7).

Where each written column goes in ASSUME (line numbers: branch Yihan_local_aggregator,
checked 2026-10-26):

    storage_units.csv column  <- source              -> used in ASSUME as
    ------------------------  ------------------     ------------------------------------
    name (index)              SimBench node          unit id
    technology                "battery" [OWN] label  unit.technology (text only)
    bidding_LEM_DA/_WM_DA     STRATEGY below         strategy per market
                                                     (loader_csv.py, read_units, 478-484)
    max_power_charge          SimBench sR (MVA)      unit.max_power_charge, made negative
                                                     (loader_csv.py, 636-640)
    max_power_discharge       SimBench sR (MVA)      unit.max_power_discharge
    efficiency_charge         SimBench etaStore      unit.efficiency_charge
                                                     (storage.py, 201-202)
    efficiency_discharge      SimBench etaStore      unit.efficiency_discharge
    min_soc / max_soc         SOC_MIN / SOC_MAX      unit.min_soc / unit.max_soc
                                                     (storage.py, 126-127)
    initial_soc               SOC_INIT               unit.initial_soc (storage.py, 136)
    capacity                  SimBench eStore (MWh)  unit.capacity (storage.py, 125)
    additional_cost_*         0                      marginal cost of the unit
    natural_inflow            0                      pumped hydro only, not used here
    unit_operator             one per battery (D5)   operator that sends the bids
                                                     (loader_csv.py, 487-495)

    The path from a row to the unit: loader_csv.py, line 620 reads the file;
    read_units (lines 478-505) turns each row into ``unit_params``; world.py,
    lines 524-528 calls ``Storage(**unit_params)`` (assume/units/storage.py,
    lines 56-75).

    forecasts_df.csv column price_LEM_DA -> unit.forecaster.price["LEM_DA"]
    (assume/common/forecaster.py, calculate_base_forecasts, line 71). Without this
    column ASSUME would compute its own merit-order forecast ("price_naive_forecast"),
    which is not the naive forecast of Weron (2014).

Run from the repository root:
    python examples/inputs/LocalRetailer_test/round2_add_batteries.py

Sources:
    Battery data: SimBench LV4.101, simbench_raw/Storage.csv (sR [MVA], eStore [MWh],
        etaStore). etaStore is used as one-way efficiency for charging and
        discharging; its exact meaning is not yet checked in the SimBench documentation.
    SOC limits 10-90 %: Wagner, L. P., Bisson, R., Gehlhoff, F. (2026). Market Strategy
        Evaluation for Prosumers in Local Electricity Markets. arXiv:2607.18272v1
        (preprint), Sect. III (decision E2).
    Initial SOC 50 %: Qussous, R., Harder, N., Weidlich, A. (2022). Understanding Power
        Market Dynamics by Reflecting Market Interrelations and Flexibility-Oriented
        Bidding Strategies. Energies 15(2), 494, Sect. 4 (decision E3).
    Naive forecast: Weron, R. (2014). Electricity price forecasting: A review of the
        state-of-the-art with a look into the future. International Journal of
        Forecasting 30(4), 1030-1081, Sect. 3.8.1, p. 1049-1050.
    WM prices: SMARD, read and converted to simulation time as in round 1
        (simbench_to_assume.py: WM_PRICE_FILE, WM_PRICE_COLUMN, to_simulation_time).
"""

from pathlib import Path

import pandas as pd

from simbench_to_assume import WM_PRICE_COLUMN, WM_PRICE_FILE, to_simulation_time

SCENARIO_PATH = Path(__file__).resolve().parent
STORAGE_FILE = SCENARIO_PATH / "simbench_raw" / "Storage.csv"
FORECASTS_FILE = SCENARIO_PATH / "forecasts_df.csv"
STORAGE_UNITS_FILE = SCENARIO_PATH / "storage_units.csv"

SOC_MIN = 0.10  # decision E2
SOC_MAX = 0.90  # decision E2
SOC_INIT = 0.50  # decision E3
STRATEGY = "storage_two_market"  # registered in assume/strategies/__init__.py


def build_storage_units(path: Path = STORAGE_FILE) -> pd.DataFrame:
    """
    One row per SimBench battery in the format of ASSUME's storage_units.csv.

    Power in MW, capacity in MWh (as in SimBench). ASSUME makes max_power_charge
    negative when loading (loader_csv.py, line 638).
    """
    raw = pd.read_csv(path, sep=";")
    names = [f"Battery_Bus{node.split()[-1]}" for node in raw["node"]]
    return pd.DataFrame(
        {
            "name": names,
            "technology": "battery",
            "bidding_LEM_DA": STRATEGY,
            "bidding_WM_DA": STRATEGY,
            "max_power_charge": raw["sR"],
            "max_power_discharge": raw["sR"],
            "efficiency_charge": raw["etaStore"],
            "efficiency_discharge": raw["etaStore"],
            "min_soc": SOC_MIN,
            "max_soc": SOC_MAX,
            "initial_soc": SOC_INIT,
            "capacity": raw["eStore"],
            "additional_cost_charge": 0.0,  # marginal cost negligible (Qussous 2022, p. 12)
            "additional_cost_discharge": 0.0,
            "natural_inflow": 0.0,
            "unit_operator": [f"battery_operator_{i + 1}" for i in range(len(raw))],
        }
    ).set_index("name")


def read_wm_price_hourly() -> pd.Series:
    """SMARD hourly WM price in simulation time (same reading as round 1)."""
    raw = pd.read_csv(WM_PRICE_FILE, sep=";", decimal=",", na_values="-")
    local_time = pd.DatetimeIndex(pd.to_datetime(raw["Datum von"], format="%d.%m.%Y %H:%M"))
    hourly = pd.Series(raw[WM_PRICE_COLUMN].to_numpy(), index=to_simulation_time(local_time))
    return hourly.sort_index()


def naive_forecast(prices: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """
    Naive similar-day forecast on the given index.

    Monday, Saturday, Sunday: same time one week earlier; Tuesday to Friday: same
    time one day earlier (Weron 2014, p. 1049-1050).
    """
    lag = [pd.Timedelta(days=7) if t.dayofweek in (0, 5, 6) else pd.Timedelta(days=1)
           for t in index]
    source_times = index - pd.TimedeltaIndex(lag)
    values = prices.reindex(source_times, method="ffill")  # hourly value for 4 quarter hours
    if values.isna().any():
        raise ValueError("price history does not cover the forecast period")
    return pd.Series(values.to_numpy(), index=index)


def main() -> None:
    storage = build_storage_units()
    storage.to_csv(STORAGE_UNITS_FILE)
    print(f"wrote {STORAGE_UNITS_FILE.name}: {len(storage)} batteries")

    forecasts = pd.read_csv(FORECASTS_FILE, index_col=0, parse_dates=True)
    forecasts["price_LEM_DA"] = naive_forecast(read_wm_price_hourly(), forecasts.index)
    forecasts.to_csv(FORECASTS_FILE)
    print(f"added column price_LEM_DA to {FORECASTS_FILE.name}")
    print("check 2016-06-21 12:00:", forecasts.loc["2016-06-21 12:00", "price_LEM_DA"],
          "(expected 25.24 = WM price of Monday 2016-06-20 12:00)")


if __name__ == "__main__":
    main()
