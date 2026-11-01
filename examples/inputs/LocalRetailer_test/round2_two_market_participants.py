# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Round 2: independent demand nodes and community PV bid on LEM_DA and WM_DA.

Run order (from the repository root):
    1. python examples/inputs/LocalRetailer_test/simbench_to_assume.py
    2. python examples/inputs/LocalRetailer_test/round2_add_batteries.py   (optional)
    3. python examples/inputs/LocalRetailer_test/round2_two_market_participants.py
Step 1 rewrites all unit tables and forecasts_df.csv, so steps 2 and 3 must be run
again after every run of step 1.

What this script changes in the scenario folder:

    file · rows                                    column            old -> new
    ---------------------------------------------  ----------------  ---------------------------------------
    demand_units.csv · unit_operator =             bidding_LEM_DA    demand_energy_naive
      independent_operator (9 rows)                                  -> demand_two_market_fixed_share
                                                   bidding_WM_DA     "" -> demand_two_market_fixed_share
    powerplant_units.csv · unit_operator =         bidding_LEM_DA    powerplant_energy_naive
      LEC_operator (CommunityPV_Bus41)                               -> powerplant_two_market_fixed_share
                                                   bidding_WM_DA     "" -> powerplant_two_market_fixed_share
    forecasts_df.csv                               price_LEM_DA      written if missing (naive forecast,
                                                                     same function as round2_add_batteries.py)

Where the changed columns go in ASSUME (branch Yihan_local_aggregator, checked
2026-10-26):
    bidding_LEM_DA / bidding_WM_DA -> loader_csv.py, read_units, lines 478-484 ->
        assume/strategies/__init__.py (names registered at the end of the file) ->
        assume/strategies/fixed_share_two_market.py, FixedShareTwoMarketStrategy
    price_LEM_DA -> loader_csv.py 651 -> forecaster.py line 71 ->
        unit.forecaster.price["LEM_DA"] (LEM price of the strategy)

The aggregator (unit_operator = local_retailer) and the wholesale units
(Grid_operator) are not changed.

Sources: see the module docstring of fixed_share_two_market.py (rules R6, R7, R8,
decisions D1, D3, D4 in the model document, section 7.1).
"""

from pathlib import Path

import pandas as pd

from round2_add_batteries import naive_forecast, read_wm_price_hourly

SCENARIO_PATH = Path(__file__).resolve().parent
DEMAND_UNITS_FILE = SCENARIO_PATH / "demand_units.csv"
POWERPLANT_UNITS_FILE = SCENARIO_PATH / "powerplant_units.csv"
FORECASTS_FILE = SCENARIO_PATH / "forecasts_df.csv"

# Operators of the participants that switch to two markets (simbench_to_assume.py,
# OPERATOR_INDEPENDENT and OPERATOR_COMMUNITY)
INDEPENDENT_OPERATOR = "independent_operator"
COMMUNITY_OPERATOR = "LEC_operator"
# Strategy names, registered in assume/strategies/__init__.py
DEMAND_STRATEGY = "demand_two_market_fixed_share"
POWERPLANT_STRATEGY = "powerplant_two_market_fixed_share"
MARKETS = ("LEM_DA", "WM_DA")


def set_strategy(path: Path, operator: str, strategy: str) -> list[str]:
    """
    Write ``strategy`` into bidding_LEM_DA and bidding_WM_DA of all rows of
    ``operator`` in the unit table ``path``.

    Returns:
        list[str]: Names of the changed units.
    """
    units = pd.read_csv(path, index_col=0, dtype=str, keep_default_na=False)
    rows = units["unit_operator"] == operator
    if not rows.any():
        raise ValueError(f"no unit of {operator} in {path.name}; run simbench_to_assume.py first")
    for market in MARKETS:
        units.loc[rows, f"bidding_{market}"] = strategy
    units.to_csv(path)
    return list(units.index[rows])


def ensure_lem_forecast() -> None:
    """Write the naive forecast into price_LEM_DA of forecasts_df.csv if it is missing."""
    forecasts = pd.read_csv(FORECASTS_FILE, index_col=0, parse_dates=True)
    if "price_LEM_DA" in forecasts.columns:
        print("forecasts_df.csv already has price_LEM_DA (written by round2_add_batteries.py)")
        return
    forecasts["price_LEM_DA"] = naive_forecast(read_wm_price_hourly(), forecasts.index)
    forecasts.to_csv(FORECASTS_FILE)
    print("added column price_LEM_DA to forecasts_df.csv")


def main() -> None:
    demand = set_strategy(DEMAND_UNITS_FILE, INDEPENDENT_OPERATOR, DEMAND_STRATEGY)
    print(f"{DEMAND_UNITS_FILE.name}: {len(demand)} independent nodes -> {DEMAND_STRATEGY}")
    plants = set_strategy(POWERPLANT_UNITS_FILE, COMMUNITY_OPERATOR, POWERPLANT_STRATEGY)
    print(f"{POWERPLANT_UNITS_FILE.name}: {plants} -> {POWERPLANT_STRATEGY}")
    ensure_lem_forecast()
    forecasts = pd.read_csv(FORECASTS_FILE, index_col=0, parse_dates=True)
    print("check 2016-06-21 11:00 UTC: price_LEM_DA =",
          forecasts.loc["2016-06-21 11:00", "price_LEM_DA"],
          "(expected 25.24 = WM price of Monday 2016-06-20 11:00 UTC = 13:00 local)")


if __name__ == "__main__":
    main()
