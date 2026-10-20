"""
WP7: ex-post settlement of one week, before ASSUME exists (offline).

Without ASSUME there is no real LEM clearing, so the actually accepted LEM volume A is
unknown. Two offline cases bound the real outcome:
    "lem_as_expected"  A = a_hat  (the LEM takes exactly what the optimiser expected)
    "lem_takes_nothing" A = 0     (the LEM takes nothing, every LEM bid becomes imbalance)
The real result from ASSUME (WP8) lies between these two cases.

Reads:   wp5_inputs_2016-06-20.csv          (WP5)
Uses:    solve_alpha() from wp4_multi_period.py (your WP4)
         settle()      from ../settlement.py    (b22d8b5, unchanged)
Writes:  wp7_settlement_<case>.csv

Sources (full citations: see the reference list in wp5_inputs.py)
    Deviation D = L - A - W and cost C = C_LEM + C_WM + C_reBAP: decision of the thesis;
        implemented in settle() of b22d8b5. Deviations settled at the imbalance price:
        [Iria2019] Sect. 3.1, Eq. (1) [PART, page TODO]; [Ottesen2016] p. 830 [OK].
    One symmetric reBAP per quarter hour: netztransparenz.de, Erlaeuterungen zur
        reBAP-Berechnung (10.07.2023), Sect. 2.1, S. 8 [OK].
    WM bids are always fully accepted (price taker): [Iria2019] p. 1363 [OK].
    The two offline cases (A = a_hat, A = 0) are bounds for testing only: [OWN].
        They are not results of the thesis; the real A comes from ASSUME (WP8).

Run from the repository root (with .venv active):
    python examples/inputs/LocalRetailer_test/course/wp7_settlement.py
"""

import sys
from pathlib import Path

import pandas as pd

COURSE_PATH = Path(__file__).resolve().parent
SCENARIO_PATH = COURSE_PATH.parent
sys.path.insert(0, str(SCENARIO_PATH))

from settlement import settle  # noqa: E402
from wp4_multi_period import solve_alpha  # noqa: E402

INPUT_FILE = COURSE_PATH / "wp5_inputs_2016-06-20.csv"
DECISION_COLUMNS = ["L_hat", "p_lem", "p_wm", "r_hat", "S_hat"]  # known when deciding


def build_orders(decisions: pd.DataFrame, inputs: pd.DataFrame, case: str) -> pd.DataFrame:
    """
    Orders and accepted volumes in the column layout that settle() expects.

    All volumes in MWh per quarter hour, + buy / - sell (same sign as L_hat).
    """
    orders = pd.DataFrame(index=decisions.index)
    orders["q_lem_bid"] = decisions.alpha * decisions.L_hat
    orders["q_wm_bid"] = (1 - decisions.alpha) * decisions.L_hat
    orders["q_wm_accepted"] = orders.q_wm_bid  # WM: price taker, everything accepted
    if case == "lem_as_expected":
        orders["q_lem_accepted"] = decisions.a_hat
    elif case == "lem_takes_nothing":
        orders["q_lem_accepted"] = 0.0
    else:
        raise ValueError(f"unknown case {case}")
    # offline: LEM clearing price = the forecast (no clearing exists yet)
    orders["price_lem"] = decisions.p_lem
    orders["price_wm"] = inputs.p_wm_actual
    return orders


def check(settlement: pd.DataFrame, case: str) -> None:
    """Two identities that must hold in every case; print OK / FAIL."""
    # 1. perfect load forecast and full WM acceptance -> imbalance = unaccepted LEM volume
    gap = (settlement.imbalance - settlement.uncovered).abs().max()
    print(f"  check 1 imbalance = LEM bid - LEM accepted: max gap {gap:.2e}",
          "OK" if gap < 1e-9 else "FAIL")
    # 2. total = sum of the three parts
    gap = (settlement.cost_total
           - settlement[["cost_lem", "cost_wm", "cost_imbalance"]].sum(axis=1)).abs().max()
    print(f"  check 2 total = LEM + WM + reBAP: max gap {gap:.2e}",
          "OK" if gap < 1e-9 else "FAIL")


def main() -> None:
    inputs = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    decisions = solve_alpha(inputs[DECISION_COLUMNS])  # ex ante: only decision columns
    print(f"ex ante (optimiser):        total {decisions.cost_total.sum():9.4f} EUR")

    for case in ["lem_as_expected", "lem_takes_nothing"]:
        orders = build_orders(decisions, inputs, case)
        settlement = settle(orders, inputs.L_actual, rebap=inputs.rebap_actual)  # ex post
        totals = settlement[["cost_lem", "cost_wm", "cost_imbalance", "cost_total"]].sum()
        print(f"ex post {case:18s}: LEM {totals.cost_lem:9.4f} | WM {totals.cost_wm:9.4f} | "
              f"reBAP {totals.cost_imbalance:9.4f} | total {totals.cost_total:9.4f} EUR")
        check(settlement, case)
        settlement.to_csv(COURSE_PATH / f"wp7_settlement_{case}.csv")


if __name__ == "__main__":
    main()
