"""
WP8: run one ASSUME simulation of the local retailer and settle it.

Before running: build the 15-minute scenario once with
    python examples/inputs/LocalRetailer_test/simbench_to_assume.py

Run from the repository root (with .venv active):
    python examples/inputs/LocalRetailer_test/course/wp8_run_assume.py                 # optimised alpha, option 1
    python examples/inputs/LocalRetailer_test/course/wp8_run_assume.py --alpha 0.5     # fixed alpha (b22d8b5 strategy)
    python examples/inputs/LocalRetailer_test/course/wp8_run_assume.py --s-hat-feedback  # optimised alpha, option 2

Results (three layers, kept apart) in course/results/<run>/:
    decisions.csv   optimiser layer: inputs, alpha, expected costs (only optimised runs)
    settlement.csv  market + ex-post layer: offered and accepted volumes, clearing prices,
                    deviation D = L - A - W, costs C_LEM + C_WM + C_reBAP
    summary.csv     one row with the weekly totals
All volumes in MWh per 15 min (+ buy / - sell), prices in EUR/MWh, costs in EUR (+ pay).

Sources: see the module docstrings of aggregator_alpha_optimized.py and settlement.py.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import pandas as pd

COURSE_PATH = Path(__file__).resolve().parent
SCENARIO_PATH = COURSE_PATH.parent
REPO_ROOT = SCENARIO_PATH.parents[2]
sys.path.insert(0, str(REPO_ROOT))  # "assume" = this repository
sys.path.insert(0, str(SCENARIO_PATH))  # settlement.py

import settlement  # noqa: E402
from assume import World  # noqa: E402
from assume.scenario.loader_csv import load_scenario_folder  # noqa: E402
from assume.strategies.aggregator_alpha_optimized import (  # noqa: E402
    AlphaOptimizedLocalRetailerStrategy,
)

STUDY_CASE = "local_retailer_demo"
OPERATOR = "local_retailer"
MARKETS = ("LEM_DA", "WM_DA")
DELIVERY_START, DELIVERY_END = "2016-06-20 00:00", "2016-06-26 23:45"  # one week


def run(alpha: float | None, s_hat_feedback: bool, seed: int = 1) -> Path:
    """Run ASSUME once; return the folder with the results of this run."""
    # The LEM now clears with pay_as_clear_prorata (equal prices share pro rata), so
    # the result no longer depends on random numbers: seeds 1, 2, 3 gave identical
    # results (checked 2026-10-19). The seed is kept only as a safeguard, e.g. if a
    # market is switched back to pay_as_clear, which breaks price ties at random
    # (clearing_algorithms/simple.py, sort key (price, random.random())).
    random.seed(seed)
    if alpha is None:
        run_id = "optimised_option2" if s_hat_feedback else "optimised_option1"
    else:
        run_id = f"fixed_alpha_{alpha:.2f}".replace(".", "_")
    run_id += f"_seed{seed}"
    result_path = COURSE_PATH / "results" / run_id
    result_path.mkdir(parents=True, exist_ok=True)
    export_path = result_path / "assume_output"

    world = World(database_uri=None, export_csv_path=str(export_path))
    load_scenario_folder(
        world,
        inputs_path=str(SCENARIO_PATH.parent),
        scenario=SCENARIO_PATH.name,
        study_case=STUDY_CASE,
    )
    operator = world.unit_operators[OPERATOR]
    strategy = operator.portfolio_strategies[MARKETS[0]]
    if alpha is None:
        # replace the fixed-alpha strategy of b22d8b5; one object for both markets
        strategy = AlphaOptimizedLocalRetailerStrategy()
        strategy.S_HAT_FROM_FEEDBACK = s_hat_feedback
        for market_id in MARKETS:
            operator.portfolio_strategies[market_id] = strategy
    else:
        strategy.alpha = alpha
    world.run()

    # --- layer 1: optimiser ------------------------------------------------------
    if alpha is None:
        decisions = pd.DataFrame(strategy.decision_log).set_index("start_time")
        decisions.to_csv(result_path / "decisions.csv")

    # --- layers 2 and 3: market results and ex-post settlement --------------------
    output = export_path / f"{SCENARIO_PATH.name}_{STUDY_CASE}"
    orders = settlement.read_portfolio_orders(output)
    demand_units, building_units = settlement.read_portfolio_units(SCENARIO_PATH)
    residual = settlement.build_realised_residual_load(
        SCENARIO_PATH, demand_units, building_units
    )
    orders = orders.loc[DELIVERY_START:DELIVERY_END]
    settled = settlement.settle(orders, residual)
    settled.to_csv(result_path / "settlement.csv")

    totals = settled[
        ["q_lem_bid", "q_lem_accepted", "q_wm_accepted", "imbalance",
         "cost_lem", "cost_wm", "cost_imbalance", "cost_total"]
    ].sum()
    summary = pd.DataFrame([{"run": run_id, "seed": seed} | totals.round(6).to_dict()])
    if alpha is None:
        summary["expected_cost_total"] = round(decisions.cost_total.sum(), 6)
    summary.to_csv(result_path / "summary.csv", index=False)
    print(summary.T.to_string(header=False))
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=None,
                        help="fixed alpha (b22d8b5 strategy); omit for optimised alpha")
    parser.add_argument("--s-hat-feedback", action="store_true",
                        help="option 2: S_hat from own LEM feedback")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (safeguard only; pro-rata clearing is deterministic)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    run(args.alpha, args.s_hat_feedback, args.seed)


if __name__ == "__main__":
    main()
