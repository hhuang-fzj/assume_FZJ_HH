# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
WP15: round-2 runs for beta = 0 / 0.5 / 1 and both tie-break rules, with settlement of
all participants.

Run from the repository root (with .venv active):
    python examples/inputs/LocalRetailer_test/Course/wp15_run_round2.py
    python examples/inputs/LocalRetailer_test/Course/wp15_run_round2.py --beta 0.5 --tie lem --s-hat feedback
    python examples/inputs/LocalRetailer_test/Course/wp15_run_round2.py --no-prepare

Step 1 (skipped with --no-prepare) rebuilds the scenario folder for round 2:
    simbench_to_assume.py                -> round-1 scenario (overwrites all inputs!)
    round2_add_batteries.py              -> 4 batteries + column price_LEM_DA
    round2_two_market_participants.py    -> independent demand and community PV on
                                            LEM and WM (FixedShareTwoMarketStrategy)
    round2_lem_limits.py                 -> F4: LEM limits -500 / 3000
To go back to round 1, run simbench_to_assume.py again.

Step 2 runs one ASSUME simulation per (beta, tie, s_hat) in its own Python process (one
World per process, as in wp8_run_assume.py):
    beta  -> BETA of storage_two_market.py and fixed_share_two_market.py (decision D1)
    tie   -> F5_TIE_BREAK of AlphaRound2LocalRetailerStrategy ("lem" or "wm")
    s_hat -> S_HAT_FROM_FEEDBACK of the same class: "full" = S_hat = |L_hat|,
             "feedback" = S_hat from the own LEM results of the last 7 days
The aggregator (local_retailer) uses AlphaRound2LocalRetailerStrategy with F1-F3, F5.

Results in Course/results/round2_beta_<b>_tie_<t>_shat_<s>/:
    decisions.csv      optimiser layer of the aggregator (with columns tie, p_lem_source)
    settlement.csv     aggregator, same as round 1 (settlement.settle())
    participants.csv   all other participants (settlement_participants.py)
    summary.csv        one row: aggregator totals, totals per participant group, checks
and Course/results/wp15_summary.csv with one row per run.

Checks written into summary.csv:
    lem_balance_max_eur      max |sum of LEM payments| in one period, must be ~0
    aggregator_check_mwh     max |L - A - W - D| of the aggregator, must be ~0
    battery_dispatch_check_mw  max |rebuilt battery dispatch - unit_dispatch.csv|, ~0
Sign convention: buy +, sell -; MWh per 15 min; cost + = pays.

Sources: see the module docstrings of aggregator_alpha_round2.py,
fixed_share_two_market.py, storage_two_market.py and settlement_participants.py.
"""

import argparse
import logging
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

COURSE_PATH = Path(__file__).resolve().parent
SCENARIO_PATH = COURSE_PATH.parent
REPO_ROOT = SCENARIO_PATH.parents[2]
sys.path.insert(0, str(REPO_ROOT))  # "assume" = this repository
sys.path.insert(0, str(SCENARIO_PATH))  # settlement.py and the round-2 scripts

STUDY_CASE = "local_retailer_demo"
OPERATOR = "local_retailer"
MARKETS = ("LEM_DA", "WM_DA")
DELIVERY_START, DELIVERY_END = "2016-06-20 00:00", "2016-06-26 23:45"  # as wp8
BETAS = (0.0, 0.5, 1.0)  # decision D1
TIE_RULES = ("lem", "wm")  # decision F5
S_HAT_RULES = ("full", "feedback")  # S_hat rule (model document, section 3.6)
RESULTS_PATH = COURSE_PATH / "results"


def run_id(beta: float, tie: str, s_hat: str) -> str:
    return f"round2_beta_{beta:.2f}_tie_{tie}_shat_{s_hat}".replace(".", "_")


def prepare_scenario() -> None:
    """Rebuild the scenario folder for round 2 (overwrites the round-1 inputs)."""
    import round2_add_batteries
    import round2_lem_limits
    import round2_two_market_participants
    import simbench_to_assume

    simbench_to_assume.main()
    round2_add_batteries.main()
    round2_two_market_participants.main()
    round2_lem_limits.main()


def check_scenario() -> None:
    """Stop if the scenario folder is not the round-2 scenario."""
    import yaml

    with open(SCENARIO_PATH / "config.yaml") as f:
        lem = yaml.safe_load(f)[STUDY_CASE]["markets_config"]["LEM_DA"]
    if lem["minimum_bid_price"] != -500 or lem["maximum_bid_price"] != 3000:
        raise RuntimeError("config.yaml has no round-2 LEM limits; run without --no-prepare")
    forecasts = pd.read_csv(SCENARIO_PATH / "forecasts_df.csv", nrows=1)
    if "price_LEM_DA" not in forecasts.columns:
        raise RuntimeError("forecasts_df.csv has no column price_LEM_DA")
    if not (SCENARIO_PATH / "storage_units.csv").exists():
        raise RuntimeError("storage_units.csv missing; run without --no-prepare")


def run_single(beta: float, tie: str, s_hat: str, seed: int = 1) -> Path:
    """One simulation with settlement of all participants; returns the result folder."""
    import assume.strategies.fixed_share_two_market as fixed_share
    import assume.strategies.storage_two_market as storage
    import settlement
    import settlement_participants
    from assume import World
    from assume.scenario.loader_csv import load_scenario_folder
    from assume.strategies.aggregator_alpha_round2 import (
        AlphaRound2LocalRetailerStrategy,
    )

    random.seed(seed)  # pay_as_clear_prorata does not use random numbers; safeguard
    storage.BETA = beta
    fixed_share.BETA = beta

    result_path = RESULTS_PATH / run_id(beta, tie, s_hat)
    result_path.mkdir(parents=True, exist_ok=True)
    export_path = result_path / "assume_output"

    world = World(database_uri=None, export_csv_path=str(export_path))
    load_scenario_folder(
        world,
        inputs_path=str(SCENARIO_PATH.parent),
        scenario=SCENARIO_PATH.name,
        study_case=STUDY_CASE,
    )
    strategy = AlphaRound2LocalRetailerStrategy()
    strategy.F5_TIE_BREAK = tie
    strategy.S_HAT_FROM_FEEDBACK = s_hat == "feedback"
    operator = world.unit_operators[OPERATOR]
    for market_id in MARKETS:  # one object for both markets, as in wp8_run_assume.py
        operator.portfolio_strategies[market_id] = strategy
    world.run()

    # --- layer 1: optimiser of the aggregator --------------------------------------
    decisions = pd.DataFrame(strategy.decision_log).set_index("start_time")
    decisions.to_csv(result_path / "decisions.csv")

    # --- layer 2: aggregator settlement (same as round 1) --------------------------
    output = export_path / f"{SCENARIO_PATH.name}_{STUDY_CASE}"
    orders = settlement.read_portfolio_orders(output)
    demand_units, building_units = settlement.read_portfolio_units(SCENARIO_PATH)
    residual = settlement.build_realised_residual_load(
        SCENARIO_PATH, demand_units, building_units
    )
    full_index = pd.date_range(DELIVERY_START, DELIVERY_END, freq="15min")
    # periods without any portfolio order (L_hat = 0) get zero volumes
    orders = orders.reindex(full_index).fillna(
        {"q_lem_bid": 0.0, "q_lem_accepted": 0.0, "q_wm_bid": 0.0, "q_wm_accepted": 0.0}
    )
    settled = settlement.settle(orders, residual)
    settled.to_csv(result_path / "settlement.csv")
    aggregator_check = float(
        (settled.residual_load - settled.q_lem_accepted - settled.q_wm_accepted
         - settled.imbalance).abs().max()
    )

    # --- layer 3: all other participants -------------------------------------------
    participants = settlement_participants.settle_participants(
        output, SCENARIO_PATH, DELIVERY_START, DELIVERY_END
    )
    participants.to_csv(result_path / "participants.csv")
    groups = settlement_participants.summarise(participants)
    balance = settlement_participants.check_lem_balance(output, DELIVERY_START, DELIVERY_END)
    # rebuilt battery dispatch vs. unit_dispatch.csv (where ASSUME wrote it)
    battery_check = max(
        (
            settlement_participants.dispatch_check(
                output, unit,
                settlement_participants.battery_dispatch(output, SCENARIO_PATH, unit),
            )
            for unit, group in settlement_participants.participant_groups(
                SCENARIO_PATH
            ).items()
            if group == settlement_participants.GROUP_BATTERY
        ),
        default=0.0,
    )

    week = decisions.loc[DELIVERY_START:DELIVERY_END]
    row = {
        "run": run_id(beta, tie, s_hat),
        "beta": beta,
        "tie_rule": tie,
        "s_hat_rule": s_hat,
        "agg_q_lem_accepted": settled.q_lem_accepted.sum(),
        "agg_q_wm_accepted": settled.q_wm_accepted.sum(),
        "agg_imbalance": settled.imbalance.sum(),
        "agg_cost_lem": settled.cost_lem.sum(),
        "agg_cost_wm": settled.cost_wm.sum(),
        "agg_cost_imbalance": settled.cost_imbalance.sum(),
        "agg_cost_total": settled.cost_total.sum(),
        "agg_expected_cost_total": week.cost_total.sum(),
        "agg_periods_alpha_gt_0": int((week.alpha > 1e-9).sum()),
        "agg_periods_tie": int(week.tie.sum()),
        "agg_periods_p_lem_from_history": int((week.p_lem_source == "history").sum()),
    }
    for group, totals in groups.iterrows():
        key = group.replace(" ", "_")
        row[f"{key}_A"] = totals.A
        row[f"{key}_W"] = totals.W
        row[f"{key}_D"] = totals.D
        row[f"{key}_cost_total"] = totals.cost_total
    row["lem_balance_max_eur"] = balance
    row["aggregator_check_mwh"] = aggregator_check
    row["battery_dispatch_check_mw"] = battery_check
    summary = pd.DataFrame([row]).round(6)
    summary.to_csv(result_path / "summary.csv", index=False)
    print(summary.T.to_string(header=False))
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--beta", type=float, nargs="*", default=list(BETAS))
    parser.add_argument("--tie", nargs="*", default=list(TIE_RULES), choices=TIE_RULES)
    parser.add_argument("--s-hat", nargs="*", default=list(S_HAT_RULES),
                        choices=S_HAT_RULES)
    parser.add_argument("--no-prepare", action="store_true",
                        help="use the scenario folder as it is")
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single:  # child process: exactly one run
        logging.basicConfig(level=logging.WARNING)
        run_single(args.beta[0], args.tie[0], args.s_hat[0])
        return

    if not args.no_prepare:
        prepare_scenario()
    check_scenario()

    rows = []
    for s_hat in args.s_hat:
        for beta in args.beta:
            for tie in args.tie:
                print(f"--- run beta = {beta}, tie = {tie}, s_hat {s_hat}")
                subprocess.run(
                    [sys.executable, __file__, "--single", "--no-prepare",
                     "--beta", str(beta), "--tie", tie, "--s-hat", s_hat],
                    check=True,
                )
                rows.append(
                    pd.read_csv(RESULTS_PATH / run_id(beta, tie, s_hat) / "summary.csv")
                )
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(RESULTS_PATH / "wp15_summary.csv", index=False)
    print(table.T.to_string(header=False))


if __name__ == "__main__":
    main()
