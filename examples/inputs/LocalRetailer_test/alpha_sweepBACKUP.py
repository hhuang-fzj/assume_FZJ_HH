# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Sweep of the allocation parameter alpha for the local retailer.

The allocation parameter alpha is the share of the forecast portfolio residual load that
the aggregator offers on the local energy market (LEM); the remaining share goes to the
wholesale market (WM). This script runs one simulation per alpha value, settles each run
and collects the results in one table and one figure.

Pipeline:
    1. **Run** one simulation per alpha value.
    2. **Settle** each run with the functions of :mod:`settlement`.
    3. **Collect** the totals of all runs in one table.
    4. **Plot** procured volumes, uncovered volume and costs over alpha.

Note:
    The prices used in the scenario are placeholders. The shape of the curves shows the
    mechanism; the absolute values change once the prices are replaced by referenced
    data.
"""

import logging
import shutil
import sys
from pathlib import Path

import matplotlib
import pandas as pd

# The repository root has to be importable, so that "assume" resolves to this fork
sys.path.insert(0, str(Path(__file__).parents[3]))

matplotlib.use("Agg")  # write files, no interactive window
import matplotlib.pyplot as plt  # noqa: E402
from assume.scenario.loader_csv import load_scenario_folder  # noqa: E402

from assume import World  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import settlement  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIO_PATH = Path(__file__).parent
INPUTS_PATH = SCENARIO_PATH.parent
OUTPUT_ROOT = SCENARIO_PATH.parent.parent / "outputs"
SCENARIO = SCENARIO_PATH.name
STUDY_CASE = "local_retailer_demo"

AGGREGATOR_OPERATOR = "local_retailer"
LEM_MARKET_ID = "LEM_DA"

ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

RESULT_FILE = SCENARIO_PATH / "alpha_sweep.csv"
FIGURE_FILE = SCENARIO_PATH / "alpha_sweep.png"


def run_simulation(alpha: float) -> Path:
    """
    Run one simulation with a given allocation parameter.

    Args:
        alpha (float): Share of the portfolio residual load offered on the LEM.

    Returns:
        Path: Folder with the CSV output of this run.

    Note:
        Alpha is set on the strategy object after the scenario is loaded, so the CSV
        inputs stay untouched and all runs use exactly the same data.
    """
    run_id = f"alpha_{alpha:.2f}".replace(".", "_")
    export_path = OUTPUT_ROOT / run_id
    shutil.rmtree(export_path, ignore_errors=True)

    world = World(database_uri=None, export_csv_path=str(export_path))
    load_scenario_folder(
        world, inputs_path=str(INPUTS_PATH), scenario=SCENARIO, study_case=STUDY_CASE
    )

    # The same strategy object serves both markets, so setting it once is enough
    strategy = world.unit_operators[AGGREGATOR_OPERATOR].portfolio_strategies[
        LEM_MARKET_ID
    ]
    strategy.alpha = alpha

    world.run()
    return export_path / f"{SCENARIO}_{STUDY_CASE}"


def settle_run(output_path: Path, alpha: float) -> dict[str, float]:
    """
    Settle one simulation run and return its totals.

    Args:
        output_path (Path): Folder with the CSV output of the run.
        alpha (float): Allocation parameter of this run, kept as a column.

    Returns:
        dict[str, float]: Totals of one run: procured volumes, uncovered volume,
        imbalance and the three cost components.
    """
    orders = settlement.read_portfolio_orders(output_path)
    demand_units, building_units = settlement.read_portfolio_units(SCENARIO_PATH)
    residual_load = settlement.build_realised_residual_load(
        SCENARIO_PATH, demand_units, building_units
    )
    table = settlement.settle(orders, residual_load)

    return {
        "alpha": alpha,
        "q_lem_accepted": table.q_lem_accepted.sum(),
        "q_wm_accepted": table.q_wm_accepted.sum(),
        "uncovered": table.uncovered.sum(),
        "imbalance": table.imbalance.sum(),
        "cost_lem": table.cost_lem.sum(),
        "cost_wm": table.cost_wm.sum(),
        "cost_imbalance": table.cost_imbalance.sum(),
        "cost_total": table.cost_total.sum(),
    }


def plot_sweep(results: pd.DataFrame, figure_file: Path) -> None:
    """
    Plot the sweep results over alpha.

    Two panels are drawn: the procured volumes with the uncovered volume, and the cost
    components with the total cost.

    Args:
        results (pd.DataFrame): Output of the sweep, one row per alpha value.
        figure_file (Path): Path of the figure to write.
    """
    fig, (ax_volume, ax_cost) = plt.subplots(1, 2, figsize=(12, 4.5))

    # 1. Volumes
    ax_volume.plot(
        results.alpha, results.q_lem_accepted, marker="o", label="LEM procured"
    )
    ax_volume.plot(
        results.alpha, results.q_wm_accepted, marker="s", label="WM procured"
    )
    ax_volume.plot(
        results.alpha, results.uncovered, marker="^", label="uncovered LEM volume"
    )
    ax_volume.set_xlabel(r"$\alpha$ (share offered on the LEM)")
    ax_volume.set_ylabel("energy (MWh)")
    ax_volume.set_title("Procured and uncovered volume")
    ax_volume.legend()
    ax_volume.grid(alpha=0.3)

    # 2. Costs
    ax_cost.plot(results.alpha, results.cost_lem, marker="o", label="LEM cost")
    ax_cost.plot(results.alpha, results.cost_wm, marker="s", label="WM cost")
    ax_cost.plot(
        results.alpha, results.cost_imbalance, marker="^", label="imbalance cost"
    )
    ax_cost.plot(
        results.alpha, results.cost_total, marker="D", linewidth=2, label="total cost"
    )
    ax_cost.set_xlabel(r"$\alpha$ (share offered on the LEM)")
    ax_cost.set_ylabel("cost (EUR)")
    ax_cost.set_title("Cost components")
    ax_cost.legend()
    ax_cost.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figure_file, dpi=150)
    logger.info(f"Figure written to {figure_file.name}")


def main() -> None:
    """Run the sweep over all alpha values, write the result table and the figure."""
    rows = []
    for alpha in ALPHA_VALUES:
        logger.info(f"--- running alpha = {alpha} ---")
        output_path = run_simulation(alpha)
        row = settle_run(output_path, alpha)
        rows.append(row)
        logger.info(
            f"alpha={alpha}: uncovered {row['uncovered']:.4f} MWh, "
            f"total cost {row['cost_total']:.2f} EUR"
        )

    results = pd.DataFrame(rows)
    results.round(6).to_csv(RESULT_FILE, index=False)
    logger.info(f"Results written to {RESULT_FILE.name}")

    plot_sweep(results, FIGURE_FILE)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
