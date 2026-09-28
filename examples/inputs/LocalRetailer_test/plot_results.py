# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Figures of the local retailer study.

The script reads the result files that the simulation and the settlement have written
and draws the figures of the thesis. It does not run any simulation, so the layout of a
figure can be changed without repeating the runs.

Figures:
    1. ``fig_alpha_volume.png``: procured and uncovered volume over alpha.
    2. ``fig_alpha_cost.png``: cost components over alpha.
    3. ``fig_timeseries.png``: bid, accepted and uncovered volume of one day, for the
       run with ``EXAMPLE_ALPHA``.

Note:
    The prices of the scenario are placeholders, so the cost figure shows the mechanism
    and not final results.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files instead of opening a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIO_PATH = Path(__file__).parent

SWEEP_FILE = SCENARIO_PATH / "alpha_sweep.csv"
# Which run the time series figure shows. The file is written by alpha_sweep.py, so the
# figure always matches the sweep results.
EXAMPLE_ALPHA = 0.50
SETTLEMENT_FILE = SCENARIO_PATH / f"settlement_alpha_{EXAMPLE_ALPHA:.2f}.csv"

FIGURE_VOLUME = SCENARIO_PATH / "fig_alpha_volume.png"
FIGURE_COST = SCENARIO_PATH / "fig_alpha_cost.png"
FIGURE_TIMESERIES = SCENARIO_PATH / "fig_timeseries.png"

# Day shown in the time series figure
EXAMPLE_DAY = None

FIGURE_SIZE = (7.0, 4.5)
DPI = 200
ALPHA_LABEL = r"$\alpha$ (share of the residual load offered on the LEM)"


def load_sweep(path: Path) -> pd.DataFrame:
    """
    Read the result table of the alpha sweep.

    Args:
        path (Path): Path to ``alpha_sweep.csv``.

    Returns:
        pd.DataFrame: One row per alpha value, sorted by alpha.

    Raises:
        FileNotFoundError: If the file does not exist. Run ``alpha_sweep.py`` first.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found, run alpha_sweep.py first")
    return pd.read_csv(path).sort_values("alpha")


def plot_volumes(sweep: pd.DataFrame, figure_file: Path) -> None:
    """
    Draw procured and uncovered volume over alpha.

    Args:
        sweep (pd.DataFrame): Result table of the sweep.
        figure_file (Path): Path of the figure to write.

    Note:
        The LEM curve and the WM curve cross where both markets procure the same volume.
        The uncovered volume grows with alpha because the local supply is limited.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(sweep.alpha, sweep.q_lem_accepted, marker="o", label="procured on the LEM")
    ax.plot(sweep.alpha, sweep.q_wm_accepted, marker="s", label="procured on the WM")
    ax.plot(
        sweep.alpha,
        sweep.uncovered,
        marker="^",
        linestyle="--",
        label="uncovered LEM volume",
    )
    ax.set_xlabel(ALPHA_LABEL)
    ax.set_ylabel("energy (MWh)")
    ax.set_title("Procured and uncovered volume")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_file, dpi=DPI)
    plt.close(fig)
    logger.info(f"wrote {figure_file.name}")


def plot_costs(sweep: pd.DataFrame, figure_file: Path) -> None:
    """
    Draw the cost components over alpha as stacked bars with the total as a line.

    Args:
        sweep (pd.DataFrame): Result table of the sweep.
        figure_file (Path): Path of the figure to write.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    width = (sweep.alpha.diff().dropna().min() or 0.2) * 0.6

    bottom = sweep.cost_lem * 0
    for column, label in [
        ("cost_lem", "LEM procurement"),
        ("cost_wm", "WM procurement"),
        ("cost_imbalance", "imbalance"),
    ]:
        ax.bar(sweep.alpha, sweep[column], bottom=bottom, width=width, label=label)
        bottom = bottom + sweep[column]

    ax.plot(
        sweep.alpha,
        sweep.cost_total,
        marker="D",
        color="black",
        linewidth=1.5,
        label="total cost",
    )
    ax.set_xlabel(ALPHA_LABEL)
    ax.set_ylabel("cost (EUR)")
    ax.set_title("Cost components of the aggregator")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figure_file, dpi=DPI)
    plt.close(fig)
    logger.info(f"wrote {figure_file.name}")


def plot_timeseries(path: Path, day: str, figure_file: Path) -> None:
    """
    Draw bid, accepted and uncovered LEM volume of one day.

    Args:
        path (Path): Path to ``settlement.csv`` of a single run.
        day (str): Day to show, e.g. ``"2016-01-03"``.
        figure_file (Path): Path of the figure to write.

    Note:
        This figure shows why the uncovered volume arises: at night the local supply is
        zero, so the whole LEM bid stays unexecuted.
    """
    if not path.exists():
        logger.warning(f"{path.name} not found, skipping the time series figure")
        return

    table = pd.read_csv(path, parse_dates=[0], index_col=0)
    if day is None:
        day = str(table.index.normalize().unique()[1].date())
    table = table.loc[day]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    hours = table.index.hour
    ax.plot(hours, table.q_lem_bid, marker="o", label="LEM bid")
    ax.plot(hours, table.q_lem_accepted, marker="s", label="LEM accepted")
    ax.fill_between(
        hours,
        table.q_lem_accepted,
        table.q_lem_bid,
        alpha=0.25,
        label="uncovered",
    )
    ax.set_xlabel(f"hour of {day}")
    ax.set_ylabel("energy (MWh)")
    ax.set_title(f"LEM bid and execution over one day (alpha = {EXAMPLE_ALPHA})")
    ax.set_xticks(range(0, 24, 3))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_file, dpi=DPI)
    plt.close(fig)
    logger.info(f"wrote {figure_file.name}")


def main() -> None:
    """Draw all figures from the result files."""
    sweep = load_sweep(SWEEP_FILE)
    plot_volumes(sweep, FIGURE_VOLUME)
    plot_costs(sweep, FIGURE_COST)
    plot_timeseries(SETTLEMENT_FILE, EXAMPLE_DAY, FIGURE_TIMESERIES)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
