# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Entry point of the local retailer study: one command runs the whole chain.

Stages:
    1. **Scenario** (:mod:`simbench_to_assume`): build the ASSUME scenario from the
       SimBench low-voltage grid.
    2. **Sweep** (:mod:`alpha_sweep`): run one simulation per alpha value and settle
       each run.
    3. **Figures** (:mod:`plot_results`): draw the figures from the result files.

Each stage can also be run on its own; this script only calls them in order. Stages can
be skipped with command line flags, for example to redraw the figures without repeating
the simulations.

Usage:
    python run_study.py                 # all stages
    python run_study.py --skip-scenario # keep the current scenario files
    python run_study.py --only-figures  # only redraw the figures
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import alpha_sweep  # noqa: E402
import plot_results  # noqa: E402
import simbench_to_assume  # noqa: E402

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Read the command line flags that select the stages to run.

    Returns:
        argparse.Namespace: Flags ``skip_scenario``, ``skip_sweep`` and ``only_figures``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-scenario",
        action="store_true",
        help="keep the current scenario files instead of rebuilding them",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="keep the current simulation results instead of running the sweep",
    )
    parser.add_argument(
        "--only-figures",
        action="store_true",
        help="only redraw the figures from the existing result files",
    )
    return parser.parse_args()


def run_stage(name: str, function) -> None:
    """
    Run one stage and log how long it took.

    Args:
        name (str): Name of the stage, used in the log messages.
        function: Callable that runs the stage, taking no arguments.
    """
    logger.info(f"=== {name} ===")
    started = time.perf_counter()
    function()
    logger.info(f"=== {name} finished in {time.perf_counter() - started:.1f} s ===\n")


def main() -> None:
    """Run the selected stages in order."""
    arguments = parse_arguments()

    skip_scenario = arguments.skip_scenario or arguments.only_figures
    skip_sweep = arguments.skip_sweep or arguments.only_figures

    if not skip_scenario:
        run_stage("1/3 scenario from SimBench", simbench_to_assume.main)
    if not skip_sweep:
        run_stage("2/3 alpha sweep and settlement", alpha_sweep.main)
    run_stage("3/3 figures", plot_results.main)

    logger.info("study finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
