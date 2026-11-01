# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Round 2, decision F4: LEM price limits -500 / 3000 instead of the corridor 123.1 / 288.

Run after simbench_to_assume.py (which writes config.yaml with the round-1 corridor):
    python examples/inputs/LocalRetailer_test/round2_lem_limits.py

What changes:
    config.yaml · markets_config · LEM_DA · minimum_bid_price   123.1 -> -500
    config.yaml · markets_config · LEM_DA · maximum_bid_price   288   -> 3000

Where it goes in ASSUME (branch Yihan_local_aggregator, checked 2026-10-26):
    loader_csv.py setup_world() 986-1003 -> MarketConfig of LEM_DA
    -> base_market.py validate_orderbook(), lines 374-383: bids outside the limits are
       set to the limit. With the corridor, every naive-forecast bid of round 2
       (about 20-40 EUR/MWh) was set to 123.1 (model document, section 7.3).

Why: with the corridor the LEM price cannot reflect the bids of round 2 (F3: the
aggregator bids its opportunity cost; R6: all participants do). The WM uses the same
limits, so a bid at the WM limit is always accepted in both markets.

Sources:
    WM limits -500 / 3000: same as WM_DA in config.yaml (round 1); -500 as price floor
        for sell bids: Qussous, R., Harder, N., Weidlich, A. (2022). Understanding
        Power Market Dynamics by Reflecting Market Interrelations and
        Flexibility-Oriented Bidding Strategies. Energies 15(2), 494, Sect. 3.2.1, p. 9.
    Same limits in the LEM: decision F4 [OWN].
"""

from pathlib import Path

import yaml

SCENARIO_PATH = Path(__file__).resolve().parent
CONFIG_FILE = SCENARIO_PATH / "config.yaml"
STUDY_CASE = "local_retailer_demo"
LEM_MARKET_ID = "LEM_DA"
LEM_MIN_PRICE_EUR_MWH = -500.0
LEM_MAX_PRICE_EUR_MWH = 3000.0


def set_lem_limits(
    path: Path = CONFIG_FILE,
    min_price: float = LEM_MIN_PRICE_EUR_MWH,
    max_price: float = LEM_MAX_PRICE_EUR_MWH,
) -> dict:
    """
    Write the LEM price limits into config.yaml.

    Returns:
        dict: The LEM market configuration after the change.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    lem = config[STUDY_CASE]["markets_config"][LEM_MARKET_ID]
    lem["minimum_bid_price"] = float(min_price)
    lem["maximum_bid_price"] = float(max_price)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    return lem


def main() -> None:
    lem = set_lem_limits()
    print(
        f"{CONFIG_FILE.name}: LEM_DA limits {lem['minimum_bid_price']} / "
        f"{lem['maximum_bid_price']} EUR/MWh"
    )


if __name__ == "__main__":
    main()
