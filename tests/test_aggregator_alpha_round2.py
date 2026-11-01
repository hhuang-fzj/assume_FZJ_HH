# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests of the round-2 aggregator (assume/strategies/aggregator_alpha_round2.py).

The cases are the hand calculation of WP14, section 5 (course document): one period
in which the aggregator sells, L_hat = -0.0051243 MWh, p_wm_hat = 25.24 EUR/MWh,
r_hat = p_wm_hat - 1 = 24.24 EUR/MWh. The test runs the same chain as the strategy:
solve_alpha() (MILP of round 1, alpha_model.py) -> apply_tie_break() (F5).
"""

import pandas as pd
import pytest

from assume.strategies.aggregator_alpha_round2 import apply_tie_break, naive_lag
from assume.strategies.alpha_model import solve_alpha

L_HAT = -0.0051243
P_WM = 25.24
R_HAT = 24.24


def decide(p_lem, S_hat, rule):
    inputs = pd.DataFrame(
        {"L_hat": [L_HAT], "p_lem": [p_lem], "p_wm": [P_WM], "r_hat": [R_HAT],
         "S_hat": [S_hat]},
        index=pd.DatetimeIndex(["2016-06-21 11:00"]),
    )
    return apply_tie_break(solve_alpha(inputs), rule).iloc[0]


@pytest.mark.parametrize("rule, alpha", [("lem", 1.0), ("wm", 0.0)])
def test_flat_cost_is_a_tie_and_the_rule_decides(rule, alpha):
    # p_lem_hat = p_wm_hat: every alpha costs the same (WP14 table, row 1)
    row = decide(25.24, abs(L_HAT), rule)
    assert row.tie
    assert row.alpha == pytest.approx(alpha)
    assert row.cost_total == pytest.approx(-0.12934, abs=1e-5)


@pytest.mark.parametrize("rule", ["lem", "wm"])
def test_lem_more_attractive_gives_alpha_one(rule):
    row = decide(30.0, abs(L_HAT), rule)  # WP14 table, row 2
    assert not row.tie
    assert row.alpha == pytest.approx(1.0)
    assert row.cost_total == pytest.approx(-0.15373, abs=1e-5)


@pytest.mark.parametrize("rule", ["lem", "wm"])
def test_lem_less_attractive_gives_alpha_zero(rule):
    row = decide(22.0, abs(L_HAT), rule)  # WP14 table, row 3
    assert not row.tie
    assert row.alpha == pytest.approx(0.0)
    assert row.cost_total == pytest.approx(-0.12934, abs=1e-5)


def test_limited_lem_volume_gives_the_kink():
    row = decide(30.0, 0.002, "lem")  # WP14 table, last row
    assert row.alpha == pytest.approx(0.002 / abs(L_HAT), abs=1e-6)  # 0.39030
    assert row.cost_total == pytest.approx(-0.13885, abs=1e-5)


def test_naive_lag():
    # Weron (2014): Monday, Saturday, Sunday -> one week; Tuesday-Friday -> one day
    assert naive_lag(pd.Timestamp("2016-06-20")).days == 7  # Monday
    assert naive_lag(pd.Timestamp("2016-06-21")).days == 1  # Tuesday
    assert naive_lag(pd.Timestamp("2016-06-25")).days == 7  # Saturday
