"""
WP14: inputs of M1 in round 2 and the resulting alpha (learning script).

It does not change ASSUME. It reproduces the hand calculation of WP14 in the course
document. M1 itself is the model of round 1 (WP1-WP4); here it is evaluated in
closed form, because the expected cost is piecewise linear in alpha with one kink
at alpha = S_hat / |L_hat|: the optimum lies at 0, at the kink or at 1 (WP1).

Changes of the inputs in round 2 (decisions F1-F5 of WP14):
    F1  p_wm_hat: naive forecast instead of the realised price.
    F2  p_lem_hat: naive forecast of the LEM price; p_wm_hat if no LEM history.
    F3  LEM bid price of the aggregator = p_wm_hat (buy and sell).
    F4  LEM price limits -500 / 3000 in config.yaml (otherwise ASSUME clips bids).
    F5  If several alpha give the same expected cost: alpha = 0 (all to the WM,
        which accepts every bid) [OWN].

Sign convention (same as round 1): buy +, sell -; cost + = pays.

Sources:
    M1, r_hat = p_wm_hat -/+ 1: round 1, WP1 and WP5 (course document);
        aggregator_alpha_optimized.py (R_HAT_PENALTY = 1.0).
    Naive forecast: Weron, R. (2014). Electricity price forecasting: A review of
        the state-of-the-art with a look into the future. International Journal of
        Forecasting 30(4), 1030-1081, Sect. 3.8.1, p. 1049-1050. Applied to the
        LEM price series: [OWN].
    Bid at the opportunity cost: Mello, J., de Lorenzo, C., Campos, F. A., Villar, J.
        (2023). Pricing and Simulating Energy Transactions in Energy Communities.
        Energies 16(4), 1949, Eq. (1), p. 4 and Sect. 3.1, p. 8. WM price as the
        opportunity cost of the aggregator: [OWN].
    Price clipping: ASSUME assume/markets/base_market.py, lines 374-383.
"""

import math

import pandas as pd

from wp13_battery_bid import naive_forecast

R_HAT_PENALTY = 1.0  # EUR/MWh, as in round 1


def expected_cost(alpha, L_hat, p_lem, p_wm, r_hat, S_hat):
    """
    Expected cost C_hat(alpha) of M1 for one period (EUR).

    a_hat = sign(L_hat) * min(alpha * |L_hat|, S_hat): expected LEM acceptance.
    Volumes in MWh per period, prices in EUR/MWh.
    """
    sign = 1.0 if L_hat > 0 else -1.0
    a_hat = sign * min(alpha * abs(L_hat), S_hat)
    return p_lem * a_hat + p_wm * (1 - alpha) * L_hat + r_hat * (alpha * L_hat - a_hat)


def best_alpha(L_hat, p_lem, p_wm, r_hat, S_hat):
    """
    Optimal alpha: the cost is linear between 0, the kink and 1, so only these
    three candidates need to be compared. Ties are broken towards alpha = 0 (F5).

    Returns:
        tuple: (alpha, expected cost)
    """
    kink = min(1.0, S_hat / abs(L_hat))
    candidates = [0.0, kink, 1.0]  # sorted ascending, so the first minimum is the smallest alpha
    costs = [expected_cost(a, L_hat, p_lem, p_wm, r_hat, S_hat) for a in candidates]
    lowest = min(costs)
    for alpha, cost in zip(candidates, costs):
        if math.isclose(cost, lowest, abs_tol=1e-9):
            return alpha, cost


def r_hat_for(L_hat, p_wm):
    """Expected imbalance price: selling -> p_wm - 1, buying -> p_wm + 1 (round 1)."""
    return p_wm - R_HAT_PENALTY if L_hat < 0 else p_wm + R_HAT_PENALTY


def ok(name, value, expected, tol=1e-5):
    status = "OK" if math.isclose(value, expected, abs_tol=tol) else "FAIL"
    print(f"{status:4s} {name}: {value:.5f} (expected {expected:.5f})")


def check():
    """Reproduce the table of WP14, section 5."""
    from wp5_inputs import read_wm_price_15min  # SMARD prices of round 1

    index = pd.date_range("2016-06-13", "2016-06-22", freq="15min", inclusive="left")
    p_wm_hat = naive_forecast(read_wm_price_15min(index), "2016-06-21")["2016-06-21 12:00"]
    ok("p_wm_hat 12:00 (naive)", p_wm_hat, 25.24)

    L_hat = -0.0051243  # MWh, course/wp5_inputs_2016-06-20.csv, 2016-06-21 12:00
    r_hat = r_hat_for(L_hat, p_wm_hat)
    ok("r_hat", r_hat, 24.24)

    cases = [  # (S_hat, p_lem_hat, expected alpha, expected cost)
        (abs(L_hat), 25.24, 0.0, -0.12934),  # flat -> tie -> alpha = 0 (F5)
        (abs(L_hat), 30.00, 1.0, -0.15373),
        (abs(L_hat), 22.00, 0.0, -0.12934),
        (0.002, 30.00, 0.002 / abs(L_hat), -0.13885),
    ]
    for S_hat, p_lem, alpha_exp, cost_exp in cases:
        alpha, cost = best_alpha(L_hat, p_lem, p_wm_hat, r_hat, S_hat)
        ok(f"S={S_hat:.5f}, p_lem={p_lem}: alpha", alpha, alpha_exp)
        ok(f"S={S_hat:.5f}, p_lem={p_lem}: cost ", cost, cost_exp)


if __name__ == "__main__":
    check()
