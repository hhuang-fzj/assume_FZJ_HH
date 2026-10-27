"""
WP11 (revised): cost of one market participant in one delivery period.

Learning script of the course (round 2). It does not change ASSUME.

All participants are settled with the same rule as the aggregator (research design
of the thesis):
    1. LEM and WM clear in parallel (day-ahead, 15-minute products).
    2. All WM bids are accepted (bids at the WM price limits).
    3. Volume that is not accepted in the LEM is settled as imbalance at the reBAP.
       (The intraday auction is added later, WP16.)
    4. Dispatchable units (battery) only run with their accepted volume, so they
       have no imbalance: realised = LEM accepted + WM accepted.

Cost of one period (same formula as settle() of settlement.py, commit b22d8b5):
    C = p_LEM * A + p_WM * W + reBAP * D,   D = L - A - W
    L = realised net volume, A = accepted in the LEM, W = accepted in the WM.

Sign convention (same as round 1): buy +, sell -; cost + = pays, cost - = receives.

Sources:
    Formula: settlement.py, settle(), commit b22d8b5 (round 1, WP7).
    reBAP for buying and selling: netztransparenz.de (2023). reBAP-Berechnung,
        Stand 10.07.2023, Section 2.1, p. 8.
    Dispatch only with accepted volume: ASSUME assume/common/base.py,
        set_dispatch_plan(), lines 121-157.
    Parallel clearing, all WM bids accepted, unmatched volume to intraday and
        reBAP: research design of the thesis (decision of the user and supervisor).
"""

import math

# Length of one delivery period in hours (15-minute products, config.yaml)
DT_H = 0.25

# Real prices on 2016-06-21 (EUR/MWh), from course/wp5_inputs_2016-06-20.csv:
# WM price = SMARD day-ahead price, reBAP = netztransparenz rebap_2016_06.xls
WM_PRICE_1200, REBAP_1200 = 37.64, 44.71
WM_PRICE_1900, REBAP_1900 = 37.44, 95.56

# Teaching price of WP10: battery discharge price = average price / efficiency
# (ASSUME flexable_storage.py, line 135), average = 35 EUR/MWh, efficiency 0.95
BATTERY_DISCHARGE_PRICE = 35 / 0.95  # = 36.84


def participant_cost(
    realised_kw, lem_kw, wm_kw, price_lem, price_wm, rebap, dt_h=DT_H
):
    """
    Cost of one participant in one delivery period.

    Args:
        realised_kw (float): Realised net volume L in kW (buy +, sell -). For a
            dispatchable unit (battery) pass lem_kw + wm_kw.
        lem_kw (float): Volume A accepted in the LEM in kW.
        wm_kw (float): Volume W accepted in the WM in kW (all WM bids are accepted).
        price_lem (float): LEM clearing price in EUR/MWh.
        price_wm (float): WM clearing price in EUR/MWh.
        rebap (float): Imbalance price in EUR/MWh.
        dt_h (float): Length of the delivery period in hours.

    Returns:
        float: Cost in EUR. Positive = the participant pays, negative = receives.
    """
    # Imbalance: what was realised minus what was traded on both markets
    imbalance_kw = realised_kw - lem_kw - wm_kw

    cost_lem = price_lem * lem_kw
    cost_wm = price_wm * wm_kw
    cost_imbalance = rebap * imbalance_kw

    # kW -> MW (/ 1000), power -> energy (* dt_h)
    return (cost_lem + cost_wm + cost_imbalance) * dt_h / 1000


# Test cases (name, L, A, W, p_LEM, p_WM, reBAP, expected cost in EUR)
TESTS = [
    # Same numbers as WP1 of round 1 (aggregator, MWh / 0.25 h -> kW):
    # L = -0.022 MWh, A = -0.006, W = -0.012; expected C = -1.1346 EUR
    ("T1 aggregator, WP1 example", -88, -24, -48, 123.1, 28, 15, -1.1346),
    # Independent nodes at 19:00, beta = 1: no seller in the LEM, all to reBAP
    ("T2 independent nodes 19:00, beta = 1", 40.641, 0, 0, 0,
     WM_PRICE_1900, REBAP_1900, 0.970913),
    # Same with beta = 0.5: half in the WM (accepted), half in the LEM (not)
    ("T3 independent nodes 19:00, beta = 0.5", 40.641, 0, 20.3205, 0,
     WM_PRICE_1900, REBAP_1900, 0.675657),
    # Community PV at 12:00, beta = 1, WP10 task 4b: 53.733 kW accepted at 37.64
    ("T4 community PV 12:00, beta = 1", -76.758, -53.733, 0, 37.64,
     WM_PRICE_1200, REBAP_1200, -0.762989),
    # Battery discharge, WP10 case B2: accepted 10 kW, realised = A + W
    ("T5 battery discharge, accepted", -10, -10, 0, BATTERY_DISCHARGE_PRICE,
     45, 45, -0.092105),
    # Battery not accepted: it does not run, realised = 0, no imbalance
    ("T6 battery, not accepted", 0, 0, 0, 0, 45, 45, 0.0),
]


def check():
    """Run all test cases and the money balance; print OK / FAIL."""
    for name, realised, lem, wm, p_lem, p_wm, rebap, expected in TESTS:
        cost = participant_cost(realised, lem, wm, p_lem, p_wm, rebap)
        status = "OK" if math.isclose(cost, expected, abs_tol=1e-6) else "FAIL"
        print(f"{status:4s} {name}: {cost:.6f} EUR (expected {expected:.6f})")

    # Money balance: in one period the LEM payments of all participants sum to
    # zero, because with one uniform price the buyers pay what the sellers receive.
    # Case A variant b of WP10: accepted volumes in kW at price 30 EUR/MWh.
    accepted_case_a = {
        "independent nodes": 40,
        "battery charge": 5,
        "community PV": -50 * 45 / 70,
        "aggregator": -20 * 45 / 70,
    }
    balance = sum(30 * x for x in accepted_case_a.values()) * DT_H / 1000
    status = "OK" if math.isclose(balance, 0.0, abs_tol=1e-9) else "FAIL"
    print(f"{status:4s} money balance case A variant b: {balance:.2e} EUR")


if __name__ == "__main__":
    check()
