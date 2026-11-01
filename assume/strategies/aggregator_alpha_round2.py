# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Local retailer strategy of round 2: M1 with the changed inputs F1-F5 (WP14).

Subclass of AlphaOptimizedLocalRetailerStrategy (round 1, unchanged). The model M1
(alpha_model.py, solve_alpha) is the same; only its inputs, the LEM bid price and the
choice between equally good alphas change. Every change is a switch, so round 1 can
still be reproduced with the parent class.

    F1  p_wm_hat = naive similar-day forecast of the WM price instead of the realised
        price. Read from unit.forecaster.price["LEM_DA"]: the column price_LEM_DA of
        forecasts_df.csv holds the naive WM forecast (written by
        round2_add_batteries.py, naive_forecast()); all participants use it (rule R7).
    F2  p_lem_hat = naive similar-day forecast of the LEM clearing price, built from the
        LEM prices the aggregator has seen in its own clearing results; p_wm_hat when
        that price is not known.
    F3  LEM bid price of the aggregator = p_wm_hat (buy and sell), its opportunity cost
        (rule R6).
    F4  LEM price limits -500 / 3000. Not in this file: config.yaml of the scenario
        (round2_lem_limits.py). The class constants LEM_SELL/BUY_PRICE of round 1 are
        no longer used when F3 is on.
    F5  Choice between alphas with the same expected cost (tie):
            "lem": the largest such alpha (the aggregator enters the LEM when it is
                   indifferent, so it observes the LEM price)  [OWN]
            "wm":  the smallest such alpha (alpha = 0 side)          [OWN]
        Why it matters: without LEM history F2 gives p_lem_hat = p_wm_hat, the cost is
        flat in alpha and every alpha is a tie. With "wm" the aggregator then never
        bids in the LEM and never learns the LEM price (model document, F2/F5).

Data path:
    p_wm_hat: SMARD 2016 -> round2_add_batteries.py naive_forecast() -> forecasts_df.csv
        column price_LEM_DA -> loader_csv.py 651 -> forecaster.py line 71
        -> unit.forecaster.price["LEM_DA"] -> _wm_price_forecast() below.
    p_lem_hat: LEM clearing message (base_market.py 682-711; also rejected orders
        carry the clearing price, set in clearing_algorithms/simple.py lines 171-173;
        base_market.py 664-669 only fills it in when it is missing) -> units_operator.py
        handle_market_feedback() 273-287 -> handle_market_feedback() below
        -> self.lem_price_history -> _lem_price_forecast() below.
    r_hat = p_wm_hat -/+ R_HAT_PENALTY_EUR_MWH and S_hat: unchanged from round 1.

Lag of the naive forecast (same rule as round2_add_batteries.py): delivery on Monday,
Saturday or Sunday -> same quarter hour one week earlier; Tuesday to Friday -> one day
earlier. The LEM result of day d-1 is known at the decision for day d (the LEM of day
d-1 cleared one day earlier).

A price of exactly 0 in a rejected order is not stored: ASSUME sets the clearing
price to 0 when nothing was accepted (simple.py, lines 162-163), so 0 then means "no
trade", not a price [OWN].

Sources:
    Naive forecast: Weron, R. (2014). Electricity price forecasting: A review of the
        state-of-the-art with a look into the future. International Journal of
        Forecasting 30(4), 1030-1081, Sect. 3.8.1, p. 1049-1050. Applied to the LEM
        price series: [OWN].
    Bid at the opportunity cost: Mello, J., de Lorenzo, C., Campos, F. A., Villar, J.
        (2023). Pricing and Simulating Energy Transactions in Energy Communities.
        Energies 16(4), 1949, Sect. 3.1, p. 8. WM price as opportunity cost of the
        aggregator: [OWN].
    Only own clearing results are known: Ottesen, S. O., Tomasgard, A., Fleten, S.-E.
        (2016). Prosumer bidding and scheduling in electricity markets. Energy 94,
        828-843, p. 830.
    Expected cost of M1 and its candidates 0, S_hat/|L_hat|, 1: model document,
        section 3.4; Course/wp14_alpha_inputs.py.
"""

import logging
from datetime import timedelta

import pandas as pd

from assume.common.local_retailer_state import ForecastSnapshot
from assume.common.market_objects import Orderbook, Product
from assume.strategies.aggregator_alpha_optimized import (
    LEM_MARKET_ID,
    AlphaOptimizedLocalRetailerStrategy,
)
from assume.strategies.alpha_model import solve_alpha

logger = logging.getLogger(__name__)

PORTFOLIO_UNIT_ID = "__portfolio__"
# Market whose forecast column holds the naive WM forecast (price_LEM_DA)
NAIVE_FORECAST_MARKET_ID = "LEM_DA"
# |L_hat| below this is not optimised (same threshold as alpha_model.py, lines 43-49)
ZERO_VOLUME_MWH = 1e-9
# Two expected costs closer than this are a tie (EUR)
TIE_TOLERANCE_EUR = 1e-9
# MILP cost worse than the exact optimum by more than this -> warning (EUR)
MILP_CHECK_TOLERANCE_EUR = 1e-6


def naive_lag(t: pd.Timestamp) -> timedelta:
    """Lag of the naive similar-day forecast (Weron 2014, p. 1049-1050)."""
    return timedelta(days=7) if t.dayofweek in (0, 5, 6) else timedelta(days=1)


def expected_cost(alpha, L_hat, p_lem, p_wm, r_hat, S_hat) -> dict:
    """
    Expected cost of M1 for one period, split into its three terms (EUR).

    Same formula as alpha_model.py (model document, section 3.2):
        a_hat = sign(L_hat) * min(alpha * |L_hat|, S_hat)
        C_hat = p_lem * a_hat + p_wm * (1 - alpha) * L_hat + r_hat * (alpha * L_hat - a_hat)
    """
    sign = 1.0 if L_hat > 0 else -1.0
    a_hat = sign * min(alpha * abs(L_hat), S_hat)
    cost_lem = p_lem * a_hat
    cost_wm = p_wm * (1 - alpha) * L_hat
    cost_rebap = r_hat * (alpha * L_hat - a_hat)
    return {
        "alpha": alpha,
        "a_hat": a_hat,
        "cost_lem": cost_lem,
        "cost_wm": cost_wm,
        "cost_rebap": cost_rebap,
        "cost_total": cost_lem + cost_wm + cost_rebap,
    }


def apply_tie_break(decision: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Choose between alphas with the same expected cost (F5).

    The expected cost is linear in alpha between 0, the kink S_hat/|L_hat| and 1
    (model document, section 3.4), so these three candidates contain every optimum.
    All candidates with the lowest cost (within TIE_TOLERANCE_EUR) are ties; "lem"
    takes the largest, "wm" the smallest. If the MILP result is worse than this exact
    optimum, a warning is logged and the exact optimum is used.

    Args:
        decision (pd.DataFrame): Output of solve_alpha() (inputs + six result columns).
        rule (str): "lem" or "wm".

    Returns:
        pd.DataFrame: Copy of ``decision`` with the chosen alpha and its costs, and the
        column ``tie`` (True if more than one candidate had the lowest cost).
    """
    if rule not in ("lem", "wm"):
        raise ValueError(f"unknown tie-break rule {rule!r}, use 'lem' or 'wm'")
    result = decision.copy()
    result["tie"] = False
    for t, row in decision.iterrows():
        if abs(row.L_hat) < ZERO_VOLUME_MWH:
            continue  # not optimised, alpha = 0 (alpha_model.py)
        kink = min(1.0, row.S_hat / abs(row.L_hat))
        candidates = [
            expected_cost(alpha, row.L_hat, row.p_lem, row.p_wm, row.r_hat, row.S_hat)
            for alpha in sorted({0.0, kink, 1.0})
        ]
        lowest = min(c["cost_total"] for c in candidates)
        if row.cost_total > lowest + MILP_CHECK_TOLERANCE_EUR:
            logger.warning(
                "MILP cost %.9f above exact optimum %.9f at %s", row.cost_total, lowest, t
            )
        ties = [c for c in candidates if c["cost_total"] <= lowest + TIE_TOLERANCE_EUR]
        chosen = ties[-1] if rule == "lem" else ties[0]  # candidates sorted by alpha
        for column, value in chosen.items():
            result.at[t, column] = value
        result.at[t, "tie"] = len(ties) > 1
    return result


class AlphaRound2LocalRetailerStrategy(AlphaOptimizedLocalRetailerStrategy):
    """M1 of round 1 with the inputs F1-F5 of round 2 (switches below)."""

    # --- settings (see module docstring for the sources) --------------------------
    F1_NAIVE_WM_FORECAST = True
    F2_LEM_PRICE_FROM_HISTORY = True
    F3_LEM_BID_AT_WM_FORECAST = True
    F5_TIE_BREAK = "lem"  # "lem" or "wm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LEM clearing price per delivery start, from own clearing results (F2)
        self.lem_price_history: dict = {}

    # ------------------------------------------------------------------------------
    # F1: WM price forecast
    # ------------------------------------------------------------------------------
    def _wm_price_forecast(self, index: pd.DatetimeIndex) -> pd.Series:
        """p_wm_hat (EUR/MWh): naive forecast (F1) or the realised price (round 1)."""
        if not self.F1_NAIVE_WM_FORECAST:
            return super()._wm_price_forecast(index)
        for unit in self._units_operator.units.values():
            price = getattr(unit.forecaster, "price", {}).get(NAIVE_FORECAST_MARKET_ID)
            if price is not None:
                return pd.Series([float(price[t]) for t in index], index=index)
        raise ValueError(
            "no unit of the portfolio has the naive forecast price_LEM_DA; "
            "run round2_add_batteries.py or round2_two_market_participants.py first"
        )

    # ------------------------------------------------------------------------------
    # F2: LEM price forecast
    # ------------------------------------------------------------------------------
    def _lem_price_forecast(
        self, index: pd.DatetimeIndex, p_wm: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        p_lem_hat (EUR/MWh): naive forecast from the LEM price history, p_wm_hat if
        the price of the lagged quarter hour is not known.

        Returns:
            tuple[pd.Series, pd.Series]: forecast and its source ("history" or "wm").
        """
        values, sources = [], []
        for t in index:
            seen = self.lem_price_history.get(t - naive_lag(t))
            if seen is None:
                values.append(float(p_wm[t]))
                sources.append("wm")
            else:
                values.append(seen)
                sources.append("history")
        return pd.Series(values, index=index), pd.Series(sources, index=index)

    # ------------------------------------------------------------------------------
    # Joint bids (same steps as the parent, lines 120-193, with F1, F2, F3, F5)
    # ------------------------------------------------------------------------------
    def _calculate_joint_bids(
        self,
        forecast_snapshot: ForecastSnapshot,
        product_tuples: list[Product],
    ) -> tuple[Orderbook, Orderbook]:
        index = pd.DatetimeIndex([product[0] for product in product_tuples])
        hours = pd.Series(
            [(p[1] - p[0]) / timedelta(hours=1) for p in product_tuples], index=index
        )

        # decision inputs, all in MWh and EUR/MWh
        residual_mw = forecast_snapshot.residual_load_forecast.loc[index].astype(float)
        inputs = pd.DataFrame(index=index)
        inputs["L_hat"] = residual_mw * hours  # MW -> MWh
        sell = inputs.L_hat < 0
        inputs["p_wm"] = self._wm_price_forecast(index)  # F1
        if self.F2_LEM_PRICE_FROM_HISTORY:
            p_lem, p_lem_source = self._lem_price_forecast(index, inputs.p_wm)
            inputs["p_lem"] = p_lem
        else:
            inputs["p_lem"] = sell.map(
                {True: self.LEM_SELL_PRICE_EUR_MWH, False: self.LEM_BUY_PRICE_EUR_MWH}
            )
            p_lem_source = pd.Series("corridor", index=index)
        inputs["r_hat"] = inputs.p_wm - self.R_HAT_PENALTY_EUR_MWH * sell.map(
            {True: 1.0, False: -1.0}
        )
        inputs["S_hat"] = self._expected_lem_volume(inputs.L_hat)
        # same column order as round 1, so solve_alpha() sees the same table
        inputs = inputs[["L_hat", "p_lem", "p_wm", "r_hat", "S_hat"]]

        decision = apply_tie_break(solve_alpha(inputs), self.F5_TIE_BREAK)  # F5
        decision["p_lem_source"] = p_lem_source
        for t, row in decision.iterrows():
            self.decision_log.append(
                {"decision_time": forecast_snapshot.information_time, "start_time": t}
                | row.to_dict()
            )

        wm_orderbook: Orderbook = []
        lem_orderbook: Orderbook = []
        for product in product_tuples:
            start_time, end_time, only_hours = product[0], product[1], product[2]
            alpha = float(decision.at[start_time, "alpha"])
            residual = float(residual_mw[start_time])  # MW, + buy / - sell
            if abs(residual) < 1e-12:
                continue
            buying = residual > 0
            # portfolio sign (+ buy) -> ASSUME sign (buy < 0)
            lem_volume = -alpha * residual
            wm_volume = -(1.0 - alpha) * residual
            if self.F3_LEM_BID_AT_WM_FORECAST:
                lem_price = float(inputs.at[start_time, "p_wm"])  # F3
            else:
                lem_price = (
                    self.LEM_BUY_PRICE_EUR_MWH if buying else self.LEM_SELL_PRICE_EUR_MWH
                )
            wm_price = self.WM_BUY_PRICE_EUR_MWH if buying else self.WM_SELL_PRICE_EUR_MWH
            if abs(wm_volume) > 1e-12:
                wm_orderbook.append(
                    {
                        "bid_id": f"portfolio_WM_{start_time.isoformat()}",
                        "unit_id": PORTFOLIO_UNIT_ID,
                        "start_time": start_time,
                        "end_time": end_time,
                        "only_hours": only_hours,
                        "price": wm_price,
                        "volume": wm_volume,
                    }
                )
            if abs(lem_volume) > 1e-12:
                lem_orderbook.append(
                    {
                        "bid_id": f"portfolio_LEM_{start_time.isoformat()}",
                        "unit_id": PORTFOLIO_UNIT_ID,
                        "start_time": start_time,
                        "end_time": end_time,
                        "only_hours": only_hours,
                        "price": lem_price,
                        "volume": lem_volume,
                    }
                )
        return wm_orderbook, lem_orderbook

    # ------------------------------------------------------------------------------
    # Feedback: also store the LEM clearing price (F2)
    # ------------------------------------------------------------------------------
    def handle_market_feedback(
        self,
        market_id: str,
        accepted_orders: Orderbook,
        rejected_orders: Orderbook,
    ) -> None:
        super().handle_market_feedback(market_id, accepted_orders, rejected_orders)
        if market_id != LEM_MARKET_ID:
            return
        for order in accepted_orders:
            if order.get("unit_id") == PORTFOLIO_UNIT_ID:
                self.lem_price_history[order["start_time"]] = float(order["accepted_price"])
        for order in rejected_orders:
            if order.get("unit_id") != PORTFOLIO_UNIT_ID:
                continue
            price = float(order.get("accepted_price", 0.0) or 0.0)
            if price != 0.0:  # 0 = nothing cleared (simple.py 162-163), not a price
                self.lem_price_history[order["start_time"]] = price
