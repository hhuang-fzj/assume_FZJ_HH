"""
Local retailer strategy with optimised LEM/WM allocation (WP8).

Same as LocalRetailerCoordinatedStrategy of b22d8b5 (aggregator.py), but alpha is no
longer a fixed number: for every delivery period the strategy builds the decision inputs
and calls solve_alpha() (alpha_model.py, written in WP4). Everything else (forecast of
the residual load, one joint plan for both markets, price cap, storing the feedback) is
inherited unchanged.

Information used by the optimiser (all known at the decision time, nothing about other
participants):
    L_hat   own portfolio residual load forecast        build_forecast_residual_load()
    p_wm    WM price forecast                           forecasts_df.csv, column price_WM_DA
    p_lem   LEM price forecast                          corridor bound (rule below)
    r_hat   expected imbalance price                    p_wm -/+ penalty
    S_hat   expected LEM volume the LEM takes           |L_hat| (option 1) or own feedback

Sources (full citations: reference list in the course document)
    WM price taker, bids at the price limits so that they are always accepted:
        [Iria2019] p. 1363 [OK] (buy at the cap); selling at the floor is the symmetric
        case [OWN].
    LEM limit prices = corridor bounds (sell 123.1, buy 288 EUR/MWh): corridor
        [Lezama2021] p. 4 Eq. (1c)(2c), [Zepter2019] p. 166 [OK]; 288: [BDEW2020]
        S. 7-8 [OK]; 123.1: [FIT2016] [TODO]. Using the bound as own limit price [OWN].
    LEM price forecast = same bound: PV surplus pushes the LEM price to the lower bound,
        [Lezama2021] p. 12 [OK]; using the bound as forecast [OWN].
    r_hat = p_wm -/+ penalty: form [Ottesen2016] p. 830, 839 [OK]; no speculation on the
        imbalance price [EBGL] Art. 17(1), [BK6-13-104] [OK]; value 1 EUR/MWh [OWN].
    S_hat: option 1 (default) full acceptance, [Iria2019] p. 1363 [OK].
        Option 2 (switch S_HAT_FROM_FEEDBACK): learn from own clearing results,
        principle [Kuehnbach2022] Sect. 2.3: participants only learn whether their bids
        were accepted (step 5, p. 7) and adapt to prior market results (step 3,
        p. 6-7) [OK]; there it is the bid price that is learned, here the volume;
        concrete rule [OWN, proposed by Claude]:
        LEM took only part of the offer in the same quarter hour of the previous
        delivery day -> S_hat = accepted volume; otherwise S_hat = |L_hat|.
        Reason for |L_hat| after full acceptance: full acceptance only shows that the
        LEM could take at least the offered volume, so the aggregator tries the full
        volume again (drawback: alternates between full and partial offers).
    Units: ASSUME order volumes are power in MW for the product duration; the model
        works in MWh, E = P * duration [OWN conversion]. ASSUME sign: demand (buy) < 0,
        supply (sell) > 0 (b22d8b5 aggregator.py, step 3).
"""

from datetime import timedelta

import pandas as pd

from assume.common.local_retailer_state import ForecastSnapshot
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.strategies.aggregator import LocalRetailerCoordinatedStrategy
from assume.strategies.alpha_model import solve_alpha

WM_MARKET_ID = "WM_DA"
LEM_MARKET_ID = "LEM_DA"


class AlphaOptimizedLocalRetailerStrategy(LocalRetailerCoordinatedStrategy):
    """Joint WM/LEM bidding of the local retailer with alpha from solve_alpha()."""

    # --- settings (see module docstring for the sources) --------------------------
    LEM_SELL_PRICE_EUR_MWH = 123.1  # lower corridor bound
    LEM_BUY_PRICE_EUR_MWH = 288.0  # upper corridor bound
    WM_SELL_PRICE_EUR_MWH = -500.0  # WM price floor -> sell order always accepted
    WM_BUY_PRICE_EUR_MWH = 3000.0  # WM price cap -> buy order always accepted
    R_HAT_PENALTY_EUR_MWH = 1.0
    S_HAT_FROM_FEEDBACK = False  # False = option 1, True = option 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._units_operator = None
        # one row per delivery period and decision: optimiser layer of the results
        self.decision_log: list[dict] = []
        # own LEM feedback per delivery start: offered and accepted energy in MWh
        self.lem_feedback: dict = {}

    # ------------------------------------------------------------------------------
    # 1. Keep a reference to the units operator, so the WM price forecast can be read
    # ------------------------------------------------------------------------------
    def calculate_bids(
        self,
        units_operator,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        self._units_operator = units_operator
        return super().calculate_bids(
            units_operator, market_config, product_tuples, **kwargs
        )

    def _wm_price_forecast(self, index: pd.DatetimeIndex) -> pd.Series:
        """WM price forecast (EUR/MWh) from the forecaster of the portfolio units."""
        for unit in self._units_operator.units.values():
            price = getattr(unit.forecaster, "price", {}).get(WM_MARKET_ID)
            if price is not None:
                return pd.Series([float(price[t]) for t in index], index=index)
        raise ValueError(f"no unit of the portfolio has a {WM_MARKET_ID} price forecast")

    # ------------------------------------------------------------------------------
    # 2. Expected LEM volume S_hat
    # ------------------------------------------------------------------------------
    def _expected_lem_volume(self, L_hat: pd.Series) -> pd.Series:
        """S_hat in MWh: option 1 = |L_hat|; option 2 = from own previous-day feedback."""
        S_hat = L_hat.abs()
        if not self.S_HAT_FROM_FEEDBACK:
            return S_hat
        for t in L_hat.index:
            previous = self.lem_feedback.get(t - timedelta(days=1))
            if previous is None:
                continue  # no feedback yet -> full acceptance
            offered, accepted = previous
            if offered > 1e-12 and accepted < offered - 1e-12:
                S_hat[t] = accepted  # LEM took only part: that is what it can take
        return S_hat

    # ------------------------------------------------------------------------------
    # 3. Joint bids: inputs -> solve_alpha() -> orders
    # ------------------------------------------------------------------------------
    def _calculate_joint_bids(
        self,
        forecast_snapshot: ForecastSnapshot,
        product_tuples: list[Product],
    ) -> tuple[Orderbook, Orderbook]:
        index = pd.DatetimeIndex([product[0] for product in product_tuples])
        duration_h = {
            product[0]: (product[1] - product[0]) / timedelta(hours=1)
            for product in product_tuples
        }
        hours = pd.Series([duration_h[t] for t in index], index=index)

        # decision inputs, all in MWh and EUR/MWh
        residual_mw = forecast_snapshot.residual_load_forecast.loc[index].astype(float)
        inputs = pd.DataFrame(index=index)
        inputs["L_hat"] = residual_mw * hours  # MW -> MWh
        sell = inputs.L_hat < 0
        inputs["p_lem"] = sell.map(
            {True: self.LEM_SELL_PRICE_EUR_MWH, False: self.LEM_BUY_PRICE_EUR_MWH}
        )
        inputs["p_wm"] = self._wm_price_forecast(index)
        inputs["r_hat"] = inputs.p_wm - self.R_HAT_PENALTY_EUR_MWH * sell.map(
            {True: 1.0, False: -1.0}
        )
        inputs["S_hat"] = self._expected_lem_volume(inputs.L_hat)

        decision = solve_alpha(inputs)
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
            lem_price = (
                self.LEM_BUY_PRICE_EUR_MWH if buying else self.LEM_SELL_PRICE_EUR_MWH
            )
            wm_price = self.WM_BUY_PRICE_EUR_MWH if buying else self.WM_SELL_PRICE_EUR_MWH
            if abs(wm_volume) > 1e-12:
                wm_orderbook.append(
                    {
                        "bid_id": f"portfolio_WM_{start_time.isoformat()}",
                        "unit_id": "__portfolio__",
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
                        "unit_id": "__portfolio__",
                        "start_time": start_time,
                        "end_time": end_time,
                        "only_hours": only_hours,
                        "price": lem_price,
                        "volume": lem_volume,
                    }
                )
        return wm_orderbook, lem_orderbook

    # ------------------------------------------------------------------------------
    # 4. Feedback: store own LEM results (used only by option 2)
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
        for order in accepted_orders + rejected_orders:
            if order.get("unit_id") != "__portfolio__":
                continue
            hours = (order["end_time"] - order["start_time"]) / timedelta(hours=1)
            offered = abs(float(order["volume"])) * hours
            accepted = abs(float(order.get("accepted_volume", 0.0) or 0.0)) * hours
            self.lem_feedback[order["start_time"]] = (offered, accepted)
