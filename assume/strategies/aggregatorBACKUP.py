from dataclasses import dataclass
from datetime import datetime

from assume.common.market_objects import (
    MarketConfig,
    Orderbook,
    Product,
)
from assume.common.utils import timestamp2datetime
from assume.strategies.portfolio_strategies import UnitOperatorStrategy

from assume.common.local_retailer_state import (
    ForecastSnapshot,
    build_forecast_residual_load,
)


@dataclass
class JointBidPlan:
    decision_time: datetime
    forecast_snapshot: ForecastSnapshot
    wm_orderbook: Orderbook
    lem_orderbook: Orderbook


@dataclass
class PortfolioMarketResult:
    market_id: str
    accepted_orders: Orderbook
    rejected_orders: Orderbook


class LocalRetailerCoordinatedStrategy(UnitOperatorStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # -------------------------------------------------
        # Joint WM + LEM bidding plan
        # -------------------------------------------------
        self.joint_plan: JointBidPlan | None = None

        # -------------------------------------------------
        # Temporary allocation parameter
        #
        # alpha = share of total portfolio bidding volume
        #         allocated to LEM
        #
        # alpha = 0.5:
        # 50 % LEM
        # 50 % WM
        # -------------------------------------------------
        self.alpha = kwargs.get("alpha", 0.5)

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                f"alpha must be between 0 and 1, got {self.alpha}"
            )

        # -------------------------------------------------
        # Temporary test bid prices
        # -------------------------------------------------
        self.buy_bid_price = kwargs.get(
            "buy_bid_price",
            3000.0,
        )

        self.sell_bid_price = kwargs.get(
            "sell_bid_price",
            -500.0,
        )
        #### Todo: 价格应该为正

        # -------------------------------------------------
        # Clearing results of WM and LEM
        # -------------------------------------------------
        self.market_results: dict[
            str,
            list[PortfolioMarketResult],
        ] = {}

    def _calculate_joint_bids(
        self,
        forecast_snapshot: ForecastSnapshot,
        product_tuples: list[Product],
    ) -> tuple[Orderbook, Orderbook]:

        wm_orderbook: Orderbook = []
        lem_orderbook: Orderbook = []

        for product in product_tuples:

            start_time = product[0]
            end_time = product[1]
            only_hours = product[2]

            # -------------------------------------------------
            # 1. Total portfolio residual load
            # -------------------------------------------------
            residual_load = float(
                forecast_snapshot.residual_load_forecast.loc[
                    start_time
                ]
            )

            # Nothing to trade
            if residual_load == 0.0:
                continue

            # -------------------------------------------------
            # 2. Parallel WM / LEM allocation
            #
            # q_LEM = alpha * residual_load
            # q_WM  = (1 - alpha) * residual_load
            # -------------------------------------------------
            lem_quantity = (
                self.alpha
                * residual_load
            )

            wm_quantity = (
                (1.0 - self.alpha)
                * residual_load
            )

            # -------------------------------------------------
            # 3. Convert portfolio sign convention
            #    to ASSUME market-order convention
            #
            # Portfolio:
            # positive = electricity needed
            # negative = surplus
            #
            # ASSUME:
            # negative volume = demand / buy
            # positive volume = supply / sell
            # -------------------------------------------------
            lem_volume = -lem_quantity
            wm_volume = -wm_quantity

            # -------------------------------------------------
            # 4. Temporary test prices
            # -------------------------------------------------
            if residual_load > 0:
                wm_price = self.buy_bid_price
                lem_price = self.buy_bid_price

            else:
                wm_price = self.sell_bid_price
                lem_price = self.sell_bid_price

            # -------------------------------------------------
            # 5. WM portfolio order
            # -------------------------------------------------
            if wm_volume != 0.0:

                wm_orderbook.append(
                    {
                        "bid_id": (
                            f"portfolio_WM_"
                            f"{start_time.isoformat()}"
                        ),
                        "unit_id": "__portfolio__",
                        "start_time": start_time,
                        "end_time": end_time,
                        "only_hours": only_hours,
                        "price": wm_price,
                        "volume": wm_volume,
                    }
                )

            # -------------------------------------------------
            # 6. LEM portfolio order
            # -------------------------------------------------
            if lem_volume != 0.0:

                lem_orderbook.append(
                    {
                        "bid_id": (
                            f"portfolio_LEM_"
                            f"{start_time.isoformat()}"
                        ),
                        "unit_id": "__portfolio__",
                        "start_time": start_time,
                        "end_time": end_time,
                        "only_hours": only_hours,
                        "price": lem_price,
                        "volume": lem_volume,
                    }
                )

        return wm_orderbook, lem_orderbook

    def _ensure_joint_plan(
        self,
        units_operator,
        product_tuples: list[Product],
        decision_time: datetime,
    ) -> JointBidPlan:

        # -------------------------------------------------
        # 1. Same decision time -> reuse same joint plan
        # -------------------------------------------------
        if (
            self.joint_plan is not None
            and self.joint_plan.decision_time == decision_time
        ):
            return self.joint_plan

        # -------------------------------------------------
        # 2. Delivery periods
        # -------------------------------------------------
        delivery_index = [
            product[0]
            for product in product_tuples
        ]

        # -------------------------------------------------
        # 3. Build one common portfolio forecast
        # -------------------------------------------------
        forecast_snapshot = build_forecast_residual_load(
            units=units_operator.units,
            delivery_index=delivery_index,
            information_time=decision_time,
        )

        # -------------------------------------------------
        # 4. Calculate WM + LEM bids simultaneously
        # -------------------------------------------------
        wm_orderbook, lem_orderbook = (
            self._calculate_joint_bids(
                forecast_snapshot=forecast_snapshot,
                product_tuples=product_tuples,
            )
        )
        #fixme: 注意orderbook 复写的风险，只复写当前time的值

        # -------------------------------------------------
        # 5. Store common joint plan
        # -------------------------------------------------
        self.joint_plan = JointBidPlan(
            decision_time=decision_time,
            forecast_snapshot=forecast_snapshot,
            wm_orderbook=wm_orderbook,
            lem_orderbook=lem_orderbook,
        )

        return self.joint_plan

    def calculate_bids(
        self,
        units_operator,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:

        # -------------------------------------------------
        # 1. Current decision time
        # -------------------------------------------------
        decision_time = timestamp2datetime(
            units_operator.context.current_timestamp
        )

        # -------------------------------------------------
        # 2. Get or create common WM + LEM plan
        # -------------------------------------------------
        joint_plan = self._ensure_joint_plan(
            units_operator=units_operator,
            product_tuples=product_tuples,
            decision_time=decision_time,
        )

        # -------------------------------------------------
        # 3. Return the orderbook requested by ASSUME
        # -------------------------------------------------
        market_id = market_config.market_id

        if market_id == "WM_DA":
            return joint_plan.wm_orderbook

        elif market_id == "LEM_DA":
            return joint_plan.lem_orderbook

        else:
            raise ValueError(
                f"Unsupported market for Local Retailer: "
                f"{market_id}"
            )

    def handle_market_feedback(
        self,
        market_id: str,
        accepted_orders: Orderbook,
        rejected_orders: Orderbook,
    ) -> None:

        # -------------------------------------------------
        # Keep only Aggregator portfolio orders
        # -------------------------------------------------
        portfolio_accepted_orders = [
            order.copy()
            for order in accepted_orders
            if order.get("unit_id") == "__portfolio__"
        ]

        portfolio_rejected_orders = [
            order.copy()
            for order in rejected_orders
            if order.get("unit_id") == "__portfolio__"
        ]

        # -------------------------------------------------
        # Store clearing result
        # -------------------------------------------------
        result = PortfolioMarketResult(
            market_id=market_id,
            accepted_orders=portfolio_accepted_orders,
            rejected_orders=portfolio_rejected_orders,
        )

        self.market_results.setdefault(
            market_id,
            [],
        ).append(result)