from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from assume.units.demand import Demand
from assume.units.building import Building


@dataclass
class ForecastSnapshot:
    information_time: datetime
    delivery_index: pd.DatetimeIndex

    demand_forecast: pd.Series
    prosumer_load_forecast: pd.Series
    pv_forecast: pd.Series
    residual_load_forecast: pd.Series


def build_forecast_residual_load(
    units,
    delivery_index,
    information_time,
):

    delivery_index = pd.DatetimeIndex(delivery_index)

    if delivery_index.min() <= pd.Timestamp(information_time):
        raise ValueError(
        f"Delivery period starts at {delivery_index.min()}, which is not after "
        f"the information time {information_time}."
    )

    # -------------------------------------------------
    # 1. Initialize portfolio forecast components
    # -------------------------------------------------
    demand_forecast = pd.Series(
        0.0,
        index=delivery_index,
    )

    prosumer_load_forecast = pd.Series(
        0.0,
        index=delivery_index,
    )

    pv_forecast = pd.Series(
        0.0,
        index=delivery_index,
    )

    # -------------------------------------------------
    # 2. Aggregate all units of the Local Retailer
    # -------------------------------------------------
    for unit_id, unit in units.items():

        # ---------------------------------------------
        # Pure Demand
        # ---------------------------------------------
        if isinstance(unit, Demand):

            demand_values = -np.asarray(
                unit.forecaster.demand[delivery_index],
                dtype=float,
            )

            demand_forecast += pd.Series(
                demand_values,
                index=delivery_index,
            )

        # ---------------------------------------------
        # Prosumer / Building
        # ---------------------------------------------
        elif isinstance(unit, Building):

            # Household consumption
            load_values = np.asarray(
                unit.forecaster.load_profile[delivery_index],
                dtype=float,
            )

            prosumer_load_forecast += pd.Series(
                load_values,
                index=delivery_index,
            )

            # -------------------------------------------------
            # Temporary simple PV forecast for Aggregator test
            #
            # Current House_X_pv_profile is normalized (0...1).
            # Convert it to absolute PV power using installed
            # PV capacity.
            # -------------------------------------------------
            pv_profile_values = np.asarray(
                unit.forecaster.pv_profile[delivery_index],
                dtype=float,
            )

            total_pv_capacity = sum(
                float(unit.components[pv_key].max_power)
                for pv_key in unit.pv_plants
            )

            pv_values = (
                pv_profile_values
                * total_pv_capacity
            )

            pv_forecast += pd.Series(
                pv_values,
                index=delivery_index,
            )

    # -------------------------------------------------
    # 3. Calculate portfolio residual-load forecast
    # -------------------------------------------------
    residual_load_forecast = (
        demand_forecast
        + prosumer_load_forecast
        - pv_forecast
    )

    # -------------------------------------------------
    # 4. Return one forecast snapshot
    # -------------------------------------------------
    return ForecastSnapshot(
        information_time=information_time,
        delivery_index=delivery_index,
        demand_forecast=demand_forecast,
        prosumer_load_forecast=prosumer_load_forecast,
        pv_forecast=pv_forecast,
        residual_load_forecast=residual_load_forecast,
    )
# Feedbacks: pv预测出来的