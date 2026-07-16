# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

from assume.common.base import BaseUnit
from assume.common.market_objects import MarketConfig, Orderbook

from assume.common.base import SupportsMinMaxCharge
from assume.common.fast_pandas import FastSeries
from assume.common.forecasts import Forecaster

logger = logging.getLogger(__name__)
EPS = 1e-4

class V2gChargingRB(SupportsMinMaxCharge):
    """
    A class for a bidirectional EV charging station.

    The charging station is modeled as an aggregated storage unit with a dynamic
    virtual battery capacity based on connected EVs. EV arrival SOC, departure SOC,
    and virtual battery capacity are read from the forecaster.

    Args:
        id (str): The ID of the charging station.
        unit_operator (str): The operator of the charging station.
        technology (str): The technology of the charging station.
        bidding_strategies (dict): The bidding strategies of the charging station.
        forecaster (Forecaster): The forecaster providing market data and EV schedules.
        node (str): The node of the charging station.
        max_power_charge (float): The maximum power input of the charging station in MW.
        min_power_charge (float): The minimum power input of the charging station in MW.
        max_power_discharge (float): The maximum power output of the charging station in MW.
        min_power_discharge (float): The minimum power output of the charging station in MW.
        max_soc (float): The maximum state of charge in MWh. For this class, it is overwritten by the maximum virtual battery capacity from the forecaster.
        min_soc (float): The minimum state of charge of the charging station in MWh.
        initial_soc (float): The initial state of charge of the charging station in MWh.
        soc_tick (float): The SOC discretization step.
        efficiency_charge (float): The efficiency of the charging station while charging.
        efficiency_discharge (float): The efficiency of the charging station while discharging.
        additional_cost_charge (float, optional): Additional costs associated with power consumption, in EUR/MWh. Defaults to 0.
        additional_cost_discharge (float, optional): Additional costs associated with power generation, in EUR/MWh. Defaults to 0.
        ramp_up_charge (float): The ramp up rate of charging the charging station in MW/15 minutes.
        ramp_down_charge (float): The ramp down rate of charging the charging station in MW/15 minutes.
        ramp_up_discharge (float): The ramp up rate of discharging the charging station in MW/15 minutes.
        ramp_down_discharge (float): The ramp down rate of discharging the charging station in MW/15 minutes.
        hot_start_cost (float): The hot start cost of the charging station in €/MW.
        warm_start_cost (float): The warm start cost of the charging station in €/MW.
        cold_start_cost (float): The cold start cost of the charging station in €/MW.
        downtime_hot_start (float): Definition of downtime before hot start in h.
        downtime_warm_start (float): Definition of downtime before warm start in h.
        min_operating_time (float): The minimum operating time of the charging station in hours.
        min_down_time (float): The minimum down time of the charging station in hours.
        location (tuple[float, float]): The location of the charging station.
        bid_start (datetime.datetime): Start timestamp of the currently evaluated bid timestep.
        bid_end (datetime.datetime): End timestamp of the currently evaluated bid timestep.

    Forecaster inputs:
        charging_Virtual_Battery [kWh]: Aggregated capacity of connected EVs.
        charging_Arrival_SOC [kWh]: SOC added by arriving EVs.
        charging_Departure_SOC [kWh]: Required SOC for departing EVs.
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        max_power_charge: float,
        max_power_discharge: float,
        max_soc: float,
        min_power_charge: float = 0.0,
        min_power_discharge: float = 0.0,
        min_soc: float = 0.0,
        initial_soc: float = 0.0,
        soc_tick: float = 0.01,
        efficiency_charge: float = 1,
        efficiency_discharge: float = 1,
        additional_cost_charge: float = 0.0,
        additional_cost_discharge: float = 0.0,
        ramp_up_charge: float = None,
        ramp_down_charge: float = None,
        ramp_up_discharge: float = None,
        ramp_down_discharge: float = None,
        hot_start_cost: float = 0,
        warm_start_cost: float = 0,
        cold_start_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 8,  # hours
        downtime_warm_start: int = 48,  # hours
        location: tuple[float, float] = (0, 0),
        node: str = "node0",
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )

        # EV schedule inputs are given in kWh and converted to MWh.
        self.virtual_battery = forecaster['charging_Virtual_Battery [kWh]']/1000
        self.arr_SOC = forecaster['charging_Arrival_SOC [kWh]']/1000
        self.dep_SOC = forecaster['charging_Departure_SOC [kWh]']/1000

        self.max_soc = max(forecaster['charging_Virtual_Battery [kWh]']/1000)
        self.virtual_battery = forecaster['charging_Virtual_Battery [kWh]']/1000
        self.min_soc = min_soc
        if initial_soc is None:
            initial_soc = 0
        self.initial_soc = initial_soc

        self.max_power_charge = -abs(max_power_charge)
        self.min_power_charge = -abs(min_power_charge)
        self.max_power_discharge = abs(max_power_discharge)
        self.min_power_discharge = abs(min_power_discharge)

        self.outputs["soc"] = FastSeries(value=self.initial_soc, index=self.index)
        self.outputs["cost_stored_energy"] = FastSeries(value=0.0, index=self.index)

        self.soc_tick = soc_tick

        # The efficiency of the storage unit while charging.
        self.efficiency_charge = efficiency_charge if 0 < efficiency_charge < 1 else 1
        self.efficiency_discharge = (
            efficiency_discharge if 0 < efficiency_discharge < 1 else 1
        )

        # The variable costs to charge/discharge the storage unit.
        self.additional_cost_charge = additional_cost_charge
        self.additional_cost_discharge = additional_cost_discharge

        # The ramp up/down rate of charging/discharging the storage unit.
        # if ramp_up_charge == 0, the ramp_up_charge is set to enable ramping between full charge and discharge power
        # else the ramp_up_charge is set to the negative value of the ramp_up_charge
        self.ramp_up_charge = (
            self.max_power_charge - self.max_power_discharge
            if not ramp_up_charge
            else -abs(ramp_up_charge)
        )
        self.ramp_down_charge = (
            self.max_power_charge - self.max_power_discharge
            if not ramp_down_charge
            else -abs(ramp_down_charge)
        )
        self.ramp_up_discharge = (
            self.max_power_discharge - self.max_power_charge
            if not ramp_up_discharge
            else ramp_up_discharge
        )
        self.ramp_down_discharge = (
            self.max_power_discharge - self.max_power_charge
            if not ramp_down_discharge
            else ramp_down_discharge
        )

        # How long the storage unit has to be in operation before it can be shut down.
        self.min_operating_time = min_operating_time
        # How long the storage unit has to be shut down before it can be started.
        self.min_down_time = min_down_time
        # The downtime before hot start of the storage unit.
        self.downtime_hot_start = downtime_hot_start
        # The downtime before warm start of the storage unit.
        self.downtime_warm_start = downtime_warm_start

        self.hot_start_cost = hot_start_cost * max_power_discharge
        self.warm_start_cost = warm_start_cost * max_power_discharge
        self.cold_start_cost = cold_start_cost * max_power_discharge

        self.bid_start = self.index[0]
        self.bid_end = self.index[1]

    def set_dispatch_plan(
            self,
            marketconfig:MarketConfig,
            orderbook:Orderbook,
    ) -> None:
        """
        Set accepted dispatch for the EV charging station.

        This intentionally bypasses SupportsMinMaxCharge.set_dispatch_plan(),
        because that method updates SOC with normal storage logic.

        For V2gChargingRB, SOC is updated only in execute_current_dispatch()
        using EV-specific arrival/departure logic.
        """

        BaseUnit.set_dispatch_plan(self, marketconfig, orderbook)
    def execute_current_dispatch(self, start: datetime, end: datetime) -> np.ndarray:
        """
        Executes the current dispatch of the unit based on the provided timestamps.

        The dispatch is only executed, if it is in the constraints given by the unit.
        Returns the volume of the unit within the given time range.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            end (datetime.datetime): The end time of the dispatch.

        Returns:
            np.ndarray: The volume of the unit within the given time range.
        """
        start = max(start, self.index[0])
        time_delta = self.index.freq / timedelta(hours=1)

        for t in self.index[start:end]:
            current_power = self.outputs["energy"].at[t]

            # adjust power to constraints of the unit
            if current_power > self.max_power_discharge:
                current_power = self.max_power_discharge
            elif current_power < self.max_power_charge:
                current_power = self.max_power_charge
            elif (
                current_power < self.min_power_discharge
                and current_power > self.min_power_charge
                and current_power != 0
            ):
                current_power = 0

            # calculate the change in state of charge
            delta_soc = 0
            soc = self.outputs["soc"].at[t]

            # discharging
            if current_power > 0:
                max_soc_discharge = self.calculate_soc_max_discharge(soc)

                if current_power > max_soc_discharge:
                    current_power = max_soc_discharge

                delta_soc = -current_power * time_delta / self.efficiency_discharge

            # charging
            elif current_power < 0:
                max_soc_charge = self.calculate_soc_max_charge(soc)

                if current_power < max_soc_charge:
                    current_power = max_soc_charge

                delta_soc = -current_power * time_delta * self.efficiency_charge

            # update the values of the state of charge and the energy
            next_freq = t + self.index.freq
            if next_freq in self.index:
                self.outputs["soc"].at[next_freq] = soc + delta_soc
            self.outputs["energy"].at[t] = current_power

        return self.outputs["energy"].loc[start:end]

    @lru_cache(maxsize=256)
    def calculate_marginal_cost(
        self,
        start: datetime,
        power: float,
    ) -> float:
        """
        Calculates the marginal cost of the unit based on the provided start time and power output and returns it.
        Returns the marginal cost of the unit.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: The marginal cost of the unit.
        """

        if power > 0:
            additional_cost = self.additional_cost_discharge
        else:
            additional_cost = self.additional_cost_charge

        marginal_cost = additional_cost

        return marginal_cost

    def calculate_soc_max_discharge(
            self,
            soc,
    ) -> float:
        """
        Calculates the maximum discharge power for the current bid timestep.

        The discharge limit is based on the current SOC and the required target SOC.
        If an EV departs at self.bid_start, the departure SOC is used as the target
        SOC. Otherwise, min_soc is used.

        Args:
            soc (float): The current state of charge in MWh.

        Returns:
            float: The maximum discharge power in MW.
        """
        duration = self.index.freq / timedelta(hours=1)
        dep_soc = self.dep_SOC.at[self.bid_start]

        if dep_soc > 0:
            target_soc = dep_soc
        else:
            target_soc = self.min_soc

        power = max(
            0,
            (soc - target_soc) * self.efficiency_discharge / duration,
        )
        return power

    def calculate_soc_max_charge(
        self,
        soc,
    ) -> float:
        """
        Calculates the maximum charge power for the current bid timestep.

        The charge limit is based on the current SOC and the allowed target SOC.
        If an EV departs at self.bid_start, the departure SOC is used as the target
        SOC. Otherwise, the virtual battery capacity at self.bid_start is used.

        Args:
            soc (float): The current state of charge in MWh.

        Returns:
            float: The maximum charge power in MW as a negative value.
        """
        duration = self.index.freq / timedelta(hours=1)
        dep_soc = self.dep_SOC.at[self.bid_start]

        if dep_soc > 0:
            # If an EV departs at the end of this timestep,
            # charge only until its requested departure SOC.
            target_soc = dep_soc
        else:
            # Otherwise charge up to connected virtual battery capacity.
            target_soc = self.virtual_battery.at[self.bid_start]

        power = min(
            0,
            (soc - target_soc) / self.efficiency_charge / duration,
        )
        return power

    def calculate_min_max_charge(
        self, start: datetime, end: datetime, soc: float = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the min and max charging power for the given time period.
        This is relative to the already sold output on other markets for the same period.
        It also adheres to reserved positive and negative capacities reserved for other markets.

        Args:
            start (datetime.datetime): The start of the current dispatch.
            end (datetime.datetime): The end of the current dispatch.
            soc (float): The current state-of-charge. Defaults to None, then using soc at given start time.

        Returns:
            tuple[np.ndarray, np.ndarray]: The minimum and maximum charge power levels of the storage unit in MW.
        """
        # end includes the end of the last product, to get the last products' start time we deduct the frequency once
        end_excl = end - self.index.freq

        base_load = self.outputs["energy"].loc[start:end_excl]
        capacity_pos = self.outputs["capacity_pos"].loc[start:end_excl]
        capacity_neg = self.outputs["capacity_neg"].loc[start:end_excl]

        min_power_charge = self.min_power_charge - (base_load + capacity_pos)
        min_power_charge = min_power_charge.clip(max=0)

        max_power_charge = self.max_power_charge - (base_load + capacity_neg)
        max_power_charge = np.where(
            max_power_charge <= min_power_charge, max_power_charge, 0
        )
        min_power_charge = np.where(
            min_power_charge >= max_power_charge, min_power_charge, 0
        )

        # restrict charging according to max_soc
        if soc is None:
            soc = self.outputs["soc"].at[start]
        max_soc_charge = self.calculate_soc_max_charge(soc)
        max_power_charge = max_power_charge.clip(min=max_soc_charge)

        return min_power_charge, max_power_charge

    def calculate_min_max_discharge(
        self, start: datetime, end: datetime, soc: float = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the min and max discharging power for the given time period.
        This is relative to the already sold output on other markets for the same period.
        It also adheres to reserved positive and negative capacities.

        Args:
            start (datetime.datetime): The start of the current dispatch.
            end (datetime.datetime): The end of the current dispatch.
            soc (float): The current state-of-charge. Defaults to None, then using soc at given start time.

        Returns:
            tuple[np.ndarray, np.ndarray]: The minimum and maximum discharge power levels of the storage unit in MW.
        """
        # end includes the end of the last product, to get the last products' start time we deduct the frequency once
        end_excl = end - self.index.freq

        base_load = self.outputs["energy"].loc[start:end_excl]
        capacity_pos = self.outputs["capacity_pos"].loc[start:end_excl]
        capacity_neg = self.outputs["capacity_neg"].loc[start:end_excl]

        min_power_discharge = self.min_power_discharge - (base_load + capacity_neg)
        min_power_discharge = min_power_discharge.clip(min=0)

        max_power_discharge = self.max_power_discharge - (base_load + capacity_pos)

        # Adjust max_power_discharge using np.where
        max_power_discharge = np.where(
            max_power_discharge >= min_power_discharge, max_power_discharge, 0
        )

        # Adjust min_power_discharge using np.where
        min_power_discharge = np.where(
            min_power_discharge < max_power_discharge, min_power_discharge, 0
        )

        # restrict according to min_soc
        if soc is None:
            soc = self.outputs["soc"].at[start]
        max_soc_discharge = self.calculate_soc_max_discharge(soc)
        max_power_discharge = max_power_discharge.clip(max=max_soc_discharge)

        return min_power_discharge, max_power_discharge

    def calculate_ramp_discharge(
        self,
        soc: float,
        previous_power: float,
        power_discharge: float,
        current_power: float = 0,
        min_power_discharge: float = 0,
    ) -> float:
        """
        Adjusts the discharging power to the ramping constraints.

        Args:
            soc (float): The current state of charge.
            previous_power (float): The previous power output of the unit.
            power_discharge (float): The discharging power output of the unit.
            current_power (float, optional): The current power output of the unit. Defaults to 0.
            min_power_discharge (float, optional): The minimum discharging power output of the unit. Defaults to 0.

        Returns:
            float: The discharging power adjusted to the ramping constraints.
        """
        power_discharge = super().calculate_ramp_discharge(
            previous_power,
            power_discharge,
            current_power,
        )
        # restrict according to min_SOC

        max_soc_discharge = self.calculate_soc_max_discharge(soc)
        power_discharge = min(power_discharge, max_soc_discharge)
        if power_discharge < min_power_discharge:
            power_discharge = 0

        return power_discharge

    def calculate_ramp_charge(
        self,
        soc: float,
        previous_power: float,
        power_charge: float,
        current_power: float = 0,
        min_power_charge: float = 0,
    ) -> float:
        """
        Adjusts the charging power to the ramping constraints.

        Args:
            soc (float): The current state of charge.
            previous_power (float): The previous power output of the unit.
            power_charge (float): The charging power output of the unit.
            current_power (float, optional): The current power output of the unit. Defaults to 0.
            min_power_charge (float, optional): The minimum charging power output of the unit. Defaults to 0.

        Returns:
            float: The charging power adjusted to the ramping constraints.
        """
        power_charge = super().calculate_ramp_charge(
            previous_power,
            power_charge,
            current_power,
        )

        # restrict charging according to max_SOC
        max_soc_charge = self.calculate_soc_max_charge(soc)

        power_charge = max(power_charge, max_soc_charge)
        if power_charge > min_power_charge:
            power_charge = 0

        return power_charge

    def get_starting_costs(self, op_time):
        """
        Calculates the starting costs of the unit depending on how long it was shut down

        Args:
            op_time (float): The time the unit was shut down in hours.

        Returns:
            float: The starting costs of the unit.
        """
        if op_time > 0:
            # unit is running
            return 0
        if -op_time < self.downtime_hot_start:
            return self.hot_start_cost
        elif -op_time < self.downtime_warm_start:
            return self.warm_start_cost
        else:
            return self.cold_start_cost

    def as_dict(self) -> dict:
        """
        Return the storage unit's attributes as a dictionary, including specific attributes.

        Returns:
            dict: The storage unit's attributes as a dictionary.
        """
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "max_soc": self.max_soc,
                "min_soc": self.min_soc,
                "max_power_charge": self.max_power_charge,
                "max_power_discharge": self.max_power_discharge,
                "min_power_charge": self.min_power_charge,
                "min_power_discharge": self.min_power_discharge,
                "efficiency_charge": self.efficiency_discharge,
                "efficiency_discharge": self.efficiency_charge,
                "unit_type": "storage",# also decide which meta table the data from this unit will be written
            }
        )

        return unit_dict

    def update_virtual_battery_soc(self, start):
        """
        Updates the SOC according to EV arrivals and departures.

        The method assumes that EVs arrive at the beginning of a timestep and leave
        at the end of a timestep. Therefore, arrival SOC at the current timestep is
        added to the SOC, while departure SOC from the previous timestep is
        subtracted from the SOC.

        Args:
            start (datetime.datetime): The current simulation timestep.

        Returns:
            float: The updated state of charge in MWh.
        """

        theoretic_SOC = self.outputs["soc"].at[start]
        arrive_energy = self.arr_SOC.at[start]
        leave_energy = self.dep_SOC.at[start-self.index.freq]


        if arrive_energy:
            updated_soc = theoretic_SOC + arrive_energy
            self.outputs["soc"].at[start] = updated_soc
        elif leave_energy :
            updated_soc = theoretic_SOC - leave_energy
            self.outputs["soc"].at[start] = updated_soc
        else:
            updated_soc = theoretic_SOC

        return updated_soc
