# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseUnit
from assume.units.demand import Demand
from assume.units.energy_hub import EnergyHub
from assume.units.exchange import Exchange
from assume.units.powerplant import PowerPlant
from assume.units.storage import Storage
from assume.units.charging_station import V2gChargingRB
from assume.units.steel_plant import SteelPlant
from assume.units.steam_generation_plant import SteamPlant
from assume.units.hydrogen_plant import HydrogenPlant
from assume.units.building import Building
from assume.units.dst_components import demand_side_technologies
from assume.units.comando_components import comando_dst

unit_types: dict[str, type[BaseUnit]] = {
    "power_plant": PowerPlant,
    "demand": Demand,
    "exchange": Exchange,
    "storage": Storage,
    "steel_plant": SteelPlant,
    "hydrogen_plant": HydrogenPlant,
    "steam_plant": SteamPlant,
    "building": Building,
    "energy_hub": EnergyHub,
    "V2G_Charging": V2gChargingRB,
}
