# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for two helpers of examples/inputs/LocalRetailer_test/simbench_to_assume.py:

- to_simulation_time(): German local time -> UTC (simulation time since 2026-10-29)
- split_nodes(): SimBench profile names decide household vs commercial customer
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "inputs"
    / "LocalRetailer_test"
    / "simbench_to_assume.py"
)
spec = importlib.util.spec_from_file_location("simbench_to_assume", SCRIPT)
s2a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2a)


# ---------------------------------------------------------------------------
# to_simulation_time
# ---------------------------------------------------------------------------
def test_summer_time_is_two_hours_ahead_of_utc():
    local = pd.DatetimeIndex(["2016-06-21 14:00"])
    assert s2a.to_simulation_time(local)[0] == pd.Timestamp("2016-06-21 12:00")


def test_winter_time_is_one_hour_ahead_of_utc():
    local = pd.DatetimeIndex(["2016-01-15 13:00"])
    assert s2a.to_simulation_time(local)[0] == pd.Timestamp("2016-01-15 12:00")


def test_daylight_saving_days_give_a_gapless_utc_axis():
    # local quarter hours of 27.03.2016 (02:00-02:45 missing) and 30.10.2016
    # (02:00-02:45 twice), as in the SimBench CSV files
    march = pd.date_range("2016-03-27 00:00", "2016-03-27 05:45", freq="15min")
    march = march[(march.hour != 2)]
    october = (
        pd.date_range("2016-10-30 00:00", "2016-10-30 02:45", freq="15min")
        .append(pd.date_range("2016-10-30 02:00", "2016-10-30 02:45", freq="15min"))
        .append(pd.date_range("2016-10-30 03:00", "2016-10-30 05:45", freq="15min"))
    )  # file order: the repeated hour follows the first one
    for local in (march, october):
        utc = s2a.to_simulation_time(local)
        assert utc.is_unique
        assert (utc[1:] - utc[:-1] == pd.Timedelta("15min")).all()


# ---------------------------------------------------------------------------
# split_nodes
# ---------------------------------------------------------------------------
def _tables(commercial_mw: float):
    """Three nodes: household with rooftop PV, household, commercial with community PV."""
    index = pd.date_range("2016-01-01", periods=4 * 24 * 366, freq="15min")
    node_load = pd.DataFrame(
        {"N1": 0.0005, "N2": 0.0005, "N3": commercial_mw}, index=index
    )
    load = pd.DataFrame(
        {"node": ["N1", "N2", "N2", "N3"], "profile": ["H0-B", "H0-G", "H0-L", "G1-A"]}
    )
    res = pd.DataFrame(
        {"node": ["N1", "N3"], "type": ["PV", "PV"], "pRES": [0.0087, 0.1288]}
    )
    return node_load, res, load


def test_profile_names_decide_the_split():
    node_load, res, load = _tables(commercial_mw=0.002)  # 17.6 MWh per year
    prosumers, aggregator, independent, community = s2a.split_nodes(
        node_load, res, load
    )
    assert prosumers == ["N1"]
    assert aggregator == ["N2"]  # H0-G and H0-L are household series
    assert independent == ["N3"]
    assert list(community.node) == ["N3"]


def test_small_commercial_node_stops_the_script():
    node_load, res, load = _tables(commercial_mw=0.001)  # 8.8 MWh per year
    with pytest.raises(ValueError, match="household limit"):
        s2a.split_nodes(node_load, res, load)


def test_commercial_prosumer_stops_the_script():
    node_load, res, load = _tables(commercial_mw=0.002)
    res.loc[1, "pRES"] = 0.01  # rooftop PV on the commercial node
    with pytest.raises(ValueError, match="commercial profile"):
        s2a.split_nodes(node_load, res, load)
