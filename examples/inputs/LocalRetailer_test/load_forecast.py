# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
 
"""
Load forecast input generation for the LocalRetailer_test scenario.
 
This script builds forecast and realised load time series for the pure demand
unit (``Demand_1``) and the prosumer building (``House_1``) based on the BDEW
standard load profile H25, and writes them into the scenario input files.
 
Pipeline:
    1. **Read** the raw BDEW H25 table (month x day type x quarter hour).
    2. **Expand** the table onto the simulation calendar and apply the BDEW
       dynamisation function, then aggregate to hourly resolution.
    3. **Scale** the profile to an annual consumption in kWh.
    4. **Realise** load by adding autocorrelated noise (stand-in for smart meter data).
    5. **Forecast** load as the mean of the same weekday and hour of the previous weeks.
    6. **Write** forecast and realised series into the ASSUME input files.
 
Note:
    The realised load is synthetic (profile plus noise) and serves as a placeholder
    for measured smart meter (iMSys) data. Replacing it only requires changing step 4.
"""
 
import logging
import shutil
from datetime import datetime
from pathlib import Path
 
import holidays
import numpy as np
import pandas as pd
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Configuration (all modelling assumptions are collected here)
# ---------------------------------------------------------------------------
SCENARIO_PATH = Path(__file__).parent
RAW_FILE = SCENARIO_PATH / "bdew_slp_2025.xlsx"  # BDEW download, renamed to ASCII
RAW_SHEET = "H25"
HOLIDAY_SUBDIVISION = "NW"  # federal state for public holidays (North Rhine-Westphalia)
 
HISTORY_START = "2018-12-01 00:00"  # four weeks of history are needed for the forecast
SIMULATION_START = "2019-01-01 00:00"
SIMULATION_END = "2019-12-31 23:00"
 
# Annual consumption per unit in kWh (assumption, source required)
ANNUAL_CONSUMPTION_KWH: dict[str, float] = {
    "Demand_1": 3500.0,
    "House_1": 4000.0,
}
 
# Parameters of the AR(1) noise per unit (assumption, source required).
# Different seeds keep the deviations of the two units uncorrelated.
NOISE_PARAMS: dict[str, dict[str, float | int]] = {
    "Demand_1": {"sigma": 0.15, "rho": 0.8, "seed": 1},
    "House_1": {"sigma": 0.15, "rho": 0.8, "seed": 2},
}
 
# Conversion from kW to the volume unit of the scenario (MW)
KW_TO_SCENARIO_UNIT = 1 / 1000
 
N_WEEKS_FORECAST = 4
 
# Target files and columns of the scenario inputs
FORECAST_TARGETS: dict[str, tuple[str, str]] = {
    "Demand_1": ("demand_df.csv", "Demand_1"),
    "House_1": ("forecasts_df.csv", "House_1_load_profile"),
}
REALISED_TARGETS: dict[str, tuple[str, str]] = {
    "Demand_1": ("actuals_df.csv", "demand_actual_Demand_1"),
    "House_1": ("actuals_df.csv", "load_actual_House_1"),
}
 
# Previous versions of the input files are archived here before they are modified
ARCHIVE_PATH = SCENARIO_PATH / "input_history"
 
# BDEW table layout (0-based indices of the profile sheets)
ROW_MONTH = 2
ROW_DAY_TYPE = 3
ROWS_VALUES = slice(4, 100)
FIRST_VALUE_COLUMN = 2
N_QUARTER_HOURS = 96
N_PROFILE_COLUMNS = 36  # 12 months x 3 day types
 
 
def load_h25_raw(
    path: Path, sheet_name: str = RAW_SHEET
) -> dict[tuple[int, str], np.ndarray]:
    """
    Load the BDEW 2025 standard load profile table of one customer group.
 
    Layout of each profile sheet in the BDEW file (0-based row/column indices):
 
    - Row 2, columns 2..37: a date per column. Only the month is relevant
      (the year 2012 is a placeholder).
    - Row 3, columns 2..37: the day type per column, one of ``SA`` (Saturday),
      ``FT`` (Sunday and public holiday) and ``WT`` (working day).
    - Rows 4..99, columns 2..37: 96 quarter-hourly values in kWh per quarter hour,
      normalised to an annual consumption of 1,000,000 kWh.
 
    Args:
        path (Path): Path to the BDEW Excel file.
        sheet_name (str, optional): Profile sheet to read. Defaults to ``"H25"``.
 
    Returns:
        dict[tuple[int, str], np.ndarray]: Mapping from ``(month, day_type)`` to an
        array of 96 quarter-hourly values in kWh. The mapping has 36 entries.
 
    Raises:
        FileNotFoundError: If the Excel file does not exist.
        ValueError: If the sheet does not contain 12 months x 3 day types.
    """
    # 1. Read the sheet without interpreting any row as header
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
 
    # 2. Extract month and day type of each column
    months = pd.to_datetime(raw.iloc[ROW_MONTH, FIRST_VALUE_COLUMN:]).dt.month
    day_types = raw.iloc[ROW_DAY_TYPE, FIRST_VALUE_COLUMN:]
 
    # 3. Extract the 96 quarter-hourly values of each column
    values = raw.iloc[ROWS_VALUES, FIRST_VALUE_COLUMN:].to_numpy(dtype=float)
 
    # 4. Map each (month, day type) combination to its 96 values
    profile_table = {
        (int(month), str(day_type)): values[:, col]
        for col, (month, day_type) in enumerate(zip(months, day_types))
    }
 
    if len(profile_table) != N_PROFILE_COLUMNS or values.shape[0] != N_QUARTER_HOURS:
        raise ValueError(
            f"Unexpected layout in sheet '{sheet_name}': found {len(profile_table)} "
            f"(month, day type) combinations and {values.shape[0]} quarter hours, "
            f"expected {N_PROFILE_COLUMNS} and {N_QUARTER_HOURS}."
        )
 
    return profile_table
 
 
def get_day_type(day: pd.Timestamp, public_holidays: holidays.HolidayBase) -> str:
    """
    Classify a calendar day into the BDEW day types.
 
    Args:
        day (pd.Timestamp): Calendar day.
        public_holidays (holidays.HolidayBase): Holiday calendar, e.g.
            ``holidays.Germany(subdiv="NW", years=[...])``.
 
    Returns:
        str: ``"FT"`` for Sundays and public holidays, ``"SA"`` for Saturdays and
        ``"WT"`` otherwise.
 
    Note:
        The treatment of 24.12 and 31.12 is defined in the BDEW application guide
        for the 2025 profiles and has to be checked there before finalising.
    """
    # Public holidays are checked first, so a holiday on a Saturday counts as FT
    if day in public_holidays or day.dayofweek == 6:
        return "FT"
    if day.dayofweek == 5:
        return "SA"
    return "WT"
 
 
def dynamisation_factor(day_of_year: int) -> float:
    """
    Compute the BDEW dynamisation factor for a given day of the year.
 
    The factor is ``-3.92e-10 t^4 + 3.20e-7 t^3 - 7.02e-5 t^2 + 2.10e-3 t + 1.24``
    with ``t`` the day of the year (1 on 1 January).
 
    Args:
        day_of_year (int): Day of the year of the calendar day, starting at 1.
 
    Returns:
        float: Multiplicative factor applied to all 96 quarter-hourly values of the day.
 
    Note:
        According to the BDEW file the dynamisation function must be applied to
        H25, P25 and S25, and must not be applied to G25 and L25.
    """
    t = day_of_year
    return -3.92e-10 * t**4 + 3.20e-7 * t**3 - 7.02e-5 * t**2 + 2.10e-3 * t + 1.24
 
 
def build_shape_series(
    profile_table: dict[tuple[int, str], np.ndarray],
    start: str,
    end: str,
) -> pd.Series:
    """
    Expand the BDEW profile table onto the calendar and aggregate to hourly resolution.
 
    Steps:
        1. **Iterate over calendar days** from ``start`` to ``end``.
        2. **Select the 96 values** of the day via its month and day type
           (:func:`get_day_type`).
        3. **Apply the dynamisation factor** of the day (:func:`dynamisation_factor`),
           using the day of the year of that calendar day.
        4. **Concatenate** the days to a quarter-hourly series and aggregate with
           ``resample("1h").sum()``.
 
    Args:
        profile_table (dict[tuple[int, str], np.ndarray]): Output of :func:`load_h25_raw`.
        start (str): First timestamp of the output series.
        end (str): Last timestamp of the output series.
 
    Returns:
        pd.Series: Hourly energy in kWh per hour (numerically equal to the mean power
        in kW) with a timezone-naive DatetimeIndex from ``start`` to ``end``.
 
    Note:
        For the calendar year 2019 the sum of the returned series is approximately
        1,000,000 kWh. Reference values for self-checking (kWh per hour, before scaling):
        2019-01-01 12:00 (FT) ~ 205.19, 2019-01-02 12:00 (WT) ~ 130.58,
        2019-01-05 12:00 (SA) ~ 188.71, 2019-07-01 12:00 (WT) ~ 104.51.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    days = pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D")
    public_holidays = holidays.Germany(
        subdiv=HOLIDAY_SUBDIVISION, years=sorted(set(days.year))
    )
 
    # 1. - 3. Look up the 96 values of each day and apply the dynamisation factor
    daily_values = [
        profile_table[(day.month, get_day_type(day, public_holidays))]
        * dynamisation_factor(day.dayofyear)
        for day in days
    ]
 
    # 4. Concatenate to a quarter-hourly series and aggregate to hourly energy
    quarter_hourly = pd.Series(
        np.concatenate(daily_values),
        index=pd.date_range(days[0], periods=len(days) * N_QUARTER_HOURS, freq="15min"),
    )
    hourly = quarter_hourly.resample("1h").sum()
 
    return hourly.loc[start_ts:end_ts]
 
 
def scale_to_annual_consumption(shape: pd.Series, annual_kwh: float) -> pd.Series:
    """
    Scale a load shape so that the simulation year matches a given annual consumption.
 
    The scaling factor is computed on the simulation year only and applied to the
    whole series, so the normalisation of the raw BDEW data does not matter.
 
    Args:
        shape (pd.Series): Hourly load shape in arbitrary units.
        annual_kwh (float): Target annual consumption in kWh.
 
    Returns:
        pd.Series: Hourly load in kW. At hourly resolution the sum over the
        simulation year equals ``annual_kwh``.
    """
    factor = annual_kwh / shape.loc[SIMULATION_START:SIMULATION_END].sum()
    return shape * factor
 
 
def realise_load(
    expected_load: pd.Series,
    sigma: float,
    rho: float,
    seed: int,
) -> pd.Series:
    """
    Generate a realised load by adding multiplicative AR(1) noise to the expected load.
 
    The realised load is computed as ``expected_load * (1 + eps)`` with
    ``eps[i] = rho * eps[i - 1] + sqrt(1 - rho**2) * z[i]``, ``z ~ N(0, 1)``,
    rescaled to a standard deviation of ``sigma``. Negative values are clipped to zero.
 
    Args:
        expected_load (pd.Series): Hourly expected load in kW.
        sigma (float): Standard deviation of the relative noise.
        rho (float): Lag-one autocorrelation of the noise.
        seed (int): Seed of the random number generator.
 
    Returns:
        pd.Series: Hourly realised load in kW with the same index as ``expected_load``.
    """
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal(len(expected_load))
 
    # 1. Build the AR(1) process step by step (each value depends on the previous one)
    eps = np.zeros(len(expected_load))
    for i in range(1, len(eps)):
        eps[i] = rho * eps[i - 1] + np.sqrt(1 - rho**2) * innovations[i]
 
    # 2. Rescale to the target standard deviation
    eps = eps / eps.std() * sigma
 
    # 3. Apply as relative deviation and clip, since load cannot be negative
    realised = expected_load * (1 + eps)
    return realised.clip(lower=0)
 
 
def forecast_same_weekday(
    realised_load: pd.Series,
    n_weeks: int = N_WEEKS_FORECAST,
) -> pd.Series:
    """
    Forecast load as the mean of the same weekday and hour of the previous weeks.
 
    The forecast for timestamp ``t`` is the mean of the realised load at
    ``t - 7 d``, ``t - 14 d``, ..., ``t - 7 * n_weeks d``. Since the smallest lag is
    one week, the forecast only uses information available at day-ahead gate closure.
 
    Args:
        realised_load (pd.Series): Hourly realised load in kW.
        n_weeks (int, optional): Number of previous weeks to average. Defaults to 4.
 
    Returns:
        pd.Series: Hourly load forecast in kW with the same index as ``realised_load``.
        The first ``n_weeks`` weeks are NaN.
    """
    # Shift by time (not by row count), so gaps in the index cannot misalign values
    lagged = [
        realised_load.shift(freq=pd.Timedelta(weeks=k)).reindex(realised_load.index)
        for k in range(1, n_weeks + 1)
    ]
    return pd.concat(lagged, axis=1).mean(axis=1, skipna=False)
 
 
def _archive_input_file(file_path: Path, timestamp: str) -> Path:
    """
    Copy an input file into the archive folder before it is modified.
 
    Previous versions are never deleted, so every earlier state of the scenario
    inputs can be restored and compared.
 
    Args:
        file_path (Path): Path to the input file that will be modified.
        timestamp (str): Timestamp of the current run, used in the archive file name.
 
    Returns:
        Path: Path to the archived copy.
    """
    ARCHIVE_PATH.mkdir(exist_ok=True)
    archived = ARCHIVE_PATH / f"{file_path.stem}_{timestamp}{file_path.suffix}"
    shutil.copy2(file_path, archived)
    logger.info(f"Archived {file_path.name} as {archived.name}")
    return archived
 
 
def _write_column(file_path: Path, column: str, series: pd.Series) -> None:
    """
    Write one series into one column of a scenario input file by index alignment.
 
    Args:
        file_path (Path): Path to the CSV file with a ``datetime`` column.
        column (str): Name of the column to create or overwrite.
        series (pd.Series): Values to write, indexed by timestamp.
 
    Raises:
        ValueError: If the file contains timestamps that are not covered by ``series``.
 
    Note:
        Existing columns keep their names because ASSUME reads fixed column names.
        The previous file content is preserved by :func:`_archive_input_file`.
    """
    df = pd.read_csv(file_path, parse_dates=["datetime"], index_col="datetime")
    df[column] = series
 
    n_missing = int(df[column].isna().sum())
    if n_missing > 0:
        raise ValueError(
            f"{n_missing} timestamps of {file_path.name} are not covered by the "
            f"series for column '{column}'. The file was not written."
        )
 
    df.to_csv(file_path)
    logger.info(f"Wrote column '{column}' to {file_path.name} ({len(df)} rows)")
 
 
def write_scenario_inputs(
    realised_load: dict[str, pd.Series],
    forecast_load: dict[str, pd.Series],
) -> None:
    """
    Write forecast and realised load into the ASSUME scenario input files.
 
    ================  ==================  ==========================
    Series            File                Column
    ================  ==================  ==========================
    Demand_1 forecast demand_df.csv       Demand_1 (positive values)
    House_1 forecast  forecasts_df.csv    House_1_load_profile
    Demand_1 realised actuals_df.csv      demand_actual_Demand_1
    House_1 realised  actuals_df.csv      load_actual_House_1
    ================  ==================  ==========================
 
    Args:
        realised_load (dict[str, pd.Series]): Realised load per unit in scenario units.
        forecast_load (dict[str, pd.Series]): Forecast load per unit in scenario units.
 
    Raises:
        ValueError: If a column contains NaN after alignment.
 
    Note:
        ``actuals_df.csv`` already contains ``pv_actual_House_1``. Columns are added,
        the file is not overwritten. Every touched file is archived once per run
        before it is modified (see :func:`_archive_input_file`).
    """
    # 1. Archive each touched file once, before any modification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    touched_files = {
        file_name
        for targets in (FORECAST_TARGETS, REALISED_TARGETS)
        for file_name, _ in targets.values()
    }
    for file_name in sorted(touched_files):
        _archive_input_file(SCENARIO_PATH / file_name, timestamp)
 
    # 2. Write forecast and realised series
    for unit_id, (file_name, column) in FORECAST_TARGETS.items():
        _write_column(SCENARIO_PATH / file_name, column, forecast_load[unit_id])
 
    for unit_id, (file_name, column) in REALISED_TARGETS.items():
        _write_column(SCENARIO_PATH / file_name, column, realised_load[unit_id])
 
 
def log_forecast_quality(
    realised_load: pd.Series,
    forecast_load: pd.Series,
    unit_id: str,
) -> None:
    """
    Log consistency checks and the forecast error of one unit for the simulation year.
 
    Logged values:
        - Annual consumption in kWh (should match ``ANNUAL_CONSUMPTION_KWH``).
        - Number of NaN in the forecast (should be zero).
        - nRMSE, defined as RMSE divided by the mean realised load.
 
    Args:
        realised_load (pd.Series): Hourly realised load in kW for the simulation year.
        forecast_load (pd.Series): Hourly forecast load in kW for the simulation year.
        unit_id (str): Identifier of the unit, used in the log messages.
    """
    annual_kwh = realised_load.sum()
    n_nan = int(forecast_load.isna().sum())
    rmse = np.sqrt(((forecast_load - realised_load) ** 2).mean())
    nrmse = rmse / realised_load.mean()
 
    logger.info(
        f"{unit_id}: realised annual consumption {annual_kwh:.0f} kWh "
        f"(target {ANNUAL_CONSUMPTION_KWH[unit_id]:.0f} kWh)"
    )
    logger.info(f"{unit_id}: NaN in forecast: {n_nan}")
    logger.info(f"{unit_id}: nRMSE (RMSE / mean realised load): {nrmse:.3f}")
 
    if n_nan > 0:
        logger.warning(f"{unit_id}: forecast contains NaN, check HISTORY_START")
 
 
def main() -> None:
    """Run the load forecast pipeline for all units and write the scenario inputs."""
    # 1. Read and expand the standard load profile
    profile_table = load_h25_raw(RAW_FILE)
    shape = build_shape_series(profile_table, HISTORY_START, SIMULATION_END)
 
    realised_scenario_unit: dict[str, pd.Series] = {}
    forecast_scenario_unit: dict[str, pd.Series] = {}
 
    for unit_id, annual_kwh in ANNUAL_CONSUMPTION_KWH.items():
        # 2. Scale, realise and forecast the load of each unit
        expected_load = scale_to_annual_consumption(shape, annual_kwh)
        realised_load = realise_load(expected_load, **NOISE_PARAMS[unit_id])
        forecast_load = forecast_same_weekday(realised_load)
 
        # 3. Restrict to the simulation year and check the forecast quality
        realised_year = realised_load.loc[SIMULATION_START:SIMULATION_END]
        forecast_year = forecast_load.loc[SIMULATION_START:SIMULATION_END]
        log_forecast_quality(realised_year, forecast_year, unit_id)
 
        # 4. Convert from kW to the volume unit of the scenario
        realised_scenario_unit[unit_id] = realised_year * KW_TO_SCENARIO_UNIT
        forecast_scenario_unit[unit_id] = forecast_year * KW_TO_SCENARIO_UNIT
 
    # 5. Write the scenario input files
    write_scenario_inputs(realised_scenario_unit, forecast_scenario_unit)
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()