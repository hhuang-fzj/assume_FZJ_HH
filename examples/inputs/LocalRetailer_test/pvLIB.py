import pandas as pd, pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain

from pathlib import Path
folder = Path(__file__).parent
 
# 1. Location
loc = Location(latitude=50.78, longitude=6.08, tz="Europe/Berlin", altitude=180)
 
# 2. System (8.1 kWp, facing south, tilt 30 degrees)
system = PVSystem(
    surface_tilt=30, surface_azimuth=180,          # south = 180
    module_parameters={"pdc0": 8100, "gamma_pdc": -0.004},
    inverter_parameters={"pdc0": 8100},
    temperature_model_parameters=pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"],
)
mc = ModelChain.with_pvwatts(system, loc)
 
# 3. Fetch weather data (returns 2 values)
raw, meta = pvlib.iotools.get_pvgis_hourly(
    latitude=50.78, longitude=6.08, start=2018, end=2019,   # one extra year of history
    surface_tilt=30, surface_azimuth=180, map_variables=True)
 
# 4. PVGIS returns plane-of-array components, assemble the three columns
#    required by run_model_from_poa
weather = pd.DataFrame({
    "poa_global":  raw[["poa_direct", "poa_sky_diffuse", "poa_ground_diffuse"]].sum(axis=1),
    "poa_direct":  raw["poa_direct"],
    "poa_diffuse": raw["poa_sky_diffuse"] + raw["poa_ground_diffuse"],
    "temp_air":    raw["temp_air"],
    "wind_speed":  raw["wind_speed"],
})
 
# 5. Run the model
mc.run_model_from_poa(weather)
 
# 6. Convert to the 0-1 capacity factor required by ASSUME
cf = (mc.results.ac / 8100).clip(lower=0)
cf.index = cf.index.tz_convert("Etc/GMT-1").tz_localize(None)
 
cf.index = cf.index.floor("h")   # align to full hours
 
out = pd.DataFrame({
    "datetime": cf.index,
    "pv_generation_kW": (cf * 8100 / 1000).round(4).values,   # p.u. times capacity gives kW
    "pv_pu": cf.round(4).values,
})
 
out = out.set_index("datetime")
out["pv_pu_forecast"] = out.groupby(out.index.hour)["pv_pu"].transform(
    lambda s: s.shift(2).rolling(7).mean()
)
out = out.loc["2019-01-01 00:00":"2019-12-31 23:00"].reset_index()
 
out.to_csv(folder / "pv_profile.csv", index=False)
print(out.head(12))
 
 
print("rows:", len(out))                                    # expected 8760
print("duplicate timestamps:", out["datetime"].duplicated().sum())   # expected 0
print("first row:", out["datetime"].iloc[0])                # expected xxxx-01-01 00:00
print("last row:", out["datetime"].iloc[-1])                # expected xxxx-12-31 23:00
print("annual yield kWh/kWp:", round(out["pv_generation_kW"].sum() / 8.1, 1))
 
 

 
# Read forecasts_df.csv and use the datetime column as index
fc = pd.read_csv(folder / "forecasts_df.csv", parse_dates=["datetime"], index_col="datetime")
 
# Write the PV forecast into House_1_pv_profile (replaces the previous values)
pv = out.set_index("datetime")["pv_pu_forecast"]
fc["House_1_pv_profile"] = pv      # pandas aligns by index (time), not by row number
 
# Check whether all timestamps are covered
n_missing = fc["House_1_pv_profile"].isna().sum()
print("uncovered rows:", n_missing)   # must be 0
 
if n_missing == 0:
    fc.to_csv(folder / "forecasts_df.csv")
    print("written to forecasts_df.csv")
 
    actuals_path = folder / "actuals_df.csv"
    pv_actual = out.set_index("datetime")["pv_pu"]
 
    if actuals_path.exists():
        ac = pd.read_csv(actuals_path, parse_dates=["datetime"], index_col="datetime")
        ac["pv_actual_House_1"] = pv_actual
    else:
        ac = pv_actual.rename("pv_actual_House_1").to_frame()
        ac.index.name = "datetime"
 
    if ac["pv_actual_House_1"].isna().sum() == 0:
        ac.to_csv(actuals_path)
        print("written to actuals_df.csv")
    else:
        print("timestamps do not match, actuals_df.csv was not written")
else:
    print("timestamps do not match, nothing was written. Check the year.")
 
print(out[["datetime", "pv_pu", "pv_pu_forecast"]].iloc[8:16])