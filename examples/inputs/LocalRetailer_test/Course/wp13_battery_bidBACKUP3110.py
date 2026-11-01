"""
WP13: bid of one battery in one delivery period (learning script, round 2).

It does not change ASSUME. It reproduces the hand calculation of WP13 in the course
document, so that the later ASSUME integration can be checked against it.

Steps for one delivery period t of day D:
    1. Price forecast for day D: naive similar-day method (rule R7).
    2. Reference price: mean of the 96 forecast values of day D (decision E4).
    3. Direction: discharge if forecast(t) >= reference price, else charge.
    4. Volume: what the state of charge allows (Eq. 20 / 21).
    5. Price: reference price, with or without efficiency (decision E1).
    6. Split: share BETA of the volume to the LEM, the rest to the WM (rule R8).

Sign convention (same as round 1): buy (charge) +, sell (discharge) -.

Sources:
    Rule: Qussous, R., Harder, N., Weidlich, A. (2022). Understanding Power Market
        Dynamics by Reflecting Market Interrelations and Flexibility-Oriented Bidding
        Strategies. Energies 15(2), 494, Sect. 3.2.3, Eq. (20)-(24), p. 12-13.
        No reserve market in this round, so the CRM terms of Eq. (20)/(21) are zero.
    ASSUME version of the price (with efficiency): assume/strategies/
        flexable_storage.py, lines 134-138 (commit b22d8b5).
    Naive forecast: Weron, R. (2014). Electricity price forecasting: A review of the
        state-of-the-art with a look into the future. International Journal of
        Forecasting 30(4), 1030-1081, Sect. 3.8.1, p. 1049-1050.
    Battery data: SimBench LV4.101, simbench_raw/Storage.csv (sR, eStore, etaStore).
    SOC limits 10-90 %: Wagner, L. P., Bisson, R., Gehlhoff, F. (2026). Market
        Strategy Evaluation for Prosumers in Local Electricity Markets.
        arXiv:2607.18272v1 (preprint), Sect. III.
    Initial SOC 50 %: Qussous et al. (2022), see above, Sect. 4.
    Mean over day D instead of the next 96 steps (Eq. 22): information boundary
        of the thesis [OWN], decision E4.
"""

import math
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Settings (decisions of WP13; change here if you decide differently)
# ---------------------------------------------------------------------------
PRICE_RULE = "assume"  # E1: "assume" (with efficiency) or "paper" (Eq. 24)
SOC_MIN_SHARE = 0.10  # E2: lower SOC limit as share of the capacity
SOC_MAX_SHARE = 0.90  # E2: upper SOC limit
SOC_INIT_SHARE = 0.50  # E3: initial SOC
BETA = 0.5  # D1: share of the volume bid in the LEM (scenarios 0 / 0.5 / 1)
WM_BUY_PRICE = 3000.0  # WM price cap: buy order always accepted
WM_SELL_PRICE = -500.0  # WM price floor: sell order always accepted
DT_H = 0.25  # length of one delivery period in hours

STORAGE_FILE = Path(__file__).resolve().parent.parent / "simbench_raw" / "Storage.csv"


def read_batteries(path: Path = STORAGE_FILE) -> pd.DataFrame:
    """
    Read the four SimBench batteries.

    Returns:
        pd.DataFrame: One row per battery with node, p_max_kw, e_max_kwh, eta.
        sR [MVA] and eStore [MWh] are converted to kW and kWh. etaStore is used as
        one-way efficiency for charging and discharging (meaning not yet checked in
        the SimBench documentation).
    """
    raw = pd.read_csv(path, sep=";")
    return pd.DataFrame(
        {
            "node": raw["node"],
            "p_max_kw": raw["sR"] * 1000,
            "e_max_kwh": raw["eStore"] * 1000,
            "eta": raw["etaStore"],
        }
    )


def naive_forecast(prices: pd.Series, day: str) -> pd.Series:
    """
    Naive similar-day forecast of the 96 prices of one day.

    Monday, Saturday, Sunday: same weekday of the previous week.
    Tuesday to Friday: previous day.
    (Weron 2014, p. 1049-1050, method of Nogales et al. 2002.)

    Args:
        prices (pd.Series): Known prices on a 15-minute index (EUR/MWh).
        day (str): Delivery day, e.g. "2016-06-21".

    Returns:
        pd.Series: Forecast on the 15-minute index of the delivery day.
    """
    day_start = pd.Timestamp(day)
    lag_days = 7 if day_start.dayofweek in (0, 5, 6) else 1  # Mon=0, Sat=5, Sun=6
    source_start = day_start - pd.Timedelta(days=lag_days)
    source = prices[source_start : source_start + pd.Timedelta("23h45min")]
    if len(source) != 96:
        raise ValueError(f"no complete price history for {source_start.date()}")
    return pd.Series(source.to_numpy(), index=source.index + pd.Timedelta(days=lag_days))


def battery_bid(soc_kwh, p_max_kw, e_max_kwh, eta, forecast_t, reference_price):
    """
    Volume and price of the battery for one delivery period.

    Args:
        soc_kwh (float): State of charge at the start of the period.
        p_max_kw (float): Maximum charge and discharge power.
        e_max_kwh (float): Capacity.
        eta (float): One-way efficiency (charge = discharge).
        forecast_t (float): Price forecast of this period (EUR/MWh).
        reference_price (float): Mean forecast price of the day (EUR/MWh).

    Returns:
        tuple: (volume_kw, price) with charge +, discharge -.
    """
    soc_min = SOC_MIN_SHARE * e_max_kwh
    soc_max = SOC_MAX_SHARE * e_max_kwh

    if forecast_t >= reference_price:  # Eq. (23): discharge
        p_sup = min((soc_kwh - soc_min) / DT_H * eta, p_max_kw)  # Eq. (20)
        volume_kw = -max(p_sup, 0.0)
        price = reference_price / eta if PRICE_RULE == "assume" else reference_price
    else:  # charge
        p_dem = min((soc_max - soc_kwh) / DT_H / eta, p_max_kw)  # Eq. (21)
        volume_kw = max(p_dem, 0.0)
        price = reference_price * eta if PRICE_RULE == "assume" else reference_price
    return volume_kw, price  # Eq. (24) / ASSUME lines 135, 138


def split_bid(volume_kw, price, beta=BETA):
    """
    Split the bid between LEM and WM (rule R8).

    Returns:
        dict: LEM volume and price; WM volume and price (WM at the price limit,
        so the WM part is always accepted).
    """
    wm_price = WM_BUY_PRICE if volume_kw > 0 else WM_SELL_PRICE
    return {
        "lem_kw": beta * volume_kw,
        "lem_price": price,
        "wm_kw": (1 - beta) * volume_kw,
        "wm_price": wm_price,
    }


def soc_after(soc_kwh, accepted_kw, eta):
    """SOC after one period: charging stores eta * energy, discharging takes energy / eta."""
    if accepted_kw >= 0:
        return soc_kwh + eta * accepted_kw * DT_H
    return soc_kwh + accepted_kw * DT_H / eta


def ok(name, value, expected, tol=1e-3):
    status = "OK" if math.isclose(value, expected, abs_tol=tol) else "FAIL"
    print(f"{status:4s} {name}: {value:.4f} (expected {expected:.4f})")


def check():
    """Reproduce the expected results of WP13, section 6."""
    from wp5_inputs import read_wm_price_15min  # SMARD prices of round 1

    index = pd.date_range("2016-06-13", "2016-06-22", freq="15min", inclusive="left")
    prices = read_wm_price_15min(index)
    forecast = naive_forecast(prices, "2016-06-21")
    reference = forecast.mean()
    ok("1 reference price", reference, 26.928)

    hourly = forecast.resample("1h").first()
    discharge = [t.hour for t, p in hourly.items() if p >= reference]
    print("2 discharge hours:", discharge)
    print("  expected:       ", [5, 6, 7, 8, 10, 16, 17, 18, 19, 20, 21])

    battery = read_batteries().iloc[0]  # battery 1, Bus 14
    soc0 = SOC_INIT_SHARE * battery.e_max_kwh
    v00, p00 = battery_bid(soc0, battery.p_max_kw, battery.e_max_kwh, battery.eta,
                           forecast["2016-06-21 00:00"], reference)
    v05, p05 = battery_bid(soc0, battery.p_max_kw, battery.e_max_kwh, battery.eta,
                           forecast["2016-06-21 05:00"], reference)
    ok("3 charge price (ASSUME)", p00, 25.582)
    ok("3 discharge price (ASSUME)", p05, 28.345)
    ok("4 charge volume 00:00 [kW]", v00, 13.7)
    ok("4 SOC at 00:15 [kWh]", soc_after(soc0, v00, battery.eta), 16.954)
    ok("4 discharge volume 05:00 [kW]", v05, -13.7)
    bid = split_bid(v00, p00)
    ok("5 LEM volume 00:00 [kW]", bid["lem_kw"], 6.85)
    ok("5 WM volume 00:00 [kW]", bid["wm_kw"], 6.85)
    print(f"     WM price {bid['wm_price']} (expected 3000)")
    print("6 09:00 forecast", forecast["2016-06-21 09:00"], "-> charge (expected),",
          "actual price", prices["2016-06-21 09:00"])


if __name__ == "__main__":
    check()
