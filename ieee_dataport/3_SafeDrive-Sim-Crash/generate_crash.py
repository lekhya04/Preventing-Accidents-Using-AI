#!/usr/bin/env python3
"""
SafeDrive-Sim-Crash  --  SYNTHETIC road-crash records generator (v1.0)

*** ALL DATA PRODUCED BY THIS SCRIPT IS SIMULATED. THESE ARE NOT REAL CRASHES. ***
Locations are random points inside a bounding box around Coimbatore, Tamil Nadu (India). They do NOT
correspond to real accident black spots, and no real accident record was used.

One row = one simulated crash with context (when / where / weather / road / vehicle) and outcomes.

Columns
  crash_id, date, hour, weekday, latitude, longitude,
  road_type        Expressway/Bypass | Highway/Arterial | Local street
  traffic_signal   True if the crash location is a signalised junction
  light_condition  Day | Night
  weather          Clear, Partly Cloudy, Overcast, Haze, Fog, Drizzle, Light Rain, Rain, Heavy Rain, Thunderstorm
  temperature_c, visibility_km, wind_direction (16-point compass or CALM)
  speed_limit_kmh, primary_vehicle (two_wheeler, car, auto_rickshaw, bus, goods_vehicle)
  severity         1 = property damage only, 2 = minor injury, 3 = grievous injury, 4 = fatal
  road_affected_km length of road affected (0 = point incident)
  clearance_time_min  minutes until the road was cleared

How severity is generated (stylised assumptions, NOT calibrated to real statistics)
  latent risk = vehicle type + speed limit + night + weather + low visibility - signal + hotspot effect
  + logistic noise (scale 0.6, so outcomes stay uncertain); the latent score is then cut into 4 classes
  at fixed quantiles (32 / 38 / 22 / 8 %).
Seasonality (monsoon rain, winter fog) and daily temperature cycle are also stylised.
The file contains crashes only (no non-crash exposure), so it supports severity / outcome modelling,
NOT estimating how likely a crash is.

Usage:  python dataset_generators/generate_crash.py [--seed 2026] [--n 30000]
"""
import os, argparse
import numpy as np
import pandas as pd

GENERATOR_VERSION = "1.0"
DATASET_NAME = "SafeDrive-Sim-Crash"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "crash", "safedrive_sim_crash.csv")
BOX = (10.93, 11.09, 76.90, 77.06)          # lat_min, lat_max, lng_min, lng_max

WEATHERS = ["Clear", "Partly Cloudy", "Overcast", "Haze", "Fog", "Drizzle", "Light Rain", "Rain", "Heavy Rain", "Thunderstorm"]
RAIN_P = [0.04, 0.04, 0.06, 0.12, 0.15, 0.18, 0.20, 0.18, 0.20, 0.35, 0.38, 0.15]     # monthly chance of rain
MONTH_T = [25, 26.5, 28.5, 29.5, 29.5, 27, 26, 26.5, 27, 26.5, 25.5, 24.5]            # monthly mean temp (C)
HOUR_W = np.array([.3, .2, .15, .15, .25, .5, 1.3, 2.8, 4.5, 4.2, 3.5, 3.2, 3.4, 3.3, 3.2, 3.6, 4.2, 5.2, 5.8, 5.0, 3.6, 2.3, 1.3, .7])
WEATHER_EFF = {"Clear": 0, "Partly Cloudy": 0, "Overcast": .08, "Haze": .3, "Fog": .8, "Drizzle": .25, "Light Rain": .4,
               "Rain": .6, "Heavy Rain": .85, "Thunderstorm": .7}
VIS_BASE = {"Clear": 9, "Partly Cloudy": 9, "Overcast": 8, "Haze": 4, "Fog": 1.2, "Drizzle": 6, "Light Rain": 5,
            "Rain": 3, "Heavy Rain": 1.5, "Thunderstorm": 2.5}
VEH = ["two_wheeler", "car", "auto_rickshaw", "bus", "goods_vehicle"]
VEH_P = {"Expressway/Bypass": [.30, .28, .02, .10, .30], "Highway/Arterial": [.50, .22, .07, .07, .14], "Local street": [.60, .18, .12, .03, .07]}
VEH_EFF = {"two_wheeler": 1.0, "car": 0.0, "auto_rickshaw": .35, "bus": .75, "goods_vehicle": .90}
SPEEDS = {"Expressway/Bypass": [60, 80], "Highway/Arterial": [40, 50, 60], "Local street": [20, 30, 40]}
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
HOTSPOT_TYPES = {  # road_type probs, P(signal), risk effect
    "arterial_junction": ([0.05, 0.70, 0.25], 0.70, 0.05),
    "bypass_stretch": ([0.80, 0.20, 0.00], 0.03, 0.35),
    "market_area": ([0.00, 0.15, 0.85], 0.20, 0.00)}
ROADS = ["Expressway/Bypass", "Highway/Arterial", "Local street"]


def make_hotspots(r, k=16):
    la0, la1, lo0, lo1 = BOX
    kinds = r.choice(list(HOTSPOT_TYPES), k, p=[0.4, 0.25, 0.35])
    return pd.DataFrame({"lat": r.uniform(la0 + 0.015, la1 - 0.015, k), "lng": r.uniform(lo0 + 0.015, lo1 - 0.015, k),
                         "sigma": r.uniform(0.004, 0.012, k), "weight": r.lognormal(0, 0.8, k),
                         "kind": kinds, "offset": r.normal(0, 0.25, k)})


def generate(seed=2026, n=30000):
    r = np.random.default_rng(seed)
    la0, la1, lo0, lo1 = BOX
    hs = make_hotspots(r)

    # ---- where / which road
    in_hot = r.random(n) < 0.72
    hid = r.choice(len(hs), n, p=hs.weight / hs.weight.sum())
    lat = np.where(in_hot, hs.lat.values[hid] + r.normal(0, 1, n) * hs.sigma.values[hid], r.uniform(la0, la1, n))
    lng = np.where(in_hot, hs.lng.values[hid] + r.normal(0, 1, n) * hs.sigma.values[hid], r.uniform(lo0, lo1, n))
    lat, lng = lat.clip(la0, la1), lng.clip(lo0, lo1)
    road, signal, hot_eff = [], [], np.zeros(n)
    for i in range(n):
        if in_hot[i]:
            probs, ps, eff = HOTSPOT_TYPES[hs.kind.values[hid[i]]]
            road.append(ROADS[r.choice(3, p=probs)]); signal.append(r.random() < ps)
            hot_eff[i] = eff + hs.offset.values[hid[i]]
        else:
            road.append(ROADS[r.choice(3, p=[0.10, 0.30, 0.60])]); signal.append(r.random() < 0.15)
    road, signal = np.array(road), np.array(signal)

    # ---- when
    days = r.integers(0, (pd.Timestamp("2025-12-31") - pd.Timestamp("2022-01-01")).days + 1, n)
    date = pd.Timestamp("2022-01-01") + pd.to_timedelta(days, unit="D")
    hour = r.choice(24, n, p=HOUR_W / HOUR_W.sum())
    month = date.month.values - 1
    night = ~((hour >= 6) & (hour < 18))

    # ---- weather
    weather = []
    for m, h in zip(month, hour):
        if r.random() < RAIN_P[m]:
            weather.append(r.choice(WEATHERS[5:], p=[.25, .35, .25, .10, .05]))
        elif m in (11, 0, 1) and 4 <= h <= 8 and r.random() < 0.15:
            weather.append("Fog")
        elif r.random() < 0.05:
            weather.append("Haze")
        else:
            weather.append(r.choice(["Clear", "Partly Cloudy", "Overcast"], p=[.5, .3, .2]))
    weather = np.array(weather)
    rainy = np.isin(weather, ["Drizzle", "Light Rain", "Rain", "Heavy Rain", "Thunderstorm"])
    temp = (np.array(MONTH_T)[month] + 4 * np.sin(2 * np.pi * (hour - 9) / 24) - 2.5 * rainy + r.normal(0, 1.2, n)).round(1)
    vis = (np.array([VIS_BASE[w] for w in weather]) * r.lognormal(0, 0.2, n)).clip(0.1, 10).round(1)
    summer = np.isin(month, [5, 6, 7, 8]); winter = np.isin(month, [11, 0, 1])
    wind = []
    for s, w in zip(summer, winter):
        if r.random() < 0.10: wind.append("CALM")
        elif s: wind.append(r.choice(["W", "WSW", "SW", "WNW"], p=[.3, .3, .25, .15]))
        elif w: wind.append(r.choice(["NE", "ENE", "E", "NNE"], p=[.35, .25, .25, .15]))
        else: wind.append(r.choice(COMPASS))

    # ---- vehicle / speed limit
    veh = np.array([r.choice(VEH, p=VEH_P[rt]) for rt in road])
    spd = np.array([r.choice(SPEEDS[rt]) for rt in road])

    # ---- severity: latent score + logistic noise, cut at fixed quantiles
    z = (np.array([VEH_EFF[v] for v in veh]) + 0.035 * (spd - 40) + 0.7 * night
         + np.array([WEATHER_EFF[w] for w in weather]) + 0.15 * np.clip(3 - vis, 0, None)
         - 0.4 * signal + 0.25 * night * (~signal) + 1.5 * hot_eff)
    ystar = z + r.logistic(0, 0.6, n)
    cuts = np.quantile(ystar, [0.32, 0.70, 0.92])
    sev = 1 + (ystar > cuts[0]).astype(int) + (ystar > cuts[1]) + (ystar > cuts[2])

    # ---- outcomes
    p_point = np.clip(0.45 - 0.08 * (sev - 1), 0.1, 0.9)
    dist = np.where(r.random(n) < p_point, 0.0,
                    r.lognormal(np.log(0.4) + 0.25 * (sev - 1) + 0.4 * (road == "Expressway/Bypass"), 0.7, n)).round(2)
    clear = (18 * 1.5 ** (sev - 1) * (1 + 0.15 * rainy) * r.lognormal(0, 0.45, n)).clip(5, 480).round(0).astype(int)

    return pd.DataFrame({
        "crash_id": [f"SSC-{i:06d}" for i in range(1, n + 1)], "date": date.strftime("%Y-%m-%d"), "hour": hour,
        "weekday": date.day_name().values, "latitude": lat.round(5), "longitude": lng.round(5),
        "road_type": road, "traffic_signal": signal, "light_condition": np.where(night, "Night", "Day"),
        "weather": weather, "temperature_c": temp, "visibility_km": vis, "wind_direction": wind,
        "speed_limit_kmh": spd, "primary_vehicle": veh, "severity": sev,
        "road_affected_km": dist, "clearance_time_min": clear})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    df = generate(a.seed, a.n)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{DATASET_NAME}: {len(df)} rows -> {a.out}")
    print("severity:", df.severity.value_counts(normalize=True).sort_index().round(3).to_dict())
    print("weather :", df.weather.value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
