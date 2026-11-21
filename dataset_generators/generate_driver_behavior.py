#!/usr/bin/env python3
"""
SafeDrive-Sim-IMU  --  SYNTHETIC smartphone-IMU driving-event dataset generator (v1.0)

*** ALL DATA PRODUCED BY THIS SCRIPT IS SIMULATED. NO REAL VEHICLE, DRIVER OR SENSOR WAS RECORDED. ***

What it simulates
  A phone rigidly mounted in a car, recording accelerometer, linear acceleration and gyroscope at
  ~50 Hz in a vehicle-aligned frame:  x = right,  y = forward,  z = up   (gyro z > 0 = turning left).

How it works (physics-inspired, deliberately imperfect)
  1. A target-speed profile (city / highway) and a driver "speed-keeping" controller give normal driving.
  2. Labelled events are injected as smooth pulses: braking, acceleration, turns, lane changes
     (aggressive AND gentle versions), plus road events: speed bump and pothole.
  3. Lateral acceleration and yaw rate are tied together physically (a_lat = v * yaw_rate).
  4. Body pitch / roll respond to longitudinal / lateral acceleration (gyro x / y).
  5. Realism layers: road vibration, engine tone, slow steering/pedal noise, sensor noise, per-trip
     gyro bias, per-trip phone-mount misalignment, timestamp jitter, dropped samples,
     per-driver aggressiveness, per-vehicle suspension, and label-boundary noise (+-0.15 s,
     mimicking human video annotation).
  6. Aggressive and gentle intensity ranges OVERLAP on purpose so classes are not trivially separable.

Usage
    python dataset_generators/generate_driver_behavior.py            # writes data/driver_behavior/trip_1..12
    python dataset_generators/generate_driver_behavior.py --seed 7   # different random world
"""
import os, json, argparse, shutil
from datetime import datetime, timedelta
from types import SimpleNamespace
import numpy as np
import pandas as pd

GENERATOR_VERSION = "1.0"
DATASET_NAME = "SafeDrive-Sim-IMU"
FS = 50.0
GRAV = 9.81

DRIVERS = {"D1": dict(gain=0.90, noise=0.12),     # smooth driver
           "D2": dict(gain=1.00, noise=0.18),     # average driver
           "D3": dict(gain=1.12, noise=0.25)}     # sporty driver
VEHICLES = {"hatchback": dict(pitch_k=0.012, roll_k=0.010, vib=1.15),
            "sedan":     dict(pitch_k=0.008, roll_k=0.007, vib=0.85)}

# number of labelled events per class over the whole dataset
QUOTA = {"non_aggressive": 40, "aggressive_braking": 18, "aggressive_acceleration": 18,
         "aggressive_left_turn": 18, "aggressive_right_turn": 18,
         "aggressive_left_lane_change": 18, "aggressive_right_lane_change": 18,
         "speed_bump": 18, "pothole": 18}

# trip plan: (driver, vehicle, road, surface)
PLAN = [("D1", "hatchback", "city", "dry"),     ("D2", "hatchback", "city", "dry"),
        ("D3", "hatchback", "highway", "dry"),  ("D1", "sedan", "city", "wet"),
        ("D2", "sedan", "highway", "dry"),      ("D3", "sedan", "city", "dry"),
        ("D1", "hatchback", "city", "dry"),     ("D2", "hatchback", "city", "wet"),
        ("D3", "hatchback", "highway", "dry"),  ("D1", "sedan", "highway", "dry"),
        ("D2", "sedan", "city", "dry"),         ("D3", "sedan", "city", "wet")]


# ----------------------------------------------------------------------------- signal helpers
def tukey(t, t0, dur, alpha=0.6):
    """Smooth 0 -> 1 -> 0 pulse with a flat top (raised-cosine edges)."""
    x = (t - t0) / dur
    y = np.zeros_like(t)
    inside = (x >= 0) & (x <= 1)
    xi = x[inside]
    w = np.ones_like(xi)
    r = xi < alpha / 2
    w[r] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (xi[r] - alpha / 2)))
    f = xi > 1 - alpha / 2
    w[f] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (xi[f] - 1 + alpha / 2)))
    y[inside] = w
    return y


def gauss(t, t0, sig):
    return np.exp(-0.5 * ((t - t0) / sig) ** 2)


def damped(t, t0, f, tau):
    """Damped sine starting at t0 (suspension / body ringing)."""
    s = np.zeros_like(t)
    m = t >= t0
    s[m] = np.exp(-(t[m] - t0) / tau) * np.sin(2 * np.pi * f * (t[m] - t0))
    return s


def bandnoise(rng, n, lo, hi, std):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    X[(f < lo) | (f > hi)] = 0
    y = np.fft.irfft(X, n)
    return y * (std / (y.std() + 1e-12))


def smooth(x, sec):
    k = max(1, int(sec * FS))
    return np.convolve(x, np.ones(k) / k, mode="same")


def rotation(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


# ----------------------------------------------------------------------------- event builders
def v_at(C, t0):
    return float(np.interp(t0, C.t, C.vt))


def ev_brake(C, t0, A, dur):
    dur = min(dur, max(1.4, 0.875 * v_at(C, t0) / A))      # never brake below ~zero speed
    p = tukey(C.t, t0, dur)
    C.ev_lon -= A * p
    C.active |= p > 0.01
    return t0, t0 + dur


def ev_accel(C, t0, A, dur):
    p = tukey(C.t, t0, dur)
    C.ev_lon += A * p
    C.active |= p > 0.01
    return t0, t0 + dur


def ev_turn(C, t0, A, dur, sign):          # sign: +1 right, -1 left (lateral accel toward the right is +)
    C.ev_lat += sign * A * tukey(C.t, t0, dur)
    return t0, t0 + dur


def ev_lane(C, t0, A, T, sign):            # S-shaped lateral manoeuvre, sign -1 = to the left
    x = (C.t - t0) / T
    m = (x >= 0) & (x <= 1)
    C.ev_lat[m] += sign * A * np.sin(2 * np.pi * x[m])
    return t0, t0 + T


def ev_speed_bump(C, t0):
    r = C.rng
    v = v_at(C, t0)
    Ab = r.uniform(0.8, 2.2)                                # driver slows before the bump
    p = tukey(C.t, t0, 1.8)
    C.ev_lon -= Ab * p
    C.active |= p > 0.01
    vb = max(v - Ab * 0.7 * 1.8, 2.5)
    tb = t0 + 2.0 + r.uniform(-0.2, 0.2)
    amp = max(r.normal(4.3, 1.1), 1.5) * np.clip(vb / 6, 0.6, 1.5)
    lag = 2.6 / vb                                          # wheelbase / speed: rear axle hits later
    pitch = r.uniform(0.15, 0.40)
    for tt, s in ((tb, 1.0), (tb + lag, 0.8)):
        C.ez += s * amp * (0.75 * damped(C.t, tt, 11, 0.07) + 0.5 * damped(C.t, tt, 1.8, 0.45))
        C.gx_ev += s * pitch * damped(C.t, tt, 1.8, 0.5)
    return t0 - 0.1, tb + lag + 1.3


def ev_pothole(C, t0):
    r = C.rng
    v = v_at(C, t0)
    amp = max(r.normal(5.2, 1.5), 1.8) * np.clip(v / 9, 0.6, 1.5)
    side = r.choice([-1, 1])                                # which side wheel drops in
    lag = 2.6 / max(v, 4)
    for tt, s in ((t0, 1.0), (t0 + lag, 0.7)):
        C.ez += s * amp * (-0.5 * gauss(C.t, tt, 0.05) + gauss(C.t, tt + 0.12, 0.04) + 0.3 * damped(C.t, tt + 0.12, 14, 0.1))
        C.ex += s * side * r.uniform(0.6, 1.6) * gauss(C.t, tt + 0.1, 0.06)
        C.gy_ev += s * side * r.uniform(0.2, 0.6) * damped(C.t, tt, 2.5, 0.25)
        C.gx_ev += s * r.uniform(-0.2, 0.2) * damped(C.t, tt, 2.0, 0.3)
    return t0 - 0.7, t0 + 1.6


def make_event(kind, t0, C):
    """Inject one labelled event; returns (label_start, label_end)."""
    r, g, wet = C.rng, C.gain, C.wet
    if kind == "aggressive_braking":
        return ev_brake(C, t0, float(np.clip(g * wet * r.normal(4.4, 0.9), 2.4, 7.0)), r.uniform(1.8, 3.2))
    if kind == "aggressive_acceleration":
        return ev_accel(C, t0, float(np.clip(g * wet * r.normal(3.0, 0.6), 1.8, 4.5)), r.uniform(2.2, 3.8))
    if kind in ("aggressive_left_turn", "aggressive_right_turn"):
        sign = -1 if "left" in kind else 1
        return ev_turn(C, t0, float(np.clip(g * r.normal(3.6, 0.8), 2.0, 5.5)), r.uniform(2.8, 4.5), sign)
    if kind in ("aggressive_left_lane_change", "aggressive_right_lane_change"):
        sign = -1 if "left" in kind else 1
        T, D = r.uniform(1.8, 3.0), r.uniform(3.0, 3.8)
        return ev_lane(C, t0, g * 2 * np.pi * D / T ** 2, T, sign)
    if kind == "speed_bump":
        return ev_speed_bump(C, t0)
    if kind == "pothole":
        return ev_pothole(C, t0)
    if kind == "non_aggressive":                            # gentle versions of ordinary manoeuvres
        sub = r.choice(["brake", "accel", "turn", "lane"])
        if sub == "brake":
            return ev_brake(C, t0, r.uniform(0.9, 2.3), r.uniform(2.5, 5.0))
        if sub == "accel":
            return ev_accel(C, t0, r.uniform(0.6, 1.6), r.uniform(3.0, 6.0))
        if sub == "turn":
            return ev_turn(C, t0, r.uniform(0.9, 2.1), r.uniform(3.0, 5.5), r.choice([-1, 1]))
        T, D = r.uniform(4.0, 6.5), r.uniform(2.8, 3.6)
        return ev_lane(C, t0, 2 * np.pi * D / T ** 2, T, r.choice([-1, 1]))
    raise ValueError(kind)


def make_background(C, t0):
    """Very mild UNLABELLED manoeuvre so 'normal cruising' is not perfectly flat."""
    r = C.rng
    s = r.choice(["brake", "accel", "turn"])
    if s == "brake":
        ev_brake(C, t0, r.uniform(0.3, 0.9), r.uniform(2.0, 4.0))
    elif s == "accel":
        ev_accel(C, t0, r.uniform(0.3, 0.8), r.uniform(2.0, 4.0))
    else:
        ev_turn(C, t0, r.uniform(0.4, 1.0), r.uniform(2.5, 4.5), r.choice([-1, 1]))
    return t0, t0 + 4.5


# ----------------------------------------------------------------------------- one trip
def simulate_trip(seed, trip_id, kinds, driver, vehicle, road, surface):
    rng = np.random.default_rng([seed, trip_id])
    duration = float(rng.uniform(540, 660))
    n = int(duration * FS)
    t = np.arange(n) / FS
    d, veh = DRIVERS[driver], VEHICLES[vehicle]
    rough = rng.uniform(1.0, 1.6) if road == "city" else rng.uniform(0.7, 1.0)

    # target speed profile (m/s)
    base = 9.5 if road == "city" else 17.0
    knots = np.arange(0, duration + 60, 30.0)
    vt = np.interp(t, knots, base * rng.uniform(0.65, 1.25, len(knots)))
    vt = smooth(vt, 6.0)

    C = SimpleNamespace(t=t, rng=rng, vt=vt, gain=d["gain"], wet=0.9 if surface == "wet" else 1.0,
                        ev_lon=np.zeros(n), ev_lat=np.zeros(n), ez=np.zeros(n), ex=np.zeros(n),
                        gx_ev=np.zeros(n), gy_ev=np.zeros(n), active=np.zeros(n, bool))

    # schedule labelled events in slots so they never overlap
    lo, hi = 14.0, duration - 14.0
    slot = (hi - lo) / len(kinds)
    events = []
    for i, kind in enumerate(rng.permutation(kinds)):
        t0 = lo + i * slot + rng.uniform(1.0, slot - 9.0)
        a, b = make_event(kind, t0, C)
        events.append((kind, a + rng.normal(0, 0.15), b + rng.normal(0, 0.15)))   # annotation noise
    # unlabelled mild background manoeuvres, kept out of labelled windows
    placed = 0
    for _ in range(200):
        if placed >= 6:
            break
        t0 = rng.uniform(8, duration - 14)
        if all(t0 + 4.5 < e[1] - 1 or t0 > e[2] + 1 for e in events):
            make_background(C, t0)
            placed += 1

    # longitudinal dynamics (driver keeps target speed except during events)
    steer_lon = bandnoise(rng, n, 0.05, 0.6, d["noise"])
    v = np.zeros(n)
    a_lon = np.zeros(n)
    v[0] = vt[0]
    for i in range(1, n):
        ctrl = 0.0 if C.active[i - 1] else float(np.clip(0.35 * (vt[i - 1] - v[i - 1]), -0.8, 1.2))
        a = ctrl + steer_lon[i - 1] + C.ev_lon[i - 1]
        v[i] = max(1.5, v[i - 1] + a / FS)
        a_lon[i - 1] = (v[i] - v[i - 1]) * FS
    a_lon[-1] = a_lon[-2]

    lat_noise = bandnoise(rng, n, 0.05, 0.8, 0.6 * d["noise"])
    a_lat = C.ev_lat + lat_noise                                  # +x = right
    omega = -a_lat / np.maximum(v, 4.0)                           # a_lat = -v * yaw_rate  (left turn: yaw > 0)

    # body attitude response
    theta = veh["pitch_k"] * smooth(a_lon, 0.4)
    phi = -veh["roll_k"] * smooth(a_lat, 0.4)
    gx = np.gradient(theta) * FS + C.gx_ev
    gy = np.gradient(phi) * FS + C.gy_ev
    gz = omega

    # vibration + engine tone
    vib_sd = veh["vib"] * rough * (0.10 + 0.012 * v)
    vib_z = bandnoise(rng, n, 2, 22, 1.0) * vib_sd
    vib_x = bandnoise(rng, n, 2, 22, 1.0) * vib_sd * 0.4
    vib_y = bandnoise(rng, n, 2, 22, 1.0) * vib_sd * 0.4
    eng = 0.04 * np.sin(2 * np.pi * np.cumsum(8 + 0.35 * v) / FS)

    lin = np.stack([a_lat + vib_x + C.ex, a_lon + vib_y, vib_z + C.ez + eng], 1)
    acc = lin + np.stack([GRAV * np.sin(phi), GRAV * np.sin(theta), np.full(n, GRAV)], 1)
    gyr = np.stack([gx, gy, gz], 1)

    # phone-mount misalignment (residual after orientation correction) + sensor imperfections
    mis = np.deg2rad(rng.normal(0, [1.5, 1.5, 2.0]))
    R = rotation(*mis)
    acc, lin, gyr = acc @ R.T, lin @ R.T, gyr @ R.T
    acc += rng.normal(0, 0.03, acc.shape)
    lin += rng.normal(0, 0.03, lin.shape)
    gyr += rng.normal(0, 0.004, gyr.shape) + rng.normal(0, 0.003, 3)      # white noise + per-trip bias

    # timestamps, jitter, dropped samples
    start = datetime(2026, 1, 10, 9, 0, 0) + timedelta(days=int(trip_id) // 3, hours=int(trip_id) % 3 * 3)
    up0 = int(rng.integers(10 ** 12, 2 * 10 ** 13))
    out = {}
    for name, arr, phase in (("accelerometer", acc, 0.0), ("linear_acceleration", lin, 0.0019), ("gyroscope", gyr, 0.0037)):
        ts = t + phase + rng.normal(0, 0.0008, n)
        ts = np.maximum.accumulate(ts)
        keep = rng.random(n) > 0.003
        df = pd.DataFrame({"timestamp": [(start + timedelta(seconds=float(s))).strftime("%d/%m/%Y %H:%M:%S") for s in ts],
                           "uptimeNanos": (up0 + ts * 1e9).astype(np.int64),
                           "x": arr[:, 0].round(5), "y": arr[:, 1].round(5), "z": arr[:, 2].round(5)})
        out[name] = df[keep].reset_index(drop=True)

    gt = pd.DataFrame([(k, round(max(a, 0.0), 2), round(min(b, duration), 2)) for k, a, b in events],
                      columns=["event", "start", "end"]).sort_values("start").reset_index(drop=True)
    info = {"synthetic": True, "dataset": DATASET_NAME, "generator_version": GENERATOR_VERSION, "seed": seed,
            "trip_id": trip_id, "driver_id": driver, "vehicle_type": vehicle, "road_type": road, "surface": surface,
            "road_roughness_factor": round(float(rough), 3), "duration_s": round(duration, 1), "nominal_sampling_hz": FS,
            "frame": "vehicle-aligned: x=right, y=forward, z=up; gyro z>0 = left turn",
            "phone_mount_misalignment_deg": [round(float(x), 2) for x in np.rad2deg(mis)],
            "simulated_start": start.strftime("%Y-%m-%d %H:%M:%S"), "n_labelled_events": int(len(gt))}
    return out, gt, info


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "driver_behavior"))
    a = ap.parse_args()

    n_trips = len(PLAN)
    rng = np.random.default_rng(a.seed)
    pool = [k for k, q in QUOTA.items() for _ in range(q)]
    pool = list(rng.permutation(pool))
    per_trip = [pool[i::n_trips] for i in range(n_trips)]          # deal events round-robin into trips

    if os.path.isdir(a.out):
        for p in os.listdir(a.out):
            if p.startswith("trip_") or p in ("trips_metadata.csv", "events_all.csv", "SOURCE_README.md"):
                q = os.path.join(a.out, p)
                shutil.rmtree(q) if os.path.isdir(q) else os.remove(q)
    os.makedirs(a.out, exist_ok=True)

    meta, allev = [], []
    for i, (driver, vehicle, road, surface) in enumerate(PLAN, start=1):
        files, gt, info = simulate_trip(a.seed, i, per_trip[i - 1], driver, vehicle, road, surface)
        d = os.path.join(a.out, f"trip_{i}")
        os.makedirs(d, exist_ok=True)
        for name, df in files.items():
            df.to_csv(os.path.join(d, name + ".csv"), index=False)
        gt.to_csv(os.path.join(d, "ground_truth.csv"), index=False)
        with open(os.path.join(d, "trip_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        meta.append({k: info[k] for k in ("trip_id", "driver_id", "vehicle_type", "road_type", "surface", "duration_s", "n_labelled_events")})
        allev.append(gt.assign(trip_id=i, driver_id=driver))
        print(f"trip_{i:<2d} {driver} {vehicle:<9s} {road:<8s} {surface:<3s} {info['duration_s']:6.1f}s  events={len(gt)}")
    pd.DataFrame(meta).to_csv(os.path.join(a.out, "trips_metadata.csv"), index=False)
    ev = pd.concat(allev, ignore_index=True)[["trip_id", "driver_id", "event", "start", "end"]]
    ev.to_csv(os.path.join(a.out, "events_all.csv"), index=False)
    print("\nEvents per class:\n" + ev.event.value_counts().to_string())
    print(f"\nTotal labelled events: {len(ev)}  ->  {a.out}")


if __name__ == "__main__":
    main()
