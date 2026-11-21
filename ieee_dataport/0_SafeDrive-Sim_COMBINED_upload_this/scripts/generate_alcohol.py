#!/usr/bin/env python3
"""
SafeDrive-Sim-Alcohol  --  SYNTHETIC in-cabin alcohol-impairment sensor dataset generator (v1.0)

*** ALL DATA PRODUCED BY THIS SCRIPT IS SIMULATED. NO PERSON WAS TESTED AND NO SENSOR WAS READ. ***

One row = one simulated driving session of one simulated participant, described by five sensor features:
    alcohol_ml_l      MQ-3 gas-sensor reading of cabin alcohol vapour (ml/L, proxy units)
    cabin_temp_c      cabin air temperature (deg C)
    face_temp_min_c   coolest pixel on the driver's face, thermal array (deg C)
    face_temp_max_c   warmest pixel on the driver's face, thermal array (deg C)
    pupil_ratio       pupil-to-iris diameter ratio from an in-cabin camera
and a label:  0 = no alcohol,  1 = alcohol present / impaired.

Assumptions built into the simulator (all stylised, NOT physiological measurements):
  * impaired sessions have, on average, higher cabin vapour, slightly warmer face and larger pupil ratio
  * participants differ in baseline face temperature and pupil size (random effects)
  * cabin temperature (A/C on or off) shifts face-temperature and gas-sensor readings
  * confounders: ~8 % of sober sessions have a vapour spike (sanitiser, perfume, food);
    ~18 % of impaired sessions have a weak vapour reading (windows open, sensor far from driver)
  * sensor noise on every channel
Because classes overlap by design, a classifier should NOT reach 100 %.

Usage:  python dataset_generators/generate_alcohol.py [--seed 2026] [--n 2000] [--participants 60]
"""
import os, argparse
import numpy as np
import pandas as pd

GENERATOR_VERSION = "1.0"
DATASET_NAME = "SafeDrive-Sim-Alcohol"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "alcohol", "safedrive_sim_alcohol.csv")


def generate(seed=2026, n=2000, n_participants=60):
    r = np.random.default_rng(seed)
    label = np.array([0] * (n // 2) + [1] * (n - n // 2))
    r.shuffle(label)
    pid = r.integers(1, n_participants + 1, n)                       # who the session belongs to

    face_off = r.normal(0, 0.6, n_participants + 1)[pid]             # participant baseline face temperature
    pupil_off = r.normal(0, 0.025, n_participants + 1)[pid]          # participant baseline pupil ratio

    ac_on = r.random(n) < 0.4
    cabin = np.where(ac_on, r.normal(23.0, 1.5, n), r.normal(29.0, 2.2, n)).clip(17, 38)

    # MQ-3 cabin vapour
    sober = r.gamma(2.0, 0.022, n)
    spike = (r.random(n) < 0.08) * r.uniform(0.08, 0.30, n)
    imp = r.uniform(0.05, 0.55, n)
    weak = np.where(r.random(n) < 0.18, r.uniform(0.1, 0.4, n), 1.0)
    conc = np.where(label == 1, imp * weak, sober + spike) + 0.002 * (cabin - 26) + r.normal(0, 0.015, n)
    conc = conc.clip(0, None)

    fmin = r.normal(30.3 + 0.10 * (cabin - 26) + 0.6 * label, 0.8, n) + face_off
    fmax = fmin + r.normal(4.0 + 0.3 * label, 0.5, n).clip(1.5, None)
    pupil = (r.normal(0.32 + 0.035 * label, 0.045, n) + pupil_off).clip(0.15, 0.60)

    df = pd.DataFrame({"sample_id": [f"SSA-{i:05d}" for i in range(1, n + 1)],
                       "participant_id": [f"P{p:03d}" for p in pid],
                       "alcohol_ml_l": conc.round(4), "cabin_temp_c": cabin.round(2),
                       "face_temp_min_c": fmin.round(2), "face_temp_max_c": fmax.round(2),
                       "pupil_ratio": pupil.round(4), "label": label})
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--participants", type=int, default=60)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    df = generate(a.seed, a.n, a.participants)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{DATASET_NAME}: {len(df)} rows, {df.participant_id.nunique()} participants, "
          f"label counts {df.label.value_counts().to_dict()} -> {a.out}")


if __name__ == "__main__":
    main()
