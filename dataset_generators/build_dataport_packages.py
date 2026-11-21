#!/usr/bin/env python3
"""
Builds ready-to-upload IEEE DataPort packages for the three SYNTHETIC SafeDrive datasets.

Output (ieee_dataport/):
    0_..._COMBINED_upload_this/  ONE dataset for the whole team (recommended)
    1_SafeDrive-Sim-IMU/      data zip, documentation PDF, cover image, form_text.md, generator code
    2_SafeDrive-Sim-Alcohol/  csv, documentation PDF, cover image, form_text.md, generator code
    3_SafeDrive-Sim-Crash/    csv, documentation PDF, cover image, form_text.md, generator code

Run AFTER the datasets exist and the models are trained (benchmark numbers are read from reports/).
    python dataset_generators/build_dataport_packages.py
"""
import os, json, shutil, zipfile, subprocess, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, REPORTS, OUT = (os.path.join(ROOT, x) for x in ("data", "reports", "ieee_dataport"))
GEN = os.path.join(ROOT, "dataset_generators")
AUTHORS = "Lekhya [surname], Srinivas [surname], Likhitha [surname]"
AFFIL = "Amrita Vishwa Vidyapeetham"
LICENSE = "CC BY 4.0 (Creative Commons Attribution 4.0 International)"
SYN = ("**IMPORTANT: this dataset is entirely SYNTHETIC.** It was produced by a documented, seeded computer simulation. "
       "No real vehicle, person, sensor or crash was recorded. It must not be used to draw conclusions about real drivers or roads.")


def rj(name):
    p = os.path.join(REPORTS, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def to_pdf(md_path, pdf_path):
    html = md_path[:-3] + ".html"
    css = "<style>body{font-family:Arial,Helvetica,sans-serif;font-size:12px;line-height:1.5;margin:20px;max-width:none!important;padding:0!important}table{border-collapse:collapse}td,th{border:1px solid #999;padding:4px 8px}code{background:#eee}</style>"
    subprocess.run(["pandoc", md_path, "-f", "gfm", "-t", "html", "-s", "--metadata", "title=Documentation", "-H", "/dev/stdin", "-o", html],
                   input=css.encode(), check=True, capture_output=True)
    subprocess.run(["wkhtmltopdf", "--quiet", "--enable-local-file-access", html, pdf_path], capture_output=True)
    os.remove(html)


def table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def finish(folder, name, doc_md, form_md, files):
    os.makedirs(folder, exist_ok=True)
    md = os.path.join(folder, f"{name}_documentation.md")
    open(md, "w").write(doc_md)
    to_pdf(md, os.path.join(folder, f"{name}_documentation.pdf"))
    os.remove(md)
    open(os.path.join(folder, "form_text.md"), "w").write(form_md)
    for src in files:
        shutil.copy(src, folder)



# =========================================================================================== 0. COMBINED (one dataset, whole team)
def pkg_combined():
    name = "SafeDrive-Sim"
    folder = os.path.join(OUT, "0_SafeDrive-Sim_COMBINED_upload_this")
    os.makedirs(os.path.join(folder, "scripts"), exist_ok=True)
    ev = pd.read_csv(os.path.join(DATA, "driver_behavior", "events_all.csv"))
    al = pd.read_csv(os.path.join(DATA, "alcohol", "safedrive_sim_alcohol.csv"))
    cr = pd.read_csv(os.path.join(DATA, "crash", "safedrive_sim_crash.csv"))
    n_trips = ev.trip_id.nunique()
    dd_al = pd.read_csv(os.path.join(OUT, "2_SafeDrive-Sim-Alcohol", "SafeDrive-Sim-Alcohol_data_dictionary.csv"))
    dd_cr = pd.read_csv(os.path.join(OUT, "3_SafeDrive-Sim-Crash", "SafeDrive-Sim-Crash_data_dictionary.csv"))
    counts = ev.event.value_counts().rename_axis("event").reset_index(name="count")
    b, a_, c_ = rj("behavior.json"), rj("alcohol.json"), rj("crash_sim.json")
    bm = b["models"][b["deployed"]]; am = a_["models"][a_["deployed"]]
    cd = c_["standard_paper"]["deployed"]; cm = {r["model"]: r for r in c_["standard_paper"]["comparison"]}[cd]

    # combined cover image
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    base = os.path.join(DATA, "driver_behavior", "trip_1")
    L = pd.read_csv(os.path.join(base, "linear_acceleration.csv")); t = (L.uptimeNanos - L.uptimeNanos.iloc[0]) / 1e9
    m = t < 120
    for col, lab in zip("xyz", "xyz"):
        ax[0].plot(t[m], L[col][m], lw=0.6, label=lab)
    ax[0].set_title("1  Driving IMU (linear accel., m/s²)", fontsize=10); ax[0].set_xlabel("time (s)"); ax[0].legend(ncol=3, fontsize=7)
    for lab, col in ((0, "tab:blue"), (1, "tab:red")):
        sub = al[al.label == lab]; ax[1].scatter(sub.alcohol_ml_l, sub.pupil_ratio, s=5, alpha=0.4, c=col, label=["no alcohol", "alcohol"][lab])
    ax[1].set_title("2  Alcohol sensors (class overlap)", fontsize=10); ax[1].set_xlabel("MQ-3 alcohol (ml/L)"); ax[1].set_ylabel("pupil ratio"); ax[1].legend(fontsize=7)
    sm = cr.sample(5000, random_state=0); sc = ax[2].scatter(sm.longitude, sm.latitude, c=sm.severity, s=4, cmap="RdYlGn_r", alpha=0.6)
    ax[2].set_title("3  Crash records (synthetic locations)", fontsize=10); ax[2].set_xlabel("longitude"); ax[2].set_ylabel("latitude"); plt.colorbar(sc, ax=ax[2], label="severity")
    fig.suptitle("SafeDrive-Sim (SYNTHETIC) - three linked datasets for driver-risk research", fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(folder, f"{name}_cover.png"), dpi=130); plt.close(fig)

    # data zip
    zp = os.path.join(folder, f"{name}_data.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}/README.txt", "SafeDrive-Sim: SYNTHETIC data (no real drivers, vehicles, people or crashes). See SafeDrive-Sim_documentation.pdf.\n"
                   "1_driver_behavior_imu/   12 trips of simulated smartphone IMU data + labelled events\n"
                   "2_alcohol_sensors/       2,000 simulated sessions, 5 sensor features + label\n"
                   "3_crash_records/         30,000 simulated crash records (Coimbatore-area bounding box)\n")
        for r, _, fs in os.walk(os.path.join(DATA, "driver_behavior")):
            for f in fs:
                q = os.path.join(r, f); z.write(q, os.path.join(name, "1_driver_behavior_imu", os.path.relpath(q, os.path.join(DATA, "driver_behavior"))))
        z.write(os.path.join(DATA, "alcohol", "safedrive_sim_alcohol.csv"), f"{name}/2_alcohol_sensors/safedrive_sim_alcohol.csv")
        z.write(os.path.join(OUT, "2_SafeDrive-Sim-Alcohol", "SafeDrive-Sim-Alcohol_data_dictionary.csv"), f"{name}/2_alcohol_sensors/data_dictionary.csv")
        z.write(os.path.join(DATA, "crash", "safedrive_sim_crash.csv"), f"{name}/3_crash_records/safedrive_sim_crash.csv")
        z.write(os.path.join(OUT, "3_SafeDrive-Sim-Crash", "SafeDrive-Sim-Crash_data_dictionary.csv"), f"{name}/3_crash_records/data_dictionary.csv")
    for g in ("generate_driver_behavior.py", "generate_alcohol.py", "generate_crash.py"):
        shutil.copy(os.path.join(GEN, g), os.path.join(folder, "scripts", g))

    doc = f"""# {name}: A Synthetic Multi-Source Dataset for Driver-Risk Research

{SYN} **Crash records are not real crashes and locations do not correspond to real accident black spots.**

**Authors:** {AUTHORS} ({AFFIL})  |  **Version:** 1.0  |  **License:** {LICENSE}  |  **Generators:** `generate_driver_behavior.py`, `generate_alcohol.py`, `generate_crash.py` v1.0, seed 2026

## 1. Overview
SafeDrive-Sim bundles three linked synthetic datasets built for one research system that fuses how a driver drives, whether the driver is impaired, and how dangerous the road context is:

| Part | Folder in the zip | Content |
|---|---|---|
| 1. Driving behaviour (IMU) | `1_driver_behavior_imu/` | {n_trips} simulated trips, accelerometer / linear acceleration / gyroscope at about 50 Hz, {len(ev)} labelled events in 9 classes |
| 2. Alcohol impairment | `2_alcohol_sensors/` | {len(al):,} simulated sessions ({(al.label==0).sum()} no alcohol, {(al.label==1).sum()} alcohol), 60 simulated participants, 5 sensor features |
| 3. Crash records | `3_crash_records/` | {len(cr):,} simulated crash records, 2022-2025, bounding box around Coimbatore, India |

The parts were generated independently (they do not describe the same simulated people or trips); they share a common purpose and documentation.

## 2. Part 1 - Driving behaviour (IMU)
Each `trip_<1..{n_trips}>/` folder: `accelerometer.csv`, `linear_acceleration.csv`, `gyroscope.csv` (columns `timestamp, uptimeNanos, x, y, z`), `ground_truth.csv` (`event, start, end`, seconds from trip start), `trip_info.json`. Top level: `trips_metadata.csv`, `events_all.csv`.
Axes: x right, y forward, z up; gyro z > 0 = left turn. `uptimeNanos` is the simulated sensor clock; dates in `timestamp` are artificial.

{table(counts)}

Generation: target-speed profile with driver speed-keeping; events as smooth pulses (aggressive and gentle ranges overlap); lateral acceleration tied to yaw rate and speed; pitch/roll response; road vibration, engine tone, per-trip gyro bias, phone-mount misalignment, sensor noise, timestamp jitter, about 0.3% dropped samples; 3 driver profiles, 2 vehicles, city/highway, dry/wet; label-boundary noise about ±0.15 s.

## 3. Part 2 - Alcohol impairment sensors
`safedrive_sim_alcohol.csv` ({len(al):,} rows):

{table(dd_al)}

Generation assumptions (stylised, NOT physiological measurements): alcohol sessions have on average higher cabin vapour, slightly warmer face, larger pupil ratio; participant random effects; A/C-dependent cabin temperature; confounders (about 8% of sober sessions with vapour spikes, about 18% of alcohol sessions with weak readings); sensor noise. Use `participant_id` for grouped cross-validation.

## 4. Part 3 - Crash records
`safedrive_sim_crash.csv` ({len(cr):,} rows, {cr.shape[1]} columns):

{table(dd_cr)}

Generation: 16 random hotspot clusters (arterial junctions, bypass stretches, market areas) give 72% of locations; stylised monsoon rain, winter fog, daily temperature cycle and commuter peaks; severity = latent risk (vehicle, speed limit, night, weather, low visibility, signal, hotspot) plus logistic noise, cut at fixed quantiles (32 / 38 / 22 / 8 %). Contains crashes only, so it supports severity / outcome modelling, not crash-likelihood estimation.

## 5. Reference benchmarks (synthetic only)
- Part 1: leave-one-trip-out CV, {b['deployed']}: event accuracy {bm['event']['accuracy']*100:.1f}%, macro-F1 {bm['event']['macro_f1']:.3f}.
- Part 2: 5-fold CV grouped by participant, {a_['deployed']}: accuracy {am['accuracy']*100:.1f}%, ROC-AUC {am['roc_auc']:.3f}.
- Part 3: 80/20 split, {cd}, risk level Low/Medium/High: accuracy {cm['accuracy']*100:.1f}%, macro-F1 {cm['macro_f1']:.3f}, ROC-AUC {cm['roc_auc']:.3f}.
These describe controlled synthetic benchmarks and are NOT estimates of real-world performance.

## 6. Reproducibility
Run the three scripts in `scripts/` (`python generate_driver_behavior.py`, `generate_alcohol.py`, `generate_crash.py`, default seed 2026) to regenerate every file exactly.

## 7. Intended use and limitations
Intended: prototyping, teaching, method comparison, pipeline testing. Not intended: safety validation, legal or medical decisions, policy or planning, or any claim about real drivers, people or roads.
Limitations: simplified dynamics and physiology; all effect sizes are the authors' assumptions; classifiers may learn the generators' rules; the three parts are not linked at row level.

## 8. Ethics
No human participants, no personal data, no real vehicles, incidents or locations. Place names only define a plausible coordinate range.

## 9. Citation
{AUTHORS}, "{name}: A Synthetic Multi-Source Dataset for Driver-Risk Research," IEEE DataPort, 2026, doi: [assigned by IEEE DataPort after publishing].
"""
    md = os.path.join(folder, f"{name}_documentation.md"); open(md, "w").write(doc)
    to_pdf(md, os.path.join(folder, f"{name}_documentation.pdf")); os.remove(md)

    form = f"""# Fill-in sheet for the IEEE DataPort "Submit a Dataset" form  (ONE dataset for the whole team)

The form has these parts: Metadata -> Dataset Authors, Title, Category, Keywords, Data Format, Related Dataset, License, Dataset Image, Abstract; then Dataset -> Dataset Files, Scripts.

## Dataset Authors (one row per author: First Name, Last Name, ORCID optional, Affiliation optional) - in citation order
1. Lekhya  [surname]  - {AFFIL}
2. Srinivas  [surname]  - {AFFIL}
3. Likhitha  [surname]  - {AFFIL}
(Add your faculty guide as an author only if the faculty agrees.)

## Title
{name}: A Synthetic Multi-Source Dataset (Smartphone IMU Driving Events, In-Cabin Alcohol Sensors and Road-Crash Records) for Driver-Risk Research

## Category
Choose the closest options in the dropdown (for example something like Transportation, Machine Learning, Sensors).

## Keywords
synthetic data; driver behaviour; smartphone sensors; alcohol detection; road crash; accident severity; road safety; multi-source fusion

## Data Format
CSV (inside a ZIP; the ZIP also contains JSON metadata files)

## Related Dataset
(leave empty)

## License
{LICENSE}  (DataPort's default for datasets)

## Dataset Image
{name}_cover.png

## Abstract (paste)
SafeDrive-Sim is a fully synthetic, openly documented multi-source dataset for driver-risk research. It bundles three linked parts generated by seeded simulators: (1) {n_trips} simulated car trips with smartphone accelerometer, linear-acceleration and gyroscope signals at about 50 Hz and {len(ev)} labelled events in nine classes (aggressive braking, acceleration, turns and lane changes, gentle manoeuvres, speed bumps and potholes); (2) {len(al):,} simulated driving sessions from 60 simulated participants with five in-cabin sensor features (gas, cabin and facial temperature, pupil ratio) labelled alcohol / no alcohol; and (3) {len(cr):,} simulated road-crash records for a Coimbatore-area bounding box with weather, road, vehicle and time context and three outcomes (severity, road length affected, clearance time). Class overlap, sensor noise, confounders and participant-level effects are built in so that the tasks are non-trivial. No real vehicle, person, sensor or crash was recorded; crash locations do not correspond to real accident black spots. The data are intended for prototyping, teaching and benchmarking of driver-behaviour, impairment and crash-severity models and of fusion systems; generator scripts are included so every file can be regenerated exactly.

## Dataset Files  (upload)
{name}_data.zip
{name}_documentation.pdf

## Scripts  (optional upload field)
scripts/generate_driver_behavior.py, scripts/generate_alcohol.py, scripts/generate_crash.py
"""
    open(os.path.join(folder, "FORM_FILL_IN_SHEET.md"), "w").write(form)


# =========================================================================================== 1. IMU
def pkg_imu():
    name = "SafeDrive-Sim-IMU"
    folder = os.path.join(OUT, f"1_{name}")
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(DATA, "driver_behavior")
    ev = pd.read_csv(os.path.join(base, "events_all.csv"))
    meta = pd.read_csv(os.path.join(base, "trips_metadata.csv"))
    counts = ev.event.value_counts().rename_axis("event").reset_index(name="count")
    nrows = 0
    for t in meta.trip_id:
        nrows += sum(1 for _ in open(os.path.join(base, f"trip_{t}", "accelerometer.csv"))) - 1
    rep = rj("behavior.json"); dep = rep.get("deployed", "n/a")
    m = rep.get("models", {}).get(dep, {})
    bench = (f"leave-one-trip-out cross-validation with the {dep}: event-level accuracy {m['event']['accuracy']*100:.1f}%, "
             f"macro-F1 {m['event']['macro_f1']:.3f}; window-level accuracy {m['window']['accuracy']*100:.1f}%") if m else "n/a"

    # cover image
    d = os.path.join(base, "trip_1")
    L = pd.read_csv(os.path.join(d, "linear_acceleration.csv")); G = pd.read_csv(os.path.join(d, "gyroscope.csv"))
    gt = pd.read_csv(os.path.join(d, "ground_truth.csv"))
    t0 = L.uptimeNanos.iloc[0]; tl = (L.uptimeNanos - t0) / 1e9; tg = (G.uptimeNanos - t0) / 1e9
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for a, (x, cols, ttl) in zip(ax, [(tl, [L.x, L.y, L.z], "Linear acceleration (m/s²): x right, y forward, z up"), (tg, [G.x, G.y, G.z], "Gyroscope (rad/s): x pitch, y roll, z yaw")]):
        for c, lab in zip(cols, "xyz"):
            a.plot(x, c, lw=0.7, label=lab)
        a.set_title(ttl, fontsize=10); a.legend(loc="upper right", ncol=3, fontsize=8)
        for _, r in gt.iterrows():
            if r.start < 200:
                a.axvspan(r.start, r.end, color="orange" if r.event != "non_aggressive" else "green", alpha=0.18)
    ax[1].set_xlim(0, 200); ax[1].set_xlabel("time (s)  —  shaded = labelled event")
    fig.suptitle("SafeDrive-Sim-IMU (SYNTHETIC) – example trip", fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(folder, f"{name}_cover.png"), dpi=130); plt.close(fig)

    # data zip
    zp = os.path.join(folder, f"{name}_data.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for r, _, fs in os.walk(base):
            for f in fs:
                p = os.path.join(r, f); z.write(p, os.path.join(name, os.path.relpath(p, base)))

    doc = f"""# {name}: Synthetic Smartphone Inertial-Sensor Dataset of Driving Events

{SYN}

**Authors:** {AUTHORS} ({AFFIL})  |  **Version:** 1.0  |  **License:** {LICENSE}  |  **Generator:** `generate_driver_behavior.py` v1.0, seed 2026

## 1. Overview
SafeDrive-Sim-IMU contains 12 simulated car trips (about 9-11 minutes each) as they would be recorded by a smartphone mounted in a car:
accelerometer, linear acceleration and gyroscope at about 50 Hz, with {len(ev)} labelled driving / road events in 9 classes.
It was created for developing and benchmarking driver-behaviour classifiers when real labelled data is unavailable or unsafe to collect.

## 2. Contents
`{name}/trip_<1..12>/` each contain:

| File | Description |
|---|---|
| `accelerometer.csv` | specific force incl. gravity: `timestamp, uptimeNanos, x, y, z` (m/s²) |
| `linear_acceleration.csv` | gravity-removed acceleration, same columns (m/s²) |
| `gyroscope.csv` | angular rate, same columns (rad/s) |
| `ground_truth.csv` | labelled events: `event, start, end` (seconds from trip start) |
| `trip_info.json` | per-trip metadata (driver, vehicle, road, surface, mount misalignment, seed) |

Top level: `trips_metadata.csv` (one row per trip) and `events_all.csv` (all {len(ev)} events).
Total sensor rows per file type: about {nrows:,}. Axes: x = right, y = forward, z = up; gyro z > 0 = turning left (vehicle-aligned frame).
`uptimeNanos` is the simulated sensor clock; `timestamp` is a nominal simulated wall clock (dates are artificial).

## 3. Event classes
{table(counts)}

`non_aggressive` = gentle braking / acceleration / turns / lane changes. Unlabelled stretches are normal cruising (with very mild unlabelled manoeuvres).

## 4. How the data was generated
Physics-inspired simulation (see `generate_driver_behavior.py`): a target-speed profile and driver speed-keeping controller;
events injected as smooth pulses (aggressive and gentle intensity ranges overlap on purpose); lateral acceleration tied to yaw rate and speed;
pitch/roll response to acceleration; road vibration and engine tone; per-trip gyro bias, phone-mount misalignment, sensor noise, timestamp jitter and about 0.3% dropped samples;
3 simulated drivers (smooth / average / sporty), 2 vehicle types, city and highway roads, dry and wet surface, and label-boundary noise of about ±0.15 s imitating human annotation.
Event counts, trip plan and seed are fixed in the script, so the dataset is exactly reproducible.

## 5. Reference benchmark
{bench}. Because the data is synthetic, scores show performance on a controlled benchmark only and are **not** estimates of real-road accuracy.

## 6. Intended use and limitations
Intended: prototyping, teaching, method comparison, pipeline testing. Not intended: validating safety-critical systems or claims about real drivers.
Limitations: simplified vehicle dynamics; no GPS, magnetometer, traffic, or phone handling; event shapes are idealised; classifiers may learn the generator's rules.

## 7. Ethics
No personal data. No human participants. No real locations or vehicles.

## 8. Citation
{AUTHORS}, "{name}: Synthetic Smartphone Inertial-Sensor Dataset of Driving Events," IEEE DataPort, 2026, doi: [assigned by IEEE DataPort after publishing].
"""
    form = f"""# Form text for IEEE DataPort - {name}

**Title:** {name}: A Synthetic Smartphone Inertial-Sensor Dataset of Driving Events (Aggressive Manoeuvres, Speed Bumps and Potholes)

**Authors:** {AUTHORS}  –  {AFFIL}

**Keywords:** synthetic data; driver behaviour; smartphone sensors; accelerometer; gyroscope; aggressive driving; road safety; time series

**Abstract (paste):**
SafeDrive-Sim-IMU is a fully synthetic dataset of smartphone inertial-sensor recordings for driving-event detection. It contains 12 simulated car trips (about 10 minutes each) with accelerometer, linear-acceleration and gyroscope signals at about 50 Hz and {len(ev)} labelled events in nine classes: aggressive braking, aggressive acceleration, aggressive left/right turns, aggressive left/right lane changes, non-aggressive (gentle) manoeuvres, speed bumps and potholes. Trips were generated by a physics-inspired, seeded simulator with three simulated driver profiles, two vehicle types, city and highway roads, dry and wet surfaces, realistic sensor noise, phone-mount misalignment, timestamp jitter, dropped samples and annotation noise. Aggressive and gentle intensity ranges deliberately overlap so that classification is non-trivial. No real vehicles or people were recorded; the dataset is intended for prototyping, education and benchmarking of driver-behaviour algorithms, and the generator code is included for full reproducibility.

**Instructions (paste):**
Unzip {name}_data.zip. Each trip folder holds accelerometer.csv, linear_acceleration.csv, gyroscope.csv, ground_truth.csv and trip_info.json; trips_metadata.csv and events_all.csv are at the top level. Axes: x right, y forward, z up. Event start/end are in seconds from trip start (convert uptimeNanos with (uptimeNanos - first uptimeNanos)/1e9). See the documentation PDF. To regenerate: python generate_driver_behavior.py --seed 2026.

**Files to upload:** {name}_data.zip, {name}_documentation.pdf, generate_driver_behavior.py   |   **Dataset image:** {name}_cover.png
**Suggested license (state in documentation):** {LICENSE}
"""
    finish(folder, name, doc, form, [os.path.join(GEN, "generate_driver_behavior.py")])


# =========================================================================================== 2. Alcohol
def pkg_alcohol():
    name = "SafeDrive-Sim-Alcohol"
    folder = os.path.join(OUT, f"2_{name}")
    os.makedirs(folder, exist_ok=True)
    csv = os.path.join(DATA, "alcohol", "safedrive_sim_alcohol.csv")
    d = pd.read_csv(csv)
    rep = rj("alcohol.json"); dep = rep.get("deployed", "n/a"); m = rep.get("models", {}).get(dep, {})
    bench = (f"5-fold cross-validation grouped by participant with {dep}: accuracy {m['accuracy']*100:.1f}%, macro-F1 {m['macro_f1']:.3f}, ROC-AUC {m['roc_auc']:.3f}") if m else "n/a"
    st = d.groupby("label")[["alcohol_ml_l", "cabin_temp_c", "face_temp_min_c", "face_temp_max_c", "pupil_ratio"]].mean().round(3).reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for lab, c, nm in ((0, "tab:blue", "no alcohol"), (1, "tab:red", "alcohol")):
        s = d[d.label == lab]
        ax[0].scatter(s.alcohol_ml_l, s.pupil_ratio, s=6, alpha=0.4, c=c, label=nm)
        ax[1].scatter(s.face_temp_min_c, s.face_temp_max_c, s=6, alpha=0.4, c=c, label=nm)
    ax[0].set_xlabel("MQ-3 cabin alcohol (ml/L)"); ax[0].set_ylabel("pupil ratio"); ax[1].set_xlabel("face temp min (°C)"); ax[1].set_ylabel("face temp max (°C)")
    ax[0].legend(); fig.suptitle("SafeDrive-Sim-Alcohol (SYNTHETIC) – class overlap", fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(folder, f"{name}_cover.png"), dpi=130); plt.close(fig)

    dd = pd.DataFrame([
        ("sample_id", "text", "-", "Unique row id (SSA-00001 ...)"),
        ("participant_id", "text", "-", "Simulated participant (P001-P060); use for grouped cross-validation"),
        ("alcohol_ml_l", "float", "ml/L (proxy)", "Simulated MQ-3 gas-sensor reading of cabin alcohol vapour"),
        ("cabin_temp_c", "float", "°C", "Simulated cabin air temperature"),
        ("face_temp_min_c", "float", "°C", "Simulated coolest pixel on driver's face (thermal array)"),
        ("face_temp_max_c", "float", "°C", "Simulated warmest pixel on driver's face (thermal array)"),
        ("pupil_ratio", "float", "ratio", "Simulated pupil-to-iris diameter ratio (camera)"),
        ("label", "int", "0/1", "0 = no alcohol, 1 = alcohol present / impaired")], columns=["column", "type", "unit", "description"])
    dd.to_csv(os.path.join(folder, f"{name}_data_dictionary.csv"), index=False)

    doc = f"""# {name}: Synthetic In-Cabin Alcohol-Impairment Sensor Dataset

{SYN}

**Authors:** {AUTHORS} ({AFFIL})  |  **Version:** 1.0  |  **License:** {LICENSE}  |  **Generator:** `generate_alcohol.py` v1.0, seed 2026

## 1. Overview
{len(d):,} simulated driving sessions from {d.participant_id.nunique()} simulated participants ({(d.label==0).sum()} no-alcohol, {(d.label==1).sum()} alcohol). Each row has five features
modelled on a low-cost in-cabin set-up (MQ-3 gas sensor, thermal array, camera) and a binary label.

## 2. File
`safedrive_sim_alcohol.csv` - {len(d):,} rows, 8 columns (see data dictionary).

{table(dd)}

Class means:

{table(st)}

## 3. How the data was generated
Seeded random simulation (`generate_alcohol.py`). Assumptions (stylised, NOT physiological measurements): alcohol sessions have on average higher cabin vapour, slightly warmer face and larger pupil ratio;
participants differ in baseline face temperature and pupil size (random effects); A/C on/off changes cabin temperature, which shifts face temperature and gas readings;
confounders - about 8% of sober sessions contain a vapour spike (sanitiser, perfume, food) and about 18% of alcohol sessions have a weak vapour reading (windows open, sensor far away); sensor noise on all channels.
Classes therefore overlap and no classifier should reach 100%.

## 4. Reference benchmark
{bench}. These numbers describe a controlled synthetic benchmark only.

## 5. Intended use and limitations
Intended: prototyping, teaching, algorithm comparison, pipeline testing. Not intended: any real impairment testing, legal or medical decisions.
Limitations: all relationships are assumptions written by the authors; real physiology and sensor behaviour are more complex; no blood-alcohol values are provided.

## 6. Ethics
No human participants, no personal data. Participant IDs are simulated.

## 7. Citation
{AUTHORS}, "{name}: Synthetic In-Cabin Alcohol-Impairment Sensor Dataset," IEEE DataPort, 2026, doi: [assigned by IEEE DataPort after publishing].
"""
    form = f"""# Form text for IEEE DataPort - {name}

**Title:** {name}: A Synthetic In-Cabin Sensor Dataset (Gas, Thermal and Pupil Features) for Driver Alcohol-Impairment Classification

**Authors:** {AUTHORS}  –  {AFFIL}

**Keywords:** synthetic data; alcohol detection; driver impairment; MQ-3 gas sensor; thermal imaging; pupil; classification; road safety

**Abstract (paste):**
SafeDrive-Sim-Alcohol is a fully synthetic tabular dataset for developing driver alcohol-impairment classifiers. It contains {len(d):,} simulated driving sessions from {d.participant_id.nunique()} simulated participants, balanced between no-alcohol and alcohol classes. Each session is described by five in-cabin sensor features modelled on a low-cost set-up: cabin alcohol vapour (MQ-3 proxy), cabin temperature, minimum and maximum facial temperature from a thermal array, and pupil-to-iris ratio from a camera. Data were produced by a seeded, documented simulator with participant-level random effects, A/C-dependent temperature shifts, realistic confounders (sanitiser or perfume vapour in sober sessions, weak readings in some alcohol sessions) and sensor noise, so that classes overlap and the task is non-trivial. No person was tested; relationships are stylised assumptions rather than physiological measurements. The dataset is intended for prototyping, teaching and benchmarking, and the generator is included for full reproducibility.

**Instructions (paste):**
Load safedrive_sim_alcohol.csv. Feature columns: alcohol_ml_l, cabin_temp_c, face_temp_min_c, face_temp_max_c, pupil_ratio. Target: label (0 = no alcohol, 1 = alcohol). Use participant_id for grouped cross-validation so that no simulated participant appears in both training and test folds. See the documentation PDF and data dictionary. To regenerate: python generate_alcohol.py --seed 2026.

**Files to upload:** safedrive_sim_alcohol.csv, {name}_data_dictionary.csv, {name}_documentation.pdf, generate_alcohol.py   |   **Dataset image:** {name}_cover.png
**Suggested license (state in documentation):** {LICENSE}
"""
    finish(folder, name, doc, form, [csv, os.path.join(GEN, "generate_alcohol.py")])


# =========================================================================================== 3. Crash
def pkg_crash():
    name = "SafeDrive-Sim-Crash"
    folder = os.path.join(OUT, f"3_{name}")
    os.makedirs(folder, exist_ok=True)
    csv = os.path.join(DATA, "crash", "safedrive_sim_crash.csv")
    d = pd.read_csv(csv)
    rep = rj("crash_sim.json"); sp = rep.get("standard_paper", {}); dep = sp.get("deployed", "n/a")
    cmp_ = {r["model"]: r for r in sp.get("comparison", [])}
    m = cmp_.get(dep)
    bench = (f"80/20 split, {dep} predicting risk level (Low / Medium / High): accuracy {m['accuracy']*100:.1f}%, macro-F1 {m['macro_f1']:.3f}, ROC-AUC {m['roc_auc']:.3f}") if m else "n/a"
    sev = d.severity.value_counts().sort_index().rename_axis("severity").reset_index(name="records")
    sev["meaning"] = sev.severity.map({1: "property damage only", 2: "minor injury", 3: "grievous injury", 4: "fatal"})

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    s = d.sample(6000, random_state=0)
    sc = ax[0].scatter(s.longitude, s.latitude, c=s.severity, s=5, cmap="RdYlGn_r", alpha=0.6)
    ax[0].set_xlabel("longitude"); ax[0].set_ylabel("latitude"); ax[0].set_title("Synthetic crash locations", fontsize=10); plt.colorbar(sc, ax=ax[0], label="severity")
    d.groupby("primary_vehicle").severity.mean().sort_values().plot.barh(ax=ax[1], color="tab:purple")
    ax[1].set_xlabel("mean severity"); ax[1].set_title("Mean severity by vehicle type", fontsize=10)
    fig.suptitle("SafeDrive-Sim-Crash (SYNTHETIC)", fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(folder, f"{name}_cover.png"), dpi=130); plt.close(fig)

    dd = pd.DataFrame([
        ("crash_id", "text", "-", "Unique id (SSC-000001 ...)"), ("date", "date", "YYYY-MM-DD", "Simulated crash date, 2022-01-01 to 2025-12-31"),
        ("hour", "int", "0-23", "Hour of day"), ("weekday", "text", "-", "Day of week"),
        ("latitude", "float", "deg", "Simulated location inside a bounding box around Coimbatore (lat 10.93-11.09)"),
        ("longitude", "float", "deg", "Simulated location (lng 76.90-77.06)"),
        ("road_type", "text", "-", "Expressway/Bypass | Highway/Arterial | Local street"), ("traffic_signal", "bool", "-", "Signalised junction"),
        ("light_condition", "text", "-", "Day | Night (06:00-18:00 = Day)"),
        ("weather", "text", "-", "Clear, Partly Cloudy, Overcast, Haze, Fog, Drizzle, Light Rain, Rain, Heavy Rain, Thunderstorm"),
        ("temperature_c", "float", "°C", "Air temperature"), ("visibility_km", "float", "km", "Visibility (0.1-10)"),
        ("wind_direction", "text", "-", "16-point compass or CALM"), ("speed_limit_kmh", "int", "km/h", "Posted speed limit"),
        ("primary_vehicle", "text", "-", "two_wheeler, car, auto_rickshaw, bus, goods_vehicle"),
        ("severity", "int", "1-4", "1 property damage only, 2 minor injury, 3 grievous injury, 4 fatal"),
        ("road_affected_km", "float", "km", "Length of road affected (0 = point incident)"),
        ("clearance_time_min", "int", "min", "Minutes until the road was cleared")], columns=["column", "type", "unit", "description"])
    dd.to_csv(os.path.join(folder, f"{name}_data_dictionary.csv"), index=False)

    doc = f"""# {name}: Synthetic Road-Crash Records for Severity and Outcome Modelling

{SYN} **The records are not real crashes and the locations do not correspond to real accident black spots.**

**Authors:** {AUTHORS} ({AFFIL})  |  **Version:** 1.0  |  **License:** {LICENSE}  |  **Generator:** `generate_crash.py` v1.0, seed 2026

## 1. Overview
{len(d):,} simulated crash records (2022-2025) located inside a bounding box around Coimbatore, Tamil Nadu, India, with context (time, location, weather, road, vehicle) and outcomes (severity, road length affected, clearance time).

## 2. File
`safedrive_sim_crash.csv` - {len(d):,} rows, {d.shape[1]} columns.

{table(dd)}

Severity distribution (set by design):

{table(sev)}

## 3. How the data was generated
Seeded random simulation (`generate_crash.py`). 16 random hotspot clusters (arterial junctions, bypass stretches, market areas) generate 72% of locations; the rest are uniform.
Season-dependent rain and winter morning fog, a daily temperature cycle and commuter-peak hours are stylised. Severity = a latent risk score (vehicle type, speed limit, night, weather,
low visibility, signalised junction, hotspot effect) plus logistic noise, cut at fixed quantiles (32 / 38 / 22 / 8 %). Road length affected and clearance time depend on severity, road type and rain.
All effect sizes are assumptions chosen by the authors, not statistics from real crash data.
The file contains crashes only (no non-crash exposure), so it supports severity / outcome modelling, not estimating crash likelihood.

## 4. Reference benchmark
{bench}. Scores describe a controlled synthetic benchmark only.

## 5. Intended use and limitations
Intended: teaching, prototyping, method comparison (tabular classification, multi-task learning, hotspot mapping demos). Not intended: policy, planning, insurance or any inference about real crashes in any city.

## 6. Ethics
No personal data and no real incidents. Place names are used only to define a plausible coordinate range.

## 7. Citation
{AUTHORS}, "{name}: Synthetic Road-Crash Records for Severity and Outcome Modelling," IEEE DataPort, 2026, doi: [assigned by IEEE DataPort after publishing].
"""
    form = f"""# Form text for IEEE DataPort - {name}

**Title:** {name}: A Synthetic Road-Crash Dataset with Weather, Road and Vehicle Context for Severity and Outcome Prediction

**Authors:** {AUTHORS}  –  {AFFIL}

**Keywords:** synthetic data; road crash; accident severity; weather; hotspots; tabular data; multi-task learning; road safety

**Abstract (paste):**
SafeDrive-Sim-Crash is a fully synthetic dataset of {len(d):,} simulated road-crash records (2022-2025) placed in a bounding box around Coimbatore, India. Each record contains time, location, road type, signalised-junction flag, light condition, weather, temperature, visibility, wind direction, speed limit, primary vehicle type and three outcomes: severity (1 property damage only, 2 minor injury, 3 grievous injury, 4 fatal), road length affected, and clearance time. Records were produced by a seeded, documented simulator with spatial hotspot clusters, monsoon and winter-fog seasonality, commuter-peak hours and a latent-risk model with substantial random noise so that outcomes remain uncertain. The records are not real crashes, locations do not correspond to real accident black spots, and effect sizes are authors' assumptions. The dataset supports teaching, prototyping and benchmarking of severity classification, multi-task learning and hotspot-aware routing; generator code is included for reproducibility.

**Instructions (paste):**
Load safedrive_sim_crash.csv. Targets: severity (1-4), road_affected_km, clearance_time_min. Suggested tasks: risk level (severity 1-2 Low, 3 Medium, 4 High), road impact (0 / <=0.8 km / >0.8 km), clearance (<=30 / >30 min). The file contains crashes only, so it is not suitable for estimating crash probability. See the documentation PDF and data dictionary. To regenerate: python generate_crash.py --seed 2026.

**Files to upload:** safedrive_sim_crash.csv, {name}_data_dictionary.csv, {name}_documentation.pdf, generate_crash.py   |   **Dataset image:** {name}_cover.png
**Suggested license (state in documentation):** {LICENSE}
"""
    finish(folder, name, doc, form, [csv, os.path.join(GEN, "generate_crash.py")])


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    pkg_imu(); pkg_alcohol(); pkg_crash(); pkg_combined()
    for r, _, fs in sorted(os.walk(OUT)):
        for f in sorted(fs):
            print(os.path.relpath(os.path.join(r, f), ROOT), f"{os.path.getsize(os.path.join(r, f)) / 1e6:.2f} MB")
