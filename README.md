# SafeDrive AI
### AI-Based Multi-Source Driver Risk Prediction and Safety Intervention System
Amrita Vishwa Vidyapeetham · Research Methodology team project

SafeDrive AI combines three things into one live risk score:
1. **How the driver is driving**: aggressive manoeuvres detected from smartphone motion sensors.
2. **Whether the driver is impaired**: alcohol detected from in-cabin sensors.
3. **How dangerous the road conditions are**: crash risk and severity predicted from weather, road and location.

It then sends graded alerts and recommends a route that avoids accident hotspots.

---

## 1. Run it (Windows / macOS / Linux)

You need Python 3.9 or newer.

**Windows:** double-click `run_windows.bat`
**macOS / Linux:** `bash run_mac_linux.sh`

or by hand:

```bash
pip install -r requirements.txt
python train_all.py          # trains every model: about 5–10 minutes the first time
python src/server.py         # opens http://localhost:8000 in your browser
```

Once trained, you only need `python src/server.py` to start the dashboard again.

Optional: `pip install xgboost lightgbm` adds the exact XGBoost / LightGBM rows from the standard paper's comparison table. Re-run `train_all.py` afterwards.

---

## 2. Team and modules

| Member | Sub-area | Code | Dashboard page |
|---|---|---|---|
| **Lekhya** | Driver Behaviour Detection | `src/behavior.py` | Driver Behaviour |
| **Lekhya** | Crash Risk & Accident Severity | `src/crash.py`, `src/mtftt.py` | Crash Severity |
| **Member 2** | Alcohol Impairment Detection | `src/alcohol.py` | Alcohol Detection |
| **Member 3** | AI Integration: fusion, alerts, safer routes | `src/fusion.py`, `src/server.py`, `web/` | Live Risk Monitor, Hotspots & Safe Routes |

Names and contribution lists are in **`team.json`**. Edit them and refresh the browser; no retraining needed.

---

## 3. Datasets

| Module | Dataset | Status |
|---|---|---|
| Behaviour | **Driver Behavior Dataset**: Ferreira Júnior et al., PLoS ONE 2017 (github.com/jair-jr/driverBehaviorDataset). 4 real trips, 69 video-labelled events, accelerometer, linear accelerometer, gyroscope | ✅ Real, bundled |
| Crash | **US-Accidents** (Moosavi et al.), cleaned subset of 314,285 records 2016–2018 (github.com/nkhanna94/Road-Accident-Analysis-And-Traffic-Severity-Prediction) | ✅ Real, bundled |
| Crash | **IEEE DataPort**: *Uncovering Influential Mechanisms of Traffic Accident Severity through Explainable ML* (Chen Ma, DOI 10.21227/79s6-qr16, UK STATS19) | ⏳ Needs an IEEE DataPort login. Put the files in `data/crash/stats19_ieee_dataport/` and re-run `train_all.py` |
| Alcohol | **IEEE DataPort**: *Alcohol Detection in Drivers by Sensors and Computer Vision* (Rosero-Montalvo & Peluffo-Ordóñez, DOI 10.21227/mnja-3553, 390 samples) | ⏳ Needs a login. Until then a **simulated** file with the same 5 attributes and 217/173 class split is used, and the dashboard marks it "Simulated". Put the real CSV in `data/alcohol/` and re-run `train_all.py` |

---

## 4. What we took from the standard paper

*AI-Driven Accident Risk Prediction and Safer Route Recommendation Using Multi-Model Data*, CCIC 2026, DOI 10.1109/CCIC68129.2026.11486065

**Taken from the paper:**
- Multi-source data: accident history, weather, road features, location.
- Pre-processing with encoding and normalisation, plus engineered features.
- Low / Medium / High risk target.
- The same classifier comparison (LR, RF, Extra Trees, GB, AdaBoost, boosted trees incl. XGBoost / LightGBM), 80/20 split and grid search.
- The same metrics: accuracy, macro-F1, ROC-AUC, confusion matrix, feature importance, time.
- The best-balanced boosted model is the one deployed.
- REST API + dashboard, heat map, threshold alerts to an emergency contact, and safer-route recommendation.

**Our extensions:**
- The driver's own state (behaviour + alcohol) is fused with road context.
- Enhanced CCP tree (Review-2 Paper 3).
- Multi-Task FT-Transformer with uncertainty-weighted loss (Review-2 Paper 4), written from scratch in NumPy and gradient-checked.
- Leave-one-trip-out behaviour evaluation.
- Live replay of real trips through all models.

---

## 5. Project layout

```
train_all.py            trains everything, writes models/ and reports/
team.json               team names, roles, contributions (shown in the UI)
src/behavior.py         behaviour detection (windows, FFT features, RBF-SVM / RF)
src/alcohol.py          alcohol detection (classifier comparison, 5-fold CV)
src/crash.py            crash risk: standard-paper comparison, Enhanced CCP, MT-FTT, STATS19 loader
src/mtftt.py            Multi-Task FT-Transformer (NumPy, manual backprop, gradient check)
src/fusion.py           risk fusion, alert actions, safety-weighted routing
src/server.py           web server + JSON API (standard library only)
web/                    dashboard (HTML / CSS / JS, no build step)
data/                   datasets
reports/                metrics JSON produced by training
```

`python src/mtftt.py` runs the numerical gradient check of the transformer (prints an error around 1e-8).

## 6. Notes on honesty of results
- Every number on the dashboard is computed by `train_all.py` on the data in `data/`. Nothing is typed in by hand except the standard paper's own Table I, which is shown for comparison.
- **Crash models:** use only conditions known *before* a crash, on a strongly imbalanced 3-class target. Their accuracy is lower than the standard paper's, and the dashboard explains why.
- **Behaviour models:** evaluated on trips they never saw (leave-one-trip-out), with only 69 labelled events.
- **Alcohol results:** come from simulated data until the IEEE DataPort file is added.
- **Simulated features:** emergency SMS/calls are simulated, and routing runs on a city grid built from real accident locations, not a real road network (see Future Scope).
