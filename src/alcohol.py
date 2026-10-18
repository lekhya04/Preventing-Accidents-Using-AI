"""
Sub-area 3 (Member 2): Alcohol Impairment Detection

Dataset: "Alcohol Detection in Drivers by Sensors and Computer Vision" - P. D. Rosero-Montalvo,
D. Peluffo-Ordóñez, IEEE DataPort, DOI 10.21227/mnja-3553  (390 instances: 217 no-alcohol, 173 alcohol)
Attributes: cabin alcohol concentration (MQ-3, ml/L), cabin temperature (°C), face temperature min and
max (MLX90621 thermal sensor, °C), pupil ratio (Raspberry Pi camera); class 1 = no alcohol, 2 = alcohol.

The file needs an IEEE DataPort login, so it is NOT bundled. Put the CSV in data/alcohol/ and run
train_all.py - the loader picks it up automatically. Until then the module runs on a SIMULATED dataset
with the same schema and class balance, and the dashboard labels every result as "simulated".
"""
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from common import DATA, MODELS, metrics, save_json, Timer, optional_boosters

ADIR = os.path.join(DATA, "alcohol")
FEATURES = ["alcohol_ml_l", "cabin_temp_c", "face_temp_min_c", "face_temp_max_c", "pupil_ratio"]
NICE = {"alcohol_ml_l": "Cabin alcohol (MQ-3, ml/L)", "cabin_temp_c": "Cabin temperature (°C)",
        "face_temp_min_c": "Face temp. min (°C)", "face_temp_max_c": "Face temp. max (°C)", "pupil_ratio": "Pupil ratio"}


def simulate(seed=7):
    """Schema-matched stand-in: 217 sober / 173 alcohol, overlapping distributions (NOT real measurements)."""
    r = np.random.default_rng(seed)
    def block(n, alc):
        conc = r.gamma(2.0, 0.025, n) if not alc else r.uniform(0.04, 0.55, n) + r.normal(0, 0.03, n)
        env = r.normal(25, 2.2, n)
        fmin = r.normal(30.4 + (0.7 if alc else 0), 0.9, n) + 0.08 * (env - 25)
        fmax = fmin + r.normal(4.0 + (0.35 if alc else 0), 0.5, n)
        pup = r.normal(0.32 + (0.035 if alc else 0), 0.045, n)
        return pd.DataFrame({"alcohol_ml_l": np.clip(conc, 0, None), "cabin_temp_c": env, "face_temp_min_c": fmin,
                             "face_temp_max_c": fmax, "pupil_ratio": np.clip(pup, 0.15, 0.6), "label": int(alc)})
    return pd.concat([block(217, False), block(173, True)]).sample(frac=1, random_state=seed).reset_index(drop=True)


def load():
    files = [f for f in glob.glob(os.path.join(ADIR, "*.csv")) if "simulated" not in os.path.basename(f).lower()]
    if files:
        d = pd.read_csv(files[0])
        num = d.select_dtypes("number")
        cls_col = [c for c in d.columns if "class" in str(c).lower()]
        y = d[cls_col[0]] if cls_col else num.iloc[:, -1]
        X = num.drop(columns=[y.name], errors="ignore").iloc[:, :5]
        X.columns = FEATURES[: X.shape[1]]
        lab = (y.astype(float) >= 2).astype(int) if y.max() >= 2 else y.astype(int)
        out = X.copy(); out["label"] = lab.values
        return out, {"real": True, "file": os.path.basename(files[0])}
    d = simulate(); d.to_csv(os.path.join(ADIR, "alcohol_SIMULATED_schema_demo.csv"), index=False)
    return d, {"real": False, "file": "alcohol_SIMULATED_schema_demo.csv"}


def train(log=print):
    d, src = load()
    X, y = d[FEATURES].values, d["label"].values
    models = {
        "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=0)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=0),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, random_state=0),
    }
    models.update({k: v(2) for k, v in optional_boosters().items()})
    cv = StratifiedKFold(5, shuffle=True, random_state=0); res = {}
    for name, m in models.items():
        with Timer() as t:
            p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")
        r = metrics(y, p.argmax(1), p, labels=[0, 1]); r["seconds"] = round(t.s, 2); res[name] = r
        log(f"   {name:<24s} acc={r['accuracy']:.3f} F1={r['macro_f1']:.3f} AUC={r['roc_auc']:.3f}")
    best = max(res, key=lambda k: (res[k]["macro_f1"], res[k]["roc_auc"] or 0))
    final = models[best].fit(X, y)
    rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X, y)
    stats = {f: {"sober_mean": float(d.loc[d.label == 0, f].mean()), "alcohol_mean": float(d.loc[d.label == 1, f].mean()),
                 "min": float(d[f].min()), "max": float(d[f].max())} for f in FEATURES}
    report = {"source": src, "n": int(len(d)), "class_counts": {"No alcohol": int((y == 0).sum()), "Alcohol present": int((y == 1).sum())},
              "cv": "5-fold stratified cross-validation", "models": res, "deployed": best,
              "feature_importance": {NICE[f]: float(v) for f, v in sorted(zip(FEATURES, rf.feature_importances_), key=lambda kv: -kv[1])},
              "feature_stats": stats, "nice": NICE,
              "samples": d.sample(12, random_state=3).round(3).to_dict("records")}
    with open(os.path.join(MODELS, "alcohol.pkl"), "wb") as f:
        pickle.dump({"model": final, "name": best, "real": src["real"]}, f)
    save_json(report, "alcohol.json")
    log(f"   -> deployed: {best}  (data: {'REAL IEEE DataPort file' if src['real'] else 'SIMULATED schema demo'})")
    return report


_M = None


def predict(vals):
    global _M
    if _M is None:
        with open(os.path.join(MODELS, "alcohol.pkl"), "rb") as f:
            _M = pickle.load(f)
    x = np.array([[float(vals.get(f, 0)) for f in FEATURES]])
    p = float(_M["model"].predict_proba(x)[0, 1])
    return {"p_alcohol": p, "label": "Alcohol detected" if p >= 0.5 else "No alcohol", "model": _M["name"], "real_data": _M["real"]}


if __name__ == "__main__":
    train()
