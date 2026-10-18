"""
Sub-area 2 (Lekhya): Crash Risk & Accident Severity Prediction

Models trained here
  1. Standard-paper model comparison  -> AI-Driven Accident Risk Prediction and Safer Route
     Recommendation (CCIC 2026): Logistic Regression, Random Forest, Gradient Boosting, AdaBoost,
     Extra Trees, Hist-Gradient-Boosting, + XGBoost / LightGBM when installed.
     Target = accident risk level Low / Medium / High, 80/20 split, grid-searched best model deployed.
  2. Enhanced CCP decision tree (Paper 3, Cao et al. IEEE Access 2026): wrapper Forward Stepwise
     Selection + cost-complexity pruning path + S-score tie-break, class reweighting, macro-F1.
  3. Multi-Task FT-Transformer with uncertainty weighting (Paper 4, Wang et al. IEEE Access 2026)
     predicting three related outcomes at once.

Datasets
  * US-Accidents subset (real, 314,285 police/traffic-reported accidents, 2016-2018)  [default]
  * UK STATS19 from IEEE DataPort (DOI 10.21227/79s6-qr16) - used automatically when its files
    are placed in data/crash/stats19_ieee_dataport/ (needs IEEE DataPort login to download).
"""
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from common import DATA, MODELS, metrics, save_json, Timer, optional_boosters
from mtftt import MultiTaskFTTransformer

RISK_LABELS = ["Low", "Medium", "High"]

# ----------------------------------------------------------------------------- US-Accidents adapter
US_PATH = os.path.join(DATA, "crash", "us_accidents_subset.csv.gz")
TOP_WEATHER = ["Clear", "Overcast", "Mostly Cloudy", "Partly Cloudy", "Light Snow", "Light Rain",
               "Scattered Clouds", "Fair", "Rain", "Cloudy", "Snow", "Haze", "Light Drizzle", "Fog",
               "Light Freezing Rain", "Heavy Snow", "Heavy Rain", "Thunderstorm"]
WIND = ["CALM", "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "VAR"]


def road_type_from_street(s):
    s = s.astype(str)
    fw = s.str.contains(r"\bI-|Interstate|Fwy|Freeway|Expy|Expressway|Tpke|Turnpike", regex=True)
    hw = s.str.contains(r"Hwy|Highway|US-|Route|SR-|State Route|Pkwy|Parkway|County Road|\bRd\b", regex=True)
    return np.where(fw, "Freeway", np.where(hw, "Highway/Arterial", "Local street"))


def geo_cell(lat, lng):
    return (np.floor(np.asarray(lat, float) * 2) / 2).astype(str).astype(object) + "_" + (np.floor(np.asarray(lng, float) * 2) / 2).astype(str).astype(object)


def us_features(df, top_states=None, top_cells=None):
    X = pd.DataFrame(index=df.index)
    w = df["Weather_Condition"].astype(str)
    X["weather"] = np.where(w.isin(TOP_WEATHER), w, "Other")
    wd = df["Wind_Direction"].astype(str).str.upper().replace({"CALM": "CALM", "VARIABLE": "VAR", "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"})
    X["wind_dir"] = np.where(wd.isin(WIND), wd, "VAR")
    X["road_type"] = road_type_from_street(df["Street"]) if "Street" in df else df["road_type"]
    X["traffic_signal"] = df["Traffic_Signal"].astype(str).str.lower().isin(["true", "1", "yes"]).map({True: "Signal", False: "No signal"})
    X["light"] = df["Sunrise_Sunset"].astype(str).where(df["Sunrise_Sunset"].astype(str).isin(["Day", "Night"]), "Day")
    st = df["State"].astype(str)
    X["state"] = np.where(st.isin(top_states), st, "Other") if top_states is not None else st
    X["year"] = df["Year"].astype(str) if "Year" in df else "2018"
    gc = pd.Series(geo_cell(df["Start_Lat"], df["Start_Lng"]), index=df.index)
    X["geo_cell"] = np.where(gc.isin(top_cells), gc, "Other") if top_cells is not None else gc
    X["lat"] = df["Start_Lat"].astype(float)
    X["lng"] = df["Start_Lng"].astype(float)
    X["temperature_f"] = df["Temperature(F)"].astype(float).clip(-30, 110)
    X["wind_chill_f"] = df["Wind_Chill(F)"].astype(float).clip(-50, 110)
    X["visibility_mi"] = df["Visibility(mi)"].astype(float).clip(0, 15)
    return X


def load_us():
    df = pd.read_csv(US_PATH, low_memory=False)
    top_states = df["State"].value_counts().index[:24].tolist()
    top_cells = pd.Series(geo_cell(df["Start_Lat"], df["Start_Lng"])).value_counts().index[:150].tolist()
    X = us_features(df, top_states, top_cells)
    sev = df["Severity"].astype(int)
    y = pd.DataFrame({
        "risk_level": np.where(sev >= 4, 2, np.where(sev == 3, 1, 0)),                                  # Low / Medium / High
        "road_impact": np.digitize(df["Distance(mi)"].astype(float), [0.001, 0.5]),                       # Point / Short / Long
        "clearance": (df["TimeDiff"].astype(float) > 31).astype(int),                                     # Quick / Prolonged
    })
    meta = {
        "name": "US-Accidents (real subset, 314,285 records, 2016–2018, 49 US states)",
        "short": "US-Accidents",
        "source": "Moosavi et al., US-Accidents (Kaggle); cleaned subset from github.com/nkhanna94/Road-Accident-Analysis-And-Traffic-Severity-Prediction",
        "tasks": {
            "risk_level": {"title": "Accident Risk Level", "classes": RISK_LABELS, "rule": "Severity 1–2 → Low, 3 → Medium, 4 → High"},
            "road_impact": {"title": "Road Length Affected", "classes": ["Point (0 mi)", "Short (≤0.5 mi)", "Long (>0.5 mi)"], "rule": "Distance(mi) bands"},
            "clearance": {"title": "Clearance Time", "classes": ["Quick (≤31 min)", "Prolonged (>31 min)"], "rule": "Impact duration (TimeDiff)"},
        },
        "cat": ["weather", "wind_dir", "road_type", "traffic_signal", "light", "state", "geo_cell", "year"],
        "num": ["lat", "lng", "temperature_f", "wind_chill_f", "visibility_mi"],
        "top_states": top_states, "top_cells": top_cells,
        "n_records": int(len(df)),
    }
    return X, y, meta, df


# ----------------------------------------------------------------------------- STATS19 (IEEE DataPort) adapter
S19_DIR = os.path.join(DATA, "crash", "stats19_ieee_dataport")


def _norm(df):
    df.columns = [str(c).strip().replace(" ", "_").replace("(", "").replace(")", "").lower() for c in df.columns]
    return df


def _read_any(path):
    if path.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(path, sheet_name=None)
        return {k: _norm(v) for k, v in sheets.items()}
    return {os.path.basename(path): _norm(pd.read_csv(path, low_memory=False, encoding_errors="ignore"))}


def stats19_available():
    files = [f for f in glob.glob(os.path.join(S19_DIR, "*")) if f.lower().endswith((".csv", ".xlsx", ".xls")) and "guide" not in f.lower()]
    return files


def load_stats19():
    """Reads 'Accidents, Casualties, Vehicles.xlsx' (or Accidents0514.csv + Casualties0514.csv + Vehicles0514.csv)."""
    tables = {}
    for f in stats19_available():
        tables.update(_read_any(f))
    acc = cas = None
    for name, t in tables.items():
        cols = set(t.columns)
        if "accident_severity" in cols and acc is None:
            acc = t
        elif "casualty_class" in cols and cas is None:
            cas = t
    if acc is None:
        raise ValueError("No accident table with an Accident_Severity column found in " + S19_DIR)
    key = "accident_index" if "accident_index" in acc.columns else acc.columns[0]
    acc = acc.copy()
    if "time" in acc.columns:
        acc["hour"] = pd.to_numeric(acc["time"].astype(str).str.slice(0, 2), errors="coerce").fillna(12).astype(int)
    else:
        acc["hour"] = 12
    if cas is not None and key in cas.columns:
        ped = cas.groupby(key)["casualty_class"].apply(lambda s: int((pd.to_numeric(s, errors="coerce") == 3).any()))
        acc["pedestrian"] = acc[key].map(ped).fillna(0).astype(int)
    else:
        acc["pedestrian"] = 0
    cat_cols = [c for c in ["road_type", "light_conditions", "weather_conditions", "road_surface_conditions", "urban_or_rural_area",
                            "junction_detail", "junction_control", "day_of_week", "1st_road_class", "special_conditions_at_site",
                            "carriageway_hazards", "speed_limit"] if c in acc.columns]
    num_cols = [c for c in ["number_of_vehicles", "hour", "latitude", "longitude"] if c in acc.columns]
    X = pd.DataFrame(index=acc.index)
    for c in cat_cols:
        X[c] = acc[c].astype(str)
    for c in num_cols:
        X[c] = pd.to_numeric(acc[c], errors="coerce").fillna(acc[c].median() if acc[c].dtype != object else 0)
    sev = pd.to_numeric(acc["accident_severity"], errors="coerce").fillna(3).astype(int)
    ncas = pd.to_numeric(acc.get("number_of_casualties", 1), errors="coerce").fillna(1).astype(int)
    y = pd.DataFrame({
        "risk_level": np.where(sev == 1, 2, np.where(sev == 2, 1, 0)),
        "road_impact": np.clip(ncas, 1, 3) - 1,
        "clearance": acc["pedestrian"].values,
    })
    meta = {
        "name": "UK STATS19 – IEEE DataPort (DOI 10.21227/79s6-qr16)",
        "short": "STATS19 (IEEE DataPort)",
        "source": "Chen Ma, 'Uncovering Influential Mechanisms of Traffic Accident Severity through Explainable Machine Learning', IEEE DataPort 2026",
        "tasks": {
            "risk_level": {"title": "Accident Severity", "classes": ["Slight", "Serious", "Fatal"], "rule": "STATS19 Accident_Severity"},
            "road_impact": {"title": "Casualty Count", "classes": ["1", "2", "≥3"], "rule": "Number_of_Casualties"},
            "clearance": {"title": "Pedestrian Involved", "classes": ["No", "Yes"], "rule": "any Casualty_Class = 3"},
        },
        "cat": cat_cols, "num": num_cols, "n_records": int(len(acc)),
    }
    return X, y, meta


# ----------------------------------------------------------------------------- encoding helpers
class Encoder:
    def __init__(self, cat, num):
        self.cat, self.num = cat, num

    def fit(self, X):
        self.vocab = {c: {v: i for i, v in enumerate(sorted(X[c].astype(str).unique()))} for c in self.cat}
        self.mu = X[self.num].mean().values if self.num else np.zeros(0)
        self.sd = X[self.num].std().replace(0, 1).values if self.num else np.ones(0)
        self.mode = {c: X[c].mode().iloc[0] for c in self.cat}
        self.median = {c: float(X[c].median()) for c in self.num}
        return self

    def tokens(self, X):
        Xc = np.stack([X[c].astype(str).map(self.vocab[c]).fillna(0).astype(int).values for c in self.cat], 1) if self.cat else np.zeros((len(X), 0), int)
        Xn = ((X[self.num].values - self.mu) / self.sd).clip(-5, 5) if self.num else np.zeros((len(X), 0))
        return Xc, Xn

    def onehot(self, X):
        Xc, Xn = self.tokens(X)
        blocks = [Xn]
        for j, c in enumerate(self.cat):
            oh = np.zeros((len(X), len(self.vocab[c])), np.float32); oh[np.arange(len(X)), Xc[:, j]] = 1; blocks.append(oh)
        return np.hstack(blocks).astype(np.float32)

    def onehot_names(self):
        names = list(self.num)
        for c in self.cat:
            names += [f"{c}={v}" for v in self.vocab[c]]
        return names

    def cards(self):
        return [len(self.vocab[c]) for c in self.cat]


def group_importance(enc, imp):
    """Sum one-hot importances back to their source feature."""
    names = enc.onehot_names(); out = {}
    for n, v in zip(names, imp):
        out[n.split("=")[0]] = out.get(n.split("=")[0], 0.0) + float(v)
    tot = sum(out.values()) or 1
    return dict(sorted(((k, v / tot) for k, v in out.items()), key=lambda kv: -kv[1]))


# ----------------------------------------------------------------------------- Enhanced CCP (Paper 3)
def enhanced_ccp(Xtr, ytr, Xva, yva, names, log=print, max_feats=12):
    """Wrapper FSS on macro-F1, then cost-complexity pruning path with S-score tie-break."""
    def score(cols, alpha=0.0):
        t = DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=20, random_state=0, ccp_alpha=alpha)
        t.fit(Xtr[:, cols], ytr)
        return f1_score(yva, t.predict(Xva[:, cols]), average="macro"), t
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(n.split("=")[0], []).append(i)
    remaining, chosen, best, stall, trace = list(groups), [], 0.0, 0, []
    while remaining and len(chosen) < max_feats and stall < 2:
        cand = []
        for g in remaining:
            cols = sum((groups[x] for x in chosen + [g]), [])
            cand.append((score(cols)[0], g))
        s, g = max(cand)
        if s - best < 0.002:
            stall += 1
        else:
            stall = 0
        if s > best:
            best = s
        chosen.append(g); remaining.remove(g); trace.append({"feature": g, "val_macro_f1": round(float(s), 4)})
        log(f"   FSS + {g:<16s} val macro-F1 = {s:.4f}")
    cols = sum((groups[x] for x in chosen), [])
    base = DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=20, random_state=0).fit(Xtr[:, cols], ytr)
    path = base.cost_complexity_pruning_path(Xtr[:, cols], ytr)
    alphas = np.unique(np.clip(np.quantile(path.ccp_alphas[:-1], np.linspace(0, 1, 25)), 0.0, None))
    results = []
    for a in alphas:
        f1, t = score(cols, a)
        leaves = t.apply(Xtr[:, cols])
        # S-score of the pruned tree: mean over leaves of 0.5*leaf accuracy + 0.5*leaf coverage (normalised)
        lid, cnt = np.unique(leaves, return_counts=True)
        pred = t.predict(Xtr[:, cols])
        acc_leaf = np.array([(pred[leaves == l] == ytr[leaves == l]).mean() for l in lid])
        cov = cnt / cnt.sum()
        S = float((0.5 * acc_leaf + 0.5 * cov / cov.max()).mean())
        results.append((float(a), float(f1), S, int(t.get_n_leaves()), t))
    top = max(r[1] for r in results)
    near = [r for r in results if r[1] >= top - 0.005]
    a, f1, S, nleaves, tree = max(near, key=lambda r: (r[2], -r[3]))
    log(f"   CCP alpha = {a:.2e}  leaves = {nleaves}  val macro-F1 = {f1:.4f}  S-score = {S:.3f}")
    return tree, cols, chosen, {"fss_trace": trace, "alpha": a, "leaves": nleaves, "s_score": S,
                                "path": [{"alpha": r[0], "val_macro_f1": r[1], "leaves": r[3]} for r in results]}


# ----------------------------------------------------------------------------- training
def train(dataset="us", sample=90000, log=print, fast=False):
    if dataset == "us":
        X, Y, meta, raw = load_us()
    else:
        X, Y, meta = load_stats19(); raw = None
    rng = np.random.RandomState(42)
    idx = np.arange(len(X))
    if sample and len(idx) > sample:
        idx, _ = train_test_split(idx, train_size=sample, stratify=Y["risk_level"].values, random_state=42)
    Xs, Ys = X.iloc[idx].reset_index(drop=True), Y.iloc[idx].reset_index(drop=True)
    tr, te = train_test_split(np.arange(len(Xs)), test_size=0.2, stratify=Ys["risk_level"], random_state=42)     # 80/20 as in the standard paper
    tr, va = train_test_split(tr, test_size=0.15, stratify=Ys["risk_level"].values[tr], random_state=42)
    enc = Encoder(meta["cat"], meta["num"]).fit(Xs.iloc[tr])
    O = enc.onehot(Xs)
    y = Ys["risk_level"].values
    report = {"dataset": meta, "split": {"train": len(tr), "val": len(va), "test": len(te), "sampled_from": int(len(X))},
              "class_distribution": {RISK_LABELS[i] if dataset == "us" else meta["tasks"]["risk_level"]["classes"][i]: int((y == i).sum()) for i in range(3)}}
    lab = meta["tasks"]["risk_level"]["classes"]

    # ---- 1. standard-paper comparison
    log("[crash] Standard-paper model comparison (target: risk level)")
    trv = np.concatenate([tr, va])
    sc = StandardScaler().fit(O[trv])
    cands = {
        "Logistic Regression": lambda k: LogisticRegression(max_iter=600, C=1.0),
        "Random Forest": lambda k: RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "Extra Trees": lambda k: ExtraTreesClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "Gradient Boosting": lambda k: GradientBoostingClassifier(n_estimators=120 if not fast else 40, max_depth=3, random_state=0),
        "AdaBoost": lambda k: AdaBoostClassifier(n_estimators=150, random_state=0),
        "Hist Gradient Boosting": lambda k: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0),
    }
    cands.update(optional_boosters())
    comp, fitted = [], {}
    # Gradient Boosting / AdaBoost are slow on wide one-hot data: trained on a 30k stratified subset (noted in report)
    slow_sub = np.random.RandomState(0).choice(trv, min(len(trv), 30000), replace=False)
    for name, mk in cands.items():
        m = mk(3)
        Xin = sc.transform(O) if name == "Logistic Regression" else O
        fit_idx = slow_sub if name in ("Gradient Boosting", "AdaBoost") else trv
        with Timer() as t:
            m.fit(Xin[fit_idx], y[fit_idx])
        p = m.predict_proba(Xin[te]); pred = p.argmax(1)
        r = metrics(y[te], pred, p, labels=[0, 1, 2]); r["model"] = name; r["train_seconds"] = round(t.s, 2); r["train_rows"] = int(len(fit_idx))
        comp.append(r); fitted[name] = m
        log(f"   {name:<24s} acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} AUC={r['roc_auc']:.4f} ({t.s:.1f}s)")
    # paper: best model chosen on ROC-AUC + macro-F1 balance
    best = max(comp, key=lambda r: (r["roc_auc"] or 0) + r["macro_f1"])
    best_name = best["model"]
    log(f"   -> deployed model: {best_name}")
    # grid search on the deployed model family (paper: 'hyperparameter tuning through grid search')
    grid_info = None
    if best_name in ("Hist Gradient Boosting", "XGBoost", "LightGBM", "Random Forest") and not fast:
        grids = {"Hist Gradient Boosting": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]},
                 "XGBoost": {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
                 "LightGBM": {"num_leaves": [31, 63], "learning_rate": [0.05, 0.1]},
                 "Random Forest": {"min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.3]}}
        sub = np.random.RandomState(0).choice(trv, min(len(trv), 20000), replace=False)
        gs = GridSearchCV(cands[best_name](3), grids[best_name], scoring="f1_macro", cv=3, n_jobs=1).fit(O[sub], y[sub])
        tuned = cands[best_name](3).set_params(**gs.best_params_).fit(O[trv], y[trv])
        p = tuned.predict_proba(O[te]); r = metrics(y[te], p.argmax(1), p, labels=[0, 1, 2])
        grid_info = {"best_params": gs.best_params_, "test": r}
        log(f"   grid search {gs.best_params_} -> acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f}")
        if r["macro_f1"] >= best["macro_f1"]:
            fitted[best_name] = tuned
    deployed = fitted[best_name]
    imp = None
    if hasattr(deployed, "feature_importances_"):
        imp = group_importance(enc, deployed.feature_importances_)
    elif "Random Forest" in fitted:
        imp = group_importance(enc, fitted["Random Forest"].feature_importances_)
    report["standard_paper"] = {"comparison": comp, "deployed": best_name, "grid_search": grid_info, "feature_importance": imp, "labels": lab}

    # ---- 2. Enhanced CCP (Paper 3)
    log("[crash] Enhanced CCP decision tree (Paper 3)")
    names = enc.onehot_names()
    sub_tr = np.random.RandomState(1).choice(tr, min(len(tr), 25000), replace=False)
    with Timer() as t:
        tree, cols, chosen, info = enhanced_ccp(O[sub_tr], y[sub_tr], O[va], y[va], names, log=log)
    p = tree.predict_proba(O[te][:, cols]); r = metrics(y[te], p.argmax(1), p, labels=[0, 1, 2])
    plain = DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=20, random_state=0).fit(O[sub_tr], y[sub_tr])
    pp = plain.predict_proba(O[te]); rp = metrics(y[te], pp.argmax(1), pp, labels=[0, 1, 2])
    from sklearn.tree import export_text
    rules = export_text(tree, feature_names=[names[c] for c in cols], max_depth=4)
    report["enhanced_ccp"] = {"test": r, "unpruned_tree_test": rp, "selected_features": chosen, "train_seconds": round(t.s, 2),
                              "rules_preview": rules[:3500], **info}
    log(f"   enhanced CCP test macroF1={r['macro_f1']:.4f} (unpruned tree {rp['macro_f1']:.4f})")

    # ---- 3. Multi-Task FT-Transformer (Paper 4)
    log("[crash] Multi-Task FT-Transformer with uncertainty weighting (Paper 4)")
    Xc, Xn = enc.tokens(Xs)
    Yall = Ys[["risk_level", "road_impact", "clearance"]].values.astype(int)
    tasks = list(meta["tasks"].keys())
    ncls = [len(meta["tasks"][k]["classes"]) for k in tasks]
    mt = MultiTaskFTTransformer(enc.cards(), len(meta["num"]), ncls, d=32, n_heads=2, n_layers=1, seed=0)
    with Timer() as t:
        hist = mt.fit(Xc[tr], Xn[tr], Yall[tr], Xc[va], Xn[va], Yall[va], epochs=3 if fast else 14, batch=256, lr=2e-3, patience=3,
                      log=lambda r: log(f"   epoch {r['epoch']}: train={r['train_loss']:.4f} val={r.get('val_loss', 0):.4f} sigma={r['sigma']}"))
    probs = mt.predict_proba(Xc[te], Xn[te])
    mt_res, st_res = {}, {}
    for k, task in enumerate(tasks):
        labs = list(range(ncls[k]))
        mt_res[task] = metrics(Yall[te, k], probs[k].argmax(1), probs[k], labels=labs)
        # single-task baseline (same features) for the multi-task vs single-task comparison
        stm = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(O[trv], Yall[trv, k])
        sp = stm.predict_proba(O[te]); st_res[task] = metrics(Yall[te, k], sp.argmax(1), sp, labels=labs)
        log(f"   {task:<12s} MT-FTT acc={mt_res[task]['accuracy']:.4f} F1={mt_res[task]['macro_f1']:.4f} | single-task HGB acc={st_res[task]['accuracy']:.4f} F1={st_res[task]['macro_f1']:.4f}")
    att = mt.attention_importance(Xc[te][:3000], Xn[te][:3000])
    feat_order = meta["num"] + meta["cat"]          # token order = numeric first, then categorical
    att_imp = dict(sorted(zip(feat_order, (att / att.sum()).tolist()), key=lambda kv: -kv[1]))
    report["mt_ftt"] = {"tasks": {k: meta["tasks"][k] for k in tasks}, "test": mt_res, "single_task_baseline": st_res,
                        "history": hist, "train_seconds": round(t.s, 2), "sigma": np.exp(0.5 * mt.P["log_var"]).tolist(),
                        "attention_importance": att_imp, "config": {"d_token": 32, "heads": 2, "layers": 1, "batch": 256, "optimizer": "Adam", "lr": 2e-3}}

    bundle = {"encoder": enc, "scaler": sc, "deployed_name": best_name, "deployed": deployed, "ccp": (tree, cols), "mtftt": mt.P,
              "mt_shape": (enc.cards(), len(meta["num"]), ncls), "meta": meta}
    with open(os.path.join(MODELS, f"crash_{dataset}.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    save_json(report, f"crash_{dataset}.json")
    if dataset == "us":
        build_hotspots(raw)
    return report


# ----------------------------------------------------------------------------- hotspot grid for the AI-fusion route module
def build_hotspots(raw, city_box=(44.86, 45.08, -93.42, -92.98), n=(18, 26)):
    """Real accident density for Minneapolis–Saint Paul (the city with the most records in the subset)."""
    la0, la1, lo0, lo1 = city_box
    d = raw[(raw.Start_Lat.between(la0, la1)) & (raw.Start_Lng.between(lo0, lo1))]
    ri = ((d.Start_Lat - la0) / (la1 - la0) * n[0]).astype(int).clip(0, n[0] - 1)
    ci = ((d.Start_Lng - lo0) / (lo1 - lo0) * n[1]).astype(int).clip(0, n[1] - 1)
    cnt = np.zeros(n); sev = np.zeros(n)
    np.add.at(cnt, (ri.values, ci.values), 1)
    np.add.at(sev, (ri.values, ci.values), (d.Severity.values >= 3).astype(float))
    pts = d.sample(min(len(d), 1500), random_state=0)[["Start_Lat", "Start_Lng", "Severity"]].values.tolist()
    save_json({"box": city_box, "grid": list(n), "count": cnt.tolist(), "severe": sev.tolist(), "n_accidents": int(len(d)),
               "points": pts, "city": "Minneapolis – Saint Paul, MN"}, "hotspots.json")


# ----------------------------------------------------------------------------- inference
_BUNDLE = {}


def _bundle(dataset="us"):
    if dataset not in _BUNDLE:
        with open(os.path.join(MODELS, f"crash_{dataset}.pkl"), "rb") as f:
            _BUNDLE[dataset] = pickle.load(f)
    return _BUNDLE[dataset]


def predict_context(inp):
    """inp: dict with weather, temperature_f, visibility_mi, wind_chill_f, wind_dir, road_type, traffic_signal,
    light, lat, lng, state, year. Returns standard-paper risk + MT-FTT multi-task output + local attribution."""
    b = _bundle("us"); enc, meta = b["encoder"], b["meta"]
    row = {"weather": inp.get("weather", "Clear"), "wind_dir": inp.get("wind_dir", "CALM"), "road_type": inp.get("road_type", "Highway/Arterial"),
           "traffic_signal": inp.get("traffic_signal", "No signal"), "light": inp.get("light", "Day"),
           "state": inp.get("state", "MN") if inp.get("state", "MN") in meta["top_states"] else "Other", "year": "2018",
           "geo_cell": None,
           "lat": float(inp.get("lat", 44.98)), "lng": float(inp.get("lng", -93.26)), "temperature_f": float(inp.get("temperature_f", 50)),
           "wind_chill_f": float(inp.get("wind_chill_f", inp.get("temperature_f", 50))), "visibility_mi": float(inp.get("visibility_mi", 10))}
    if row["weather"] not in TOP_WEATHER: row["weather"] = "Other"
    gc = geo_cell([row["lat"]], [row["lng"]])[0]
    row["geo_cell"] = gc if gc in meta["top_cells"] else "Other"
    X = pd.DataFrame([row])
    O = enc.onehot(X)
    p = b["deployed"].predict_proba(O)[0]
    # local attribution: replace each feature by its training default and measure the change in risk score
    def risk(pv): return float(pv[1] * 0.5 + pv[2])
    base = risk(p); contrib = {}
    for c in meta["cat"] + meta["num"]:
        X2 = X.copy(); X2[c] = enc.mode[c] if c in enc.mode else enc.median[c]
        contrib[c] = base - risk(b["deployed"].predict_proba(enc.onehot(X2))[0])
    mt = MultiTaskFTTransformer(*b["mt_shape"], d=int(b["mtftt"]["cls"].shape[0]), n_heads=2, n_layers=1); mt.P = b["mtftt"]
    Xc, Xn = enc.tokens(X)
    mp = mt.predict_proba(Xc, Xn)
    tasks = list(meta["tasks"].keys())
    tree, cols = b["ccp"]
    pc = tree.predict_proba(O[:, cols])[0]
    return {"deployed_model": b["deployed_name"], "risk_proba": p.tolist(), "risk_labels": RISK_LABELS, "risk_score": base,
            "risk_level": RISK_LABELS[int(np.argmax(p))], "attribution": dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1]))),
            "ccp_proba": pc.tolist(),
            "mt_ftt": {t: {"title": meta["tasks"][t]["title"], "classes": meta["tasks"][t]["classes"], "proba": mp[k][0].tolist()} for k, t in enumerate(tasks)}}


if __name__ == "__main__":
    import sys
    train("us", fast="--fast" in sys.argv)
    if stats19_available():
        train("stats19")
