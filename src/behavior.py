"""
Sub-area 1 (Lekhya): Driver Behaviour Detection

Dataset (real): Driver Behavior Dataset - Ferreira Júnior et al., "Driver behavior profiling: An
investigation with different smartphone sensors and machine learning", PLoS ONE 2017.
(github.com/jair-jr/driverBehaviorDataset). 4 trips, 2011 Honda Civic, smartphone accelerometer,
linear accelerometer and gyroscope, 69 video-labelled driving events in 7 classes.
This is the same public benchmark used by Review-1 Paper 2 (Hong et al., Edge-Gated Attention + FFT).

Method (taken from the Review-1 papers)
  * sliding windows over 9 IMU channels, time-domain statistics          (Review-1 Paper 1 / 3)
  * FFT spectral band energies per channel                               (Review-1 Paper 2, FFT branch)
  * Gaussian (RBF) SVM classifier                                        (Review-1 Paper 1)
  * Random Forest comparison
  * evaluation: leave-one-trip-out cross validation, window- and event-level scores
"""
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import DATA, MODELS, metrics, save_json, Timer

BDIR = os.path.join(DATA, "driver_behavior")
FS = 50.0           # resample rate (Hz)
WIN = 2.0           # window length (s)
STEP = 0.25         # step inside labelled events (s)

LABELS = {
    "evento_nao_agressivo": "Non-aggressive",
    "aceleracao_agressiva": "Aggressive acceleration",
    "freada_agressiva": "Aggressive braking",
    "curva_esquerda_agressiva": "Aggressive left turn",
    "curva_direita_agressiva": "Aggressive right turn",
    "troca_faixa_esquerda_agressiva": "Aggressive left lane change",
    "troca_faixa_direita_agressiva": "Aggressive right lane change",
    "cruise": "Normal cruising",
}
CLASSES = list(LABELS.values())
SAFE = {"Non-aggressive", "Normal cruising"}
CHANNELS = ["acc_x", "acc_y", "acc_z", "lin_x", "lin_y", "lin_z", "gyr_x", "gyr_y", "gyr_z"]
BANDS = [(0, 0.5), (0.5, 1.5), (1.5, 3), (3, 6), (6, 12)]


def load_trip(tid):
    d = os.path.join(BDIR, f"trip_{tid}")
    parts = {"acc": "acelerometro_terra", "lin": "aceleracaoLinear_terra", "gyr": "giroscopio_terra"}
    raw = {k: pd.read_csv(os.path.join(d, f + ".csv")) for k, f in parts.items()}
    t0 = min(r.uptimeNanos.iloc[0] for r in raw.values())
    t1 = min(r.uptimeNanos.iloc[-1] for r in raw.values())
    grid = np.arange(0, (t1 - t0) / 1e9, 1 / FS)
    sig = np.zeros((len(grid), 9))
    for j, (k, r) in enumerate(raw.items()):
        ts = (r.uptimeNanos.values - t0) / 1e9
        for a, ax in enumerate("xyz"):
            sig[:, j * 3 + a] = np.interp(grid, ts, r[ax].values)
    gt = pd.read_csv(os.path.join(d, "groundTruth.csv"))
    gt.columns = [c.strip() for c in gt.columns]
    gt = gt[gt["evento"].isin(LABELS)].rename(columns={"evento": "event", "inicio": "start", "fim": "end"})
    gt["label"] = gt["event"].map(LABELS)
    return grid, sig, gt.reset_index(drop=True)


def window_features(w):
    """w: (n_samples, 9). Time-domain stats + FFT band energies + jerk."""
    f = []
    f += list(w.mean(0)); f += list(w.std(0)); f += list(w.min(0)); f += list(w.max(0))
    f += list(w.max(0) - w.min(0)); f += list(np.sqrt((w ** 2).mean(0)))
    jerk = np.diff(w[:, 3:6], axis=0) * FS
    f += list(np.abs(jerk).mean(0))
    spec = np.abs(np.fft.rfft(w - w.mean(0), axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(w), 1 / FS)
    tot = spec.sum(0) + 1e-9
    for lo, hi in BANDS:
        f += list(spec[(freqs >= lo) & (freqs < hi)].sum(0) / tot)
    f += [np.linalg.norm(w[:, 3:6], axis=1).max(), np.abs(w[:, 8]).max()]       # peak linear accel, peak yaw rate
    return np.array(f, dtype=np.float32)


def feature_names():
    n = []
    for stat in ["mean", "std", "min", "max", "range", "rms"]:
        n += [f"{c}_{stat}" for c in CHANNELS]
    n += [f"jerk_{a}" for a in "xyz"]
    for lo, hi in BANDS:
        n += [f"{c}_fft_{lo}-{hi}Hz" for c in CHANNELS]
    return n + ["peak_lin_acc", "peak_yaw_rate"]


def build_windows(tid, rng):
    grid, sig, gt = load_trip(tid)
    n = int(WIN * FS); X, y, ev = [], [], []
    for i, r in gt.iterrows():
        dur = r.end - r.start
        starts = np.arange(r.start - 0.25 * WIN, max(r.start - 0.25 * WIN, r.end - 0.75 * WIN) + 1e-9, STEP) if dur >= WIN * 0.75 else [r.start + dur / 2 - WIN / 2]
        for s in starts:
            a = int(s * FS)
            if a < 0 or a + n > len(sig): continue
            X.append(window_features(sig[a:a + n])); y.append(r.label); ev.append(f"{tid}-{i}")
    # normal cruising: windows at least 3 s away from every labelled event (assumption: unlabelled = normal driving)
    busy = np.zeros(len(grid), bool)
    for _, r in gt.iterrows():
        busy[max(0, int((r.start - 3) * FS)):int((r.end + 3) * FS)] = True
    free = np.where(~busy[: len(grid) - n])[0]
    for k, a in enumerate(rng.choice(free, size=min(len(free), 60), replace=False)):
        if busy[a:a + n].any(): continue
        X.append(window_features(sig[a:a + n])); y.append("Normal cruising"); ev.append(f"{tid}-cruise{k // 6}")
    return np.array(X), np.array(y), np.array(ev)


def trips():
    return sorted(int(os.path.basename(p).split("_")[1]) for p in glob.glob(os.path.join(BDIR, "trip_*")))


def make_models():
    return {
        "RBF-SVM (Gaussian kernel)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, class_weight="balanced", random_state=0)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=1, class_weight="balanced", random_state=0, n_jobs=-1),
    }


def train(log=print):
    rng = np.random.default_rng(0)
    data = {t: build_windows(t, rng) for t in trips()}
    X = np.vstack([d[0] for d in data.values()]); y = np.concatenate([d[1] for d in data.values()])
    ev = np.concatenate([d[2] for d in data.values()]); trip_of = np.concatenate([[t] * len(d[1]) for t, d in data.items()])
    labs = [c for c in CLASSES if c in set(y)]
    log(f"[behaviour] windows={len(y)}  events={len(set(e for e in ev if 'cruise' not in e))}  features={X.shape[1]}")
    report = {"dataset": {"name": "Driver Behavior Dataset (Ferreira Júnior et al., PLoS ONE 2017)", "trips": trips(),
                          "n_windows": int(len(y)), "n_events": int(len(set(e for e in ev if "cruise" not in e))),
                          "class_windows": {c: int((y == c).sum()) for c in labs}, "sensors": "Accelerometer, linear accelerometer, gyroscope (3 axes each), resampled to 50 Hz",
                          "window": f"{WIN:.0f} s windows, {STEP} s step", "n_features": int(X.shape[1])}, "models": {}}
    best = None
    for name, _ in make_models().items():
        yp, pp, yt, evs = [], [], [], []
        with Timer() as tm:
            for t in trips():                                  # leave-one-trip-out
                trn, tst = trip_of != t, trip_of == t
                m = make_models()[name].fit(X[trn], y[trn])
                P = np.zeros((tst.sum(), len(labs))); pr = m.predict_proba(X[tst])
                for j, c in enumerate(m.classes_): P[:, labs.index(c)] = pr[:, j]
                yp += [labs[i] for i in P.argmax(1)]; pp.append(P); yt += list(y[tst]); evs += list(ev[tst])
        yp, yt, pp, evs = np.array(yp), np.array(yt), np.vstack(pp), np.array(evs)
        win = metrics(yt, yp, labels=labs)
        # event-level: average window probabilities per event
        e_true, e_pred = [], []
        for e in np.unique(evs):
            k = evs == e; e_true.append(yt[k][0]); e_pred.append(labs[pp[k].mean(0).argmax()])
        evm = metrics(np.array(e_true), np.array(e_pred), labels=labs)
        safe_true = np.isin(yt, list(SAFE)); safe_pred = np.isin(yp, list(SAFE))
        binary = float((safe_true == safe_pred).mean())
        report["models"][name] = {"window": win, "event": evm, "aggressive_vs_safe_accuracy": binary, "seconds": round(tm.s, 1)}
        log(f"   {name:<28s} window acc={win['accuracy']:.3f} F1={win['macro_f1']:.3f} | event acc={evm['accuracy']:.3f} | aggressive-vs-safe={binary:.3f}")
        if best is None or evm["macro_f1"] > report["models"][best]["event"]["macro_f1"]:
            best = name
    report["labels"] = labs; report["deployed"] = best
    final = make_models()[best].fit(X, y)
    rf = make_models()["Random Forest"].fit(X, y)
    imp = sorted(zip(feature_names(), rf.feature_importances_), key=lambda kv: -kv[1])[:15]
    report["top_features"] = [{"feature": a, "importance": float(b)} for a, b in imp]
    with open(os.path.join(MODELS, "behavior.pkl"), "wb") as f:
        pickle.dump({"model": final, "labels": labs, "name": best}, f)
    save_json(report, "behavior.json")
    log(f"   -> deployed: {best}")
    for t in trips():
        save_json(replay(t, final), f"replay_trip_{t}.json")
    return report


def replay(tid, model=None, step=0.5):
    """Run the deployed model over a whole real trip (as if streaming live)."""
    if model is None:
        with open(os.path.join(MODELS, "behavior.pkl"), "rb") as f:
            model = pickle.load(f)["model"]
    grid, sig, gt = load_trip(tid)
    n = int(WIN * FS); out = []
    starts = list(range(0, len(sig) - n, int(step * FS)))
    P = model.predict_proba(np.stack([window_features(sig[a:a + n]) for a in starts]))
    for a, pr in zip(starts, P):
        t = (a + n / 2) / FS
        k = int(pr.argmax()); lab = model.classes_[k]
        truth = gt[(gt.start <= t) & (gt.end >= t)]
        safe_p = float(sum(pr[j] for j, c in enumerate(model.classes_) if c in SAFE))
        out.append({"t": round(t, 2), "pred": lab, "conf": round(float(pr[k]), 3), "aggr": round(1 - safe_p, 3),
                    "truth": truth.label.iloc[0] if len(truth) else None,
                    "ax": round(float(sig[a + n // 2, 3]), 3), "ay": round(float(sig[a + n // 2, 4]), 3), "gz": round(float(sig[a + n // 2, 8]), 3)})
    return {"trip": tid, "duration": float(grid[-1]), "events": gt[["label", "start", "end"]].to_dict("records"), "stream": out}


if __name__ == "__main__":
    train()
