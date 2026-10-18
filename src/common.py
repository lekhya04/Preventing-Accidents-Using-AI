import json, os, time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)


def metrics(y_true, y_pred, proba=None, labels=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if proba is not None:
        try:
            if proba.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
            else:
                out["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        except Exception:
            out["roc_auc"] = None
    out["confusion"] = confusion_matrix(y_true, y_pred, labels=labels).tolist() if labels is not None else confusion_matrix(y_true, y_pred).tolist()
    return out


def save_json(obj, name):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(REPORTS, name), "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def load_json(name, default=None):
    p = os.path.join(REPORTS, name)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self
    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t


def optional_boosters():
    """XGBoost / LightGBM are used when installed (as in the standard paper); otherwise skipped."""
    out = {}
    try:
        from xgboost import XGBClassifier
        out["XGBoost"] = lambda n_cls: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                                                     colsample_bytree=0.9, tree_method="hist", n_jobs=-1,
                                                     eval_metric="mlogloss" if n_cls > 2 else "logloss")
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        out["LightGBM"] = lambda n_cls: LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, verbose=-1)
    except Exception:
        pass
    return out
