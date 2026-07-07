"""Experiment 2: train the repo predictor + compare candidate model families.

Trains: Ridge (linear baseline), RandomForest (repo default), ExtraTrees,
HistGradientBoosting, MLP — all sharing the repo's preprocessing.
Reports per-target MAE/R2, classifier metrics, train time, single-frame
inference latency (median over 200 calls), and serialized model size.
Also saves the repo's official bundle to models/predictor.pkl.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

from src.predictor.model import (_clf_pipeline, _load_split, _preprocessor,
                                 _reg_pipeline, CLF_TARGET, REG_TARGETS)
from config import MODEL_PATH

OUT = Path(__file__).parent / "results_exp2.json"

X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = _load_split(ROOT / "data" / "network_dataset.csv")
print(f"split: {len(X_tr)} train / {len(X_te)} test")

REG_CANDIDATES = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    "HistGB": MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42)),
    "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300,
                        early_stopping=True, random_state=42),
}
CLF_CANDIDATES = {
    "LogisticReg": LogisticRegression(max_iter=1000),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "HistGB": HistGradientBoostingClassifier(random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                         early_stopping=True, random_state=42),
}

res = {"regressors": {}, "classifiers": {}}

single_row = X_te.iloc[[0]]

for name, est in REG_CANDIDATES.items():
    pipe = Pipeline([("prep", _preprocessor()), ("est", est)])
    t0 = time.perf_counter()
    pipe.fit(X_tr, yr_tr)
    fit_s = time.perf_counter() - t0
    pred = pipe.predict(X_te)
    per_target = {}
    for i, col in enumerate(REG_TARGETS):
        per_target[col] = {
            "mae": round(float(mean_absolute_error(yr_te.iloc[:, i], pred[:, i])), 2),
            "r2": round(float(r2_score(yr_te.iloc[:, i], pred[:, i])), 4),
        }
    # single-frame latency
    lats = []
    for _ in range(200):
        t0 = time.perf_counter()
        pipe.predict(single_row)
        lats.append((time.perf_counter() - t0) * 1e3)
    size_kb = len(pickle.dumps(pipe)) // 1024
    res["regressors"][name] = {
        "per_target": per_target,
        "mean_r2": round(float(np.mean([v["r2"] for v in per_target.values()])), 4),
        "fit_s": round(fit_s, 1),
        "pred_ms_p50": round(float(np.percentile(lats, 50)), 2),
        "size_kb": size_kb,
    }
    print(name, res["regressors"][name]["mean_r2"], f"{fit_s:.1f}s")

for name, est in CLF_CANDIDATES.items():
    pipe = Pipeline([("prep", _preprocessor()), ("est", est)])
    t0 = time.perf_counter()
    pipe.fit(X_tr, yc_tr)
    fit_s = time.perf_counter() - t0
    pred = pipe.predict(X_te)
    lats = []
    for _ in range(200):
        t0 = time.perf_counter()
        pipe.predict(single_row)
        lats.append((time.perf_counter() - t0) * 1e3)
    res["classifiers"][name] = {
        "accuracy": round(float(accuracy_score(yc_te, pred)), 4),
        "precision": round(float(precision_score(yc_te, pred)), 4),
        "recall": round(float(recall_score(yc_te, pred)), 4),
        "f1": round(float(f1_score(yc_te, pred)), 4),
        "fit_s": round(fit_s, 1),
        "pred_ms_p50": round(float(np.percentile(lats, 50)), 2),
        "size_kb": len(pickle.dumps(pipe)) // 1024,
    }
    print(name, res["classifiers"][name]["f1"], f"{fit_s:.1f}s")

# ── Feature importance from the repo RF (impurity-based, mapped to input names)
rf_pipe = Pipeline([("prep", _preprocessor()),
                    ("est", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42))])
rf_pipe.fit(X_tr, yr_tr)
prep = rf_pipe.named_steps["prep"]
feat_names = list(prep.get_feature_names_out())
imps = rf_pipe.named_steps["est"].feature_importances_
# collapse one-hot scenario cols into one
agg = {}
for n, v in zip(feat_names, imps):
    key = "scenario" if n.startswith("cat__") else n.replace("num__", "")
    agg[key] = agg.get(key, 0.0) + float(v)
res["rf_feature_importance"] = {k: round(v, 4) for k, v in
                                sorted(agg.items(), key=lambda kv: -kv[1])}

# ── Save the official repo bundle (RF + GBM) so closed-loop eval can run
reg = _reg_pipeline(); reg.fit(X_tr, yr_tr)
clf = _clf_pipeline(); clf.fit(X_tr, yc_tr)
bundle = {"reg": reg, "clf": clf, "reg_targets": REG_TARGETS, "clf_target": CLF_TARGET}
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)
print(f"bundle saved -> {MODEL_PATH}")

OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
