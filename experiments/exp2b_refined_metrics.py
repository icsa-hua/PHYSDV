"""Experiment 2b: operationally honest predictor metrics.

- mec_network_ms scored ONLY on test rows where MEC is truly reachable
  (the 999-ms sentinel rows are handled by the availability classifier, not
  the regressor, at inference time).
- cloud_network_ms scored on all test rows.
- capacity targets reported separately as observable pass-through telemetry.
Also saves an alternative HistGB bundle for closed-loop comparison.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import (ExtraTreesRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from src.predictor.model import (_clf_pipeline, _load_split, _preprocessor,
                                 CLF_TARGET, REG_TARGETS)

OUT = Path(__file__).parent / "results_exp2b.json"
HISTGB_BUNDLE = ROOT / "models" / "predictor_histgb.pkl"

X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = _load_split(ROOT / "data" / "network_dataset.csv")
avail_te = (yc_te == 1).values
i_mec = REG_TARGETS.index("mec_network_ms")
i_cld = REG_TARGETS.index("cloud_network_ms")

CANDIDATES = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    "HistGB": MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42)),
    "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300,
                        early_stopping=True, random_state=42),
}

res = {}
single_row = X_te.iloc[[0]]
for name, est in CANDIDATES.items():
    pipe = Pipeline([("prep", _preprocessor()), ("est", est)])
    t0 = time.perf_counter()
    pipe.fit(X_tr, yr_tr)
    fit_s = time.perf_counter() - t0
    pred = pipe.predict(X_te)
    lats = []
    for _ in range(200):
        t0 = time.perf_counter()
        pipe.predict(single_row)
        lats.append((time.perf_counter() - t0) * 1e3)
    res[name] = {
        "mec_rtt_mae_reachable": round(float(mean_absolute_error(
            yr_te.iloc[avail_te, i_mec], pred[avail_te, i_mec])), 2),
        "mec_rtt_r2_reachable": round(float(r2_score(
            yr_te.iloc[avail_te, i_mec], pred[avail_te, i_mec])), 4),
        "cloud_rtt_mae": round(float(mean_absolute_error(
            yr_te.iloc[:, i_cld], pred[:, i_cld])), 2),
        "cloud_rtt_r2": round(float(r2_score(
            yr_te.iloc[:, i_cld], pred[:, i_cld])), 4),
        "fit_s": round(fit_s, 1),
        "pred_ms_p50": round(float(np.percentile(lats, 50)), 2),
        "size_mb": round(len(pickle.dumps(pipe)) / 1024 / 1024, 2),
    }
    print(name, res[name])
    if name == "HistGB":
        histgb_reg = pipe

# save alternative bundle: HistGB regressor + repo GBM classifier
clf = _clf_pipeline()
clf.fit(X_tr, yc_tr)
bundle = {"reg": histgb_reg, "clf": clf,
          "reg_targets": REG_TARGETS, "clf_target": CLF_TARGET}
with open(HISTGB_BUNDLE, "wb") as f:
    pickle.dump(bundle, f)
print(f"HistGB bundle -> {HISTGB_BUNDLE}")

OUT.write_text(json.dumps(res, indent=2))
