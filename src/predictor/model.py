"""
Multi-output ML predictor for MEC and cloud network conditions.

Trains two scikit-learn pipelines on the HUA synthetic dataset:

* **Regressor** — ``RandomForestRegressor`` (200 trees) predicting four
  continuous targets: ``mec_network_ms``, ``mec_capacity_pct``,
  ``cloud_network_ms``, ``cloud_capacity_pct``.

* **Classifier** — ``GradientBoostingClassifier`` (100 trees) predicting
  the binary ``mec_available`` label.

Both pipelines share the same preprocessing: one-hot encoding for the
``scenario`` categorical feature and standard scaling for all 12 numeric
features.

Usage::

    python -m src.predictor.model            # train and save
    python -m src.predictor.model --eval     # load and evaluate
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_PATH, MODEL_PATH

CAT_COLS = ["scenario"]
NUM_COLS = [
    "car_speed_kmh",
    "time_sin",
    "time_cos",
    "mec_distance_m",
    "mec_rssi_dbm",
    "mec_channel_bw_mhz",
    "mec_load_pct",
    "num_connected_ues",
    "handover_active",
    "payload_kb",
    "cloud_congestion",
    "cloud_rtt_base_ms",
    "cloud_load_pct",
]
FEATURE_COLS = CAT_COLS + NUM_COLS
REG_TARGETS  = ["mec_network_ms", "mec_capacity_pct", "cloud_network_ms", "cloud_capacity_pct"]
CLF_TARGET   = "mec_available"


def _encode_time(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ``time_of_day_h`` with a sin/cos pair encoding the cyclic 24-hour clock."""
    df           = df.copy()
    df["time_sin"] = np.sin(2 * np.pi * df["time_of_day_h"] / 24.0)
    df["time_cos"] = np.cos(2 * np.pi * df["time_of_day_h"] / 24.0)
    return df


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ],
        remainder="drop",
    )


def _reg_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor()),
        ("rf",   RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)),
    ])


def _clf_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor()),
        ("clf",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ])


def _load_split(data_path: Path):
    """Load CSV, engineer cyclic time features, return stratified 80/20 split."""
    df = pd.read_csv(data_path)
    df = _encode_time(df)
    return train_test_split(
        df[FEATURE_COLS], df[REG_TARGETS], df[CLF_TARGET],
        test_size=0.2, random_state=42,
    )


def _print_metrics(
    X_te:  pd.DataFrame,
    yr_te: pd.DataFrame,
    yc_te: pd.Series,
    reg:   Pipeline,
    clf:   Pipeline,
) -> None:
    yr_pred = reg.predict(X_te)
    yc_pred = clf.predict(X_te)

    print("\n── Regression metrics (test set) ────────────────────────────────────")
    for i, col in enumerate(REG_TARGETS):
        mae = mean_absolute_error(yr_te.iloc[:, i], yr_pred[:, i])
        r2  = r2_score(yr_te.iloc[:, i], yr_pred[:, i])
        print(f"  {col:<28}  MAE = {mae:7.2f}   R² = {r2:.4f}")

    print("\n── Classifier metrics — mec_available (test set) ────────────────────")
    print(f"  Accuracy : {accuracy_score(yc_te, yc_pred):.4f}")
    print(classification_report(
        yc_te, yc_pred,
        target_names=["unavailable", "available"],
        zero_division=0,
    ))


def train(
    data_path:  Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> dict:
    """
    Train the regressor and classifier, print metrics, and persist the bundle.

    Parameters
    ----------
    data_path:
        Path to the dataset CSV produced by ``src.dataset.generate``.
    model_path:
        Destination path for the serialised model bundle (pickle).

    Returns
    -------
    dict
        The saved bundle: ``{"reg": Pipeline, "clf": Pipeline}``.
    """
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = _load_split(data_path)
    print(f"Dataset : {len(X_tr):,} train / {len(X_te):,} test")

    reg = _reg_pipeline()
    clf = _clf_pipeline()

    print("Training regressor  (RandomForest, 200 trees) …")
    reg.fit(X_tr, yr_tr)

    print("Training classifier (GradientBoosting, 100 trees) …")
    clf.fit(X_tr, yc_tr)

    _print_metrics(X_te, yr_te, yc_te, reg, clf)

    bundle = {"reg": reg, "clf": clf, "reg_targets": REG_TARGETS, "clf_target": CLF_TARGET}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModel saved → {model_path}")
    return bundle


def evaluate(
    data_path:  Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> None:
    """Load a saved bundle and re-print metrics on the held-out test split."""
    bundle = load_bundle(model_path)
    _, X_te, _, yr_te, _, yc_te = _load_split(data_path)
    _print_metrics(X_te, yr_te, yc_te, bundle["reg"], bundle["clf"])


def load_bundle(model_path: Path = MODEL_PATH) -> dict:
    """Deserialise and return the model bundle from disk."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(bundle: dict, state: dict) -> dict:
    """
    Predict MEC and cloud network conditions for a single frame state.

    Parameters
    ----------
    bundle:
        Model bundle returned by :func:`load_bundle`.
    state:
        Dict with keys: ``scenario``, ``car_speed_kmh``, ``time_of_day_h``,
        ``mec_distance_m``, ``mec_rssi_dbm``, ``mec_channel_bw_mhz``,
        ``mec_load_pct``, ``num_connected_ues``, ``handover_active``,
        ``payload_kb``, ``cloud_congestion``, ``cloud_rtt_base_ms``,
        ``cloud_load_pct``.

    Returns
    -------
    dict
        Keys: ``mec_available`` (int), ``mec_network_ms`` (float | None),
        ``mec_capacity_pct`` (float), ``cloud_network_ms`` (float),
        ``cloud_capacity_pct`` (float).
    """
    row      = _encode_time(pd.DataFrame([state]))
    X        = row[FEATURE_COLS]
    reg_vals = bundle["reg"].predict(X)[0]
    clf_val  = int(bundle["clf"].predict(X)[0])

    result = dict(zip(REG_TARGETS, reg_vals))
    result[CLF_TARGET] = clf_val

    result["mec_network_ms"]     = max(5.0,  float(result["mec_network_ms"]))
    result["cloud_network_ms"]   = max(40.0, float(result["cloud_network_ms"]))
    result["mec_capacity_pct"]   = float(np.clip(result["mec_capacity_pct"],   0.0, 100.0))
    result["cloud_capacity_pct"] = float(np.clip(result["cloud_capacity_pct"], 0.0, 100.0))

    if not clf_val:
        result["mec_network_ms"] = None

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="Evaluate saved model instead of training")
    args = ap.parse_args()

    if args.eval:
        evaluate()
    else:
        train()
