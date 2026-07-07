"""
Risk-aware quantile RTT predictors for uncertainty-aware QoS gating.

Point predictions of network round-trip time (RTT) are unbiased on average
but symmetric: roughly half of all under-estimates admit a mode into the
feasible set whose *true* latency exceeds the QoS budget.  This module
trains upper-quantile regressors (``HistGradientBoostingRegressor`` with
pinball loss) for MEC and cloud RTT so the offloading optimizer can gate
feasibility on a conservative :math:`\\alpha`-quantile estimate instead of
the conditional mean — closing the latency-compliance gap to the oracle at
a small cost in edge utilisation.

The MEC quantile model is trained only on reachable rows
(``mec_available == 1``); unreachable frames are handled upstream by the
availability classifier, so the 999-ms sentinel never pollutes the
quantile fit.

Usage::

    python -m src.predictor.quantile               # train, report coverage
    python -m src.predictor.quantile --alpha 0.95
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import DATA_PATH, QUANTILE_PATH
from src.predictor.model import FEATURE_COLS, _encode_time, _preprocessor


def _q_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor()),
        ("hgb",  HistGradientBoostingRegressor(
            loss="quantile", quantile=alpha, random_state=42)),
    ])


def train_quantile(
    alpha:      float = 0.9,
    data_path:  Path  = DATA_PATH,
    model_path: Path  = QUANTILE_PATH,
) -> dict:
    """
    Train upper-quantile RTT regressors and persist the bundle.

    Parameters
    ----------
    alpha:
        Target quantile level (e.g. 0.9 gates feasibility on the p90 RTT).
    data_path:
        Dataset CSV produced by ``src.dataset.generate``.
    model_path:
        Destination for the serialised quantile bundle.

    Returns
    -------
    dict
        ``{"mec_q", "cloud_q", "alpha", "coverage"}`` — the fitted
        pipelines plus empirical coverage on a held-out 20% split
        (fraction of true RTTs at or below the predicted quantile;
        well-calibrated models score close to ``alpha``).
    """
    df = _encode_time(pd.read_csv(data_path))
    tr, te = train_test_split(df, test_size=0.2, random_state=42)

    mec_tr, mec_te = tr[tr.mec_available == 1], te[te.mec_available == 1]
    mec_q = _q_pipeline(alpha)
    mec_q.fit(mec_tr[FEATURE_COLS], mec_tr["mec_network_ms"])

    cloud_q = _q_pipeline(alpha)
    cloud_q.fit(tr[FEATURE_COLS], tr["cloud_network_ms"])

    coverage = {
        "mec": round(float(
            (mec_te["mec_network_ms"] <=
             mec_q.predict(mec_te[FEATURE_COLS])).mean()), 4),
        "cloud": round(float(
            (te["cloud_network_ms"] <=
             cloud_q.predict(te[FEATURE_COLS])).mean()), 4),
    }
    print(f"q{alpha:.2f} empirical coverage — "
          f"MEC: {coverage['mec']*100:.1f}%  cloud: {coverage['cloud']*100:.1f}%")

    bundle = {"mec_q": mec_q, "cloud_q": cloud_q,
              "alpha": alpha, "coverage": coverage}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Quantile bundle saved → {model_path}")
    return bundle


def load_quantile_bundle(model_path: Path = QUANTILE_PATH) -> dict:
    """Deserialise and return the quantile bundle from disk."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_quantile(qbundle: dict, state: dict) -> dict:
    """
    Predict the upper-quantile MEC and cloud RTT for a single frame state.

    Parameters
    ----------
    qbundle:
        Bundle returned by :func:`train_quantile` / :func:`load_quantile_bundle`.
    state:
        Same observable state dict consumed by ``src.predictor.model.predict``.

    Returns
    -------
    dict
        ``mec_network_ms_q`` and ``cloud_network_ms_q`` (floats, ms).
    """
    row = _encode_time(pd.DataFrame([state]))[FEATURE_COLS]
    return {
        "mec_network_ms_q":   max(5.0,  float(qbundle["mec_q"].predict(row)[0])),
        "cloud_network_ms_q": max(40.0, float(qbundle["cloud_q"].predict(row)[0])),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train risk-aware quantile RTT models.")
    ap.add_argument("--alpha", type=float, default=0.9, help="Quantile level")
    args = ap.parse_args()
    train_quantile(alpha=args.alpha)
