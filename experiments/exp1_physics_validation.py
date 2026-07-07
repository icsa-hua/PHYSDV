"""Experiment 1: physics self-consistency validation + per-scenario dataset stats."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import SCENARIOS, DATA_PATH  # noqa: E402

OUT = Path(__file__).parent / "results_exp1.json"

df = pd.read_csv(DATA_PATH)
res = {"n_rows": len(df), "per_scenario": {}, "pathloss_recovery": {}}

for sc, prof in SCENARIOS.items():
    d = df[df["scenario"] == sc]
    reach = d[d["mec_available"] == 1]
    res["per_scenario"][sc] = {
        "n": len(d),
        "share_pct": round(100 * len(d) / len(df), 1),
        "mec_avail_pct": round(100 * d["mec_available"].mean(), 1),
        "rssi_mean": round(d["mec_rssi_dbm"].mean(), 1),
        "rssi_p5": round(d["mec_rssi_dbm"].quantile(0.05), 1),
        "mec_rtt_med": round(reach["mec_network_ms"].median(), 1),
        "mec_rtt_p95": round(reach["mec_network_ms"].quantile(0.95), 1),
        "cloud_rtt_med": round(d["cloud_network_ms"].median(), 1),
        "cloud_rtt_p95": round(d["cloud_network_ms"].quantile(0.95), 1),
        "ho_rate_pct": round(100 * d["handover_active"].mean(), 1),
        # % of samples where MEC RTT beats cloud RTT (when reachable)
        "mec_beats_cloud_pct": round(
            100 * (reach["mec_network_ms"] < reach["cloud_network_ms"]).mean(), 1),
    }

    # Path-loss exponent recovery: RSSI = C - path_exp*log10(d) + noise(shadow+NLOS)
    # Robust fit: use LOS-dominated subset via iterative sigma clipping to strip NLOS tail.
    x = np.log10(d["mec_distance_m"].values)
    y = d["mec_rssi_dbm"].values
    mask = np.ones(len(x), bool)
    for _ in range(3):
        A = np.vstack([x[mask], np.ones(mask.sum())]).T
        coef, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
        resid = y - (coef[0] * x + coef[1])
        mask = np.abs(resid - np.median(resid[mask])) < 2.0 * resid[mask].std()
    res["pathloss_recovery"][sc] = {
        "configured_exp": prof["path_exp"],
        "recovered_exp": round(-coef[0], 2),
        "err_pct": round(100 * abs(-coef[0] - prof["path_exp"]) / prof["path_exp"], 1),
        "shadow_cfg_db": prof["shadow_db"],
        "resid_std_db": round(resid[mask].std(), 2),
    }

# Global stats
reach_all = df[df["mec_available"] == 1]
res["global"] = {
    "mec_avail_pct": round(100 * df["mec_available"].mean(), 1),
    "mec_rtt_med": round(reach_all["mec_network_ms"].median(), 1),
    "mec_rtt_p95": round(reach_all["mec_network_ms"].quantile(0.95), 1),
    "cloud_rtt_med": round(df["cloud_network_ms"].median(), 1),
    "cloud_rtt_p95": round(df["cloud_network_ms"].quantile(0.95), 1),
    # rush hour effect
    "mec_load_rush": round(df[((df.time_of_day_h > 7) & (df.time_of_day_h < 9)) |
                              ((df.time_of_day_h > 17) & (df.time_of_day_h < 19))]["mec_load_pct"].mean(), 1),
    "mec_load_offpeak": round(df[~(((df.time_of_day_h > 7) & (df.time_of_day_h < 9)) |
                                   ((df.time_of_day_h > 17) & (df.time_of_day_h < 19)))]["mec_load_pct"].mean(), 1),
}

OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
