"""Experiment 4: uncertainty-aware (q90) QoS gating in the closed loop.

Compares point-mean gating vs quantile gating (both HistGB-based), plus the
dwell stabilizer with safety-bypass. Also produces the dual outage timeline
(raw EMA decisions vs dwell+force-stabilized) for the paper figure.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRATCH = Path(__file__).parent

from config import (INTERCELL_DB, LATENCY_BUDGET_MS, MODES, NOISE_FIG_DB,
                    SCENARIOS, SINR_MIN_DB, SPEC_EFF_MAX, ACC_LOSS_BUDGET,
                    QUANTILE_PATH)
import src.predictor.client as client_mod
from src.predictor.client import InferenceClient
from src.offloading.table import PerformanceTable
from src.offloading.optimizer import DwellTimeStabilizer, select_mode

FRAMES, HZ, EMA_ALPHA, DWELL_K = 1000, 10, 0.3, 5
HISTGB_BUNDLE = ROOT / "models" / "predictor_histgb.pkl"
OUT = SCRATCH / "results_exp4.json"

_orig_predict = client_mod.predict
_captured = {}


def _capturing_predict(bundle, state):
    _captured["state"] = dict(state)
    return _orig_predict(bundle, state)


client_mod.predict = _capturing_predict


def _sinr_db(rssi_dbm, bw_mhz):
    thermal = -174.0 + 10.0 * np.log10(bw_mhz * 1e6) + NOISE_FIG_DB
    return rssi_dbm - thermal - INTERCELL_DB


def true_mec(state):
    sinr = _sinr_db(state["mec_rssi_dbm"], state["mec_channel_bw_mhz"])
    if sinr <= SINR_MIN_DB:
        return False, None
    spec = min(np.log2(1.0 + 10.0 ** (sinr / 10.0)) * 0.6, SPEC_EFF_MAX)
    user_bw = state["mec_channel_bw_mhz"] / max(state["num_connected_ues"], 1)
    eff_bw = max(spec * user_bw * (1.0 - state["mec_load_pct"] / 100.0 * 0.7), 0.5)
    upload = (state["payload_kb"] * 8.0) / (eff_bw * 1_000.0) * 1_000.0
    prop = state["mec_distance_m"] / 2e8 * 1e3 * 2.0
    queuing = (state["mec_load_pct"] / 100.0) ** 2 * 15.0
    return True, max(5.0, upload + prop + queuing + 20.0 * state["handover_active"] + 0.1)


def true_cloud(state, prof):
    eff_bw = prof["cloud_bw_mbps"] / (1.0 + state["cloud_congestion"] * 3.0)
    upload = (state["payload_kb"] * 8.0) / (eff_bw * 1_000.0) * 1_000.0
    return max(40.0, state["cloud_rtt_base_ms"] + upload
               + state["cloud_congestion"] * 30.0)


def true_total_ms(mode, t_avail, t_mec, t_cld):
    row = MODES[mode]
    if mode in ("A", "B"):
        return row["inference_ms"]
    if mode == "C":
        return None if not t_avail else row["inference_ms"] + t_mec
    return row["inference_ms"] + t_cld


class EmaTable:
    def __init__(self, alpha):
        self.alpha, self.pt = alpha, PerformanceTable()
        self.ema_mec = self.ema_cld = None

    def push(self, avail, mec_ms, cld_ms, mec_cap, cld_cap):
        net, cap = {}, {}
        if avail and mec_ms is not None:
            self.ema_mec = (mec_ms if self.ema_mec is None
                            else self.alpha * mec_ms + (1 - self.alpha) * self.ema_mec)
            net["C"], cap["C"] = self.ema_mec, mec_cap
        else:
            self.ema_mec = None
            self.pt.mark_mec_unavailable()
        self.ema_cld = (cld_ms if self.ema_cld is None
                        else self.alpha * cld_ms + (1 - self.alpha) * self.ema_cld)
        net["D"], cap["D"] = self.ema_cld, cld_cap
        self.pt.update(net, cap)
        return self.pt.get()


POLICIES = ["point_ema", "q90_ema", "q90_ema_dwell", "oracle"]
results = {"config": {"frames": FRAMES, "hz": HZ, "alpha": EMA_ALPHA,
                      "dwell_k": DWELL_K, "quantile": 0.9},
           "scenarios": {}}

for scenario in SCENARIOS:
    prof = SCENARIOS[scenario]
    client = InferenceClient(scenario=scenario, hz=HZ,
                             model_path=HISTGB_BUNDLE,
                             quantile_path=QUANTILE_PATH)
    tables = {p: EmaTable(EMA_ALPHA if "ema" in p else 1.0) for p in
              ["point_ema", "q90_ema", "q90_ema_dwell"]}
    tables["oracle"] = EmaTable(1.0)
    dwell = DwellTimeStabilizer(k=DWELL_K)
    stats = {p: {"modes": [], "true_total": [], "lat_ok": 0, "acc_ok": 0,
                 "map_sum": 0.0} for p in POLICIES}
    tick_ms = []

    for f in range(FRAMES):
        t0 = time.perf_counter()
        pred = client.tick()
        tick_ms.append((time.perf_counter() - t0) * 1e3)
        state = _captured["state"]
        t_avail, t_mec = true_mec(state)
        t_cld = true_cloud(state, prof)

        decisions = {}
        decisions["point_ema"] = select_mode(tables["point_ema"].push(
            pred["mec_available"], pred["mec_network_ms"],
            pred["cloud_network_ms"], pred["mec_capacity_pct"],
            pred["cloud_capacity_pct"]))[0]
        q_mec, q_cld = pred["mec_network_ms_q"], pred["cloud_network_ms_q"]
        decisions["q90_ema"] = select_mode(tables["q90_ema"].push(
            pred["mec_available"], q_mec, q_cld,
            pred["mec_capacity_pct"], pred["cloud_capacity_pct"]))[0]
        m, reason = select_mode(tables["q90_ema_dwell"].push(
            pred["mec_available"], q_mec, q_cld,
            pred["mec_capacity_pct"], pred["cloud_capacity_pct"]))
        decisions["q90_ema_dwell"] = dwell.apply(
            m, force=(reason == "fallback_no_feasible_mode"))
        decisions["oracle"] = select_mode(tables["oracle"].push(
            t_avail, t_mec, t_cld, state["mec_load_pct"],
            state["cloud_load_pct"]))[0]

        for p, mode in decisions.items():
            s = stats[p]
            s["modes"].append(mode)
            tt = true_total_ms(mode, t_avail, t_mec, t_cld)
            s["lat_ok"] += int(tt is not None and tt <= LATENCY_BUDGET_MS)
            s["acc_ok"] += int(MODES[mode]["acc_loss"] <= ACC_LOSS_BUDGET)
            s["map_sum"] += MODES[mode]["map50_95"]
            if tt is not None:
                s["true_total"].append(tt)

    sc = {"tick_ms_p50": round(float(np.percentile(tick_ms, 50)), 1),
          "agree_point": round(100 * sum(
              a == b for a, b in zip(stats["point_ema"]["modes"],
                                     stats["oracle"]["modes"])) / FRAMES, 1),
          "agree_q90": round(100 * sum(
              a == b for a, b in zip(stats["q90_ema"]["modes"],
                                     stats["oracle"]["modes"])) / FRAMES, 1),
          "policies": {}}
    for p in POLICIES:
        s = stats[p]
        modes = s["modes"]
        sc["policies"][p] = {
            "share": {m: round(100 * modes.count(m) / FRAMES, 1) for m in "ABCD"},
            "switches_per_1k": sum(1 for a, b in zip(modes, modes[1:]) if a != b),
            "lat_ok_pct": round(100 * s["lat_ok"] / FRAMES, 1),
            "acc_ok_pct": round(100 * s["acc_ok"] / FRAMES, 1),
            "mean_map": round(s["map_sum"] / FRAMES, 1),
            "mean_true_total_ms": round(float(np.mean(s["true_total"])), 1)
                                  if s["true_total"] else None,
        }
    results["scenarios"][scenario] = sc
    print(scenario, "point lat_ok:", sc["policies"]["point_ema"]["lat_ok_pct"],
          "| q90 lat_ok:", sc["policies"]["q90_ema"]["lat_ok_pct"])

# ── dual outage timeline (urban_dense) for the paper figure ──────────────────
client = InferenceClient(scenario="urban_dense", hz=HZ,
                         model_path=HISTGB_BUNDLE, quantile_path=QUANTILE_PATH)
ema = EmaTable(EMA_ALPHA)
dwell = DwellTimeStabilizer(k=DWELL_K)
OUTAGE_AT, OUTAGE_DUR, N = 80, 30, 200
raw_tl, stab_tl = [], []
for f in range(1, N + 1):
    pred = client.tick()
    if OUTAGE_AT <= f < OUTAGE_AT + OUTAGE_DUR:
        pred = dict(pred)
        pred["mec_available"] = False
        pred["cloud_network_ms_q"] = 200.0
    mode, reason = select_mode(ema.push(
        pred["mec_available"], pred["mec_network_ms_q"],
        pred["cloud_network_ms_q"], pred["mec_capacity_pct"],
        pred["cloud_capacity_pct"]))
    raw_tl.append(mode)
    stab_tl.append(dwell.apply(mode, force=(reason == "fallback_no_feasible_mode")))


def rle(tl):
    out, cur, cnt = [], tl[0], 0
    for m in tl:
        if m == cur:
            cnt += 1
        else:
            out.append([cur, cnt]); cur, cnt = m, 1
    out.append([cur, cnt])
    return out


fb_raw = next((i + 1 for i, m in enumerate(raw_tl) if m == "B"), None)
fb_stab = next((i + 1 for i, m in enumerate(stab_tl) if m == "B"), None)
results["outage"] = {
    "outage_at": OUTAGE_AT, "outage_dur": OUTAGE_DUR,
    "fallback_delay_raw": fb_raw - OUTAGE_AT, "fallback_delay_stab": fb_stab - OUTAGE_AT,
    "raw_rle": rle(raw_tl), "stab_rle": rle(stab_tl),
}

OUT.write_text(json.dumps(results, indent=2))
print(json.dumps(results["outage"], indent=2))
