"""Experiment 3: closed-loop evaluation across all 5 scenarios.

Per frame we capture the client's observable state, compute a deterministic
physics ORACLE (same equations as the generator, noise factor = 1), and
evaluate policies:

  always_B / always_C / always_D  - static tier baselines
  proposed_raw                    - QoS optimizer on raw per-frame predictions
  proposed_ema                    - + EMA smoothing (alpha=0.3, as in demo.py)
  proposed_ema_dwell              - + min-dwell hysteresis (k=5 frames)
  oracle                          - QoS optimizer on ground-truth table

Metrics: mode share, switches/1k frames, oracle decision agreement, true mean
latency of selected tier, latency-budget compliance (true total <= 100 ms),
accuracy-budget compliance, mean effective mAP, WAN (cloud) share,
per-frame control-loop latency. Plus an outage failover trace.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (INTERCELL_DB, LATENCY_BUDGET_MS, MODES, NOISE_FIG_DB,
                    SCENARIOS, SINR_MIN_DB, SPEC_EFF_MAX, ACC_LOSS_BUDGET)
import src.predictor.client as client_mod
from src.predictor.client import InferenceClient
from src.offloading.table import PerformanceTable
from src.offloading.optimizer import select_mode

FRAMES = 1000
HZ = 10
EMA_ALPHA = 0.3
DWELL_K = 5

# optional args: [bundle_path] [output_suffix]
BUNDLE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else None
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""

OUT = Path(__file__).parent / f"results_exp3{SUFFIX}.json"

# ── capture the client's internal state via a wrapper around predict ─────────
_orig_predict = client_mod.predict
_captured = {}


def _capturing_predict(bundle, state):
    _captured["state"] = dict(state)
    return _orig_predict(bundle, state)


client_mod.predict = _capturing_predict


# ── deterministic physics oracle (generator equations, noise factor = 1) ─────
def _sinr_db(rssi_dbm, bw_mhz):
    thermal = -174.0 + 10.0 * np.log10(bw_mhz * 1e6) + NOISE_FIG_DB
    return rssi_dbm - thermal - INTERCELL_DB


def true_mec(state):
    sinr = _sinr_db(state["mec_rssi_dbm"], state["mec_channel_bw_mhz"])
    avail = sinr > SINR_MIN_DB
    if not avail:
        return False, None
    spec = min(np.log2(1.0 + 10.0 ** (sinr / 10.0)) * 0.6, SPEC_EFF_MAX)
    user_bw = state["mec_channel_bw_mhz"] / max(state["num_connected_ues"], 1)
    eff_bw = max(spec * user_bw * (1.0 - state["mec_load_pct"] / 100.0 * 0.7), 0.5)
    upload = (state["payload_kb"] * 8.0) / (eff_bw * 1_000.0) * 1_000.0
    prop = state["mec_distance_m"] / 2e8 * 1e3 * 2.0
    queuing = (state["mec_load_pct"] / 100.0) ** 2 * 15.0
    total = upload + prop + queuing + 20.0 * state["handover_active"] + 0.1
    return True, max(5.0, total)


def true_cloud(state, prof):
    eff_bw = prof["cloud_bw_mbps"] / (1.0 + state["cloud_congestion"] * 3.0)
    upload = (state["payload_kb"] * 8.0) / (eff_bw * 1_000.0) * 1_000.0
    total = state["cloud_rtt_base_ms"] + upload + state["cloud_congestion"] * 30.0
    return max(40.0, total)


def true_total_ms(mode, t_mec_avail, t_mec_ms, t_cld_ms):
    """Ground-truth end-to-end latency of executing on a given tier."""
    row = MODES[mode]
    if mode in ("A", "B"):
        return row["inference_ms"]
    if mode == "C":
        return None if not t_mec_avail else row["inference_ms"] + t_mec_ms
    return row["inference_ms"] + t_cld_ms


# ── policy state helpers ──────────────────────────────────────────────────────
class EmaTable:
    """Replicates demo.py's EMA-smoothed PerformanceTable feeding."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.pt = PerformanceTable()
        self.ema_mec = None
        self.ema_cld = None

    def push(self, pred):
        net, cap = {}, {}
        raw_mec, raw_cld = pred["mec_network_ms"], pred["cloud_network_ms"]
        if pred["mec_available"] and raw_mec is not None:
            self.ema_mec = (raw_mec if self.ema_mec is None
                            else self.alpha * raw_mec + (1 - self.alpha) * self.ema_mec)
            net["C"] = self.ema_mec
            cap["C"] = pred["mec_capacity_pct"]
        else:
            self.ema_mec = None
            self.pt.mark_mec_unavailable()
        self.ema_cld = (raw_cld if self.ema_cld is None
                        else self.alpha * raw_cld + (1 - self.alpha) * self.ema_cld)
        net["D"] = self.ema_cld
        cap["D"] = pred["cloud_capacity_pct"]
        self.pt.update(net, cap)
        return self.pt.get()


class DwellPolicy:
    """Min-dwell hysteresis: only commit to a switch after it persists k frames."""

    def __init__(self, k):
        self.k = k
        self.current = None
        self.candidate = None
        self.streak = 0

    def apply(self, proposed):
        if self.current is None:
            self.current = proposed
            return proposed
        if proposed == self.current:
            self.candidate, self.streak = None, 0
            return self.current
        if proposed == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = proposed, 1
        if self.streak >= self.k:
            self.current = self.candidate
            self.candidate, self.streak = None, 0
        return self.current


POLICIES = ["always_B", "always_C", "always_D",
            "proposed_raw", "proposed_ema", "proposed_ema_dwell", "oracle"]

results = {"config": {"frames": FRAMES, "hz": HZ, "ema_alpha": EMA_ALPHA,
                      "dwell_k": DWELL_K},
           "scenarios": {}}

for scenario in SCENARIOS:
    prof = SCENARIOS[scenario]
    client = (InferenceClient(scenario=scenario, hz=HZ, model_path=BUNDLE_PATH)
              if BUNDLE_PATH else InferenceClient(scenario=scenario, hz=HZ))
    tables = {"raw": EmaTable(1.0), "ema": EmaTable(EMA_ALPHA),
              "oracle": EmaTable(1.0)}
    dwell = DwellPolicy(DWELL_K)

    stats = {p: {"modes": [], "true_total": [], "lat_ok": 0, "acc_ok": 0,
                 "map_sum": 0.0} for p in POLICIES}
    agree_oracle = 0
    tick_lat_ms = []

    for f in range(FRAMES):
        t0 = time.perf_counter()
        pred = client.tick()
        tick_lat_ms.append((time.perf_counter() - t0) * 1e3)
        state = _captured["state"]

        t_avail, t_mec = true_mec(state)
        t_cld = true_cloud(state, prof)

        decisions = {
            "always_B": "B", "always_C": "C", "always_D": "D",
            "proposed_raw": select_mode(tables["raw"].push(pred))[0],
            "proposed_ema": select_mode(tables["ema"].push(pred))[0],
        }
        decisions["proposed_ema_dwell"] = dwell.apply(decisions["proposed_ema"])

        oracle_pred = {"mec_available": int(t_avail), "mec_network_ms": t_mec,
                       "mec_capacity_pct": state["mec_load_pct"],
                       "cloud_network_ms": t_cld,
                       "cloud_capacity_pct": state["cloud_load_pct"]}
        decisions["oracle"] = select_mode(tables["oracle"].push(oracle_pred))[0]

        if decisions["proposed_ema"] == decisions["oracle"]:
            agree_oracle += 1

        for p, m in decisions.items():
            s = stats[p]
            s["modes"].append(m)
            tt = true_total_ms(m, t_avail, t_mec, t_cld)
            lat_ok = tt is not None and tt <= LATENCY_BUDGET_MS
            s["lat_ok"] += int(lat_ok)
            s["acc_ok"] += int(MODES[m]["acc_loss"] <= ACC_LOSS_BUDGET)
            s["map_sum"] += MODES[m]["map50_95"]
            if tt is not None:
                s["true_total"].append(tt)

    sc_res = {"tick_ms_p50": round(float(np.percentile(tick_lat_ms, 50)), 1),
              "tick_ms_p95": round(float(np.percentile(tick_lat_ms, 95)), 1),
              "oracle_agreement_pct": round(100 * agree_oracle / FRAMES, 1),
              "policies": {}}
    for p in POLICIES:
        s = stats[p]
        modes = s["modes"]
        switches = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
        share = {m: round(100 * modes.count(m) / FRAMES, 1) for m in "ABCD"}
        sc_res["policies"][p] = {
            "share": share,
            "switches_per_1k": switches,
            "lat_ok_pct": round(100 * s["lat_ok"] / FRAMES, 1),
            "acc_ok_pct": round(100 * s["acc_ok"] / FRAMES, 1),
            "mean_map": round(s["map_sum"] / FRAMES, 1),
            "mean_true_total_ms": round(float(np.mean(s["true_total"])), 1)
                                  if s["true_total"] else None,
            "wan_share_pct": share["D"],
        }
    results["scenarios"][scenario] = sc_res
    print(scenario, "done | oracle agree:", sc_res["oracle_agreement_pct"], "%")

# ── outage failover trace (urban_dense, injection identical to demo.py) ──────
client = InferenceClient(scenario="urban_dense", hz=HZ)
ema = EmaTable(EMA_ALPHA)
OUTAGE_AT, OUTAGE_DUR, N = 80, 30, 200
timeline = []
for f in range(1, N + 1):
    pred = client.tick()
    if OUTAGE_AT <= f < OUTAGE_AT + OUTAGE_DUR:
        pred = dict(pred)
        pred["mec_available"] = False
        pred["cloud_network_ms"] = 200.0
    mode, reason = select_mode(ema.push(pred))
    timeline.append(mode)

first_b = next((i + 1 for i, m in enumerate(timeline) if m == "B"), None)
rec = next((i + 1 for i, m in enumerate(timeline[OUTAGE_AT + OUTAGE_DUR - 1:],
                                        start=OUTAGE_AT + OUTAGE_DUR - 1)
            if m in ("C", "D")), None)
results["outage"] = {
    "outage_at": OUTAGE_AT, "outage_dur": OUTAGE_DUR,
    "frames_to_fallback": (first_b - OUTAGE_AT) if first_b else None,
    "recovery_frame": rec,
    "frames_to_recover": (rec - (OUTAGE_AT + OUTAGE_DUR)) if rec else None,
    "timeline_rle": [],
}
# run-length encode the timeline for the paper
cur, cnt = timeline[0], 0
for m in timeline:
    if m == cur:
        cnt += 1
    else:
        results["outage"]["timeline_rle"].append([cur, cnt])
        cur, cnt = m, 1
results["outage"]["timeline_rle"].append([cur, cnt])

OUT.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
