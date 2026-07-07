"""
Shared configuration for the HUA CloudNet 2026 artifact.

All scenario profiles, execution-mode definitions, radio-physics constants,
YOLOv8 parameters, and QoS budgets live here so every module imports from
one authoritative source.
"""

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent

DATA_PATH     = ROOT / "data"  / "network_dataset.csv"
MODEL_PATH    = ROOT / "models" / "predictor.pkl"
QUANTILE_PATH = ROOT / "models" / "predictor_q90.pkl"
OUTPUT_DIR    = ROOT / "output"

# ── QoS budgets ───────────────────────────────────────────────────────────────

LATENCY_BUDGET_MS        = 100.0
ACC_LOSS_BUDGET          = 0.15
CAPACITY_LIMIT_PCT       = 95.0
MEC_PREFERENCE_MAP_DELTA = 5.0

# ── Dataset generation ────────────────────────────────────────────────────────

N_SAMPLES   = 50_000
RANDOM_SEED = 42

SCENARIO_WEIGHTS = {
    "urban_dense":  0.30,
    "urban_sparse": 0.25,
    "suburban":     0.20,
    "highway":      0.15,
    "rural":        0.10,
}

SCENARIOS = {
    "urban_dense": dict(
        mec_spacing_m   = 300,
        speed_range     = (20,  50),
        bw_mhz          = 100,
        freq_ghz        = 3.5,
        shadow_db       = 8,
        ue_lambda       = 40,
        path_exp        = 35,
        nlos_prob       = 0.25,
        cloud_rtt_range = (40,  60),
        cloud_bw_mbps   = 1000,
    ),
    "urban_sparse": dict(
        mec_spacing_m   = 800,
        speed_range     = (30,  70),
        bw_mhz          = 100,
        freq_ghz        = 3.5,
        shadow_db       = 8,
        ue_lambda       = 20,
        path_exp        = 30,
        nlos_prob       = 0.20,
        cloud_rtt_range = (40,  70),
        cloud_bw_mbps   = 1000,
    ),
    "suburban": dict(
        mec_spacing_m   = 2000,
        speed_range     = (50,  90),
        bw_mhz          = 40,
        freq_ghz        = 2.1,
        shadow_db       = 6,
        ue_lambda       = 10,
        path_exp        = 26,
        nlos_prob       = 0.15,
        cloud_rtt_range = (45,  75),
        cloud_bw_mbps   = 800,
    ),
    "highway": dict(
        mec_spacing_m   = 5000,
        speed_range     = (80, 130),
        bw_mhz          = 20,
        freq_ghz        = 2.1,
        shadow_db       = 4,
        ue_lambda       = 5,
        path_exp        = 22,
        nlos_prob       = 0.10,
        cloud_rtt_range = (45,  80),
        cloud_bw_mbps   = 600,
    ),
    "rural": dict(
        mec_spacing_m   = 10_000,
        speed_range     = (60, 100),
        bw_mhz          = 10,
        freq_ghz        = 0.9,
        shadow_db       = 4,
        ue_lambda       = 2,
        path_exp        = 20,
        nlos_prob       = 0.25,
        cloud_rtt_range = (50,  90),
        cloud_bw_mbps   = 400,
    ),
}

# ── Radio-physics constants ───────────────────────────────────────────────────

BS_TX_DBM    = 46
BS_GAIN_DBI  = 15
NOISE_FIG_DB = 7
INTERCELL_DB = 5
SINR_MIN_DB  = 0
SPEC_EFF_MAX = 8.0

# ── YOLOv8 execution modes ────────────────────────────────────────────────────

TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
IMG_SIZE       = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
FRAME_H        = 720
FRAME_W        = 1280
WARMUP_RUNS    = 5
BENCHMARK_RUNS = 50

_CLASS_AP = {
    "yolov8n": {"person": 52.1, "car": 46.8, "motorcycle": 47.2, "bus": 58.3, "truck": 36.4},
    "yolov8m": {"person": 59.4, "car": 56.1, "motorcycle": 56.8, "bus": 68.9, "truck": 47.2},
    "yolov8l": {"person": 61.8, "car": 59.3, "motorcycle": 59.7, "bus": 71.4, "truck": 50.6},
}


def _weighted_acc_loss(model_name: str) -> float:
    baseline = _CLASS_AP["yolov8l"]
    target   = _CLASS_AP[model_name]
    drops    = [(baseline[c] - target[c]) / baseline[c] for c in baseline]
    return round(sum(drops) / len(drops), 4)


MODES: dict[str, dict] = {
    "A": {
        "model":        "yolov8n",
        "hw":           "Car GPU 1",
        "inference_ms": 8.0,
        "network_ms":   0.0,
        "total_ms":     8.0,
        "map50_95":     37.3,
        "class_ap":     _CLASS_AP["yolov8n"],
        "acc_loss":     _weighted_acc_loss("yolov8n"),
        "capacity_pct": 0.0,
    },
    "B": {
        "model":        "yolov8n",
        "hw":           "Car GPU 2",
        "inference_ms": 5.0,
        "network_ms":   0.0,
        "total_ms":     5.0,
        "map50_95":     37.3,
        "class_ap":     _CLASS_AP["yolov8n"],
        "acc_loss":     _weighted_acc_loss("yolov8n"),
        "capacity_pct": 0.0,
    },
    "C": {
        "model":        "yolov8m",
        "hw":           "MEC",
        "inference_ms": 6.0,
        "network_ms":   18.0,
        "total_ms":     24.0,
        "map50_95":     50.2,
        "class_ap":     _CLASS_AP["yolov8m"],
        "acc_loss":     _weighted_acc_loss("yolov8m"),
        "capacity_pct": 0.0,
    },
    "D": {
        "model":        "yolov8l",
        "hw":           "Cloud",
        "inference_ms": 4.0,
        "network_ms":   60.0,
        "total_ms":     84.0,
        "map50_95":     52.9,
        "class_ap":     _CLASS_AP["yolov8l"],
        "acc_loss":     0.0,
        "capacity_pct": 0.0,
    },
}


def get_device() -> str:
    """Return the best available PyTorch compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
