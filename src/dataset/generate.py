"""
Physics-based synthetic dataset generator for vehicular edge-offloading research.

Simulates a vehicle transmitting 640×640 JPEG inference frames to a nearby MEC
server and to the cloud across five road environment types.  Radio link quality
is derived from a 3GPP-style path-loss model; per-UE throughput follows the
Shannon–Hartley theorem with a practical spectral-efficiency cap.  Cloud latency
models WAN RTT plus a congestion-dependent transmission delay.

The output CSV contains 13 observable input features, 4 continuous regression
targets, and 1 binary classification target — exactly the schema consumed by
``src/predictor/model.py``.

Usage::

    python -m src.dataset.generate
    python -m src.dataset.generate --n 100000 --out data/custom.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BS_GAIN_DBI, BS_TX_DBM, INTERCELL_DB, N_SAMPLES, NOISE_FIG_DB,
    RANDOM_SEED, SCENARIO_WEIGHTS, SCENARIOS, SINR_MIN_DB, SPEC_EFF_MAX,
    DATA_PATH,
)


def _path_loss_db(distance_m: float, freq_ghz: float, path_exp: float) -> float:
    """3GPP-style path loss (dB): base offset + environment exponent + frequency term."""
    return 32.4 + path_exp * np.log10(max(distance_m, 1.0)) + 20.0 * np.log10(freq_ghz)


def _rssi_dbm(
    distance_m: float,
    freq_ghz:   float,
    shadow_db:  float,
    path_exp:   float,
    rng:        np.random.Generator,
) -> float:
    """Received signal strength (dBm) with log-normal shadowing."""
    pl = _path_loss_db(distance_m, freq_ghz, path_exp)
    return BS_TX_DBM + BS_GAIN_DBI - pl + rng.normal(0.0, shadow_db)


def _sinr_db(rssi_dbm: float, bw_mhz: float) -> float:
    """
    Signal-to-Interference-plus-Noise Ratio (dB).

    In OFDMA (LTE/5G NR) scheduled UEs are orthogonal within a cell, so only
    thermal noise and a fixed inter-cell interference term matter.
    """
    thermal_dbm = -174.0 + 10.0 * np.log10(bw_mhz * 1e6) + NOISE_FIG_DB
    return rssi_dbm - thermal_dbm - INTERCELL_DB


def _spectral_efficiency(sinr_db: float) -> float:
    """Shannon capacity (bits/s/Hz) scaled by a 0.6 practical factor, capped at 256-QAM limit."""
    return min(np.log2(1.0 + 10.0 ** (sinr_db / 10.0)) * 0.6, SPEC_EFF_MAX)


def _mec_rtt_ms(
    distance_m: float,
    rssi_dbm:   float,
    bw_mhz:     float,
    load_pct:   float,
    num_ues:    int,
    payload_kb: float,
    handover:   int,
    rng:        np.random.Generator,
) -> float:
    """
    Physics-based MEC round-trip time (ms).

    Combines uplink transmission delay (Shannon throughput, per-UE bandwidth
    share), propagation delay, M/D/1-style queuing delay, and a handover
    penalty.
    """
    sinr    = _sinr_db(rssi_dbm, bw_mhz)
    spec    = _spectral_efficiency(sinr)
    user_bw = bw_mhz / max(num_ues, 1)
    eff_bw  = max(spec * user_bw * (1.0 - load_pct / 100.0 * 0.7), 0.5)
    upload  = (payload_kb * 8.0) / (eff_bw * 1_000.0) * 1_000.0
    prop    = distance_m / 2e8 * 1e3 * 2.0
    queuing = (load_pct / 100.0) ** 2 * 15.0
    total   = upload + prop + queuing + 20.0 * handover + 0.1
    return max(5.0, total * rng.uniform(0.90, 1.10))


def _cloud_rtt_ms(
    base_ms:      float,
    congestion:   float,
    payload_kb:   float,
    backbone_mbps: float,
    rng:          np.random.Generator,
) -> float:
    """
    Cloud round-trip time (ms).

    Fixed regional RTT base plus WAN upload delay and a congestion penalty
    that degrades effective backbone bandwidth non-linearly.
    """
    eff_bw = backbone_mbps / (1.0 + congestion * 3.0)
    upload = (payload_kb * 8.0) / (eff_bw * 1_000.0) * 1_000.0
    total  = base_ms + upload + congestion * 30.0
    return max(40.0, total * rng.uniform(0.95, 1.05))


def _generate_sample(scenario: str, rng: np.random.Generator) -> dict:
    """
    Generate one labeled sample for a given road scenario.

    Parameters
    ----------
    scenario:
        One of the keys in ``config.SCENARIOS``.
    rng:
        Seeded NumPy random generator — ensures reproducibility.

    Returns
    -------
    dict
        All 18 columns of the dataset schema.
    """
    sc = SCENARIOS[scenario]

    speed_kmh  = rng.uniform(*sc["speed_range"])
    time_h     = rng.uniform(0.0, 24.0)
    distance_m = rng.uniform(10.0, sc["mec_spacing_m"])

    rssi = _rssi_dbm(distance_m, sc["freq_ghz"], sc["shadow_db"], sc["path_exp"], rng)
    if rng.random() < sc["nlos_prob"]:
        rssi -= rng.uniform(25.0, 45.0)

    mec_avail = int(_sinr_db(rssi, sc["bw_mhz"]) > SINR_MIN_DB)

    rush_factor = 1.5 if (7 < time_h < 9 or 17 < time_h < 19) else 1.0
    mec_load    = float(np.clip(rng.beta(2, 5) * 100.0 * rush_factor, 0.0, 100.0))
    num_ues     = max(1, int(rng.poisson(sc["ue_lambda"] * rush_factor)))

    cell_edge = distance_m > sc["mec_spacing_m"] * 0.7
    ho_prob   = 0.30 if (cell_edge and speed_kmh > 60) else 0.05
    handover  = int(rng.random() < ho_prob)

    payload_kb    = float(np.clip(rng.normal(80.0, 20.0), 20.0, 200.0))
    cloud_cong    = float(rng.beta(1.5, 4.0))
    cloud_rtt_b   = float(rng.uniform(*sc["cloud_rtt_range"]))
    cloud_load    = float(rng.beta(1.5, 4.0) * 100.0)

    mec_ms = (
        _mec_rtt_ms(distance_m, rssi, sc["bw_mhz"], mec_load, num_ues,
                    payload_kb, handover, rng)
        if mec_avail else 999.0
    )
    cloud_ms = _cloud_rtt_ms(cloud_rtt_b, cloud_cong, payload_kb,
                              sc["cloud_bw_mbps"], rng)

    return {
        "scenario":           scenario,
        "car_speed_kmh":      round(speed_kmh,   1),
        "time_of_day_h":      round(time_h,      2),
        "mec_distance_m":     round(distance_m,  1),
        "mec_rssi_dbm":       round(rssi,         2),
        "mec_channel_bw_mhz": sc["bw_mhz"],
        "mec_load_pct":       round(mec_load,    1),
        "num_connected_ues":  num_ues,
        "handover_active":    handover,
        "payload_kb":         round(payload_kb,  1),
        "cloud_congestion":   round(cloud_cong,  4),
        "cloud_rtt_base_ms":  round(cloud_rtt_b, 1),
        "cloud_load_pct":     round(cloud_load,  1),
        "mec_available":      mec_avail,
        "mec_network_ms":     round(mec_ms,      2),
        "mec_capacity_pct":   round(mec_load,    1),
        "cloud_network_ms":   round(cloud_ms,    2),
        "cloud_capacity_pct": round(cloud_load,  1),
    }


def generate_dataset(
    n:    int  = N_SAMPLES,
    out:  Path = DATA_PATH,
    seed: int  = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate the full synthetic dataset and save it to a CSV file.

    Parameters
    ----------
    n:
        Number of samples to generate.
    out:
        Destination CSV path.  Parent directories are created if missing.
    seed:
        Random seed for full reproducibility.

    Returns
    -------
    pd.DataFrame
        The generated dataset (also persisted to ``out``).
    """
    rng       = np.random.default_rng(seed)
    scenarios = list(SCENARIO_WEIGHTS)
    weights   = [SCENARIO_WEIGHTS[s] for s in scenarios]
    chosen    = rng.choice(scenarios, size=n, p=weights)
    rows      = [_generate_sample(s, rng) for s in chosen]
    df        = pd.DataFrame(rows)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Saved {len(df):,} samples → {out}")
    print(f"\nScenario distribution:\n{df['scenario'].value_counts().to_string()}")
    print(f"\nMEC availability : {df['mec_available'].mean() * 100:.1f}% reachable")
    print(f"\nTarget statistics:")
    print(df[["mec_network_ms", "cloud_network_ms",
              "mec_capacity_pct", "cloud_capacity_pct"]].describe().to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the HUA synthetic dataset.")
    ap.add_argument("--n",    type=int,  default=N_SAMPLES, help="Number of samples")
    ap.add_argument("--out",  type=Path, default=DATA_PATH,  help="Output CSV path")
    ap.add_argument("--seed", type=int,  default=RANDOM_SEED)
    args = ap.parse_args()
    generate_dataset(n=args.n, out=args.out, seed=args.seed)
