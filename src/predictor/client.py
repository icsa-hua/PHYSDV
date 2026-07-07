"""
Real-time inference client for the HUA network predictor.

``InferenceClient`` simulates a vehicle driving back and forth between two MEC
towers in a chosen road scenario.  On each call to :meth:`tick` it advances
the car's position by one frame interval, computes a physics-based RSSI value,
builds the observable state vector, and returns a latency/capacity prediction
from the trained ML model.

Usage::

    from src.predictor import InferenceClient

    client = InferenceClient(scenario="urban_dense", hz=10)
    for _ in range(100):
        result = client.tick()
        print(result)
"""

import numpy as np

from config import BS_GAIN_DBI, BS_TX_DBM, MODEL_PATH, SCENARIOS
from src.predictor.model import load_bundle, predict


def _sim_rssi(
    dist_m:   float,
    freq_ghz: float,
    shadow_db: float,
    path_exp: float,
    rng:      np.random.Generator,
) -> float:
    """3GPP-style instantaneous RSSI with log-normal shadowing."""
    pl = 32.4 + path_exp * np.log10(max(dist_m, 1.0)) + 20.0 * np.log10(freq_ghz)
    return BS_TX_DBM + BS_GAIN_DBI - pl + rng.normal(0.0, shadow_db)


class InferenceClient:
    """
    Stateful per-frame network condition predictor for a simulated vehicle.

    The car starts at the midpoint between two MEC towers and drives at constant
    speed (with small Gaussian jitter) until it reaches either boundary, then
    reverses direction.  A 5% probability of handover is applied when the car
    is near a cell edge and travelling above 60 km/h.

    Parameters
    ----------
    scenario:
        Road environment profile — one of ``urban_dense``, ``urban_sparse``,
        ``suburban``, ``highway``, ``rural``.
    hz:
        Prediction rate in frames per second (1–20).
    model_path:
        Path to the serialised model bundle produced by ``src.predictor.model``.
    """

    def __init__(
        self,
        scenario:      str  = "urban_dense",
        hz:            int  = 10,
        model_path          = MODEL_PATH,
        quantile_path       = None,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {list(SCENARIOS)}")

        self._bundle    = load_bundle(model_path)
        self._qbundle   = None
        if quantile_path is not None:
            from src.predictor.quantile import load_quantile_bundle
            self._qbundle = load_quantile_bundle(quantile_path)
        self._scenario  = scenario
        self._profile   = SCENARIOS[scenario]
        self._hz        = hz
        self._rng       = np.random.default_rng(0)
        self._dist      = float(self._profile["mec_spacing_m"] * 0.5)
        self._direction = -1.0

    def tick(self) -> dict:
        """
        Advance the simulation by one frame interval and return a prediction.

        Returns
        -------
        dict
            Keys: ``mec_available``, ``mec_network_ms``, ``mec_capacity_pct``,
            ``cloud_network_ms``, ``cloud_capacity_pct``, ``dist_m``,
            ``rssi_dbm``, ``scenario``.
        """
        prof     = self._profile
        interval = 1.0 / self._hz
        speed    = prof["speed_range"][0] + (prof["speed_range"][1] - prof["speed_range"][0]) / 2.0

        self._dist += self._direction * (speed / 3.6) * interval
        if self._dist < 30.0:
            self._direction = 1.0
        elif self._dist > prof["mec_spacing_m"] * 1.05:
            self._direction = -1.0

        rssi     = _sim_rssi(self._dist, prof["freq_ghz"], prof["shadow_db"],
                              prof["path_exp"], self._rng)
        cell_edge = self._dist > prof["mec_spacing_m"] * 0.7
        ho_prob   = 0.30 if (cell_edge and speed > 60) else 0.05

        state = {
            "scenario":           self._scenario,
            "car_speed_kmh":      speed + self._rng.normal(0.0, 3.0),
            "time_of_day_h":      8.5,
            "mec_distance_m":     max(10.0, self._dist + self._rng.normal(0.0, 5.0)),
            "mec_rssi_dbm":       rssi,
            "mec_channel_bw_mhz": prof["bw_mhz"],
            "mec_load_pct":       float(np.clip(50.0 + self._rng.normal(0.0, 15.0), 0.0, 100.0)),
            "num_connected_ues":  max(1, int(self._rng.poisson(prof["ue_lambda"]))),
            "handover_active":    int(self._rng.random() < ho_prob),
            "payload_kb":         float(np.clip(self._rng.normal(80.0, 15.0), 20.0, 200.0)),
            "cloud_congestion":   float(np.clip(self._rng.beta(1.5, 4.0), 0.0, 1.0)),
            "cloud_rtt_base_ms":  (prof["cloud_rtt_range"][0] + prof["cloud_rtt_range"][1]) / 2.0
                                  + self._rng.normal(0.0, 5.0),
            "cloud_load_pct":     float(np.clip(30.0 + self._rng.normal(0.0, 10.0), 0.0, 100.0)),
        }

        result              = predict(self._bundle, state)
        if self._qbundle is not None:
            from src.predictor.quantile import predict_quantile
            result.update(predict_quantile(self._qbundle, state))
        result["dist_m"]    = round(self._dist, 1)
        result["rssi_dbm"]  = round(rssi, 1)
        result["scenario"]  = self._scenario
        return result


if __name__ == "__main__":
    import time

    SCENARIO = "urban_dense"
    HZ       = 10

    client = InferenceClient(scenario=SCENARIO, hz=HZ)
    hdr    = (f"{'Frame':>6}  {'Dist(m)':>8}  {'RSSI':>10}  {'MEC':>5}  "
              f"{'MEC ms':>8}  {'Cloud ms':>9}  {'MEC cap%':>9}  {'Cloud cap%':>11}")
    print(f"\nInference stream  |  scenario={SCENARIO}  |  {HZ} Hz  (Ctrl+C to stop)\n")
    print(hdr)
    print("─" * len(hdr))

    for frame in range(1, 10_001):
        res     = client.tick()
        avail   = "YES" if res["mec_available"] else "NO"
        mec_str = f"{res['mec_network_ms']:.1f}" if res["mec_available"] else "N/A"
        print(
            f"{frame:>6}  {res['dist_m']:>8.0f}  {res['rssi_dbm']:>10.1f}  {avail:>5}  "
            f"{mec_str:>8}  {res['cloud_network_ms']:>9.1f}  "
            f"{res['mec_capacity_pct']:>9.1f}  {res['cloud_capacity_pct']:>11.1f}"
        )
        time.sleep(1.0 / HZ)
