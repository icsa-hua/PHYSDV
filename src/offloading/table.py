"""
Live performance table for the four YOLOv8 execution modes.

``PerformanceTable`` merges the static hardware benchmarks from ``config.MODES``
with runtime network latency and server capacity values supplied by the HUA
predictor.  The merged table is the direct input to the offloading optimizer.
"""

from config import MODES


class PerformanceTable:
    """
    Combines static hardware benchmarks with live HUA network predictions.

    Modes A and B are on-board (zero network latency) and are never updated
    by the HUA predictor.  Modes C (MEC) and D (Cloud) receive live ``network_ms`` and
    ``capacity_pct`` values on each call to :meth:`update`.

    Parameters
    ----------
    network_ms:
        Live round-trip network latency per mode, e.g. ``{"C": 22.5, "D": 63.1}``.
    capacity_pct:
        Live compute utilisation per mode (0–100), e.g. ``{"C": 47.0, "D": 28.3}``.
    """

    def __init__(
        self,
        network_ms:   dict[str, float] | None = None,
        capacity_pct: dict[str, float] | None = None,
    ) -> None:
        self._table = {mode: dict(row) for mode, row in MODES.items()}
        if network_ms:
            for mode, val in network_ms.items():
                if mode in self._table:
                    self._table[mode]["network_ms"] = val
                    self._table[mode]["total_ms"]   = (
                        self._table[mode]["inference_ms"] + val
                    )
        if capacity_pct:
            for mode, val in capacity_pct.items():
                if mode in self._table:
                    self._table[mode]["capacity_pct"] = val

    def update(
        self,
        network_ms:   dict[str, float],
        capacity_pct: dict[str, float],
    ) -> None:
        """
        Push new HUA predictor predictions into the table.

        Parameters
        ----------
        network_ms:
            Updated round-trip latencies for modes C and/or D.
        capacity_pct:
            Updated capacity utilisation values for modes C and/or D.
        """
        for mode, val in network_ms.items():
            if mode in self._table:
                self._table[mode]["network_ms"] = val
                self._table[mode]["total_ms"]   = (
                    self._table[mode]["inference_ms"] + val
                )
        for mode, val in capacity_pct.items():
            if mode in self._table:
                self._table[mode]["capacity_pct"] = val

    def get(self) -> dict[str, dict]:
        """Return the full merged performance table."""
        return self._table

    def mark_mec_unavailable(self) -> None:
        """
        Mark Mode C as unreachable by setting network_ms to None.

        Called when the HUA classifier predicts ``mec_available = 0``.
        """
        self._table["C"]["network_ms"] = None
        self._table["C"]["total_ms"]   = None

    def update_inference_latency(self, timing: dict[str, dict]) -> None:
        """
        Inject measured inference latencies from a live benchmark run.

        Parameters
        ----------
        timing:
            Dict keyed by model name (e.g. ``"yolov8n"``), each containing
            ``mean_ms`` from the benchmark.
        """
        for mode, row in self._table.items():
            name = row["model"]
            if name in timing:
                row["inference_ms"] = timing[name]["mean_ms"]
                net                 = row.get("network_ms") or 0.0
                row["total_ms"]     = round(row["inference_ms"] + net, 1)
