"""
QoS-aware offloading decision function.

``select_mode`` takes the live performance table and returns the single best
execution mode for the current frame, together with a human-readable reason
string that explains the decision.

Decision policy
---------------
1. Filter to *feasible* modes: total latency ≤ budget, accuracy loss ≤ budget,
   server capacity < saturation threshold, and MEC network reachable for Mode C.
2. Apply edge-first preference: if Mode C (MEC) is feasible and Mode D (Cloud)
   offers less than ``MEC_PREFERENCE_MAP_DELTA`` mAP points of improvement,
   prefer C — it has lower latency and conserves WAN bandwidth.
3. Among the remaining feasible modes, pick the one with the highest mAP.
4. If no mode is feasible (e.g. extreme congestion), fall back to the fastest
   on-board mode (B) and flag the reason.

The edge-first rule produces the switching behaviour visible in the demo:
the system normally offloads to MEC, falls back to Cloud when MEC is
unavailable or saturated, and uses on-board GPU as last resort.

QoS budgets and preference delta are read from ``config``:

* ``LATENCY_BUDGET_MS``        = 100 ms
* ``ACC_LOSS_BUDGET``          = 0.15
* ``CAPACITY_LIMIT_PCT``       = 95 %
* ``MEC_PREFERENCE_MAP_DELTA`` = 5.0 mAP points
"""

from config import (
    ACC_LOSS_BUDGET, CAPACITY_LIMIT_PCT, LATENCY_BUDGET_MS,
    MEC_PREFERENCE_MAP_DELTA,
)

_FALLBACK_MODE = "B"


def _is_feasible(mode: str, row: dict) -> tuple[bool, str]:
    """
    Check whether a mode satisfies all QoS constraints.

    Parameters
    ----------
    mode:
        Mode key (``"A"``, ``"B"``, ``"C"``, or ``"D"``).
    row:
        Single row of the performance table.

    Returns
    -------
    (feasible, reason)
        ``feasible`` is True when the mode passes all constraints.
        ``reason`` names the first violated constraint (empty string if feasible).
    """
    if mode == "C" and row.get("network_ms") is None:
        return False, "mec_unreachable"

    total_ms = row.get("total_ms") or (row["inference_ms"] + (row.get("network_ms") or 0.0))

    if total_ms > LATENCY_BUDGET_MS:
        return False, f"latency_{total_ms:.0f}ms_over_budget"
    if row["acc_loss"] > ACC_LOSS_BUDGET:
        return False, f"acc_loss_{row['acc_loss']*100:.0f}pct_over_budget"
    if row["capacity_pct"] >= CAPACITY_LIMIT_PCT:
        return False, f"capacity_{row['capacity_pct']:.0f}pct_saturated"

    return True, ""


def select_mode(table: dict[str, dict]) -> tuple[str, str]:
    """
    Select the optimal execution mode for the current frame.

    Parameters
    ----------
    table:
        Live performance table returned by ``PerformanceTable.get()``.

    Returns
    -------
    (mode, reason)
        ``mode`` is the selected key (``"A"``–``"D"``).
        ``reason`` is a short human-readable string explaining the decision.

    Examples
    --------
    >>> from src.offloading import PerformanceTable, select_mode
    >>> pt = PerformanceTable(network_ms={"C": 22.0, "D": 60.0},
    ...                       capacity_pct={"C": 40.0, "D": 25.0})
    >>> mode, reason = select_mode(pt.get())
    >>> print(mode, reason)
    C edge_first_within_map_delta
    """
    feasible: dict[str, dict] = {}

    for mode, row in table.items():
        ok, _ = _is_feasible(mode, row)
        if ok:
            feasible[mode] = row

    if not feasible:
        return _FALLBACK_MODE, "fallback_no_feasible_mode"

    if "C" in feasible and "D" in feasible:
        map_delta = feasible["D"]["map50_95"] - feasible["C"]["map50_95"]
        if map_delta < MEC_PREFERENCE_MAP_DELTA:
            return "C", "edge_first_within_map_delta"
        return "D", "cloud_accuracy_gain_justifies_wan"

    best_mode = max(feasible, key=lambda m: feasible[m]["map50_95"])

    if len(feasible) == 1:
        reason = "only_feasible_mode"
    elif best_mode == "D":
        reason = "mec_unavailable_cloud_fallback"
    elif best_mode in ("A", "B"):
        reason = "remote_tiers_infeasible_onboard_fallback"
    else:
        reason = "best_feasible"

    return best_mode, reason


class DwellTimeStabilizer:
    """
    Minimum-dwell hysteresis filter for the per-frame mode decision.

    Suppresses ping-pong switching under noisy network predictions: a
    proposed mode change is committed only after it persists for ``k``
    consecutive frames.  Wrap :func:`select_mode` output::

        stab = DwellTimeStabilizer(k=5)
        mode, reason = select_mode(table)
        mode = stab.apply(mode)

    Parameters
    ----------
    k:
        Number of consecutive frames a new mode must be proposed before
        the switch is committed (``k=1`` disables hysteresis).
    """

    def __init__(self, k: int = 5) -> None:
        self.k         = k
        self._current  = None
        self._cand     = None
        self._streak   = 0

    def apply(self, proposed: str, force: bool = False) -> str:
        """
        Return the stabilised mode for the current frame.

        Parameters
        ----------
        proposed:
            Mode proposed by :func:`select_mode` for this frame.
        force:
            Commit the switch immediately, bypassing hysteresis.  Used for
            safety-critical transitions (e.g. ``fallback_no_feasible_mode``),
            where delaying the fallback would leave the vehicle on an
            unreachable tier.
        """
        if force or self._current is None:
            self._current = proposed
            self._cand, self._streak = None, 0
            return proposed
        if proposed == self._current:
            self._cand, self._streak = None, 0
            return self._current
        if proposed == self._cand:
            self._streak += 1
        else:
            self._cand, self._streak = proposed, 1
        if self._streak >= self.k:
            self._current = self._cand
            self._cand, self._streak = None, 0
        return self._current

    def reset(self) -> None:
        """Clear all hysteresis state."""
        self._current = None
        self._cand    = None
        self._streak  = 0


def explain(table: dict[str, dict]) -> dict[str, dict]:
    """
    Return per-mode feasibility annotations for display purposes.

    Parameters
    ----------
    table:
        Live performance table.

    Returns
    -------
    dict
        Same keys as ``table``; each value is a dict with ``feasible`` (bool)
        and ``reason`` (str).
    """
    return {
        mode: {"feasible": ok, "reason": reason}
        for mode, row in table.items()
        for ok, reason in [_is_feasible(mode, row)]
    }
