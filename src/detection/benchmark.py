"""
YOLOv8 inference latency benchmark across all four execution modes.

Measures per-model inference time statistics (mean, std, p50, p95, p99) on
the local device using randomised frames at the target resolution.  Results
are written to ``output/benchmark_results.json`` and injected into the
performance table for accurate total-latency computation.

Usage::

    python -m src.detection.benchmark
    python -m src.detection.benchmark --runs 100
    python -m src.detection.benchmark --net C:22 D:55 --cap C:60 D:10
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config import (
    BENCHMARK_RUNS, CONF_THRESHOLD, FRAME_H, FRAME_W,
    IMG_SIZE, IOU_THRESHOLD, MODES, OUTPUT_DIR,
    TARGET_CLASSES, WARMUP_RUNS, get_device,
)
from src.offloading.table import PerformanceTable


def _load_model(name: str, device: str) -> YOLO:
    m = YOLO(f"{name}.pt")
    m.to(device)
    return m


def measure_latency(model: YOLO, device: str, n_runs: int) -> dict:
    """
    Measure single-frame inference latency statistics over ``n_runs`` trials.

    A warm-up phase is performed first to exclude JIT/CUDA initialization
    overhead from the reported numbers.

    Parameters
    ----------
    model:
        Loaded Ultralytics YOLO model.
    device:
        Target compute device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    n_runs:
        Number of timed inference runs.

    Returns
    -------
    dict
        Keys: ``mean_ms``, ``std_ms``, ``p50_ms``, ``p95_ms``, ``p99_ms``.
    """
    frame = np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)

    for _ in range(WARMUP_RUNS):
        model(
            [frame], imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
            classes=list(TARGET_CLASSES.keys()), verbose=False, device=device,
        )

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(
            [frame], imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            classes=list(TARGET_CLASSES.keys()), verbose=False, device=device,
        )
        times.append((time.perf_counter() - t0) * 1_000.0)

    t = np.array(times)
    return {
        "mean_ms": round(float(t.mean()),                  2),
        "std_ms":  round(float(t.std()),                   2),
        "p50_ms":  round(float(np.percentile(t, 50)),      2),
        "p95_ms":  round(float(np.percentile(t, 95)),      2),
        "p99_ms":  round(float(np.percentile(t, 99)),      2),
    }


def run_benchmark(
    n_runs:       int                    = BENCHMARK_RUNS,
    network_ms:   dict[str, float] | None = None,
    capacity_pct: dict[str, float] | None = None,
    output_dir:   Path                  = OUTPUT_DIR,
) -> dict:
    """
    Run the full benchmark, update the performance table, and save results.

    Parameters
    ----------
    n_runs:
        Timed inference runs per model.
    network_ms:
        Optional live network latencies (from HUA predictor) to inject.
    capacity_pct:
        Optional live capacity utilisation to inject.
    output_dir:
        Directory for the JSON result file.

    Returns
    -------
    dict
        The merged performance table.
    """
    device = get_device()
    print(f"\nDevice : {device}")
    print(f"Runs   : {n_runs} per model  (+ {WARMUP_RUNS} warm-up)\n")

    pt = PerformanceTable(network_ms=network_ms, capacity_pct=capacity_pct)

    model_cache:     dict[str, YOLO] = {}
    timing_by_model: dict[str, dict] = {}
    timing_by_mode:  dict[str, dict] = {}

    for mode, row in pt.get().items():
        name = row["model"]
        if name not in model_cache:
            print(f"  Loading {name}.pt …", end=" ", flush=True)
            model_cache[name]    = _load_model(name, device)
            timing_by_model[name] = measure_latency(model_cache[name], device, n_runs)
            print(
                f"ok  mean {timing_by_model[name]['mean_ms']:.1f} ms  "
                f"p95 {timing_by_model[name]['p95_ms']:.1f} ms"
            )
        timing_by_mode[mode] = timing_by_model[name]

    pt.update_inference_latency(timing_by_model)
    table = pt.get()

    _print_optimizer_table(table)
    _print_timing_table(timing_by_mode)

    out_path = Path(output_dir) / "benchmark_results.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\n  Saved → {out_path}")
    return table


def _print_optimizer_table(table: dict) -> None:
    print("\n" + "=" * 78)
    print("  PERFORMANCE TABLE")
    print("=" * 78)
    print(
        f"  {'Mode':<5} {'Model':<10} {'HW':<11} {'Inf (ms)':>9} {'Net (ms)':>9} "
        f"{'Total (ms)':>11} {'mAP':>7} {'Acc.Loss':>9} {'Cap (%)':>8}"
    )
    print("  " + "-" * 76)
    for mode, r in table.items():
        net_str   = f"{r['network_ms']:.0f}"  if r.get("network_ms")  is not None else "N/A"
        total_str = f"{r['total_ms']:.1f}"    if r.get("total_ms")    is not None else "N/A"
        print(
            f"  {mode:<5} {r['model']:<10} {r['hw']:<11} "
            f"{r['inference_ms']:>9.1f} {net_str:>9} "
            f"{total_str:>11} {r['map50_95']:>7.1f} "
            f"{r['acc_loss']*100:>8.1f}% {r['capacity_pct']:>7.0f}%"
        )


def _print_timing_table(timing: dict) -> None:
    print(
        f"\n  {'Mode':<5} {'mean (ms)':>10} {'p50 (ms)':>9} "
        f"{'p95 (ms)':>9} {'p99 (ms)':>9} {'std':>7}"
    )
    print("  " + "-" * 52)
    for mode, t in timing.items():
        print(
            f"  {mode:<5} {t['mean_ms']:>10.1f} {t['p50_ms']:>9.1f} "
            f"{t['p95_ms']:>9.1f} {t['p99_ms']:>9.1f} {t['std_ms']:>7.2f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=BENCHMARK_RUNS)
    ap.add_argument("--net",  nargs="*", metavar="MODE:MS")
    ap.add_argument("--cap",  nargs="*", metavar="MODE:PCT")
    args = ap.parse_args()

    def _parse_kv(pairs):
        return {k: float(v) for k, v in (p.split(":") for p in pairs)} if pairs else None

    run_benchmark(
        n_runs=args.runs,
        network_ms=_parse_kv(args.net),
        capacity_pct=_parse_kv(args.cap),
    )
