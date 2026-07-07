"""
Road-video object detection across all four execution modes.

Annotates each frame with YOLOv8 bounding boxes, mode metadata, and live
inference timing, then writes one MP4 per mode (or an optional side-by-side
grid) to ``output/``.

Detected classes: person, car, motorcycle, bus, truck.

Usage::

    python -m src.detection.detect_video --video road.mp4
    python -m src.detection.detect_video --video road.mp4 --mode D
    python -m src.detection.detect_video --video road.mp4 --mode all --side-by-side
    python -m src.detection.detect_video --source 0          # webcam
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CONF_THRESHOLD, IMG_SIZE, IOU_THRESHOLD,
    MODES, OUTPUT_DIR, TARGET_CLASSES, get_device,
)

_CLASS_COLORS = {
    "person":     (  0, 255, 100),
    "car":        (  0, 180, 255),
    "motorcycle": (255, 100,   0),
    "bus":        (255, 220,   0),
    "truck":      (180,   0, 255),
}
_MODE_COLORS = {
    "A": (255, 200,  50),
    "B": ( 50, 200, 255),
    "C": (100, 255, 100),
    "D": (255, 100, 100),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _load_models(modes: list[str], device: str) -> dict[str, YOLO]:
    cache: dict[str, YOLO] = {}
    for mode in modes:
        name = MODES[mode]["model"]
        if name not in cache:
            print(f"  Loading {name}.pt …", end=" ", flush=True)
            m = YOLO(f"{name}.pt")
            m.to(device)
            cache[name] = m
            print("ok")
    return cache


def _infer(model: YOLO, frame: np.ndarray, device: str) -> tuple[list[dict], float]:
    """Run inference and return detections + elapsed time in ms."""
    t0      = time.perf_counter()
    results = model(
        [frame], imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
        classes=list(TARGET_CLASSES.keys()), verbose=False, device=device,
    )
    inf_ms = (time.perf_counter() - t0) * 1_000.0
    dets   = []
    r      = results[0]
    if r.boxes is not None:
        for xyxy, cls_id, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
        ):
            dets.append({
                "bbox":  xyxy.tolist(),
                "class": TARGET_CLASSES[cls_id],
                "conf":  float(conf),
            })
    return dets, inf_ms


def _annotate(frame: np.ndarray, dets: list[dict], mode: str, inf_ms: float) -> np.ndarray:
    """Draw bounding boxes and a mode-status overlay onto the frame."""
    out = frame.copy()
    row = MODES[mode]
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        col             = _CLASS_COLORS.get(d["class"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label       = f"{d['class']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    net_str   = f"{row['network_ms']:.0f}" if row.get("network_ms") is not None else "N/A"
    total_str = f"{row['total_ms']:.0f}"   if row.get("total_ms")   is not None else "N/A"
    status = (
        f"Mode {mode} | {row['model']} | {row['hw']} | "
        f"inf: {inf_ms:.1f} ms | net: {net_str} ms | total: {total_str} ms | "
        f"acc.loss: {row['acc_loss']*100:.0f}% | cap: {row['capacity_pct']:.0f}%"
    )
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(out, status, (8, 19), _FONT, 0.52, _MODE_COLORS[mode], 1, cv2.LINE_AA)
    return out


def run_detection(
    source:       str,
    modes:        list[str],
    side_by_side: bool = False,
    output_dir:   Path = OUTPUT_DIR,
) -> None:
    """
    Run YOLOv8 detection on a video source and write annotated MP4 output.

    Parameters
    ----------
    source:
        File path to an MP4 or an integer string for webcam index.
    modes:
        List of mode keys to run (subset of ``["A", "B", "C", "D"]``).
    side_by_side:
        If True and multiple modes are given, write a single grid MP4.
    output_dir:
        Destination directory for output videos.
    """
    device = get_device()
    models = _load_models(modes, device)
    cap    = cv2.VideoCapture(int(source) if source.isdigit() else source)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if side_by_side and len(modes) > 1:
        cols      = min(len(modes), 2)
        grid_rows = (len(modes) + 1) // 2
        out_path  = str(Path(output_dir) / "detection_sbs.mp4")
        writer    = cv2.VideoWriter(out_path, fourcc, fps, (w * cols, h * grid_rows))
        writers   = None
    else:
        writer  = None
        writers = {
            m: (
                cv2.VideoWriter(
                    str(Path(output_dir) / f"detection_mode{m}.mp4"), fourcc, fps, (w, h)
                ),
                str(Path(output_dir) / f"detection_mode{m}.mp4"),
            )
            for m in modes
        }

    stats:    dict[str, list] = {m: [] for m in modes}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        annotated: dict[str, np.ndarray] = {}

        for mode in modes:
            model           = models[MODES[mode]["model"]]
            dets, inf_ms    = _infer(model, frame, device)
            stats[mode].append(inf_ms)
            annotated[mode] = _annotate(frame, dets, mode, inf_ms)

        if writer is not None:
            cols = min(len(modes), 2)
            grid = []
            for i in range(0, len(modes), cols):
                row_frames = [
                    annotated[modes[j]] for j in range(i, min(i + cols, len(modes)))
                ]
                while len(row_frames) < cols:
                    row_frames.append(np.zeros_like(frame))
                grid.append(np.hstack(row_frames))
            writer.write(np.vstack(grid))
        else:
            for mode in modes:
                writers[mode][0].write(annotated[mode])

        if frame_idx % 30 == 0:
            parts = "  ".join(f"Mode {m}: {np.mean(stats[m]):.1f} ms" for m in modes)
            print(f"  Frame {frame_idx:5d}  |  {parts}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\n  Saved → {out_path}")
    else:
        for mode, (wr, p) in writers.items():
            wr.release()
            print(f"  Saved → {p}")

    print(f"\n  {'Mode':<5} {'Model':<10} {'Mean (ms)':>10} {'p95 (ms)':>9}")
    print("  " + "-" * 38)
    for mode in modes:
        t = np.array(stats[mode])
        print(
            f"  {mode:<5} {MODES[mode]['model']:<10} "
            f"{t.mean():>10.1f} {np.percentile(t, 95):>9.1f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",        type=str, default="video.mp4")
    ap.add_argument("--source",       type=str, default=None)
    ap.add_argument("--mode",         type=str, action="append", default=None,
                    choices=["A", "B", "C", "D", "all"])
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--output-dir",   type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    src   = args.source or args.video
    raw   = args.mode or ["all"]
    modes = list(MODES.keys()) if "all" in raw else list(dict.fromkeys(raw))

    run_detection(
        source=src,
        modes=modes,
        side_by_side=args.side_by_side,
        output_dir=args.output_dir,
    )
