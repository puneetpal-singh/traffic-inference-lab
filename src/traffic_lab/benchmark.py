"""Predecoded model benchmarks; these are explicitly NOT live pipeline latency."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil

from .metrics import distribution, match_detections
from .resources import ResourceMonitor
from .runtime import Detector, environment, execution_device, file_hash, save_json, video_frames


def benchmark(config, source, output, frames=60, repeats=3, warmup=5, batch=1):
    if min(frames, repeats, batch) < 1 or warmup < 1:
        raise ValueError("frames, repeats, batch and warmup must be positive")
    detector = Detector(config)
    images = video_frames(source, frames)
    detector.warmup(warmup, batch)
    process = psutil.Process()
    raw, trials = [], []
    with ResourceMonitor() as monitor:
        for repeat in range(repeats):
            cpu_start = sum(process.cpu_times()[:2])
            start = time.perf_counter()
            for offset in range(0, len(images), batch):
                group = images[offset : offset + batch]
                tick = time.perf_counter()
                results = detector.predict(group)
                elapsed = (time.perf_counter() - tick) * 1000
                # CPU conversion synchronizes pending device work before reporting completion.
                detections = [r.boxes.cpu().numpy().data for r in results]
                elapsed = (time.perf_counter() - tick) * 1000
                raw.append(
                    {
                        "repeat": repeat,
                        "offset": offset,
                        "batch_size": len(group),
                        "batch_wall_ms": elapsed,
                        "stages_per_image_ms": {
                            k: float(np.mean([r.speed[k] for r in results]))
                            for k in ("preprocess", "inference", "postprocess")
                        },
                        "detections": sum(len(x) for x in detections),
                        "rss_mib": process.memory_info().rss / 1024**2,
                    }
                )
            wall = time.perf_counter() - start
            trials.append(
                {
                    "repeat": repeat,
                    "seconds": wall,
                    "fps": len(images) / wall,
                    "process_cpu_one_core_percent": 100 * (sum(process.cpu_times()[:2]) - cpu_start) / wall,
                }
            )
    result = {
        "kind": "predecoded_model_benchmark",
        "schema_version": 1,
        "environment": environment(),
        **detector.metadata(),
        "input": {"path": str(source), "sha256": file_hash(source), "frames": len(images)},
        "warmup_calls": warmup,
        "batch": batch,
        "repeats": repeats,
        "batch_wall_ms": distribution([r["batch_wall_ms"] for r in raw]),
        "fps": distribution([r["fps"] for r in trials]),
        "aggregate_fps": len(images) * repeats / sum(r["seconds"] for r in trials),
        "peak_sampled_rss_mib": max(r["rss_mib"] for r in raw),
        "quality": None,
        "resources": monitor.summary(),
        "trials": trials,
        "samples": raw,
        "limitations": [
            "Decode, queues, tracking, rendering excluded.",
            "RSS includes predecoded frames and runtime; sampled, not true peak.",
            "Backend stage timers are implementation dependent; compare wall time.",
            "No accuracy claim without a separately labeled evaluation.",
        ],
    }
    save_json(output, result)
    return result


def parity(reference, candidate, source, output, frames=30, min_iou=0.9, max_conf_delta=0.02):
    images = video_frames(source, frames)
    baseline = Detector(reference)
    baseline.warmup()
    ref = [baseline.predict([frame])[0].boxes.cpu().numpy().data.copy() for frame in images]
    del baseline
    target = Detector(candidate)
    target.warmup()
    comparisons = []
    for idx, (frame, boxes) in enumerate(zip(images, ref)):
        other = target.predict([frame])[0].boxes.cpu().numpy().data
        row = match_detections(boxes, other, min_iou)
        row["frame"] = idx
        row["passed"] = row["matched"] == row["reference"] == row["candidate"] and (
            row["max_conf_delta"] is None or row["max_conf_delta"] <= max_conf_delta
        )
        comparisons.append(row)
    result = {
        "kind": "detection_parity",
        "reference": asdict(reference),
        "candidate": target.metadata(),
        "reference_model_sha256": file_hash(reference.model),
        "source_sha256": file_hash(source),
        "environment": environment(),
        "min_iou": min_iou,
        "max_conf_delta": max_conf_delta,
        "frames": comparisons,
        "passed": all(r["passed"] for r in comparisons),
        "note": "Detection agreement on sampled frames is not mAP or domain accuracy.",
    }
    save_json(output, result)
    return result


def evaluate(config, data, output, split="val"):
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(config.threads)
    model = YOLO(config.model, task="detect")
    # Use validation's low confidence threshold; deployment threshold would truncate AP.
    result = model.val(
        data=data,
        split=split,
        imgsz=config.imgsz,
        device=execution_device(config),
        batch=1,
        workers=0,
        plots=False,
        verbose=False,
        conf=0.001,
        iou=config.iou,
        rect=False,
        project="runs/validation",
        name=Path(output).stem,
        quantize=16 if config.half else 32,
    )
    metrics = {
        "kind": "labeled_detection_evaluation",
        "environment": environment(),
        "config": asdict(config),
        "model_sha256": file_hash(config.model),
        "data": str(data),
        "dataset_yaml_sha256": file_hash(data) if Path(data).is_file() else None,
        "dataset_manifest_sha256": file_hash("results/full-matrix/dataset-manifest.json")
        if Path(data).resolve() == Path("data/coco-deployment/holdout.yaml").resolve()
        and Path("results/full-matrix/dataset-manifest.json").exists()
        else None,
        "split": split,
        "map50_95": float(result.box.map),
        "map50": float(result.box.map50),
        "per_class_map50_95": {
            str(int(k)): float(v) for k, v in zip(result.box.ap_class_index, result.box.ap)
        },
        "note": "All COCO classes evaluated; use a traffic holdout before deployment.",
    }
    save_json(output, metrics)
    return metrics
