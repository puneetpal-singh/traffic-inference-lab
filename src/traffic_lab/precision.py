"""Planning and reporting for one-checkpoint precision studies; no inference at import time."""

import math
import random
from statistics import mean, median, stdev

from .confirmation import json_digest, percentile


def cases(target="all"):
    variants = []
    for backend, device, precisions in (
        ("pytorch", "cpu", (32,)), ("pytorch", "0", (32,)),
        ("onnx", "cpu", (32, 16, 8)), ("onnx", "0", (32, 16)),
        ("openvino", "cpu", (32, 16, 8)), ("tensorrt", "0", (32, 16, 8)),
    ):
        if target != "all" and (target == "cpu") != (device == "cpu"):
            continue
        for precision in precisions:
            label = "int8" if precision == 8 else f"fp{precision}"
            variants.append({"name": f"{backend}-{'cpu' if device == 'cpu' else 'cuda'}-{label}",
                             "backend": backend, "device": device, "precision": precision,
                             "export_key": None if backend == "pytorch" else f"{backend}-{label}"})
    return variants


def randomized_rounds(variants, repeats, seed):
    if repeats < 2:
        raise ValueError("At least two trial rounds required")
    rng = random.Random(seed)
    orders = []
    for _ in range(repeats):
        order = [v["name"] for v in variants]
        rng.shuffle(order)
        orders.append(order)
    return orders


def split_images(images, calibration_count, evaluation_limit, seed):
    ordered = sorted(images, key=lambda row: row["id"])
    if len({row["id"] for row in ordered}) != len(ordered):
        raise ValueError("Duplicate dataset image IDs")
    if not 0 < calibration_count < len(ordered) or evaluation_limit < 0:
        raise ValueError("Calibration must leave a nonempty disjoint evaluation set")
    random.Random(seed).shuffle(ordered)
    calibration, evaluation = ordered[:calibration_count], ordered[calibration_count:]
    if evaluation_limit:
        evaluation = evaluation[:evaluation_limit]
    return calibration, sorted(evaluation, key=lambda row: row["id"])


def spread(values):
    values = list(values)
    return {"n": len(values), "median": median(values), "min": min(values), "max": max(values)}


def paired_ratio(candidate, reference):
    """Pair complete randomized rounds only; never substitute frame observations for trials."""
    common = sorted(set(candidate) & set(reference))
    ratios = [candidate[r] / reference[r] for r in common]
    if not ratios:
        return None
    if any(not math.isfinite(v) or v <= 0 for v in ratios):
        raise ValueError("Ratios must be finite and positive")
    logs = [math.log(v) for v in ratios]
    interval = None
    if len(logs) >= 2:
        from scipy.stats import t

        margin = float(t.ppf(.975, len(logs) - 1)) * stdev(logs) / math.sqrt(len(logs))
        interval = [math.exp(mean(logs) - margin), math.exp(mean(logs) + margin)]
    return {"rounds": common, "ratios": ratios, "geometric_mean": math.exp(mean(logs)),
            "log_t_95_interval": interval,
            "caution": "Exploratory single-session interval; independent normal log ratios assumed, drift may violate this."}


def evidence_identity(job, bindings):
    return json_digest({"job": job, "bindings": bindings})


def validate_timing(row, cfg):
    samples = row["samples"]
    if row["seconds"] < cfg["seconds"] or row["frames"] != len(samples):
        raise ValueError("Timing duration/frame count mismatch")
    expected = list(range(0, cfg["pool_frames"] * cfg["frame_step"], cfg["frame_step"]))
    if row["cycles"] < 1 or [s["source_index"] for s in samples] != expected * row["cycles"]:
        raise ValueError("Timing frame-pool composition mismatch")
    if not math.isclose(row["fps"], row["frames"] / row["seconds"], rel_tol=1e-12):
        raise ValueError("Timing FPS mismatch")
    walls = [s["wall_ms"] for s in samples]
    if any(not math.isfinite(w) or w <= 0 for w in walls) or sum(walls) > row["seconds"] * 1000 + 1:
        raise ValueError("Invalid timing samples")
    for key, fraction in (("p50_ms", .5), ("p95_ms", .95)):
        if not math.isclose(row[key], percentile(walls, fraction), rel_tol=1e-12):
            raise ValueError("Timing percentile mismatch")
