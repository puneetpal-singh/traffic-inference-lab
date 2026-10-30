"""Validate published paired-trial evidence without a GPU, model downloads or inference."""

import argparse
import json
import math
from pathlib import Path

from traffic_lab.confirmation import balanced_orders, json_digest, paired_speedup, percentile


def audit(root=Path("results/confirmation")):
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    plan, summary = read(root / "plan.json"), read(root / "summary.json")
    errors, trials = [], []
    if summary["plan_json_sha256"] != json_digest(plan):
        errors.append("Plan identity changed")
    if plan["orders"] != balanced_orders(len(plan["orders"]), plan["seed"]):
        errors.append("Order is not the declared balanced randomized schedule")
    if plan["pool_frames"] != 60 or plan["source_indices"] != list(range(0, 120, 2)):
        errors.append("Unexpected frame-pool composition")
    if plan["minimum_trial_seconds"] < 20 or len(plan["orders"]) < 6:
        errors.append("Published confirmation requires at least six pairs and twenty seconds per trial")
    export = read("results/full-matrix/exports/tensorrt-fp16-320.json")
    if (
        export["source_sha256"] != plan["execution"]["pytorch"]["model_sha256"]
        or export["artifact_sha256"] != plan["execution"]["tensorrt"]["model_sha256"]
        or export["arguments"]["quantize"] != 16
    ):
        errors.append("TensorRT artifact is not the declared FP16 export of the same PyTorch weights")
    baseline_config = plan["execution"]["pytorch"]["config"]
    candidate_config = plan["execution"]["tensorrt"]["config"]
    if any(
        baseline_config[key] != candidate_config[key]
        for key in ("imgsz", "device", "threads", "conf", "iou", "classes", "half")
    ):
        errors.append("Detection settings differ between runtimes")
    original = read("results/full-matrix/timing-pytorch-cuda-fp32-320.json")
    if plan["source_sha256"] != original["input"]["sha256"]:
        errors.append("Source differs from the declared traffic workload")
    qualities = {}
    for runtime, evidence in plan["quality"].items():
        quality = read(evidence["path"])
        qualities[runtime] = quality
        if evidence["json_sha256"] != json_digest(quality):
            errors.append(f"Quality evidence changed: {runtime}")
        execution = plan["execution"][runtime]
        if execution["model_sha256"] != quality["model_sha256"]:
            errors.append(f"Evaluated model does not match timed model: {runtime}")
        if not execution["output_boxes_device"].startswith("cuda:"):
            errors.append(f"No CUDA output evidence: {runtime}")
        if execution["config"]["imgsz"] != 320 or execution["config"]["device"] != "0":
            errors.append(f"Configuration changed: {runtime}")
    if qualities["pytorch"]["dataset_manifest_sha256"] != qualities["tensorrt"]["dataset_manifest_sha256"]:
        errors.append("Quality datasets differ")
    global_drop = 100 * (qualities["pytorch"]["map50_95"] - qualities["tensorrt"]["map50_95"])

    def vehicle(q):
        return sum(q["per_class_map50_95"][str(c)] for c in (2, 3, 5, 7)) / 4

    vehicle_drop = 100 * (vehicle(qualities["pytorch"]) - vehicle(qualities["tensorrt"]))
    for item in summary["trials"]:
        trial = read(root / item["path"])
        trials.append(trial)
        name = item["path"]
        if item["json_sha256"] != json_digest(trial):
            errors.append(f"Trial identity changed: {name}")
        if plan["orders"][trial["pair"]][trial["position"]] != trial["runtime"]:
            errors.append(f"Trial does not follow randomized order: {name}")
        if trial["model_sha256"] != plan["execution"][trial["runtime"]]["model_sha256"]:
            errors.append(f"Wrong model: {name}")
        if (
            trial["frames"] != len(trial["samples"])
            or trial["frames"] != 60 * trial["cycles"]
            or [s["source_index"] for s in trial["samples"]] != plan["source_indices"] * trial["cycles"]
        ):
            errors.append(f"Frame composition mismatch: {name}")
        if trial["seconds"] < plan["minimum_trial_seconds"]:
            errors.append(f"Trial too short: {name}")
        if not math.isclose(trial["fps"], trial["frames"] / trial["seconds"], rel_tol=1e-12):
            errors.append(f"FPS mismatch: {name}")
        walls = [s["wall_ms"] for s in trial["samples"]]
        if any(not math.isfinite(value) or value <= 0 for value in walls):
            errors.append(f"Invalid latency observation: {name}")
        for key, quantile in (("p50_ms", 0.5), ("p95_ms", 0.95)):
            if not math.isclose(trial[key], percentile(walls, quantile), rel_tol=1e-12):
                errors.append(f"Latency summary mismatch: {name}")
    expected_count = 2 * len(plan["orders"])
    if len(trials) != expected_count:
        errors.append(f"Expected {expected_count} trials")
    derived = paired_speedup(trials, plan["seed"])
    for key in ("geometric_mean_speedup", "paired_fps_ratios", "trial_bootstrap_95_interval"):
        actual = summary[key] if isinstance(summary[key], list) else [summary[key]]
        expected = derived[key] if isinstance(derived[key], list) else [derived[key]]
        if len(actual) != len(expected) or not all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(actual, expected)
        ):
            errors.append(f"Paired comparison mismatch: {key}")
    for runtime in ("pytorch", "tensorrt"):
        aggregate = sum(t["frames"] for t in trials if t["runtime"] == runtime) / sum(
            t["seconds"] for t in trials if t["runtime"] == runtime
        )
        if not math.isclose(summary["aggregate_fps"][runtime], aggregate, rel_tol=1e-12):
            errors.append(f"Aggregate FPS mismatch: {runtime}")
    report = {
        "passed": not errors,
        "trials": len(trials),
        "pairs": derived["pairs"],
        "errors": errors,
        "global_map_drop_points": global_drop,
        "vehicle_map_drop_points": vehicle_drop,
        "accuracy_budget_passed": global_drop <= 1 and vehicle_drop <= 1,
    }
    (root / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/confirmation"))
    result = audit(parser.parse_args().input)
    print(json.dumps(result, indent=2))
    raise SystemExit(int(not result["passed"]))
