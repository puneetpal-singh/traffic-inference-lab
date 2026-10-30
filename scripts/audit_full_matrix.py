"""Audit full matrix coverage, matching quality evidence, device placement, and calibration separation."""

import argparse
import hashlib
import json
from pathlib import Path

root = Path("results/full-matrix")
parser = argparse.ArgumentParser()
parser.add_argument("--require-pipelines", action="store_true")
args = parser.parse_args()


def read(name):
    return json.loads((root / name).read_text(encoding="utf-8"))


plan = read("plan.json")
manifest = read("dataset-manifest.json")
manifest_digest = hashlib.sha256((root / "dataset-manifest.json").read_bytes()).hexdigest()
errors = []
calibration = {row["id"] for row in manifest["calibration"]}
holdout = {row["id"] for row in manifest["holdout"]}
if len(calibration) != 512 or len(holdout) != 256 or calibration & holdout:
    errors.append("Calibration/evaluation split is incomplete or overlaps")
completed = 0
for spec in plan:
    name = spec["name"]
    if not (root / f"{name}.json").exists():
        errors.append(f"Missing completed case: {name}")
        continue
    completed += 1
    timing, quality = read(f"timing-{name}.json"), read(f"quality-{name}.json")
    case = read(f"{name}.json")
    if (
        case["map50_95"] != quality["map50_95"]
        or case["p95_ms"] != timing["batch_wall_ms"]["p95"]
        or case["throughput_fps"] != timing["aggregate_fps"]
    ):
        errors.append(f"Case summary does not match its source measurements: {name}")
    if quality.get("dataset_manifest_sha256") != manifest_digest:
        errors.append(f"Evaluation dataset identity mismatch: {name}")
    if timing["input"]["frames"] != 120 or timing["repeats"] != 5 or timing["warmup_calls"] != 20:
        errors.append(f"Full-study timing protocol mismatch: {name}")
    if timing["model_sha256"] != quality["model_sha256"]:
        errors.append(f"Mismatched model hashes: {name}")
    if "holdout.yaml" not in quality["data"]:
        errors.append(f"Unexpected quality dataset: {name}")
    for repeat in range(timing["repeats"]):
        actual = sum(s["batch_size"] for s in timing["samples"] if s["repeat"] == repeat)
        if actual != timing["input"]["frames"]:
            errors.append(f"Frame accounting error: {name} repeat {repeat}")
    if spec["device"] != "cpu" and not timing["cuda_available"]:
        errors.append(f"CUDA requested without CUDA: {name}")
    inspection = read(f"inspection-{name}.json")
    if spec["device"] != "cpu" and spec["backend"] == "onnx":
        if not inspection.get("weighted_kernel_providers", {}).get("CUDAExecutionProvider"):
            errors.append(f"ONNX weighted kernels did not execute on CUDA: {name}")
    if spec["backend"] == "onnx" and spec["precision"] == 8:
        if not any(
            key.endswith((":int8", ":uint8")) and count > 0
            for key, count in inspection.get("weighted_kernel_input_types", {}).items()
        ):
            errors.append(f"No executed INT8 weighted-kernel tensor evidence for ONNX: {name}")
    if spec["backend"] == "openvino" and inspection.get("execution_devices") != ["CPU"]:
        errors.append(f"OpenVINO CPU case ran elsewhere: {name}")
    if spec["backend"] == "openvino" and spec["precision"] == 8:
        if not any(
            "i8" in key.lower() and count > 0
            for key, count in inspection.get("runtime_precision_counts", {}).items()
        ):
            errors.append(f"No native INT8 execution evidence for OpenVINO: {name}")
    if spec["backend"] == "tensorrt" and spec["precision"] == 8:
        layers = inspection.get("engine_layers", {}).get("Layers", [])
        weighted_int8 = [
            layer
            for layer in layers
            if any(op in layer.get("LayerType", "").lower() for op in ("conv", "gemm", "matmul"))
            and "int8" in json.dumps(layer).lower()
        ]
        if not weighted_int8:
            errors.append(f"No native INT8 weighted TensorRT layer evidence: {name}")
report = {
    "passed": not errors,
    "planned": len(plan),
    "completed": completed,
    "calibration_images": len(calibration),
    "holdout_images": len(holdout),
    "errors": errors,
}
pipeline_count = 0
if args.require_pipelines:
    for path in (root / "pipelines").glob("*/summary.json"):
        pipeline_count += 1
        pipeline = json.loads(path.read_text())
        if pipeline["status"] != "complete":
            errors.append(f"Incomplete pipeline: {path.parent.name}")
        expected_batches = list(range(1, min(pipeline["requested_batch"], len(pipeline["cameras"])) + 1))
        if pipeline.get("warmup") != {"zero_image_calls": 5, "real_image_batch_sizes": expected_batches}:
            errors.append(f"Serving warm-up protocol mismatch: {path.parent.name}")
        expected_grid = 64 if path.parent.name.endswith("-sampled-telemetry") else 0
        if pipeline.get("brightness_grid", 0) != expected_grid:
            errors.append(f"Telemetry sampling protocol mismatch: {path.parent.name}")
        for camera in pipeline["cameras"]:
            accounted = sum(camera[k] for k in ("processed", "queue_dropped", "stride_skipped", "remaining"))
            if accounted != camera["decoded"]:
                errors.append(f"Pipeline accounting mismatch: {path.parent.name} camera {camera['camera']}")
            if camera["queue_high_water"] > pipeline["queue_capacity_per_camera"]:
                errors.append(f"Queue bound exceeded: {path.parent.name}")
            if camera.get("decoder_threads") != 1:
                errors.append(f"Decoder thread budget not applied: {path.parent.name}")
    if pipeline_count < 24:
        errors.append(f"Expected 24 serving experiments, found {pipeline_count}")
report.update(passed=not errors, errors=errors, pipeline_runs=pipeline_count)
(root / "audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
raise SystemExit(int(bool(errors)))
