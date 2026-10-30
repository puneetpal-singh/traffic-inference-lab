"""Audit full matrix coverage, matching quality evidence, device placement, and calibration separation."""

import json
from pathlib import Path

root = Path("results/full-matrix")


def read(name):
    return json.loads((root / name).read_text(encoding="utf-8"))


plan = read("plan.json")
manifest = read("dataset-manifest.json")
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
    if spec["backend"] == "openvino" and inspection.get("execution_devices") != ["CPU"]:
        errors.append(f"OpenVINO CPU case ran elsewhere: {name}")
report = {
    "passed": not errors,
    "planned": len(plan),
    "completed": completed,
    "calibration_images": len(calibration),
    "holdout_images": len(holdout),
    "errors": errors,
}
(root / "audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
raise SystemExit(int(bool(errors)))
