"""Audit measurement accounting and provenance without rerunning inference."""

import json
from pathlib import Path


def main():
    errors, benchmarks, pipelines, parity = [], 0, 0, 0
    for path in Path("results").rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        kind = data.get("kind")
        if kind == "predecoded_model_benchmark":
            benchmarks += 1
            if not data["model_sha256"] or not data["input"]["sha256"]:
                errors.append(f"{path}: missing hash")
            for repeat in range(data["repeats"]):
                frames = sum(s["batch_size"] for s in data["samples"] if s["repeat"] == repeat)
                if frames != data["input"]["frames"]:
                    errors.append(f"{path}: repeat {repeat} frame accounting mismatch")
        elif kind == "paced_recorded_multistream":
            pipelines += 1
            if data["status"] != "complete":
                errors.append(f"{path}: incomplete")
            for camera in data["cameras"]:
                total = sum(camera[k] for k in ("processed", "queue_dropped", "stride_skipped", "remaining"))
                if total != camera["decoded"]:
                    errors.append(f"{path}: camera frame accounting mismatch")
                if camera["queue_high_water"] > data["queue_capacity_per_camera"]:
                    errors.append(f"{path}: exceeded queue bound")
        elif kind == "detection_parity":
            parity += 1
            if not data["passed"]:
                errors.append(f"{path}: parity gate failed")
    if benchmarks < 6 or pipelines < 6 or parity < 4:
        errors.append(
            f"Expected >=6 benchmarks, >=6 pipelines, >=4 parity checks; got {benchmarks}/{pipelines}/{parity}"
        )
    print(
        json.dumps(
            {
                "passed": not errors,
                "benchmarks": benchmarks,
                "pipelines": pipelines,
                "parity_checks": parity,
                "errors": errors,
            },
            indent=2,
        )
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
