"""One isolated process per case, with identical data and preprocessing across all backends."""

import argparse
import gc
import json
from pathlib import Path

from traffic_lab.benchmark import benchmark, evaluate, parity
from traffic_lab.inspection import inspect_artifact
from traffic_lab.runtime import ModelConfig, file_hash, save_json


def clean():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.case)
    root = Path("results/full-matrix")
    config = ModelConfig(model=spec["model"], imgsz=spec["imgsz"], device=spec["device"], half=spec["half"])
    model_hash = file_hash(config.model)
    video_hash = file_hash("data/traffic.mp4")
    dataset_hash = file_hash("data/coco-deployment/holdout.yaml")

    def cached(prefix, candidate=False):
        path = root / f"{prefix}-{spec['name']}.json"
        if args.force or not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        metadata = data.get("candidate", {}) if candidate else data
        if metadata.get("model_sha256") != model_hash:
            return None
        old = metadata.get("config", {})
        if any(
            old.get(k) != getattr(config, k)
            for k in ("model", "imgsz", "device", "half", "threads", "conf", "iou")
        ):
            return None
        if prefix == "timing" and (
            data.get("repeats") != args.repeats
            or data.get("warmup_calls") != args.warmup
            or data.get("input", {}).get("frames") != args.frames
        ):
            return None
        if prefix == "timing" and data.get("input", {}).get("sha256") != video_hash:
            return None
        if prefix == "quality" and data.get("dataset_yaml_sha256") != dataset_hash:
            return None
        if prefix == "parity" and (
            data.get("source_sha256") != video_hash
            or data.get("reference_model_sha256") != file_hash("models/yolo11n.pt")
        ):
            return None
        print(f"Reusing completed {prefix} artifact", flush=True)
        return data

    print(f"Timing {spec['name']}: {args.frames} frames x {args.repeats} trials", flush=True)
    timing = cached("timing") or benchmark(
        config,
        "data/traffic.mp4",
        root / f"timing-{spec['name']}.json",
        frames=args.frames,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    clean()
    print("Timing complete; evaluating 256 held-out images", flush=True)
    quality = cached("quality") or evaluate(
        config, "data/coco-deployment/holdout.yaml", root / f"quality-{spec['name']}.json"
    )
    clean()
    print("Accuracy evaluation complete; checking detection agreement", flush=True)
    reference = ModelConfig(imgsz=spec["imgsz"], device="cpu", half=False)
    agreement = cached("parity", candidate=True) or parity(
        reference, config, "data/traffic.mp4", root / f"parity-{spec['name']}.json"
    )
    clean()
    print("Inspecting actual execution placement and precision", flush=True)
    inspect_artifact(config, root / f"inspection-{spec['name']}.json")
    clean()
    evidence = {
        "kind": "full_matrix_case",
        "spec": spec,
        "throughput_fps": timing["aggregate_fps"],
        "mean_trial_fps": timing["fps"]["mean"],
        "p95_ms": timing["batch_wall_ms"]["p95"],
        "p99_ms": timing["batch_wall_ms"]["p99"],
        "map50_95": quality["map50_95"],
        "map50": quality["map50"],
        "strict_detection_parity_passed": agreement["passed"],
        "note": "Strict detection parity is diagnostic for quantized models; evaluate mAP loss separately.",
    }
    save_json(root / f"{spec['name']}.json", evidence)
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
