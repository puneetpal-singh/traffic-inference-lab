"""One isolated process per case, with identical data and preprocessing across all backends."""

import argparse
import gc
import json
from pathlib import Path

from traffic_lab.benchmark import benchmark, evaluate, parity
from traffic_lab.inspection import inspect_artifact
from traffic_lab.runtime import ModelConfig, save_json


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
    args = parser.parse_args()
    spec = json.loads(args.case)
    root = Path("results/full-matrix")
    config = ModelConfig(model=spec["model"], imgsz=spec["imgsz"], device=spec["device"], half=spec["half"])
    print(f"Timing {spec['name']}: {args.frames} frames x {args.repeats} trials", flush=True)
    timing = benchmark(
        config,
        "data/traffic.mp4",
        root / f"timing-{spec['name']}.json",
        frames=args.frames,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    clean()
    print("Timing complete; evaluating 256 held-out images", flush=True)
    quality = evaluate(config, "data/coco-deployment/holdout.yaml", root / f"quality-{spec['name']}.json")
    clean()
    print("Accuracy evaluation complete; checking detection agreement", flush=True)
    reference = ModelConfig(imgsz=spec["imgsz"], device="cpu", half=False)
    agreement = parity(reference, config, "data/traffic.mp4", root / f"parity-{spec['name']}.json")
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
