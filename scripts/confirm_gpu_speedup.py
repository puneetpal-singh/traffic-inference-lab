"""Confirm a same-GPU optimization with balanced randomized order and real-image warm-up."""

import argparse
import json
import time
from pathlib import Path

from traffic_lab.confirmation import balanced_orders, json_digest, paired_speedup, percentile
from traffic_lab.resources import ResourceMonitor
from traffic_lab.runtime import Detector, ModelConfig, environment, file_hash, save_json, video_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/confirmation"))
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.seconds < 10:
        raise ValueError("Use at least ten seconds per trial")
    orders = balanced_orders(args.pairs, args.seed)
    if (args.output / "plan.json").exists():
        raise FileExistsError("Choose a new --output directory to preserve the existing experiment")
    configs = {
        "pytorch": ModelConfig(model="models/yolo11n.pt", imgsz=320, device="0"),
        "tensorrt": ModelConfig(
            model="models/full-matrix/tensorrt-fp16-320/yolo11n.engine", imgsz=320, device="0"
        ),
    }
    quality = {}
    for runtime, case in (("pytorch", "pytorch-cuda-fp32-320"), ("tensorrt", "tensorrt-cuda-fp16-320")):
        path = Path(f"results/full-matrix/quality-{case}.json")
        record = json.loads(path.read_text())
        if record["model_sha256"] != file_hash(configs[runtime].model):
            raise RuntimeError("The evaluated model changed; evaluate the current artifact first")
        quality[runtime] = {
            "path": path.as_posix(),
            "json_sha256": json_digest(record),
            "model_sha256": record["model_sha256"],
        }
    # 60 every-other frames span the same first 120 source frames with a smaller host-memory footprint.
    images = video_frames("data/traffic.mp4", limit=60, step=2)
    if len(images) != 60:
        raise RuntimeError("Expected the complete declared 60-frame pool")
    detectors, execution = {}, {}
    for name, config in configs.items():
        detector = Detector(config)
        detector.warmup()
        result = detector.predict([images[0]])[0]
        if not result.boxes.data.is_cuda:
            raise RuntimeError("Expected CUDA output for both resident models")
        execution[name] = {**detector.metadata(), "output_boxes_device": str(result.boxes.data.device)}
        detectors[name] = detector
    plan = {
        "kind": "same_gpu_confirmation_plan",
        "schema_version": 1,
        "environment": environment(),
        "orders": orders,
        "seed": args.seed,
        "minimum_trial_seconds": args.seconds,
        "pool_frames": len(images),
        "source_indices": list(range(0, 120, 2)),
        "source_sha256": file_hash("data/traffic.mp4"),
        "runner_sha256": file_hash(__file__),
        "quality": quality,
        "execution": execution,
        "warmup": "One complete real-image pool before every trial, outside measurement",
        "endpoint": "Batch-one predict through completed CPU box transfer; decode/tracking/rendering excluded",
        "design": "Both models resident; sequential trials, balanced AB/BA order randomized by seed; full pool cycles only",
    }
    save_json(args.output / "plan.json", plan)
    trials, references = [], []
    for pair, order in enumerate(orders):
        for position, name in enumerate(order):
            detector = detectors[name]
            for frame in images:
                detector.predict([frame])[0].boxes.cpu().numpy()
            samples = []
            with ResourceMonitor() as monitor:
                start = time.perf_counter()
                cycles = 0
                while time.perf_counter() - start < args.seconds:
                    for index, frame in enumerate(images):
                        tick = time.perf_counter()
                        boxes = detector.predict([frame])[0].boxes.cpu().numpy()
                        elapsed = (time.perf_counter() - tick) * 1000
                        samples.append(
                            {"source_index": index * 2, "wall_ms": elapsed, "detections": len(boxes)}
                        )
                    cycles += 1
                seconds = time.perf_counter() - start
            trial = {
                "pair": pair,
                "position": position,
                "runtime": name,
                "seconds": seconds,
                "frames": len(samples),
                "cycles": cycles,
                "fps": len(samples) / seconds,
                "p50_ms": percentile([s["wall_ms"] for s in samples], 0.5),
                "p95_ms": percentile([s["wall_ms"] for s in samples], 0.95),
                "model_sha256": execution[name]["model_sha256"],
                "samples": samples,
                "resources": monitor.summary(),
            }
            path = f"pair-{pair:02d}-{name}.json"
            save_json(args.output / path, trial)
            references.append({"path": path, "json_sha256": json_digest(trial)})
            trials.append(trial)
            print(
                f"Pair {pair + 1}/{args.pairs}, {name}: {trial['fps']:.2f} FPS, p95 {trial['p95_ms']:.2f} ms",
                flush=True,
            )
    summary = {
        "kind": "same_gpu_confirmation",
        "plan_json_sha256": json_digest(plan),
        "trials": references,
        **paired_speedup(trials, args.seed),
        "aggregate_fps": {
            name: sum(t["frames"] for t in trials if t["runtime"] == name)
            / sum(t["seconds"] for t in trials if t["runtime"] == name)
            for name in configs
        },
        "limitations": [
            "One shared laptop and a short cyclic video pool, not diverse or sustained production input.",
            "This supplements the 28-case matrix; its warm-up, frame pool and residency protocol differ.",
            "The interval resamples six paired trials, not thousands of correlated frames.",
            "Held-out quality is reused only for the exact previously evaluated model hashes.",
        ],
    }
    save_json(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("trials",)}, indent=2))


if __name__ == "__main__":
    main()
