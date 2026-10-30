"""Compare CPU and GPU serving capacity, then measure tracker, stride, and batching choices."""

import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from traffic_lab.benchmark import evaluate
from traffic_lab.pipeline import run_pipeline
from traffic_lab.runtime import ModelConfig, environment, file_hash, save_json

ROOT = Path("results/full-matrix")


def clean():
    import torch

    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def main():
    current_environment = environment()
    video_hash = file_hash("data/traffic.mp4")
    baseline = json.loads((ROOT / "pytorch-cpu-fp32-320.json").read_text())["map50_95"]

    def vehicle_map(quality):
        return sum(quality["per_class_map50_95"][str(c)] for c in (2, 3, 5, 7)) / 4

    baseline_vehicle = vehicle_map(json.loads((ROOT / "quality-pytorch-cpu-fp32-320.json").read_text()))
    selections = []
    decisions = []
    for prefix in ("openvino-cpu", "onnx-cpu", "onnx-cuda", "tensorrt-cuda"):
        cases = [json.loads(p.read_text()) for p in ROOT.glob(f"{prefix}-*-320.json")]
        eligible = []
        for case in cases:
            quality = json.loads((ROOT / f"quality-{case['spec']['name']}.json").read_text())
            global_drop = 100 * (baseline - case["map50_95"])
            vehicle_drop = 100 * (baseline_vehicle - vehicle_map(quality))
            accepted = global_drop <= 1 and vehicle_drop <= 1
            decisions.append(
                {
                    "name": case["spec"]["name"],
                    "global_drop_points": global_drop,
                    "vehicle_drop_points": vehicle_drop,
                    "eligible": accepted,
                }
            )
            if accepted:
                eligible.append(case)
        if not eligible:
            raise RuntimeError(f"No {prefix} candidate satisfies global and vehicle 1-point accuracy limits")
        selections.append(min(eligible, key=lambda c: c["p95_ms"])["spec"])
    save_json(
        ROOT / "pipeline-selections.json",
        {
            "baseline_map50_95": baseline,
            "baseline_vehicle_map50_95": baseline_vehicle,
            "max_map_drop_points": 1,
            "max_vehicle_drop_points": 1,
            "candidate_decisions": decisions,
            "selections": selections,
        },
    )
    ledger_path = ROOT / "pipeline-ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}

    def run(
        spec, streams, batch=1, stride=1, tracker="bytetrack", render=False, suffix="", brightness_grid=0
    ):
        name = f"{spec['name']}-streams-{streams}-batch-{batch}-stride-{stride}-{tracker}{suffix}"
        output = ROOT / "pipelines" / name
        if ledger.get(name, {}).get("status") == "complete" and (output / "summary.json").exists():
            previous = json.loads((output / "summary.json").read_text())
            if (
                previous.get("status") == "complete"
                and previous.get("schema_version") == 2
                and previous.get("model_sha256") == file_hash(spec["model"])
                and previous.get("input_hashes") == [video_hash] * streams
                and previous["environment"]["packages"] == current_environment["packages"]
                and previous["environment"]["gpu"] == current_environment["gpu"]
                and previous["environment"]["processor"] == current_environment["processor"]
                and previous.get("requested_seconds") == 5
                and previous["config"]["imgsz"] == 320
                and previous.get("requested_batch") == batch
                and previous.get("stride") == stride
                and previous.get("tracker") == tracker
                and previous.get("render") == render
                and previous.get("brightness_grid", 0) == brightness_grid
                and previous.get("requested_decoder_threads") == 1
                and previous.get("warmup", {}).get("real_image_batch_sizes")
                == list(range(1, min(batch, streams) + 1))
            ):
                print(f"{name}: cached matching model, input and configuration", flush=True)
                return
        start = time.perf_counter()
        print(f"Running {name}", flush=True)
        try:
            config = ModelConfig(model=spec["model"], imgsz=320, device=spec["device"], half=spec["half"])
            summary = run_pipeline(
                config,
                ["data/traffic.mp4"] * streams,
                output,
                duration=5,
                batch=batch,
                stride=stride,
                tracker=tracker,
                render=render,
                decode_threads=1,
                brightness_grid=brightness_grid,
            )
            ledger[name] = {
                "status": "complete",
                "seconds": time.perf_counter() - start,
                "processed": sum(c["processed"] for c in summary["cameras"]),
                "decoded": sum(c["decoded"] for c in summary["cameras"]),
                "output": str(output),
                "model": spec["model"],
                "streams": streams,
                "batch": batch,
                "stride": stride,
                "tracker": tracker,
                "render": render,
                "brightness_grid": brightness_grid,
            }
        except Exception as exc:
            traceback.print_exc()
            ledger[name] = {"status": "failed", "error": str(exc), "seconds": time.perf_counter() - start}
        save_json(ledger_path, ledger)
        clean()

    for spec in selections:
        for streams in (1, 2, 4, 8):
            run(spec, streams)
    trt = next(s for s in selections if s["backend"] == "tensorrt")
    run(trt, 1, tracker="botsort")
    run(trt, 1, render=True, suffix="-render")
    run(trt, 4, stride=2)
    for streams in (1, 4, 8):
        run(trt, streams, brightness_grid=64, suffix="-sampled-telemetry")
    dynamic_dir = Path("models/full-matrix/tensorrt-dynamic-batch4")
    dynamic_model = dynamic_dir / "yolo11n.engine"
    manifest_path = dynamic_dir / "export.json"
    reusable_dynamic = False
    if dynamic_model.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        reusable_dynamic = (
            manifest["arguments"]["quantize"] == trt["precision"]
            and manifest["arguments"]["batch"] == 4
            and manifest["arguments"]["dynamic"]
            and manifest["source_sha256"] == file_hash("models/yolo11n.pt")
            and manifest["artifact_sha256"] == file_hash(dynamic_model)
            and manifest["environment"]["packages"] == current_environment["packages"]
            and manifest["environment"]["gpu"] == current_environment["gpu"]
        )
        if trt["precision"] == 8:
            reusable_dynamic = (
                reusable_dynamic
                and manifest.get("calibration_yaml_sha256")
                == file_hash("data/coco-deployment/calibration.yaml")
                and manifest.get("calibration_manifest_sha256") == file_hash(ROOT / "dataset-manifest.json")
            )
    if not reusable_dynamic:
        arguments = [
            sys.executable,
            "-m",
            "traffic_lab.cli",
            "export",
            "--format",
            "engine",
            "--device",
            "0",
            "--imgsz",
            "320",
            "--batch",
            "4",
            "--dynamic",
            "--quantize",
            str(trt["precision"]),
            "--workspace",
            "0.5",
            "--output-dir",
            str(dynamic_dir),
        ]
        if trt["precision"] == 8:
            arguments += ["--data", "data/coco-deployment/calibration.yaml"]
        with (ROOT / "logs/export-tensorrt-dynamic-batch4.log").open("w") as log:
            subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=True)
    dynamic = {**trt, "name": "tensorrt-cuda-dynamic-320", "model": str(dynamic_model)}
    quality_path = ROOT / "quality-tensorrt-dynamic-batch4.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text())
    else:
        quality = {}
    if (
        quality.get("model_sha256") != file_hash(dynamic_model)
        or quality.get("dataset_yaml_sha256") != file_hash("data/coco-deployment/holdout.yaml")
        or quality.get("dataset_manifest_sha256") != file_hash(ROOT / "dataset-manifest.json")
        or quality.get("environment", {}).get("packages") != current_environment["packages"]
    ):
        quality = evaluate(
            ModelConfig(model=str(dynamic_model), imgsz=320, device="0"),
            "data/coco-deployment/holdout.yaml",
            quality_path,
        )
    clean()
    if 100 * (baseline - quality["map50_95"]) > 1 or 100 * (baseline_vehicle - vehicle_map(quality)) > 1:
        raise RuntimeError("Dynamic batching engine failed accuracy budget; do not adopt it")
    for streams in (4, 8):
        run(dynamic, streams, batch=4)
    failed = [k for k, v in ledger.items() if v["status"] != "complete"]
    print(json.dumps({"pipeline_runs": len(ledger), "failed": failed}, indent=2))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
