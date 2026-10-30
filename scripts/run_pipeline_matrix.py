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
from traffic_lab.runtime import ModelConfig, save_json

ROOT = Path("results/full-matrix")


def clean():
    import torch

    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def main():
    baseline = json.loads((ROOT / "pytorch-cpu-fp32-320.json").read_text())["map50_95"]
    selections = []
    for prefix in ("openvino-cpu", "onnx-cuda", "tensorrt-cuda"):
        cases = [json.loads(p.read_text()) for p in ROOT.glob(f"{prefix}-*-320.json")]
        eligible = [c for c in cases if 100 * (baseline - c["map50_95"]) <= 1]
        if not eligible:
            raise RuntimeError(f"No {prefix} candidate satisfies the 1-point accuracy-loss limit")
        selections.append(min(eligible, key=lambda c: c["p95_ms"])["spec"])
    save_json(
        ROOT / "pipeline-selections.json",
        {"baseline_map50_95": baseline, "max_map_drop_points": 1, "selections": selections},
    )
    ledger_path = ROOT / "pipeline-ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}

    def run(spec, streams, batch=1, stride=1, tracker="bytetrack", render=False, suffix=""):
        name = f"{spec['name']}-streams-{streams}-batch-{batch}-stride-{stride}-{tracker}{suffix}"
        output = ROOT / "pipelines" / name
        if ledger.get(name, {}).get("status") == "complete" and (output / "summary.json").exists():
            print(f"{name}: cached", flush=True)
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
    dynamic_dir = Path("models/full-matrix/tensorrt-dynamic-batch4")
    dynamic_model = dynamic_dir / "yolo11n.engine"
    if not dynamic_model.exists():
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
        quality = evaluate(
            ModelConfig(model=str(dynamic_model), imgsz=320, device="0"),
            "data/coco-deployment/holdout.yaml",
            quality_path,
        )
    clean()
    if 100 * (baseline - quality["map50_95"]) > 1:
        raise RuntimeError("Dynamic batching engine failed accuracy budget; do not adopt it")
    for streams in (4, 8):
        run(dynamic, streams, batch=4)
    failed = [k for k, v in ledger.items() if v["status"] != "complete"]
    print(json.dumps({"pipeline_runs": len(ledger), "failed": failed}, indent=2))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
