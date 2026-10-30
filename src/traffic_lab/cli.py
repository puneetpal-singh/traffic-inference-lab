from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from .runtime import ModelConfig, environment, file_hash, save_json


def model_args(parser):
    parser.add_argument("--model", default="models/yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)


def config(args):
    return ModelConfig(
        **{k: getattr(args, k) for k in ("model", "imgsz", "device", "half", "threads", "conf", "iou")}
    )


def parser():
    p = argparse.ArgumentParser(description="Traffic Inference Lab — benchmark, verify, deploy")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Print hardware and installed runtime metadata")
    fetch = sub.add_parser("fetch", help="Download the official checkpoint and replay video")
    fetch.add_argument("--model", default="yolo11n.pt")
    export = sub.add_parser("export")
    model_args(export)
    export.add_argument("--format", choices=["onnx", "openvino", "engine"], required=True)
    export.add_argument("--int8", action="store_true")
    export.add_argument("--data", help="Calibration dataset YAML, disjoint from evaluation")
    export.add_argument("--batch", type=int, default=1)
    export.add_argument("--dynamic", action="store_true")
    export.add_argument("--workspace", type=float, default=0.5, help="TensorRT builder workspace in GiB")
    export.add_argument("--quantize", type=int, choices=[8, 16, 32], help="Explicit export precision")
    export.add_argument("--output-dir", default="models/export")
    bench = sub.add_parser("benchmark")
    model_args(bench)
    bench.add_argument("--source", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--frames", type=int, default=60)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--batch", type=int, default=1)
    par = sub.add_parser("parity")
    model_args(par)
    par.add_argument("--candidate", required=True)
    par.add_argument("--source", required=True)
    par.add_argument("--output", required=True)
    par.add_argument("--frames", type=int, default=30)
    par.add_argument("--min-iou", type=float, default=0.9)
    par.add_argument("--max-conf-delta", type=float, default=0.02)
    val = sub.add_parser("evaluate")
    model_args(val)
    val.add_argument("--data", required=True)
    val.add_argument("--split", choices=["val", "test", "train"], default="val")
    val.add_argument("--output", required=True)
    run = sub.add_parser("run")
    model_args(run)
    run.add_argument("--sources", nargs="+", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--duration", type=float, default=10)
    run.add_argument("--queue-size", type=int, default=1)
    run.add_argument("--batch", type=int, default=1)
    run.add_argument("--stride", type=int, default=1)
    run.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    run.add_argument("--line", nargs=4, type=float, default=[0.1, 0.55, 0.9, 0.55])
    run.add_argument("--render", action="store_true")
    run.add_argument("--zone", nargs=4, type=float, default=[0.2, 0.3, 0.8, 0.8])
    run.add_argument("--decode-threads", type=int, default=1)
    report = sub.add_parser("report")
    report.add_argument("--input", default="results")
    report.add_argument("--output", default="results/report.html")
    shift = sub.add_parser("drift")
    shift.add_argument("--reference", required=True, help="Pipeline frames.jsonl")
    shift.add_argument("--current", required=True)
    shift.add_argument("--output", required=True)
    counts = sub.add_parser("count-score")
    counts.add_argument("--predictions", required=True, help="Pipeline events.json")
    counts.add_argument("--labels", required=True, help="Independent annotated crossing events")
    counts.add_argument("--tolerance", type=float, default=0.5)
    counts.add_argument("--output", required=True)
    gate = sub.add_parser("gate", help="Check measured model latency and labeled accuracy budgets")
    gate.add_argument("--benchmark", required=True)
    gate.add_argument("--evaluation", required=True)
    gate.add_argument("--baseline-evaluation", required=True)
    gate.add_argument("--p95-budget-ms", type=float, default=100)
    gate.add_argument("--max-map-drop-points", type=float, default=1)
    gate.add_argument("--output", required=True)
    return p


def fetch(args):
    from urllib.request import urlretrieve

    from ultralytics import YOLO

    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    name = Path(args.model).name
    checkpoint = Path("models") / name
    if not checkpoint.exists():
        YOLO(str(checkpoint))
    video = Path("data/traffic.mp4")
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/anpr-demo-video.mp4"
    if not video.exists():
        temp = video.with_suffix(".download")
        urlretrieve(url, temp)
        temp.replace(video)
    manifest = {
        "model": str(checkpoint),
        "model_sha256": file_hash(checkpoint),
        "video": str(video),
        "video_sha256": file_hash(video),
        "video_url": url,
        "note": "Upstream demo footage; not redistributed here. See docs/DATA.md.",
    }
    save_json("data/manifest.json", manifest)
    return manifest


def export(args):
    from ultralytics import YOLO

    precision = args.quantize or (8 if args.int8 else 16 if args.half else 32)
    if precision == 8 and not args.data:
        raise ValueError("INT8 requires --data with representative calibration data")
    if args.half and args.int8:
        raise ValueError("Choose half or INT8, not both")
    if args.format == "engine" and args.device == "cpu":
        raise ValueError("TensorRT export requires a CUDA device, e.g. --device 0")
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    staged = destination / Path(args.model).name
    if staged.resolve() != Path(args.model).resolve():
        shutil.copy2(args.model, staged)
    model = YOLO(str(staged))
    kwargs = dict(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
        quantize=precision,
        batch=args.batch,
        dynamic=args.dynamic,
    )
    if args.data:
        kwargs["data"] = args.data
    if args.format == "onnx":
        kwargs.update(opset=17, simplify=True)
    if args.format == "engine":
        kwargs["workspace"] = args.workspace
    path = model.export(**kwargs)
    manifest = {
        "kind": "export",
        "source_sha256": file_hash(args.model),
        "artifact": str(path),
        "artifact_sha256": file_hash(path),
        "arguments": kwargs,
        "calibration_yaml_sha256": file_hash(args.data) if args.data and Path(args.data).is_file() else None,
        "calibration_manifest_sha256": file_hash("results/full-matrix/dataset-manifest.json")
        if args.data
        and Path(args.data).resolve() == Path("data/coco-deployment/calibration.yaml").resolve()
        and Path("results/full-matrix/dataset-manifest.json").exists()
        else None,
        "environment": environment(),
    }
    # Keep manifest outside OpenVINO directory so its hash does not change after saving it.
    save_json(destination / "export.json", manifest)
    return manifest


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = environment()
        elif args.command == "fetch":
            result = fetch(args)
        elif args.command == "export":
            result = export(args)
        elif args.command == "benchmark":
            from .benchmark import benchmark

            result = benchmark(
                config(args), args.source, args.output, args.frames, args.repeats, args.warmup, args.batch
            )
            result = {k: v for k, v in result.items() if k != "samples"}
        elif args.command == "parity":
            from .benchmark import parity

            candidate = ModelConfig(**{**asdict(config(args)), "model": args.candidate})
            result = parity(
                config(args),
                candidate,
                args.source,
                args.output,
                args.frames,
                args.min_iou,
                args.max_conf_delta,
            )
            print(json.dumps({"passed": result["passed"], "output": args.output}, indent=2))
            return 0 if result["passed"] else 2
        elif args.command == "evaluate":
            from .benchmark import evaluate

            result = evaluate(config(args), args.data, args.output, args.split)
        elif args.command == "run":
            from .pipeline import run_pipeline

            result = run_pipeline(
                config(args),
                args.sources,
                args.output,
                args.duration,
                args.queue_size,
                args.batch,
                args.stride,
                args.tracker,
                args.line,
                args.render,
                args.zone,
                args.decode_threads,
            )
        elif args.command == "report":
            matrix = Path(args.input)
            if not (matrix / "plan.json").exists():
                matrix = matrix / "full-matrix"
            if (matrix / "plan.json").exists():
                from .matrix_report import build_matrix_report

                summary = build_matrix_report(matrix, args.output)
                result = {k: v for k, v in summary.items() if k not in ("rows", "pipelines")}
            else:
                from .report import build_report

                result = build_report(args.input, args.output)
        elif args.command == "drift":
            from .metrics import drift

            def confidences(path):
                rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
                cameras = {r["camera"] for r in rows}
                if len(cameras) != 1:
                    raise ValueError(
                        "Drift inputs must each contain one camera to avoid pooling scene changes"
                    )
                return [c for r in rows[::10] for c in r["confidences"]]

            result = drift(confidences(args.reference), confidences(args.current))
            save_json(args.output, result)
        elif args.command == "gate":
            from .gate import deployment_gate

            def read(path):
                return json.loads(Path(path).read_text(encoding="utf-8"))

            result = deployment_gate(
                read(args.benchmark),
                read(args.evaluation),
                read(args.baseline_evaluation),
                args.p95_budget_ms,
                args.max_map_drop_points,
            )
            save_json(args.output, result)
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 2
        elif args.command == "count-score":
            from .scoring import score_events

            result = score_events(
                json.loads(Path(args.labels).read_text()),
                json.loads(Path(args.predictions).read_text()),
                args.tolerance,
            )
            save_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
