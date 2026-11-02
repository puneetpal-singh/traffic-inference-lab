"""One YOLO checkpoint, three export runtimes, FP32/FP16/INT8. Default action is a dry run."""

import argparse
import contextlib
import csv
import ctypes
import functools
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from traffic_lab.confirmation import json_digest, percentile
from traffic_lab.precision import (
    cases,
    evidence_identity,
    paired_ratio,
    randomized_rounds,
    split_images,
    spread,
    validate_timing,
)
from traffic_lab.progress import ledger_progress, log_event, wait_with_progress, worker_environment
from traffic_lab.provenance import file_hash

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Print the plan; no model loading or downloads (default)")
    action.add_argument("--prepare-data", action="store_true", help="Download/verify COCO and create disjoint splits")
    action.add_argument("--run", action="store_true", help="Explicitly start exports, timing and accuracy evaluation")
    action.add_argument("--report-only", action="store_true", help="Regenerate tables from saved evidence")
    p.add_argument("--output", type=Path, default=Path("results/precision-study"))
    p.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"))
    p.add_argument("--source", type=Path, default=Path("data/traffic.mp4"))
    p.add_argument("--data-root", type=Path, default=Path("data/precision-study"))
    p.add_argument("--target", choices=("all", "cpu", "cuda"), default="all")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--pool-frames", type=int, default=60)
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--timing-classes", choices=("all", "vehicles"), default="all")
    p.add_argument("--calibration-images", type=int, default=512)
    p.add_argument("--evaluation-limit", type=int, default=0, help="0=all remaining 4488; small subsets are exploratory")
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--quality-budget-points", type=float, default=1.)
    p.add_argument("--stage-timeout", type=int, default=7200)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


def config(args):
    if (args.imgsz < 32 or args.imgsz % 32 or min(args.threads, args.pool_frames, args.frame_step) < 1
            or args.seconds < 20 or args.trials < 2 or not 0 < args.calibration_images < 5000
            or args.evaluation_limit < 0 or not math.isfinite(args.seconds)
            or not math.isfinite(args.quality_budget_points) or args.quality_budget_points < 0
            or args.stage_timeout < 60):
        raise ValueError("Use valid image size, threads, splits, >=2 trials, >=20s/trial and >=60s timeout")
    return {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
            if k not in ("dry_run", "prepare_data", "run", "report_only", "worker", "stage_timeout")}


def data_recipe(cfg):
    return {k: cfg[k] for k in ("seed", "calibration_images", "evaluation_limit")}


def prepare_data(cfg):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from prepare_coco_holdout import download

    root = Path(cfg["data_root"])
    existing = root / "manifest.json"
    if existing.exists() and read(existing)["recipe"] != data_recipe(cfg):
        raise ValueError("This data root has a different split; choose a fresh --data-root")
    annotation = root / "instances_val2017.json"
    annotation.parent.mkdir(parents=True, exist_ok=True)
    if not annotation.exists():
        previous = Path("data/coco-deployment/instances_val2017.json")
        if previous.exists():
            shutil.copy2(previous, annotation)
        else:
            archive = Path(".build-cache/annotations_trainval2017.zip")
            download("https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip", archive)
            with zipfile.ZipFile(archive) as source:
                annotation.write_bytes(source.read("annotations/instances_val2017.json"))
    annotations = read(annotation)
    calibration, evaluation = split_images(annotations["images"], cfg["calibration_images"],
                                            cfg["evaluation_limit"], cfg["seed"])
    categories = sorted(annotations["categories"], key=lambda row: row["id"])
    category_map = {row["id"]: i for i, row in enumerate(categories)}
    labels = {}
    for box in annotations["annotations"]:
        if not box.get("iscrowd"):
            labels.setdefault(box["image_id"], []).append(box)
    tasks, rows = [], []
    for split, images in (("calibration", calibration), ("evaluation", evaluation)):
        for image in images:
            path = root / "images" / split / image["file_name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                candidates = [Path("data/coco-deployment/images") / old / image["file_name"]
                              for old in ("calibration", "holdout")]
                candidates.append(Path("data/coco-full/images") / image["file_name"])
                for previous in candidates:
                    if previous.exists():
                        try:
                            os.link(previous, path)
                        except OSError:
                            shutil.copy2(previous, path)
                        break
            tasks.append(("https://s3.amazonaws.com/images.cocodataset.org/val2017/" + image["file_name"], path))
            rows.append({"id": image["id"], "split": split, "path": str(path.resolve())})
            text = []
            for box in labels.get(image["id"], []):
                x, y, w, h = box["bbox"]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(image["width"], x+w), min(image["height"], y+h)
                if x2 > x1 and y2 > y1:
                    text.append(f"{category_map[box['category_id']]} {(x1+x2)/2/image['width']:.8f} "
                                f"{(y1+y2)/2/image['height']:.8f} {(x2-x1)/image['width']:.8f} "
                                f"{(y2-y1)/image['height']:.8f}")
            target = root / "labels" / split / Path(image["file_name"]).with_suffix(".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(text) + "\n", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, url, path) for url, path in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 250 == 0:
                print(f"Downloaded/verified {index}/{len(tasks)}", flush=True)
    for row in rows:
        row["sha256"] = file_hash(row["path"])
    save(root / "calibration.yaml", {"path": str(root.resolve()), "train": "images/calibration",
         "val": "images/calibration", "names": {i: c["name"] for i, c in enumerate(categories)}})
    manifest = {"recipe": data_recipe(cfg), "annotation_sha256": file_hash(annotation),
                "calibration_yaml_sha256": file_hash(root / "calibration.yaml"), "images": rows,
                "category_ids": [c["id"] for c in categories], "source": "Official COCO val2017",
                "note": "No training. Calibration/evaluation image IDs are disjoint; primary AP excludes calibration."}
    if existing.exists() and json_digest(read(existing)) != json_digest(manifest):
        raise ValueError("Dataset content changed; preserve the old study and choose a fresh data root")
    save(existing, manifest)
    print(f"Ready: {len(calibration)} calibration / {len(evaluation)} evaluation images; no overlap.")


def model_path(cfg, case):
    if case["backend"] == "pytorch":
        return cfg["model"]
    manifest = read(Path(cfg["output"]) / "exports" / case["export_key"] / "export.json")
    path = Path(manifest["artifact"])
    if file_hash(path) != manifest["artifact_sha256"] or manifest["source_sha256"] != file_hash(cfg["model"]):
        raise ValueError("Export or base checkpoint changed")
    return str(path)


def export_directory(cfg, case):
    identity = json_digest({k: cfg[k] for k in ("output", "model", "imgsz")})[:12]
    return Path("models/precision-study") / identity / case["export_key"]


def detector(cfg, case, quality=False):
    import cv2
    import torch

    from traffic_lab.runtime import Detector, ModelConfig

    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(cfg["threads"])
    cv2.setNumThreads(1)
    if case["backend"] == "onnx":
        import onnxruntime as ort
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.nn.backends.onnx import ONNXBackend

        options = ort.SessionOptions()
        options.intra_op_num_threads = cfg["threads"]
        options.inter_op_num_threads = 1
        AutoBackend._BACKEND_MAP["onnx"] = functools.partial(ONNXBackend, session_options=options)
    classes = (2, 3, 5, 7) if not quality and cfg.get("timing_classes") == "vehicles" else None
    obj = Detector(ModelConfig(model=model_path(cfg, case), imgsz=cfg["imgsz"], device=case["device"],
                               threads=cfg["threads"], classes=classes, conf=.001 if quality else .25))
    obj.warmup(1)
    if case["backend"] == "openvino":
        import openvino as ov

        path = Path(obj.config.model)
        xml = next(path.glob("*.xml")) if path.is_dir() else path
        core = ov.Core()
        backend = obj.model.predictor.model.backend
        backend.ov_compiled_model = core.compile_model(core.read_model(xml), "CPU",
            {"INFERENCE_NUM_THREADS": cfg["threads"], "NUM_STREAMS": 1,
             "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": ov.Type.f32})
    if case["device"] != "cpu":
        import numpy as np

        result = obj.predict([np.zeros((cfg["imgsz"], cfg["imgsz"], 3), dtype=np.uint8)])[0]
        if not result.boxes.data.is_cuda:
            raise RuntimeError("CUDA requested, but result boxes are on CPU")
    return obj


def timed_trial(cfg, case, round_index):
    from traffic_lab.resources import ResourceMonitor
    from traffic_lab.runtime import video_frames

    obj = detector(cfg, case)
    frames = video_frames(cfg["source"], cfg["pool_frames"], cfg["frame_step"])
    if len(frames) != cfg["pool_frames"]:
        raise ValueError("Source is shorter than the declared timing pool")
    for frame in frames:
        obj.predict([frame])[0].boxes.cpu().numpy()
    samples, cycles, clock_events = [], 0, []
    with ResourceMonitor() as monitor:
        started = time.perf_counter()
        while time.perf_counter() - started < cfg["seconds"]:
            for index, frame in enumerate(frames):
                tick = time.perf_counter()
                result = obj.predict([frame])[0]
                count = len(result.boxes.cpu().numpy())
                elapsed = (time.perf_counter() - tick) * 1000
                samples.append({"source_index": index * cfg["frame_step"], "wall_ms": elapsed,
                                "detections": count, "backend_stages_ms": dict(result.speed)})
            cycles += 1
            if monitor.handle is not None:
                try:
                    clock_events.append(int(monitor.nvml.nvmlDeviceGetCurrentClocksThrottleReasons(monitor.handle)))
                except Exception:
                    pass
        seconds = time.perf_counter() - started
    resources = monitor.summary()
    flags = []
    if any(mask & (32 | 64) for mask in clock_events):
        flags.append("NVML thermal throttle bit observed at a pool boundary")
    if os.name == "nt" and not on_ac_power():
        flags.append("AC power absent or unknown at trial end")
    temperatures = [r["gpu_temperature_c"] for r in resources["samples"] if r.get("gpu_temperature_c") is not None]
    if temperatures and max(temperatures) >= 85:
        flags.append("GPU temperature reached 85C; diagnostic, not proof of throttling")
    if any(r["system_available_ram_mib"] < 512 for r in resources["samples"]):
        flags.append("Available system RAM fell below 512 MiB")
    result = {"kind": "precision_timing_trial", "case": case, "round": round_index,
            "model_sha256": file_hash(obj.config.model), "execution": obj.metadata(), "seconds": seconds,
            "frames": len(samples), "cycles": cycles, "fps": len(samples) / seconds,
            "p50_ms": percentile([r["wall_ms"] for r in samples], .5),
            "p95_ms": percentile([r["wall_ms"] for r in samples], .95),
            "samples": samples, "resources": resources, "flags": flags,
            "clock_event_masks_at_pool_boundaries": clock_events,
            "endpoint": "Predecoded full-resolution frame to completed CPU boxes; decode/tracking excluded",
            "stage_timer_caution": "Backend stage timers are diagnostic, not a common GPU-only CUDA-event measurement"}
    validate_timing(result, cfg)
    return result


def quality_evaluation(cfg, case):
    import cv2
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    root = Path(cfg["data_root"])
    manifest = read(root / "manifest.json")
    images = sorted((r for r in manifest["images"] if r["split"] == "evaluation"), key=lambda r: r["id"])
    obj, predictions = detector(cfg, case, quality=True), []
    for index, row in enumerate(images, 1):
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Evaluation image changed")
        image = cv2.imread(row["path"])
        if image is None:
            raise ValueError(f"Cannot decode {row['path']}")
        for x1, y1, x2, y2, score, cls in obj.predict([image])[0].boxes.cpu().numpy().data:
            predictions.append({"image_id": row["id"], "category_id": manifest["category_ids"][int(cls)],
                                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)], "score": float(score)})
        if index % 250 == 0:
            print(f"{case['name']}: {index}/{len(images)} evaluation images", flush=True)
    predictions_path = Path(cfg["data_root"]) / "predictions" / Path(cfg["output"]).name / f"{case['name']}.json"
    save(predictions_path, predictions)
    coco = COCO(str(root / "instances_val2017.json"))
    if predictions:
        detected = coco.loadRes(str(predictions_path))
    else:
        detected = COCO()
        detected.dataset = {"images": coco.dataset["images"], "categories": coco.dataset["categories"],
                            "annotations": []}
        detected.createIndex()
    evaluator = COCOeval(coco, detected, "bbox")
    evaluator.params.imgIds = [r["id"] for r in images]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    aps = {}
    for position, category in enumerate(evaluator.params.catIds):
        values = evaluator.eval["precision"][:, :, position, 0, -1]
        valid = values[values >= 0]
        aps[str(category)] = float(valid.mean()) if valid.size else None
    vehicle = [aps[str(c)] for c in (3, 4, 6, 8) if aps.get(str(c)) is not None]
    return {"kind": "precision_accuracy", "case": case, "model_sha256": file_hash(obj.config.model),
            "images": len(images), "map50_95": float(evaluator.stats[0]), "map50": float(evaluator.stats[1]),
            "vehicle_ap": float(np.mean(vehicle)) if len(vehicle) == 4 else None, "per_category_ap": aps,
            "prediction_sha256": file_hash(predictions_path),
            "evaluator": "Official COCOeval bbox: crowd handling, IoU .50:.95, area all, maxDets100",
            "note": "Calibration-disjoint evaluation. AP point estimates only; near-budget differences are inconclusive."}


def environment_snapshot():
    from traffic_lab.runtime import environment

    snapshot = environment()
    snapshot.pop("utc", None)
    if os.name == "nt":
        snapshot["power_scheme"] = subprocess.check_output(["powercfg", "/getactivescheme"], text=True).strip()
    return snapshot


def check_dependencies(cfg):
    modules = ["torch", "ultralytics", "onnx", "onnxruntime", "pycocotools", "psutil"]
    if cfg["target"] != "cuda":
        modules += ["openvino", "nncf"]
    if cfg["target"] != "cpu":
        # runtime registers the process-local Windows TensorRT DLL directory before this check.
        from traffic_lab import runtime  # noqa: F401

        modules += ["tensorrt"]
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; fix the environment or choose --target cpu")
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(f"Missing packages: {', '.join(missing)}. See docs/PRECISION_STUDY.md")
    installed = importlib.metadata.version("ultralytics")
    if installed != "8.4.193":
        raise RuntimeError("This runner uses the pinned Ultralytics 8.4.193 quantize API; use this project's environment")


def verify_inputs(cfg):
    manifest = read(Path(cfg["data_root"]) / "manifest.json")
    if manifest["recipe"] != data_recipe(cfg):
        raise ValueError("Data split differs from requested settings; run --prepare-data with a fresh data root")
    calibration = {r["id"] for r in manifest["images"] if r["split"] == "calibration"}
    evaluation = {r["id"] for r in manifest["images"] if r["split"] == "evaluation"}
    if calibration & evaluation or len(calibration) != cfg["calibration_images"] or not evaluation:
        raise ValueError("Invalid or overlapping dataset splits")
    if file_hash(Path(cfg["data_root"]) / "instances_val2017.json") != manifest["annotation_sha256"]:
        raise ValueError("Annotations changed")
    if file_hash(Path(cfg["data_root"]) / "calibration.yaml") != manifest["calibration_yaml_sha256"]:
        raise ValueError("Calibration YAML changed")
    for row in manifest["images"]:
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError(f"Dataset content changed: {row['path']}")
    return manifest


def on_ac_power():
    if os.name != "nt":
        return True  # AC status is not verified on non-Windows hosts.
    status = ctypes.create_string_buffer(12)
    return bool(ctypes.windll.kernel32.GetSystemPowerStatus(status)) and status.raw[0] == 1


@contextlib.contextmanager
def exclusive_run(root):
    import psutil

    root.mkdir(parents=True, exist_ok=True)
    path = root / "run.lock"
    if path.exists():
        previous = read(path)
        try:
            process = psutil.Process(previous["pid"])
            if math.isclose(process.create_time(), previous["created"], rel_tol=0, abs_tol=.01):
                raise RuntimeError("Another precision study is running in this output directory")
        except psutil.NoSuchProcess:
            pass
        path.unlink()  # Only a stale lock belonging to this study; no recursive filesystem operations.
    with path.open("x", encoding="utf-8") as lock:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, lock)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


@contextlib.contextmanager
def awake():
    if os.name == "nt":
        # Request system wakefulness for this process only; leave display and power-plan settings alone.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        yield
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def worker(job):
    cfg, case = job["config"], job["case"]
    if job["stage"] == "export":
        from traffic_lab.cli import export
        from traffic_lab.cli import parser as cli_parser

        fmt = "engine" if case["backend"] == "tensorrt" else case["backend"]
        argv = ["export", "--model", cfg["model"], "--format", fmt, "--imgsz", str(cfg["imgsz"]),
                "--quantize", str(case["precision"]), "--device", "0" if fmt == "engine" else "cpu",
                "--batch", "1", "--workspace", ".5", "--output-dir", str(export_directory(cfg, case))]
        if case["precision"] == 8:
            argv += ["--data", str(Path(cfg["data_root"]) / "calibration.yaml")]
        manifest = export(cli_parser().parse_args(argv))
        manifest["dataset_manifest_sha256"] = json_digest(read(Path(cfg["data_root"]) / "manifest.json"))
        manifest["calibration_note"] = "Same declared image pool; backend algorithms/subsampling may differ. See export log."
        save(job["artifact"], manifest)
        return
    if job["stage"] == "inspect":
        from traffic_lab.inspection import inspect_artifact
        from traffic_lab.runtime import ModelConfig

        inspect_artifact(ModelConfig(model=model_path(cfg, case), imgsz=cfg["imgsz"],
                                    device=case["device"], threads=cfg["threads"]), job["artifact"])
        return
    result = timed_trial(cfg, case, job["round"]) if job["stage"] == "timing" else quality_evaluation(cfg, case)
    result["job_identity"] = job["identity"]
    save(job["artifact"], result)


def execute(job, ledger, root, timeout):
    key = job["key"]
    prior = ledger.get(key)
    artifact = Path(job["artifact"])
    if prior and prior["status"] == "complete" and artifact.exists():
        if prior["identity"] != job["identity"] or prior["artifact_sha256"] != file_hash(artifact):
            raise ValueError(f"Cached evidence changed: {key}; choose a fresh --output")
        if job["stage"] == "timing":
            validate_timing(read(artifact), job["config"])
        log_event(f"CACHED {key} | {ledger_progress(ledger, job.get('planned_stages'))}")
        return True
    attempt = len(prior.get("attempts", [])) + 1 if prior else 1
    job_path = root / "jobs" / f"{key}.json"
    save(job_path, job)
    log_path = root / "logs" / f"{key}.attempt-{attempt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"status": "running", "identity": job["identity"], "artifact": str(artifact),
             "attempts": prior.get("attempts", []) if prior else []}
    ledger[key] = entry
    save(root / "ledger.json", ledger)
    started = time.time()
    log_event(f"START {key} | attempt={attempt} | {ledger_progress(ledger, job.get('planned_stages'))}")
    code, error = -1, None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(job_path)],
                                   stdout=log, stderr=subprocess.STDOUT, env=worker_environment())
        try:
            code = wait_with_progress(process, log_path, key, timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            error = type(exc).__name__
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=log, stderr=subprocess.STDOUT, check=False)
            else:
                process.kill()
            process.wait()
            if isinstance(exc, KeyboardInterrupt):
                entry["status"] = "interrupted"
                save(root / "ledger.json", ledger)
                raise
    success = code == 0 and artifact.exists()
    # Export hashes bind the actual model/engine as well as the small manifest.
    if success and job["stage"] == "export":
        exported = read(artifact)
        success = file_hash(exported["artifact"]) == exported["artifact_sha256"]
    entry.update(status="complete" if success else "failed", artifact_sha256=file_hash(artifact) if success else None)
    entry["attempts"].append({"exit_code": code, "error": error, "elapsed_seconds": time.time()-started,
                              "log": str(log_path), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    save(root / "ledger.json", ledger)
    log_event(f"{'OK' if success else 'FAILED'} {key} | elapsed={time.time()-started:.1f}s | "
              f"{ledger_progress(ledger, job.get('planned_stages'))} | log={log_path}")
    return success


def run(cfg, timeout):
    if not on_ac_power():
        raise RuntimeError("Connect the laptop to AC power before starting this study")
    check_dependencies(cfg)
    root = Path(cfg["output"])
    manifest = verify_inputs(cfg)
    variants = cases(cfg["target"])
    planned_stages = len({c["export_key"] for c in variants if c["export_key"]}) + len(variants) * (cfg["trials"] + 2)
    orders = randomized_rounds(variants, cfg["trials"], cfg["seed"])
    bindings = {"model_sha256": file_hash(cfg["model"]), "source_sha256": file_hash(cfg["source"]),
                "dataset_sha256": json_digest(manifest), "runner_sha256": file_hash(__file__),
                "helpers_sha256": file_hash(REPO / "src/traffic_lab/precision.py"),
                "runtime_sha256": file_hash(REPO / "src/traffic_lab/runtime.py"),
                "environment": environment_snapshot()}
    plan = {"config": cfg, "bindings": bindings, "cases": variants, "orders": orders,
            "scope": "One checkpoint; batch one; one resolution; no architecture change or retraining",
            "limitations": ["Free-running shared laptop clocks, one session; no causal FP16 claim from cross-runtime ratios",
                            "ONNX CUDA INT8 deliberately omitted: no verified CUDA INT8 execution path in this stack",
                            "FP16 graph/storage and TensorRT precision flags do not imply every operation has that precision",
                            "AP point estimates have no CI; near-budget differences cannot establish pass/fail"]}
    if (root / "plan.json").exists() and json_digest(read(root / "plan.json")) != json_digest(plan):
        raise ValueError("Inputs, environment, source or settings changed. Use a fresh --output; old evidence is preserved.")
    save(root / "plan.json", plan)
    ledger = read(root / "ledger.json") if (root / "ledger.json").exists() else {}
    by_name = {c["name"]: c for c in variants}

    def stage(kind, case, round_index=None):
        key = f"{kind}-{case['export_key'] if kind == 'export' else case['name']}"
        if round_index is not None:
            key += f"-round-{round_index:02d}"
        artifact = root / "exports" / case["export_key"] / "export.json" if kind == "export" else root / kind / f"{key}.json"
        job = {"key": key, "stage": kind, "config": cfg, "case": case, "round": round_index,
               "artifact": str(artifact), "planned_stages": planned_stages}
        dependency = dict(bindings)
        if kind != "export":
            dependency["artifact_sha256"] = file_hash(model_path(cfg, case))
        job["identity"] = evidence_identity(job, dependency)
        return execute(job, ledger, root, timeout)

    with awake():
        exported, attempted = set(), set()
        for case in variants:
            key = case["export_key"]
            if key and key not in attempted:
                attempted.add(key)
                if stage("export", case):
                    exported.add(key)
        ready = [c["name"] for c in variants if not c["export_key"] or c["export_key"] in exported]
        verified = [name for name in ready if stage("inspect", by_name[name])]
        for round_index, order in enumerate(orders):
            log_event(f"TIMING ROUND {round_index+1}/{cfg['trials']} | {len(order)} configurations | "
                      f">={cfg['seconds']}s each, plus real-image warm-up")
            for name in order:
                if name in verified:
                    if not on_ac_power():
                        raise RuntimeError("AC disconnected. Completed stages are saved; reconnect and resume.")
                    stage("timing", by_name[name], round_index)
            report(root)
        for name in verified:
            stage("quality", by_name[name])
            report(root)
    report(root)
    failures = [k for k, v in ledger.items() if v["status"] != "complete"]
    print(f"Study finished. Failed/incomplete stages: {failures}. Report: {root / 'REPORT.md'}")
    return int(bool(failures))


def report(root):
    root = Path(root).resolve()
    plan, ledger = read(root / "plan.json"), read(root / "ledger.json")

    def valid(path):
        entry = next((v for v in ledger.values() if v["artifact"] == str(path)), None)
        return entry and entry["status"] == "complete" and path.exists() and file_hash(path) == entry["artifact_sha256"]

    timings, quality = {}, {}
    for case in plan["cases"]:
        name = case["name"]
        timings[name] = [read(p) for p in sorted((root / "timing").glob(f"timing-{name}-round-*.json")) if valid(p)]
        for trial in timings[name]:
            validate_timing(trial, plan["config"])
        path = root / "quality" / f"quality-{name}.json"
        if valid(path):
            quality[name] = read(path)
    reference_name = ("pytorch-cuda-fp32" if any(c["name"] == "pytorch-cuda-fp32" for c in plan["cases"])
                      else "pytorch-cpu-fp32")
    reference = quality.get(reference_name)
    rows, comparisons = [], {}
    for case in plan["cases"]:
        name, trials = case["name"], timings[case["name"]]
        q = quality.get(name)
        fp32_name = f"{case['backend']}-{'cpu' if case['device'] == 'cpu' else 'cuda'}-fp32"
        fp32 = quality.get(fp32_name)
        drop = 100 * (reference["map50_95"] - q["map50_95"]) if reference and q else None
        within_drop = 100 * (fp32["map50_95"] - q["map50_95"]) if fp32 and q else None
        entry = {"case": name, "device": "CPU" if case["device"] == "cpu" else "CUDA",
                 "trials": len(trials), "expected_trials": plan["config"]["trials"],
                 "fps_median": median_value([t["fps"] for t in trials]),
                 "fps_min": min((t["fps"] for t in trials), default=None),
                 "fps_max": max((t["fps"] for t in trials), default=None),
                 "trial_p95_median_ms": median_value([t["p95_ms"] for t in trials]),
                 "map50_95": q["map50_95"] if q else None, "vehicle_ap": q["vehicle_ap"] if q else None,
                 "loss_vs_pytorch_points": drop, "loss_vs_runtime_fp32_points": within_drop,
                 "quality_budget": "not evaluated" if drop is None else
                 ("point estimate within budget; uncertainty unmeasured" if drop <= plan["config"]["quality_budget_points"]
                  else "point estimate exceeds budget; uncertainty unmeasured"),
                 "flagged_trials": sum(bool(t["flags"]) for t in trials)}
        for field, metric in (("peak_sampled_rss_mib", "process_rss_mib"), ("peak_whole_gpu_mib", "gpu_used_mib")):
            observed = [t["resources"]["metrics"].get(metric, {}).get("max") for t in trials]
            entry[field] = max((value for value in observed if value is not None), default=None)
        try:
            path = Path(model_path(plan["config"], case))
            entry["artifact_mib"] = (sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                                     if path.is_dir() else path.stat().st_size) / 1024**2
        except (OSError, KeyError, ValueError):
            entry["artifact_mib"] = None
        if trials and timings.get(fp32_name):
            comparisons[name] = paired_ratio({t["round"]: t["fps"] for t in trials},
                                              {t["round"]: t["fps"] for t in timings[fp32_name]})
        rows.append(entry)
    summary = {"kind": "one_checkpoint_precision_study", "reference": reference_name,
               "rows": rows, "within_runtime_paired_fps_ratios": comparisons,
               "completed_stages": sum(v["status"] == "complete" for v in ledger.values()),
               "failed_stages": [k for k, v in ledger.items() if v["status"] != "complete"]}
    save(root / "summary.json", summary)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# One-checkpoint YOLO precision study", "", f"Accuracy reference: `{reference_name}`.", "",
             "Missing measurements are shown as `pending`; incomplete studies are not completed results.", ""]
    for device in ("CPU", "CUDA"):
        lines += [f"## {device}", "", "| Case | Trials | FPS median [min, max] | Median trial p95 ms | mAP50–95 | AP loss vs reference (points) | Model MiB |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["device"] == device:
                lines.append(f"| {row['case']} | {row['trials']}/{row['expected_trials']} | "
                             f"{fmt(row['fps_median'])} [{fmt(row['fps_min'])}, {fmt(row['fps_max'])}] | "
                             f"{fmt(row['trial_p95_median_ms'])} | {fmt(row['map50_95'],4)} | "
                             f"{fmt(row['loss_vs_pytorch_points'],4)} | {fmt(row['artifact_mib'])} |")
        lines.append("")
    lines += ["## Interpretation", "", "- Compare precisions within a runtime/device first; CPU/GPU are separate experiments.",
              "- Warmed, predecoded batch-one wall time includes preprocessing, prediction, NMS and completed CPU boxes.",
              f"- Timing uses conf=.25, class filter `{plan['config'].get('timing_classes', 'all')}`; accuracy uses conf=.001 and all classes.",
              "- Ten trials are not ten independent days. Single-session log-ratio intervals are exploratory.",
              "- OpenVINO FP16 means stored constants; CPU arithmetic may be FP32. ONNX FP16 CPU may also execute FP32 kernels.",
              "- TensorRT flags permit mixed precision. Inspect execution evidence rather than assuming pure FP16/INT8.",
              "- AP has no confidence interval here. Do not declare a near-threshold precision variant safe or unsafe from point estimates alone.",
              "- RSS includes runtime plus decoded pool; NVML memory is whole-device; unsupported metrics remain missing.",
              "- Inspect logs/ledger for failed stages. Failed cases are not silently removed or replaced with old measurements."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def median_value(values):
    return spread(values)["median"] if values else None


def fmt(value, digits=2):
    return "pending" if value is None else f"{value:.{digits}f}"


def main(argv=None):
    os.chdir(REPO)
    args = parser().parse_args(argv)
    if args.worker:
        worker(read(args.worker))
        return 0
    cfg = config(args)
    if args.prepare_data:
        prepare_data(cfg)
    elif args.run:
        with exclusive_run(Path(cfg["output"])):
            return run(cfg, args.stage_timeout)
    elif args.report_only:
        report(args.output)
    else:
        variants = cases(args.target)
        print(json.dumps({"action": "DRY RUN — no downloads, exports, inference or results written",
                          "config": cfg, "cases": variants,
                          "minimum_timing_minutes_excluding_warmup_and_loading": len(variants)*args.trials*args.seconds/60,
                          "evaluation_images": min(args.evaluation_limit or 5000, 5000-args.calibration_images),
                          "next": "Use --prepare-data, then --run. Read docs/PRECISION_STUDY.md."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
