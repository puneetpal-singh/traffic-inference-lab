"""Kaggle smoke gate and full multi-camera traffic study. Never run automatically on import."""

import argparse
import collections
import html
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from statistics import median

import precision_study as precision

from traffic_lab.confirmation import json_digest
from traffic_lab.precision import cases
from traffic_lab.progress import (
    ledger_progress,
    log_event,
    stage_progress,
    wait_with_progress,
    worker_environment,
)
from traffic_lab.provenance import file_hash

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return precision.read(path)


def save(path, value):
    precision.save(path, value)


def configuration(path):
    cfg = read(path)
    defaults = {"mode": "smoke", "phase": "all", "run_id": "kaggle-v1", "image_sizes": [320, 640],
                "threads": 4, "seed": 20261102, "camera_counts": [1, 2, 4, 8],
                "serving_repeats": 3, "serving_seconds": 30, "model_trials": 10, "model_seconds": 20,
                "min_camera_fps": 15, "max_frame_age_p95_ms": 100, "max_ap_loss_points": 1,
                "min_count_f1": .95, "max_count_error_fraction": .05, "min_label_events": 5,
                "line": [.1, .55, .9, .55], "traffic_clips": [], "event_labels": {}, "run_tuning": True}
    cfg = {**defaults, **cfg}
    if cfg["mode"] not in ("smoke", "full") or cfg["phase"] not in ("all", "models", "serving", "tuning"):
        raise ValueError("Unknown mode/phase")
    if not cfg["run_id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in cfg["run_id"]):
        raise ValueError("run_id must contain only letters, digits, hyphens and underscores")
    if (min(cfg["camera_counts"]) < 1 or cfg["serving_repeats"] < 3 or cfg["serving_seconds"] < 30
            or cfg["model_trials"] < 2 or cfg["model_seconds"] < 20 or cfg["threads"] < 1
            or 320 not in cfg["image_sizes"] or any(s < 32 or s % 32 for s in cfg["image_sizes"])):
        raise ValueError("Full study requires 320px, valid sizes/counts, >=3x30s serving and >=2x20s timing")
    if (not 0 <= cfg["min_count_f1"] <= 1 or cfg["max_count_error_fraction"] < 0
            or cfg["max_ap_loss_points"] < 0 or cfg["min_label_events"] < 1
            or cfg["min_camera_fps"] <= 0 or cfg["max_frame_age_p95_ms"] <= 0):
        raise ValueError("Invalid predeclared budgets")
    numeric = ("model_seconds", "serving_seconds", "min_camera_fps", "max_frame_age_p95_ms",
               "max_ap_loss_points", "min_count_f1", "max_count_error_fraction")
    if any(not math.isfinite(cfg[key]) for key in numeric):
        raise ValueError("Budgets and durations must be finite")
    if len(cfg["line"]) != 4 or any(not 0 <= p <= 1 for p in cfg["line"]) or cfg["line"][:2] == cfg["line"][2:]:
        raise ValueError("Invalid normalized counting line")
    return cfg


def study_root(cfg):
    return REPO / "results" / cfg["run_id"] / cfg["mode"]


def model_cfg(cfg, size, smoke=False):
    args = ["--imgsz", str(size), "--threads", str(cfg["threads"]), "--timing-classes", "vehicles",
            "--output", str(study_root(cfg) / f"precision-{size}"),
            "--data-root", str(REPO / "data" / f"{cfg['run_id']}-{'smoke' if smoke else 'full'}"),
            "--calibration-images", "32" if smoke else "512", "--evaluation-limit", "8" if smoke else "0",
            "--trials", "2" if smoke else str(cfg["model_trials"]), "--seconds", str(cfg["model_seconds"]),
            "--pool-frames", "8" if smoke else "60", "--seed", str(cfg["seed"])]
    result = precision.config(precision.parser().parse_args(args))
    result["source"] = str(Path(cfg["traffic_clips"][0]).resolve())
    return result


def duration(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or fps <= 0 or frames <= 0:
            raise ValueError(f"Cannot read video metadata: {path}")
        return frames / fps
    finally:
        cap.release()


def verify_accounting(summary):
    cameras = summary["cameras"]
    if summary["status"] != "complete" or not cameras:
        raise ValueError("Pipeline did not complete")
    for camera in cameras:
        if camera["decoder_threads"] != 1:
            raise ValueError("Decoder did not honor the one-thread setting")
        if camera["decoded"] != sum(camera[k] for k in ("processed", "queue_dropped", "stride_skipped", "remaining")):
            raise ValueError("Decoded/processed/drop/skip accounting mismatch")
        if camera["queue_high_water"] > summary["queue_capacity_per_camera"]:
            raise ValueError("Queue exceeded its declared bound")


def label_document(path, source, line):
    if not path:
        return None, "missing independent event labels"
    document = read(path)
    if not document.get("reviewed"):
        return None, "label template has not been independently reviewed"
    if document.get("source_sha256") != file_hash(source) or document.get("line") != list(line):
        raise ValueError("Labels do not match source bytes/counting line")
    for event in document["events"]:
        if (event["direction"] not in ("positive", "negative") or event["class_id"] not in (2, 3, 5, 7)
                or not math.isfinite(event["source_seconds"]) or event["source_seconds"] < 0):
            raise ValueError("Invalid crossing event label")
    return document, None


def count_quality(job, predictions):
    from traffic_lab.scoring import score_events

    cfg, truths, reasons = job["study"], [], []
    for camera, original in enumerate(job["original_sources"]):
        if job["looped"][camera]:
            reasons.append("looped short source: natural traffic counting accuracy not established")
            continue
        document, error = label_document(cfg["event_labels"].get(Path(original).name), original, cfg["line"])
        if error:
            reasons.append(error)
            continue
        events = [e for e in document["events"] if e["source_seconds"] < job["seconds"]]
        if len(events) < cfg["min_label_events"]:
            reasons.append("insufficient independently labelled crossings in evaluation window")
        truths.extend({**event, "camera": camera} for event in events)
    if reasons:
        return {"status": "unvalidated", "reasons": sorted(set(reasons))}
    by_camera = []
    for camera in range(job["streams"]):
        truth = [e for e in truths if e["camera"] == camera]
        predicted = [e for e in predictions if e["camera"] == camera]
        score = score_events(truth, predicted, tolerance=.5)
        score["relative_count_error"] = score["absolute_count_error"] / len(truth)
        score["passed"] = score["f1"] >= cfg["min_count_f1"] and score["relative_count_error"] <= cfg["max_count_error_fraction"]
        by_camera.append(score)
    return {"status": "evaluated", "passed": all(c["passed"] for c in by_camera), "cameras": by_camera}


def pipeline_metrics(summary, cfg):
    verify_accounting(summary)
    cameras = summary["cameras"]
    rates = [c["processed_fps"] for c in cameras]
    ages = [c["frame_age_ms"]["p95"] for c in cameras]
    if any(a is None for a in ages):
        raise ValueError("Camera has no frame-age measurements")
    return {"min_camera_fps": min(rates), "aggregate_fps": sum(rates), "worst_camera_p95_ms": max(ages),
            "fairness_jain": sum(rates)**2 / (len(rates) * sum(r*r for r in rates)) if any(rates) else 0,
            "decoded": sum(c["decoded"] for c in cameras), "processed": sum(c["processed"] for c in cameras),
            "dropped": sum(c["queue_dropped"] for c in cameras),
            "timing_passed": min(rates) >= cfg["min_camera_fps"] and max(ages) <= cfg["max_frame_age_p95_ms"]}


def process_job(job):
    cfg, stage = job["study"], job["stage"]
    if stage in ("export", "inspect", "quality"):
        precision.worker({**job, "config": job["model_config"]})
        return
    if stage == "probe":
        from traffic_lab.runtime import video_frames

        obj = precision.detector(job["model_config"], job["case"])
        frames = video_frames(cfg["traffic_clips"][0], 3)
        if len(frames) != 3:
            raise ValueError("Smoke source needs at least three frames")
        for frame in frames:
            obj.predict([frame])[0].boxes.cpu().numpy()
        save(job["artifact"], {"kind": "compatibility_probe_only", "case": job["case"],
                              "frames": len(frames), "execution": obj.metadata(), "not_a_benchmark": True})
        return
    if stage == "dynamic-export":
        from traffic_lab.cli import export, parser

        case, mc = job["case"], job["model_config"]
        argv = ["export", "--model", mc["model"], "--imgsz", "320", "--device", "0", "--format", "engine",
                "--quantize", str(case["precision"]), "--batch", "4", "--dynamic", "--workspace", ".5",
                "--output-dir", str(precision.export_directory(mc, case))]
        if case["precision"] == 8:
            argv += ["--data", str(Path(mc["data_root"]) / "calibration.yaml")]
        save(job["artifact"], export(parser().parse_args(argv)))
        return
    if stage == "pipeline":
        from traffic_lab.pipeline import run_pipeline
        from traffic_lab.runtime import ModelConfig

        mc, case = job["model_config"], job["case"]
        output = Path(job["artifact"]).parent
        summary = run_pipeline(ModelConfig(model=precision.model_path(mc, case), imgsz=320, device=case["device"],
                                          threads=cfg["threads"]), job["sources"], output, duration=job["seconds"],
                               batch=job.get("batch", 1), stride=job.get("stride", 1),
                               tracker="bytetrack", line=tuple(cfg["line"]), render=job.get("render", False),
                               decode_threads=1, brightness_grid=job.get("brightness_grid", 0))
        metrics = pipeline_metrics(summary, cfg)
        if job.get("render") and not summary.get("browser_videos"):
            raise ValueError("Browser-compatible annotated rendering did not complete")
        if any(c["decoded"] < math.ceil(job["seconds"] * c["source_fps"] - .01) for c in summary["cameras"]):
            raise ValueError("Replay ended before requested duration")
        save(job["artifact"], {"kind": "kaggle_pipeline_experiment", "job": job,
                               "summary_sha256": file_hash(output / "summary.json"), **metrics,
                               "counting": count_quality(job, read(output / "events.json"))})
        return
    raise ValueError(f"Unknown stage: {stage}")


def execute(job, root, bindings, timeout=1800):
    job = {k: v for k, v in job.items() if k != "identity"}
    if job["stage"] not in ("export", "dynamic-export"):
        bindings = {**bindings, "model_sha256": file_hash(precision.model_path(job["model_config"], job["case"]))}
    root = Path(root)
    ledger_path = root / "ledger.json"
    ledger = read(ledger_path) if ledger_path.exists() else {}
    job["identity"] = json_digest({"job": job, "bindings": bindings})
    previous = ledger.get(job["key"], {})
    planned = (len({c["export_key"] for c in cases() if c["export_key"]}) + 3 * len(cases()) + 3
               if job["study"]["mode"] == "smoke" else None)
    artifact = Path(job["artifact"])
    if previous.get("status") == "complete" and artifact.exists():
        if previous["identity"] != job["identity"] or previous["sha256"] != file_hash(artifact):
            raise ValueError("Cached study evidence changed; use a new run_id")
        if job["stage"] in ("export", "dynamic-export"):
            manifest = read(artifact)
            if file_hash(manifest["artifact"]) != manifest["artifact_sha256"]:
                raise ValueError("Cached engine changed; use a new run_id")
        log_event(f"CACHED {job['key']} — {ledger_progress(ledger, planned)}")
        return True
    attempt = len(previous.get("attempts", [])) + 1
    job_file = root / "jobs" / f"{job['key']}.json"
    save(job_file, job)
    log_file = root / "logs" / f"{job['key']}.attempt-{attempt}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {"status": "running", "identity": job["identity"], "artifact": str(artifact),
             "attempts": previous.get("attempts", [])}
    ledger[job["key"]] = entry
    save(ledger_path, ledger)
    log_event(f"START {job['key']} | attempt={attempt} | {ledger_progress(ledger, planned)} | log={log_file}")
    tick, code, error = time.time(), -1, None
    with log_file.open("w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(job_file)],
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt",
                                 env=worker_environment())
        try:
            code = wait_with_progress(child, log_file, job["key"], timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            error = type(exc).__name__
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], stdout=log, stderr=log, check=False)
            else:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            if isinstance(exc, KeyboardInterrupt):
                entry["status"] = "interrupted"
                save(ledger_path, ledger)
                raise
    success = code == 0 and artifact.exists()
    entry.update(status="complete" if success else "failed", sha256=file_hash(artifact) if success else None)
    entry["attempts"].append({"exit_code": code, "error": error, "seconds": time.time()-tick, "log": str(log_file)})
    save(ledger_path, ledger)
    log_event(f"{'OK' if success else 'FAILED'} {job['key']} | elapsed={time.time()-tick:.1f}s | {ledger_progress(ledger, planned)}")
    return success


def preflight(cfg):
    log_event("PREFLIGHT: checking GPU, CUDA arithmetic, FFmpeg and dependencies")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator, then rerun")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required for replay extension and browser-compatible rendering")
    if torch.cuda.get_device_capability(0) < (7, 5):
        raise RuntimeError("Use a T4 or newer GPU for this pinned TensorRT stack")
    tensor = torch.ones((32, 32), device="cuda:0")
    if float((tensor @ tensor)[0, 0].cpu()) != 32:
        raise RuntimeError("CUDA arithmetic failed")
    precision.check_dependencies({"target": "all"})
    probe = {"environment": precision.environment_snapshot(), "selected_gpu": torch.cuda.get_device_name(0),
             "selected_device": "cuda:0", "cuda_runtime": torch.version.cuda,
             "note": "One GPU only; no parallel configurations. CPU host and accelerator can vary across sessions."}
    save(study_root(cfg) / "preflight.json", probe)
    log_event(f"PREFLIGHT PASS: {probe['selected_gpu']} | CUDA {probe['cuda_runtime']} | GPU0 only")
    return probe


def assets(cfg):
    log_event("ASSETS: downloading/verifying YOLO checkpoint and traffic recordings")
    from traffic_lab.cli import fetch, parser

    fetch(parser().parse_args(["fetch"]))
    clips = cfg["traffic_clips"] or [str(REPO / "data/traffic.mp4")]
    if len({Path(p).name for p in clips}) != len(clips):
        raise ValueError("Use unique video basenames for independent label matching")
    for clip in clips:
        if duration(clip) < 2:
            raise ValueError("Each traffic clip must have at least two seconds of footage")
    cfg["traffic_clips"] = [str(Path(p).resolve()) for p in clips]
    kit = study_root(cfg) / "annotation-kit"
    kit.mkdir(parents=True, exist_ok=True)
    for source in cfg["traffic_clips"]:
        template = kit / f"{Path(source).stem}.events.template.json"
        if not template.exists():
            save(template, {"reviewed": False, "source_sha256": file_hash(source), "line": cfg["line"], "events": [],
                            "instructions": "Independently watch the original video and label every finite-line crossing. "
                            "Events: source_seconds, class_id (2 car/3 motorcycle/5 bus/7 truck), direction. "
                            "For the default left-to-right horizontal line, positive means downward; negative upward. "
                            "Use bottom-center of each vehicle; do not copy predicted events. Then set reviewed=true."})
        label_document(cfg["event_labels"].get(Path(source).name), source, cfg["line"])


def bind_plan(cfg, preflight):
    sources = {p: file_hash(p) for p in cfg["traffic_clips"]}
    labels = {k: file_hash(p) for k, p in cfg["event_labels"].items()}
    # Phase is execution control, not an experiment setting: a phased resume must have the same identity.
    settings = {k: v for k, v in cfg.items() if k != "phase"}
    source_files = [Path(__file__), REPO / "scripts/precision_study.py"] + sorted((REPO / "src/traffic_lab").glob("*.py"))
    bindings = {"settings": settings, "sources": sources, "labels": labels,
                "code": {p.relative_to(REPO).as_posix(): file_hash(p) for p in source_files},
                "environment": preflight["environment"], "checkpoint": file_hash("models/yolo11n.pt")}
    plan = {"bindings": bindings, "question": "Camera capacity under declared freshness, COCO proxy and counting budgets",
            "counting_requirement": "Independent reviewed event labels on non-looped recordings; otherwise unvalidated"}
    path = study_root(cfg) / "plan.json"
    if path.exists() and json_digest(read(path)) != json_digest(plan):
        raise ValueError("Hardware/software/data/settings changed. Choose a fresh RUN_ID; never mix studies")
    save(path, plan)
    return bindings


def basic_job(cfg, mc, stage, case, artifact, key=None):
    return {"study": {**cfg, "phase": "all"}, "model_config": mc, "case": case, "stage": stage,
            "key": key or f"{stage}-{case['name']}", "artifact": str(artifact)}


def smoke(cfg, bindings):
    root, mc = study_root(cfg), model_cfg(cfg, 320, smoke=True)
    log_event("SMOKE DATA: preparing 32 calibration / 8 evaluation images")
    with stage_progress("SMOKE DATA download/verification"):
        precision.prepare_data(mc)
    log_event("SMOKE: 9 exports, 13 runtime probes/evaluations, then 3 pipeline checks")
    variants, success, exported = cases(), True, {}
    data_binding = {**bindings, "dataset": json_digest(read(Path(mc["data_root"]) / "manifest.json"))}
    for case in variants:
        key = case["export_key"]
        if key and key not in exported:
            artifact = Path(mc["output"]) / "exports" / key / "export.json"
            exported[key] = execute(basic_job(cfg, mc, "export", case, artifact, "export-"+key), root, data_binding)
        if key and not exported[key]:
            success = False
            continue
        for stage in ("inspect", "probe", "quality"):
            artifact = root / stage / f"{case['name']}.json"
            if not execute(basic_job(cfg, mc, stage, case, artifact), root, data_binding):
                success = False
                break
    # Independent trackers, queue accounting and browser-compatible annotated video are exercised too.
    for case_name, streams, render in (("pytorch-cuda-fp32", 1, False), ("tensorrt-cuda-fp16", 1, True),
                                       ("tensorrt-cuda-fp16", 2, False)):
        case = next(c for c in variants if c["name"] == case_name)
        if case["export_key"] and not exported.get(case["export_key"]):
            success = False
            continue
        key = f"pipeline-{case_name}-{streams}cam-render{int(render)}"
        source = cfg["traffic_clips"][0]
        job = {**basic_job(cfg, mc, "pipeline", case, root / "pipelines" / key / "measurement.json", key),
               "sources": [source]*streams, "original_sources": [source]*streams, "looped": [False]*streams,
               "seconds": 2, "streams": streams, "render": render}
        success = execute(job, root, data_binding) and success
    ledger = read(root / "ledger.json")
    success = success and all(v["status"] == "complete" for v in ledger.values())
    result = {"passed": success, "binding_sha256": json_digest(bindings), "preflight": read(root / "preflight.json"),
              "warning": "32 calibration / 8 evaluation images, three-frame probes and 2s replays: compatibility only, NOT benchmark evidence",
              "failed": [k for k, v in ledger.items() if v["status"] != "complete"]}
    save(root / "smoke-status.json", result)
    print(f"SMOKE {'PASS' if success else 'FAIL'}: {result['failed']}", flush=True)
    return success


def validate_smoke_gate(cfg, preflight):
    path = study_root({**cfg, "mode": "smoke"}) / "smoke-status.json"
    if not path.exists() or not read(path).get("passed"):
        raise RuntimeError("Run MODE='smoke' successfully before starting the full study")
    old = read(path)["preflight"]["environment"]
    if old != preflight["environment"]:
        raise RuntimeError("Assigned environment changed since smoke. Use a new RUN_ID and repeat smoke")


def selections(cfg):
    mc = model_cfg(cfg, 320)
    root = Path(mc["output"])
    ledger = read(root / "ledger.json")
    groups, decisions = collections.defaultdict(list), []
    reference = read(root / "quality/quality-pytorch-cuda-fp32.json")
    reference_entry = ledger.get("quality-pytorch-cuda-fp32", {})
    if (reference_entry.get("status") != "complete" or
            file_hash(root / "quality/quality-pytorch-cuda-fp32.json") != reference_entry.get("artifact_sha256")):
        raise ValueError("Reference accuracy is missing or changed")
    for case in cases():
        path = root / "quality" / f"quality-{case['name']}.json"
        entry = ledger.get("quality-"+case["name"], {})
        if entry.get("status") != "complete" or not path.exists() or file_hash(path) != entry["artifact_sha256"]:
            continue
        quality = read(path)
        timing = []
        for index in range(cfg["model_trials"]):
            key = f"timing-{case['name']}-round-{index:02d}"
            record = ledger.get(key, {})
            target = root / "timing" / f"{key}.json"
            if record.get("status") == "complete" and target.exists() and file_hash(target) == record["artifact_sha256"]:
                timing.append(read(target))
        if len(timing) != cfg["model_trials"] or quality["vehicle_ap"] is None:
            continue
        drop = 100 * (reference["map50_95"] - quality["map50_95"])
        vehicle_drop = 100 * (reference["vehicle_ap"] - quality["vehicle_ap"])
        eligible = max(drop, vehicle_drop) <= cfg["max_ap_loss_points"]
        item = {"case": case, "global_loss_points": drop, "vehicle_loss_points": vehicle_drop,
                "eligible_by_point_estimate": eligible, "median_p95_ms": median(t["p95_ms"] for t in timing),
                "isolated_model_reference_fps": median(t["fps"] for t in timing)}
        decisions.append(item)
        if eligible:
            groups[(case["backend"], case["device"])].append(item)
    selected = [min(items, key=lambda x: x["median_p95_ms"]) for items in groups.values()]
    save(study_root(cfg) / "selections.json", {"candidates": decisions, "selected": selected,
         "scope": "Point-estimate COCO proxy gate, no AP CI. First-clip isolated FPS is a reference, not a hard serving ceiling."})
    if not selected:
        raise RuntimeError("No fully measured candidate is eligible; inspect precision study failures")
    return selected


def replay_sources(cfg):
    sources, loops = [], []
    root = REPO / "data" / cfg["run_id"] / "replays"
    root.mkdir(parents=True, exist_ok=True)
    for source in cfg["traffic_clips"]:
        looped = duration(source) < cfg["serving_seconds"] + .1
        if looped:
            target = root / f"{file_hash(source)[:12]}-{cfg['serving_seconds']+2}s.mp4"
            if not target.exists():
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", source,
                                "-t", str(cfg["serving_seconds"]+2), "-an", "-c:v", "copy", str(target)], check=True)
            sources.append(str(target))
        else:
            sources.append(source)
        loops.append(looped)
    return sources, loops


def serving(cfg, bindings, tuning=False):
    root, mc = study_root(cfg), model_cfg(cfg, 320)
    selected = selections(cfg)
    sources, looped = replay_sources(cfg)
    scenarios = []
    for item in selected:
        if not tuning:
            for streams in cfg["camera_counts"]:
                scenarios.append({"item": item, "streams": streams, "variant": "baseline"})
        elif item["case"]["backend"] == "tensorrt":
            scenarios += [{"item": item, "streams": 1, "variant": "render", "render": True},
                          {"item": item, "streams": 4, "variant": "sampled-telemetry", "brightness_grid": 64},
                          {"item": item, "streams": 4, "variant": "stride2", "stride": 2}]
            dynamic = {**item["case"], "name": "tensorrt-cuda-dynamic-b4", "export_key": "tensorrt-dynamic-b4"}
            artifact = Path(mc["output"]) / "exports" / dynamic["export_key"] / "export.json"
            if execute(basic_job(cfg, mc, "dynamic-export", dynamic, artifact), root, bindings):
                quality_path = root / "quality/dynamic-b4.json"
                if execute(basic_job(cfg, mc, "quality", dynamic, quality_path), root, bindings, 7200):
                    q = read(quality_path)
                    base = read(Path(mc["output"]) / "quality/quality-pytorch-cuda-fp32.json")
                    if (q["vehicle_ap"] is not None and
                            max(100*(base["map50_95"]-q["map50_95"]), 100*(base["vehicle_ap"]-q["vehicle_ap"])) <= cfg["max_ap_loss_points"]):
                        for streams in (4, 8):
                            scenarios.append({"item": {**item, "case": dynamic}, "streams": streams,
                                              "variant": "dynamic-batch4", "batch": 4})
    for repeat in range(cfg["serving_repeats"]):
        log_event(f"{'TUNING' if tuning else 'SERVING'}: repeat {repeat+1}/{cfg['serving_repeats']}, {len(scenarios)} scenarios")
        order = scenarios.copy()
        random.Random(cfg["seed"] + repeat + (1000 if tuning else 0)).shuffle(order)
        for scenario_index, scenario in enumerate(order, 1):
            item, streams = scenario["item"], scenario["streams"]
            indices = [i % len(sources) for i in range(streams)]
            name = f"{item['case']['name']}-{streams}cam-{scenario['variant']}-r{repeat}"
            log_event(f"SCENARIO {scenario_index}/{len(order)} | repeat {repeat+1}/{cfg['serving_repeats']} | {name}")
            job = {**basic_job(cfg, mc, "pipeline", item["case"], root / "pipelines" / name / "measurement.json", name),
                   **{k: v for k, v in scenario.items() if k not in ("item", "variant")},
                   "variant": scenario["variant"], "repeat": repeat, "seconds": cfg["serving_seconds"],
                   "sources": [sources[i] for i in indices], "original_sources": [cfg["traffic_clips"][i] for i in indices],
                   "looped": [looped[i] for i in indices], "isolated_model_reference_fps": item["isolated_model_reference_fps"]}
            artifact_hash = file_hash(precision.model_path(mc, item["case"]))
            execute(job, root, {**bindings, "engine": artifact_hash,
                               "replay_hashes": [file_hash(s) for s in job["sources"]]}, 1200)
        report(cfg)
        archive(cfg)


def report(cfg):
    root = study_root(cfg)
    ledger = read(root / "ledger.json") if (root / "ledger.json").exists() else {}
    records = []
    for entry in ledger.values():
        path = Path(entry["artifact"])
        if entry["status"] == "complete" and path.exists() and file_hash(path) == entry["sha256"]:
            record = read(path)
            if record.get("kind") == "kaggle_pipeline_experiment":
                records.append(record)
    groups = collections.defaultdict(list)
    for record in records:
        job = record["job"]
        groups[(job["case"]["name"], job["streams"], job.get("variant", "smoke"))].append(record)
    rows = []
    for (name, streams, variant), runs in sorted(groups.items()):
        enough = len(runs) == cfg["serving_repeats"] and cfg["mode"] == "full"
        timing = enough and all(r["timing_passed"] for r in runs)
        count = enough and all(r["counting"].get("passed", False) for r in runs)
        rows.append({"case": name, "streams": streams, "variant": variant, "repeats": len(runs),
                     "min_camera_fps_median": median(r["min_camera_fps"] for r in runs),
                     "min_camera_fps_range": [min(r["min_camera_fps"] for r in runs), max(r["min_camera_fps"] for r in runs)],
                     "worst_p95_ms_median": median(r["worst_camera_p95_ms"] for r in runs),
                     "worst_p95_ms_range": [min(r["worst_camera_p95_ms"] for r in runs), max(r["worst_camera_p95_ms"] for r in runs)],
                     "isolated_reference_fps": runs[0]["job"].get("isolated_model_reference_fps"),
                     "all_repeats_timing_passed": timing, "all_repeats_counting_passed": count,
                     "counting_status": "evaluated" if all(r["counting"]["status"] == "evaluated" for r in runs) else "unvalidated",
                     "fairness_jain_median": median(r["fairness_jain"] for r in runs)})
    capacity = []
    for name in sorted({r["case"] for r in rows if r["variant"] == "baseline"}):
        baseline = [r for r in rows if r["case"] == name and r["variant"] == "baseline"]
        capacity.append({"case": name,
                         "largest_tested_timing_pass": max((r["streams"] for r in baseline if r["all_repeats_timing_passed"]), default=None),
                         "largest_tested_timing_and_counting_pass": max((r["streams"] for r in baseline
                              if r["all_repeats_timing_passed"] and r["all_repeats_counting_passed"]), default=None),
                         "complete_tested_loads": sorted(r["streams"] for r in baseline if r["repeats"] == cfg["serving_repeats"]),
                         "scope": "Largest passing tested load, not a production ceiling; model accuracy uses a COCO point-estimate proxy"})
    result = {"mode": cfg["mode"], "rows": rows, "capacity": capacity,
              "failed": [k for k, v in ledger.items() if v["status"] != "complete"],
              "notes": ["COCO is a detection-quality proxy; traffic-domain detection and occupancy accuracy remain unvalidated.",
                        "Looped clips are compute-load tests. Counting requires reviewed non-looped source labels.",
                        "No production capacity claim; same-session repeated replay on assigned Kaggle hardware.",
                        "Model-only reference excludes decode/tracking, uses the first clip and is not a hard ceiling."]}
    save(root / "study-summary.json", result)
    head = "# Kaggle multi-camera traffic study\n\n" + "\n".join("- " + n for n in result["notes"]) + "\n\n"
    if capacity:
        head += "## Largest passing tested loads\n\n| Case | Timing only | Timing + independently scored counting |\n|---|---:|---:|\n"
        for item in capacity:
            timing = item["largest_tested_timing_pass"]
            counted = item["largest_tested_timing_and_counting_pass"]
            head += f"| {item['case']} | {timing if timing is not None else 'none established'} | {counted if counted is not None else 'none established'} |\n"
        head += "\n"
    table = "| Case | Streams | Variant | Repeats | Min-camera FPS median [range] | Worst p95 ms median [range] | Timing all-pass | Counting |\n|---|---:|---|---:|---:|---:|---|---|\n"
    for row in rows:
        table += (f"| {row['case']} | {row['streams']} | {row['variant']} | {row['repeats']} | "
                  f"{row['min_camera_fps_median']:.2f} {row['min_camera_fps_range']} | "
                  f"{row['worst_p95_ms_median']:.2f} {row['worst_p95_ms_range']} | "
                  f"{row['all_repeats_timing_passed']} | {row['counting_status']} |\n")
    (root / "REPORT.md").write_text(head + table, encoding="utf-8")
    # Standalone results viewer: no externally hosted service or network dependency.
    body = ["<!doctype html><meta charset='utf-8'><title>Traffic study</title>",
            "<style>body{font:15px system-ui;margin:2rem;background:#101827;color:#e5edf7}table{border-collapse:collapse;width:100%}td,th{padding:.6rem;border:1px solid #41516b}video{max-width:650px}a{color:#8ac7ff}</style>",
            "<h1>Kaggle traffic analytics study</h1>", "<p>Fresh assigned-hardware results; read limitations before interpreting capacity.</p>",
            "<ul>" + "".join("<li>"+html.escape(n)+"</li>" for n in result["notes"]) + "</ul>",
            "<h2>Largest passing tested loads</h2><pre>"+html.escape(json.dumps(capacity, indent=2))+"</pre>",
            "<table><tr>"+"".join("<th>"+html.escape(k)+"</th>" for k in (rows[0] if rows else {}))+"</tr>"]
    for row in rows:
        body.append("<tr>" + "".join("<td>"+html.escape(str(v))+"</td>" for v in row.values()) + "</tr>")
    body.append("</table>")
    for video in root.rglob("*-browser.mp4"):
        rel = video.relative_to(root).as_posix()
        body.append(f"<h2>{html.escape(video.parent.name)}</h2><video controls src='{html.escape(rel, quote=True)}'></video>")
    for record in records:
        job = record["job"]
        folder = Path(job["artifact"]).parent
        raw = folder / "summary.json"
        if file_hash(raw) != record["summary_sha256"]:
            raise ValueError("Raw pipeline summary changed")
        summary = read(raw)
        stage_values, occupancy = collections.defaultdict(list), collections.defaultdict(list)
        log = folder / "frames.jsonl"
        if log.exists():
            with log.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    occupancy[row["camera"]].append(row["zone_occupancy"])
                    for key in ("decode_ms", "decoder_lag_ms", "queue_wait_ms", "batch_predict_ms",
                                "tracking_count_ms", "render_ms", "telemetry_ms"):
                        if key in row:
                            stage_values[key].append(row[key])
        body.append("<details><summary>" + html.escape(job["key"]) + " — camera statistics and stage profile</summary>")
        body.append("<p>Counting: " + html.escape(str(record["counting"])) + "</p>")
        body.append("<table><tr><th>Camera</th><th>Processed FPS</th><th>Dropped</th><th>Crossings (unvalidated unless scored)</th><th>Mean observed occupancy</th></tr>")
        for camera in summary["cameras"]:
            values = occupancy[camera["camera"]]
            mean_occupancy = sum(values)/len(values) if values else None
            body.append(f"<tr><td>{camera['camera']}</td><td>{camera['processed_fps']:.2f}</td>"
                        f"<td>{camera['queue_dropped']}</td><td>{html.escape(str(camera['counts']))}</td>"
                        f"<td>{mean_occupancy}</td></tr>")
        body.append("</table><p>Stage p50/p95 milliseconds; independent quantiles must not be summed:</p><ul>")
        for name, values in stage_values.items():
            body.append(f"<li>{html.escape(name)}: {precision.percentile(values, .5):.2f} / "
                        f"{precision.percentile(values, .95):.2f}</li>")
        body.append("</ul></details>")
    (root / "dashboard.html").write_text("\n".join(body), encoding="utf-8")
    return result


def archive(cfg):
    target = REPO.parent / f"{cfg['run_id']}-checkpoint.zip"
    folders = [REPO / "results" / cfg["run_id"], REPO / "models/precision-study", REPO / "models/yolo11n.pt",
               REPO / "data/traffic.mp4"]
    folders += [REPO / "data" / name for name in
                (cfg["run_id"], cfg["run_id"]+"-smoke", cfg["run_id"]+"-full")]
    with stage_progress(f"CHECKPOINT {target.name}"), zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as output:
        for folder in folders:
            files = folder.rglob("*") if folder.is_dir() else [folder]
            for path in files:
                if path.is_file() and path.name != "run.lock" and not any(part in ("images", "predictions") for part in path.relative_to(REPO).parts):
                    output.write(path, path.relative_to(REPO).as_posix())
    print(f"Checkpoint saved: {target} (image cache omitted; data can be re-downloaded)", flush=True)


def run(cfg):
    os.chdir(REPO)
    with stage_progress("ASSETS download/verification"):
        assets(cfg)
    preflight_result = preflight(cfg)
    bindings = bind_plan(cfg, preflight_result)
    with precision.exclusive_run(study_root(cfg)):
        try:
            if cfg["mode"] == "smoke":
                return 0 if smoke(cfg, bindings) else 1
            validate_smoke_gate(cfg, preflight_result)
            if cfg["phase"] in ("all", "models"):
                for index, size in enumerate(cfg["image_sizes"], 1):
                    log_event(f"MODEL STUDY: resolution {index}/{len(cfg['image_sizes'])}: {size}px")
                    mc = model_cfg(cfg, size)
                    with stage_progress(f"FULL DATA download/verification for {size}px"):
                        precision.prepare_data(mc)
                    with precision.exclusive_run(Path(mc["output"])):
                        status = precision.run(mc, 7200)
                    if status:
                        raise RuntimeError("Precision stage has failures; inspect logs before camera selection")
                    archive(cfg)
            if cfg["phase"] in ("all", "serving"):
                serving(cfg, bindings)
            if cfg["run_tuning"] and cfg["phase"] in ("all", "tuning"):
                serving(cfg, bindings, tuning=True)
            ledger_path = study_root(cfg) / "ledger.json"
            failed = [v for v in (read(ledger_path).values() if ledger_path.exists() else []) if v["status"] != "complete"]
            return int(bool(failed))
        finally:
            report(cfg)
            archive(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        os.chdir(REPO)
        process_job(read(args.worker))
        return 0
    if not args.config:
        parser.error("--config is required; use the exported Kaggle notebook")
    return run(configuration(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
