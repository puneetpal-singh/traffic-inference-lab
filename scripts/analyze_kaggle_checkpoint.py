"""Audit an offline Kaggle checkpoint and derive a report without loading models or running inference."""

import argparse
import ast
import collections
import hashlib
import io
import json
import math
import statistics
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

from traffic_lab.confirmation import json_digest, percentile
from traffic_lab.precision import randomized_rounds, validate_timing

PREFIX = "/kaggle/working/traffic-inference-lab/"
FULL = "results/traffic-kaggle-v1/full/"
MODELS = FULL + "precision-320/"


class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(path)
        self.names = set(self.zip.namelist())
        self.hashes = {}

    def member(self, path):
        value = str(path)
        return value[len(PREFIX):] if value.startswith(PREFIX) else value

    def read(self, path):
        return json.loads(self.zip.read(self.member(path)))

    def sha(self, path):
        member = self.member(path)
        if member not in self.hashes:
            if member in self.names:
                self.hashes[member] = hashlib.sha256(self.zip.read(member)).hexdigest()
            else:
                prefix = member.rstrip("/") + "/"
                children = sorted(n for n in self.names if n.startswith(prefix) and not n.endswith("/"))
                if not children:
                    raise FileNotFoundError(member)
                digest = hashlib.sha256()
                for name in children:
                    digest.update(name[len(prefix):].encode())
                    digest.update(bytes.fromhex(self.sha(name)))
                self.hashes[member] = digest.hexdigest()
        return self.hashes[member]


def near(actual, expected):
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def paired(candidate, baseline):
    rounds = sorted(set(candidate) & set(baseline))
    logs = [math.log(candidate[i] / baseline[i]) for i in rounds]
    if len(logs) != 10:
        raise ValueError("This saved study requires ten complete rounds")
    center = statistics.mean(logs)
    margin = 2.2621571628540993 * statistics.stdev(logs) / math.sqrt(10)  # t(.975, df=9)
    return {"rounds": 10, "geometric_mean_ratio": math.exp(center),
            "exploratory_log_t_95_interval": [math.exp(center-margin), math.exp(center+margin)],
            "first_five_geomean": math.exp(statistics.mean(logs[:5])),
            "last_five_geomean": math.exp(statistics.mean(logs[5:])),
            "limitation": "Single session, approximately independent normal block log ratios assumed; not a multi-day CI"}


def audit(path, output):
    archive = Archive(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    errors, checks = [], 0

    def require(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    require(all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts
                and "\\" not in n for n in archive.names), "Unsafe archive member")
    print("Checking ZIP CRCs...", flush=True)
    require(archive.zip.testzip() is None, "ZIP CRC failure")
    full_plan, plan = archive.read(FULL+"plan.json"), archive.read(MODELS+"plan.json")
    cfg, bindings = plan["config"], plan["bindings"]
    manifest = archive.read("data/traffic-kaggle-v1-full/manifest.json")
    calibration = {r["id"] for r in manifest["images"] if r["split"] == "calibration"}
    evaluation = {r["id"] for r in manifest["images"] if r["split"] == "evaluation"}
    require(len(calibration) == 512 and len(evaluation) == 4488 and not calibration & evaluation, "Data split invalid")
    require(len({r["id"] for r in manifest["images"]}) == 5000, "Dataset IDs duplicated")
    require(json_digest(manifest) == bindings["dataset_sha256"], "Dataset manifest identity mismatch")
    require(archive.sha("data/traffic-kaggle-v1-full/instances_val2017.json") == manifest["annotation_sha256"], "Annotation hash mismatch")
    require(archive.sha("data/traffic-kaggle-v1-full/calibration.yaml") == manifest["calibration_yaml_sha256"], "Calibration YAML hash mismatch")
    require(archive.sha(cfg["model"]) == bindings["model_sha256"], "Base model hash mismatch")
    require(archive.sha(cfg["source"]) == bindings["source_sha256"], "Video hash mismatch")
    require(plan["orders"] == randomized_rounds(plan["cases"], cfg["trials"], cfg["seed"]), "Randomized schedule changed")
    require(cfg["imgsz"] == 320 and cfg["trials"] == 10 and cfg["seconds"] >= 20, "Unexpected protocol")

    # Match the archived source fingerprints to the exact committed, embedded notebook source.
    try:
        notebook = json.loads(subprocess.check_output(
            ["git", "show", "795d811:notebooks/kaggle/Traffic_Analytics_Kaggle.ipynb"], text=True))
        import base64

        source = None
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse(cell["source"]).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SOURCE_BUNDLE_BASE64":
                    source = base64.b64decode(ast.literal_eval(node.value))
        with zipfile.ZipFile(io.BytesIO(source)) as embedded:
            for name, digest in full_plan["bindings"]["code"].items():
                require(hashlib.sha256(embedded.read(name)).hexdigest() == digest, f"Historical source mismatch: {name}")
        source_verified = True
    except (subprocess.CalledProcessError, KeyError, TypeError, ValueError) as exc:
        source_verified = False
        errors.append(f"Historical source audit unavailable: {exc}")

    ledger = archive.read(MODELS+"ledger.json")
    require(len(ledger) == 165 and all(v["status"] == "complete" for v in ledger.values()), "Model stage coverage incomplete")
    for key, entry in ledger.items():
        require(archive.sha(entry["artifact"]) == entry["artifact_sha256"], f"Model artifact hash: {key}")
        job = archive.read(MODELS+"jobs/"+key+".json")
        dependency = dict(bindings)
        if job["stage"] != "export":
            dependency["artifact_sha256"] = archive.read(entry["artifact"]).get("model_sha256")
            if job["stage"] == "inspect":
                case = job["case"]
                dependency["artifact_sha256"] = archive.sha(cfg["model"]) if case["backend"] == "pytorch" else archive.read(
                    MODELS+"exports/"+case["export_key"]+"/export.json")["artifact_sha256"]
        expected = json_digest({"job": {k: v for k, v in job.items() if k != "identity"}, "bindings": dependency})
        require(expected == entry["identity"] == job["identity"], f"Model job identity mismatch: {key}")
        if job["stage"] == "export":
            record = archive.read(entry["artifact"])
            require(archive.sha(record["artifact"]) == record["artifact_sha256"], f"Exported model changed: {key}")
            require(record["source_sha256"] == bindings["model_sha256"], f"Export source changed: {key}")

    print("Auditing 130 timing trials and 13 accuracy records...", flush=True)
    timings, qualities = {}, {}
    for case in plan["cases"]:
        name = case["name"]
        trials = []
        model_sha = (bindings["model_sha256"] if case["backend"] == "pytorch" else archive.read(
            MODELS+"exports/"+case["export_key"]+"/export.json")["artifact_sha256"])
        for index in range(10):
            key = f"timing-{name}-round-{index:02d}"
            row = archive.read(MODELS+"timing/"+key+".json")
            try:
                validate_timing(row, cfg)
            except ValueError as exc:
                errors.append(f"{key}: {exc}")
            require(row["round"] == index and row["model_sha256"] == model_sha, f"Trial model/round mismatch: {key}")
            require(row["job_identity"] == ledger[key]["identity"], f"Trial job binding mismatch: {key}")
            trials.append(row)
        timings[name] = trials
        quality = archive.read(MODELS+"quality/quality-"+name+".json")
        require(quality["model_sha256"] == model_sha and quality["images"] == 4488, f"Quality identity: {name}")
        values = [v for v in quality["per_category_ap"].values() if v is not None]
        require(near(quality["map50_95"], statistics.mean(values)), f"Global AP arithmetic: {name}")
        require(near(quality["vehicle_ap"], statistics.mean(quality["per_category_ap"][str(c)] for c in (3, 4, 6, 8))), f"Vehicle AP arithmetic: {name}")
        qualities[name] = quality

    summary = archive.read(MODELS+"summary.json")
    model_rows = summary["rows"]
    for row in model_rows:
        name, trials = row["case"], timings[row["case"]]
        require(near(row["fps_median"], statistics.median(t["fps"] for t in trials)), f"Median FPS mismatch: {name}")
        require(near(row["trial_p95_median_ms"], statistics.median(t["p95_ms"] for t in trials)), f"Median p95 mismatch: {name}")
        require(near(row["map50_95"], qualities[name]["map50_95"]), f"Summary AP mismatch: {name}")
        require(near(row["loss_vs_pytorch_points"], 100*(qualities["pytorch-cuda-fp32"]["map50_95"]-qualities[name]["map50_95"])), f"AP loss mismatch: {name}")
    comparison_pairs = [("tensorrt-cuda-fp32", "pytorch-cuda-fp32"),
                        ("tensorrt-cuda-fp16", "pytorch-cuda-fp32"),
                        ("tensorrt-cuda-fp16", "tensorrt-cuda-fp32"),
                        ("tensorrt-cuda-int8", "tensorrt-cuda-fp16"),
                        ("openvino-cpu-int8", "openvino-cpu-fp32")]
    comparisons = {a+" / "+b: paired({r["round"]: r["fps"] for r in timings[a]},
                                     {r["round"]: r["fps"] for r in timings[b]}) for a, b in comparison_pairs}

    print("Auditing 72 serving runs and per-frame logs...", flush=True)
    serving_ledger, serving_rows, profiles = archive.read(FULL+"ledger.json"), [], []
    require(len(serving_ledger) == 72 and all(v["status"] == "complete" for v in serving_ledger.values()), "Serving coverage incomplete")
    for key, entry in serving_ledger.items():
        require(archive.sha(entry["artifact"]) == entry["sha256"], f"Serving artifact hash: {key}")
        record = archive.read(entry["artifact"])
        job = archive.read(FULL+"jobs/"+key+".json")
        require(job == record["job"], f"Serving job differs: {key}")
        case = job["case"]
        model_sha = qualities[case["name"]]["model_sha256"]
        dependency = {**full_plan["bindings"], "engine": model_sha, "replay_hashes": [archive.sha(p) for p in job["sources"]],
                      "model_sha256": model_sha}
        expected = json_digest({"job": {k: v for k, v in job.items() if k != "identity"}, "bindings": dependency})
        require(expected == entry["identity"] == job["identity"], f"Serving job identity: {key}")
        folder = str(PurePosixPath(archive.member(entry["artifact"])).parent) + "/"
        raw = archive.read(folder+"summary.json")
        require(archive.sha(folder+"summary.json") == record["summary_sha256"], f"Raw summary identity: {key}")
        require(raw["model_sha256"] == model_sha and raw["requested_seconds"] == 30 and raw["status"] == "complete", f"Serving protocol: {key}")
        by_camera = collections.defaultdict(list)
        stage = collections.defaultdict(list)
        for line in archive.zip.open(folder+"frames.jsonl"):
            frame = json.loads(line)
            by_camera[frame["camera"]].append(frame)
            for field in ("decode_ms", "decoder_lag_ms", "queue_wait_ms", "batch_predict_ms", "tracking_count_ms", "render_ms", "telemetry_ms", "frame_age_ms"):
                stage[field].append(frame[field])
        for camera in raw["cameras"]:
            frames = by_camera[camera["camera"]]
            require(camera["processed"] == len(frames), f"Frame accounting: {key}/{camera['camera']}")
            require(camera["decoded"] == sum(camera[k] for k in ("processed", "queue_dropped", "stride_skipped", "remaining")), f"Drop accounting: {key}")
            require(camera["decoded"] == 900 and camera["queue_high_water"] <= 1 and camera["decoder_threads"] == 1, f"Replay/queue/threads: {key}")
            require(near(camera["processed_fps"], len(frames)/raw["elapsed_seconds"]), f"Camera FPS mismatch: {key}")
            require(near(camera["frame_age_ms"]["p95"], percentile([f["frame_age_ms"] for f in frames], .95)), f"Camera p95 mismatch: {key}")
        rates = [c["processed_fps"] for c in raw["cameras"]]
        ages = [c["frame_age_ms"]["p95"] for c in raw["cameras"]]
        require(near(record["min_camera_fps"], min(rates)) and near(record["worst_camera_p95_ms"], max(ages)), f"Worst-camera gate arithmetic: {key}")
        require(record["timing_passed"] == (min(rates) >= 15 and max(ages) <= 100), f"Timing gate mismatch: {key}")
        require(record["counting"]["status"] == "unvalidated", f"Unexpected counting claim: {key}")
        serving_rows.append(record)
        profiles.append({"case": case["name"], "streams": job["streams"], "repeat": job["repeat"],
                         "aggregate_fps": sum(rates), "processed": sum(c["processed"] for c in raw["cameras"]),
                         "elapsed_seconds": raw["elapsed_seconds"],
                         "stages": {field: {"mean": statistics.mean(values), "p50": percentile(values, .5), "p95": percentile(values, .95)}
                                    for field, values in stage.items()},
                         "gpu_utilization": raw["resources"]["metrics"].get("gpu_utilization_percent")})

    serving_summary = archive.read(FULL+"study-summary.json")
    for row in serving_summary["rows"]:
        selected = [r for r in serving_rows if r["job"]["case"]["name"] == row["case"] and r["job"]["streams"] == row["streams"]]
        require(len(selected) == 3, f"Serving repeat count: {row['case']}/{row['streams']}")
        require(near(row["min_camera_fps_median"], statistics.median(r["min_camera_fps"] for r in selected)), "Serving FPS median mismatch")
        require(near(row["worst_p95_ms_median"], statistics.median(r["worst_camera_p95_ms"] for r in selected)), "Serving p95 median mismatch")
        require(row["all_repeats_timing_passed"] == all(r["timing_passed"] for r in selected), "Serving all-repeat gate mismatch")

    original_hash = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            original_hash.update(block)
    result = {"passed": not errors, "checks": checks, "errors": errors, "archive_sha256": original_hash.hexdigest(),
              "archive_bytes": Path(path).stat().st_size, "source_payload_commit": "795d811", "source_verified": source_verified,
              "model_stages": len(ledger), "timing_trials": sum(map(len, timings.values())), "serving_runs": len(serving_ledger),
              "calibration_images": len(calibration), "evaluation_images": len(evaluation),
              "flagged_model_trials": sum(bool(r["flags"]) for records in timings.values() for r in records),
              "coverage_note": "320 complete; 640 intentionally incomplete; no tuning phase present",
              "limitations": ["Audits identities/arithmetic, not independent reproduction of measured performance.",
                              "Image cache and per-image predictions omitted from checkpoint: AP cannot be independently recomputed or bootstrapped from this archive.",
                              "Counting labels absent; looped short video tests compute capacity only.",
                              "One assigned Kaggle host/session; baseline PyTorch eager with cuDNN autotuning disabled."]}
    flagged = [{"case": name, "round": row["round"], "flags": row["flags"]}
               for name, records in timings.items() for row in records if row["flags"]]
    result["flagged_trials"] = flagged
    (output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (output / "stage-profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    # Preserve the authoritative aggregate JSONs unchanged alongside the derived analysis.
    for name, member in (("precision-320-summary.json", MODELS+"summary.json"), ("serving-summary.json", FULL+"study-summary.json"),
                         ("selections.json", FULL+"selections.json"), ("precision-plan.json", MODELS+"plan.json")):
        (output / name).write_bytes(archive.zip.read(member))
    write_report(output, result, model_rows, serving_summary, comparisons, profiles)
    print(json.dumps(result, indent=2))
    return result


def write_report(output, audit_result, models, serving, comparisons, profiles):
    lines = ["# Kaggle T4 traffic inference study — verified checkpoint analysis", "",
             f"Offline artifact audit: **{'PASS' if audit_result['passed'] else 'FAIL'}**, {audit_result['checks']} checks.",
             "No inference, engine builds or new performance experiments were run for this analysis.", "",
             "## Completed scope", "", "- YOLO11n at 320px; one T4, 2 physical / 4 logical CPU cores, 31.35 GiB RAM.",
             "- 13 runtime/device/precision cases, 10 randomized >=20s timing trials each; 512 calibration and 4488 disjoint COCO evaluation images.",
             "- 72 serving runs: six chosen backend/device configurations × 1/2/4/8 cameras × three 30-second replays.",
             "- 640px was stopped; tuning and independent counting validation are not completed.", "",
             "## Model results", "", "FPS is median trial throughput; AP losses are absolute percentage points against PyTorch CUDA FP32.", "",
             "| Case | Median FPS [min, max] | Median trial p95 ms | mAP50–95 | Global / vehicle AP loss (points) | Model MiB |",
             "|---|---:|---:|---:|---:|---:|"]
    reference_vehicle = next(r["vehicle_ap"] for r in models if r["case"] == "pytorch-cuda-fp32")
    for r in models:
        lines.append(f"| {r['case']} | {r['fps_median']:.2f} [{r['fps_min']:.2f}, {r['fps_max']:.2f}] | "
                     f"{r['trial_p95_median_ms']:.2f} | {r['map50_95']:.4f} | {r['loss_vs_pytorch_points']:.4f} / "
                     f"{100*(reference_vehicle-r['vehicle_ap']):.4f} | {r['artifact_mib']:.2f} |")
    lines += ["", "## Paired randomized-round throughput comparisons", "",
              "Exploratory log-ratio t-intervals over ten rounds. They assume approximately independent normal log ratios; one session does not establish multi-day repeatability.", "",
              "| Candidate / baseline | Geometric-mean ratio | Exploratory 95% interval | First / last five rounds |",
              "|---|---:|---:|---:|"]
    for name, r in comparisons.items():
        lo, hi = r["exploratory_log_t_95_interval"]
        lines.append(f"| {name} | {r['geometric_mean_ratio']:.3f}× | [{lo:.3f}, {hi:.3f}] | {r['first_five_geomean']:.3f} / {r['last_five_geomean']:.3f} |")
    lines += ["", "## Interpretation", "",
              "- TensorRT FP16 is a strong model-level candidate: approximately the same throughput as INT8 with a much smaller measured AP loss.",
              "- The full TensorRT gain over PyTorch is a runtime/configuration comparison, not a speedup caused solely by FP16.",
              "- Auto-selection chose TensorRT INT8 because its median p95 was marginally lower. This is not proof it is a better deployment tradeoff than FP16.",
              "- CPU ONNX INT8 was slower than ONNX FP32 and lost substantial AP in this recipe. Quantization is not universally beneficial.",
              "- OpenVINO FP16 denotes storage compression, and TensorRT flags allow mixed precision; refer to retained runtime inspection.", "",
              f"- {audit_result['flagged_model_trials']} timing trial(s) were flagged and retained: " +
              "; ".join(f"{r['case']} round {r['round']}: {', '.join(r['flags'])}" for r in audit_result.get("flagged_trials", [])) + ".", "",
              "## Serving results", "", "Passing requires every camera to sustain >=15 FPS and <=100ms p95 frame age in all three repeats. Rendering was off in these baseline capacity runs.", "",
              "| Case | Cameras | Min-camera FPS median | Worst p95 ms median [range] | All three timing pass |",
              "|---|---:|---:|---:|---|"]
    for r in serving["rows"]:
        low, high = r["worst_p95_ms_range"]
        lines.append(f"| {r['case']} | {r['streams']} | {r['min_camera_fps_median']:.2f} | {r['worst_p95_ms_median']:.2f} [{low:.2f}, {high:.2f}] | {r['all_repeats_timing_passed']} |")
    lines += ["", "## TensorRT INT8 pipeline profile", "",
              "Shown for repeat 0; full profiles for all 72 runs are in stage-profiles.json. Decode time includes source read; decoder lag measures lateness against scheduled arrival. Quantiles must not be summed.", "",
              "| Cameras | Aggregate FPS | Decode p95 ms | Decoder-lag p95 ms | Queue p95 ms | Predict p95 ms | Tracking p95 ms | Telemetry p95 ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in sorted((p for p in profiles if p["case"] == "tensorrt-cuda-int8" and p["repeat"] == 0), key=lambda p: p["streams"]):
        s = p["stages"]
        lines.append(f"| {p['streams']} | {p['aggregate_fps']:.2f} | " + " | ".join(f"{s[k]['p95']:.2f}" for k in
                     ("decode_ms", "decoder_lag_ms", "queue_wait_ms", "batch_predict_ms", "tracking_count_ms", "telemetry_ms")) + " |")
    lines += ["", "## Scope limits and next experiment", "",
              "- Two cameras are the largest passing tested GPU load under these replay conditions. This is timing-only capacity; counting remains unvalidated.",
              "- Multiple cameras repeat the same short recording. Independent views, RTSP/network behaviour, long-running reliability and human crossing labels remain future work.",
              "- Large multi-camera frame ages coexist with bounded queues because the decoder can deliver frames that are already late. Profile decoder pacing and CPU work before changing the detector.",
              "- A useful next comparison is TensorRT FP16 versus INT8 at two/four streams under the same protocol; FP16 was not selected for these full serving runs.",
              "- AP uncertainty is unmeasured: predictions/image caches were deliberately excluded from the archive. Avoid claims based on tiny AP differences.",
              "- The archive includes real trial/frame evidence, but this audit is not an independent rerun and cannot establish traffic counting correctness.",
              "", "## Figures", "", "Figures use the saved aggregates; error bars show trial/repeat min/max, not confidence intervals.",
              "", "![Runtime throughput](runtime-throughput.png)", "", "![Camera scaling](camera-scaling.png)"]
    (output / "Kaggle_Results_Analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive, args.output)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
