"""Write the portfolio findings directly from run artifacts, not hand-entered performance claims."""

import json
from pathlib import Path


def main():
    root = Path("results")

    def read(p):
        return json.loads(p.read_text(encoding="utf-8"))

    rows = []
    lines = [
        "# Measured findings",
        "",
        "Generated from the checked-in JSON artifacts. No estimated speedups.",
        "",
        "## Environment and scope",
        "",
    ]
    for path in sorted(root.glob("*.json")):
        d = read(path)
        if isinstance(d, dict) and d.get("kind") == "predecoded_model_benchmark":
            rows.append((path, d))
    if not rows:
        raise RuntimeError("No benchmark evidence")
    env = rows[0][1]["environment"]
    lines += [
        f"- OS: {env['platform']}",
        f"- CPU: {env['processor']}",
        f"- Memory: {env['ram_gib']:.2f} GiB; GPU inventory: {env['gpu']}",
        f"- PyTorch: {env['packages']['torch']}; ONNX Runtime: {env['packages']['onnxruntime']}; "
        f"OpenVINO: {env['packages']['openvino']}",
        "- All reported inference runs use CPU. CUDA/TensorRT and INT8 were not measured.",
        "- Shared Windows desktop, short clip, fixed experiment order; no isolation or thermal control.",
        "  Treat this as a preliminary runtime study, not a hardware leaderboard or production capacity certification.",
        "- Four PyTorch threads; ONNX/OpenVINO backend threading defaults. Resource budgets are not identical.",
        "",
        "## Runtime comparison",
        "",
        "Predecoded road video; batch one; five warm-up calls; three trials, 60 frames per trial.",
        "",
        "| Run | Mean FPS | p50 / p95 / p99 wall ms | COCO128 mAP50–95 |",
        "|---|---:|---:|---:|",
    ]
    for path, d in rows:
        evaluation = root / f"eval-{path.stem}.json"
        q = f"{read(evaluation)['map50_95']:.4f}" if evaluation.exists() else "not evaluated"
        t = d["batch_wall_ms"]
        lines.append(
            f"| {path.stem} | {d['fps']['mean']:.2f} | {t['p50']:.2f} / {t['p95']:.2f} / {t['p99']:.2f} | {q} |"
        )
    lines += [
        "",
        "COCO128 is a tiny general-purpose smoke set derived from training imagery, not an independent traffic holdout.",
        "The mAP values evaluate all classes; throughput uses the vehicle class filter.",
        "",
        "## Export agreement",
        "",
    ]
    for path in sorted(root.glob("parity-*.json")):
        d = read(path)
        matched = sum(r["matched"] for r in d["frames"])
        lines.append(
            f"- {path.stem}: **{'PASS' if d['passed'] else 'FAIL'}**, {len(d['frames'])} sampled frames, "
            f"{matched} matched detections (IoU >= {d['min_iou']}, confidence delta <= {d['max_conf_delta']})."
        )
    lines += ["", "## Explicit model acceptance gates", ""]
    for path in sorted(root.glob("gate-*.json")):
        gate = read(path)
        lines.append(
            f"- {path.stem}: **{'ACCEPT' if gate['passed'] else 'REJECT'}**. "
            f"p95 budget {gate['p95_budget_ms']} ms; maximum accuracy drop "
            f"{gate['max_map_drop_points']} absolute mAP points; measured drop "
            f"{gate['measured_map_drop_points']:.2f} points. Reasons: "
            f"{'; '.join(gate['reasons']) or 'within the model-only budgets'}."
        )
    lines += [
        "",
        "Agreement does not prove accuracy; deployment requires a labeled traffic holdout.",
        "",
        "## Multi-camera load and scheduling",
        "",
        "The same short clip is replayed per camera at source FPS. These are compute-load experiments, not diverse RTSP cameras.",
        "Results below are aggregated across cameras; p95 is pooled from the per-frame telemetry.",
        "",
        "| Experiment | Decoded / processed | Queue drops | Stride skips | Pooled p95 frame age ms |",
        "|---|---:|---:|---:|---:|",
    ]
    from traffic_lab.metrics import distribution

    profiles = []
    for path in sorted(root.glob("*/summary.json")):
        d = read(path)
        c = d["cameras"]
        log = path.parent / "frames.jsonl"
        samples = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        ages = [r["frame_age_ms"] for r in samples]
        profile = [path.parent.name]
        for key in ("decoder_lag_ms", "queue_wait_ms", "batch_predict_ms", "tracking_count_ms"):
            value = distribution([r[key] for r in samples])["p95"]
            profile.append(f"{value:.2f}" if value is not None else "no samples")
        profiles.append("| " + " | ".join(profile) + " |")
        p95 = distribution(ages)["p95"]
        age_text = f"{p95:.2f}" if p95 is not None else "no samples"
        lines.append(
            f"| {path.parent.name} | {sum(r['decoded'] for r in c)} / {sum(r['processed'] for r in c)} | "
            f"{sum(r['queue_dropped'] for r in c)} | {sum(r['stride_skipped'] for r in c)} | {age_text} |"
        )
    lines += [
        "",
        "### Locating the bottleneck",
        "",
        "| Experiment | p95 decoder lag ms | p95 queue wait ms | p95 prediction batch ms | p95 tracking/count ms |",
        "|---|---:|---:|---:|---:|",
        *profiles,
        "",
        "Percentiles are computed independently and must not be added. Decoder lag includes source-schedule backlog.",
        "A bounded inference queue cannot remove backlog already accumulated inside a sequential decoder.",
        "This distinction guides the next optimization: investigate decoder throughput when lag dominates, rather than",
        "assuming another model export will fix end-to-end latency.",
    ]
    lines += [
        "",
        "Drops, stride skips, processing rate and frame age must be considered together. A fast result after dropping most inputs",
        "does not meet a high processing-rate requirement. Counts remain unvalidated. The BoT-SORT run also renders video, so it",
        "is a functional demonstration rather than a controlled tracker-speed comparison.",
        "",
        "## What to say about this project",
        "",
        "Implemented and measured a YOLO deployment pipeline across PyTorch, ONNX Runtime and OpenVINO, with export parity checks,",
        "per-camera tracking, bounded queues, explicit frame-drop accounting, and reproducible latency/quality artifacts.",
        "Compare the tables when discussing measured tradeoffs; do not turn this smoke study into an unqualified speedup or accuracy claim.",
        "",
        "## Remaining validation",
        "",
        "1. Longer, varied traffic footage and repeated run orders under stable thermal conditions.",
        "2. Independent labeled traffic frames and crossing-event annotations; quantify misses and duplicate counts.",
        "3. Sustained paced-stream acceptance against an agreed latency, processing-rate and quality budget.",
        "4. Optional CUDA/TensorRT and calibrated INT8 experiments on the deployment hardware.",
        "",
        "See [methodology](METHODOLOGY.md) and [raw results](../results/).",
    ]
    ledger = root / "experiment-status.json"
    if ledger.exists():
        statuses = read(ledger)
        passed = sum(s["exit_code"] == 0 for s in statuses)
        retries = sum(len(s.get("previous_attempts", [])) for s in statuses)
        lines += [
            "",
            "## Execution audit",
            "",
            f"Final experiment ledger: {passed}/{len(statuses)} commands successful. "
            f"Retained retry records: {retries}.",
            "Initial tracker-constructor failures were corrected and rerun. Original failure logs and attempt history",
            "are preserved beside the successful measurements. Run `python scripts/validate_artifacts.py` to audit",
            "frame accounting, queue bounds, raw benchmark frame counts, and parity gates.",
        ]
    Path("docs/RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
