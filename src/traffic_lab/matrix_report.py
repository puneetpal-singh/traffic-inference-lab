"""Report only directly paired full-matrix timing and held-out accuracy artifacts."""

import csv
import html
import json
from pathlib import Path

from .metrics import pareto_indices


def build_matrix_report(root, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, output = Path(root), Path(output)
    plan = json.loads((root / "plan.json").read_text())
    ledger = json.loads((root / "ledger.json").read_text())
    rows, missing = [], []
    for spec in plan:
        path = root / f"{spec['name']}.json"
        if not path.exists():
            missing.append(spec["name"])
            continue
        result = json.loads(path.read_text())
        timing = json.loads((root / f"timing-{spec['name']}.json").read_text())
        quality = json.loads((root / f"quality-{spec['name']}.json").read_text())
        if timing["model_sha256"] != quality["model_sha256"]:
            raise ValueError(f"Mismatched artifacts: {spec['name']}")
        model = Path(spec["model"])
        size = timing.get("model_size_bytes")
        if size is None and model.exists():
            size = (
                sum(p.stat().st_size for p in model.rglob("*") if p.is_file())
                if model.is_dir()
                else model.stat().st_size
            )
        resources = timing.get("resources", {}).get("metrics", {})
        stages = {
            key: sum(s["stages_per_image_ms"][key] for s in timing["samples"]) / len(timing["samples"])
            for key in ("preprocess", "inference", "postprocess")
        }
        per_class = quality.get("per_class_map50_95", {})
        vehicle_scores = [per_class[str(c)] for c in (2, 3, 5, 7) if str(c) in per_class]
        rows.append(
            {
                "name": spec["name"],
                "backend": spec["backend"],
                "precision": spec["precision"],
                "device": "CPU" if spec["device"] == "cpu" else "CUDA",
                "imgsz": spec["imgsz"],
                "fps": result["throughput_fps"],
                "p50_ms": timing["batch_wall_ms"]["p50"],
                "p95_ms": result["p95_ms"],
                "p99_ms": result["p99_ms"],
                "mean_preprocess_ms": stages["preprocess"],
                "mean_forward_ms": stages["inference"],
                "mean_postprocess_ms": stages["postprocess"],
                "mean_wrapper_and_transfer_residual_ms": timing["batch_wall_ms"]["mean"]
                - sum(stages.values()),
                "map50_95": result["map50_95"],
                "map50": result["map50"],
                "vehicle_map50_95": sum(vehicle_scores) / 4 if len(vehicle_scores) == 4 else None,
                "model_mib": size / 1024**2 if size is not None else None,
                "process_rss_mib": timing["peak_sampled_rss_mib"],
                "gpu_used_mib": resources.get("gpu_used_mib", {}).get("max"),
                "gpu_util_percent": resources.get("gpu_utilization_percent", {}).get("mean"),
                "gpu_temperature_c": resources.get("gpu_temperature_c", {}).get("max"),
                "gpu_power_w": resources.get("gpu_power_w", {}).get("mean"),
                "strict_parity": result["strict_detection_parity_passed"],
                "export_process_seconds": ledger.get("export-" + spec["export_key"], {}).get("seconds"),
            }
        )
    for row in rows:
        baseline = next((r for r in rows if r["name"] == f"pytorch-cpu-fp32-{row['imgsz']}"), None)
        row["map_drop_points"] = 100 * (baseline["map50_95"] - row["map50_95"]) if baseline else None
        row["model_gate_passed"] = (
            row["map_drop_points"] is not None and row["map_drop_points"] <= 1 and row["p95_ms"] <= 100
        )
    root.mkdir(parents=True, exist_ok=True)
    if rows:
        with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.09, wspace=0.55, hspace=0.45)
    for row_index, size in enumerate((320, 640)):
        for col_index, device in enumerate(("CPU", "CUDA")):
            ax = axes[row_index, col_index]
            group = [r for r in rows if r["imgsz"] == size and r["device"] == device]
            labels = [
                f"{r['backend']} {'INT8' if r['precision'] == 8 else 'FP' + str(r['precision'])}"
                for r in group
            ]
            ax.barh(
                labels,
                [r["p95_ms"] for r in group],
                color=["#18a788" if r["model_gate_passed"] else "#b9815a" for r in group],
            )
            ax.invert_yaxis()
            ax.set_title(f"{device} · {size}px · model p95 latency")
            ax.set_xlabel("Milliseconds; green meets model-only latency/accuracy gate")
            for i, result in enumerate(group):
                ax.text(result["p95_ms"], i, f" {result['p95_ms']:.1f}", va="center", fontsize=8)
            ax.margins(x=0.18)
            common_limit = max((r["p95_ms"] for r in rows if r["imgsz"] == size), default=1) * 1.2
            ax.set_xlim(0, common_limit)
    image = root / "latency-comparison.png"
    fig.savefig(image, dpi=140, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.subplots_adjust(left=0.09, right=0.68, bottom=0.08, top=0.94, hspace=0.4)
    for ax, device in zip(axes, ("CPU", "CUDA")):
        group = [r for r in rows if r["device"] == device]
        frontier = pareto_indices(group, "p95_ms", "map50_95")
        for i, row in enumerate(group):
            ax.scatter(
                row["p95_ms"],
                row["map50_95"],
                label=row["name"].replace(f"-{device.lower()}", ""),
                marker="o" if row["imgsz"] == 320 else "^",
                s=65,
                edgecolors="black" if i in frontier else "none",
            )
        ax.set_title(f"{device} accuracy–latency tradeoffs")
        ax.set_xlabel("p95 model wall time (ms)")
        ax.set_ylabel("mAP50–95, 256 held-out COCO val2017 images")
        if group:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    frontier_image = root / "quality-latency.png"
    fig.savefig(frontier_image, dpi=140, bbox_inches="tight")
    plt.close(fig)

    def number(value, digits=2):
        return "N/A" if value is None else f"{value:.{digits}f}"

    def table(group):
        header = (
            "<tr><th>Configuration</th><th>FPS</th><th>p50 / p95 / p99 ms</th><th>mAP50–95</th><th>Vehicle mAP50–95</th>"
            "<th>Δ mAP points</th><th>Model MiB</th><th>Peak sampled GPU MiB</th><th>Gate</th></tr>"
        )
        body = []
        for r in group:
            values = (
                r["name"],
                number(r["fps"]),
                f"{r['p50_ms']:.2f} / {r['p95_ms']:.2f} / {r['p99_ms']:.2f}",
                number(r["map50_95"], 4),
                number(r["vehicle_map50_95"], 4),
                number(r["map_drop_points"]),
                number(r["model_mib"]),
                number(r["gpu_used_mib"], 0),
                "PASS" if r["model_gate_passed"] else "REJECT",
            )
            body.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
        return '<div class="scroll"><table>' + header + "".join(body) + "</table></div>"

    output.parent.mkdir(parents=True, exist_ok=True)
    import base64

    latency_b64 = base64.b64encode(image.read_bytes()).decode()
    frontier_b64 = base64.b64encode(frontier_image.read_bytes()).decode()
    failed = [k for k, v in ledger.items() if v["exit_code"]]
    pipeline_rows = []
    for path in sorted((root / "pipelines").glob("*/summary.json")):
        pipeline = json.loads(path.read_text())
        cameras = pipeline["cameras"]
        ages = [c["frame_age_ms"]["p95"] for c in cameras]
        fps = min(c["processed_fps"] for c in cameras)
        worst_age = max(ages) if all(a is not None for a in ages) else None
        pipeline_rows.append(
            {
                "experiment": path.parent.name,
                "backend": pipeline["backend"],
                "streams": len(cameras),
                "batch": pipeline["requested_batch"],
                "stride": pipeline["stride"],
                "tracker": pipeline["tracker"],
                "render": pipeline["render"],
                "decoded": sum(c["decoded"] for c in cameras),
                "processed": sum(c["processed"] for c in cameras),
                "queue_dropped": sum(c["queue_dropped"] for c in cameras),
                "stride_skipped": sum(c["stride_skipped"] for c in cameras),
                "min_camera_fps": fps,
                "worst_camera_p95_ms": worst_age,
                "timing_gate_passed": pipeline["status"] == "complete"
                and worst_age is not None
                and worst_age <= 100
                and fps >= 15,
                "status": pipeline["status"],
            }
        )
    if pipeline_rows:
        with (root / "pipeline-comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pipeline_rows[0]))
            writer.writeheader()
            writer.writerows(pipeline_rows)
    serving_table = (
        "<div class='scroll'><table><tr><th>Experiment</th><th>Decoded / processed</th>"
        "<th>Queue drops</th><th>Stride skips</th><th>Minimum camera FPS</th>"
        "<th>Worst camera p95 ms</th><th>Timing gate</th></tr>"
    )
    for r in pipeline_rows:
        values = (
            r["experiment"],
            f"{r['decoded']} / {r['processed']}",
            r["queue_dropped"],
            r["stride_skipped"],
            number(r["min_camera_fps"]),
            number(r["worst_camera_p95_ms"]),
            "PASS" if r["timing_gate_passed"] else "REJECT",
        )
        serving_table += "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>"
    serving_table += "</table></div>"
    resource_table = (
        "<div class='scroll'><table><tr><th>CUDA configuration</th><th>Sampled process RSS MiB</th>"
        "<th>Peak sampled GPU MiB</th><th>Mean GPU utilization %</th><th>Max temperature °C</th>"
        "<th>Mean reported GPU power W</th></tr>"
    )
    for r in rows:
        if r["device"] != "CUDA":
            continue
        values = (
            r["name"],
            number(r["process_rss_mib"], 0),
            number(r["gpu_used_mib"], 0),
            number(r["gpu_util_percent"], 1),
            number(r["gpu_temperature_c"], 1),
            number(r["gpu_power_w"], 1),
        )
        resource_table += "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>"
    resource_table += "</table></div>"
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Traffic Inference Lab — CPU, CUDA and TensorRT</title>
<style>body{{font:15px system-ui;max-width:1400px;margin:40px auto;padding:24px;background:#0d1725;color:#e4edf6}}
p{{line-height:1.65;color:#b8c9dc}}h1{{font-size:42px}}h2{{margin-top:40px}}img{{width:100%;background:white;border-radius:10px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #304257;text-align:left;white-space:nowrap}}
.scroll{{overflow-x:auto}}.tag{{color:#44d6b1;letter-spacing:3px}}</style>
<p class="tag">PROFILE · CALIBRATE · VERIFY · BENCHMARK</p><h1>Traffic Inference Lab</h1>
<p>{len(rows)}/{len(plan)} complete configurations. PyTorch, ONNX Runtime, OpenVINO and TensorRT at 320 and 640 pixels.
CPU and CUDA use the same Python environment, checkpoint, inputs, square letterboxing and thresholds.
Timing: 120 predecoded traffic frames × 5 trials, 20 warm-up calls. Accuracy: 256 COCO val2017 holdout images.
INT8 calibration: 512 separate images; no overlap with evaluation.</p>
<img alt="Measured p95 latency by backend, precision, device, and resolution" src="data:image/png;base64,{latency_b64}">
<h2>CPU configurations</h2>{table([r for r in rows if r["device"] == "CPU"])}
<h2>CUDA configurations</h2>{table([r for r in rows if r["device"] == "CUDA"])}
<p>Gate: p95 model wall time ≤100 ms and loss ≤1 absolute mAP point relative to PyTorch CPU FP32 at the same resolution.
This is a model-only acceptance rule; it does not certify a live-camera SLA or counting accuracy.</p>
<h2>Accuracy and latency</h2><img alt="Measured accuracy versus latency" src="data:image/png;base64,{frontier_b64}">
<p>OpenVINO FP16 compresses stored constants; runtime precision is separately inspected. TensorRT FP16/INT8 enable mixed-precision
tactics, so the engine may retain higher precision layers. ONNX profiles separately verify CUDA kernel placement.
ONNX FP16 denotes the exported graph: CPU profiles show FP32 weighted-kernel inputs after runtime conversion.
Strict box-level parity is recorded as a diagnostic; quantized models are judged on held-out mAP loss.</p>
<p>PyTorch uses four CPU threads. ONNX Runtime and OpenVINO use backend threading defaults; these are not equal-thread-budget tests.</p>
<h2>Camera scaling, tracking and batching</h2><p>{len(pipeline_rows)} serving experiments. Each camera replays the same clip at source FPS,
has an independent tracker and a bounded queue, and requests one decoder thread. Model choices lose at most one absolute
point of both global mAP and mean vehicle AP against the same-size CPU PyTorch baseline.
The illustrative serving timing gate is at least 15 processed FPS and p95 frame age at most 100 ms for every camera.</p>
{serving_table}<p>Frame age includes decode backlog, queueing, inference, tracking and metadata preparation. Disk logging is excluded.
Batch size is an opportunistic maximum. Stride skips are reported separately from queue drops. BoT-SORT and rendering runs are
separate experiments. Passing this short replay gate is not proof of sustained production capacity or counting accuracy.</p>
<h2>Evidence and limitations</h2><p>GPU telemetry is sampled with NVML and covers the whole device; unsupported power readings remain N/A.
{resource_table}
<p>Process RSS includes predecoded frames and runtime libraries. GPU readings are whole-device samples, not per-process allocations.
This is a shared Windows laptop, with a short reproducible traffic clip. Images, weights and engines remain local.
Raw timing, labeled evaluations, graph/engine inspections, resource samples, dataset hashes and the complete command ledger accompany this report.
The COCO subset measures general detection quality; independently annotated traffic events are still needed for counting validation.</p>
<p>Missing configurations: {html.escape(", ".join(missing) or "none")}. Failed stages: {html.escape(", ".join(failed) or "none")}.</p></html>"""
    output.write_text(body, encoding="utf-8")
    summary = {
        "kind": "full_matrix_report",
        "complete": len(rows),
        "planned": len(plan),
        "missing": missing,
        "failed_stages": failed,
        "rows": rows,
        "pipeline_runs": len(pipeline_rows),
        "pipelines": pipeline_rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
