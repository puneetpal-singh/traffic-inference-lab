"""Generate the engineering findings from the completed matrix and serving experiments."""

import json
from pathlib import Path

root = Path("results/full-matrix")
summary = json.loads((root / "summary.json").read_text())
rows = summary["rows"]
hardware = json.loads((root / "hardware.json").read_text())
lines = [
    "# Measured findings: CPU, CUDA and TensorRT",
    "",
    f"Completed **{summary['complete']}/{summary['planned']}** planned runtime/precision/resolution configurations.",
    "All numbers below are calculated from checked-in artifacts; no backend result is estimated.",
    "",
    "## Test conditions",
    "",
    f"- GPU: {hardware['gpu_name']}, compute capability {hardware['compute_capability']}.",
    f"- PyTorch {hardware['torch']}, CUDA {hardware['torch_cuda']}, cuDNN {hardware['cudnn']}.",
    f"- TensorRT {hardware['tensorrt']}; ONNX Runtime GPU package {hardware['onnxruntime']}.",
    f"- OpenVINO {hardware['environment']['packages']['openvino']}; {hardware['environment']['ram_gib']:.2f} GiB system RAM.",
    "- One environment for all CPU/GPU cases. Batch one; 120 predecoded traffic frames × five trials; twenty warm-up calls.",
    "- Accuracy: 256 COCO val2017 holdout images. INT8 calibration: 512 different images, with asserted zero overlap.",
    "- Shared Windows desktop, fixed run order, short clip. These are measured deployment experiments, not a hardware leaderboard.",
    "",
    "## Complete comparison",
    "",
    "FPS is total frames divided by total measured trial time. Model wall time includes preprocessing, inference, postprocessing",
    "and completion of the CPU box transfer. It excludes decoding and camera queues. Full per-trial samples and NVML telemetry",
    "are preserved in the timing JSON files.",
    "",
    "| Configuration | FPS | p95 / p99 ms | mAP50–95 | Vehicle mAP50–95 | Δ mAP points | Gate |",
    "|---|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    vehicle = "N/A" if row["vehicle_map50_95"] is None else f"{row['vehicle_map50_95']:.4f}"
    lines.append(
        f"| {row['name']} | {row['fps']:.2f} | {row['p95_ms']:.2f} / {row['p99_ms']:.2f} | "
        f"{row['map50_95']:.4f} | {vehicle} | {row['map_drop_points']:.2f} | "
        f"{'PASS' if row['model_gate_passed'] else 'REJECT'} |"
    )
lines += [
    "",
    "The illustrative model gate requires p95 ≤100 ms and ≤1 absolute mAP point lost against PyTorch CPU FP32",
    "at the same resolution. Vehicle mAP is the mean over car, motorcycle, bus and truck when all four have labels.",
    "",
    "## Deployment choices supported by these measurements",
    "",
]
for size in (320, 640):
    baseline = next(r for r in rows if r["name"] == f"pytorch-cpu-fp32-{size}")
    for device in ("CPU", "CUDA"):
        eligible = [
            r for r in rows if r["imgsz"] == size and r["device"] == device and r["model_gate_passed"]
        ]
        if eligible:
            best = min(eligible, key=lambda r: r["p95_ms"])
            lines.append(
                f"- {device}, {size}px: lowest accepted p95 is **{best['name']}**, {best['p95_ms']:.2f} ms, "
                f"{best['fps']:.2f} FPS; {best['fps'] / baseline['fps']:.2f}× this study's CPU PyTorch throughput."
            )
rejected = [r["name"] for r in rows if not r["model_gate_passed"]]
lines += [
    "",
    "Rejected by the model gate: " + (", ".join(rejected) or "none") + ".",
    "A faster configuration is not accepted if its measured accuracy loss exceeds the limit. FP16 and INT8 are experiments,",
    "not guaranteed improvements. Where runtimes are close, shared-desktop noise and the short workload limit causal conclusions.",
    "",
    "## Precision and execution evidence",
    "",
    "- ONNX CUDA cases include separate provider profiles proving weighted kernels executed on CUDA.",
    "- OpenVINO FP16 is compressed storage; CPU arithmetic is checked separately through profiling execution types.",
    "- TensorRT FP16/INT8 enable mixed-precision tactics. Engine inspection records available layer details and CUDA graph use.",
    "- Strict detection parity is diagnostic. Quantized models may fail it while still meeting the labeled mAP budget.",
    "- NVML records whole-device memory, utilization, temperature, clocks and supported power readings. It is not per-process",
    "  energy measurement; unsupported fields remain N/A.",
    "",
    "## Camera scaling and pipeline tuning",
    "",
]
pipeline_paths = sorted((root / "pipelines").glob("*/summary.json"))
if pipeline_paths:
    lines += [
        "All serving configurations use one requested decoder thread per camera, independent trackers, bounded queues,",
        "and models with at most one absolute point lost in both global mAP and mean vehicle AP versus CPU PyTorch.",
        "Replay is paced at the source FPS. Frame age includes decode backlog, queueing, prediction, tracking, metadata",
        "preparation and optional rendering; log writing is excluded. The rendered run is reported separately.",
        "",
        "| Experiment | Decoded / processed | Dropped | Skipped | Min camera FPS | Worst camera p95 ms | Timing gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for path in pipeline_paths:
        data = json.loads(path.read_text())
        cameras = data["cameras"]
        fps = min(c["processed_fps"] for c in cameras)
        ages = [c["frame_age_ms"]["p95"] for c in cameras]
        age = max(a for a in ages if a is not None) if all(a is not None for a in ages) else float("inf")
        passed = fps >= 15 and age <= 100
        lines.append(
            f"| {path.parent.name} | {sum(c['decoded'] for c in cameras)} / {sum(c['processed'] for c in cameras)} | "
            f"{sum(c['queue_dropped'] for c in cameras)} | {sum(c['stride_skipped'] for c in cameras)} | "
            f"{fps:.2f} | {age:.2f} | {'PASS' if passed else 'REJECT'} |"
        )
else:
    lines.append(
        "Camera experiments have not been assembled yet; do not infer serving capacity from model FPS."
    )
lines += [
    "",
    "The serving timing gate is ≥15 processed FPS for every camera and p95 frame age ≤100 ms for every camera.",
    "It is an illustrative short-replay gate, not production certification. Duplicate replay sources test compute load,",
    "not scene diversity or network behavior. Counts require independent human event labels; none are invented.",
    "",
    "## Reproduce and inspect",
    "",
    "See [the full protocol](FULL_MATRIX_PROTOCOL.md), [data provenance](DATA.md), [raw artifacts](../results/full-matrix/)",
    "and [the portable report](../results/report.html). `comparison.csv` contains the complete machine-readable comparison.",
    "The command ledger includes exports, timings, evaluation, inspection and retry history. Run",
    "`python scripts/audit_full_matrix.py` to verify coverage, model hashes, frame counts, CUDA placement and calibration separation.",
]
Path("docs/RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
