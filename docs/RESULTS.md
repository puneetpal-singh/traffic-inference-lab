# Measured findings

Generated from the checked-in JSON artifacts. No estimated speedups.

## Environment and scope

- OS: Windows-11-10.0.26300-SP0
- CPU: Intel64 Family 6 Model 165 Stepping 2, GenuineIntel
- Memory: 7.84 GiB; GPU inventory: NVIDIA GeForce GTX 1650 Ti, 580.97, 4096 MiB
- PyTorch: 2.15.0+cpu; ONNX Runtime: 1.31.1; OpenVINO: 2026.4.1
- All reported inference runs use CPU. CUDA/TensorRT and INT8 were not measured.
- Shared Windows desktop, short clip, fixed experiment order; no isolation or thermal control.
  Treat this as a preliminary runtime study, not a hardware leaderboard or production capacity certification.
- Four PyTorch threads; ONNX/OpenVINO backend threading defaults. Resource budgets are not identical.

## Runtime comparison

Predecoded road video; batch one; five warm-up calls; three trials, 60 frames per trial.

| Run | Mean FPS | p50 / p95 / p99 wall ms | COCO128 mAP50–95 |
|---|---:|---:|---:|
| onnx-320 | 39.56 | 22.89 / 36.58 / 55.35 | 0.3856 |
| onnx-640 | 13.87 | 70.00 / 83.98 / 100.76 | 0.5052 |
| openvino-320 | 42.43 | 19.39 / 36.19 / 74.23 | 0.3856 |
| openvino-640 | 14.25 | 61.26 / 107.37 / 177.84 | 0.5052 |
| pytorch-320 | 21.74 | 41.94 / 60.94 / 80.46 | 0.3856 |
| pytorch-640 | 12.11 | 80.04 / 93.73 / 107.04 | 0.5052 |

COCO128 is a tiny general-purpose smoke set derived from training imagery, not an independent traffic holdout.
The mAP values evaluate all classes; throughput uses the vehicle class filter.

## Export agreement

- parity-onnx-320: **PASS**, 30 sampled frames, 174 matched detections (IoU >= 0.9, confidence delta <= 0.02).
- parity-onnx-640: **PASS**, 30 sampled frames, 223 matched detections (IoU >= 0.9, confidence delta <= 0.02).
- parity-openvino-320: **PASS**, 30 sampled frames, 174 matched detections (IoU >= 0.9, confidence delta <= 0.02).
- parity-openvino-640: **PASS**, 30 sampled frames, 223 matched detections (IoU >= 0.9, confidence delta <= 0.02).

## Explicit model acceptance gates

- gate-resolution-tradeoff: **REJECT**. p95 budget 100 ms; maximum accuracy drop 1 absolute mAP points; measured drop 11.96 points. Reasons: Accuracy drop exceeds allowance or labeled quality is missing.
- gate-same-resolution: **ACCEPT**. p95 budget 100 ms; maximum accuracy drop 1 absolute mAP points; measured drop 0.00 points. Reasons: within the model-only budgets.

Agreement does not prove accuracy; deployment requires a labeled traffic holdout.

## Multi-camera load and scheduling

The same short clip is replayed per camera at source FPS. These are compute-load experiments, not diverse RTSP cameras.
Results below are aggregated across cameras; p95 is pooled from the per-frame telemetry.

| Experiment | Decoded / processed | Queue drops | Stride skips | Pooled p95 frame age ms |
|---|---:|---:|---:|---:|
| streams-1-batch-1-stride-1-botsort | 150 / 51 | 99 | 0 | 145.31 |
| streams-1-batch-1-stride-1-bytetrack | 150 / 106 | 44 | 0 | 89.63 |
| streams-2-batch-1-stride-1-bytetrack | 300 / 96 | 204 | 0 | 87.30 |
| streams-4-batch-1-stride-1-bytetrack | 600 / 55 | 545 | 0 | 199.26 |
| streams-4-batch-1-stride-2-bytetrack | 600 / 34 | 501 | 65 | 251.64 |
| streams-4-batch-4-stride-1-bytetrack | 600 / 90 | 510 | 0 | 332.13 |

### Locating the bottleneck

| Experiment | p95 decoder lag ms | p95 queue wait ms | p95 prediction batch ms | p95 tracking/count ms |
|---|---:|---:|---:|---:|
| streams-1-batch-1-stride-1-botsort | 17.98 | 34.97 | 30.28 | 42.27 |
| streams-1-batch-1-stride-1-bytetrack | 18.55 | 30.34 | 54.59 | 3.12 |
| streams-2-batch-1-stride-1-bytetrack | 22.11 | 33.07 | 44.75 | 3.55 |
| streams-4-batch-1-stride-1-bytetrack | 30.94 | 53.35 | 95.53 | 12.60 |
| streams-4-batch-1-stride-2-bytetrack | 40.30 | 26.54 | 162.68 | 15.39 |
| streams-4-batch-4-stride-1-bytetrack | 26.90 | 44.93 | 298.66 | 4.10 |

Percentiles are computed independently and must not be added. Decoder lag includes source-schedule backlog.
A bounded inference queue cannot remove backlog already accumulated inside a sequential decoder.
This distinction guides the next optimization: investigate decoder throughput when lag dominates, rather than
assuming another model export will fix end-to-end latency.

Drops, stride skips, processing rate and frame age must be considered together. A fast result after dropping most inputs
does not meet a high processing-rate requirement. Counts remain unvalidated. The BoT-SORT run also renders video, so it
is a functional demonstration rather than a controlled tracker-speed comparison.

## What to say about this project

Implemented and measured a YOLO deployment pipeline across PyTorch, ONNX Runtime and OpenVINO, with export parity checks,
per-camera tracking, bounded queues, explicit frame-drop accounting, and reproducible latency/quality artifacts.
Compare the tables when discussing measured tradeoffs; do not turn this smoke study into an unqualified speedup or accuracy claim.

## Remaining validation

1. Longer, varied traffic footage and repeated run orders under stable thermal conditions.
2. Independent labeled traffic frames and crossing-event annotations; quantify misses and duplicate counts.
3. Sustained paced-stream acceptance against an agreed latency, processing-rate and quality budget.
4. Optional CUDA/TensorRT and calibrated INT8 experiments on the deployment hardware.

See [methodology](METHODOLOGY.md) and [raw results](../results/).

## Execution audit

Final experiment ledger: 26/26 commands successful. Retained retry records: 4.
Initial tracker-constructor failures were corrected and rerun. Original failure logs and attempt history
are preserved beside the successful measurements. Run `python scripts/validate_artifacts.py` to audit
frame accounting, queue bounds, raw benchmark frame counts, and parity gates.
