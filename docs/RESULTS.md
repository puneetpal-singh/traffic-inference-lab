# Measured findings: CPU, CUDA and TensorRT

Completed **28/28** planned runtime/precision/resolution configurations.
All numbers below are calculated from checked-in artifacts; no backend result is estimated.

## Test conditions

- GPU: NVIDIA GeForce GTX 1650 Ti, compute capability [7, 5].
- PyTorch 2.6.0+cu124, CUDA 12.4, cuDNN 90100.
- TensorRT 10.8.0.43; ONNX Runtime GPU package 1.20.2.
- OpenVINO 2026.4.1; 7.84 GiB system RAM.
- One environment for all CPU/GPU cases. Batch one; 120 predecoded traffic frames × five trials; twenty warm-up calls.
- Accuracy: 256 COCO val2017 holdout images. INT8 calibration: 512 different images, with asserted zero overlap.
- Shared Windows desktop, fixed run order, short clip. These are measured deployment experiments, not a hardware leaderboard.

## Complete comparison

FPS is total frames divided by total measured trial time. Model wall time includes preprocessing, inference, postprocessing
and completion of the CPU box transfer. It excludes decoding and camera queues. Full per-trial samples and NVML telemetry
are preserved in the timing JSON files.

| Configuration | FPS | p95 / p99 ms | mAP50–95 | Vehicle mAP50–95 | Δ mAP points | Gate |
|---|---:|---:|---:|---:|---:|---|
| pytorch-cpu-fp32-320 | 23.90 | 55.87 / 63.02 | 0.2997 | 0.2801 | 0.00 | PASS |
| pytorch-cuda-fp32-320 | 36.80 | 32.96 / 36.20 | 0.2997 | 0.2801 | 0.00 | PASS |
| pytorch-cuda-fp16-320 | 28.83 | 44.14 / 51.36 | 0.2995 | 0.2792 | 0.02 | PASS |
| onnx-cpu-fp32-320 | 42.67 | 23.31 / 25.29 | 0.2997 | 0.2801 | 0.00 | PASS |
| onnx-cpu-fp16-320 | 24.94 | 47.60 / 51.11 | 0.2997 | 0.2806 | 0.01 | PASS |
| onnx-cpu-int8-320 | 24.13 | 50.92 / 62.01 | 0.2627 | 0.2530 | 3.71 | REJECT |
| onnx-cuda-fp32-320 | 42.57 | 36.87 / 45.08 | 0.2997 | 0.2801 | 0.00 | PASS |
| onnx-cuda-fp16-320 | 42.95 | 33.61 / 50.82 | 0.2993 | 0.2803 | 0.05 | PASS |
| openvino-cpu-fp32-320 | 38.33 | 46.51 / 78.93 | 0.2997 | 0.2801 | 0.00 | PASS |
| openvino-cpu-fp16-320 | 42.00 | 39.20 / 50.12 | 0.3000 | 0.2807 | -0.02 | PASS |
| openvino-cpu-int8-320 | 47.05 | 38.84 / 56.43 | 0.3001 | 0.2801 | -0.04 | PASS |
| tensorrt-cuda-fp32-320 | 73.52 | 25.22 / 29.52 | 0.2997 | 0.2801 | -0.00 | PASS |
| tensorrt-cuda-fp16-320 | 75.26 | 24.34 / 32.50 | 0.2997 | 0.2796 | -0.00 | PASS |
| tensorrt-cuda-int8-320 | 91.68 | 17.68 / 24.46 | 0.2919 | 0.2697 | 0.78 | PASS |
| pytorch-cpu-fp32-640 | 11.89 | 100.56 / 139.05 | 0.4015 | 0.3952 | 0.00 | REJECT |
| pytorch-cuda-fp32-640 | 35.00 | 36.20 / 43.11 | 0.4013 | 0.3952 | 0.03 | PASS |
| pytorch-cuda-fp16-640 | 35.24 | 37.67 / 41.94 | 0.4017 | 0.3962 | -0.02 | PASS |
| onnx-cpu-fp32-640 | 12.41 | 99.40 / 145.82 | 0.4015 | 0.3952 | -0.00 | PASS |
| onnx-cpu-fp16-640 | 7.48 | 151.36 / 180.59 | 0.4007 | 0.3958 | 0.09 | REJECT |
| onnx-cpu-int8-640 | 7.57 | 150.78 / 170.08 | 0.3539 | 0.3439 | 4.77 | REJECT |
| onnx-cuda-fp32-640 | 44.59 | 37.16 / 44.84 | 0.4015 | 0.3952 | -0.00 | PASS |
| onnx-cuda-fp16-640 | 29.36 | 49.85 / 54.94 | 0.4002 | 0.3958 | 0.14 | PASS |
| openvino-cpu-fp32-640 | 10.08 | 179.97 / 289.24 | 0.4015 | 0.3952 | -0.00 | REJECT |
| openvino-cpu-fp16-640 | 16.10 | 94.79 / 112.87 | 0.4012 | 0.3951 | 0.04 | PASS |
| openvino-cpu-int8-640 | 24.26 | 64.35 / 81.10 | 0.3984 | 0.3930 | 0.31 | PASS |
| tensorrt-cuda-fp32-640 | 59.25 | 27.62 / 32.18 | 0.4015 | 0.3952 | -0.00 | PASS |
| tensorrt-cuda-fp16-640 | 58.69 | 30.07 / 33.39 | 0.4022 | 0.3951 | -0.06 | PASS |
| tensorrt-cuda-int8-640 | 46.15 | 38.93 / 53.68 | 0.3872 | 0.3781 | 1.44 | REJECT |

The illustrative model gate requires p95 ≤100 ms and ≤1 absolute mAP point lost against PyTorch CPU FP32
at the same resolution. Vehicle mAP is the mean over car, motorcycle, bus and truck when all four have labels.

## Deployment choices supported by these measurements

- CPU, 320px: lowest accepted p95 is **onnx-cpu-fp32-320**, 23.31 ms, 42.67 FPS; 1.79× this study's CPU PyTorch throughput.
- CUDA, 320px: lowest accepted p95 is **tensorrt-cuda-int8-320**, 17.68 ms, 91.68 FPS; 3.84× this study's CPU PyTorch throughput.
- CPU, 640px: lowest accepted p95 is **openvino-cpu-int8-640**, 64.35 ms, 24.26 FPS; 2.04× this study's CPU PyTorch throughput.
- CUDA, 640px: lowest accepted p95 is **tensorrt-cuda-fp32-640**, 27.62 ms, 59.25 FPS; 4.98× this study's CPU PyTorch throughput.

### Trial variation at 320px

Zero-image warm-up does not exercise every real-image path. These trial rates are all retained; the headline
uses total frames / total trial time, not the arithmetic mean of the rates below.

| Configuration | Five trial FPS, in order |
|---|---|
| pytorch-cuda-fp32-320 | 25.08 / 41.51 / 42.04 / 41.23 / 41.90 |
| pytorch-cuda-fp16-320 | 22.67 / 34.52 / 33.17 / 25.48 / 32.28 |
| onnx-cuda-fp32-320 | 27.11 / 55.20 / 52.04 / 46.51 / 46.02 |
| onnx-cuda-fp16-320 | 26.85 / 54.43 / 48.56 / 43.29 / 58.45 |
| tensorrt-cuda-fp32-320 | 32.57 / 97.88 / 111.63 / 109.56 / 110.96 |
| tensorrt-cuda-fp16-320 | 36.71 / 95.07 / 106.84 / 106.09 / 101.13 |
| tensorrt-cuda-int8-320 | 42.88 / 109.30 / 135.09 / 136.39 / 136.35 |

### Optimizations that did not help

- pytorch-cuda-fp16-320: 28.83 FPS versus 36.80 FPS for FP32 (21.7% lower throughput).
- onnx-cpu-fp16-320: 24.94 FPS versus 42.67 FPS for FP32 (41.6% lower throughput).
- onnx-cpu-int8-320: 24.13 FPS versus 42.67 FPS for FP32 (43.5% lower throughput).
- onnx-cpu-fp16-640: 7.48 FPS versus 12.41 FPS for FP32 (39.7% lower throughput).
- onnx-cpu-int8-640: 7.57 FPS versus 12.41 FPS for FP32 (39.0% lower throughput).
- onnx-cuda-fp16-640: 29.36 FPS versus 44.59 FPS for FP32 (34.2% lower throughput).
- tensorrt-cuda-int8-640: 46.15 FPS versus 59.25 FPS for FP32 (22.1% lower throughput).

Rejected by the model gate: onnx-cpu-int8-320, pytorch-cpu-fp32-640, onnx-cpu-fp16-640, onnx-cpu-int8-640, openvino-cpu-fp32-640, tensorrt-cuda-int8-640.
A faster configuration is not accepted if its measured accuracy loss exceeds the limit. FP16 and INT8 are experiments,
not guaranteed improvements. Where runtimes are close, shared-desktop noise and the short workload limit causal conclusions.

## Precision and execution evidence

- ONNX CUDA cases include separate provider profiles proving weighted kernels executed on CUDA.
- ONNX FP16 names the exported graph. CPU execution profiles show float32 weighted-kernel inputs after conversion.
- OpenVINO FP16 is compressed storage; CPU arithmetic is checked separately through profiling execution types.
- TensorRT FP16/INT8 enable mixed-precision tactics. Engine inspection records available layer details and CUDA graph use.
- Strict detection parity is diagnostic. Quantized models may fail it while still meeting the labeled mAP budget.
- NVML records whole-device memory, utilization, temperature, clocks and supported power readings. It is not per-process
  energy measurement; unsupported fields remain N/A.

## Camera scaling and pipeline tuning

Serving model choices: openvino-cpu-int8-320, onnx-cpu-fp32-320, onnx-cuda-fp16-320, tensorrt-cuda-fp16-320.

tensorrt-cuda-int8-320 met the global budget but lost 1.04 vehicle AP points, so it was excluded from serving selection. The traffic-specific guard was added after observing this difference; see the protocol for that scope change.

All serving configurations use one requested decoder thread per camera, independent trackers, bounded queues,
and models with at most one absolute point lost in both global mAP and mean vehicle AP versus CPU PyTorch.
Before replay, five zero-image calls and real first-frame batches warm the detector/NMS; tracker state remains fresh.
Replay is paced at the source FPS. Frame age includes decode backlog, queueing, prediction, tracking, metadata
preparation and optional rendering; log writing is excluded. The rendered run is reported separately.

| Experiment | Decoded / processed | Dropped | Skipped | Min camera FPS | Worst camera p95 ms | Timing gate |
|---|---:|---:|---:|---:|---:|---|
| onnx-cpu-fp32-320-streams-1-batch-1-stride-1-bytetrack | 150 / 114 | 36 | 0 | 22.37 | 498.31 | REJECT |
| onnx-cpu-fp32-320-streams-2-batch-1-stride-1-bytetrack | 300 / 126 | 174 | 0 | 12.20 | 169.66 | REJECT |
| onnx-cpu-fp32-320-streams-4-batch-1-stride-1-bytetrack | 600 / 89 | 511 | 0 | 2.42 | 1662.75 | REJECT |
| onnx-cpu-fp32-320-streams-8-batch-1-stride-1-bytetrack | 1200 / 48 | 1152 | 0 | 0.53 | 4200.35 | REJECT |
| onnx-cuda-fp16-320-streams-1-batch-1-stride-1-bytetrack | 150 / 135 | 15 | 0 | 26.79 | 125.64 | REJECT |
| onnx-cuda-fp16-320-streams-2-batch-1-stride-1-bytetrack | 300 / 129 | 171 | 0 | 11.73 | 502.93 | REJECT |
| onnx-cuda-fp16-320-streams-4-batch-1-stride-1-bytetrack | 600 / 112 | 488 | 0 | 3.67 | 2372.64 | REJECT |
| onnx-cuda-fp16-320-streams-8-batch-1-stride-1-bytetrack | 1200 / 109 | 1091 | 0 | 0.96 | 5404.58 | REJECT |
| openvino-cpu-int8-320-streams-1-batch-1-stride-1-bytetrack | 150 / 121 | 29 | 0 | 23.85 | 190.13 | REJECT |
| openvino-cpu-int8-320-streams-2-batch-1-stride-1-bytetrack | 300 / 130 | 170 | 0 | 12.71 | 173.56 | REJECT |
| openvino-cpu-int8-320-streams-4-batch-1-stride-1-bytetrack | 600 / 73 | 527 | 0 | 2.95 | 914.39 | REJECT |
| openvino-cpu-int8-320-streams-8-batch-1-stride-1-bytetrack | 1200 / 40 | 1160 | 0 | 0.46 | 3689.59 | REJECT |
| tensorrt-cuda-dynamic-320-streams-4-batch-4-stride-1-bytetrack | 600 / 178 | 422 | 0 | 8.52 | 414.35 | REJECT |
| tensorrt-cuda-dynamic-320-streams-8-batch-4-stride-1-bytetrack | 1200 / 184 | 1016 | 0 | 3.13 | 1779.94 | REJECT |
| tensorrt-cuda-fp16-320-streams-1-batch-1-stride-1-botsort | 150 / 52 | 98 | 0 | 9.74 | 522.29 | REJECT |
| tensorrt-cuda-fp16-320-streams-1-batch-1-stride-1-bytetrack | 150 / 141 | 9 | 0 | 27.95 | 92.67 | PASS |
| tensorrt-cuda-fp16-320-streams-1-batch-1-stride-1-bytetrack-render | 150 / 57 | 93 | 0 | 11.14 | 338.28 | REJECT |
| tensorrt-cuda-fp16-320-streams-1-batch-1-stride-1-bytetrack-sampled-telemetry | 150 / 144 | 6 | 0 | 28.42 | 89.65 | PASS |
| tensorrt-cuda-fp16-320-streams-2-batch-1-stride-1-bytetrack | 300 / 188 | 112 | 0 | 18.34 | 106.64 | REJECT |
| tensorrt-cuda-fp16-320-streams-4-batch-1-stride-1-bytetrack | 600 / 152 | 448 | 0 | 6.78 | 432.61 | REJECT |
| tensorrt-cuda-fp16-320-streams-4-batch-1-stride-1-bytetrack-sampled-telemetry | 600 / 220 | 380 | 0 | 7.73 | 1758.71 | REJECT |
| tensorrt-cuda-fp16-320-streams-4-batch-1-stride-2-bytetrack | 600 / 132 | 344 | 124 | 5.40 | 961.22 | REJECT |
| tensorrt-cuda-fp16-320-streams-8-batch-1-stride-1-bytetrack | 1200 / 173 | 1027 | 0 | 2.31 | 2848.17 | REJECT |
| tensorrt-cuda-fp16-320-streams-8-batch-1-stride-1-bytetrack-sampled-telemetry | 1200 / 284 | 916 | 0 | 2.68 | 5494.26 | REJECT |

### Largest tested load meeting the baseline replay gate

- ONNXBackend CPU: **none of the tested camera counts** with batch one, stride one, ByteTrack and no rendering.
- ONNXBackend CUDA: **none of the tested camera counts** with batch one, stride one, ByteTrack and no rendering.
- OpenVINOBackend CPU: **none of the tested camera counts** with batch one, stride one, ByteTrack and no rendering.
- TensorRTBackend CUDA: **1 camera** with batch one, stride one, ByteTrack and no rendering.

These are the largest passing loads among the tested 1/2/4/8-camera replays, not production capacity ceilings.

### Observability overhead experiment

Full-frame brightness telemetry is compared with a sampled grid of at most 64×64 pixels. Only the brightness
diagnostic changes; detector input and tracker settings stay fixed. The approximation can differ from the
full-frame mean. Both runs remain visible; these short fixed-order comparisons do not isolate every source of noise.

| Cameras | Mean telemetry ms: full / sampled | Min camera FPS: full / sampled | Worst p95 ms: full / sampled |
|---|---:|---:|---:|
| 1 | 5.822 / 1.147 | 27.95 / 28.42 | 92.67 / 89.65 |
| 4 | 8.944 / 2.076 | 6.78 / 7.73 | 432.61 / 1758.71 |
| 8 | 11.077 / 2.682 | 2.31 / 2.68 | 2848.17 / 5494.26 |

Reducing diagnostic overhead is not sufficient evidence of higher serving capacity. In this run, the
end-to-end gates remain separate from the local telemetry saving. Decoder lag and shared-system variability must be considered.
Sampled runs at camera counts 4, 8 recorded worse p95 frame age than their baseline counterparts.

The serving timing gate is ≥15 processed FPS for every camera and p95 frame age ≤100 ms for every camera.
It is an illustrative short-replay gate, not production certification. Duplicate replay sources test compute load,
not scene diversity or network behavior. Counts require independent human event labels; none are invented.

## Reproduce and inspect

See [the full protocol](FULL_MATRIX_PROTOCOL.md), [data provenance](DATA.md), [raw artifacts](../results/full-matrix/)
and [the portable report](../results/report.html). `comparison.csv` contains the complete machine-readable comparison.
The command ledger includes exports, timings, evaluation, inspection and retry history. Run
`python scripts/audit_full_matrix.py` to verify coverage, model hashes, frame counts, CUDA placement and calibration separation.
