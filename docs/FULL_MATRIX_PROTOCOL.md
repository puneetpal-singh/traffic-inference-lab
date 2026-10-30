# Complete CPU/CUDA precision study

## Scope

Twenty-eight configurations: two image sizes (320 and 640) for each of these fourteen variants:

| Hardware | Runtime | Variants |
|---|---|---|
| CPU | PyTorch | FP32 |
| CPU | ONNX Runtime | FP32, FP16 graph, calibrated INT8 |
| CPU | OpenVINO | FP32, FP16 stored constants, calibrated INT8 |
| NVIDIA CUDA | PyTorch | FP32, FP16 |
| NVIDIA CUDA | ONNX Runtime | FP32, FP16 graph |
| NVIDIA CUDA | TensorRT | FP32, FP16-enabled, INT8-enabled |

ONNX Runtime's CUDA provider can partition graphs across devices. CUDA cases require actual weighted kernels on CUDA,
verified with a separate ORT profile; provider registration alone is insufficient evidence. An INT8 ONNX CPU graph is
not advertised as GPU INT8 acceleration merely because a CUDA provider is present.

OpenVINO FP16 compresses model constants; it does not guarantee native FP16 arithmetic on this i7. The compiled model's
runtime precision is inspected. TensorRT precision flags enable mixed-precision tactics; the engine may retain higher
precision layers. TensorRT INT8 includes detailed layer inspection when available.

## Consistent environment and inputs

The full matrix uses one CUDA-enabled Python environment for both CPU and GPU inference. CPU-only PyTorch is not used
as a substitute for a GPU experiment. All cases use the same original YOLO11n checkpoint, square letterboxing, confidence
.25, NMS IoU .7 and COCO vehicle class filter for timing. Labeled quality is separately evaluated on all available classes
at confidence .001, identical input size and preprocessing, and batch one.

PyTorch uses four CPU threads; ONNX Runtime and OpenVINO use their backend thread defaults. Resource budgets therefore
must not be assumed identical simply because the model adapter has a `threads` setting. No concurrent timed runs occur.

## Independent calibration and evaluation

`prepare_coco_holdout.py` shuffles the official COCO val2017 image list, initially sorted by image ID, using seed 20261030.
It selects the first 512 images for calibration and the next 256 for evaluation. The image-ID sets are asserted disjoint.
Neither subset is used for fine-tuning. This replaces the original COCO128 smoke evaluation for the expanded matrix.

Annotations are converted to YOLO boxes, clipping boxes to image bounds and omitting `iscrowd` annotations. These are
Ultralytics mAP metrics on the converted subset, not an official score over all 5,000 COCO validation images. Image hashes,
source URLs, annotation-file hash and exact selected IDs are recorded in the manifest. Keep the manifest with any published
measurement. Calibration diversity is general COCO imagery; validate on a labeled traffic holdout before deploying counting.

## Timing and resources

- 120 predecoded traffic frames per trial, five trials, twenty warm-up calls, batch one.
- Warm-up uses zero images at the configured shape. Real images can activate postprocessing paths not exercised by zeros;
  first-trial variation remains visible in the raw records. No outliers are removed. Headline FPS uses total frames divided
  by total trial time rather than the arithmetic mean of trial FPS.
- Static exports for each resolution avoid dynamic-shape compilation becoming a hidden variable.
- Wall timing stops only after detection boxes reach CPU, including any necessary GPU completion/transfer.
- Report p50/p95/p99, per-trial FPS, raw observations and backend stage timings.
- A 100 ms telemetry sampler captures process RSS and available system RAM; NVML adds whole-GPU memory, utilization,
  temperature, clocks and power when available. Unsupported fields remain N/A. These are sampled values, not exact peaks.
- NVML measures the whole device and may include unrelated processes. A short shared-desktop study is not a hardware
  leaderboard, energy-efficiency certification or sustained production-capacity test.
- Downloads, export, INT8 calibration, compilation, labeled evaluation and execution profiling are outside timing.
- TensorRT build workspace is limited to 0.5 GiB by default for the 4 GiB device. Engines are built on the deployment GPU.
  Builder optimization level remains TensorRT's default 3; complete export-job durations are recorded separately from inference.

Strict box agreement is recorded separately. Quantization can legitimately fail that strict gate; accuracy acceptance uses
the labeled evaluation. The illustrative model gate is p95 ≤100 ms and no more than one absolute mAP50–95 point lost versus
FP32 PyTorch at the same size. A camera-processing-rate and end-to-end freshness gate is a separate requirement.

Serving candidates must also lose no more than one absolute point of mean vehicle AP (car, motorcycle, bus, truck).
The global detector gate above remains unchanged. This additional traffic-specific guard is recorded separately in
`pipeline-selections.json`; it was added after the 320px INT8 result showed a larger vehicle loss than global loss.
Selection chooses the lowest measured p95 among eligible candidates for each backend. Dynamic batch engines must
pass both accuracy checks before serving measurements.

## Reproduction

Use the Windows CUDA/TensorRT setup in the README, then:

```powershell
python scripts/prepare_coco_holdout.py
python scripts/full_matrix.py
python scripts/run_pipeline_matrix.py
python scripts/audit_full_matrix.py --require-pipelines
traffic-lab report --input results --output results/report.html
python scripts/write_full_findings.py
```

The matrix runner resumes successful artifacts by default. Use `--force` to regenerate them after changing software,
inputs or configuration. `--only tensorrt` selects the TensorRT cases. The ledger retains commands, exit codes, durations
and retry history. Failed or missing cases remain visible; they are never filled with estimates.
