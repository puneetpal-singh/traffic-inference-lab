# CV-ready project description

**YOLO Inference Optimization & Multi-Camera Analytics**

Python · PyTorch · CUDA · TensorRT · ONNX Runtime · OpenVINO · OpenCV · GitHub Actions

## Copy-ready bullets

- Achieved **3.39× paired model-throughput speedup** with TensorRT FP16 versus PyTorch CUDA FP32 on a GTX 1650 Ti, confirmed across six balanced randomized trial pairs with real-image warm-up and held-out accuracy checks.
- Benchmarked 28 runtime/precision/resolution configurations using 512 calibration images and a disjoint 256-image COCO holdout; verified execution precision and rejected configurations exceeding accuracy-loss budgets.
- Implemented bounded camera queues, independent tracking/counting, resource telemetry and a results dashboard; profiled 24 serving experiments and automated evidence audits in CI.

## Defensible claims

| Claim | Supporting evidence |
|---|---|
| Same-GPU runtime optimization | [Randomized confirmation](CONFIRMATION.md), raw trial records and matched model hashes |
| Reproducible backend comparison | [Full matrix](RESULTS.md), exported-graph/engine inspection, calibration split and environment locks |
| Systems profiling and overload handling | Per-camera drops, frame age, decoder lag and telemetry measurements across 24 runs |

Keep the hardware, resolution, baseline and model-only endpoint attached to performance numbers. The original CPU-to-GPU ratio mixes hardware and runtime changes; use the same-GPU result above for an optimization claim.
Do not describe the eight-camera load test as eight cameras meeting real-time requirements. Only the measured passing loads support that claim; counting accuracy and production reliability remain unvalidated.

The repository remains private as requested. A CV link requires the reviewer to have repository access; the checked-in report and chart can also be shared as artifacts.

Use [the walkthrough](WALKTHROUGH.md) to prepare an explanation of the design, rejected optimizations and limitations.
