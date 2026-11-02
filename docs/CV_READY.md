# CV-ready project description

**YOLO Inference Optimization & Multi-Camera Analytics**

Python · PyTorch · CUDA · TensorRT · ONNX Runtime · OpenVINO · OpenCV · GitHub Actions

## Copy-ready bullets

- Measured **2.69× paired model-throughput speedup** for TensorRT FP16 enabled versus eager PyTorch CUDA FP32 on a Tesla T4 at 320px/batch one, across ten randomized trial rounds; evaluated accuracy on 4488 calibration-disjoint COCO images.
- Compared 13 runtime/device/precision configurations across PyTorch, ONNX Runtime, OpenVINO and TensorRT; quantified accuracy, throughput, latency and artifact-size tradeoffs, including INT8 configurations that reduced accuracy without improving speed.
- Built bounded camera queues, independent tracking/counting and resource telemetry; completed 72 repeated serving experiments and identified decoder backlog under overload. Two streams met timing budgets in all three TensorRT INT8 repeats; counting accuracy remains unvalidated.

## Defensible claims

| Claim | Supporting evidence |
|---|---|
| Same-GPU runtime optimization | [Latest Kaggle evidence](KAGGLE_RESULTS.md), ten randomized rounds, matched model hashes and raw archived trials |
| Reproducible backend comparison | [Verified report](../results/kaggle-t4-320-2026-11-02/Kaggle_Results_Analysis.md), 4488-image evaluation, disjoint calibration and artifact audit |
| Systems profiling and overload handling | Per-camera drops, frame age, decoder lag and telemetry across 72 repeated serving runs |

Keep hardware, resolution, baseline and endpoint attached to performance numbers. The 2.69× is a TensorRT-versus-eager-PyTorch configuration comparison, not a speedup caused solely by FP16. AP differences are point estimates; the measured FP16 global AP loss was 0.0036 absolute points, without an AP confidence interval. One flagged ONNX GPU trial was retained. The older GTX 1650 Ti studies remain separate historical evidence.
Do not describe the eight-camera load test as eight cameras meeting real-time requirements. Only the measured passing loads support that claim; counting accuracy and production reliability remain unvalidated.

The repository remains private as requested. A CV link requires the reviewer to have repository access; the checked-in report and chart can also be shared as artifacts.

Use [the walkthrough](WALKTHROUGH.md) to prepare an explanation of the design, rejected optimizations and limitations.
