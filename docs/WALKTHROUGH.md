# Presenting the optimization work

For a CV or interview, lead with the [same-GPU paired confirmation](CONFIRMATION.md) and use the
[copy-ready project description](CV_READY.md). It separates runtime optimization from a CPU-to-GPU hardware change.

Start with the engineering question: **How many cameras can this machine process while preserving detection quality
and keeping results fresh?** Open [the measured findings](RESULTS.md) and the downloadable [standalone report](../results/report.html).

## A five-minute walkthrough

1. **Establish the baseline.** Show the hardware/software record, original model hash, image size and predecoded timing
   endpoint. Explain why model FPS excludes decoding, queues and tracking, and cannot establish camera capacity by itself.
2. **Compare deployment choices.** Show CPU and CUDA measurements at 320 and 640 pixels. Explain the accuracy–latency
   tradeoff and why comparing different image sizes as an unconditional backend speedup would be misleading.
3. **Inspect precision.** Show ONNX provider profiles, OpenVINO execution types and TensorRT layer inspection. An FP16
   file can execute with FP32 CPU inputs; an enabled INT8 builder can retain higher-precision layers. Verify execution.
4. **Explain a rejected optimization.** Use the measured ONNX CPU INT8 accuracy loss or a slower FP16 result. Calibration
   and evaluation use disjoint images. Strict box agreement is a diagnostic; labeled mAP is the quality gate.
5. **Move from models to a system.** Show the one/two/four/eight-camera table, queue drops, per-camera processing rate and
   p95 frame age. Compare stride, batching, tracking and rendering as separate experiments. Explain the bounded queue and
   independent tracker state. End with the deployment choice supported by these measurements and its validation limits.

## Evidence to open when challenged

| Question | Evidence |
|---|---|
| Did CUDA actually run? | `results/full-matrix/inspection-onnx-cuda-*.json` and linked native profiles |
| Did INT8 actually execute? | OpenVINO profiled execution types; TensorRT detailed weighted-layer formats |
| Was accuracy preserved? | Same-size FP32 baseline and paired `quality-*.json`; `pipeline-selections.json` for vehicle guard |
| Can someone reproduce it? | Dataset manifest, model hashes, software lock, full command ledger and protocol |
| Were slow observations hidden? | Five trials and all 600 timed observations per configuration; none removed |
| Did overload disappear through dropping? | Each camera's decoded/processed/dropped/skipped accounting and freshness/FPS gates |
| Is the counting accurate? | Independent event scorer exists; the demo counts are not independently validated |
| How much does it cost? | Dashboard scenario arithmetic with a supplied hourly rate; no provider price is assumed |

## Scope to explain clearly

This is a short, fixed-order experiment on a shared Windows laptop. CPU backends use their documented threading settings,
not an equal-thread-budget research comparison. Zero-image warm-up may not exercise every real-image postprocessing path.
COCO subset mAP is useful for export regression and quantization checks, but does not establish traffic-domain accuracy.
Repeated copies of one clip exercise compute load, not diverse scenes or RTSP reliability. Production adoption would need
longer runs, varied run order, independent traffic labels and sustained thermal/resource observations.

Keep claims tied to the checked-in measurements. The strongest demonstration is being able to explain both a successful
optimization and a configuration you rejected with evidence.
