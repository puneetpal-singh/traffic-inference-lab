# Deployment decisions

## Use one tested environment for the expanded study

The local machine has an i7-10750H and GTX 1650 Ti (4 GiB). The original study used CPU-only PyTorch. The expanded matrix
uses a separate CUDA-enabled environment for every CPU and GPU case: PyTorch 2.6/CUDA 12.4, ONNX Runtime GPU 1.20.2,
OpenVINO 2026.4.1 and TensorRT 10.8. GPU execution and TensorRT builder creation are verified before any experiment starts.
Windows TensorRT DLLs are loaded from a local official SDK; the machine-wide PATH is unchanged.

## Treat precision as a measured choice

FP16 and INT8 are not assumed improvements. Each exported model receives timing, held-out accuracy evaluation, detection
agreement diagnostics, and execution inspection. OpenVINO FP16 stored constants are distinguished from CPU arithmetic.
ONNX CUDA profiles must show actual weighted CUDA kernels. TensorRT precision flags allow mixed precision and engines
are built on the deployment device with a bounded workspace. Reject configurations that exceed the accuracy-loss budget.

Calibration uses 512 images disjoint from the 256-image COCO val2017 evaluation set. The tiny original COCO128 study is
kept as historical smoke evidence, not reused to claim deployment accuracy for the expanded matrix.

## Use one shared model, separate trackers

Models are expensive; trackers own temporal identity state. Sharing one tracker across independent sources can mix IDs
and corrupt counts. One model predicts a group; each result goes to the matching camera's tracker and counter.

## Prefer bounded age under overload

For online traffic analytics, old results can be less useful than deliberate frame drops. The queue discards the oldest
waiting frame. We report drops, processing rate and frame age together: apparent low latency obtained by dropping almost
everything is not success. Archival processing needs a different offline policy that preserves every frame.

The expanded serving experiments request one FFmpeg decoder thread per camera and record the actual count. Decoder
oversubscription must be distinguished from model saturation. Schema-v2 frame age includes metadata preparation; disk
logging remains outside that endpoint. TensorRT batching uses a separately built and accuracy-checked dynamic engine.
Stride filtering occurs after decoding, so it does not eliminate compressed-video decode work. The serving traces expose
decoder lag separately from queueing, model, tracking and telemetry time. Sampled brightness is measured as an optional
observability optimization, preserving the full-frame diagnostic as the baseline rather than assuming its overhead is free.

## Keep the demonstration small and honest

YOLO11n avoids making model training the project. Initial work emphasizes exports, profiling and system behavior.
The short clip and COCO128 support functional checks and preliminary comparisons only. A proper next validation uses
long traffic recordings, independently labeled frames/events, longer warm-up and measurement, varied run order,
and sustained thermal monitoring. Do not select the final production runtime solely from this smoke workload.

## Defer unsupported claims

- Traffic-domain acceptance: the general COCO holdout does not replace independently labeled traffic frames and events.
- Road speed: needs known scene geometry, camera calibration and ground truth.
- Energy/cost: power telemetry is recorded where supported; the dashboard cost calculator uses a user-supplied hourly
  rate and observed processed FPS. No provider price, measured energy saving or production capacity is inferred.
- Drift: a distribution shift is a diagnostic; review ground truth before claiming model degradation.
- Tracking quality: event scoring is implemented; HOTA/IDF1 requires an appropriate annotated MOT dataset.

These extensions are useful follow-up experiments, not results implied by the existence of an export command.
