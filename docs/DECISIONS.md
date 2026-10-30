# Deployment decisions

## Start with a CPU comparison

The local machine has an i7-10750H and GTX 1650 Ti (4 GiB). A CPU-only Python environment makes the PyTorch,
ONNX Runtime CPU provider, and OpenVINO CPU comparison coherent. GPU hardware inventory is not a GPU inference result.
TensorRT is an explicit optional path requiring a compatible runtime and an engine built for the deployment device.

## Use one shared model, separate trackers

Models are expensive; trackers own temporal identity state. Sharing one tracker across independent sources can mix IDs
and corrupt counts. One model predicts a group; each result goes to the matching camera's tracker and counter.

## Prefer bounded age under overload

For online traffic analytics, old results can be less useful than deliberate frame drops. The queue discards the oldest
waiting frame. We report drops, processing rate and frame age together: apparent low latency obtained by dropping almost
everything is not success. Archival processing needs a different offline policy that preserves every frame.

## Keep the demonstration small and honest

YOLO11n avoids making model training the project. Initial work emphasizes exports, profiling and system behavior.
The short clip and COCO128 support functional checks and preliminary comparisons only. A proper next validation uses
long traffic recordings, independently labeled frames/events, longer warm-up and measurement, varied run order,
and sustained thermal monitoring. Do not select the final production runtime solely from this smoke workload.

## Defer unsupported claims

- INT8: needs representative disjoint calibration and measured accuracy loss.
- Road speed: needs known scene geometry, camera calibration and ground truth.
- Energy/cost: needs measured energy and/or dated deployment-price assumptions.
- Drift: a distribution shift is a diagnostic; review ground truth before claiming model degradation.
- Tracking quality: event scoring is implemented; HOTA/IDF1 requires an appropriate annotated MOT dataset.

These extensions are useful follow-up experiments, not results implied by the existence of an export command.
