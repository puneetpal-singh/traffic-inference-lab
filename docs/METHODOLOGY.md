# Benchmark protocol

This documents the original CPU smoke study. The expanded CPU/CUDA/TensorRT study uses
[the full matrix protocol](FULL_MATRIX_PROTOCOL.md), an independent COCO val2017 calibration/evaluation split,
and explicit decoder-thread budgets for the new serving experiments.

## Question and deployment gate

Find the runtime/configuration that sustains the required streams without excessive frame age or accuracy loss.
An example target is **two 30 FPS inputs, p95 frame age below 100 ms, at least 15 processed FPS per camera,
and no more than 1 absolute mAP50–95 percentage point lost relative to the baseline on a traffic holdout**.
This is a proposed acceptance gate, not a claim that the checked-in runs meet it.
The shorter COCO128 smoke evaluation cannot certify it. Counting also needs independent event labels.

## Model-only experiment

- Same checkpoint, input hash, vehicle classes (2, 3, 5, 7), confidence .25, NMS IoU .7, square letterboxing.
- YOLO11n is a deliberately small, established baseline. No training or novel-model claim.
- Input sizes 320 and 640; default batch one, four PyTorch CPU threads.
- Decode a fixed prefix before timing. Record actual frame count if shorter than requested.
- Warm up five calls including backend compilation and allocation. No model download/export in timing.
- Three repeated trials, complete raw batch observations, wall p50/p95/p99, trial FPS, sampled process RSS.
- GPU predictions must complete a CPU box conversion before wall timing stops.
- Batch latency is measured per batch; do not divide it and call it live frame latency.
- Only one timed experiment runs at a time. Run order is fixed, not randomized; thermal/order effects remain.
- PyTorch thread count is configured. ONNX/OpenVINO use backend defaults unless explicitly recorded otherwise;
  a wrapper's thread flag must not be interpreted as a portable backend thread budget.
- Process CPU percentage uses 100% per fully occupied logical core; it may exceed 100%.
- RSS includes predecoded frames and imported libraries, is sampled, and is not VRAM or power measurement.

## Parity and quality

`parity` matches final detections one-to-one, by class and box overlap, with IoU >= .90 and confidence delta <= .02.
It fails if any reference or candidate detection is unmatched. Empty/empty frames pass; they provide no positive evidence.
This strict gate can fail around a confidence/NMS boundary without implying a large mAP change; investigate the raw frame rows.
Exit code 2 marks a failed parity gate. Do not silently loosen tolerances to manufacture a pass.

`evaluate` computes mAP using low confidence .001 and the same square input transform. It evaluates all COCO classes,
separately from the vehicle-filtered performance workload. COCO128 is a tiny general-purpose smoke set derived from training
imagery, so performance on it is not an unbiased test of generalization. Export agreement is useful; absolute accuracy is not
a traffic deployment claim. INT8 requires a representative, disjoint calibration YAML supplied explicitly by the user.

## Paced multi-camera replay

Each producer decodes a local file at its metadata FPS, timestamped at its scheduled arrival. The oldest queued frame is
discarded when a bounded per-camera queue fills. A round-robin scheduler takes at most one frame per camera per prediction.
It does not wait to fill a batch: batch is a maximum. Dynamic exports are required for variable shapes/batches.
Warm-up covers the configured maximum batch; first use of other dynamic batch shapes can still incur compilation overhead.
One model serves all cameras, with independent ByteTrack or BoT-SORT instances and line counters.

Record decode time, decoder lag, queue wait, prediction batch time, tracking/counting time, optional rendering time,
and frame age (scheduled arrival through analytics/render completion). Logging the record occurs after that endpoint.
Track processed, stride-skipped, queue-dropped, and remaining frames. At successful completion:

`decoded = processed + stride_skipped + queue_dropped + remaining`

Processed FPS uses wall duration including final drain. Actual source duration can be shorter than requested.
Duplicating one file across cameras creates compute load, not diverse scenes or realistic RTSP/network behavior.
Dropping frames and detector stride change the tracker update time scale. Frame skipping is an experiment whose counting
quality must be checked, not a universally safe optimization. Rendered output compresses gaps between processed frames.

## Counting and observability

Line crossing uses bottom-center anchors and finite normalized line endpoints, a deadband, and per-ID/per-direction
deduplication. Tracker ID switches can still double-count. State has a finite TTL.
Annotate independent crossings with `camera`, `direction`, `class_id`, `source_seconds`, then use `count-score`.
Temporal one-to-one matching reports event precision/recall/F1 as well as absolute count error; a matching total alone can
hide both misses and duplicates. Do not use generated predictions as ground truth.

Per-frame confidence, brightness, object counts and latency are logged. `drift` compares subsampled single-camera confidence
distributions using a KS test plus effect-size threshold and minimum sample count. Correlated video and multiple tests limit
interpretation. A shift indicates review is needed; it does not prove accuracy loss. No real-world speed estimate is reported
without a calibrated scene geometry. No energy/cost claim is reported without measured power or dated price assumptions.
