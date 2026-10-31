# One checkpoint, different precisions

This runner studies **one pretrained YOLO11n checkpoint** at a fixed resolution and batch one.
It does not train a model, change its architecture, or run the tracking/serving pipeline.
The script is [scripts/precision_study.py](../scripts/precision_study.py).
**Its default action is a dry run. Nothing starts until you explicitly use `--run`.**

## Run later on this PC

Open PowerShell in `D:\M\New_Project\traffic-inference-lab` and use the existing GPU environment:

```powershell
# Preview only; no downloads or inference.
.\.venv-gpu\Scripts\python.exe scripts/precision_study.py --dry-run

# Install the official COCO evaluator if needed. No experiments run in this step.
.\.venv-gpu\Scripts\python.exe -m pip install -e ".[evaluation]"

# Prepare 512 calibration and 4488 disjoint evaluation images. Downloads data, not GPU inference.
.\.venv-gpu\Scripts\python.exe scripts/precision_study.py --prepare-data

# When ready: plug in, close background work, choose a consistent power plan, then explicitly start.
.\.venv-gpu\Scripts\python.exe -u scripts/precision_study.py --run
```

The model `models/yolo11n.pt` and clip `data/traffic.mp4` already exist in this workspace.
On a fresh checkout, first follow the README's environment setup and `traffic-lab fetch` instructions.
Use the pinned Ultralytics 8.4.193 and the existing CUDA/TensorRT environment; the runner checks prerequisites.
The default `all` target needs the CPU/ONNX/OpenVINO, GPU/TensorRT, INT8 and evaluation dependencies.
Windows TensorRT SDK DLLs are loaded from the existing `.build-cache/tensorrt-sdk` location.
On Linux, use the activated environment's `python` in place of the Windows executable.

This is a multi-hour study: timing alone has a **43.3-minute lower bound**, before model loading,
real-image warm-up, precision inspection, engine builds and 13 evaluations of 4488 images each.
The script never promises a completion time or switches power plans. It requests Windows system
wakefulness while its sequential jobs run; that does not protect against a shutdown, lost power or forced sleep.

## Configurations

| Device | Runtime | Precisions |
|---|---|---|
| CPU | PyTorch reference | FP32 |
| CPU | ONNX Runtime | FP32, FP16 graph, INT8 |
| CPU | OpenVINO | FP32, FP16 storage, INT8 |
| CUDA | PyTorch reference | FP32 |
| CUDA | ONNX Runtime | FP32, FP16 graph |
| CUDA | TensorRT | FP32, FP16-enabled, INT8-enabled |

There are **13 runtime/device/precision cases and 9 distinct exports**. ONNX CPU/CUDA reuse the same
FP32 and FP16 graph artifacts. ONNX CUDA INT8 is deliberately omitted because this project's stack
does not provide a verified CUDA INT8 execution path for that exported graph. A CPU fallback must
not be presented as GPU INT8 performance.

All exports use the exact same checkpoint, fixed 320x320 input and static batch one. Model/engine
binaries are stored under the ignored `models/precision-study/` directory. Small export manifests,
measurements, logs and reports go under `results/precision-study/`.

## Accuracy protocol

- Official COCO val2017 annotations/images; split seed **20261030**. No training or fine-tuning.
- **512 calibration images** and **4488 evaluation images**, with zero image-ID overlap.
- Existing downloaded images are reused when possible; otherwise data is downloaded from the COCO S3 mirror.
- Calibration YAML points only to calibration images. All INT8 exports receive the same declared pool.
  Backend calibration algorithms and internal sample limits can differ: review the retained export logs
  before claiming every engine consumed all 512 images. This is a comparison of deployment recipes,
  not an isolated comparison of quantization algorithms.
- Detection confidence `.001`, NMS IoU `.7`, all 80 classes, `rect=False`, batch one.
- Official `pycocotools.COCOeval` computes bbox AP, including crowd handling, IoU .50:.95,
  area all and maximum 100 detections per image for the primary score. Inference uses the library's
  default maximum 300 detections before evaluation.
- Reports include mAP50-95, mAP50 in raw quality JSON, and mean AP of car/motorcycle/bus/truck.
- Accuracy loss is in **absolute percentage points**, both against the fixed PyTorch FP32 reference
  and against the same runtime/device's FP32 result. Negative loss is not proof of an accuracy improvement.
- These are **point estimates without AP confidence intervals**. A one-point budget is reported
  descriptively; a near-threshold result is not a statistically established pass/fail.
- Primary scores cover 4488 calibration-disjoint images, not the official full 5000-image leaderboard set.
  Some overlap the earlier project's 256-image exploratory holdout; this is not an entirely untouched final test set.

## Timing protocol

- Default **10 randomized rounds**, one trial per configuration per round. Each trial lasts at least **20 seconds**.
- Each trial loads one model in an isolated subprocess. Model loading and export are outside the timer.
- A pool of 60 predecoded full-resolution clip frames uses source indices 0,2,...,118.
  One complete pool of real images warms up every trial. Timed trials use whole pool cycles only.
- Wall time includes preprocessing, prediction, postprocessing and completed CPU box transfer.
  It excludes decoding, tracking and rendering. Confidence `.25`, NMS IoU `.7`, all classes.
- Equal configured thread budgets: PyTorch 4, ORT intra-op 4/inter-op 1, OpenVINO inference threads 4
  and one stream; OpenCV one thread. Actual runtime metadata is saved. This is equal configuration,
  not proof of equal core utilization. cuDNN autotuning is explicitly disabled for the baseline.
- Reports show median/min/max FPS across trials and median trial p95 latency. Raw frame samples are retained.
- Within-runtime precision speedups pair the same randomized rounds and report exploratory log-ratio
  t-intervals. Shared-host drift can violate independence; do not present them as multi-day repeatability.
- Backend-reported preprocess/inference/postprocess timers are retained as diagnostics. They are **not**
  a standardized GPU-only CUDA-event benchmark.
- Process RSS, available RAM and supported NVML memory/utilization/temperature/clock/power measurements
  are sampled. RSS includes the decoded frame pool; GPU readings cover the whole device.
  Unsupported GPU metrics remain absent. Thermal clock-event bits are checked at pool boundaries.
- Hot/low-memory/thermal-event trials are flagged and retained, never silently discarded.

## Precision verification

Untimed inspection records ONNX kernel providers and weighted-kernel input dtypes, OpenVINO execution
types and TensorRT engine layer information. Failed inspection prevents timing/quality claims for that case.
FP16 storage does not guarantee FP16 CPU arithmetic; FP16/INT8 TensorRT flags permit mixed precision.

## Resume, failures and outputs

Run the **same command** again after interruption. Completed stages are reused only when their saved
artifact hashes and job identities match. Interrupted/failed stages retry and keep separate attempt logs.
Windows subprocess timeouts stop the worker's process tree, avoiding an orphan worker competing with the next trial.
Only one runner can use an output directory at a time; stale process locks are detected after a crash.

The saved plan binds checkpoint, clip, dataset content, runner/helpers/runtime source, package versions,
GPU/hardware metadata and the Windows power plan. If those or study settings change, the runner refuses
to mix results: use a **fresh output directory**. Changing GPU/platform also requires fresh TensorRT engines.

```powershell
# CPU-only study, separate results and exports.
.\.venv-gpu\Scripts\python.exe scripts/precision_study.py --run --target cpu --output results/precision-cpu

# 640px is a separate study, using the same prepared data.
.\.venv-gpu\Scripts\python.exe scripts/precision_study.py --run --imgsz 640 --output results/precision-640

# Rebuild reports from saved evidence without running inference.
.\.venv-gpu\Scripts\python.exe scripts/precision_study.py --report-only
```

Main outputs under `results/precision-study/`:

- `REPORT.md`: separate CPU/CUDA tables and interpretation.
- `summary.csv`, `summary.json`: accuracy, loss, artifact size, sampled memory, timing spread and paired ratios.
- `plan.json`, `ledger.json`: immutable study identity and per-stage completion/retry state.
- `timing/`, `quality/`, `inspect/`, `exports/`, `logs/`: raw evidence and failure diagnostics.

Reports update as rounds and evaluations finish. Missing cases are labelled **pending**, not fabricated,
zero-filled or replaced with historical measurements. Large prediction JSONs remain in the ignored data directory.

For a small **exploratory rehearsal**, prepare a separate dataset and output with `--evaluation-limit 256`.
Use identical flags for preparation and running; label that study as a subset, not the main accuracy result.

```powershell
.\.venv-gpu\Scripts\python.exe scripts/precision_study.py --prepare-data --evaluation-limit 256 --data-root data/precision-small
.\.venv-gpu\Scripts\python.exe scripts/precision_study.py --run --evaluation-limit 256 --data-root data/precision-small --output results/precision-small
```
