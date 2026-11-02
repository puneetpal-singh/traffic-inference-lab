# Traffic Inference Lab

**How many cameras can one machine serve while keeping results fresh and preserving accuracy?**

A computer vision deployment project built around that question: reproducible YOLO runtime comparisons,
export parity gates, independent vehicle tracking and counting, bounded multi-camera scheduling, and a local results dashboard.

**Latest measured study: Kaggle T4, 320px.** Completed 13 model configurations (130 randomized timing trials,
4488 calibration-disjoint accuracy images) and 72 repeated serving runs. TensorRT FP16 enabled recorded
**2.69× paired throughput versus eager PyTorch CUDA FP32**; TensorRT INT8 had similar throughput with a larger
measured AP loss. Two cameras were the largest timing-passing tested GPU load; counting remains unvalidated.
See the [verified Kaggle report](results/kaggle-t4-320-2026-11-02/Kaggle_Results_Analysis.md) and
[scope and evidence](docs/KAGGLE_RESULTS.md). The laptop experiments below are a separate historical study.

Measured on an i7-10750H / GTX 1650 Ti: **28 model configurations and 24 serving experiments**, including CPU/CUDA,
FP32/FP16/INT8, camera scaling, batching, tracking, rendering, stride and telemetry tuning. See the
[measured findings](docs/RESULTS.md), [portable report](results/report.html), and [raw evidence](results/full-matrix/).

For the focused **one-checkpoint FP32/FP16/INT8 study** across ONNX Runtime, OpenVINO and TensorRT,
use the [new runner and launch instructions](docs/PRECISION_STUDY.md). It defaults to a dry run;
the new study has not been executed and does not replace the historical results below.

To run the **multi-camera project on Kaggle**, upload the [self-contained notebook](notebooks/kaggle/Traffic_Analytics_Kaggle.ipynb).
It defaults to a smoke test and gates the full benchmark on a matching PASS. See [Kaggle instructions](docs/KAGGLE.md).
The notebook has been checked locally; Kaggle execution still needs to be performed in your account.

## Review the project

<!-- confirmation:begin -->
**Same-GPU confirmation:** TensorRT FP16 enabled recorded **3.39× paired model-throughput speedup** versus PyTorch CUDA FP32
at 320px on the GTX 1650 Ti. Aggregate throughput was **39.58 → 134.17 FPS** across 6 balanced randomized trial pairs
with real-image warm-up and at least 20 seconds per trial. Matched-artifact accuracy budget: **PASS**.
The [confirmation protocol](docs/CONFIRMATION.md) explains the quality checks, uncertainty interval and differences from the original matrix. Decoding, tracking and rendering are excluded.
<!-- confirmation:end -->

- [Same-GPU confirmation](docs/CONFIRMATION.md): balanced randomized trials with real-image warm-up and a paired-trial uncertainty interval.
- [CV description and defensible claims](docs/CV_READY.md): measured results and an interview walkthrough.
- Open [the standalone HTML report](results/report.html) locally to review the complete study without installing inference dependencies.

![Same-GPU optimization confirmation](results/confirmation/same-gpu-comparison.png)

```mermaid
flowchart LR
    A[Recorded camera replay] --> B[Bounded per-camera queues]
    B --> C[Fair batch scheduler]
    C --> D[Shared YOLO detector]
    D --> E[Independent camera trackers]
    E --> F[Finite-line crossing counters]
    F --> G[Frame telemetry and events]
    G --> H[HTML report / Streamlit viewer]
```

## Run locally

Python 3.11–3.13; Python 3.12 recommended. Start in this repository.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[cpu,dashboard,dev]"
traffic-lab doctor
traffic-lab fetch
```

On Linux/macOS use `python3.12 -m venv .venv` and `source .venv/bin/activate`.
The CPU wheels are intentional for the CPU comparison. Use a separate environment with a compatible CUDA-enabled
PyTorch build for GPU experiments. Do not assume that an installed NVIDIA driver makes CPU-only PyTorch use CUDA.

## Complete CPU, CUDA, ONNX, OpenVINO and TensorRT matrix

The expanded study covers **28 runtime/precision/resolution combinations** and uses **512 calibration images plus 256
disjoint COCO val2017 evaluation images**. See [the full protocol](docs/FULL_MATRIX_PROTOCOL.md).

Windows setup for the tested GTX 1650 Ti, using a separate environment:

```powershell
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[gpu,int8,dashboard,dev]"
python scripts/bootstrap_tensorrt_windows.py
python -m pip install .build-cache/tensorrt-sdk/tensorrt-10.8.0.43-cp312-none-win_amd64.whl
traffic-lab doctor
traffic-lab fetch
python scripts/prepare_coco_holdout.py
python scripts/full_matrix.py
python scripts/run_pipeline_matrix.py
python scripts/audit_full_matrix.py --require-pipelines
traffic-lab report --input results --output results/report.html
python scripts/write_full_findings.py
python -m streamlit run dashboard.py
```

The SDK helper downloads NVIDIA's official Windows archive and extracts runtime DLLs and Python bindings into the ignored
local cache. It does not change the machine-wide PATH. Engines are built locally on the deployment GPU; do not reuse them
across unrelated devices/runtimes. If the CUDA wheel mirror stalls, `scripts/download_cuda_wheel.py` retrieves the official
Windows wheel using ranges and verifies its SHA-256 against the official index.

`full_matrix.py` resumes successful cases; `--force` reruns them and `--only tensorrt` selects TensorRT. Raw timing, held-out
mAP, graph/engine inspection, NVML telemetry and failure history are under `results/full-matrix/`.
FP16 storage and native FP16 compute are distinguished in the report. Strict box parity is diagnostic for INT8; measured
accuracy loss determines whether quantization meets the chosen budget.

The original CPU smoke suite remains reproducible separately:

```powershell
python scripts/run_experiments.py --evaluate
python -m streamlit run dashboard.py
```

Open [the checked-in HTML report](results/report.html) by downloading/opening it locally, or launch the dashboard.
The HTML report is standalone; GitHub's file preview does not execute it. See [the measured findings](docs/RESULTS.md)
and [methodology](docs/METHODOLOGY.md) before quoting a result.

## Individual experiments

```powershell
traffic-lab export --format onnx --dynamic --batch 4 --output-dir models/onnx
traffic-lab benchmark --model models/onnx/yolo11n.onnx --source data/traffic.mp4 --imgsz 320 --output results/onnx-320.json
traffic-lab parity --candidate models/onnx/yolo11n.onnx --source data/traffic.mp4 --output results/parity-onnx-320.json
traffic-lab evaluate --model models/onnx/yolo11n.onnx --data coco128.yaml --output results/eval-onnx-320.json
traffic-lab run --sources data/traffic.mp4 data/traffic.mp4 --duration 5 --output results/two-cameras --render
traffic-lab report --input results --output results/report.html
traffic-lab gate --benchmark results/onnx-320.json --evaluation results/eval-onnx-320.json --baseline-evaluation results/eval-pytorch-320.json --output results/gate-onnx.json
```

For `run`, change `--tracker botsort`, `--stride 2`, `--queue-size 1`, `--batch 2`, and
`--line 0.1 0.55 0.9 0.55` independently to study their effects. Batches are opportunistic maxima;
the scheduler does not wait to fill them. Use dynamic exports when shapes/batch sizes vary.
`--zone x1 y1 x2 y2` controls the normalized occupancy rectangle. If FFmpeg is installed,
rendered videos also receive an H.264 browser copy after the timed run; otherwise use a local video player.
The dashboard plots tracked-position density from normalized bottom-center observations and includes a cost-sensitivity
calculator using your supplied machine hourly rate and the selected replay's measured processing rate.
`--brightness-grid 64` measures approximate brightness from at most 64×64 sampled pixels; the default zero uses the full
image. The serving study preserves separate baseline and sampled-telemetry runs to measure the observability overhead.

Additional export and event-quality commands:

```powershell
# Build on the NVIDIA deployment machine with compatible CUDA/TensorRT installed.
traffic-lab export --format engine --device 0 --half --dynamic --batch 4 --output-dir models/trt-fp16
# Calibrate only on an independent, representative dataset; install .[int8] first.
traffic-lab export --format openvino --int8 --data path/to/calibration.yaml --output-dir models/ov-int8
# Score human-annotated line-crossing events, not model-generated labels.
traffic-lab count-score --predictions results/two-cameras/events.json --labels path/to/labels.json --output results/count-quality.json
# Each drift input must contain one camera.
traffic-lab drift --reference path/to/reference/frames.jsonl --current path/to/current/frames.jsonl --output results/drift.json
```

## What this demonstrates

- **Profiling:** isolate decode/queue/model/tracker/render time and tail latency.
- **Runtime engineering:** consistent preprocessing, explicit export settings, and detection parity checks.
- **Load handling:** bounded latest-frame queues, drop accounting, fair scheduling, per-camera tracker isolation.
- **Optimization judgment:** compare runtime/input-size/batching/stride tradeoffs instead of assuming faster is better.
- **Evidence:** raw measurements, hashes, environment records, repeatable commands, and explicit failure logs.
- **Quality discipline:** labeled mAP evaluation and independent event scoring; unknown quality remains unknown.

The initial evaluation used COCO128 as a **smoke set**; the expanded matrix uses a separate COCO val2017 holdout. Neither is an independent traffic benchmark. The short road clip
tests the implementation and runtime behavior; repeat it on long, diverse clips for stable tail/capacity claims.
Counts in demos are unvalidated. No training, calibrated speed, production SLA, cloud savings, or energy efficiency is claimed.

## Repository map

| Path | Purpose |
|---|---|
| `src/traffic_lab/runtime.py` | Model adapter, hardware metadata, artifact hashes |
| `benchmark.py` | Repeated timing, export parity, labeled evaluation |
| `pipeline.py` | Paced replay, bounded queues, per-camera tracking |
| `metrics.py`, `scoring.py` | Counting, drift diagnostics, Pareto selection, event scoring |
| `report.py`, `dashboard.py` | Portable report and local results viewer |
| `scripts/run_experiments.py` | Sequential isolated benchmark processes |
| `results/` | Actual measurements and experiment logs |
| `tests/` | Queue/crossing/matching/drift/report regression tests |

```powershell
python -m pytest -q
python -m ruff check .
python scripts/audit_full_matrix.py --require-pipelines
python scripts/audit_confirmation.py
```

Read [data provenance](docs/DATA.md), [benchmark protocol](docs/FULL_MATRIX_PROTOCOL.md), and
[deployment decisions](docs/DECISIONS.md). Weights, videos, calibration images, and secrets are excluded from Git.
Use the [project walkthrough](docs/WALKTHROUGH.md) to present the measurements and optimization decisions.

## Learning references

Based on the questions covered by [Ultralytics YOLO in Production](https://academy.ultralytics.com/courses/yolo-in-production):
[deployment target](https://academy.ultralytics.com/courses/yolo-in-production/choose-a-deployment-target),
[ONNX parity](https://academy.ultralytics.com/courses/yolo-in-production/export-to-onnx),
[TensorRT](https://academy.ultralytics.com/courses/yolo-in-production/optimize-with-tensorrt),
[OpenVINO](https://academy.ultralytics.com/courses/yolo-in-production/openvino-on-cpu),
[tracking](https://academy.ultralytics.com/courses/yolo-in-production/tracking-with-bytetrack-and-botsort),
[counting](https://academy.ultralytics.com/courses/yolo-in-production/counting-heatmaps-and-speed),
[multi-stream inference](https://academy.ultralytics.com/courses/yolo-in-production/multi-stream-inference),
[observability](https://academy.ultralytics.com/courses/yolo-in-production/observability-and-drift), and
[cost/latency tuning](https://academy.ultralytics.com/courses/yolo-in-production/cost-and-latency-tuning).
Implementation uses the [export](https://docs.ultralytics.com/modes/export/),
[tracking](https://docs.ultralytics.com/modes/track/), and
[OpenVINO](https://docs.ultralytics.com/integrations/openvino/) APIs.
