# Kaggle traffic analytics notebook

Upload `notebooks/kaggle/Traffic_Analytics_Kaggle.ipynb` to a **private** Kaggle notebook.
Select an available GPU (T4 is the intended target), enable Internet, and leave `MODE="smoke"`.
Use **Save Version → Save & Run All** so outputs are retained. The notebook embeds its source code;
it does not need access to the private GitHub repository or any tokens.

The notebook was structurally checked locally. It has **not been executed on Kaggle** from this workspace;
no Kaggle credentials are configured here. Its smoke gate is the required environment validation.

## Smoke first

Setup creates an isolated environment with PyTorch 2.6.0+CUDA12.4, torchvision 0.21.0,
ONNX Runtime GPU 1.20.2 and TensorRT 10.8.0.43, plus the pinned project runtime.
The notebook kernel can use a newer Python. Bootstrap uses uv to download an independent CPython 3.12
and create the worker environment, so pinned ML wheels do not depend on Kaggle's kernel version.
Kaggle's driver must still support the CUDA stack. Python bootstrap/download progress has separate logs.
The environment lives under `/tmp` so installed libraries are not exported as notebook outputs.
Resolved dependency constraints are saved with the study and applied when rebuilding a resumed environment.
Python management follows [uv's documented install workflow](https://docs.astral.sh/uv/guides/install-python/).
Package installation downloads several GB; TensorRT engine builds also take time. Do not assume a five-minute smoke.

Smoke checks:

- GPU0 CUDA execution and required packages. GPU1 is unused even if two T4s are assigned.
- One YOLO11n checkpoint, nine exports, thirteen CPU/CUDA/runtime/precision combinations.
- ONNX provider/dtype profiles, OpenVINO execution types and TensorRT engine inspection.
- Three-frame inference probes, official COCO evaluation on eight calibration-disjoint images,
  and 32-image INT8 calibration (compatibility only; not valid accuracy evidence).
- One/two-camera independent tracking, bounded queues, frame/drop accounting and annotated rendering.

Download `<RUN_ID>-checkpoint.zip`, console output, and `results/<RUN_ID>/smoke/smoke-status.json`.
If any required stage fails, inspect `logs/`. Full mode refuses to run without a matching smoke PASS.
The first smoke does not turn itself into a full run.

The notebook prints UTC timestamps, stage names, attempts, cached-stage notices, model-resolution and
trial/replay round counters, completed/failed stage counts, and elapsed times. Every 15 seconds a running
worker/setup step emits a heartbeat plus up to three recent log lines, even when a native builder is quiet.
Full worker logs and the console log remain on disk. A heartbeat means the process is alive, not proof
that an internal compiler/inference operation is making progress; timeouts still apply. Progress printing
stays in the parent process rather than the timed inference loop. If replacing code from an already-run
notebook version, choose a fresh RUN_ID because the saved study binds source hashes.

## Full study

After PASS, set `MODE="full"`. Keep the same RUN_ID and settings. If starting a new session,
attach the checkpoint ZIP as a private Kaggle input and set `RESUME_ARCHIVE` to its input path.
The default full run is multi-hour; use phases if it will not fit your remaining quota/session:

1. `PHASE="models"`: 320/640px standardized precision study. One checkpoint, static batch one,
   512 calibration / 4488 disjoint evaluation images, ten randomized >=20s real-warmed timing rounds per case.
   Timing filters vehicle classes; official COCO accuracy evaluates all classes. CPU/CUDA stay separate.
2. `PHASE="serving"`: select the lowest median-p95 eligible candidate per backend/device, then
   test 1/2/4/8 streams in three randomized >=30s replays. Queues have capacity one; every camera
   must satisfy the predeclared FPS and frame-age budgets in every repeat.
3. `PHASE="tuning"`: separately measure rendering overhead, sampled telemetry, detection stride two,
   and dynamic TensorRT batch four (only after a separate accuracy evaluation of that engine).

`PHASE="all"` executes these in order. No experiment configurations run concurrently.
These are same-session cloud-host experiments, not isolated bare-metal hardware rankings.
The model-only first-clip reference is displayed alongside serving results; it is not a strict capacity ceiling.

## Traffic clips and independent labels

The default short official demo allows a self-contained compatibility/load test. Full replay loops it
to the requested duration. Identical/looped views measure compute load, not scene diversity or natural
counting accuracy. For the original project claim, attach several **original recordings longer than 30s**
under `TRAFFIC_CLIPS`, and reviewed crossing annotations under `EVENT_LABELS`.

An annotation template is saved for every clip in `annotation-kit/`. It has `reviewed:false` and zero
events; neither property is changed automatically. Independently watch the original footage, using the
declared finite line and vehicle bottom-center. Supply events like:

```json
{
  "reviewed": true,
  "source_sha256": "EXACT_VIDEO_SHA256_FROM_TEMPLATE",
  "line": [0.1, 0.55, 0.9, 0.55],
  "events": [
    {"source_seconds": 4.2, "class_id": 2, "direction": "positive"}
  ]
}
```

This is a schema illustration, not real ground truth. Class IDs are YOLO COCO IDs: car2, motorcycle3,
bus5, truck7. For the default left-to-right horizontal line, positive is downward, negative upward.
At least five labelled crossings per camera/window are required by default. Matching is one-to-one,
same camera/class/direction, within 0.5s. Every camera must meet F1 and relative-count-error budgets.
Hash or line mismatches fail explicitly. Missing/unreviewed/insufficient labels and looped sources leave
counting **unvalidated**, never implicitly correct. Occupancy is displayed but lacks independent occupancy labels.
COCO AP is a proxy; it does not prove traffic-domain detection accuracy. AP CIs are not yet calculated.

## Results and recovery

Notebook cells show Markdown tables and an annotated video. `dashboard.html` is a standalone local
results viewer with repeated-run measurements and rendered clips; it does not expose a hosted web server.
Raw samples, environments, trial spread, per-camera drops/frame age, Jain fairness, selection decisions,
precision inspection and attempt logs are retained under `results/<RUN_ID>/`.

The checkpoint ZIP includes results, model engines, metadata, label templates and replay clips.
Large image caches/prediction arrays and the isolated Python environment are omitted. Re-prepare/verify
images when resuming. Use identical source clips and labels. Resume refuses changed hardware/driver,
host metadata, packages, code, model bytes or settings. A different assigned machine needs a fresh RUN_ID
and smoke test. Do not reuse Windows/GTX engines on Kaggle; all engines build on the assigned GPU.

Kaggle availability and session/quota limits still apply. Checkpoints are written after each completed model
resolution, serving round, phase or a handled failure, not continuously uploaded to external storage. A forced VM termination can lose
work since the last saved Kaggle output. Save/download completed phases before continuing.

The optional `kernel-metadata.json` supports Kaggle CLI import; replace YOUR_KAGGLE_USERNAME first.
It requests a private GPU notebook with Internet. `kaggle kernels push` starts execution; use `--no-run`
to upload only. UI import needs no metadata editing.

Regenerate the portable notebook after source changes:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/build_kaggle_notebook.py
```

Only selected project sources are embedded. No credentials, models, datasets, historical measurements
or unfinished overnight scripts are included.
