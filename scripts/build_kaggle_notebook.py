"""Build a portable, self-contained Kaggle notebook without credentials, assets or historical results."""

import base64
import hashlib
import io
import json
import subprocess
import textwrap
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DESTINATION = REPO / "notebooks" / "kaggle"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def source_bundle():
    tracked = subprocess.check_output(["git", "ls-files", "src/traffic_lab/*.py"], cwd=REPO, text=True).splitlines()
    paths = tracked + ["pyproject.toml", "README.md", "scripts/kaggle_study.py", "scripts/precision_study.py",
                       "scripts/prepare_coco_holdout.py", "docs/KAGGLE.md", "src/traffic_lab/progress.py"]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(set(paths)):
            path = REPO / name
            if not path.is_file():
                raise FileNotFoundError(path)
            # Normalize source line endings so bundles are reproducible across Git checkouts/platforms.
            content = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 2, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, content)
    return stream.getvalue()


def build():
    data = source_bundle()
    encoded, digest = base64.b64encode(data).decode("ascii"), hashlib.sha256(data).hexdigest()
    cells = [cell("markdown", """
        # Traffic analytics: Kaggle smoke test → reproducible full study

        **Question:** How many paced traffic streams can one assigned GPU process within a freshness budget,
        while retaining COCO detection quality and independently measured crossing accuracy?

        1. Import this notebook into Kaggle. Keep it **private**. Enable **Internet** and an available **GPU**.
        2. Leave `MODE = "smoke"` and use **Save Version → Save & Run All**.
        3. Download the checkpoint ZIP and inspect `smoke-status.json` / failure logs.
        4. Only after smoke PASS, set `MODE = "full"`. A new session needs the saved checkpoint attached as input.

        Source code is embedded in this notebook. No GitHub token, private-repo clone or laptop files are required.
        A separate Python environment pins PyTorch/CUDA/TensorRT dependencies; initial setup downloads several GB.
        One GPU is used even if Kaggle assigns two. Your PC supplies only the browser.

        **Smoke is compatibility testing, not portfolio benchmark evidence.** The full study is multi-hour;
        checkpoint/resume is provided, but Kaggle availability/session limits still apply. No result has been
        precomputed or copied from the laptop study.
    """), cell("code", """
        # Edit this cell before running. Smoke is the default; full never starts automatically after smoke.
        MODE = "smoke"                    # "smoke" or "full"
        PHASE = "all"                     # full: "all", "models", "serving", or "tuning"
        RUN_ID = "traffic-kaggle-v1"       # change if hardware, packages, data, budgets or settings change
        IMAGE_SIZES = [320, 640]           # full model benchmark; serving uses 320
        TRAFFIC_CLIPS = []                 # optional /kaggle/input/.../*.mp4; default = official short demo
        EVENT_LABELS = {}                  # {"clip.mp4": "/kaggle/input/.../reviewed-events.json"}
        RESUME_ARCHIVE = ""                # optional /kaggle/input/.../traffic-kaggle-v1-checkpoint.zip
        INSTALL_DEPENDENCIES = True        # leave True; pip reuses the environment on a same-session rerun
        RUN_TUNING = True

        # Budgets are fixed BEFORE results. Model AP is a COCO proxy, not traffic-domain ground truth.
        MIN_CAMERA_FPS = 15
        MAX_FRAME_AGE_P95_MS = 100
        MAX_AP_LOSS_POINTS = 1.0
        MIN_COUNT_F1 = 0.95
        MAX_COUNT_ERROR_FRACTION = 0.05
        COUNTING_LINE = [0.1, 0.55, 0.9, 0.55]
        MODEL_TRIALS = 10
        MODEL_SECONDS = 20
        SERVING_REPEATS = 3
        SERVING_SECONDS = 30
    """), cell("code", f"""
        import base64, hashlib, io, json, os, pathlib, subprocess, sys, zipfile
        WORK = pathlib.Path("/kaggle/working")
        if not WORK.is_dir():
            raise RuntimeError("Run this notebook in a Kaggle GPU notebook, not on the laptop.")
        if sys.version_info[:2] not in ((3, 11), (3, 12)):
            raise RuntimeError("This pinned environment supports Python 3.11/3.12. Inspect the Kaggle image before proceeding.")
        LAB = WORK / "traffic-inference-lab"
        LAB.mkdir(exist_ok=True)
        SOURCE_BUNDLE_SHA256 = "{digest}"
        SOURCE_BUNDLE_BASE64 = "{encoded}"
        bundle = base64.b64decode(SOURCE_BUNDLE_BASE64)
        assert hashlib.sha256(bundle).hexdigest() == SOURCE_BUNDLE_SHA256
        def safe_extract(archive, destination):
            destination = pathlib.Path(destination).resolve()
            for item in archive.infolist():
                target = (destination / item.filename).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("Unsafe archive path")
            archive.extractall(destination)
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            safe_extract(archive, LAB)
        if RESUME_ARCHIVE:
            with zipfile.ZipFile(RESUME_ARCHIVE) as archive:
                safe_extract(archive, LAB)
        os.chdir(LAB)
        print("Verified embedded source:", SOURCE_BUNDLE_SHA256)
        print(subprocess.check_output(["nvidia-smi"], text=True))
    """), cell("code", """
        # All ML imports/work run in subprocesses, so notebook-kernel package state cannot leak into trials.
        # Keep several GB of installed libraries out of Kaggle's saved notebook outputs.
        ENV = pathlib.Path("/tmp/traffic-inference-venv")
        PYTHON = ENV / "bin/python"
        if not PYTHON.exists():
            subprocess.run([sys.executable, "-m", "venv", str(ENV)], check=True)
        sys.path.insert(0, str(LAB / "src"))
        from traffic_lab.progress import log_event, wait_with_progress
        def setup_step(arguments, logfile):
            log_event("SETUP " + logfile + ": " + " ".join(str(a) for a in arguments[:6]) + " ...")
            with open(WORK / logfile, "w") as log:
                status = subprocess.Popen([str(a) for a in arguments], stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                try:
                    code = wait_with_progress(status, WORK / logfile, "SETUP " + logfile, timeout=3600)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    import signal
                    os.killpg(status.pid, signal.SIGKILL)
                    status.wait()
                    raise
            log_event("SETUP " + logfile + ": exit=" + str(code))
            if code:
                print((WORK / logfile).read_text()[-12000:])
                raise RuntimeError("Setup failed; inspect " + logfile)
        if INSTALL_DEPENDENCIES:
            LOCKFILE = LAB / "results" / RUN_ID / "dependency-constraints.txt"
            setup_step([PYTHON, "-m", "pip", "install", "--upgrade", "pip"], "setup-pip.log")
            setup_step([PYTHON, "-m", "pip", "install", "torch==2.6.0", "torchvision==0.21.0",
                        "--index-url", "https://download.pytorch.org/whl/cu124"], "setup-torch.log")
            project_install = [PYTHON, "-m", "pip", "install", "-e", ".[gpu,int8,evaluation]"]
            if LOCKFILE.exists():
                project_install += ["-c", LOCKFILE]
            setup_step(project_install, "setup-project.log")
            packages = json.loads(subprocess.check_output([PYTHON, "-m", "pip", "list", "--format=json"], text=True))
            LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
            LOCKFILE.write_text("\\n".join(p["name"] + "==" + p["version"] for p in packages) + "\\n")
        # ORT needs access to the same CUDA/cuDNN shared libraries as PyTorch.
        site = pathlib.Path(subprocess.check_output([PYTHON, "-c", "import site; print(site.getsitepackages()[0])"], text=True).strip())
        libraries = [site / "torch/lib", site / "tensorrt_libs"] + list((site / "nvidia").glob("*/lib"))
        os.environ["LD_LIBRARY_PATH"] = ":".join(str(p) for p in libraries if p.is_dir()) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        print("Isolated Python:", PYTHON)
    """), cell("markdown", """
        ## Start the selected mode

        Smoke builds and inspects all nine exports, probes thirteen runtime/device combinations, evaluates
        eight disjoint images, and exercises one/two-camera tracking plus annotated rendering. TensorRT builds
        can make this take appreciable time. A smoke failure blocks full mode; read the corresponding log.

        Full mode runs the standardized precision study at 320/640, then selects candidates within a
        predeclared COCO AP budget for repeated 1/2/4/8-camera tests. Tuning tests rendering overhead,
        sampled telemetry, frame skipping, and separately evaluated dynamic TensorRT batching.

        For meaningful counting results, attach independently reviewed event labels and original traffic clips
        longer than 30 seconds. The short built-in clip is looped for full compute-load tests and cannot establish
        natural-traffic counting accuracy. Templates are exported in `annotation-kit/`; no labels are fabricated.
    """), cell("code", """
        settings = {
            "mode": MODE, "phase": PHASE, "run_id": RUN_ID, "image_sizes": IMAGE_SIZES,
            "traffic_clips": TRAFFIC_CLIPS, "event_labels": EVENT_LABELS, "run_tuning": RUN_TUNING,
            "min_camera_fps": MIN_CAMERA_FPS, "max_frame_age_p95_ms": MAX_FRAME_AGE_P95_MS,
            "max_ap_loss_points": MAX_AP_LOSS_POINTS, "min_count_f1": MIN_COUNT_F1,
            "max_count_error_fraction": MAX_COUNT_ERROR_FRACTION, "line": COUNTING_LINE,
            "model_trials": MODEL_TRIALS, "model_seconds": MODEL_SECONDS,
            "serving_repeats": SERVING_REPEATS, "serving_seconds": SERVING_SECONDS,
        }
        CONFIG = LAB / "kaggle-config.json"
        CONFIG.write_text(json.dumps(settings, indent=2))
        RUN_LOG = WORK / (RUN_ID + "-" + MODE + "-console.log")
        print("Starting", MODE, PHASE, "— raw progress also saved to", RUN_LOG, flush=True)
        with RUN_LOG.open("w") as logfile:
            process = subprocess.Popen([str(PYTHON), "-u", "scripts/kaggle_study.py", "--config", str(CONFIG)],
                                       cwd=LAB, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, start_new_session=True)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    logfile.write(line)
                    logfile.flush()
                returncode = process.wait()
            except KeyboardInterrupt:
                import signal
                os.killpg(process.pid, signal.SIGINT)
                process.wait()
                raise
        print("Exit code:", returncode)
        if returncode:
            print("Run failed or has failed stages. Inspect logs and smoke-status.json before full mode.")
    """), cell("code", """
        # Keep this cell after the runner: it still displays diagnostics when the runner returns nonzero.
        from IPython.display import FileLink, Markdown, Video, display
        OUTPUT = LAB / "results" / RUN_ID / MODE
        smoke_status = LAB / "results" / RUN_ID / "smoke/smoke-status.json"
        if smoke_status.exists():
            print("SMOKE STATUS:")
            print(smoke_status.read_text())
        if (OUTPUT / "REPORT.md").exists():
            display(Markdown((OUTPUT / "REPORT.md").read_text()))
        for path in sorted(OUTPUT.glob("precision-*/REPORT.md")):
            display(Markdown(path.read_text()))
        videos = list(OUTPUT.rglob("*-browser.mp4"))
        if videos:
            display(Video(str(videos[0]), embed=True))
        for path in [WORK / (RUN_ID + "-checkpoint.zip"), RUN_LOG, OUTPUT / "dashboard.html"]:
            if path.exists():
                display(FileLink(str(path)))
        print("Use Save Version → Save & Run All to retain outputs. Download the checkpoint ZIP.")
    """), cell("markdown", """
        ## Resume and interpretation

        Attach the checkpoint ZIP from a saved run as a private Kaggle input, set `RESUME_ARCHIVE` to that file,
        and keep the same RUN_ID/settings. Source, model and result identities are checked. Image caches are omitted
        from the ZIP and are re-downloaded/verified as needed. The environment must match; a different GPU/driver,
        host configuration, package version, video, labels or budget requires a new RUN_ID and another smoke test.

        `PHASE="models"`, then `"serving"`, then `"tuning"` lets you split work across saved sessions.
        Phase is not part of the experiment identity. Do not run two copies concurrently or benchmark both GPUs at once.

        A timing pass requires every camera in every repeat to meet FPS and freshness budgets. Counting is a separate
        gate requiring independent labels. COCO AP is a detection-quality proxy with no AP confidence interval here;
        a borderline difference is not a confident accuracy pass/fail. Model-only throughput is a reference, not a
        hard multi-camera ceiling. CPU results describe the assigned Kaggle host. Smoke outputs are never pooled
        with full-study results or the earlier laptop measurements.
    """)]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}, "accelerator": "GPU",
                "kaggle": {"isInternetEnabled": True, "isGpuEnabled": True, "language": "python",
                           "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 5}
    for index, item in enumerate(cells):
        item["id"] = f"traffic-{index:02d}"
    DESTINATION.mkdir(parents=True, exist_ok=True)
    target = DESTINATION / "Traffic_Analytics_Kaggle.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    metadata = {"id": "YOUR_KAGGLE_USERNAME/traffic-analytics-smoke-and-benchmark", "title": "Traffic Analytics Smoke and Benchmark",
                "code_file": target.name, "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (DESTINATION / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Built {target}; {len(cells)} cells; source bundle {len(data)} bytes / SHA256 {digest}")
    return target


if __name__ == "__main__":
    build()
