from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import psutil

from .provenance import file_hash

_DLL_HANDLES = []
if os.name == "nt":
    # Local, official Windows SDK; no machine-wide PATH or CUDA installation changes.
    sdk = Path(".build-cache/tensorrt-sdk").resolve()
    if sdk.is_dir():
        _DLL_HANDLES.append(os.add_dll_directory(str(sdk)))
        os.environ["PATH"] = str(sdk) + os.pathsep + os.environ.get("PATH", "")


def environment():
    packages = {}
    for name in (
        "traffic-inference-lab",
        "ultralytics",
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "onnxruntime",
        "onnxruntime-gpu",
        "openvino",
        "onnx",
        "nncf",
        "tensorrt",
        "tensorrt-cu12",
        "lap",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        gpu = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(),
        "ram_gib": psutil.virtual_memory().total / 1024**3,
        "gpu": gpu,
        "packages": packages,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass
class ModelConfig:
    model: str = "models/yolo11n.pt"
    imgsz: int = 320
    device: str = "cpu"
    half: bool = False
    threads: int = 4
    conf: float = 0.25
    iou: float = 0.7
    classes: tuple | None = (2, 3, 5, 7)


def execution_device(config):
    # Ultralytics may otherwise choose OpenVINO AUTO even when device='cpu'.
    model = Path(config.model)
    if config.device == "cpu" and (model.is_dir() or model.suffix == ".xml"):
        return "intel:cpu"
    return config.device


class Detector:
    def __init__(self, config: ModelConfig):
        import torch
        from ultralytics import YOLO

        if config.threads < 1 or config.imgsz < 32 or config.imgsz % 32:
            raise ValueError("threads >= 1 and imgsz a positive multiple of 32 are required")
        if config.half and config.device == "cpu":
            raise ValueError("CPU half inference is not a portable configuration; use a supported export")
        torch.set_num_threads(config.threads)
        cv2.setNumThreads(1)
        self.config = config
        self.model = YOLO(config.model, task="detect")
        self.calls = 0

    def predict(self, frames):
        c = self.config
        results = self.model.predict(
            frames,
            imgsz=c.imgsz,
            device=execution_device(c),
            conf=c.conf,
            iou=c.iou,
            classes=c.classes,
            rect=False,
            verbose=False,
            stream=False,
            quantize=16 if c.half else 32,
        )
        self.calls += 1
        return results

    def warmup(self, count=5, batch=1):
        image = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
        for _ in range(count):
            self.predict([image] * batch)
        if self.config.device not in ("cpu", "intel:cpu"):
            backend = self.model.predictor.model
            session = getattr(backend, "session", None)
            if session is not None and "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError("Requested CUDA inference but ONNX Runtime fell back to another provider")

    def metadata(self):
        import torch

        backend = self.model.predictor.model if self.model.predictor else None
        model_path = Path(self.config.model)
        size = (
            sum(p.stat().st_size for p in model_path.rglob("*") if p.is_file())
            if model_path.is_dir()
            else model_path.stat().st_size
        )
        return {
            "config": asdict(self.config),
            "model_sha256": file_hash(self.config.model),
            "model_size_bytes": size,
            "backend": type(getattr(backend, "backend", backend)).__name__,
            "execution_device_argument": execution_device(self.config),
            "openvino_execution_devices": list(backend.ov_compiled_model.get_property("EXECUTION_DEVICES"))
            if backend is not None and hasattr(backend, "ov_compiled_model")
            else [],
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "onnx_providers": getattr(getattr(backend, "session", None), "get_providers", lambda: [])(),
            "backend_input_fp16": bool(getattr(backend, "fp16", False)),
            "tensorrt_cuda_graph_active": bool(getattr(backend, "graph", None) is not None)
            if model_path.suffix == ".engine"
            else None,
            "openvino_runtime_threads": int(backend.ov_compiled_model.get_property("INFERENCE_NUM_THREADS"))
            if backend is not None and hasattr(backend, "ov_compiled_model")
            else None,
            "openvino_inference_precision_hint": str(
                backend.ov_compiled_model.get_property("INFERENCE_PRECISION_HINT")
            )
            if backend is not None and hasattr(backend, "ov_compiled_model")
            else None,
        }


def video_frames(path, limit=120, step=1):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    frames = []
    index = 0
    try:
        while len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                frames.append(frame)
            index += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("Video contained no decodable frames")
    return frames
