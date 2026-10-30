"""Inspect exported graphs and actual execution placement separately from timing."""

import json
from collections import Counter
from pathlib import Path

import numpy as np


def inspect_artifact(config, output):
    from .runtime import Detector, save_json

    path = Path(config.model)
    evidence = {"model": str(path), "device": config.device}
    if path.suffix == ".onnx":
        import onnx
        import onnxruntime as ort
        import torch

        # Loading torch first registers its CUDA/cuDNN DLL directories on Windows.
        if config.device != "cpu":
            torch.cuda.init()
        model = onnx.load(str(path), load_external_data=False)
        evidence["initializer_types"] = dict(
            Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in model.graph.initializer)
        )
        evidence["graph_operations"] = dict(Counter(n.op_type for n in model.graph.node))
        del model
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(output).with_suffix(""))
        providers = (
            ["CPUExecutionProvider"]
            if config.device == "cpu"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        input_info = session.get_inputs()[0]
        dtype = np.float16 if "float16" in input_info.type else np.float32
        image = np.zeros((1, 3, config.imgsz, config.imgsz), dtype=dtype)
        for _ in range(3):
            session.run(None, {input_info.name: image})
        profile = session.end_profiling()
        events = json.loads(Path(profile).read_text())
        kernels = [e for e in events if e.get("cat") == "Node" and e.get("args", {}).get("provider")]
        evidence["registered_providers"] = session.get_providers()
        evidence["executed_kernel_providers"] = dict(Counter(e["args"]["provider"] for e in kernels))
        weighted = [
            e
            for e in kernels
            if any(k in e.get("args", {}).get("op_name", "").lower() for k in ("conv", "gemm", "matmul"))
        ]
        evidence["weighted_kernel_providers"] = dict(Counter(e["args"]["provider"] for e in weighted))
        evidence["profile"] = profile
        if config.device != "cpu" and not any(
            e["args"]["provider"] == "CUDAExecutionProvider" for e in weighted
        ):
            raise RuntimeError("CUDA was requested but weighted kernels executed entirely on CPU")
    elif path.suffix == ".engine" or path.is_dir():
        detector = Detector(config)
        detector.warmup(2)
        backend = detector.model.predictor.model
        evidence.update(detector.metadata())
        if path.suffix == ".engine":
            import tensorrt as trt

            engine = backend.model
            inspector = engine.create_engine_inspector()
            inspector.execution_context = backend.context
            details = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
            evidence["engine_layers"] = details
            evidence["engine_layer_count"] = engine.num_layers
            evidence["engine_device_memory_bytes"] = int(engine.device_memory_size_v2)
            evidence["profiling_verbosity"] = str(engine.profiling_verbosity)
        else:
            compiled = backend.ov_compiled_model
            precisions = []
            for op in compiled.get_runtime_model().get_ops():
                info = op.get_rt_info()
                if "runtimePrecision" in info:
                    value = info["runtimePrecision"]
                    precisions.append(str(value.value if hasattr(value, "value") else value))
            evidence["runtime_precision_counts"] = dict(Counter(precisions))
            evidence["execution_devices"] = list(compiled.get_property("EXECUTION_DEVICES"))
    else:
        import torch

        evidence["torch_cuda_version"] = torch.version.cuda
        evidence["requested_fp16"] = config.half
        if config.device != "cpu":
            evidence["compute_capability"] = list(torch.cuda.get_device_capability(0))
    evidence["note"] = (
        "Inspection/profiling is a separate untimed run. TensorRT precision flags allow mixed-precision tactics."
    )
    save_json(output, evidence)
    return evidence
