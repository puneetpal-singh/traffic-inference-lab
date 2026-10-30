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
    elif path.is_dir():
        import xml.etree.ElementTree as ET

        import openvino as ov

        xml = next(path.glob("*.xml"))
        layers = ET.parse(xml).getroot().findall(".//layer")
        evidence["ir_operations"] = dict(Counter(layer.attrib.get("type") for layer in layers))
        evidence["constant_types"] = dict(
            Counter(
                layer.find("data").attrib.get("element_type", "unknown")
                for layer in layers
                if layer.attrib.get("type") == "Const" and layer.find("data") is not None
            )
        )
        # Profile a separate infer request. Traversing the executable runtime graph
        # exposed a native cleanup crash in this Windows OpenVINO build.
        core = ov.Core()
        ir_model = core.read_model(str(xml))
        profiled = core.compile_model(
            ir_model,
            "CPU",
            {"PERF_COUNT": "YES", "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"},
        )
        evidence["execution_devices"] = list(profiled.get_property("EXECUTION_DEVICES"))
        evidence["openvino_inference_precision_hint"] = str(profiled.get_property("INFERENCE_PRECISION_HINT"))
        request = profiled.create_infer_request()
        request.infer({0: np.zeros((1, 3, config.imgsz, config.imgsz), dtype=np.float32)})
        records = [
            {
                "name": str(p.node_name),
                "type": str(p.node_type),
                "execution_type": str(p.exec_type),
                "microseconds": p.real_time.total_seconds() * 1e6,
            }
            for p in request.get_profiling_info()
        ]
        evidence["profiled_nodes"] = records
        evidence["execution_type_counts"] = dict(Counter(r["execution_type"] for r in records))
        evidence["runtime_precision_counts"] = dict(
            Counter(r["execution_type"].split("_")[-1] for r in records)
        )
        del request, profiled, ir_model, core
    elif path.suffix == ".engine":
        detector = Detector(config)
        detector.warmup(2)
        backend = detector.model.predictor.model
        evidence.update(detector.metadata())
        import tensorrt as trt

        engine = backend.model
        inspector = engine.create_engine_inspector()
        inspector.execution_context = backend.context
        details = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        evidence["engine_layers"] = details
        evidence["engine_layer_count"] = engine.num_layers
        memory_size = getattr(engine, "device_memory_size_v2", None)
        evidence["engine_device_memory_bytes"] = int(
            memory_size if memory_size is not None else engine.device_memory_size
        )
        evidence["profiling_verbosity"] = str(engine.profiling_verbosity)
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
