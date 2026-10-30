"""Fail before benchmarking if the actual CUDA/TensorRT stack cannot execute."""

from traffic_lab.runtime import environment, save_json


def main():
    import torch

    assert torch.cuda.is_available(), "CUDA is unavailable in the selected Python environment"
    value = torch.randn(512, 512, device="cuda")
    result = value @ value
    torch.cuda.synchronize()
    assert result.isfinite().all(), "CUDA calculation produced invalid values"
    import onnxruntime as ort
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    evidence = {
        "environment": environment(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_matrix_multiplication_passed": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "tensorrt": trt.__version__,
        "tensorrt_builder_created": True,
        "fast_fp16": bool(builder.platform_has_fast_fp16),
        "fast_int8": bool(builder.platform_has_fast_int8),
        "onnxruntime": ort.__version__,
        "onnx_providers": ort.get_available_providers(),
    }
    save_json("results/full-matrix/hardware.json", evidence)
    print(evidence, flush=True)


if __name__ == "__main__":
    main()
