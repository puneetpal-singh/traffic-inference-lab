"""Execute the full CPU/CUDA/precision matrix, resuming completed stages from an auditable ledger."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def variants(size):
    specs = []
    for backend, precisions, devices in (
        ("pytorch", (32,), ("cpu",)),
        ("pytorch", (32, 16), ("0",)),
        ("onnx", (32, 16, 8), ("cpu",)),
        ("onnx", (32, 16), ("0",)),
        ("openvino", (32, 16, 8), ("cpu",)),
        ("tensorrt", (32, 16, 8), ("0",)),
    ):
        for precision in precisions:
            for device in devices:
                label = "int8" if precision == 8 else f"fp{precision}"
                hardware = "cpu" if device == "cpu" else "cuda"
                name = f"{backend}-{hardware}-{label}-{size}"
                export_key = f"{backend}-{label}-{size}"
                directory = Path("models/full-matrix") / export_key
                suffix = {
                    "onnx": "_int8.onnx" if precision == 8 else ".onnx",
                    "openvino": "_int8_openvino_model" if precision == 8 else "_openvino_model",
                    "tensorrt": ".engine",
                }.get(backend, ".pt")
                model = "models/yolo11n.pt" if backend == "pytorch" else str(directory / ("yolo11n" + suffix))
                specs.append(
                    {
                        "name": name,
                        "backend": backend,
                        "precision": precision,
                        "device": device,
                        "imgsz": size,
                        "model": model,
                        "half": backend == "pytorch" and precision == 16,
                        "export_key": export_key,
                        "export_dir": str(directory),
                    }
                )
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--only", help="Substring filter for an individual backend/case")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path("results/full-matrix")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    ledger_file = root / "ledger.json"
    ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else {}
    all_specs = [s for size in (320, 640) for s in variants(size)]
    (root / "plan.json").write_text(json.dumps(all_specs, indent=2))

    def run(key, arguments, artifact):
        prior = ledger.get(key)
        if not args.force and prior and prior["exit_code"] == 0 and Path(artifact).exists():
            print(f"{key}: cached successful artifact", flush=True)
            return True
        if prior:
            previous = {k: v for k, v in prior.items() if k != "previous_attempts"}
            history = prior.get("previous_attempts", []) + [previous]
            prior_log = root / "logs" / f"{key}.log"
            if prior_log.exists():
                archived = prior_log.with_name(f"{key}.attempt-{len(history)}.log")
                prior_log.replace(archived)
                previous["log"] = str(archived)
        else:
            history = []
        start = time.perf_counter()
        with (root / "logs" / f"{key}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [args.python, *arguments], stdout=log, stderr=subprocess.STDOUT, check=False, timeout=3600
            )
        ledger[key] = {
            "exit_code": result.returncode,
            "seconds": time.perf_counter() - start,
            "arguments": arguments,
            "artifact": str(artifact),
            "previous_attempts": history,
        }
        ledger_file.write_text(json.dumps(ledger, indent=2))
        print(f"{key}: {'OK' if result.returncode == 0 else 'FAILED — inspect log'}", flush=True)
        return result.returncode == 0

    selected = [s for s in all_specs if not args.only or args.only in s["name"]]
    if not run("hardware-check", ["scripts/check_accelerators.py"], root / "hardware.json"):
        return 1
    for spec in selected:
        if spec["backend"] != "pytorch":
            fmt = "engine" if spec["backend"] == "tensorrt" else spec["backend"]
            export_args = [
                "-m",
                "traffic_lab.cli",
                "export",
                "--format",
                fmt,
                "--imgsz",
                str(spec["imgsz"]),
                "--quantize",
                str(spec["precision"]),
                "--output-dir",
                spec["export_dir"],
                "--device",
                "0" if fmt == "engine" else "cpu",
                "--batch",
                "1",
            ]
            if spec["precision"] == 8:
                export_args += ["--data", "data/coco-deployment/calibration.yaml"]
            if not run("export-" + spec["export_key"], export_args, spec["model"]):
                continue
        run(
            spec["name"],
            [
                "scripts/matrix_case.py",
                "--case",
                json.dumps(spec),
                "--frames",
                str(args.frames),
                "--repeats",
                str(args.repeats),
                "--warmup",
                str(args.warmup),
            ],
            root / (spec["name"] + ".json"),
        )
    failed = [key for key, row in ledger.items() if row["exit_code"]]
    print(json.dumps({"failed_stages": failed, "total_recorded_stages": len(ledger)}, indent=2))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
