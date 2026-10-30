"""Sequential subprocesses isolate runtime state and preserve failures; no concurrent timing runs."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--duration", type=float, default=5)
    p.add_argument("--evaluate", action="store_true", help="Run COCO128 smoke evaluations")
    args = p.parse_args()
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    statuses = []

    def run(name, *arguments):
        start = time.perf_counter()
        with open(f"results/logs/{name}.log", "w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-m", "traffic_lab.cli", *map(str, arguments)],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        statuses.append(
            dict(
                name=name,
                exit_code=result.returncode,
                seconds=time.perf_counter() - start,
                arguments=list(arguments),
            )
        )
        Path("results/experiment-status.json").write_text(json.dumps(statuses, indent=2))
        print(f"{name}: {'OK' if result.returncode == 0 else 'FAILED (see log)'}", flush=True)
        return result.returncode == 0

    if not run("fetch", "fetch"):
        return 1
    models = [("pytorch", "models/yolo11n.pt")]
    for fmt, extension in (("onnx", "onnx"), ("openvino", "_openvino_model")):
        suffix = f".{extension}" if fmt == "onnx" else extension
        if run(
            f"export-{fmt}",
            "export",
            "--format",
            fmt,
            "--dynamic",
            "--batch",
            4,
            "--output-dir",
            f"models/{fmt}",
        ):
            models.append((fmt, f"models/{fmt}/yolo11n{suffix}"))
    for size in (320, 640):
        for name, model in models:
            key = f"{name}-{size}"
            run(
                key,
                "benchmark",
                "--model",
                model,
                "--source",
                "data/traffic.mp4",
                "--imgsz",
                size,
                "--frames",
                args.frames,
                "--repeats",
                args.repeats,
                "--output",
                f"results/{key}.json",
            )
            if name != "pytorch":
                run(
                    f"parity-{key}",
                    "parity",
                    "--candidate",
                    model,
                    "--source",
                    "data/traffic.mp4",
                    "--imgsz",
                    size,
                    "--output",
                    f"results/parity-{key}.json",
                )
            if args.evaluate:
                run(
                    f"eval-{key}",
                    "evaluate",
                    "--model",
                    model,
                    "--data",
                    "coco128.yaml",
                    "--imgsz",
                    size,
                    "--output",
                    f"results/eval-{key}.json",
                )
    # CPU model stays batch 1 so the change in stream count is isolated.
    preferred = models[0][1]
    for runtime_name in ("onnx", "openvino"):
        check = Path(f"results/parity-{runtime_name}-320.json")
        if check.is_file() and json.loads(check.read_text()).get("passed"):
            preferred = next((m for n, m in models if n == runtime_name), preferred)
    for streams, batch, stride, tracker, render in (
        (1, 1, 1, "bytetrack", False),
        (2, 1, 1, "bytetrack", False),
        (4, 1, 1, "bytetrack", False),
        (4, 4, 1, "bytetrack", False),
        (4, 1, 2, "bytetrack", False),
        (1, 1, 1, "botsort", True),
    ):
        key = f"streams-{streams}-batch-{batch}-stride-{stride}-{tracker}"
        run(
            key,
            "run",
            "--model",
            preferred,
            "--sources",
            *(["data/traffic.mp4"] * streams),
            "--duration",
            args.duration,
            "--batch",
            batch,
            "--stride",
            stride,
            "--tracker",
            tracker,
            "--output",
            f"results/{key}",
            *(["--render"] if render else []),
        )
    run("report", "report")
    return int(any(s["exit_code"] for s in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
