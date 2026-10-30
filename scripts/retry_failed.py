"""Rerun failed ledger entries after fixing the cause, preserving the original attempt logs."""

import json
import subprocess
import sys
import time
from pathlib import Path

ledger = Path("results/experiment-status.json")
statuses = json.loads(ledger.read_text())
for status in statuses:
    if status["exit_code"] == 0:
        continue
    name = status["name"]
    log_path = Path(f"results/logs/{name}.log")
    original = log_path.with_name(log_path.stem + "-initial-failure.log")
    if not original.exists():
        log_path.replace(original)
    previous = {k: v for k, v in status.items() if k != "previous_attempts"}
    status.setdefault("previous_attempts", []).append(previous)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-m", "traffic_lab.cli", *map(str, status["arguments"])],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status.update(exit_code=result.returncode, seconds=time.perf_counter() - start)
    ledger.write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    print(f"{name}: exit {result.returncode}", flush=True)
raise SystemExit(int(any(s["exit_code"] for s in statuses)))
