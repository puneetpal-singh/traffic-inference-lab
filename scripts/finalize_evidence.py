"""Refresh the initial baseline with the final adapter and assemble/audit the evidence bundle."""

import json
import subprocess
import sys
import time
from pathlib import Path

checks = []


def run(name, args, expected=0):
    start = time.perf_counter()
    with Path(f"results/logs/{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT, check=False)
    check = {
        "name": name,
        "exit_code": result.returncode,
        "expected_exit_code": expected,
        "passed": result.returncode == expected,
        "seconds": time.perf_counter() - start,
    }
    checks.append(check)
    Path("results/verification.json").write_text(json.dumps(checks, indent=2))
    print(f"{name}: {'PASS' if check['passed'] else 'FAIL'}", flush=True)
    if not check["passed"]:
        raise SystemExit(1)
    return check


timing = run(
    "pytorch-320",
    [
        "-m",
        "traffic_lab.cli",
        "benchmark",
        "--source",
        "data/traffic.mp4",
        "--frames",
        "60",
        "--repeats",
        "3",
        "--output",
        "results/pytorch-320.json",
    ],
)
ledger = Path("results/experiment-status.json")
statuses = json.loads(ledger.read_text())
for status in statuses:
    if status["name"] == "pytorch-320":
        status.update(exit_code=timing["exit_code"], seconds=timing["seconds"])
ledger.write_text(json.dumps(statuses, indent=2))
base = [
    "-m",
    "traffic_lab.cli",
    "gate",
    "--benchmark",
    "results/openvino-320.json",
    "--evaluation",
    "results/eval-openvino-320.json",
]
run(
    "gate-same-resolution",
    [
        *base,
        "--baseline-evaluation",
        "results/eval-pytorch-320.json",
        "--output",
        "results/gate-same-resolution.json",
    ],
)
run(
    "gate-resolution-tradeoff",
    [
        *base,
        "--baseline-evaluation",
        "results/eval-pytorch-640.json",
        "--output",
        "results/gate-resolution-tradeoff.json",
    ],
    expected=2,
)
run("capture-provenance", ["scripts/capture_provenance.py"])
run("freeze-environment", ["scripts/freeze_environment.py"])
run("report", ["-m", "traffic_lab.cli", "report"])
run("summarize-results", ["scripts/summarize_results.py"])
run("audit-artifacts", ["scripts/validate_artifacts.py"])
run("dashboard-smoke", ["scripts/check_dashboard.py"])
