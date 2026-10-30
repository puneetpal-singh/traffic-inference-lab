"""Render the real dashboard with Streamlit's app test harness, without a background server."""

import json
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

start = time.perf_counter()
app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard.py")).run(timeout=60)
errors = [str(error.message) for error in app.exception]
verification = Path("results/verification.json")
if verification.exists():
    checks = json.loads(verification.read_text())
    for check in checks:
        if check["name"] == "dashboard-smoke":
            previous = {k: v for k, v in check.items() if k != "previous_attempts"}
            check.setdefault("previous_attempts", []).append(previous)
            check.update(exit_code=int(bool(errors)), passed=not errors, seconds=time.perf_counter() - start)
    verification.write_text(json.dumps(checks, indent=2))
print(
    json.dumps(
        {
            "passed": not errors,
            "exceptions": errors,
            "dataframes": len(app.dataframe),
            "metrics": len(app.metric),
        },
        indent=2,
    )
)
raise SystemExit(int(bool(errors)))
