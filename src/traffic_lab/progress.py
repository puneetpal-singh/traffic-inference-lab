"""Parent-process progress reporting; never instrument the timed inference loop."""

import os
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def worker_environment(base=None):
    """Notebook-only rendering backends are not available in isolated worker environments."""
    environment = dict(os.environ if base is None else base)
    environment["MPLBACKEND"] = "Agg"
    return environment


def log_event(message):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def ledger_progress(ledger, planned=None):
    complete = sum(row.get("status") == "complete" for row in ledger.values())
    failed = sum(row.get("status") in ("failed", "interrupted") for row in ledger.values())
    total = f"/{planned}" if planned is not None else ""
    return f"completed={complete}{total}, failed/interrupted={failed}"


@contextmanager
def stage_progress(label, heartbeat=15):
    """Liveness for untimed downloads/setup; deliberately not used inside benchmark trials."""
    started, stopped = time.monotonic(), threading.Event()

    def report():
        while not stopped.wait(heartbeat):
            log_event(f"HEARTBEAT {label}: elapsed={time.monotonic()-started:.0f}s; stage still running")

    thread = threading.Thread(target=report, daemon=True)
    log_event(f"START {label}")
    thread.start()
    try:
        yield
    except BaseException:
        log_event(f"STOPPED {label}: elapsed={time.monotonic()-started:.1f}s")
        raise
    else:
        log_event(f"OK {label}: elapsed={time.monotonic()-started:.1f}s")
    finally:
        stopped.set()
        thread.join(timeout=1)


def wait_with_progress(process, log_path, label, timeout, heartbeat=15):
    """Show bounded log excerpts and liveness, including when native builders emit no output."""
    started = time.monotonic()
    position = 0
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(getattr(process, "args", label), timeout)
        done = False
        try:
            code = process.wait(timeout=min(heartbeat, remaining))
            done = True
        except subprocess.TimeoutExpired:
            code = None
        try:
            # Bound memory/output even if a backend writes a very large native compilation log.
            with Path(log_path).open("rb") as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(position, size - 8192))
                chunk = stream.read().decode("utf-8", errors="replace")
                position = size
            lines = [line.strip() for line in chunk.replace("\r", "\n").splitlines() if line.strip()]
            for line in lines[-3:]:
                log_event(f"{label} | {line[:500]}")
        except OSError:
            lines = []
        if done:
            return code
        elapsed = time.monotonic() - started
        activity = "recent worker output above" if lines else "worker still running; no new log output"
        log_event(f"HEARTBEAT {label}: elapsed={elapsed:.0f}s, timeout={timeout}s; {activity}")
