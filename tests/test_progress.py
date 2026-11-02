import subprocess

import pytest

from traffic_lab.progress import ledger_progress, stage_progress, wait_with_progress


def test_quiet_worker_emits_heartbeat_then_shows_new_log_output(tmp_path, capsys):
    path = tmp_path / "worker.log"
    path.write_text("")

    class Worker:
        args = ["fake-worker"]
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(self.args, timeout)
            path.write_text("finished export\n")
            return 0

    assert wait_with_progress(Worker(), path, "TensorRT export", timeout=30, heartbeat=.01) == 0
    output = capsys.readouterr().out
    assert "UTC" in output and "HEARTBEAT TensorRT export" in output
    assert "no new log output" in output and "finished export" in output


def test_log_forwarding_is_bounded_but_full_log_remains(tmp_path, capsys):
    path = tmp_path / "native.log"
    text = "\n".join(f"line-{i}" for i in range(100))
    path.write_text(text)

    class Worker:
        def wait(self, timeout):
            return 0

    assert wait_with_progress(Worker(), path, "native compile", 30) == 0
    output = capsys.readouterr().out
    assert "line-99" in output and "line-0" not in output
    assert path.read_text() == text


def test_progress_retains_failure_code_and_correct_ledger_counts(tmp_path):
    class Worker:
        def wait(self, timeout):
            return 7

    assert wait_with_progress(Worker(), tmp_path / "missing.log", "failure", 30) == 7
    assert ledger_progress({"a": {"status": "complete"}, "b": {"status": "failed"}}, 10) == (
        "completed=1/10, failed/interrupted=1"
    )


def test_untimed_stage_reports_failure_and_propagates_it(capsys):
    with pytest.raises(ValueError):
        with stage_progress("data download"):
            raise ValueError("network error")
    assert "STOPPED data download" in capsys.readouterr().out
