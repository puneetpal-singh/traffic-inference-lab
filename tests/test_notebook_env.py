import json

import pytest

from traffic_lab import notebook_env


def test_newer_kernel_bootstraps_managed_python_instead_of_creating_host_version_venv(tmp_path, monkeypatch):
    environment, calls = tmp_path / "worker", []
    managed = "/managed/cpython312/bin/python3.12"

    def step(args, logfile, env=None):
        calls.append((args, logfile, env))
        if args[1:3] == ["-m", "venv"]:
            assert args[0] == managed
            path = environment / "bin/python"
            path.parent.mkdir(parents=True)
            path.write_text("fake interpreter")

    def output(args, **kwargs):
        if "find" in args:
            assert "--managed-python" in args
            assert args[0] == str(tmp_path / "uv/bin/uv")
            assert kwargs["env"]["UV_PYTHON_INSTALL_DIR"].startswith(str(tmp_path))
            return managed + "\n"
        return json.dumps([3, 12, 10])

    monkeypatch.setattr(notebook_env.subprocess, "check_output", output)
    result = notebook_env.ensure_model_python("/kernel/python3.14", environment, step, tmp_path / "uv")
    assert result == environment / "bin/python"
    assert any("uv==0.13.6" in args for args, _, _ in calls)
    assert not any(args[:3] == ["/kernel/python3.14", "-m", "venv"] for args, _, _ in calls)
    assert any(args[1:3] == ["-m", "ensurepip"] for args, _, _ in calls)


def test_existing_worker_is_checked_without_redownloading_python(tmp_path, monkeypatch):
    python = tmp_path / "worker/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("fake interpreter")
    monkeypatch.setattr(notebook_env.subprocess, "check_output", lambda *a, **k: json.dumps([3, 12, 10]))
    calls = []
    result = notebook_env.ensure_model_python("/kernel/python3.14", python.parent.parent,
                                               lambda *a, **k: calls.append(a))
    assert result == python and len(calls) == 1
    assert "ensurepip" in calls[0][0]


def test_incompatible_existing_worker_fails_before_ml_installation(tmp_path, monkeypatch):
    python = tmp_path / "worker/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("fake interpreter")
    monkeypatch.setattr(notebook_env.subprocess, "check_output", lambda *a, **k: json.dumps([3, 14, 0]))
    with pytest.raises(RuntimeError, match="Worker environment"):
        notebook_env.ensure_model_python("/kernel/python3.14", python.parent.parent,
                                        lambda *a, **k: pytest.fail("must not install ML packages"))
