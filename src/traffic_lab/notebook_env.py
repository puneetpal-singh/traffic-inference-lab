"""Create the Kaggle worker interpreter independently of the notebook kernel's Python."""

import json
import os
import subprocess
from pathlib import Path

from .progress import log_event


def ensure_model_python(kernel_python, environment, setup_step, bootstrap_root="/tmp/traffic-python-bootstrap"):
    """Use a managed CPython 3.12; install only bootstrap uv using the notebook interpreter."""
    environment, bootstrap = Path(environment), Path(bootstrap_root)
    python = environment / "bin/python"
    if not python.exists():
        env = dict(os.environ)
        env["UV_PYTHON_INSTALL_DIR"] = str(bootstrap / "managed")
        env["UV_PYTHON_BIN_DIR"] = str(bootstrap / "bin")
        env["UV_CACHE_DIR"] = str(bootstrap / "cache")
        setup_step([kernel_python, "-m", "pip", "install", "--only-binary=:all:", "--upgrade",
                    "--target", str(bootstrap), "uv==0.13.6"], "setup-python-bootstrap.log")
        # pip --target installs the native binary here. Invoke it directly to avoid a global uv shadowing it.
        uv = [str(bootstrap / "bin/uv")]
        setup_step([*uv, "python", "install", "3.12"], "setup-python-download.log", env=env)
        managed = subprocess.check_output([*uv, "python", "find", "--managed-python", "3.12"],
                                          env=env, text=True).strip()
        setup_step([managed, "-m", "venv", str(environment)], "setup-venv.log")
    version = json.loads(subprocess.check_output(
        [str(python), "-c", "import json,sys; print(json.dumps(list(sys.version_info[:3])))"], text=True))
    if version[:2] != [3, 12]:
        raise RuntimeError(f"Worker environment has Python {version}; choose a fresh environment path for Python 3.12")
    setup_step([str(python), "-m", "ensurepip", "--upgrade"], "setup-ensurepip.log")
    log_event(f"Worker Python {'.'.join(str(v) for v in version)} ready: {python}")
    return python
