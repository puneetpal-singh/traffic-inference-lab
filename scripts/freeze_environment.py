"""Snapshot the tested CPU environment without local editable paths or duplicate metadata."""

from importlib import metadata
from pathlib import Path

names = {d.metadata["Name"] for d in metadata.distributions()}
excluded = {"traffic-inference-lab", "pip", "setuptools"}
lines = [
    "# Tested Windows / Python 3.12 CPU environment. Install project with --no-deps afterward.",
    "--extra-index-url https://download.pytorch.org/whl/cpu",
]
lines += [
    f"{name}=={metadata.version(name)}"
    for name in sorted(names, key=str.lower)
    if name.lower() not in excluded
]
Path("requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
