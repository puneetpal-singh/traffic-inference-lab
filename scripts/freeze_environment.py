"""Snapshot the tested CPU environment without local editable paths or duplicate metadata."""

import argparse
from importlib import metadata
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--output", default="requirements-lock.txt")
args = parser.parse_args()
names = {d.metadata["Name"] for d in metadata.distributions()}
excluded = {"traffic-inference-lab", "pip", "setuptools"}
lines = [
    "# Tested Windows / Python 3.12 environment. Install the Windows TensorRT SDK wheel first for CUDA.",
    "--extra-index-url https://download.pytorch.org/whl/" + ("cu124" if args.cuda else "cpu"),
]
lines += [
    f"{name}=={metadata.version(name)}"
    for name in sorted(names, key=str.lower)
    if name.lower() not in excluded
]
Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
