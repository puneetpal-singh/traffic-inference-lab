"""Capture hashes, not source imagery or model weights, for the locally evaluated smoke dataset."""

import shutil
from pathlib import Path

from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.checks import check_yaml

from traffic_lab.runtime import file_hash, save_json

Path("results").mkdir(exist_ok=True)
for source, destination in (
    ("data/manifest.json", "results/input-manifest.json"),
    ("models/onnx/export.json", "results/export-onnx.json"),
    ("models/openvino/export.json", "results/export-openvino.json"),
):
    if Path(source).exists():
        shutil.copy2(source, destination)
dataset = check_det_dataset("coco128.yaml", autodownload=False)
root = Path(dataset["path"])
files = {
    p.relative_to(root).as_posix(): file_hash(p)
    for p in sorted(root.rglob("*"))
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".txt"}
}
save_json(
    "results/dataset-manifest.json",
    {
        "dataset": "coco128",
        "root": str(root),
        "yaml_sha256": file_hash(check_yaml("coco128.yaml")),
        "files": files,
        "note": "Snapshot of the local COCO128 files used in the smoke evaluation. No dataset files redistributed.",
    },
)
