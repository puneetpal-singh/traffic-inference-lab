"""Complete provenance metadata for the unchanged inputs used by this study; do not alter measurements."""

import json
from pathlib import Path

from traffic_lab.inspection import weighted_input_types
from traffic_lab.provenance import file_hash

root = Path("results/full-matrix")
manifest = root / "dataset-manifest.json"
digest = file_hash(manifest)
for path in root.glob("quality-*.json"):
    data = json.loads(path.read_text())
    if Path(data["data"]).resolve() == Path("data/coco-deployment/holdout.yaml").resolve():
        if data["dataset_yaml_sha256"] != file_hash(data["data"]):
            raise RuntimeError("Dataset YAML changed; do not append provenance to an unrelated evaluation")
        data.setdefault("dataset_manifest_sha256", digest)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
for path in Path("models/full-matrix").glob("*/export.json"):
    data = json.loads(path.read_text())
    calibration = data["arguments"].get("data")
    if calibration and Path(calibration).resolve() == Path("data/coco-deployment/calibration.yaml").resolve():
        data.setdefault("calibration_yaml_sha256", file_hash(calibration))
        data.setdefault("calibration_manifest_sha256", digest)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    target = root / "exports" / (path.parent.name + ".json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("Captured export manifests and dataset identity; timing and quality values are unchanged.")
for path in root.glob("inspection-onnx-*.json"):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and data.get("profile"):
        events = json.loads(Path(data["profile"]).read_text())
        data["weighted_kernel_input_types"] = weighted_input_types(events)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
