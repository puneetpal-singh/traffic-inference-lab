import ast
import base64
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest


def script(name, monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kaggle_smoke_is_default_and_budgets_are_preregistered(tmp_path, monkeypatch):
    module = script("kaggle_study", monkeypatch)
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg = module.configuration(cfg_file)
    assert cfg["mode"] == "smoke"
    assert cfg["serving_repeats"] == 3 and cfg["serving_seconds"] == 30
    assert cfg["model_trials"] == 10 and cfg["event_labels"] == {}
    cfg_file.write_text(json.dumps({"serving_repeats": 1}))
    with pytest.raises(ValueError):
        module.configuration(cfg_file)


def test_full_mode_requires_successful_smoke_on_same_environment(tmp_path, monkeypatch):
    module = script("kaggle_study", monkeypatch)
    monkeypatch.setattr(module, "REPO", tmp_path)
    cfg = {"mode": "full", "run_id": "test"}
    env = {"environment": {"gpu": "T4", "driver": "x"}}
    with pytest.raises(RuntimeError, match="smoke"):
        module.validate_smoke_gate(cfg, env)
    status = module.study_root({**cfg, "mode": "smoke"}) / "smoke-status.json"
    module.save(status, {"passed": False, "preflight": env})
    with pytest.raises(RuntimeError):
        module.validate_smoke_gate(cfg, env)
    module.save(status, {"passed": True, "preflight": env})
    module.validate_smoke_gate(cfg, env)
    with pytest.raises(RuntimeError, match="changed"):
        module.validate_smoke_gate(cfg, {"environment": {"gpu": "L4"}})


def test_annotation_templates_never_pass_as_independent_ground_truth(tmp_path, monkeypatch):
    module = script("kaggle_study", monkeypatch)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"test source identity only")
    path = tmp_path / "labels.json"
    line = [.1, .55, .9, .55]
    module.save(path, {"reviewed": False, "events": []})
    assert module.label_document(path, source, line)[0] is None
    module.save(path, {"reviewed": True, "source_sha256": "wrong", "line": line, "events": []})
    with pytest.raises(ValueError, match="match"):
        module.label_document(path, source, line)
    module.save(path, {"reviewed": True, "source_sha256": module.file_hash(source), "line": line,
                       "events": [{"source_seconds": 4.2, "class_id": 2, "direction": "positive"}]})
    assert module.label_document(path, source, line)[0]["events"][0]["class_id"] == 2


def test_camera_gate_uses_worst_camera_not_pooled_average(tmp_path, monkeypatch):
    module = script("kaggle_study", monkeypatch)
    def camera(fps, age):
        return {"decoded": 100, "processed": 60, "queue_dropped": 40, "stride_skipped": 0,
                "remaining": 0, "queue_high_water": 1, "processed_fps": fps, "frame_age_ms": {"p95": age},
                "decoder_threads": 1}
    summary = {"status": "complete", "queue_capacity_per_camera": 1,
               "cameras": [camera(25, 40), camera(5, 200)]}
    result = module.pipeline_metrics(summary, {"min_camera_fps": 15, "max_frame_age_p95_ms": 100})
    assert result["aggregate_fps"] == 30
    assert result["min_camera_fps"] == 5 and not result["timing_passed"]
    summary["cameras"][0]["decoded"] = 99
    with pytest.raises(ValueError, match="accounting"):
        module.pipeline_metrics(summary, {"min_camera_fps": 15, "max_frame_age_p95_ms": 100})


def test_phase_control_does_not_change_job_identity(monkeypatch):
    module = script("kaggle_study", monkeypatch)
    cfg = {"mode": "full", "run_id": "test", "phase": "models"}
    a = module.basic_job(cfg, {}, "probe", {"name": "pt"}, "output.json")
    b = module.basic_job({**cfg, "phase": "serving"}, {}, "probe", {"name": "pt"}, "output.json")
    assert a == b


def test_source_bundle_contains_no_assets_credentials_or_old_results(monkeypatch):
    builder = script("build_kaggle_notebook", monkeypatch)
    data = builder.source_bundle()
    assert data == builder.source_bundle()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        assert "scripts/kaggle_study.py" in names
        assert "scripts/precision_study.py" in names
        assert not any(name.startswith(("data/", "models/", "results/", ".env", ".git/")) for name in names)
        assert not any("overnight_study" in name or "full_coco_quality" in name for name in names)


def test_generated_notebook_cells_parse_and_default_to_smoke(monkeypatch):
    builder = script("build_kaggle_notebook", monkeypatch)
    notebook = json.loads((builder.DESTINATION / "Traffic_Analytics_Kaggle.ipynb").read_text(encoding="utf-8"))
    source_payload = None
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            tree = ast.parse(cell["source"])
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id == "MODE":
                        assert ast.literal_eval(node.value) == "smoke"
                    if node.targets[0].id == "SOURCE_BUNDLE_BASE64":
                        source_payload = ast.literal_eval(node.value)
    assert source_payload
    payload = base64.b64decode(source_payload)
    assert payload == builder.source_bundle(), "Regenerate the notebook after changing bundled source"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if name.endswith(".py"):
                ast.parse(archive.read(name).decode(), filename=name)
    metadata = json.loads((builder.DESTINATION / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True


def test_checkpoint_omits_image_cache_and_live_process_lock(tmp_path, monkeypatch):
    module = script("kaggle_study", monkeypatch)
    repo = tmp_path / "lab"
    monkeypatch.setattr(module, "REPO", repo)
    for name in ("results/run/full/run.lock", "results/run/full/ledger.json", "models/yolo11n.pt",
                 "data/run-full/images/evaluation/image.jpg", "data/run-full/manifest.json"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    module.archive({"run_id": "run"})
    with zipfile.ZipFile(tmp_path / "run-checkpoint.zip") as archive:
        assert "results/run/full/ledger.json" in archive.namelist()
        assert "models/yolo11n.pt" in archive.namelist()
        assert not any("images/" in n or n.endswith("run.lock") for n in archive.namelist())
