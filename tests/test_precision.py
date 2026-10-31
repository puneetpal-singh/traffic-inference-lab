import importlib.util
import sys
from pathlib import Path

import pytest

from traffic_lab.precision import (
    cases,
    evidence_identity,
    paired_ratio,
    randomized_rounds,
    split_images,
    validate_timing,
)


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/precision_study.py"
    spec = importlib.util.spec_from_file_location("precision_study_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_model_matrix_separates_devices_and_does_not_invent_cuda_int8_onnx():
    all_cases = cases()
    assert len(all_cases) == 13
    assert len(cases("cpu")) == 7
    assert len(cases("cuda")) == 6
    assert not any(c["backend"] == "onnx" and c["precision"] == 8 and c["device"] != "cpu" for c in all_cases)
    onnx = [c for c in all_cases if c["backend"] == "onnx" and c["precision"] == 32]
    assert len(onnx) == 2 and onnx[0]["export_key"] == onnx[1]["export_key"]


def test_split_is_disjoint_reproducible_and_independent_of_annotation_order():
    images = [dict(id=i) for i in range(5000)]
    calibration, evaluation = split_images(images, 512, 0, 42)
    assert len(calibration) == 512 and len(evaluation) == 4488
    assert not {i["id"] for i in calibration} & {i["id"] for i in evaluation}
    assert (calibration, evaluation) == split_images(images[::-1], 512, 0, 42)
    assert len(split_images(images, 512, 256, 42)[1]) == 256
    with pytest.raises(ValueError):
        split_images(images, 5000, 0, 42)


def test_every_randomized_round_contains_each_configuration_exactly_once():
    orders = randomized_rounds(cases(), 10, 42)
    assert orders == randomized_rounds(cases(), 10, 42)
    assert len({tuple(order) for order in orders}) > 1
    expected = sorted(c["name"] for c in cases())
    assert all(sorted(order) == expected for order in orders)


def test_precision_ratio_pairs_rounds_and_does_not_weight_fast_rounds_more():
    result = paired_ratio({0: 20, 1: 500, 2: 999}, {0: 10, 1: 1000})
    assert result["rounds"] == [0, 1]
    assert result["geometric_mean"] == pytest.approx(1)
    assert paired_ratio({0: 1}, {1: 1}) is None
    assert evidence_identity({"round": 1}, {"model": "a"}) != evidence_identity({"round": 1}, {"model": "b"})


def test_default_and_explicit_dry_run_do_not_import_inference_or_write_results(tmp_path, monkeypatch, capsys):
    module = runner()
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    before = set(sys.modules)
    assert module.main([]) == 0
    assert module.main(["--dry-run"]) == 0
    assert list(tmp_path.iterdir()) == []
    assert not {"torch", "ultralytics", "tensorrt", "openvino"} & (set(sys.modules) - before)
    assert "DRY RUN" in capsys.readouterr().out


def test_resume_reuses_only_identical_successful_artifacts(tmp_path, monkeypatch):
    module = runner()
    calls = []
    job = {"key": "quality-test", "stage": "quality", "identity": "original",
           "artifact": str(tmp_path / "timing" / "test.json")}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            calls.append(args)
            module.save(job["artifact"], {"fps": 10})

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", FakeWorker)
    ledger = {}
    assert module.execute(job, ledger, tmp_path, 60)
    assert module.execute(job, ledger, tmp_path, 60)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Cached evidence changed"):
        module.execute({**job, "identity": "different-model"}, ledger, tmp_path, 60)
    module.save(job["artifact"], {"fps": 1000})
    with pytest.raises(ValueError, match="Cached evidence changed"):
        module.execute(job, ledger, tmp_path, 60)


def test_report_keeps_missing_cases_pending_and_compares_only_within_runtime(tmp_path):
    module = runner()
    root = tmp_path / "study"
    subset = [c for c in cases("cpu") if c["backend"] == "pytorch"]
    cfg = {"trials": 10, "quality_budget_points": 1, "model": str(tmp_path / "model.pt")}
    module.save(root / "plan.json", {"config": cfg, "cases": subset})
    module.save(root / "ledger.json", {})
    report = module.report(root)
    assert report["rows"][0]["trials"] == 0
    assert report["rows"][0]["fps_median"] is None
    assert report["within_runtime_paired_fps_ratios"] == {}
    assert "pending" in (root / "REPORT.md").read_text(encoding="utf-8")


def test_timing_audit_rejects_fps_and_frame_pool_tampering():
    cfg = {"seconds": 20, "pool_frames": 2, "frame_step": 2}
    row = {"seconds": 20, "frames": 2, "cycles": 1, "fps": .1, "p50_ms": 10, "p95_ms": 10,
           "samples": [{"source_index": i, "wall_ms": 10} for i in (0, 2)]}
    validate_timing(row, cfg)
    with pytest.raises(ValueError, match="FPS"):
        validate_timing({**row, "fps": 1000}, cfg)
    with pytest.raises(ValueError, match="composition"):
        validate_timing({**row, "cycles": 2}, cfg)
