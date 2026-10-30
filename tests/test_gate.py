from traffic_lab.gate import deployment_gate


def artifacts():
    config = {"imgsz": 320, "half": False, "device": "cpu"}
    bench = {
        "kind": "predecoded_model_benchmark",
        "batch": 1,
        "model_sha256": "a",
        "config": config,
        "batch_wall_ms": {"p95": 40},
    }
    evaluation = {
        "kind": "labeled_detection_evaluation",
        "model_sha256": "a",
        "config": config,
        "data": "holdout.yaml",
        "split": "val",
        "map50_95": 0.49,
    }
    baseline = {**evaluation, "map50_95": 0.5}
    return bench, evaluation, baseline


def test_gate_accepts_one_absolute_map_point_drop():
    assert deployment_gate(*artifacts())["passed"]


def test_gate_rejects_missing_quality_and_unrelated_models():
    bench, evaluation, baseline = artifacts()
    evaluation["map50_95"] = None
    evaluation["model_sha256"] = "different"
    result = deployment_gate(bench, evaluation, baseline)
    assert not result["passed"]
    assert len(result["reasons"]) == 2


def test_gate_rejects_latency_violation_and_batched_timing():
    bench, evaluation, baseline = artifacts()
    bench["batch"] = 4
    bench["batch_wall_ms"]["p95"] = 120
    result = deployment_gate(bench, evaluation, baseline)
    assert not result["passed"]
    assert len(result["reasons"]) == 2


def test_gate_rejects_nonfinite_measurements():
    bench, evaluation, baseline = artifacts()
    bench["batch_wall_ms"]["p95"] = float("nan")
    evaluation["map50_95"] = float("nan")
    assert not deployment_gate(bench, evaluation, baseline)["passed"]
