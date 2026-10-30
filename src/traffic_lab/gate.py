"""Fail closed when benchmark evidence does not meet an explicit quality/latency budget."""

import math


def deployment_gate(benchmark, evaluation, baseline, p95_budget_ms=100, max_map_drop_points=1):
    reasons = []
    if (
        not math.isfinite(p95_budget_ms)
        or not math.isfinite(max_map_drop_points)
        or p95_budget_ms <= 0
        or max_map_drop_points < 0
    ):
        raise ValueError("Latency budget must be positive; accuracy drop allowance nonnegative")
    if benchmark.get("kind") != "predecoded_model_benchmark":
        reasons.append("Expected a predecoded model benchmark")
    if benchmark.get("batch") != 1:
        reasons.append("This gate expects batch one; batched timing is not single-frame latency")
    if any(e.get("kind") != "labeled_detection_evaluation" for e in (evaluation, baseline)):
        reasons.append("Both labeled evaluation artifacts are required")
    if not benchmark.get("model_sha256") or benchmark.get("model_sha256") != evaluation.get("model_sha256"):
        reasons.append("Benchmark and candidate evaluation model hashes differ")
    for key in ("imgsz", "half", "device"):
        if benchmark.get("config", {}).get(key) != evaluation.get("config", {}).get(key):
            reasons.append(f"Benchmark and candidate evaluation {key} differ")
    if (
        not evaluation.get("data")
        or not evaluation.get("split")
        or (evaluation.get("data"), evaluation.get("split")) != (baseline.get("data"), baseline.get("split"))
    ):
        reasons.append("Candidate and baseline evaluation datasets/splits differ")
    latency = benchmark.get("batch_wall_ms", {}).get("p95")
    quality, reference = evaluation.get("map50_95"), baseline.get("map50_95")
    if latency is None or not math.isfinite(latency) or latency <= 0 or latency > p95_budget_ms:
        reasons.append("p95 model wall time exceeds budget or is missing")
    valid_quality = all(q is not None and math.isfinite(q) and 0 <= q <= 1 for q in (quality, reference))
    drop = 100 * (reference - quality) if valid_quality else None
    if drop is None or drop > max_map_drop_points + 1e-9:
        reasons.append("Accuracy drop exceeds allowance or labeled quality is missing")
    return {
        "kind": "model_deployment_gate",
        "passed": not reasons,
        "reasons": reasons,
        "p95_budget_ms": p95_budget_ms,
        "measured_p95_ms": latency,
        "max_map_drop_points": max_map_drop_points,
        "measured_map_drop_points": drop,
        "candidate_model_sha256": evaluation.get("model_sha256"),
        "baseline_model_sha256": baseline.get("model_sha256"),
        "candidate_imgsz": evaluation.get("config", {}).get("imgsz"),
        "baseline_imgsz": baseline.get("config", {}).get("imgsz"),
        "candidate_map50_95": quality,
        "baseline_map50_95": reference,
        "data": evaluation.get("data"),
        "split": evaluation.get("split"),
        "scope": "Model-only gate. Independently require paced-stream and traffic holdout acceptance.",
    }
