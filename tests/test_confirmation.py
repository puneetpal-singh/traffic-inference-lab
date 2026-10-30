import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from traffic_lab.confirmation import balanced_orders, json_digest, paired_speedup


def test_confirmation_order_is_balanced_and_reproducible():
    orders = balanced_orders(6, 42)
    assert orders == balanced_orders(6, 42)
    assert orders.count(["pytorch", "tensorrt"]) == 3
    assert orders.count(["tensorrt", "pytorch"]) == 3
    with pytest.raises(ValueError):
        balanced_orders(5, 42)


def test_constant_paired_effect_has_degenerate_trial_interval():
    trials = [
        dict(pair=p, runtime=r, fps=f) for p in range(6) for r, f in (("pytorch", 20), ("tensorrt", 40))
    ]
    result = paired_speedup(trials, resamples=100)
    assert result["geometric_mean_speedup"] == pytest.approx(2)
    assert result["trial_bootstrap_95_interval"] == pytest.approx([2, 2])
    with pytest.raises(ValueError):
        paired_speedup(trials[:-1], resamples=100)
    with pytest.raises(ValueError):
        paired_speedup(trials + [trials[0]], resamples=100)


def test_evidence_digest_is_independent_of_json_key_order_and_rejects_invalid_rate():
    assert json_digest({"a": 1, "b": 2}) == json_digest({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        paired_speedup([dict(pair=0, runtime="pytorch", fps=float("nan"))])


def test_paired_effect_does_not_weight_pairs_by_their_absolute_throughput():
    trials = []
    for pair in range(6):
        baseline, ratio = (10, 2) if pair % 2 == 0 else (1000, 0.5)
        trials.extend(
            [
                dict(pair=pair, runtime="pytorch", fps=baseline),
                dict(pair=pair, runtime="tensorrt", fps=baseline * ratio),
            ]
        )
    assert paired_speedup(trials, resamples=100)["geometric_mean_speedup"] == pytest.approx(1)


def test_published_evidence_audit_rejects_a_rehashed_fabricated_fps(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    copied = tmp_path / "confirmation"
    shutil.copytree(repo / "results/confirmation", copied)
    summary = json.loads((copied / "summary.json").read_text())
    reference = summary["trials"][0]
    path = copied / reference["path"]
    trial = json.loads(path.read_text())
    trial["fps"] *= 2
    path.write_text(json.dumps(trial))
    reference["json_sha256"] = json_digest(trial)
    (copied / "summary.json").write_text(json.dumps(summary))
    result = subprocess.run(
        [sys.executable, "scripts/audit_confirmation.py", "--input", str(copied)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "FPS mismatch" in result.stdout
