import json

from traffic_lab.report import build_report


def test_report_escapes_names_and_does_not_invent_quality(tmp_path):
    run = {
        "kind": "predecoded_model_benchmark",
        "config": {"device": "<script>", "imgsz": 320},
        "model_sha256": "abc",
        "batch_wall_ms": {"p95": 12},
        "fps": {"mean": 80},
        "peak_sampled_rss_mib": 100,
        "batch": 1,
    }
    (tmp_path / "sample.json").write_text(json.dumps(run))
    target = tmp_path / "report.html"
    assert build_report(tmp_path, target)["benchmark_runs"] == 1
    page = target.read_text(encoding="utf-8")
    assert "&lt;script&gt;" in page
    assert "not evaluated" in page
    assert "<td><script>" not in page
