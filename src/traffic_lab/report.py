"""Generate a portable HTML comparison from real run artifacts."""

import base64
import html
import io
import json
from pathlib import Path

from .metrics import pareto_indices


def build_report(directory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory, output = Path(directory), Path(output)
    benchmarks, evaluations, pipelines, parity = [], {}, [], []
    for path in sorted(directory.rglob("*.json")):
        if "validation" in path.parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        kind = data.get("kind")
        if kind == "labeled_detection_evaluation":
            key = (data["model_sha256"], data["config"]["imgsz"], data["config"].get("half", False))
            evaluations[key] = data
        elif kind == "predecoded_model_benchmark":
            benchmarks.append((path, data))
        elif kind == "paced_recorded_multistream":
            pipelines.append((path, data))
        elif kind == "detection_parity":
            parity.append((path, data))
    rows = []
    for path, data in benchmarks:
        c = data["config"]
        quality = evaluations.get((data["model_sha256"], c["imgsz"], c.get("half", False)), {})
        rows.append(
            {
                "name": path.stem,
                "latency": data["batch_wall_ms"]["p95"],
                "quality": quality.get("map50_95"),
                "fps": data["fps"]["mean"],
                "rss": data["peak_sampled_rss_mib"],
                "batch": data["batch"],
                "device": c["device"],
                "imgsz": c["imgsz"],
                "data": quality.get("data"),
            }
        )
    frontier = pareto_indices(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.9, wspace=0.33)
    names = [r["name"] for r in rows]
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(rows))]
    axes[0].barh(names, [r["latency"] for r in rows], color=colors)
    axes[0].set_xlabel("p95 batch wall time (ms); predecoded inputs")
    for i, row in enumerate(rows):
        if row["quality"] is not None:
            axes[1].scatter(
                row["latency"],
                row["quality"],
                s=90,
                color=colors[i],
                label=row["name"],
                edgecolors="#102020" if i in frontier else colors[i],
                linewidths=2,
            )
    if frontier:
        axes[1].legend(loc="upper left", fontsize=8, frameon=False)
        axes[1].margins(x=0.12, y=0.15)
    axes[1].set_xlabel("p95 batch wall time (ms)")
    axes[1].set_ylabel("mAP50–95 on recorded evaluation set")
    axes[1].set_title("Accuracy–latency frontier" if frontier else "No labeled evaluations available")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "comparison.png").write_bytes(buf.getvalue())
    plt.close(fig)
    picture = base64.b64encode(buf.getvalue()).decode()

    def esc(value):
        return html.escape(str(value))

    table = "".join(
        "<tr>"
        + "".join(
            f"<td>{esc(value)}</td>"
            for value in (
                r["name"],
                r["device"],
                r["imgsz"],
                r["batch"],
                f"{r['fps']:.2f}",
                f"{r['latency']:.2f}",
                f"{r['rss']:.1f}",
                f"{r['quality']:.4f}" if r["quality"] is not None else "not evaluated",
            )
        )
        + "</tr>"
        for r in rows
    )
    stream_rows = "".join(
        f"<tr><td>{esc(path.parent.name)}</td><td>{c['camera']}</td>"
        f"<td>{c['decoded']}</td><td>{c['processed']}</td><td>{c['queue_dropped']}</td>"
        f"<td>{c['stride_skipped']}</td><td>{esc(c['frame_age_ms']['p95'])}</td>"
        f"<td>{esc(c['counts'])}</td></tr>"
        for path, d in pipelines
        for c in d["cameras"]
    )
    checks = "".join(
        f"<li>{esc(p.name)}: {'PASS' if d['passed'] else 'FAIL'} ({len(d['frames'])} sampled frames)</li>"
        for p, d in parity
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Traffic Inference Lab</title>
<style>body{{font:16px system-ui;background:#101826;color:#e6edf5;max-width:1200px;margin:40px auto;padding:24px}}
h1{{font-size:42px}}h2{{margin-top:40px}}p{{line-height:1.6;color:#bccbdb}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:12px;border-bottom:1px solid #33445b;text-align:left}}img{{width:100%;background:white;border-radius:12px}}
.tag{{color:#54ddbe;letter-spacing:3px}}code{{color:#54ddbe}}.scroll{{overflow-x:auto}}</style>
<p class="tag">MEASURE · VERIFY · OPTIMIZE</p><h1>Traffic Inference Lab</h1>
<p>Measured runtime comparisons and bounded multi-camera traffic analytics. Model throughput and paced pipeline
latency are different experiments. Results apply to the hardware, inputs, and versions recorded in each artifact.</p>
<img alt="Runtime latency and labeled accuracy comparison" src="data:image/png;base64,{picture}">
<h2>Predecoded model benchmark</h2><div class="scroll"><table><tr><th>Run</th><th>Device</th><th>Size</th><th>Batch</th>
<th>Mean FPS</th><th>p95 batch ms</th><th>Sampled RSS MiB</th><th>mAP50–95</th></tr>{table}</table></div>
<p>FPS is averaged over repeated trials. Batch time is not per-frame live latency. RSS includes decoded inputs.
Accuracy is joined by artifact hash and input size; inspect evaluation JSON for dataset and split. Frontier comparisons
are meaningful only for the same dataset, device budget, precision, batch, and input workload.</p>
<h2>Export parity</h2><ul>{checks}</ul><p>Agreement on sampled detections does not establish domain accuracy.</p>
<h2>Paced camera replay</h2><div class="scroll"><table><tr><th>Run</th><th>Camera</th><th>Decoded</th><th>Processed</th>
<th>Queue drops</th><th>Stride skips</th><th>p95 frame age ms</th><th>Counts (unvalidated)</th></tr>{stream_rows}</table></div>
<p>Frame age includes decode backlog, queueing, prediction, tracking, and optional rendering. Each camera has an
independent tracker. Streams may repeat the same source to test load; this does not simulate scene diversity or network behavior.</p>
<h2>Evidence and limits</h2><p>Raw JSON and per-frame JSONL are the source of truth. COCO smoke evaluations do not
replace a traffic holdout. No power, cloud cost, TensorRT, INT8, or counting-accuracy claims are inferred from other measurements.</p>
</html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return {"report": str(output), "benchmark_runs": len(rows), "pipeline_runs": len(pipelines)}
