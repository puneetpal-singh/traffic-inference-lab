"""Read-only local experiment viewer. Run: streamlit run dashboard.py"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Traffic Inference Lab", page_icon="🚦", layout="wide")
st.title("Traffic Inference Lab")
st.caption("Measured deployment experiments · independent camera tracking · reproducible evidence")
root = Path(st.sidebar.text_input("Results directory", "results"))
summaries = sorted(root.rglob("summary.json")) if root.is_dir() else []
benchmarks = []
if root.is_dir():
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("kind") == "predecoded_model_benchmark":
                benchmarks.append(
                    {
                        "run": path.stem,
                        "device": data["config"]["device"],
                        "input size": data["config"]["imgsz"],
                        "batch": data["batch"],
                        "mean FPS": data["fps"]["mean"],
                        "p95 batch ms": data["batch_wall_ms"]["p95"],
                        "sampled RSS MiB": data["peak_sampled_rss_mib"],
                    }
                )
        except (OSError, ValueError, KeyError):
            continue
if benchmarks:
    st.subheader("Runtime comparison")
    st.dataframe(pd.DataFrame(benchmarks), width="stretch", hide_index=True)
    st.bar_chart(pd.DataFrame(benchmarks).set_index("run")[["mean FPS"]])
    st.caption("Predecoded model throughput excludes decoding, queues, tracking, and rendering.")
if not summaries:
    st.info("Run traffic-lab run to generate camera telemetry. Checked-in benchmarks remain viewable above.")
else:
    selected = st.selectbox("Camera experiment", summaries, format_func=lambda p: str(p.parent))
    summary = json.loads(selected.read_text())
    st.subheader("Paced multi-camera replay")
    st.dataframe(pd.DataFrame(summary["cameras"]), width="stretch")
    log = selected.parent / "frames.jsonl"
    if log.exists() and log.stat().st_size:
        data = pd.read_json(log, lines=True)
        cam = st.selectbox("Camera", sorted(data.camera.unique()))
        rows = data[data.camera == cam].set_index("source_seconds")
        a, b, c = st.columns(3)
        a.metric("p95 frame age", f"{rows.frame_age_ms.quantile(0.95):.1f} ms")
        b.metric("Processed frames", len(rows))
        c.metric("Mean tracked objects", f"{rows.tracks.mean():.1f}")
        st.line_chart(rows[["frame_age_ms", "queue_wait_ms", "batch_predict_ms"]])
        st.line_chart(rows[[k for k in ("detections", "tracks", "zone_occupancy") if k in rows]])
        video = selected.parent / f"camera-{cam}.mp4"
        browser_video = selected.parent / f"camera-{cam}-browser.mp4"
        if browser_video.exists():
            video = browser_video
        if video.exists():
            st.video(str(video))
    st.warning(
        "Counts require independent annotations. Replayed streams measure compute load, not network reliability."
    )
    with st.expander("Environment and configuration"):
        st.json(summary)
