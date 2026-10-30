"""Read-only local experiment viewer. Run: streamlit run dashboard.py"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Traffic Inference Lab", page_icon="🚦", layout="wide")
st.title("Traffic Inference Lab")
st.caption("Measured deployment experiments · independent camera tracking · reproducible evidence")
default_results = "results/full-matrix" if Path("results/full-matrix/plan.json").exists() else "results"
root = Path(st.sidebar.text_input("Results directory", default_results))
summaries = []
if root.is_dir():
    for path in sorted(root.rglob("summary.json")):
        try:
            if json.loads(path.read_text()).get("kind") == "paced_recorded_multistream":
                summaries.append(path)
        except (OSError, ValueError, AttributeError):
            continue
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
                        "throughput FPS": data.get("aggregate_fps", data["fps"]["mean"]),
                        "p95 batch ms": data["batch_wall_ms"]["p95"],
                        "sampled RSS MiB": data["peak_sampled_rss_mib"],
                    }
                )
        except (OSError, ValueError, KeyError):
            continue
if benchmarks:
    st.subheader("Runtime comparison")
    st.dataframe(pd.DataFrame(benchmarks), width="stretch", hide_index=True)
    st.bar_chart(pd.DataFrame(benchmarks).set_index("run")[["throughput FPS"]])
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
        if "track_positions_normalized" in rows:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            points = [point for frame_points in rows.track_positions_normalized for point in frame_points]
            if points:
                points = np.asarray(points)
                fig, ax = plt.subplots(figsize=(7, 4))
                density = ax.hist2d(points[:, 0], points[:, 1], bins=24, range=((0, 1), (0, 1)), cmap="magma")
                ax.invert_yaxis()
                ax.set(
                    xlabel="Normalized image x", ylabel="Normalized image y", title="Tracked-position density"
                )
                fig.colorbar(density[3], ax=ax, label="Processed-frame observations")
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "Track bottom-center observations; repeated sightings accumulate. This is not a unique-vehicle count."
                )
        with st.expander("Cost sensitivity"):
            hourly_rate = st.number_input(
                "Assumed machine cost per hour (your currency)", min_value=0.0, value=0.0, step=0.1
            )
            observed_fps = sum(camera["processed_fps"] for camera in summary["cameras"])
            if hourly_rate > 0 and observed_fps > 0:
                st.metric(
                    "Assumed cost per million processed frames",
                    f"{hourly_rate * 1e6 / (3600 * observed_fps):.2f}",
                )
                st.metric(
                    "Assumed cost per configured camera-hour", f"{hourly_rate / len(summary['cameras']):.3f}"
                )
            st.caption(
                "Scenario arithmetic from this replay's measured processing rate and your supplied hourly rate. "
                "No provider price is assumed; review each camera's freshness, FPS and drops before comparing costs."
            )
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
