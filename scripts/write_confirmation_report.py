"""Generate a recruiter-facing summary and CV wording from verified measured evidence."""

import json
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path("results/confirmation")
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    if not audit["passed"]:
        raise RuntimeError("Audit the evidence successfully before generating CV claims")
    trials = [json.loads((root / item["path"]).read_text()) for item in summary["trials"]]
    speedup = summary["geometric_mean_speedup"]
    low, high = summary["trial_bootstrap_95_interval"]
    baseline, candidate = (summary["aggregate_fps"][name] for name in ("pytorch", "tensorrt"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for pair in range(summary["pairs"]):
        values = [
            next(t["fps"] for t in trials if t["pair"] == pair and t["runtime"] == name)
            for name in ("pytorch", "tensorrt")
        ]
        axes[0].plot([0, 1], values, color="#607d8b", alpha=0.6, linewidth=1, marker="o", markersize=4)
    axes[0].scatter([0, 1], [baseline, candidate], s=90, color=["#2474b5", "#dc7820"], zorder=5)
    axes[0].set_xticks(
        [0, 1],
        [
            f"PyTorch CUDA FP32\n{baseline:.1f} aggregate FPS",
            f"TensorRT FP16 enabled\n{candidate:.1f} aggregate FPS",
        ],
    )
    axes[0].set(
        xlim=(-0.35, 1.35),
        ylim=(0, max(t["fps"] for t in trials) * 1.18),
        ylabel="Frames per second",
        title="Same GPU · each line is one paired trial",
    )
    ratios = summary["paired_fps_ratios"]
    axes[1].scatter(range(1, len(ratios) + 1), ratios, color="#2474b5", s=50)
    axes[1].errorbar(
        [len(ratios) + 2],
        [speedup],
        yerr=[[speedup - low], [high - speedup]],
        fmt="o",
        color="#dc7820",
        capsize=6,
        markersize=7,
    )
    axes[1].axhline(1, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_xticks(
        list(range(1, len(ratios) + 1)) + [len(ratios) + 2],
        [str(n) for n in range(1, len(ratios) + 1)] + ["Geo. mean"],
    )
    axes[1].set(
        ylim=(0, max(high, *ratios) * 1.2),
        ylabel="TensorRT / PyTorch throughput ratio",
        xlabel="Pair; error bar resamples paired trials",
        title=f"{speedup:.2f}× paired geometric mean",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"GTX 1650 Ti · {summary['pairs']} balanced randomized pairs · ≥{plan['minimum_trial_seconds']:g}s per trial"
    )
    fig.text(
        0.04,
        0.025,
        "Real-frame warm-up before every trial. Predecoded model-only throughput; decoding, tracking and rendering excluded.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(root / "same-gpu-comparison.png", dpi=150)
    plt.close(fig)
    lines = [
        "# Same-GPU confirmation study",
        "",
        f"TensorRT FP16 enabled versus PyTorch CUDA FP32, YOLO11n at 320px: **{speedup:.2f}×** paired geometric-mean",
        f"throughput ratio; exploratory 95% paired-trial bootstrap interval **{low:.2f}–{high:.2f}×**.",
        "",
        f"Aggregate throughput: **{baseline:.2f} FPS → {candidate:.2f} FPS**. Each backend ran {summary['pairs']} trials",
        f"of at least {plan['minimum_trial_seconds']:g} seconds, with three PyTorch-first and three TensorRT-first pairs randomized by seed.",
        "",
        "![Paired same-GPU throughput](../results/confirmation/same-gpu-comparison.png)",
        "",
        "## Protocol and quality",
        "",
        "- Both models resident on the same GTX 1650 Ti, executed sequentially with the same CPU thread setting and batch one.",
        "- A 60-frame every-other-frame pool spans the first 120 source frames. Each trial processes complete cycles only,",
        "  so the frame mixture is identical despite different processing rates. Decode is excluded.",
        "- One complete real-image pool warms the selected runtime before each trial. CPU box transfer completes before the timer stops.",
        "- Accuracy is reused from the existing 256-image holdout only after matching the exact model hashes and semantic JSON identity.",
        f"- Global mAP loss: **{audit['global_map_drop_points']:.4f} points**; mean vehicle AP loss: **{audit['vehicle_map_drop_points']:.4f} points**.",
        "  Negative loss means a higher measured value; it is not evidence of a statistically established accuracy improvement.",
        "- The interval resamples paired trials, not correlated frames. Six pairs on one shared laptop remain a small experiment.",
        "- This supplements the original 28-case matrix with a different warm-up, frame pool and model-residency protocol. Do not pool the two studies.",
        "",
        "## Individual pairs",
        "",
        "| Pair | Order | PyTorch FPS / p95 ms | TensorRT FPS / p95 ms | FPS ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for pair, order in enumerate(plan["orders"]):
        a, b = [
            next(t for t in trials if t["pair"] == pair and t["runtime"] == name)
            for name in ("pytorch", "tensorrt")
        ]
        lines.append(
            f"| {pair + 1} | {' → '.join(order)} | {a['fps']:.2f} / {a['p95_ms']:.2f} | "
            f"{b['fps']:.2f} / {b['p95_ms']:.2f} | {b['fps'] / a['fps']:.2f}× |"
        )
    lines += [
        "",
        "## Reproduce and audit",
        "",
        "```powershell",
        "python scripts/confirm_gpu_speedup.py --output results/confirmation-rerun",
        "python scripts/audit_confirmation.py --input results/confirmation-rerun",
        "```",
        "",
        "The original raw trials and resource samples are under [results/confirmation](../results/confirmation/).",
        "CI audits their identities, timing arithmetic, frame composition, paired order, quality links and derived statistics without a GPU.",
    ]
    Path("docs/CONFIRMATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    supported = low > 1 and audit["accuracy_budget_passed"]
    performance_bullet = (
        f"- Achieved **{speedup:.2f}× paired model-throughput speedup** with TensorRT FP16 versus PyTorch CUDA FP32 on a "
        "GTX 1650 Ti, confirmed across six balanced randomized trial pairs with real-image warm-up and held-out accuracy checks."
        if supported
        else "- Built a randomized paired-trial framework to compare TensorRT and PyTorch CUDA with real-image warm-up and held-out accuracy checks."
    )
    cv = [
        "# CV-ready project description",
        "",
        "**YOLO Inference Optimization & Multi-Camera Analytics**",
        "",
        "Python · PyTorch · CUDA · TensorRT · ONNX Runtime · OpenVINO · OpenCV · GitHub Actions",
        "",
        "## Copy-ready bullets",
        "",
        performance_bullet,
        "- Benchmarked 28 runtime/precision/resolution configurations using 512 calibration images and a disjoint 256-image COCO holdout; verified execution precision and rejected configurations exceeding accuracy-loss budgets.",
        "- Implemented bounded camera queues, independent tracking/counting, resource telemetry and a results dashboard; profiled 24 serving experiments and automated evidence audits in CI.",
        "",
        "## Defensible claims",
        "",
        "| Claim | Supporting evidence |",
        "|---|---|",
        "| Same-GPU runtime optimization | [Randomized confirmation](CONFIRMATION.md), raw trial records and matched model hashes |",
        "| Reproducible backend comparison | [Full matrix](RESULTS.md), exported-graph/engine inspection, calibration split and environment locks |",
        "| Systems profiling and overload handling | Per-camera drops, frame age, decoder lag and telemetry measurements across 24 runs |",
        "",
        "Keep the hardware, resolution, baseline and model-only endpoint attached to performance numbers. The original CPU-to-GPU ratio mixes hardware and runtime changes; use the same-GPU result above for an optimization claim.",
        "Do not describe the eight-camera load test as eight cameras meeting real-time requirements. Only the measured passing loads support that claim; counting accuracy and production reliability remain unvalidated.",
        "",
        "The repository remains private as requested. A CV link requires the reviewer to have repository access; the checked-in report and chart can also be shared as artifacts.",
        "",
        "Use [the walkthrough](WALKTHROUGH.md) to prepare an explanation of the design, rejected optimizations and limitations.",
    ]
    Path("docs/CV_READY.md").write_text("\n".join(cv) + "\n", encoding="utf-8")
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    begin, end = "<!-- confirmation:begin -->", "<!-- confirmation:end -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError("README needs exactly one confirmation block")
    label = "speedup" if supported else "ratio"
    headline = (
        f"**Same-GPU confirmation:** TensorRT FP16 enabled recorded **{speedup:.2f}× paired model-throughput {label}** "
        "versus PyTorch CUDA FP32\n"
        f"at 320px on the GTX 1650 Ti. Aggregate throughput was **{baseline:.2f} → {candidate:.2f} FPS** across "
        f"{summary['pairs']} balanced randomized trial pairs\n"
        f"with real-image warm-up and at least {plan['minimum_trial_seconds']:g} seconds per trial. "
        f"Matched-artifact accuracy budget: **{'PASS' if audit['accuracy_budget_passed'] else 'REJECT'}**.\n"
        "The [confirmation protocol](docs/CONFIRMATION.md) explains the quality checks, uncertainty interval and differences "
        "from the original matrix. Decoding, tracking and rendering are excluded."
    )
    readme.write_text(
        text.split(begin)[0] + begin + "\n" + headline + "\n" + end + text.split(end)[1], encoding="utf-8"
    )
    print(
        f"Generated verified CV wording and chart: {speedup:.2f}x, exploratory interval {low:.2f}–{high:.2f}x"
    )


if __name__ == "__main__":
    main()
