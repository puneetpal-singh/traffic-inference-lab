"""Small, independently testable analytics and measurement primitives."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "n": len(a),
        "mean": float(a.mean()),
        **{f"p{p}": float(np.percentile(a, p)) for p in (50, 95, 99)},
    }


def zone_occupancy(tracks, width, height, zone):
    """Count tracked bottom-center anchors inside a normalized rectangular ROI."""
    x1, y1, x2, y2 = zone
    return sum(
        x1 <= (float(t[0]) + float(t[2])) / (2 * width) <= x2 and y1 <= float(t[3]) / height <= y2
        for t in tracks
    )


def iou(a, b):
    a, b = np.asarray(a)[:4], np.asarray(b)[:4]
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = float(np.maximum(hi - lo, 0).prod())
    union = float(np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod() - inter)
    return inter / union if union > 0 else 0.0


def match_detections(reference, candidate, min_iou=0.9):
    """Rows: xyxy, confidence, class. Maximum-cardinality, class-aware matching."""
    from scipy.optimize import linear_sum_assignment

    if len(reference) == 0 or len(candidate) == 0:
        return {
            "reference": len(reference),
            "candidate": len(candidate),
            "matched": 0,
            "min_iou": None,
            "max_conf_delta": None,
        }
    overlap = np.array([[iou(a, b) if int(a[5]) == int(b[5]) else 0 for b in candidate] for a in reference])
    # An eligible edge dominates any sum of ineligible IoU improvements.
    eligible = overlap >= min_iou
    costs = -(eligible * (max(overlap.shape) + 1) + overlap)
    rr, cc = linear_sum_assignment(costs)
    pairs = [(r, c) for r, c in zip(rr, cc) if eligible[r, c]]
    return {
        "reference": len(reference),
        "candidate": len(candidate),
        "matched": len(pairs),
        "min_iou": min((float(overlap[r, c]) for r, c in pairs), default=None),
        "max_conf_delta": max(
            (abs(float(reference[r][4] - candidate[c][4])) for r, c in pairs), default=None
        ),
    }


@dataclass
class LineCounter:
    """Count finite-segment crossings once per ID and direction; use normalized anchors.

    Deadband suppresses line jitter. State expires so long-running streams remain bounded.
    Tracker identity switches can still overcount; evaluate against human event labels.
    """

    start: tuple = (0.1, 0.55)
    end: tuple = (0.9, 0.55)
    deadband: float = 0.01
    ttl: int = 120
    state: dict = field(default_factory=dict)
    totals: Counter = field(default_factory=Counter)

    def update(self, tracks, width, height, frame_index):
        self.state = {k: v for k, v in self.state.items() if frame_index - v[2] <= self.ttl}
        ax, ay = self.start
        bx, by = self.end
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length == 0:
            raise ValueError("Counting line must have nonzero length")
        events = []
        for t in tracks:
            ident = int(t[4])
            point = ((float(t[0]) + float(t[2])) / (2 * width), float(t[3]) / height)
            distance = (dx * (point[1] - ay) - dy * (point[0] - ax)) / length
            if abs(distance) <= self.deadband:
                if ident in self.state:
                    old = self.state[ident]
                    self.state[ident] = (old[0], old[1], frame_index, old[3])
                continue
            side = 1 if distance > 0 else -1
            previous = self.state.get(ident)
            seen = set(previous[3]) if previous else set()
            if previous and previous[0] != side:
                p = previous[1]
                old_dist = dx * (p[1] - ay) - dy * (p[0] - ax)
                new_dist = distance * length
                alpha = old_dist / (old_dist - new_dist)
                cross = (p[0] + alpha * (point[0] - p[0]), p[1] + alpha * (point[1] - p[1]))
                position = ((cross[0] - ax) * dx + (cross[1] - ay) * dy) / length**2
                direction = "positive" if side > 0 else "negative"
                if 0 <= position <= 1 and direction not in seen:
                    seen.add(direction)
                    self.totals[direction] += 1
                    events.append(
                        {
                            "track_id": ident,
                            "direction": direction,
                            "class_id": int(t[6]),
                            "frame": frame_index,
                        }
                    )
            self.state[ident] = (side, point, frame_index, seen)
        return events


def pareto_indices(rows, latency_key="latency", quality_key="quality"):
    """Minimize latency, maximize quality. Do not substitute unknown quality with zero."""
    valid = [
        (i, r)
        for i, r in enumerate(rows)
        if r.get(latency_key) is not None and r.get(quality_key) is not None
    ]
    return [
        i
        for i, r in valid
        if not any(
            s[latency_key] <= r[latency_key]
            and s[quality_key] >= r[quality_key]
            and (s[latency_key] < r[latency_key] or s[quality_key] > r[quality_key])
            for j, s in valid
            if j != i
        )
    ]


def drift(reference, current, alpha=0.01, min_samples=100):
    """Confidence shift is a diagnostic, not proof of accuracy loss."""
    from scipy.stats import ks_2samp

    if min(len(reference), len(current)) < min_samples:
        return {"status": "insufficient_data", "reference_n": len(reference), "current_n": len(current)}
    result = ks_2samp(reference, current)
    effect = abs(float(np.mean(reference) - np.mean(current)))
    return {
        "status": "shift" if result.pvalue < alpha and result.statistic >= 0.1 else "stable",
        "ks_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_delta": effect,
        "alpha": alpha,
        "minimum_effect_ks": 0.1,
        "note": "Correlated video samples and repeated testing affect significance; review labels.",
    }
