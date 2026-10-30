import numpy as np
import pytest

from traffic_lab.metrics import LineCounter, distribution, drift, match_detections, pareto_indices
from traffic_lab.pipeline import Mailbox
from traffic_lab.scoring import score_events


def test_fp16_large_pixel_boxes_do_not_overflow_during_parity():
    from traffic_lab.metrics import iou

    box = np.array([0, 0, 1080, 1920, 0.9, 2], dtype=np.float16)
    assert iou(box, box) == pytest.approx(1)


def test_zone_occupancy_uses_bottom_center_and_normalized_bounds():
    from traffic_lab.metrics import zone_occupancy

    tracks = [track(50, 50), track(10, 50), track(50, 95)]
    assert zone_occupancy(tracks, 100, 100, (0.2, 0.2, 0.8, 0.8)) == 1


def track(x, y, ident=1):
    return [x - 2, y - 4, x + 2, y, ident, 0.9, 2, 0]


def test_count_crossing_jitter_and_duplicate_suppression():
    counter = LineCounter((0.2, 0.5), (0.8, 0.5))
    assert not counter.update([track(50, 40)], 100, 100, 0)
    assert not counter.update([track(50, 50)], 100, 100, 1)
    assert len(counter.update([track(50, 60)], 100, 100, 2)) == 1
    assert len(counter.update([track(50, 40)], 100, 100, 3)) == 1
    assert not counter.update([track(50, 60)], 100, 100, 4)
    assert dict(counter.totals) == {"positive": 1, "negative": 1}


def test_crossing_outside_finite_segment_not_counted():
    counter = LineCounter((0.2, 0.5), (0.8, 0.5))
    counter.update([track(95, 40)], 100, 100, 0)
    assert not counter.update([track(95, 60)], 100, 100, 1)


def test_state_expires_and_cameras_are_independent():
    a, b = LineCounter(ttl=2), LineCounter(ttl=2)
    a.update([track(50, 40)], 100, 100, 0)
    assert not b.update([track(50, 70)], 100, 100, 1)
    assert not a.update([track(50, 70)], 100, 100, 5)
    assert len(a.state) == 1


def test_latest_queue_bound_and_accounting():
    q = Mailbox(2)
    for i in range(10):
        q.put(i)
    assert len(q) == q.high_water == 2
    assert q.dropped == 8
    assert [q.get(), q.get(), q.get()] == [8, 9, None]
    with pytest.raises(ValueError):
        Mailbox(0)


def test_parity_class_aware_one_to_one():
    ref = np.array([[0, 0, 10, 10, 0.9, 2], [20, 0, 30, 10, 0.8, 7]])
    assert match_detections(ref, ref[::-1])["matched"] == 2
    wrong = ref.copy()
    wrong[0, 5] = 3
    assert match_detections(ref, wrong)["matched"] == 1
    assert match_detections(ref, ref[:1])["matched"] == 1
    assert match_detections([], [ref[0]])["candidate"] == 1


def test_pareto_preserves_tradeoffs_and_ignores_unknown_quality():
    rows = [
        dict(latency=10, quality=0.5),
        dict(latency=20, quality=0.6),
        dict(latency=30, quality=0.4),
        dict(latency=1, quality=None),
    ]
    assert pareto_indices(rows) == [0, 1]


def test_distribution_handles_empty_and_tail():
    assert distribution([])["p95"] is None
    assert distribution([1, 2, 3])["p95"] == pytest.approx(2.9)


def test_drift_requires_data_and_effect():
    assert drift([0.5], [0.9])["status"] == "insufficient_data"
    assert drift([0.5] * 100, [0.9] * 100)["status"] == "shift"
    assert drift([0.5] * 100, [0.5] * 100)["status"] == "stable"


def event(t, direction="positive"):
    return dict(camera=0, direction=direction, class_id=2, source_seconds=t)


def test_event_matching_detects_duplicate_and_wrong_direction():
    result = score_events([event(1), event(2)], [event(1), event(1.1), event(2, "negative")], 0.2)
    assert result["true_positive"] == 1
    assert result["false_positive"] == 2
    assert result["false_negative"] == 1
    assert result["f1"] == pytest.approx(0.4)


def test_empty_count_score():
    assert score_events([], [])["f1"] == 1
    assert score_events([event(1)], [])["recall"] == 0
