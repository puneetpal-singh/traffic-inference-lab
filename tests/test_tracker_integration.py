"""Use the pinned upstream tracker, so constructor/state API changes cannot hide behind mocks."""

import numpy as np
from ultralytics.engine.results import Boxes

from traffic_lab.pipeline import make_tracker


def boxes(offset=0):
    return Boxes(np.array([[10 + offset, 10, 40 + offset, 50, 0.95, 2]], dtype=np.float32), (100, 100))


def test_real_bytetrack_ids_stay_stable_and_camera_state_is_independent():
    first, second = make_tracker("bytetrack", 30), make_tracker("bytetrack", 30)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    a = first.update(boxes(), image)
    b = second.update(boxes(40), image)
    assert len(a) == len(b) == 1
    a_next = first.update(boxes(1), image)
    assert int(a_next[0, 4]) == int(a[0, 4])
    assert first.frame_id == 2
    assert second.frame_id == 1


def test_real_botsort_constructs_and_tracks_without_reid_weights():
    tracker = make_tracker("botsort", 30)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert len(tracker.update(boxes(), image)) == 1
