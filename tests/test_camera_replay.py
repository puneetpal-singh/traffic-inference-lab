import threading
import time

import cv2
import numpy as np
import pytest

from traffic_lab.pipeline import Camera, Mailbox


def test_recorded_camera_limits_decoder_threads_and_accounts_for_overload(tmp_path):
    source = tmp_path / "camera.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 64))
    if not writer.isOpened():
        pytest.skip("MJPG codec unavailable in this OpenCV build")
    for value in (0, 40, 80, 120):
        writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
    writer.release()
    mailbox = Mailbox(1)
    camera = Camera(0, source, mailbox, time.perf_counter() + 0.01, 0.2, threading.Event(), decode_threads=1)
    camera.start()
    camera.join(timeout=5)
    assert camera.done.is_set()
    assert camera.error is None
    assert camera.decoded == 4
    assert mailbox.dropped == 3
    assert len(mailbox) == 1
    assert camera.actual_decode_threads == 1
