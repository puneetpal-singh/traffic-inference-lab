"""One detector, independent per-camera trackers, bounded latest-frame mailboxes."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import psutil

from .metrics import LineCounter, distribution, zone_occupancy
from .resources import ResourceMonitor
from .runtime import Detector, environment, file_hash, save_json


def open_recording(source, decode_threads=1):
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, decode_threads])
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(source))
    return cap


@dataclass
class Packet:
    camera: int
    index: int
    frame: np.ndarray
    scheduled: float
    decoded: float
    decode_ms: float


class Mailbox:
    def __init__(self, capacity=1):
        if capacity < 1:
            raise ValueError("Queue capacity must be positive")
        self.capacity = capacity
        self.items = deque()
        self.lock = threading.Lock()
        self.dropped = 0
        self.high_water = 0

    def put(self, packet):
        with self.lock:
            if len(self.items) == self.capacity:
                self.items.popleft()
                self.dropped += 1
            self.items.append(packet)
            self.high_water = max(self.high_water, len(self.items))

    def get(self):
        with self.lock:
            return self.items.popleft() if self.items else None

    def __len__(self):
        with self.lock:
            return len(self.items)


class Camera(threading.Thread):
    def __init__(self, camera, source, mailbox, start_time, duration, stop, decode_threads=1):
        super().__init__(daemon=True)
        self.camera, self.source, self.mailbox = camera, source, mailbox
        self.start_time, self.duration, self.stop = start_time, duration, stop
        self.done = threading.Event()
        self.decoded = 0
        self.error = None
        self.source_fps = None
        self.decode_threads = decode_threads
        self.actual_decode_threads = None

    def run(self):
        cap = open_recording(self.source, self.decode_threads)
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open {self.source}")
            self.source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            self.actual_decode_threads = int(cap.get(cv2.CAP_PROP_N_THREADS))
            if not np.isfinite(self.source_fps) or self.source_fps <= 0:
                raise ValueError("Recorded source requires valid FPS metadata")
            index = 0
            while index / self.source_fps < self.duration and not self.stop.is_set():
                scheduled = self.start_time + index / self.source_fps
                if self.stop.wait(max(0, scheduled - time.perf_counter())):
                    break
                tick = time.perf_counter()
                ok, frame = cap.read()
                decoded = time.perf_counter()
                if not ok:
                    break
                self.mailbox.put(
                    Packet(self.camera, index, frame, scheduled, decoded, (decoded - tick) * 1000)
                )
                self.decoded += 1
                index += 1
        except Exception as exc:
            self.error = str(exc)
        finally:
            cap.release()
            self.done.set()


def make_tracker(name, frame_rate):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    args = IterableSimpleNamespace(**YAML.load(check_yaml(f"{name}.yaml")))
    # The pinned release uses a frame-count buffer and no frame_rate constructor argument.
    args.track_buffer = max(1, round(args.track_buffer * frame_rate / 30))
    return {"bytetrack": BYTETracker, "botsort": BOTSORT}[name](args=args)


def run_pipeline(
    config,
    sources,
    output,
    duration=10,
    queue_size=1,
    batch=1,
    stride=1,
    tracker="bytetrack",
    line=(0.1, 0.55, 0.9, 0.55),
    render=False,
    zone=(0.2, 0.3, 0.8, 0.8),
    decode_threads=1,
):
    if not sources or min(duration, queue_size, batch, stride, decode_threads) <= 0:
        raise ValueError("Sources and positive duration, queue, batch, stride are required")
    if len(line) != 4 or any(not 0 <= v <= 1 for v in line) or line[:2] == line[2:]:
        raise ValueError("Line must have distinct normalized endpoints")
    if len(zone) != 4 or not (0 <= zone[0] < zone[2] <= 1 and 0 <= zone[1] < zone[3] <= 1):
        raise ValueError("Zone must be a normalized rectangle: x1 y1 x2 y2")
    detector = Detector(config)
    # The scheduler never batches two frames from the same camera.
    effective_batch = min(batch, len(sources))
    detector.warmup(batch=effective_batch)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mailboxes = [Mailbox(queue_size) for _ in sources]
    stop = threading.Event()
    start = time.perf_counter() + 0.2
    cameras = [
        Camera(i, path, mailboxes[i], start, duration, stop, decode_threads) for i, path in enumerate(sources)
    ]
    rates, warmup_frames = [], []
    for source in sources:
        probe = open_recording(source, decode_threads)
        rate = float(probe.get(cv2.CAP_PROP_FPS))
        opened = probe.isOpened()
        decoded, first_frame = probe.read() if opened else (False, None)
        probe.release()
        if not opened or not decoded or not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"Source must be a decodable recording with valid FPS: {source}")
        rates.append(rate)
        warmup_frames.append(first_frame)
    # Real detections exercise NMS paths that zero-image warm-up may never call.
    # Exercise every allowed batch shape before the replay clock starts.
    for size in range(1, effective_batch + 1):
        for result in detector.predict(warmup_frames[:size]):
            result.boxes.cpu()
    del warmup_frames
    trackers = [make_tracker(tracker, max(1, round(rate))) for rate in rates]
    counters = [LineCounter(tuple(line[:2]), tuple(line[2:])) for _ in sources]
    # Importing tracker dependencies can be slow on first use. The replay clock
    # must start after initialization, otherwise startup becomes fake decode lag.
    monitor = ResourceMonitor()
    monitor.__enter__()
    start = time.perf_counter() + 0.2
    for camera in cameras:
        camera.start_time = start
    records, events, writers = [], [], {}
    skipped = [0] * len(sources)
    process = psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    cursor = 0
    error = None
    try:
        for camera in cameras:
            camera.start()
        with (output / "frames.jsonl").open("w", encoding="utf-8") as log:
            import json

            while True:
                packets = []
                for offset in range(len(sources)):
                    idx = (cursor + offset) % len(sources)
                    packet = mailboxes[idx].get()
                    if packet is None:
                        continue
                    if packet.index % stride:
                        skipped[idx] += 1
                        continue
                    packets.append(packet)
                    if len(packets) == effective_batch:
                        break
                cursor = (cursor + 1) % len(sources)
                if not packets:
                    if all(c.done.is_set() for c in cameras) and not any(len(q) for q in mailboxes):
                        break
                    stop.wait(0.001)
                    continue
                predict_start = time.perf_counter()
                results = detector.predict([p.frame for p in packets])
                prediction_end = time.perf_counter()
                for packet, result in zip(packets, results):
                    idx = packet.camera
                    tick = time.perf_counter()
                    tracks = trackers[idx].update(result.boxes.cpu().numpy(), packet.frame)
                    height, width = packet.frame.shape[:2]
                    new_events = counters[idx].update(tracks, width, height, packet.index)
                    occupancy = zone_occupancy(tracks, width, height, zone)
                    events.extend(
                        {"camera": idx, "source_seconds": packet.index / cameras[idx].source_fps, **event}
                        for event in new_events
                    )
                    tracking_end = time.perf_counter()
                    if render:
                        canvas = packet.frame.copy()
                        for t in tracks:
                            x1, y1, x2, y2 = map(int, t[:4])
                            cv2.rectangle(canvas, (x1, y1), (x2, y2), (88, 220, 80), 2)
                            cv2.putText(
                                canvas,
                                f"ID {int(t[4])}",
                                (x1, max(20, y1)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (88, 220, 80),
                                2,
                            )
                        cv2.line(
                            canvas,
                            (int(line[0] * width), int(line[1] * height)),
                            (int(line[2] * width), int(line[3] * height)),
                            (0, 220, 255),
                            2,
                        )
                        cv2.rectangle(
                            canvas,
                            (int(zone[0] * width), int(zone[1] * height)),
                            (int(zone[2] * width), int(zone[3] * height)),
                            (220, 160, 40),
                            2,
                        )
                        cv2.putText(
                            canvas,
                            f"Camera {idx} | count {sum(counters[idx].totals.values())}",
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        if idx not in writers:
                            writers[idx] = cv2.VideoWriter(
                                str(output / f"camera-{idx}.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                cameras[idx].source_fps / stride,
                                (width, height),
                            )
                            if not writers[idx].isOpened():
                                raise RuntimeError("Video writer failed to open")
                        writers[idx].write(canvas)
                    end = time.perf_counter()
                    row = {
                        "camera": idx,
                        "frame": packet.index,
                        "source_seconds": packet.index / cameras[idx].source_fps,
                        "decode_ms": packet.decode_ms,
                        "decoder_lag_ms": max(0, (packet.decoded - packet.scheduled) * 1000),
                        "queue_wait_ms": (predict_start - packet.decoded) * 1000,
                        "batch_predict_ms": (prediction_end - predict_start) * 1000,
                        "tracking_count_ms": (tracking_end - tick) * 1000,
                        "render_ms": (end - tracking_end) * 1000,
                        "frame_age_ms": (end - packet.scheduled) * 1000,
                        "detections": len(result.boxes),
                        "tracks": len(tracks),
                        "zone_occupancy": occupancy,
                        "track_positions_normalized": [
                            [
                                float(np.clip((t[0] + t[2]) / (2 * width), 0, 1)),
                                float(np.clip(t[3] / height, 0, 1)),
                            ]
                            for t in tracks
                        ],
                        "confidences": result.boxes.conf.cpu().tolist(),
                        "brightness": float(np.mean(cv2.cvtColor(packet.frame, cv2.COLOR_BGR2GRAY))),
                        "rss_mib": process.memory_info().rss / 1024**2,
                    }
                    telemetry_end = time.perf_counter()
                    row["analytics_frame_age_ms"] = row["frame_age_ms"]
                    row["telemetry_ms"] = (telemetry_end - end) * 1000
                    row["frame_age_ms"] = (telemetry_end - packet.scheduled) * 1000
                    log.write(json.dumps(row) + "\n")
                    records.append(row)
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        stop.set()
        for camera in cameras:
            if camera.ident is not None:
                camera.join(timeout=10)
        for writer in writers.values():
            writer.release()
        monitor.__exit__()
        elapsed = max(time.perf_counter() - start, 1e-9)
        per_camera = []
        for idx, camera in enumerate(cameras):
            rows = [r for r in records if r["camera"] == idx]
            per_camera.append(
                {
                    "camera": idx,
                    "source": str(sources[idx]),
                    "source_fps": camera.source_fps,
                    "decoder_threads": camera.actual_decode_threads,
                    "decoded": camera.decoded,
                    "processed": len(rows),
                    "queue_dropped": mailboxes[idx].dropped,
                    "stride_skipped": skipped[idx],
                    "remaining": len(mailboxes[idx]),
                    "queue_high_water": mailboxes[idx].high_water,
                    "processed_fps": len(rows) / elapsed,
                    "frame_age_ms": distribution([r["frame_age_ms"] for r in rows]),
                    "queue_wait_ms": distribution([r["queue_wait_ms"] for r in rows]),
                    "counts": dict(counters[idx].totals),
                    "error": camera.error,
                }
            )
        summary = {
            "kind": "paced_recorded_multistream",
            "schema_version": 2,
            "environment": environment(),
            **detector.metadata(),
            "input_hashes": [file_hash(s) for s in sources],
            "elapsed_seconds": elapsed,
            "requested_seconds": duration,
            "requested_batch": batch,
            "effective_max_batch": effective_batch,
            "queue_capacity_per_camera": queue_size,
            "stride": stride,
            "requested_decoder_threads": decode_threads,
            "warmup": {"zero_image_calls": 5, "real_image_batch_sizes": list(range(1, effective_batch + 1))},
            "resources": monitor.summary(),
            "tracker": tracker,
            "tracker_configs": [vars(t.args) for t in trackers],
            "line": list(line),
            "zone": list(zone),
            "render": render,
            "cameras": per_camera,
            "process_cpu_one_core_percent": 100 * (sum(process.cpu_times()[:2]) - cpu_start) / elapsed,
            "peak_sampled_rss_mib": max((r["rss_mib"] for r in records), default=None),
            "error": error,
            "status": "failed" if error or any(c.error for c in cameras) else "complete",
            "limitations": [
                "Recorded sources replayed at metadata FPS; not an RTSP network test.",
                "Frame age starts at scheduled source arrival and includes decode backlog.",
                "Rendered video contains processed frames only; gaps are compressed.",
                "Counts are unvalidated until compared with independent human event labels.",
                "Tracker time uses update steps; frame skips/drops change its effective time scale.",
                "No cross-camera identity association. Metadata preparation included; log writing excluded from frame age.",
            ],
        }
        save_json(output / "summary.json", summary)
        save_json(output / "events.json", events)
    if summary["status"] != "complete":
        raise RuntimeError("Camera failed; inspect summary.json")
    summary["browser_videos"] = []
    if render and shutil.which("ffmpeg"):
        # Browser-compatible conversion is an artifact step, outside timed pipeline execution.
        for idx in writers:
            target = output / f"camera-{idx}-browser.mp4"
            conversion = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(output / f"camera-{idx}.mp4"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(target),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if conversion.returncode == 0:
                summary["browser_videos"].append(str(target))
            else:
                summary.setdefault("artifact_warnings", []).append(conversion.stderr[-500:])
    save_json(output / "summary.json", summary)
    return summary
