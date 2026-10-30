import time

from traffic_lab.resources import ResourceMonitor


def test_monitor_collects_process_samples_and_stops_cleanly():
    with ResourceMonitor(interval=0.01) as monitor:
        time.sleep(0.05)
    assert not monitor.thread.is_alive()
    summary = monitor.summary()
    assert summary["metrics"]["process_rss_mib"]["max"] > 0
    assert summary["samples"]
