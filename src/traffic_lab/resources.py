"""Low-frequency process and whole-device telemetry, measured outside stage timer instrumentation."""

import threading
import time

import psutil

from .metrics import distribution


class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()
        self.thread = None
        self.nvml = self.handle = None
        self.error = None

    def __enter__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:
            self.error = str(exc)
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            row = {
                "seconds": time.perf_counter() - self.started,
                "process_rss_mib": process.memory_info().rss / 1024**2,
                "system_available_ram_mib": psutil.virtual_memory().available / 1024**2,
            }
            if self.handle is not None:
                queries = {
                    "gpu_used_mib": lambda: self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024**2,
                    "gpu_utilization_percent": lambda: (
                        self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                    ),
                    "gpu_temperature_c": lambda: self.nvml.nvmlDeviceGetTemperature(self.handle, 0),
                    "gpu_power_w": lambda: self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000,
                    "gpu_sm_clock_mhz": lambda: self.nvml.nvmlDeviceGetClockInfo(self.handle, 1),
                }
                for name, query in queries.items():
                    try:
                        row[name] = float(query())
                    except Exception:
                        row[name] = None
            self.samples.append(row)
            self.stop.wait(self.interval)

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.nvml is not None:
            self.nvml.nvmlShutdown()

    def summary(self):
        keys = sorted({key for row in self.samples for key in row if key != "seconds"})
        return {
            "sample_interval_seconds": self.interval,
            "nvml_error": self.error,
            "metrics": {
                key: {
                    **distribution([r[key] for r in self.samples if r.get(key) is not None]),
                    "max": max((r[key] for r in self.samples if r.get(key) is not None), default=None),
                }
                for key in keys
            },
            "samples": self.samples,
            "scope": "NVML readings cover the whole GPU, not exclusively this process. Power may be unsupported.",
        }
