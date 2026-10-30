"""Fetch NVIDIA's official Windows SDK when pip's Linux-only library stub is unsuitable."""

import concurrent.futures
import hashlib
import shutil
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.8.0/zip/"
    "TensorRT-10.8.0.43.Windows.win10.cuda-12.8.zip"
)


def main():
    archive = Path(".build-cache/TensorRT-10.8.0.43.Windows.win10.cuda-12.8.zip")
    size = int(urlopen(Request(URL, method="HEAD"), timeout=60).headers["Content-Length"])
    parts = Path(".build-cache/tensorrt-parts")
    parts.mkdir(parents=True, exist_ok=True)
    block = 8 * 1024**2
    count = (size + block - 1) // block

    def download(index):
        start, end = index * block, min((index + 1) * block, size) - 1
        path = parts / f"{index:04}.part"
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            try:
                with urlopen(
                    Request(URL, headers={"Range": f"bytes={start}-{end}"}), timeout=120
                ) as response:
                    if response.status != 206:
                        raise RuntimeError("NVIDIA server ignored byte range")
                    content = response.read()
                if len(content) != end - start + 1:
                    raise RuntimeError("Incomplete SDK range")
                path.write_bytes(content)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    if not archive.exists() or archive.stat().st_size != size:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(download, i) for i in range(count)]
            for n, result in enumerate(concurrent.futures.as_completed(futures), 1):
                result.result()
                if n % 16 == 0:
                    print(f"TensorRT SDK: {n}/{count} ranges", flush=True)
        with archive.open("wb") as destination:
            for i in range(count):
                with (parts / f"{i:04}.part").open("rb") as source:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
    output = Path(".build-cache/tensorrt-sdk")
    with zipfile.ZipFile(archive) as sdk:
        for info in sdk.infolist():
            name = info.filename
            if (
                ("/lib/" in name and name.endswith(".dll"))
                or ("/python/" in name and "cp312" in name and name.endswith(".whl"))
                or name.endswith("/trtexec.exe")
            ):
                target = output / Path(name).name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.stat().st_size != info.file_size:
                    with sdk.open(info) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination, 1024 * 1024)
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    print(f"Extracted verified ZIP entries to {output}; archive SHA-256: {digest}", flush=True)


if __name__ == "__main__":
    main()
