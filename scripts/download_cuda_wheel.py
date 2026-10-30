"""Download the official CUDA wheel in verified ranges when the package mirror stalls."""

import concurrent.futures
import hashlib
import html
import re
import shutil
import time
from pathlib import Path
from urllib.parse import unquote
from urllib.request import Request, urlopen

NAME = "torch-2.6.0+cu124-cp312-cp312-win_amd64.whl"
INDEX = "https://download.pytorch.org/whl/cu124/torch/"
PAGE = urlopen(INDEX, timeout=60).read().decode()
links = [html.unescape(link) for link in re.findall(r'href="([^"]+)"', PAGE)]
link = next(link for link in links if NAME in unquote(link))
URL, digest = link.split("#sha256=")
# This official CloudFront endpoint supports byte ranges; avoid the stalled alternate mirror.
URL = "https://download.pytorch.org/whl/cu124/" + NAME.replace("+", "%2B")
size = int(urlopen(Request(URL, method="HEAD"), timeout=60).headers["Content-Length"])
root = Path(".build-cache/cuda-wheel-parts")
root.mkdir(parents=True, exist_ok=True)
block = 8 * 1024**2


def part(index):
    start, end = index * block, min((index + 1) * block, size) - 1
    target = root / f"{index:04}.part"
    if target.exists() and target.stat().st_size == end - start + 1:
        return
    for attempt in range(4):
        try:
            with urlopen(Request(URL, headers={"Range": f"bytes={start}-{end}"}), timeout=90) as response:
                if response.status != 206:
                    raise RuntimeError("Server did not honor the range request")
                content = response.read()
            if len(content) != end - start + 1:
                raise RuntimeError("Incomplete range")
            target.write_bytes(content)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


if __name__ == "__main__":
    count = (size + block - 1) // block
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(part, i) for i in range(count)]
        for n, result in enumerate(concurrent.futures.as_completed(futures), 1):
            result.result()
            if n % 16 == 0:
                print(f"CUDA wheel: {n}/{count} verified-length ranges", flush=True)
    output = Path(".build-cache/wheels") / NAME
    output.parent.mkdir(exist_ok=True)
    with output.open("wb") as destination:
        for i in range(count):
            with (root / f"{i:04}.part").open("rb") as source:
                shutil.copyfileobj(source, destination, 1024 * 1024)
    calculated = hashlib.file_digest(output.open("rb"), "sha256").hexdigest()
    if calculated != digest:
        raise RuntimeError(f"Official wheel SHA-256 mismatch: {calculated}")
    print(f"SHA-256 verified against the official wheel index: {output}", flush=True)
