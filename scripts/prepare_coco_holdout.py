"""Deterministic disjoint calibration/holdout subsets of the official COCO 2017 validation set."""

import concurrent.futures
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


def download(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "TrafficInferenceLab/0.2"}), timeout=60
            ) as response:
                content = response.read()
            if not content:
                raise ValueError("Empty download")
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(content)
            temporary.replace(path)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    root = Path("data/coco-deployment").resolve()
    archive = Path(".build-cache/annotations_trainval2017.zip")
    annotations = root / "instances_val2017.json"
    if not annotations.exists():
        download(
            "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
            archive,
        )
        with zipfile.ZipFile(archive) as source:
            annotations.parent.mkdir(parents=True, exist_ok=True)
            annotations.write_bytes(source.read("annotations/instances_val2017.json"))
    data = json.loads(annotations.read_text())
    images = sorted(data["images"], key=lambda row: row["id"])
    random.Random(20261030).shuffle(images)
    calibration, validation = images[:512], images[512:768]
    categories = sorted(data["categories"], key=lambda row: row["id"])
    category_map = {category["id"]: index for index, category in enumerate(categories)}
    labels = {}
    for annotation in data["annotations"]:
        if annotation.get("iscrowd"):
            continue
        labels.setdefault(annotation["image_id"], []).append(annotation)
    tasks = []
    manifest = {
        "seed": 20261030,
        "source": "COCO 2017 validation",
        "calibration": [],
        "holdout": [],
        "annotation_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "note": "Calibration and evaluation are disjoint. Official val2017 images, no fine-tuning.",
    }
    for split, subset in (("calibration", calibration), ("holdout", validation)):
        for image in subset:
            path = root / "images" / split / image["file_name"]
            url = "https://s3.amazonaws.com/images.cocodataset.org/val2017/" + image["file_name"]
            tasks.append((url, path))
            text = []
            for a in labels.get(image["id"], []):
                x, y, w, h = a["bbox"]
                # COCO boxes can extend slightly past the image; clip before normalization.
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(image["width"], x + w), min(image["height"], y + h)
                if x2 <= x1 or y2 <= y1:
                    continue
                text.append(
                    f"{category_map[a['category_id']]} {(x1 + x2) / 2 / image['width']:.8f} "
                    f"{(y1 + y2) / 2 / image['height']:.8f} {(x2 - x1) / image['width']:.8f} "
                    f"{(y2 - y1) / image['height']:.8f}"
                )
            target = root / "labels" / split / Path(image["file_name"]).with_suffix(".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(text) + ("\n" if text else ""))
            manifest[split].append({"id": image["id"], "file_name": image["file_name"], "url": url})
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download, url, path) for url, path in tasks]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if i % 64 == 0:
                print(f"Downloaded/verified {i}/{len(tasks)} images", flush=True)
    names = {i: c["name"] for i, c in enumerate(categories)}
    # JSON is valid YAML and avoids an extra authoring dependency.
    for name, split in (("calibration", "calibration"), ("holdout", "holdout")):
        config = {
            "path": root.as_posix(),
            "train": f"images/{split}",
            "val": f"images/{split}",
            "names": names,
        }
        (root / f"{name}.yaml").write_text(json.dumps(config, indent=2))
    assert not ({i["id"] for i in calibration} & {i["id"] for i in validation})
    for split in ("calibration", "holdout"):
        for row in manifest[split]:
            row["sha256"] = hashlib.sha256(
                (root / "images" / split / row["file_name"]).read_bytes()
            ).hexdigest()
    output = Path("results/full-matrix")
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Ready: 512 calibration images, 256 disjoint held-out images. {root}", flush=True)


if __name__ == "__main__":
    main()
