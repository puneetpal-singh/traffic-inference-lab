# Inputs, provenance, and licensing

The repository contains code and measurement artifacts, not pretrained weights or source videos.

## Expanded matrix: COCO val2017

The complete CPU/CUDA/TensorRT study uses **512 calibration images and 256 disjoint evaluation images** selected from
the official COCO 2017 validation set. `scripts/prepare_coco_holdout.py` sorts image IDs, shuffles with seed 20261030,
and selects nonoverlapping slices. Exact IDs, URLs, image hashes and the annotation archive's extracted JSON hash are
recorded in `results/full-matrix/dataset-manifest.json`. No fine-tuning uses either split.

The HTTPS path-style S3 endpoint accesses the official `images.cocodataset.org` bucket, avoiding that domain's certificate
hostname mismatch. Images and annotations remain under ignored `data/coco-deployment/`; only provenance and measurements
are committed. Review [COCO's source and terms](https://cocodataset.org/#termsofuse) for the individual images.

COCO boxes are clipped to image bounds and converted to YOLO format; `iscrowd` annotations are omitted. Reported metrics
are Ultralytics evaluation on this converted 256-image subset, not the official 5,000-image COCO leaderboard score.
The evaluation covers general detection and separately reports vehicle-class AP. It does not validate traffic line counts.

## Original smoke study and video

- Checkpoint: official Ultralytics `yolo11n.pt`, downloaded by `traffic-lab fetch`.
  See [YOLO11](https://docs.ultralytics.com/models/yolo11/) and
  [Ultralytics licensing](https://www.ultralytics.com/license). Review upstream terms for your use.
- Replay clip: [Ultralytics ANPR demo release asset](https://github.com/ultralytics/assets/releases/download/v0.0.0/anpr-demo-video.mp4).
  Used locally as a short road-vehicle workload, not a license plate recognition project. Source footage is not redistributed.
  An upstream release asset's availability does not independently establish the footage's redistribution rights.
- Optional smoke data: [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/), downloaded by Ultralytics validation.
  This is a small general-purpose sanity check, not an independent traffic holdout. No fine-tuning is performed.

`data/manifest.json` records URLs and SHA-256 hashes after fetching. Benchmark JSON includes the input and model hash.
Use your own appropriately licensed long clips and annotated traffic validation set for stronger conclusions.

Calibration and evaluation datasets must be separate. For INT8, supply a calibration YAML with representative day/night,
camera-angle and object-size diversity. Do not calibrate on the same images used to advertise accuracy.

Human crossing-label format (illustrative schema, not actual ground truth):

```json
[{"camera": 0, "direction": "positive", "class_id": 2, "source_seconds": 1.25}]
```
