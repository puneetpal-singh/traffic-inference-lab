# Kaggle T4 results, 2 November 2026

The original checkpoint was downloaded from Kaggle and audited locally without inference or engine builds.
The archive passed CRC validation and **2753 identity/arithmetic checks**, including historical embedded-source
fingerprints, model hashes, disjoint image IDs, trial composition/FPS/latency arithmetic and camera/drop accounting.
This is an artifact audit, not independent reproduction of the experiment.

The [complete analysis](../results/kaggle-t4-320-2026-11-02/Kaggle_Results_Analysis.md) includes all model and serving
rows, paired throughput comparisons, stage profiles, charts, limitations and the thermal flag.

## Main findings

- One Tesla T4; YOLO11n, 320px, static batch one; 2 physical / 4 logical CPU cores; 31.35 GiB system RAM.
- 13 runtime/device/precision cases, 10 randomized >=20s timing trials each; 512 calibration and 4488 disjoint
  COCO evaluation images. Accuracy uses official COCOeval. Timing includes preprocess/inference/postprocess
  and completed CPU box transfer, excludes decode/tracking, and filters vehicle classes.
- Median model throughput: PyTorch CUDA FP32 **95.92 FPS**, TensorRT FP32 **236.39**, FP16 enabled **258.37**,
  INT8 enabled **258.05**. FP16/PyTorch paired geometric-mean ratio **2.694×**. Baseline PyTorch is eager,
  with cuDNN autotuning disabled; this is a configuration comparison, not a claim against every tuned baseline.
- The TensorRT-only FP16/FP32 paired ratio is **1.091×**. The large total gain is not caused solely by FP16.
- INT8/FP16 paired throughput ratio **0.999×**, exploratory log-t interval **0.991–1.007×**: no clear INT8
  throughput gain in this session. Global AP loss is **0.7353 points** for INT8 versus **0.0036 points** for FP16.
  INT8 vehicle AP loss is **0.9848 points**, close to the declared one-point budget. AP uncertainty is unmeasured.
- OpenVINO INT8 improved CPU throughput versus its FP32 baseline: **65.15 → 74.33 FPS**, paired ratio **1.136×**,
  with **0.1679 global AP points** lost. ONNX CPU INT8 was slower than FP32 and lost **3.3884 global AP points**.
- The automatic serving selector picked TensorRT INT8 for its marginally lower median p95 (3.96 versus 4.00ms).
  That is not proof of a better deployment tradeoff. FP16 is a good next candidate for a matched serving comparison;
  it was not tested in the full baseline capacity matrix.
- Completed **72 serving runs**: 6 backend/device choices × 1/2/4/8 cameras × three 30s replays. Every camera
  must meet >=15 FPS and <=100ms p95 age in every repeat. Largest timing-passing tested loads: ONNX CPU 1,
  OpenVINO CPU 1, PyTorch CUDA 2, ONNX CUDA 2, TensorRT INT8 CUDA 2; PyTorch CPU had none.
- TensorRT INT8 two-camera results: minimum per-camera FPS median **29.93**, worst-camera p95 age median
  **53.20ms** (range **51.27–54.00ms**). Four-camera freshness failed despite >=15 FPS per camera.
- At eight streams in repeat 0, TensorRT decoder-lag p95 reached **22.54s**, while mean whole-GPU utilization
  was **10.14%**. Bounded mailboxes do not remove lateness accumulated before frames leave the decoder.
  This supports investigating CPU/decode pacing and other host stages before more detector quantization.

## Limits that remain

The camera tests replay copies of one short, looped clip; they are compute-load experiments. Independent crossing
labels are absent, so no camera capacity preserving counting accuracy has been established. Traffic-domain detection
AP, occupancy accuracy, varied scenes/network sources and sustained production reliability are also unvalidated.
The 640 study was intentionally interrupted; the tuning phase has not been run.

One ONNX CUDA FP32 timing trial (round 2, zero-based) observed thermal-throttle bits and reached 85C. It was retained
and flagged. Results are from one cloud host/session; exploratory paired-round confidence intervals assume roughly
independent normal log ratios and do not cover day-to-day variability. Printed AP point differences lack AP CIs.

## Evidence and reproduction

Original archive SHA256:
`6007760be6d0a2324ddf0579321d3e01b62370d0fa5f961feb3433032424a702`

The complete 486942221-byte archive, including raw frame/trial records and model binaries, is preserved locally in
the ignored `data/imported-kaggle/` directory. It is not committed to Git. Committed analysis JSON/CSV/charts are
derived from that archive; see [audit.json](../results/kaggle-t4-320-2026-11-02/audit.json).
The archived code fingerprints match the embedded source payload at Git commit `795d811`.

```powershell
.\.venv-gpu\Scripts\python.exe scripts/analyze_kaggle_checkpoint.py data/imported-kaggle/traffic-kaggle-v1-6007760be6d0.zip --output .build-cache/kaggle-audit
```

Run from a checkout with the historical Git source available. This audit uses only the Python standard library
and pure project helpers; it does not execute archive contents or run ML inference. The checkpoint omits the
large image cache and per-image prediction JSONs. Therefore this archive alone cannot independently recompute
or bootstrap AP; preserve those prediction files from the live Kaggle session if further accuracy analysis is needed.
