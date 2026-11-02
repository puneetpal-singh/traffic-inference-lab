# Browser smoke verification — 2 November 2026

## Editorial revision (current presentation)

Earlier sections below describe superseded presentations. Raw evidence is unchanged.

- Removed capacity/quality pass-fail labels based on the arbitrary historical 100 ms / 15 FPS and 1.0 AP-point targets. Their provenance and lack of application validation are stated explicitly.
- Comparison references now include numeric values for per-device PyTorch FP32, runtime-specific FP32, PyTorch CUDA FP32 accuracy, and one-camera serving/profile references.
- Seven SVG figures use visible axes, units, fixed domains and min/max whiskers where available. Frame age uses an explicitly labeled logarithmic axis. Performance bars start at zero; stage quantiles are not stacked.
- Five tests passed: linear/log scale behavior, fixed-domain containment of every plotted observation/interval, and three vision tests. JavaScript syntax checks passed. Phone-width checking found and fixed a grid min-content overflow; the document width now matches the viewport, with large figures scrolling inside their own containers. The temporary viewport override was reset.
- Browser inspection confirmed the new editorial layout, hero axes/whiskers, camera-load curves and ONNX CPU precision selection with the 26.68 FPS FP32 reference and COCO/vehicle AP references of 0.284270 / 0.332202.

## Earlier optimization case study update

- The primary page now exposes all 13 recorded model cases, CPU/GPU filtering, selectable throughput/p95/artifact-size sorting, per-configuration accuracy and RSS details, precision comparisons, all 24 serving cells, and four camera-count stage profiles.
- Tested CPU filtering and artifact-size sorting in the browser. Tested ONNX CPU precision selection: INT8 displayed 19.86 FPS, 3.3884 global and 5.0086 vehicle AP points lost, and an exceeded-budget indication.
- Tested four-camera TensorRT INT8 selection: 17.02 slowest-camera FPS, 283.64 ms median worst-camera p95, repeat range 159.37–344.20 ms, timing failure shown correctly.
- Tested two-camera TensorRT INT8 profile selection: 59.88 aggregate FPS, 16.6% mean GPU utilization, and 53.06 ms frame-age p95. Profile quantiles are not summed.
- Six downloadable evidence files were hash-compared against the curated verified source and match byte for byte. No new inference benchmark was run for this presentation update.
- Browser desktop visual inspection confirmed the new hero, runtime explorer, capacity matrix and profile charts render. The existing detector markup and inference code were retained; the three vision tests still pass.

## Earlier live-detector verification

Local static site served at localhost using Node 24.13.0. Actual interactive tests ran in the Codex in-app Chromium browser on the user's laptop. These are functionality checks, not a controlled performance benchmark.

- Automatic mode initialized WebGPU successfully, completed real YOLO11n inference and displayed changing vehicle detections and telemetry.
- Explicit CPU/WebAssembly mode initialized successfully and completed real inference.
- A local MP4 selected through the browser file chooser was processed to its end; the test displayed a downward crossing and stopped at the short video's duration.
- The initial short sample was replaced with the first 30 continuous seconds of public-domain E18-Lysaker (02) footage by Kjetil_r. The final version was tested in WebGPU mode through 00:30 / 00:30; it stopped, displayed completion, and did not loop.
- Updated `Traffic_Browser_Demo_Preview.png` shows actual boxes, IDs, WebGPU selection, live telemetry and estimated crossings on the new E18 sample.
- Frame capture uses a main-thread OffscreenCanvas followed by transferable ImageBitmap rather than directly creating an ImageBitmap from the video; this fixed an observed browser capture failure. Longest capture dimension is capped at 640px before worker-side letterboxing to 320px.
- Annotation scale accounts for portrait footage. Layering is explicit so overlays remain visible above hardware-composited video.
- Three unit tests pass: letterbox/NMS/class filtering, crossing tracking/expiry/reset, line-change reset. JavaScript syntax checks pass.
- FFprobe confirms the delivered E18 clip is exactly 30 seconds, 900 frames, 960×540, 5,304,740 bytes. Its original SHA-1 matches Wikimedia's published `2c4ee6a07cba4b51fdf9155b29a452deebc658c9`. Source attribution, public-domain status, changes and SHA-256 hashes are bundled in the site.

Observed live processing was around 6 FPS in this browser in both modes. This is uncontrolled, capped playback processing, and is not a benchmark claim or a promise for other devices. WebGPU initialization does not establish that every operation executes on GPU. No FP16/FP8 browser support, mobile throughput, browser counting accuracy, or public portfolio integration is claimed by these checks.
