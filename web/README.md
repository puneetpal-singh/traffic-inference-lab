# Traffic Inference Study — website and live demo

**Live website:** https://traffic-inference-lab.puneetsekhon07.chatgpt.site

This is the source of the published technical report, interactive recorded-result charts, and live browser inference demo. The report covers PyTorch, ONNX Runtime, OpenVINO and TensorRT, precision tradeoffs, camera-load curves and pipeline profiles. Historical study targets are not presented as application-validated deployment requirements. Each comparison states the measured reference values.

The demo runs YOLO11n ONNX FP32 on the visitor's device using ONNX Runtime Web 1.22.0, WebGPU where initialization succeeds, and CPU/WASM fallback. It uses a 30-second continuous public-domain E18 recording by Kjetil_r. Uploads stay local. Counting uses simple IoU/centroid tracking and remains unvalidated; the Python study uses ByteTrack separately.

## Run

Node 20 or newer; no npm install required.

```sh
cd web
node prepare-assets.mjs
node server.mjs
# Open http://127.0.0.1:8791
```

Five binary assets (model, sample video, poster and WASM runtimes) are downloaded from the public deployment, checked against the sizes and SHA-256 hashes in `asset-manifest.json`, and ignored by Git. Static assets total about 47 MiB; only the needed runtime is fetched during browser initialization. The official Ultralytics model source and public-domain video attribution are in `dist/about.html` and `dist/assets/traffic-source*.json`. ONNX Runtime's MIT license is in `dist/runtime/LICENSE.txt`; Ultralytics licensing is included in `dist/assets/ULTRALYTICS-LICENSE.txt`.

For a single asset, for example to test public download access:

```sh
node prepare-assets.mjs dist/assets/poster.jpg
```

## Edit and check

```sh
node build-case-study.mjs
npm run check
npm test
```

Edit the report in `case-study-sections.html`; its controls/charts are `dist/study.js` and `dist/chart-math.js`, with styling in `dist/study.css`. `dist/app.js`, `inference-worker.js` and `vision.js` implement browser detection/tracking. `dist/evidence/` contains unchanged curated study aggregates. Serve `dist/` over HTTPS to deploy; preserve JavaScript and WASM MIME types. Code is included under the AGPL-3.0 license in this directory; third-party assets retain their own licenses.
