import fs from 'node:fs';
const current=fs.readFileSync('dist/index.html','utf8');
const start=current.indexOf('<section class="workspace"');
const cap=current.indexOf('<p class="caption">',start),end=current.indexOf('</p>',cap)+4;
if(start<0||cap<0||end<4)throw new Error('Existing live demo markup was not found');
const demo=current.slice(start,end);
const sections=fs.readFileSync('case-study-sections.html','utf8');
fs.writeFileSync('dist/index.html',`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="A measured YOLO11n optimization case study: PyTorch, ONNX Runtime, OpenVINO and TensorRT; precision tradeoffs, multi-camera capacity and pipeline profiling. Includes live browser inference."><title>Traffic Inference Study</title><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="study.css"></head>
<body><header><a class="brand" href="#overview"><span class="brand-icon">PS</span> TRAFFIC INFERENCE STUDY<span class="tag">PUNEET PAL SINGH</span></a><nav class="study-nav" aria-label="Study navigation"><a href="#experiments">Runtime</a><a href="#precision">Precision</a><a href="#capacity">Camera load</a><a href="#bottleneck">Profiling</a><a href="#method">Evidence</a><a class="nav-demo" href="#live-demo">Video demo ↗</a></nav></header>
<main>${sections}${demo}<footer><span>TRAFFIC INFERENCE STUDY</span><span>Runtime · precision · pipeline</span><a href="about.html">Browser method & assets ↗</a></footer></main><script type="module" src="study.js"></script><script type="module" src="app.js"></script></body></html>\n`);
console.log('Built optimization case study; retained live detector markup.');
