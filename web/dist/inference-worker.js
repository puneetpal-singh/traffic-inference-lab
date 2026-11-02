import * as ort from './runtime/ort.webgpu.bundle.min.mjs';
import {decode,Tracker} from './vision.js';
ort.env.wasm.wasmPaths=new URL('./runtime/',import.meta.url).href;
ort.env.wasm.numThreads=1;
const canvas=new OffscreenCanvas(320,320),ctx=canvas.getContext('2d',{willReadFrequently:true});
let session,backend,tracker=new Tracker(),activeEpoch=null;
function log(message){postMessage({type:'status',message});}
async function initialize(provider){
  log('Downloading YOLO11n ONNX model…');const response=await fetch(new URL('./assets/yolo11n-320-fp32.onnx',import.meta.url));
  if(!response.ok)throw new Error(`Model download failed: HTTP ${response.status}`);
  const reader=response.body.getReader(),chunks=[];let received=0;const total=Number(response.headers.get('content-length'))||10615285;
  while(true){const {value,done}=await reader.read();if(done)break;chunks.push(value);received+=value.length;postMessage({type:'progress',percent:Math.min(100,received/total*100)});}
  const model=new Uint8Array(received);let offset=0;for(const chunk of chunks){model.set(chunk,offset);offset+=chunk.length;}
  async function create(ep){log(`Initializing ${ep==='webgpu'?'browser GPU':'CPU / WebAssembly'} and warming model…`);const s=await ort.InferenceSession.create(model,{executionProviders:[ep],graphOptimizationLevel:'all'});try{const z=new ort.Tensor('float32',new Float32Array(3*320*320),[1,3,320,320]);const out=await s.run({[s.inputNames[0]]:z});z.dispose();const v=out[s.outputNames[0]];if(v.dims[1]!==84)throw new Error(`Incompatible model output ${v.dims}`);for(const tensor of Object.values(out))tensor.dispose();return s;}catch(e){await s.release();throw e;}}
  if(provider==='auto'&&navigator.gpu){try{session=await create('webgpu');backend='WebGPU';}catch(e){log(`GPU unavailable (${e.message}); trying CPU…`);}}
  if(!session){session=await create('wasm');backend='CPU / WASM';}
  postMessage({type:'ready',backend});
}
self.onmessage=async({data:m})=>{
  try{
    if(m.type==='init')await initialize(m.provider);
    else if(m.type==='reset'){tracker.reset();activeEpoch=m.epoch;}
    else if(m.type==='frame'){
      const t0=performance.now(),{bitmap,width,height}=m,scale=Math.min(320/width,320/height),nw=Math.round(width*scale),nh=Math.round(height*scale),left=Math.round((320-nw)/2-.1),top=Math.round((320-nh)/2-.1);
      if(activeEpoch===null)activeEpoch=m.epoch;
      ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,320,320);ctx.drawImage(bitmap,left,top,nw,nh);bitmap.close();
      const rgba=ctx.getImageData(0,0,320,320).data,input=new Float32Array(3*320*320),plane=320*320;
      for(let i=0;i<plane;i++){input[i]=rgba[i*4]/255;input[plane+i]=rgba[i*4+1]/255;input[plane*2+i]=rgba[i*4+2]/255;}
      const tensor=new ort.Tensor('float32',input,[1,3,320,320]),a=performance.now();let out;
      try{out=await session.run({[session.inputNames[0]]:tensor});if(m.epoch!==activeEpoch){postMessage({type:'result',epoch:m.epoch});return;}const inferenceMs=performance.now()-a,v=out[session.outputNames[0]],boxes=decode(v.data,v.dims,{width,height,scale,left,top},m.confidence),result=tracker.update(boxes,m.time,m.line);postMessage({type:'result',...result,inferenceMs,workerMs:performance.now()-t0,time:m.time,epoch:m.epoch});}
      finally{tensor.dispose();if(out)for(const v of Object.values(out))v.dispose();}
    }
  }catch(e){postMessage({type:'error',message:e.message||String(e),stack:e.stack});}
};
