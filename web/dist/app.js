import {VEHICLES} from './vision.js';
const $=id=>document.getElementById(id),video=$('video'),overlay=$('overlay'),ctx=overlay.getContext('2d');
const capture=new OffscreenCanvas(320,320),captureCtx=capture.getContext('2d');
let worker=null,ready=false,initializing=false,running=false,busy=false,epoch=0,source='sample',url=null,tracks=[],latencies=[],completionTimes=[],sentAt=0,lastSentMedia=-1,lastResult=0,startedAt=0,elapsed=0;
const limit=()=>source==='sample'?30:Math.min(60,video.duration||60);
const clock=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}`;
function status(s){$('status').textContent=s;}
function clearMetrics(){tracks=[];latencies=[];completionTimes=[];$('fps').innerHTML='—<small> FPS</small>';$('latency').innerHTML='—<small> ms</small>';$('p95').innerHTML='—<small> ms</small>';$('vehicles').textContent='—';$('occupancy').textContent='—';$('up').textContent='0';$('down').textContent='0';lastResult=0;}
function resetTracking(){epoch++;tracks=[];lastSentMedia=-1;worker?.postMessage({type:'reset',epoch});$('up').textContent='0';$('down').textContent='0';$('vehicles').textContent='—';$('occupancy').textContent='—';}
function controls(){ $('run').disabled=initializing;$('run').textContent=initializing?'Loading model…':running?'Ⅱ Pause':elapsed>=limit()?'↺ Run again':'▶ Run demo';$('reset').disabled=initializing;$('provider').disabled=running||initializing;$('upload').disabled=initializing;$('sample').disabled=initializing;$('state').textContent=initializing?'LOADING':running?'RUNNING':elapsed?'PAUSED':'IDLE';$('liveDot').style.background=running?'var(--mint)':'var(--muted)';}
function stop(message){if(running)elapsed+=(performance.now()-startedAt)/1000;running=false;video.pause();controls();if(message)status(message);}
function reset(){stop();elapsed=0;video.currentTime=0;clearMetrics();resetTracking();$('position').textContent=`00:00 / ${clock(limit())}`;$('splash').hidden=false;status('Reset. Ready to run.');controls();}
async function initialize(){
  initializing=true;controls();$('progress').hidden=false;$('progress').value=0;
  worker=new Worker(new URL('./inference-worker.js',import.meta.url),{type:'module'});
  return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>{worker?.terminate();worker=null;initializing=false;ready=false;$('progress').hidden=true;reject(new Error('Model setup timed out. Try CPU mode or another browser.'));},180000);
    worker.onmessage=({data:m})=>{
      if(m.type==='status')status(m.message);
      if(m.type==='progress')$('progress').value=m.percent;
      if(m.type==='ready'){clearTimeout(timer);ready=true;initializing=false;$('backend').textContent=`${m.backend.toUpperCase()} · YOLO11n`;$('progress').hidden=true;controls();resolve();}
      if(m.type==='result'){
        busy=false;if(m.epoch!==epoch)return;const now=performance.now(),latency=now-sentAt;tracks=m.tracks;lastResult=now;latencies.push(latency);if(latencies.length>60)latencies.shift();completionTimes.push(now);completionTimes=completionTimes.filter(t=>now-t<3000);
        const fps=completionTimes.length>1?(completionTimes.length-1)*1000/(now-completionTimes[0]):0;
        $('fps').innerHTML=`${fps?fps.toFixed(1):'…'}<small> FPS</small>`;$('latency').innerHTML=`${latency.toFixed(0)}<small> ms</small>`;const sorted=[...latencies].sort((a,b)=>a-b);$('p95').innerHTML=`${sorted[Math.max(0,Math.ceil(sorted.length*.95)-1)].toFixed(0)}<small> ms</small>`;$('vehicles').textContent=tracks.length;$('occupancy').textContent=m.occupancy;$('up').textContent=m.up;$('down').textContent=m.down;
      }
      if(m.type==='error'){clearTimeout(timer);busy=false;stop(`Inference error: ${m.message}. Try CPU mode, then Run demo.`);ready=false;initializing=false;worker?.terminate();worker=null;$('progress').hidden=true;controls();reject(new Error(m.message));}
    };
    worker.onerror=e=>{clearTimeout(timer);initializing=false;ready=false;busy=false;stop(`Worker error: ${e.message}`);worker?.terminate();worker=null;$('progress').hidden=true;controls();reject(new Error(e.message));};
    worker.postMessage({type:'init',provider:$('provider').value});
  });
}
async function start(){
  if(running){stop('Paused. Press Run demo to continue.');return;}
  if(elapsed>=limit())reset();
  try{if(!ready)await initialize();if(video.readyState<2)await new Promise((resolve,reject)=>{video.addEventListener('loadeddata',resolve,{once:true});video.addEventListener('error',()=>reject(new Error('Video could not be decoded. Try an MP4/H.264 clip.')),{once:true});video.load();});await video.play();completionTimes=[];running=true;startedAt=performance.now();$('splash').hidden=true;controls();status(source==='sample'?'Live inference on a continuous 30-second E18 traffic recording.':'Live inference. Your video is processed locally; the session stops after 60 seconds.');}
  catch(e){initializing=false;stop(e.message);controls();}
}
$('run').onclick=start;$('reset').onclick=reset;
$('provider').onchange=()=>{worker?.terminate();worker=null;ready=false;busy=false;reset();$('backend').textContent='READY TO INITIALIZE';};
$('confidence').oninput=()=>{$('confValue').value=Number($('confidence').value).toFixed(2);resetTracking();};
$('line').oninput=()=>{$('lineValue').value=`${$('line').value}%`;resetTracking();};
function changeSource(src,name,isSample){reset();if(url){URL.revokeObjectURL(url);url=null;}source=isSample?'sample':'upload';if(!isSample)url=src;video.src=src;video.load();$('feedLabel').textContent=isSample?'E18 / LYSAKER, NORWAY':`LOCAL VIDEO / ${name}`;$('sourceNote').textContent=isSample?'30 continuous seconds · E18, Lysaker, Norway · Kjetil_r / public domain.':'Your video stays on this device. The first 60 seconds are processed.';status('Video selected. Press Run demo.');$('position').textContent=`00:00 / ${clock(isSample?30:60)}`;}
$('upload').onchange=()=>{const f=$('upload').files[0];if(!f)return;if(f.size>100*1024*1024){status('Choose a video smaller than 100 MB.');$('upload').value='';return;}changeSource(URL.createObjectURL(f),f.name,false);};
$('sample').onclick=()=>{changeSource('assets/traffic.mp4','sample',true);$('upload').value='';};
video.addEventListener('loadedmetadata',()=>{$('position').textContent=`${clock(elapsed)} / ${clock(limit())}`;});
video.addEventListener('error',()=>stop('This browser could not decode the video. Try an MP4/H.264 clip.'));
video.addEventListener('ended',()=>{if(running){stop('Video complete. Reset to replay.');elapsed=limit();controls();}});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)stop('Paused while the tab is hidden. Press Run demo to continue.');});
function draw(){
  const w=video.videoWidth||960,h=video.videoHeight||540;if(overlay.width!==w||overlay.height!==h){overlay.width=w;overlay.height=h;}
  ctx.clearRect(0,0,w,h);const unit=Math.max(1,w/960,h/540);ctx.lineWidth=2*unit;
  ctx.fillStyle='#b6f0cc0a';ctx.fillRect(.15*w,.35*h,.7*w,.55*h);ctx.strokeStyle='#b6f0cc55';ctx.setLineDash([6*unit,7*unit]);ctx.strokeRect(.15*w,.35*h,.7*w,.55*h);
  const line=Number($('line').value)/100;ctx.strokeStyle='#ffb978';ctx.setLineDash([10*unit,7*unit]);ctx.beginPath();ctx.moveTo(w*.1,h*line);ctx.lineTo(w*.9,h*line);ctx.stroke();ctx.setLineDash([]);ctx.font=`${12*unit}px monospace`;ctx.fillStyle='#ffb978';ctx.fillText('COUNTING LINE',w*.1,h*line-9*unit);
  if(performance.now()-lastResult<1200)for(const b of tracks){ctx.strokeStyle='#b6f0cc';ctx.strokeRect(b.x*w,b.y*h,b.w*w,b.h*h);const label=`#${b.id} ${VEHICLES.get(b.cls)} ${Math.round(b.score*100)}%`,tw=ctx.measureText(label).width;ctx.fillStyle='#b6f0cc';ctx.fillRect(b.x*w,Math.max(0,b.y*h-20*unit),tw+10*unit,20*unit);ctx.fillStyle='#0b1217';ctx.fillText(label,b.x*w+5*unit,Math.max(14*unit,b.y*h-6*unit));}
}
async function tick(){
  if(running){const duration=elapsed+(performance.now()-startedAt)/1000;$('position').textContent=`${clock(Math.min(duration,limit()))} / ${clock(limit())}`;
    if(duration>=limit()){stop('Session complete. Reset to replay or choose your own clip.');elapsed=limit();controls();}
    else if(!busy&&ready&&video.readyState>=2&&video.currentTime>.02&&Math.abs(video.currentTime-lastSentMedia)>=1/15){busy=true;sentAt=performance.now();const time=video.currentTime,e=epoch;lastSentMedia=time;try{const ratio=Math.min(1,640/Math.max(video.videoWidth,video.videoHeight)),cw=Math.round(video.videoWidth*ratio),ch=Math.round(video.videoHeight*ratio);if(capture.width!==cw||capture.height!==ch){capture.width=cw;capture.height=ch;}captureCtx.drawImage(video,0,0,cw,ch);const bitmap=capture.transferToImageBitmap();if(!running||epoch!==e){bitmap.close();busy=false;}else worker.postMessage({type:'frame',bitmap,width:video.videoWidth,height:video.videoHeight,time,line:Number($('line').value)/100,confidence:Number($('confidence').value),epoch:e},[bitmap]);}catch(err){busy=false;stop(`Frame capture failed: ${err.message}`);}}
  }
  draw();requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
