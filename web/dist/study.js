import {linear,logarithmic,domains} from './chart-math.js';
const $=id=>document.getElementById(id),n=(v,d=2)=>Number(v).toFixed(d),names={pytorch:'PyTorch',onnx:'ONNX Runtime',openvino:'OpenVINO',tensorrt:'TensorRT'},colors={pytorch:'#718087',onnx:'#416882',openvino:'#2c7a72',tensorrt:'#cf4b2c'},precColors={fp32:'#718087',fp16:'#cf4b2c',int8:'#74618c'};
const label=id=>{const [r,d,p]=id.split('-');return `${names[r]} ${p.toUpperCase()} / ${d==='cpu'?'CPU':'GPU'}`;};
const short=id=>label(id).replace(' / GPU','').replace(' / CPU','');
const safe=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
const apLoss=v=>Math.abs(v)<.00005?'0.0000':n(v,4),rng=(v,d=2)=>`${n(v[0],d)}–${n(v[1],d)}`;
let models,serving,profiles,reference,comparisons,device='CUDA',selected='tensorrt-cuda-fp16',metric='fps_median',servingCase='tensorrt-cuda-int8',streamCount=2;
const vehicleLoss=r=>(reference.vehicle_ap-r.vehicle_ap)*100;
const svg=(w,h,title,content)=>`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${safe(title)}"><title>${safe(title)}</title>${content}</svg>`;
const text=(x,y,s,cls='',anchor='start')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${safe(s)}</text>`;
function barChart(rows,{max,ticks,min=0,width=900,step=48,left=190,right=100,unit='FPS',whiskers=false,interactive=false,referenceValues=false,title=''}){
  const top=24,height=top+rows.length*step+48,edge=width-right,x=v=>linear(v,min,max,left,edge),zero=x(0),axisY=height-39;
  let content=ticks.map(t=>`<line x1="${x(t)}" y1="12" x2="${x(t)}" y2="${axisY}" class="grid"/>${text(x(t),height-17,t,'axis','middle')}`).join('');
  content+=text(edge,height-2,unit,'axis-unit','end');
  rows.forEach((r,i)=>{const y=top+i*step+step/2,barX=Math.min(zero,x(r.value||0)),barW=Math.abs(x(r.value||0)-zero),color=r.color||'#416882';
    content+=`<g ${interactive?`class="chart-select" data-case="${r.id}" role="button" tabindex="0" aria-pressed="${r.id===selected}" aria-label="${safe(r.name)}, ${r.missing?'not measured':n(r.value)+' '+unit}"><rect x="0" y="${y-step/2+2}" width="${width}" height="${step-4}" fill="${r.id===selected?'#eeebe1':'transparent'}"/>`:'>'}`;
    content+=text(left-14,y+4,r.name,'plot-label','end');
    if(r.missing){content+=text(left+10,y+4,'Not measured','axis');}
    else{
      content+=`<rect x="${barX}" y="${y-8}" width="${barW}" height="16" fill="${color}"/>`;
      if(whiskers&&r.range){const [lo,hi]=r.range;content+=`<path d="M${x(lo)},${y}H${x(hi)}M${x(lo)},${y-5}V${y+5}M${x(hi)},${y-5}V${y+5}" class="whisker"/>`;}
      if(Number.isFinite(r.reference)){const a=x(r.reference);content+=`<path d="M${a},${y-7}l7,7l-7,7l-7,-7Z" class="reference-marker"/>`;}
      content+=text(width-10,y+4,referenceValues?`${n(r.value)} / ${n(r.reference)}`:n(r.value),'plot-value','end');
    }content+='</g>';
  });return svg(width,height,title,content);
}
function wireSelections(){document.querySelectorAll('#runtimeBars [data-case]').forEach(el=>{const choose=()=>{selected=el.dataset.case;renderModels();};el.onclick=choose;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}};});}
function renderModels(){
  const rows=models.filter(r=>r.device===device).sort((a,b)=>metric==='fps_median'?b[metric]-a[metric]:a[metric]-b[metric]);if(!rows.some(r=>r.case===selected))selected=rows[0].case;
  const base=models.find(r=>r.case===`pytorch-${device==='CUDA'?'cuda':'cpu'}-fp32`),domain=domains[metric][device],units={fps_median:'FPS',trial_p95_median_ms:'ms',artifact_mib:'MiB'},titles={fps_median:'Median throughput',trial_p95_median_ms:'Median trial p95 latency',artifact_mib:'Serialized artifact size'};
  $('runtimeReference').innerHTML=`Reference on this device: <b>${label(base.case)} · ${n(base[metric])} ${units[metric]}</b>. Accuracy reference: PyTorch CUDA FP32 · COCO mAP <b>${n(reference.map50_95,6)}</b> / vehicle AP <b>${n(reference.vehicle_ap,6)}</b>.`;
  $('chartMetricLabel').textContent=`${titles[metric]} / ${device==='CUDA'?'Tesla T4':'CPU'}`;
  $('runtimeBars').innerHTML=barChart(rows.map(r=>({id:r.case,name:short(r.case)+(r.flagged_trials?' †':''),value:r[metric],color:colors[r.case.split('-')[0]],range:[r.fps_min,r.fps_max]})),{...domain,unit:units[metric],whiskers:metric==='fps_median',interactive:true,title:`${titles[metric]}, zero-based linear scale. ${rows.map(r=>`${short(r.case)} ${n(r[metric])} ${units[metric]}`).join('; ')}`});wireSelections();
  $('runtimeScaleNote').textContent=`Linear axis, 0–${domain.max} ${units[metric]}. ${metric==='fps_median'?'Bars: median throughput of ten trials. Whiskers: observed min–max, not a confidence interval.':metric==='artifact_mib'?'Weights / graph / engine file size; runtime memory use is a separate measurement.':'Median of ten trial p95 values; this chart has no error bars because trial p95 ranges are not in the saved summary.'}`;
  const r=models.find(r=>r.case===selected);
  $('configurationDetail').innerHTML=`<div class="reading-title"><span>Selected measurement</span><h3>${label(r.case)}</h3></div><dl><div><dt>Median throughput</dt><dd>${n(r.fps_median)} FPS</dd></div><div><dt>Ten-trial min–max</dt><dd>${n(r.fps_min)}–${n(r.fps_max)} FPS</dd></div><div><dt>Median trial p95</dt><dd>${n(r.trial_p95_median_ms)} ms</dd></div><div><dt>COCO / vehicle AP</dt><dd>${n(r.map50_95,6)} / ${n(r.vehicle_ap,6)}</dd></div><div><dt>Global / vehicle loss</dt><dd>${apLoss(r.loss_vs_pytorch_points)} / ${apLoss(vehicleLoss(r))} pt</dd></div><div><dt>Artifact / peak sampled RSS</dt><dd>${n(r.artifact_mib)} / ${n(r.peak_sampled_rss_mib,0)} MiB</dd></div></dl><p class="figure-caption">Throughput reference: ${label(base.case)}, <b>${n(base.fps_median)} FPS</b>. Selected / reference ratio of medians: <b>${n(r.fps_median/base.fps_median,3)}×</b>; this is distinct from paired-round inference.${r.flagged_trials?' † One thermal diagnostic flag was retained.':''}</p>`;
  $('deviceTakeaway').textContent=device==='CUDA'?'Interpretation: TensorRT FP32 reached 236.39 FPS versus the 95.92 FPS eager PyTorch FP32 reference. TensorRT FP16 reached 258.37 FPS. Most of this measured gain is associated with the runtime/configuration change, not solely reduced precision.':'Interpretation: OpenVINO FP32 reached 65.15 FPS versus the 32.40 FPS PyTorch CPU FP32 reference. OpenVINO INT8 reached 74.33 FPS. ONNX CPU INT8 reached 19.86 FPS versus its own 26.68 FPS FP32 result; quantization did not improve that recipe.';
}
function renderPrecision(){
  const key=$('precisionRuntime').value,fp32=models.find(r=>r.case===key+'-fp32'),d=key.endsWith('cpu')?'CPU':'CUDA',rows=['fp32','fp16','int8'].map(p=>models.find(r=>r.case===key+'-'+p)),domain=domains.fps_median[d];
  $('precisionReference').innerHTML=`Speed reference: <b>${label(fp32.case)} · ${n(fp32.fps_median)} FPS</b>. Accuracy-loss reference for every variant: <b>PyTorch CUDA FP32 · COCO ${n(reference.map50_95,6)} / vehicle ${n(reference.vehicle_ap,6)} AP</b>.`;
  $('precisionSpeed').innerHTML=barChart(rows.map((r,i)=>({name:['FP32','FP16','INT8'][i],missing:!r,value:r?.fps_median,range:r?[r.fps_min,r.fps_max]:null,color:precColors[['fp32','fp16','int8'][i]]})),{...domain,width:550,left:65,right:70,step:67,whiskers:true,title:`${key} throughput. FP32 reference ${n(fp32.fps_median)} FPS.`});
  const W=550,H=265,L=65,R=135,edge=W-R,x=v=>linear(v,-.1,6,L,edge);let out=[0,1,2,3,4,5,6].map(t=>`<line x1="${x(t)}" y1="20" x2="${x(t)}" y2="224" class="grid"/>${text(x(t),245,t,'axis','middle')}`).join('')+`<line x1="${x(0)}" y1="20" x2="${x(0)}" y2="224" class="zero-line"/>`;
  rows.forEach((r,i)=>{const y=55+i*67;out+=text(L-12,y+4,['FP32','FP16','INT8'][i],'plot-label','end');if(!r){out+=text(L+15,y+4,'Not measured','axis');return;}const values=[r.loss_vs_pytorch_points,vehicleLoss(r)];values.forEach((v,j)=>{const py=y-10+j*18;out+=`<rect x="${Math.min(x(0),x(v))}" y="${py-5}" width="${Math.abs(x(v)-x(0))}" height="10" fill="${j?'#416882':'#cf4b2c'}"/>`;});out+=text(W-8,y-3,apLoss(values[0]),'plot-value ap-global','end')+text(W-8,y+15,apLoss(values[1]),'plot-value ap-vehicle','end');});
  $('precisionLoss').innerHTML=svg(W,H,'Global and vehicle AP loss, fixed -0.1 to 6.0 percentage-point scale; exact values printed at right.',out);
  $('precisionCards').innerHTML=`<table><caption>Exact measurements; AP is the original 0–1 score.</caption><thead><tr><th>Precision</th><th>Median FPS</th><th>COCO mAP</th><th>Vehicle AP</th><th>Artifact MiB</th></tr></thead><tbody>${rows.map((r,i)=>r?`<tr><th scope="row">${['FP32','FP16','INT8'][i]}</th><td>${n(r.fps_median)}</td><td>${n(r.map50_95,6)}</td><td>${n(r.vehicle_ap,6)}</td><td>${n(r.artifact_mib)}</td></tr>`:`<tr><th>INT8</th><td colspan="4">Not measured on this runtime/device</td></tr>`).join('')}</tbody></table>`;
  const conclusions={
    'tensorrt-cuda':'TensorRT INT8: 258.05 FPS; FP16 reference: 258.37 FPS. The paired INT8 / FP16 ratio was 0.999× (exploratory 95% interval 0.991–1.007×), so there was no clear throughput gain. INT8 used 6.91 MiB versus 8.29 MiB for FP16, with 0.7353 versus 0.0036 global AP points lost against the stated PyTorch accuracy reference.',
    'openvino-cpu':'OpenVINO INT8: 74.33 FPS; its FP32 reference: 65.15 FPS. Paired-round improvement: 13.6% (exploratory interval 11.7–15.6%). The global AP point loss against PyTorch CUDA FP32 mAP 0.284270 was 0.1679. FP16 here means compressed stored weights, not necessarily FP16 CPU execution.',
    'onnx-cpu':'ONNX CPU INT8: 19.86 FPS; FP32 reference: 26.68 FPS. INT8 reduced artifact size from 10.12 to 2.84 MiB, but global mAP fell from 0.284270 to 0.250386. The smaller artifact was neither faster nor equally accurate in this configuration.',
    'onnx-cuda':'ONNX CUDA FP16: 175.25 FPS; FP32 reference: 194.34 FPS. FP16 was slower in this recipe. One FP32 trial had a retained thermal diagnostic flag. INT8 was not measured on ONNX CUDA.'
  };$('precisionConclusion').textContent=conclusions[key];
}
function servingPlot(key,log){
  const W=560,H=340,L=65,R=24,T=28,B=55,ex=W-R,ey=H-B,x=v=>linear(v,1,8,L,ex),y=v=>log?logarithmic(v,10,100000,ey,T):linear(v,0,35,ey,T),ticks=log?[10,100,1000,10000,100000]:[0,5,10,15,20,25,30,35];
  let out=ticks.map(t=>`<path d="M${L},${y(t)}H${ex}" class="grid"/>${text(L-10,y(t)+4,log&&t>=1000?(t/1000)+'k':t,'axis','end')}`).join('');
  [1,2,4,8].forEach(c=>{out+=`<path d="M${x(c)},${T}V${ey}" class="grid vertical-grid"/>${text(x(c),ey+24,c,'axis','middle')}`;});out+=text((L+ex)/2,H-7,'Camera streams · linear scale','axis-unit','middle');
  const cases=[...new Set(serving.map(r=>r.case))].sort((a,b)=>(a===servingCase?1:0)-(b===servingCase?1:0));
  for(const c of cases){const rr=serving.filter(r=>r.case===c).sort((a,b)=>a.streams-b.streams),is=c===servingCase,color=colors[c.split('-')[0]],dash=c.includes('-cpu-')?'6 4':'none';out+=`<path d="${rr.map((r,i)=>`${i?'L':'M'}${x(r.streams)},${y(r[key])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${is?3:1.4}" stroke-dasharray="${dash}" opacity="${is?1:.35}"/>`;
    rr.forEach(r=>{const px=x(r.streams),py=y(r[key]);if(is){const values=r[key==='min_camera_fps_median'?'min_camera_fps_range':'worst_p95_ms_range'];out+=`<path d="M${px},${y(values[0])}V${y(values[1])}M${px-4},${y(values[0])}H${px+4}M${px-4},${y(values[1])}H${px+4}" stroke="${color}" fill="none"/>`;}
      out+=`<circle cx="${px}" cy="${py}" r="${is&&r.streams===streamCount?6:is?4:2}" fill="${is?color:'#f7f5ef'}" stroke="${color}" opacity="${is?1:.6}"><title>${safe(label(c))}, ${r.streams} cameras: ${n(r[key])} ${log?'ms':'FPS'}</title></circle>`;});
  }return svg(W,H,`${log?'Worst-camera p95 frame age, logarithmic 10 to 100000 milliseconds':'Slowest-camera FPS, linear 0 to 35'}. Selected ${label(servingCase)}.`,out);
}
function renderServing(){
  document.querySelectorAll('[data-streams]').forEach(b=>b.setAttribute('aria-pressed',Number(b.dataset.streams)===streamCount));
  $('servingFps').innerHTML=servingPlot('min_camera_fps_median',false);$('servingAge').innerHTML=servingPlot('worst_p95_ms_median',true);
  const cases=[...new Set(serving.map(r=>r.case))];$('servingLegend').innerHTML=cases.map(c=>`<span class="${c===servingCase?'legend-selected':''}"><i style="border-color:${colors[c.split('-')[0]]};border-top-style:${c.includes('-cpu-')?'dashed':'solid'}"></i>${label(c)}</span>`).join('');
  const r=serving.find(r=>r.case===servingCase&&r.streams===streamCount),base=serving.find(r=>r.case===servingCase&&r.streams===1);
  $('capacityDetail').innerHTML=`<div class="reading-title"><span>${streamCount} cameras / 3 repeats</span><h3>${label(servingCase)}</h3></div><dl><div><dt>Slowest-camera FPS · median</dt><dd>${n(r.min_camera_fps_median)} FPS</dd><small>Repeat range ${rng(r.min_camera_fps_range)} FPS</small></div><div><dt>Worst-camera p95 age · median</dt><dd>${n(r.worst_p95_ms_median)} ms</dd><small>Repeat range ${rng(r.worst_p95_ms_range)} ms</small></div></dl><p class="figure-caption">One-camera reference on the same backend: <b>${n(base.min_camera_fps_median)} FPS</b> and <b>${n(base.worst_p95_ms_median)} ms</b>. Counting accuracy remains unvalidated. No deployment pass/fail is inferred.</p>`;
}
function renderProfile(streams=8){
  document.querySelectorAll('[data-profile]').forEach(b=>b.setAttribute('aria-pressed',Number(b.dataset.profile)===streams));
  const r=profiles.find(r=>r.case==='tensorrt-cuda-int8'&&r.streams===streams&&r.repeat===0),base=profiles.find(r=>r.case==='tensorrt-cuda-int8'&&r.streams===1&&r.repeat===0),stages=[['decode_ms','Decode / read'],['queue_wait_ms','Queue wait'],['batch_predict_ms','Preprocess + predict'],['tracking_count_ms','Track + count'],['telemetry_ms','Telemetry']];
  $('profileBars').innerHTML=barChart(stages.map(([key,name])=>({name,value:r.stages[key].p95,reference:base.stages[key].p95,color:'#416882'})),{max:120,ticks:[0,20,40,60,80,100,120],width:840,left:170,right:150,unit:'ms · selected / 1 cam',step:50,referenceValues:true,title:`Stage p95, 0–120 ms fixed scale. Selected ${streams} cameras versus one camera. Values at right are selected / one-camera reference.`});
  const rows=[['Decoder lag p95',r.stages.decoder_lag_ms.p95,base.stages.decoder_lag_ms.p95,'ms'],['Mean GPU utilization',r.gpu_utilization.mean,base.gpu_utilization.mean,'%'],['Aggregate throughput',r.aggregate_fps,base.aggregate_fps,'FPS'],['Frame-age p95',r.stages.frame_age_ms.p95,base.stages.frame_age_ms.p95,'ms']];
  $('profileKpis').innerHTML=`<table><caption>Same backend / first repeat</caption><thead><tr><th>Measurement</th><th>${streams} cam${streams>1?'s':''}</th><th>1 cam ref.</th></tr></thead><tbody>${rows.map(([name,a,b,u])=>`<tr><th>${name}</th><td>${n(a)} ${u}</td><td>${n(b)} ${u}</td></tr>`).join('')}</tbody></table>`;
}
async function init(){
  const files=['precision-320-summary.json','serving-summary.json','stage-profiles.json','comparisons.json'];const data=await Promise.all(files.map(async f=>{const r=await fetch('evidence/'+f);if(!r.ok)throw new Error('Cannot load '+f);return r.json();}));[models,serving,profiles,comparisons]=[data[0].rows,data[1].rows,data[2],data[3]];reference=models.find(r=>r.case==='pytorch-cuda-fp32');
  const hero=['pytorch-cuda-fp32','tensorrt-cuda-fp32','tensorrt-cuda-fp16'].map(id=>models.find(r=>r.case===id));$('heroChart').innerHTML=barChart(hero.map(r=>({name:short(r.case),value:r.fps_median,range:[r.fps_min,r.fps_max],color:colors[r.case.split('-')[0]]})),{max:300,ticks:[0,100,200,300],width:640,left:165,right:75,step:62,whiskers:true,title:'PyTorch FP32 reference 95.92 FPS; TensorRT FP32 236.39 FPS; TensorRT FP16 258.37 FPS. Zero-based 0–300 FPS scale.'});
  document.querySelectorAll('[data-device]').forEach(b=>b.onclick=()=>{device=b.dataset.device;document.querySelectorAll('[data-device]').forEach(t=>t.setAttribute('aria-pressed',t===b));renderModels();});$('benchmarkMetric').onchange=()=>{metric=$('benchmarkMetric').value;renderModels();};$('precisionRuntime').onchange=renderPrecision;
  $('servingRuntime').innerHTML=[...new Set(serving.map(r=>r.case))].map(c=>`<option value="${c}" ${c===servingCase?'selected':''}>${label(c)}</option>`).join('');$('servingRuntime').onchange=()=>{servingCase=$('servingRuntime').value;renderServing();};document.querySelectorAll('[data-streams]').forEach(b=>b.onclick=()=>{streamCount=Number(b.dataset.streams);renderServing();});document.querySelectorAll('[data-profile]').forEach(b=>b.onclick=()=>renderProfile(Number(b.dataset.profile)));
  $('allModelRows').innerHTML=models.map(r=>`<tr><th>${label(r.case)}${r.flagged_trials?' †':''}</th><td>${n(r.fps_median)} [${n(r.fps_min)}, ${n(r.fps_max)}]</td><td>${n(r.trial_p95_median_ms)}</td><td>${n(r.map50_95,6)}</td><td>${apLoss(r.loss_vs_pytorch_points)}</td><td>${apLoss(vehicleLoss(r))}</td><td>${n(r.artifact_mib)}</td></tr>`).join('');
  renderModels();renderPrecision();renderServing();renderProfile();
}
init().catch(e=>{console.error(e);$('runtimeBars').textContent='Results could not load. The original report is available under Method & evidence.';});
