export const VEHICLES = new Map([[2,'car'],[3,'motorcycle'],[5,'bus'],[7,'truck']]);
export function iou(a,b){const w=Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x)),h=Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));return w*h/(a.w*a.h+b.w*b.h-w*h+1e-9);}
export function decode(data,dims,geometry,confidence=.25){
  const [_,channels,n]=dims;if(channels!==84||!n)throw new Error(`Expected YOLO detection output [1,84,N], received ${dims}`);
  const {width,height,scale,left,top}=geometry, candidates=[];
  for(let i=0;i<n;i++){
    // Match all-class argmax before restricting to traffic classes.
    let cls=0,score=-Infinity;for(let c=0;c<80;c++){const v=data[(c+4)*n+i];if(v>score){score=v;cls=c;}}
    if(score<confidence||!VEHICLES.has(cls))continue;
    const cx=data[i],cy=data[n+i],bw=data[n*2+i],bh=data[n*3+i];
    const x1=Math.max(0,Math.min(1,((cx-bw/2-left)/scale)/width));
    const y1=Math.max(0,Math.min(1,((cy-bh/2-top)/scale)/height));
    const x2=Math.max(0,Math.min(1,((cx+bw/2-left)/scale)/width));
    const y2=Math.max(0,Math.min(1,((cy+bh/2-top)/scale)/height));
    if(x2>x1&&y2>y1)candidates.push({x:x1,y:y1,w:x2-x1,h:y2-y1,cls,score});
  }
  candidates.sort((a,b)=>b.score-a.score);const keep=[];
  for(const b of candidates.slice(0,1000)){if(!keep.some(a=>a.cls===b.cls&&iou(a,b)>.7))keep.push(b);if(keep.length===100)break;}
  return keep;
}
// Lightweight greedy IoU + centroid association. This is not the Python ByteTrack implementation.
export class Tracker{
  constructor(){this.reset();}
  reset(){this.tracks=[];this.next=1;this.up=0;this.down=0;this.line=.55;}
  update(boxes,time,line=.55){
    if(line!==this.line){this.reset();this.line=line;}
    this.tracks=this.tracks.filter(t=>time-t.seen<.8);const assigned=new Set(),used=new Set(),pairs=[];
    for(let i=0;i<this.tracks.length;i++)for(let j=0;j<boxes.length;j++){
      const a=this.tracks[i],b=boxes[j];if(a.cls!==b.cls)continue;
      const overlap=iou(a,b),distance=Math.hypot(a.x+a.w/2-b.x-b.w/2,a.y+a.h/2-b.y-b.h/2);
      if(overlap>.12||(distance<.045&&Math.abs(Math.log((a.w*a.h)/(b.w*b.h)))<.8))pairs.push({i,j,score:overlap-distance});
    }
    pairs.sort((a,b)=>b.score-a.score);
    for(const {i,j} of pairs){if(assigned.has(i)||used.has(j))continue;assigned.add(i);used.add(j);const t=this.tracks[i];Object.assign(t,boxes[j],{seen:time,hits:t.hits+1});}
    boxes.forEach((b,j)=>{if(!used.has(j))this.tracks.push({...b,id:this.next++,seen:time,hits:1,side:0,counted:false});});
    const visible=this.tracks.filter(t=>t.seen===time);
    for(const t of visible){const cx=t.x+t.w/2,cy=t.y+t.h/2,side=cy<line-.012?-1:cy>line+.012?1:0;
      if(side&&t.hits>=3){if(t.side&&t.side!==side&&!t.counted&&cx>=.1&&cx<=.9){if(side===1)this.down++;else this.up++;t.counted=true;}t.side=side;}}
    return {tracks:visible.map(t=>({...t})),up:this.up,down:this.down,occupancy:visible.filter(t=>{const x=t.x+t.w/2,y=t.y+t.h/2;return x>=.15&&x<=.85&&y>=.35&&y<=.9;}).length};
  }
}
