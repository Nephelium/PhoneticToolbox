import type {ReconstructionConfig} from '../../platform/research.ts';
type Stroke=NonNullable<ReconstructionConfig['strokes']>[number];

// Same pixel-center distance rule as phonetic_core.spec2wav.editing.pixel_brush.
function nearestEven(value:number){const low=Math.floor(value);return value-low===.5?low+(low%2):Math.round(value);}
export function pixelStroke(width:number,height:number,stroke:Stroke){
 const radius=stroke.size/2,offset=stroke.size%2?.5:0;
 const points=stroke.points.map(p=>({x:nearestEven(p.x*(width-1))+offset,y:nearestEven(p.y*(height-1))+offset}));
 const left=Math.max(0,Math.floor(Math.min(...points.map(p=>p.x))-radius)),top=Math.max(0,Math.floor(Math.min(...points.map(p=>p.y))-radius));
 const right=Math.min(width,Math.ceil(Math.max(...points.map(p=>p.x))+radius)),bottom=Math.min(height,Math.ceil(Math.max(...points.map(p=>p.y))+radius));
 const w=right-left,h=bottom-top,mask=new Uint8Array(w*h);
 for(let i=0;i<Math.max(1,points.length-1);i++){
  const a=points[i],b=points[Math.min(i+1,points.length-1)],dx=b.x-a.x,dy=b.y-a.y,denominator=dx*dx+dy*dy;
  const x0=Math.max(left,Math.floor(Math.min(a.x,b.x)-radius)),x1=Math.min(right,Math.ceil(Math.max(a.x,b.x)+radius));
  const y0=Math.max(top,Math.floor(Math.min(a.y,b.y)-radius)),y1=Math.min(bottom,Math.ceil(Math.max(a.y,b.y)+radius));
  for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){
   const t=denominator?Math.max(0,Math.min(1,((x+.5-a.x)*dx+(y+.5-a.y)*dy)/denominator)):0;
   if((x+.5-a.x-t*dx)**2+(y+.5-a.y-t*dy)**2<=radius*radius)mask[(y-top)*w+x-left]=1;
  }
 }
 return {left,top,width:w,height:h,mask};
}
