// Source: V2 metrics.py; one versioned index specification, float32 legacy coordinates.
import spec from '../../../../resources/m05/metric-spec.json' with {type:'json'};
export type Point = [number, number];
export type Metrics = Record<string, number | null>;
const f = Math.fround;
export function pixelPoints(points:number[][],width:number,height:number):Point[]{
  return points.map(p=>[f(p[0]*width),f(p[1]*height)]);
}
function area(points:Point[]){
  // Original vertices[:,0/1] has stride 2. OpenBLAS 0.3.29 sdot strided path
  // groups two float32 products before double accumulation, then returns float32.
  let a=0,b=0;
  for(let i=0;i<points.length;i+=2){const p=points[i],q=points[(i+points.length-1)%points.length],r=points[i+1];a+=f(f(p[0]*q[1])+f(r[0]*p[1]));b+=f(f(p[1]*q[0])+f(r[1]*p[0]));}
  return .5*Math.abs(f(f(a)-f(b)));
}
function perimeter(points:Point[]){let total=0;for(let i=0;i<points.length;i++){const p=points[i],q=points[(i+1)%points.length],dx=f(p[0]-q[0]),dy=f(p[1]-q[1]);total+=f(Math.sqrt(f(f(dx*dx)+f(dy*dy))));}return total;}
export function lipMetrics(points:Point[]):Metrics{
  if(points.length!==478||points.some(p=>p.length!==2||p.some(x=>!Number.isFinite(x))))throw Error('invalid_landmarks');
  const outer=spec.OUTER_LIP_LANDMARKS.map(i=>points[i]),inner=spec.INNER_LIP_LANDMARKS.map(i=>points[i]),face=spec.FACE_OVAL.map(i=>points[i]);
  const fw=Math.abs(f(points[spec.RIGHT_FACE][0]-points[spec.LEFT_FACE][0])),fh=Math.abs(f(points[spec.BOTTOM_FACE][1]-points[spec.TOP_FACE][1]));
  const span=(p:Point[],axis:number)=>f(Math.max(...p.map(x=>x[axis]))-Math.min(...p.map(x=>x[axis])));
  const height=span(outer,1),ow=span(outer,0),iw=span(inner,0),total=ow+iw,open=points[14][1]-points[13][1];
  const fa=area(face),la=area(outer)-area(inner),length=perimeter(outer),ratio=(a:number,b:number)=>b>0?a/b:null;
  return {area:ratio(la,fa),face_width:fw,face_height:fh,height_px:height,outer_width_px:ow,inner_width_px:iw,total_width_px:total,
    open_px:open,length,height:ratio(height,fh),outer_width:ratio(ow,fw),inner_width:ratio(iw,fw),total_width:ratio(total,fw),open:ratio(open,fh),circularity:ratio(4*Math.PI*la,length*length)};
}
export const metricSpec=spec;
