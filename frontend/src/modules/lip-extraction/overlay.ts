import {metricSpec,type Point} from './metrics.ts';
// V2 _mesh_connections = FACEMESH_TESSELATION | FACEMESH_CONTOURS.
// Canonical frozen adjacency has the same undirected edges. Iris points are
// drawn as points, as in V2, without inventing additional iris connections.
export const meshEdges=metricSpec.neighbors.flatMap((neighbors,i)=>neighbors.filter(j=>i<j).map(j=>[i,j] as const));
export function faceBounds(frames:Iterable<{points?:Point[]|null}>){
 let left=Infinity,top=Infinity,right=-Infinity,bottom=-Infinity;
 for(const frame of frames)for(const [x,y] of frame.points??[]){if(!Number.isFinite(x)||!Number.isFinite(y))continue;left=Math.min(left,x);right=Math.max(right,x);top=Math.min(top,y);bottom=Math.max(bottom,y);}
 return right>left&&bottom>top?{left,top,right,bottom}:null;
}
export function fitFace(points:Point[],width:number,height:number,bounds=faceBounds([{points}]),fill=.82):Point[]{
 if(!bounds)return points;
 const scale=Math.min(width*fill/(bounds.right-bounds.left),height*fill/(bounds.bottom-bounds.top));
 const cx=(bounds.left+bounds.right)/2,cy=(bounds.top+bounds.bottom)/2;
 return points.map(([x,y])=>[(x-cx)*scale+width/2,(y-cy)*scale+height/2]);
}
export function drawLegacyOverlay(context:CanvasRenderingContext2D,points:Point[],width:number,height:number){
 context.strokeStyle='#00ff00';context.fillStyle='#00ff00';context.lineWidth=1;
 const pixel=(p:Point):Point=>[Math.trunc(p[0]),Math.trunc(p[1])];
 const inside=(p:Point)=>p[0]>=0&&p[0]<width&&p[1]>=0&&p[1]<height;
 context.beginPath();
 for(const [a,b] of meshEdges){if(!points[a]||!points[b])continue;const p=pixel(points[a]),q=pixel(points[b]);if(inside(p)&&inside(q)){context.moveTo(...p);context.lineTo(...q);}}
 context.stroke();context.beginPath();
 for(const point of points){const p=pixel(point);context.moveTo(p[0]+2,p[1]);context.arc(p[0],p[1],2,0,2*Math.PI);}
 context.fill();
}
