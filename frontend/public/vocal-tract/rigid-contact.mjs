// One-way display contact. Native teeth and acoustic/contact surfaces stay fixed.
// Each pair of dental ribs bounds a small convex solid in the current jaw pose.
const EPS=1e-8;
const sub=(a,b)=>a.map((v,k)=>v-b[k]);
const dot=(a,b)=>a.reduce((s,v,k)=>s+v*b[k],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const box=points=>[0,1,2].map(k=>[Math.min(...points.map(p=>p[k])),Math.max(...points.map(p=>p[k]))]);
const overlaps=(a,b)=>a.every((v,k)=>v[1]>=b[k][0]-EPS&&b[k][1]>=v[0]-EPS);
let cachedKey='',cachedSolids=[];
export function dentalSolids(meshes){
  const teeth=meshes.filter(m=>m.name.endsWith('_teeth'));
  const key=JSON.stringify(teeth.map(m=>[m.name,m.ribs,m.points,m.vertices]));
  if(key===cachedKey)return cachedSolids;
  const solids=[];
  for(const mesh of teeth){
    for(let r=1;r<mesh.ribs;r++){
      const points=[];
      for(const rib of [r-1,r])for(let j=0;j<mesh.points-1;j++)points.push(mesh.vertices.slice((rib*mesh.points+j)*3,(rib*mesh.points+j)*3+3));
      const planes=[];
      for(let a=0;a<points.length;a++)for(let b=a+1;b<points.length;b++)for(let c=b+1;c<points.length;c++){
        let n=cross(sub(points[b],points[a]),sub(points[c],points[a])),length=Math.hypot(...n);if(length<EPS)continue;
        n=n.map(v=>v/length);let d=dot(n,points[a]),dist=points.map(p=>dot(n,p)-d);
        if(Math.min(...dist)<-EPS&&Math.max(...dist)>EPS)continue;
        if(Math.max(...dist)>EPS){n=n.map(v=>-v);d=-d;}
        if(!planes.some(p=>Math.abs(p.d-d)<EPS&&Math.hypot(...sub(p.n,n))<EPS))planes.push({n,d});
      }
      if(planes.length>=4)solids.push({planes,box:box(points),name:mesh.name,rib:r});
    }
  }
  cachedKey=key;cachedSolids=solids;return solids;
}
export function splitPolygon(poly,plane,epsilon=EPS){
  const inside=[],outside=[];
  for(let i=0;i<poly.length;i++){
    const a=poly[i],b=poly[(i+1)%poly.length],da=dot(plane.n,a)-plane.d,db=dot(plane.n,b)-plane.d;
    if(da<=epsilon)inside.push(a);if(da>=-epsilon)outside.push(a);
    if((da>epsilon&&db< -epsilon)||(da< -epsilon&&db>epsilon)){
      const t=da/(da-db),p=a.map((v,k)=>v+t*(b[k]-v));inside.push(p);outside.push(p);
    }
  }
  return {inside,outside};
}
function subtract(poly,solid){
  if(!overlaps(box(poly),solid.box)||solid.planes.some(p=>poly.every(v=>dot(p.n,v)-p.d>=-EPS)))return [poly];
  const parts=[];let pending=poly;
  for(const plane of solid.planes){
    const {inside,outside}=splitPolygon(pending,plane);
    if(outside.length>=3)parts.push(outside);
    pending=inside;if(pending.length<3)break;
  }
  return parts;
}
export function subtractTeeth(mesh,solids,protectedVertices=0){
  const vertices=[...mesh.vertices],triangles=[];
  for(let i=0;i<mesh.triangles.length;i+=3){
    const a=mesh.triangles[i],b=mesh.triangles[i+1],c=mesh.triangles[i+2];
    if(a<protectedVertices&&b<protectedVertices&&c<protectedVertices){triangles.push(a,b,c);continue;}
    const v=mesh.vertices,ax=a*3,bx=b*3,cx=c*3;
    const xmin=Math.min(v[ax],v[bx],v[cx]),xmax=Math.max(v[ax],v[bx],v[cx]),
      ymin=Math.min(v[ax+1],v[bx+1],v[cx+1]),ymax=Math.max(v[ax+1],v[bx+1],v[cx+1]),
      zmin=Math.min(v[ax+2],v[bx+2],v[cx+2]),zmax=Math.max(v[ax+2],v[bx+2],v[cx+2]);
    let parts=null,original;
    // Every clipped fragment remains inside this original triangle's bounds.
    for(const solid of solids){const q=solid.box;
      if(xmax<q[0][0]-EPS||xmin>q[0][1]+EPS||ymax<q[1][0]-EPS||ymin>q[1][1]+EPS||zmax<q[2][0]-EPS||zmin>q[2][1]+EPS)continue;
      if(parts===null){original=[v.slice(ax,ax+3),v.slice(bx,bx+3),v.slice(cx,cx+3)];parts=[original];}
      parts=parts.flatMap(p=>subtract(p,solid));if(!parts.length)break;
    }
    if(parts===null||(parts.length===1&&parts[0]===original)){triangles.push(a,b,c);continue;}
    for(const poly of parts)for(let j=1;j<poly.length-1;j++){
      const face=[poly[0],poly[j],poly[j+1]];
      if(Math.hypot(...cross(sub(face[1],face[0]),sub(face[2],face[0])))<EPS)continue;
      const start=vertices.length/3;vertices.push(...face.flat());triangles.push(start,start+1,start+2);
    }
  }
  return {...mesh,vertices,triangles};
}
