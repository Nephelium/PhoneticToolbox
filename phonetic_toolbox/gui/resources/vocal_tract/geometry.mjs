// Display geometry only. The native VTL tube remains the acoustic authority.
import {velumModel} from './anatomy.mjs';
export const point=(mesh,i)=>mesh.vertices.slice(i*3,i*3+3);
// VTL's reference axis is not necessarily inside the changing air space.
// Draw at the actual midline profile midpoint; retain native distance/index.
export function airwayGuide(state){
  return state.airway_sections.map(sec=>{
    const c=state.centerline[sec.index],up=sec.upper[48],lo=sec.lower[48];
    if(up===null||lo===null)return null; // lateral-only openings have no midline air
    const p=d=>[c[0]+c[3]*d,c[1]+c[4]*d];
    return {index:sec.index,position:c[2],point:p((up+lo)/2),upper:p(up),lower:p(lo)};
  });
}
export function nearestSection(state,position){
  return state.centerline.reduce((best,c,i)=>Math.abs(c[2]-position)<Math.abs(state.centerline[best][2]-position)?i:best,0);
}

// Local display relief at the vermilion/teeth join. Acoustic surfaces are intact.
// Both scene views and lip handles consume the same adjusted vertices.
export function displayLips(state){
  const contours={...state.contours};
  const meshes=state.meshes.map(mesh=>{
    if(!mesh.name.endsWith('_lip'))return mesh;
    const teeth=state.meshes.find(m=>m.name===mesh.name.replace('_lip','_teeth'));
    const vertices=[...mesh.vertices];
    for(let i=0;i<vertices.length;i+=3){
      const index=i/3%mesh.points;if(index<1||index>6)continue;
      const z=vertices[i+2];let front=-Infinity,low=Infinity,high=-Infinity;
      for(let j=0;j<teeth.vertices.length;j+=3)if(Math.abs(teeth.vertices[j+2]-z)<.22){front=Math.max(front,teeth.vertices[j]);low=Math.min(low,teeth.vertices[j+1]);high=Math.max(high,teeth.vertices[j+1]);}
      if(vertices[i+1]>=low-.12&&vertices[i+1]<=high+.12)vertices[i]=Math.max(vertices[i],front+.055);
    }
    const midline=Array.from({length:mesh.ribs},(_,r)=>r).reduce((best,r)=>Math.abs(vertices[(r*mesh.points+4)*3+2])<Math.abs(vertices[(best*mesh.points+4)*3+2])?r:best,0);
    contours[mesh.name]=Array.from({length:mesh.points},(_,i)=>vertices.slice((midline*mesh.points+i)*3,(midline*mesh.points+i)*3+2));
    return {...mesh,vertices};
  });
  return {...state,meshes,contours};
}
const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
const mean=points=>points[0].map((_,k)=>points.reduce((s,p)=>s+p[k],0)/points.length);
export function planeSegments(vertices,indices,z=0){
  const out=[];
  for(let f=0;f<indices.length;f+=3){const points=[];
    for(let e=0;e<3;e++){const a=indices[f+e]*3,b=indices[f+(e+1)%3]*3,za=vertices[a+2]-z,zb=vertices[b+2]-z;
      if((za<=0&&zb>0)||(zb<=0&&za>0)){const t=za/(za-zb);points.push([vertices[a]+t*(vertices[b]-vertices[a]),vertices[a+1]+t*(vertices[b+1]-vertices[a+1])]);}}
    if(points.length===2)out.push(points);
  }return out;
}
export function stitch(segments){
  const key=p=>p.map(v=>Math.round(v*1000)).join(','),nodes=new Map(),edges=[];
  segments.forEach(([a,b])=>{const ka=key(a),kb=key(b);if(ka===kb)return;const id=edges.length;edges.push([a,b,false]);for(const k of [ka,kb]){if(!nodes.has(k))nodes.set(k,[]);nodes.get(k).push(id);}});
  const paths=[];
  for(let first=0;first<edges.length;first++){if(edges[first][2])continue;let id=first,point=edges[first][0],path=[point];
    while(id!==undefined){const e=edges[id];e[2]=true;point=key(point)===key(e[0])?e[1]:e[0];path.push(point);id=nodes.get(key(point)).find(i=>!edges[i][2]);}
    point=path[0];id=nodes.get(key(point)).find(i=>!edges[i][2]);
    while(id!==undefined){const e=edges[id];e[2]=true;point=key(point)===key(e[0])?e[1]:e[0];path.unshift(point);id=nodes.get(key(point)).find(i=>!edges[i][2]);}
    if(path.length>3)paths.push(path);
  }return paths.sort((a,b)=>b.length-a.length);
}
export function boundaryLoops(mesh){
  const edges=new Map(),nv=mesh.vertices.length/3;
  for(let i=0;i<mesh.triangles.length;i+=3)for(let j=0;j<3;j++){
    const a=mesh.triangles[i+j],b=mesh.triangles[i+(j+1)%3],key=Math.min(a,b)*nv+Math.max(a,b);
    if(edges.has(key))edges.delete(key);else edges.set(key,[a,b]);
  }
  const adj=new Map();for(const [a,b] of edges.values()){if(!adj.has(a))adj.set(a,[]);if(!adj.has(b))adj.set(b,[]);adj.get(a).push(b);adj.get(b).push(a);}
  const seen=new Set(),loops=[];
  for(const first of adj.keys()){if(seen.has(first))continue;let i=first,loop=[];
    while(!seen.has(i)){seen.add(i);loop.push(i);const next=adj.get(i).find(k=>!seen.has(k));if(next===undefined)break;i=next;}
    if(loop.length>2){const edge=edges.get(Math.min(loop[0],loop[1])*nv+Math.max(loop[0],loop[1]));if(edge[0]!==loop[0])loop.reverse();loops.push(loop);}
  }return loops;
}
export function nasalOutlet(mesh){
  const target=mesh.landmarks.nasopharynx.reference_cm;
  return boundaryLoops(mesh).sort((a,b)=>distance(mean(a.map(i=>point(mesh,i))),target)-distance(mean(b.map(i=>point(mesh,i))),target))[0];
}
export function sectionPaths(mesh,z=0){
  const segments=planeSegments(mesh.vertices,mesh.triangles,z);
  // Close only the slice at real open outlets (glottis, lips, nostrils).
  for(const loop of boundaryLoops(mesh)){
    const hits=[];for(let i=0;i<loop.length;i++){const a=point(mesh,loop[i]),b=point(mesh,loop[(i+1)%loop.length]);if((a[2]<=z&&b[2]>z)||(b[2]<=z&&a[2]>z)){const t=(z-a[2])/(b[2]-a[2]);hits.push([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]);}}
    for(let i=0;i+1<hits.length;i+=2)segments.push([hits[i],hits[i+1]]);
  }return stitch(segments);
}
function resampleClosed(points,n){
  const lengths=[0];for(let i=1;i<=points.length;i++)lengths.push(lengths.at(-1)+distance(points[i%points.length],points[i-1]));
  const out=[];let j=0;for(let i=0;i<n;i++){const d=i/n*lengths.at(-1);while(j<points.length-1&&lengths[j+1]<d)j++;const a=points[j],b=points[(j+1)%points.length],t=(d-lengths[j])/(lengths[j+1]-lengths[j]||1);out.push(a.map((v,k)=>v+t*(b[k]-v)));}return out;
}
export function oralMesh(state){
  const vertices=[],triangles=[],N=64;
  for(const sec of state.airway_sections){const c=state.centerline[sec.index],valid=sec.upper.map((v,i)=>v===null?null:i).filter(v=>v!==null);let points=[];
    if(valid.length>1){
      const first=valid[0],last=valid.at(-1),sample=(values,index)=>{let a=Math.floor(index),b=Math.ceil(index);while(a>first&&values[a]===null)a--;while(b<last&&values[b]===null)b++;const t=(index-a)/(b-a||1);return values[a]*(1-t)+values[b]*t;};
      // Fixed upper/lower correspondence avoids rotating a ring when its
      // perimeter changes, particularly at /u/ and /o/ constrictions.
      points=Array.from({length:N},(_,j)=>{const index=first+(last-first)*(1-Math.cos(j/N*2*Math.PI))/2;return [index*7/96-3.5,sample(j<=N/2?sec.upper:sec.lower,index)];});
    }
    else points=Array.from({length:N},(_,i)=>[.005*Math.cos(i/N*2*Math.PI),.005*Math.sin(i/N*2*Math.PI)]);
    for(const [z,d] of points)vertices.push(c[0]+c[3]*d,c[1]+c[4]*d,z);
  }
  for(let r=1;r<state.airway_sections.length;r++)for(let j=0;j<N;j++){const a=(r-1)*N+j,b=(r-1)*N+(j+1)%N,c=r*N+j,d=r*N+(j+1)%N;triangles.push(a,b,c,b,d,c);}
  return {vertices,triangles,ringSize:N};
}
// Align cyclic boundaries before a zipper loft. Shared indices eliminate cracks.
function align(mesh,a,b){
  let best=b,bestCost=Infinity;
  for(const order of [b,b.toReversed()])for(let offset=0;offset<b.length;offset++){
    let cost=0;for(let i=0;i<a.length;i+=Math.max(1,Math.floor(a.length/12)))cost+=distance(point(mesh,a[i]),point(mesh,order[(Math.round(i/a.length*b.length)+offset)%b.length]));
    if(cost<bestCost){bestCost=cost;best=order.map((_,i)=>order[(i+offset)%b.length]);}
  }return best;
}
function loft(mesh,a,b){
  b=align(mesh,a,b);let i=0,j=0;
  while(i<a.length||j<b.length){
    if((i+1)/a.length<=(j+1)/b.length){mesh.triangles.push(a[i%a.length],a[(i+1)%a.length],b[j%b.length]);i++;}
    else{mesh.triangles.push(a[i%a.length],b[(j+1)%b.length],b[j%b.length]);j++;}
  }return b;
}
function cap(mesh,loop){const i=mesh.vertices.length/3;mesh.vertices.push(...mean(loop.map(j=>point(mesh,j))));for(let j=0;j<loop.length;j++)mesh.triangles.push(i,loop[j],loop[(j+1)%loop.length]);}
export function connectedAirway(state,nose,outlet=nasalOutlet(nose)){
  const oral=oralMesh(state),N=oral.ringSize,rows=state.airway_sections.length;
  const nearest=state.airway_sections.reduce((best,s,i)=>Math.abs(state.centerline[s.index][2]-state.nasal.port_position)<Math.abs(state.centerline[state.airway_sections[best].index][2]-state.nasal.port_position)?i:best,0);
  const r0=Math.max(1,Math.min(rows-4,nearest-1)),r1=r0+2,target=nose.landmarks.nasopharynx.reference_cm;
  let jCenter=0;for(let j=1;j<N;j++)if(point(oral,nearest*N+j)[0]<point(oral,nearest*N+jCenter)[0])jCenter=j;
  const width=8,j0=jCenter-width,j1=jCenter+width,id=(r,j)=>r*N+(j+N*2)%N;
  const patch=[];for(let j=j0;j<=j1;j++)patch.push(id(r0,j));for(let r=r0+1;r<=r1;r++)patch.push(id(r,j1));for(let j=j1-1;j>=j0;j--)patch.push(id(r1,j));for(let r=r1-1;r>r0;r--)patch.push(id(r,j0));
  const holeCells=new Set();for(let j=j0;j<j1;j++)holeCells.add((j+N*2)%N);
  const vertices=[...oral.vertices,...nose.vertices],triangles=[];
  for(let r=0;r<rows-1;r++)for(let j=0;j<N;j++){
    if(r>=r0&&r<r1&&holeCells.has(j))continue;
    const a=id(r,j),b=id(r,j+1),c=id(r+1,j),d=id(r+1,j+1);triangles.push(a,b,c,b,d,c);
  }
  const noseOffset=oral.vertices.length/3;for(const i of nose.triangles)triangles.push(noseOffset+i);
  const mesh={vertices,triangles},start=mean(patch.map(i=>point(mesh,i))),end=mean(outlet.map(i=>point(nose,i)));
  // Local registration uses a horizontal ellipse between the native branch and
  // measured nasal outlet. Its polygon area is VO, independent of render density.
  const palate=velumModel(state),center=palate.gate,area=Math.max(0,state.nasal.port_area),K=48;
  const ring=(a,yOffset=0)=>{const radius=Math.sqrt(Math.max(a,.000001)/(K*.5*Math.sin(2*Math.PI/K))),ids=[];
    const halfWidth=palate.width,halfGap=a/(K*.5*Math.sin(2*Math.PI/K)*halfWidth);
    for(let k=0;k<K;k++){const t=k/K*2*Math.PI;ids.push(mesh.vertices.length/3);mesh.vertices.push(center[0]+halfGap*Math.cos(t),center[1]+yOffset,center[2]+halfWidth*Math.sin(t));}return ids;};
  const opened=area>0,lower=ring(opened?area:.002,opened?0:-.01),upper=opened?lower:ring(.002,.01);
  const lowerAligned=loft(mesh,patch,lower),upperAligned=opened?lowerAligned:upper;
  const noseLoop=outlet.map(i=>noseOffset+i),noseAligned=loft(mesh,upperAligned,noseLoop);
  // A zipper may reverse a ring to avoid twisting. Propagate that winding
  // across shared seams, including the nasal surface and closed-port caps.
  if((noseLoop.indexOf(noseAligned[1])-noseLoop.indexOf(noseAligned[0])+noseLoop.length)%noseLoop.length!==1){
    const first=(rows-1)*N*6-2*width*2*6;
    for(let i=first;i<first+nose.triangles.length;i+=3)[mesh.triangles[i+1],mesh.triangles[i+2]]=[mesh.triangles[i+2],mesh.triangles[i+1]];
  }
  if(!opened){cap(mesh,lowerAligned);cap(mesh,upperAligned.toReversed());}
  return {...mesh,oralVertexCount:noseOffset,oralIndexCount:(rows-1)*N*6-2*width*2*6,
    junction:{open:opened,area,center,start,end,throat:lower.map(i=>point(mesh,i)),seamVertices:patch.length+outlet.length,reference:true}};
}
