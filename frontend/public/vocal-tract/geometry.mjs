// Display geometry only. The native VTL tube remains the acoustic authority.
import {velumModel,smoothContour,tongueEnvelope} from './anatomy.mjs';
export const point=(mesh,i)=>mesh.vertices.slice(i*3,i*3+3);
// VTL's reference axis is not necessarily inside the changing air space.
// Draw at the actual midline profile midpoint; retain native distance/index.
export function airwayGuide(state){
  return state.airway_sections.map(sec=>{
    const c=state.centerline[sec.index],up=sec.upper[48],lo=sec.lower[48];
    if(up===null||lo===null||up<=lo)return null; // lateral-only openings have no midline air
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
  const contours={...state.contours},lipHandles={},lipEdges={};
  const nativeUpper=state.meshes.find(m=>m.name==='upper_lip'),nativeLower=state.meshes.find(m=>m.name==='lower_lip');
  const meshes=state.meshes.map(mesh=>{
    if(!mesh.name.endsWith('_lip'))return mesh;
    // The native strips include long gum attachment rays. Closing/smoothing
    // those rays as a lip volume made crossing fins at the mouth corners.
    // Loft only the vermilion envelope, anchored to each native aperture edge.
    // Shared transverse stations and separated half-spaces prevent interpenetration.
    const midline=Math.floor(mesh.ribs/2),upper=mesh.name==='upper_lip',edges=[];
    const profiles=Array.from({length:mesh.ribs},(_,r)=>{
      const a=point(nativeUpper,r*nativeUpper.points+4),b=point(nativeLower,r*nativeLower.points+4),seam=(a[1]+b[1])/2;
      const edge=[...(upper?a:b)];edge[1]=upper?Math.max(a[1],seam):Math.min(b[1],seam);edge[2]=(a[2]+b[2])/2;edges.push(edge);
      const taper=Math.cos(Math.abs(r-midline)/midline*Math.PI/2),direction=upper?1:-1;
      return Array.from({length:40},(_,j)=>{const angle=j/40*2*Math.PI-Math.PI/2,co=Math.cos(angle);return [edge[0]+co*(co>=0?.72:.40)*taper,edge[1]+direction*.75*taper*(1+Math.sin(angle))/2,edge[2]];});
    });
    lipHandles[mesh.name]=edges[midline].slice(0,2);lipEdges[mesh.name]=edges;
    const N=profiles[0].length,triangles=[];
    for(let r=1;r<profiles.length;r++)for(let j=0;j<N;j++){const a=(r-1)*N+j,b=(r-1)*N+(j+1)%N,c=r*N+j,d=r*N+(j+1)%N;triangles.push(a,b,c,b,d,c);}
    contours[mesh.name]=profiles[midline].map(p=>p.slice(0,2));
    return {...mesh,vertices:profiles.flat(2),triangles,points:N};
  });
  const tongue=tongueEnvelope(state);meshes[meshes.findIndex(m=>m.name==='tongue')]=tongue.mesh;
  return {...state,meshes,contours,lipHandles,lipEdges,tongueOutline:tongue.outline};
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
    else points=Array.from({length:N},()=>[0,0]); // no fabricated air tunnel at full closure
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
  // The upper half-ring's midline is the posterior pharyngeal wall here.
  // Searching for minimum x picked an off-midline facet and twisted the branch.
  const jCenter=N/4;
  const width=8,j0=jCenter-width,j1=jCenter+width,id=(r,j)=>r*N+(j+N*2)%N;
  const patch=[];for(let j=j0;j<=j1;j++)patch.push(id(r0,j));for(let r=r0+1;r<=r1;r++)patch.push(id(r,j1));for(let j=j1-1;j>=j0;j--)patch.push(id(r1,j));for(let r=r1-1;r>r0;r--)patch.push(id(r,j0));
  const holeCells=new Set();for(let j=j0;j<j1;j++)holeCells.add((j+N*2)%N);
  const vertices=[...oral.vertices,...nose.vertices],triangles=[];
  const palate=velumModel(state);
  // Keep the reference posterior wall independent of the moving soft palate.
  // The native upper-cover bridge is a construction surface, not tissue pulled
  // into the velum. Replace only its posterior display patch, not acoustic areas.
  for(let r=Math.max(0,r0-5);r<=Math.min(rows-1,r1+5);r++)for(let j=1;j<N/2;j++){
    const i=id(r,j)*3,y=vertices[i+1],x=vertices[i],wall=palate.wallAt(y);
    if(y<state.contours.upper_cover[7][1]||x>wall+.9)continue;
    const blend=Math.min(1,Math.max(0,(r-(r0-5))/3),Math.max(0,((r1+5)-r)/3));
    vertices[i]=x+(wall-x)*blend;
  }
  for(let r=0;r<rows-1;r++)for(let j=0;j<N;j++){
    if(r>=r0&&r<r1&&holeCells.has(j))continue;
    const a=id(r,j),b=id(r,j+1),c=id(r+1,j),d=id(r+1,j+1);triangles.push(a,b,c,b,d,c);
  }
  const noseOffset=oral.vertices.length/3;for(const i of nose.triangles)triangles.push(noseOffset+i);
  const mesh={vertices,triangles},start=mean(patch.map(i=>point(mesh,i))),end=mean(outlet.map(i=>point(nose,i)));
  // Local registration uses a horizontal ellipse between the native branch and
  // measured nasal outlet. Its polygon area is VO, independent of render density.
  const center=palate.gate,area=Math.max(0,state.nasal.port_area),K=48;
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
  const oralIndexCount=(rows-1)*N*6-2*width*2*6;
  const connector={vertices:mesh.vertices,triangles:mesh.triangles.slice(oralIndexCount+nose.triangles.length)};
  return {...mesh,connector,oralVertexCount:noseOffset,oralIndexCount,
    junction:{open:opened,area,center,start,end,throat:lower.map(i=>point(mesh,i)),seamVertices:patch.length+outlet.length,reference:true}};
}
