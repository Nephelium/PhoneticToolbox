import {fitSoftTissue} from './surface-contact.mjs';
// The display reconstruction has one 3-D authority; sagittal lines are slices.
// Native VTL defines the oral surface. MRI informs the dorsal tissue envelope.
export function registerNose(source){
  // Reference-only affine registration: retain posterior palatal clearance,
  // lower the anterior outlet into the external head's nasal envelope.
  // Apply the SAME map to every surface vertex and every landmark in 2-D/3-D.
  const transform=([x,y,z])=>[x,y+1.14-.28*x,z];
  const vertices=[];
  for(let i=0;i<source.vertices.length;i+=3)vertices.push(...transform(source.vertices.slice(i,i+3)));
  const landmarks=Object.fromEntries(Object.entries(source.landmarks).map(([name,p])=>[name,{...p,reference_cm:transform(p.reference_cm)}]));
  const bounds=[0,1].map(side=>[0,1,2].map(k=>vertices.reduce((v,p,i)=>i%3===k?(side?Math.max(v,p):Math.min(v,p)):v,side?-Infinity:Infinity)));
  return {...source,vertices,landmarks,bounds,registration:'m10 nasal/3: y += 1.14 - 0.28*x cm; display reference, not acoustic geometry'};
}
const smooth=(points,steps=5)=>{
  // Corner cutting stays inside the control polygon: no spline overshoot into
  // the pharyngeal wall or back through the thin uvular tip.
  let out=points;
  for(let n=0;n<2;n++)out=out.flatMap((a,i)=>{const b=out[(i+1)%out.length];return [a.map((v,k)=>.75*v+.25*b[k]),a.map((v,k)=>.25*v+.75*b[k])];});
  return out;
};
const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
function clipProfile(points,distance){
  const out=[];
  for(let i=0;i<points.length;i++){
    const a=points[i],b=points[(i+1)%points.length],da=distance(a),db=distance(b);
    if(da>=0)out.push(a);
    if((da>=0)!==(db>=0)){const t=da/(da-db);out.push(a.map((v,k)=>v+t*(b[k]-v)));}
  }return out;
}
function triangulate(points){
  const signed=points.reduce((sum,p,i)=>sum+p[0]*points[(i+1)%points.length][1]-points[(i+1)%points.length][0]*p[1],0),direction=Math.sign(signed);
  const ids=points.map((_,i)=>i),triangles=[];let attempts=0;
  while(ids.length>3&&attempts++<points.length*points.length){let found=false;
    for(let j=0;j<ids.length;j++){const a=ids[(j+ids.length-1)%ids.length],b=ids[j],c=ids[(j+1)%ids.length];if(direction*cross(points[a],points[b],points[c])<1e-9)continue;
      if(ids.some(k=>k!==a&&k!==b&&k!==c&&direction*cross(points[a],points[b],points[k])>=0&&direction*cross(points[b],points[c],points[k])>=0&&direction*cross(points[c],points[a],points[k])>=0))continue;
      triangles.push(a,b,c);ids.splice(j,1);found=true;break;
    }if(!found)throw Error('Invalid soft-palate profile');
  }triangles.push(...ids);return triangles;
}
export function velumModel(state){
  const c=state.contours,area=Math.max(0,state.nasal.port_area),free=c.upper_cover[9],root=c.upper_cover[13];
  // Continue the posterior wall's slope above its last native sample. A fixed
  // x clipping plane leaves tissue touching the sloping wall at small VO.
  const a=c.upper_cover[7],b=c.upper_cover[8],wallAt=y=>a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]);
  const u=c.uvula,tip=u.at(-1);
  const wall=wallAt(free[1]+.65);
  // The native uvula keeps its dimensions. Fit the port into the space behind
  // it, spreading the requested area transversely instead of shaving tissue.
  // The continuation and exported contour differ by <0.001 cm at closure.
  // Absorb that rounding offset to avoid a transverse flare at tiny openings.
  const clearance=Math.max(.001,u[0][0]-wall+.001);
  const gap=Math.min(2*area/(Math.PI*1.6),clearance),width=gap>0?2*area/(Math.PI*gap):1.6;
  const contact=[wall+gap,free[1]+.65],gate=[wall+gap/2,contact[1],0];
  // Native UVULA has two meridians. The acoustic-facing meridian must stay
  // exact: smoothing it with the dorsal envelope moved apparent contact by mm.
  const uvula=state.meshes.find(m=>m.name==='uvula'),P=uvula.points;
  const front=Array.from({length:uvula.ribs},(_,r)=>uvula.vertices.slice((r*P)*3,(r*P)*3+2));
  const lower=[...c.upper_cover.slice(9,14).toReversed(),...front.slice(1)];
  // Continuous vertex deformation, without clipping away/adding polygon
  // corners as VO crosses a threshold. The posterior contour joins smoothly.
  // Follow the posterior native meridian up to its shoulder, then a cubic roof
  // to the hard-palate attachment. The loop has fixed topology during dragging.
  const back=u.slice(0,-1).toReversed().map(p=>p.slice(0,2));
  const shoulder=back.at(-1),roof=[];
  // Carry the dorsal attachment up to the front of the velopharyngeal port.
  // Turning straight towards the hard palate below that port leaves a false
  // pocket between the oral connector and tissue, visible as a pointed spur.
  const joint=[contact[0],Math.max(contact[1],shoulder[1]+.3)];
  for(let i=1;i<=20;i++){
    const first=i<=6,t=first?i/6:(i-6)/14,v=1-t;
    const a=first?shoulder:joint,b=first?joint:[root[0],root[1]+.32],rise=(joint[1]-shoulder[1])/3;
    const c=first?[shoulder[0],shoulder[1]+rise]:[joint[0],joint[1]+.4],d=first?[joint[0],joint[1]-rise]:[root[0]-.65,root[1]+.48];
    roof.push([0,1].map(k=>v*v*v*a[k]+3*v*v*t*c[k]+3*v*t*t*d[k]+t*t*t*b[k]));
  }
  const profile=[...lower,...back,...roof];
  const vertices=[],triangles=[],N=profile.length,Q=12;
  const cover=state.meshes.find(m=>m.name==='upper_cover');
  const at=(m,r,j)=>m.vertices.slice((r*m.points+j)*3,(r*m.points+j)*3+3);
  const onZ=(points,z)=>{
    for(let j=1;j<points.length;j++){
      const a=points[j-1],b=points[j];if(z<Math.min(a[2],b[2])-1e-8||z>Math.max(a[2],b[2])+1e-8)continue;
      const t=(z-a[2])/(b[2]-a[2]||1);return a.map((v,k)=>v+t*(b[k]-v));
    }return points.at(-1);
  };
  const uvulaSide=(r,back,q)=>{
    const ids=back?(q<0?[4,3,2]:[4,5,6]):(q<0?[0,1,2]:[8,7,6]);
    const t=Math.abs(q)*2,lo=Math.min(1,Math.floor(t)),a=at(uvula,r,ids[lo]),b=at(uvula,r,ids[lo+1]);
    // Subdivide the actual polygon edges, including their native corners.
    // Sampling only by z cut inside those corners and changed the body shape.
    return a.map((v,k)=>v+(t-lo)*(b[k]-v));
  };
  // Tissue tapers from the broad palate to the narrow uvula, across the midline.
  for(let q=0;q<=Q;q++){const z=2*q/Q-1;
    for(let j=0;j<N;j++){const p=profile[j];
      if(j<4){vertices.push(...onZ(Array.from({length:cover.points},(_,k)=>at(cover,13-j,k)),z*1.15));}
      else if(j<8){vertices.push(...uvulaSide(j-4,false,z));}
      else if(j<11){vertices.push(...uvulaSide(10-j,true,z));}
      else {const t=(j-10)/20,width=.35+(1.15-.35)*t;vertices.push(p[0],p[1]+.08*z*z,z*width);}
    }
  }
  for(let q=0;q<Q;q++)for(let j=0;j<N;j++){const a=q*N+j,b=q*N+(j+1)%N,c=(q+1)*N+j,d=(q+1)*N+(j+1)%N;triangles.push(a,c,b,b,c,d);}
  const caps=triangulate(profile);for(let i=0;i<caps.length;i+=3){triangles.push(caps[i],caps[i+1],caps[i+2]);triangles.push(Q*N+caps[i],Q*N+caps[i+2],Q*N+caps[i+1]);}
  fitSoftTissue(vertices,triangles,state.meshes.find(m=>m.name==='tongue'),i=>i%N>=4&&i%N<=10);
  const section=Array.from({length:N},(_,j)=>vertices.slice((Q/2*N+j)*3,(Q/2*N+j)*3+2));
  return {vertices,triangles,profile:section,exposedProfile:section.slice(0,lower.length+back.length),gate,gap,width,wall,wallAt,contact,tip,root,area};
}

// Display smoothing stays inside the control polygon, never changes native areas.
export const smoothContour=points=>smooth(points);

// VTL's four ventral ribs are a sparse construction boundary (including a
// vertical ray to the floor). Keep the 34 dorsal/contact ribs exactly and loft
// a display-only ventral surface back to the native floor attachment.
export function tongueEnvelope(state){
  const native=state.meshes.find(m=>m.name==='tongue'),P=native.points,D=34,Q=24;
  const tip=state.contours.tongue[D-1],floor=state.contours.lower_cover;
  const p=state.limited,curl=Math.max(0,Math.min(1,(p[12]-p[10])/.75))*Math.max(0,Math.min(1,(p[11]-p[13])/.5));
  let anchor=7;
  for(let i=7;i<floor.length;i++)if(floor[i][0]<tip[0]-.8+1.6*curl&&floor[i][1]<tip[1]-.15)anchor=i;
  const end=floor[anchor],vertices=native.vertices.slice(0,D*P*3),triangles=[];
  // Retain the native diagonals too: flipping a non-planar quad's diagonal
  // changes the contact surface even when all four vertices are unchanged.
  for(let i=0;i<native.triangles.length;i+=3){const face=native.triangles.slice(i,i+3);if(face.every(j=>j<D*P))triangles.push(...face);}
  const floorMesh=state.meshes.find(m=>m.name==='lower_cover');
  for(let r=1;r<=Q;r++)for(let j=0;j<P;j++){
    const a=native.vertices.slice(((D-1)*P+j)*3,((D-1)*P+j)*3+3),t=r/Q,u=1-t;
    const z=j/(P-1)*(floorMesh.points-1),lo=Math.floor(z),hi=Math.ceil(z),f=z-lo;
    const b=[0,1,2].map(k=>floorMesh.vertices[(anchor*floorMesh.points+lo)*3+k]*(1-f)+floorMesh.vertices[(anchor*floorMesh.points+hi)*3+k]*f);
    // Smooth turn under the tip, then a shallow tangent into the mouth floor.
    const front=Math.max(...Array.from({length:D},(_,r)=>native.vertices[(r*P+j)*3]));
    // A curled tip's underside returns around the anterior blade, not through
    // the descending dorsal surface. Blend with the ordinary ventral loft.
    const prev=native.vertices.slice(((D-2)*P+j)*3,((D-2)*P+j)*3+3),length=Math.hypot(a[0]-prev[0],a[1]-prev[1])||1;
    const cx=a[0]-.14,dx=b[0]+Math.max(.12,(a[0]-b[0])*.26);
    const c=[cx+curl*(a[0]+.38*(a[0]-prev[0])/length-cx),a[1]-.36+curl*(.36+.38*(a[1]-prev[1])/length),a[2]],d=[dx+curl*(Math.max(dx,front+.45)-dx),b[1]+.08,b[2]];
    for(let k=0;k<3;k++)vertices.push(u*u*u*a[k]+3*u*u*t*c[k]+3*u*t*t*d[k]+t*t*t*b[k]);
  }
  const R=D+Q;
  for(let r=D;r<R;r++)for(let j=0;j<P-1;j++){const a=(r-1)*P+j,b=a+1,c=r*P+j,d=c+1;triangles.push(a,b,c,b,d,c);}
  const contour=Array.from({length:R},(_,r)=>vertices.slice((r*P+Math.floor(P/2))*3,(r*P+Math.floor(P/2))*3+2));
  return {mesh:{...native,vertices,triangles,ribs:R,dynamicRibs:D},outline:[...contour,...floor.slice(4,anchor).toReversed()],contour,anchor,end};
}
