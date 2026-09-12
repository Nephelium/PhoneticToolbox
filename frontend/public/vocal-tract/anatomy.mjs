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
  const wall=wallAt(free[1]+.65),width=1.05,gap=2*area/(Math.PI*width),contact=[wall+gap,free[1]+.65],gate=[wall+gap/2,contact[1],0];
  const u=c.uvula,tip=u.at(-1),lower=[...c.upper_cover.slice(9,14).toReversed(),[free[0],free[1]-.42],[free[0]-.09,free[1]-.65],tip];
  const control=[...lower,u[2],u[1],[Math.max(u[0][0],wall+gap),u[0][1]],contact,[free[0]-.08,free[1]+.87],[root[0]-.65,root[1]+.5],[root[0],root[1]+.38]];
  const relief=.095*Math.min(1,area/.04); // clear display strokes, ramp from closure
  const clipped=clipProfile(clipProfile(control,p=>p[0]-wall-gap),p=>p[0]-wallAt(p[1])-gap-relief);
  const profile=smooth(clipped).filter((p,i,a)=>Math.abs(cross(a[(i+a.length-1)%a.length],p,a[(i+1)%a.length]))>1e-9);
  const vertices=[],triangles=[],N=profile.length,Q=12;
  // Tissue tapers from the broad palate to the narrow uvula, across the midline.
  for(let q=0;q<=Q;q++){const z=2*q/Q-1;
    for(const p of profile){const uvula=Math.max(0,Math.min(1,(free[1]-.05-p[1])/.6)),halfWidth=1.15*(1-uvula)+.28*uvula;
      vertices.push(p[0],p[1]+.08*z*z,z*halfWidth);
    }
  }
  for(let q=0;q<Q;q++)for(let j=0;j<N;j++){const a=q*N+j,b=q*N+(j+1)%N,c=(q+1)*N+j,d=(q+1)*N+(j+1)%N;triangles.push(a,c,b,b,c,d);}
  const caps=triangulate(profile);for(let i=0;i<caps.length;i+=3){triangles.push(caps[i],caps[i+1],caps[i+2]);triangles.push(Q*N+caps[i],Q*N+caps[i+2],Q*N+caps[i+1]);}
  return {vertices,triangles,profile,gate,gap,width,wall,wallAt,relief,contact,tip,root,area};
}

// Display smoothing stays inside the control polygon, never changes native areas.
export const smoothContour=points=>smooth(points);

// VTL's four ventral ribs are a sparse construction boundary (including a
// vertical ray to the floor). Keep the 34 dorsal/contact ribs exactly and loft
// a display-only ventral surface back to the native floor attachment.
export function tongueEnvelope(state){
  const native=state.meshes.find(m=>m.name==='tongue'),P=native.points,D=34,Q=24;
  const tip=state.contours.tongue[D-1],floor=state.contours.lower_cover;
  let anchor=7;
  for(let i=7;i<floor.length;i++)if(floor[i][0]<tip[0]-.8&&floor[i][1]<tip[1]-.15)anchor=i;
  const end=floor[anchor],vertices=native.vertices.slice(0,D*P*3),triangles=[];
  const floorMesh=state.meshes.find(m=>m.name==='lower_cover');
  for(let r=1;r<=Q;r++)for(let j=0;j<P;j++){
    const a=native.vertices.slice(((D-1)*P+j)*3,((D-1)*P+j)*3+3),t=r/Q,u=1-t;
    const z=j/(P-1)*(floorMesh.points-1),lo=Math.floor(z),hi=Math.ceil(z),f=z-lo;
    const b=[0,1,2].map(k=>floorMesh.vertices[(anchor*floorMesh.points+lo)*3+k]*(1-f)+floorMesh.vertices[(anchor*floorMesh.points+hi)*3+k]*f);
    // Smooth turn under the tip, then a shallow tangent into the mouth floor.
    const c=[a[0]-.14,a[1]-.36,a[2]],d=[b[0]+Math.max(.12,(a[0]-b[0])*.26),b[1]+.08,b[2]];
    for(let k=0;k<3;k++)vertices.push(u*u*u*a[k]+3*u*u*t*c[k]+3*u*t*t*d[k]+t*t*t*b[k]);
  }
  const R=D+Q;
  for(let r=1;r<R;r++)for(let j=0;j<P-1;j++){const a=(r-1)*P+j,b=a+1,c=r*P+j,d=c+1;triangles.push(a,b,c,b,d,c);}
  const contour=Array.from({length:R},(_,r)=>vertices.slice((r*P+Math.floor(P/2))*3,(r*P+Math.floor(P/2))*3+2));
  return {mesh:{...native,vertices,triangles,ribs:R,dynamicRibs:D},outline:[...contour,...floor.slice(4,anchor).toReversed()],contour,anchor,end};
}
