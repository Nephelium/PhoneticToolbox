// The display reconstruction has one 3-D authority; sagittal lines are slices.
// Native VTL defines the oral surface. MRI informs the dorsal tissue envelope.
export function registerNose(source){
  const vertices=source.vertices.map((v,i)=>i%3===1?v+1.0+.7*Math.max(0,Math.min(1,(1-source.vertices[i-1])/3)):v);
  const p=source.landmarks.nasopharynx.reference_cm;
  return {...source,vertices,landmarks:{...source.landmarks,nasopharynx:{...source.landmarks.nasopharynx,reference_cm:[p[0],p[1]+1.0+.7*Math.max(0,Math.min(1,(1-p[0])/3)),p[2]]}},registration:'palatal clearance + posterior nasal outlet; display reference'};
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
