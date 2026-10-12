// Loft the native upper/lower sheets over their open domain. A closed strip
// joins the sheets locally; it never becomes a bridge between two air channels.
export function openRuns(section,epsilon=1e-6){
  const runs=[];
  section.upper.forEach((u,j)=>{
    const l=section.lower[j];if(u===null||l===null||u-l<=epsilon)return;
    if(!runs.length||runs.at(-1).at(-1)!==j-1)runs.push([]);
    runs.at(-1).push(j);
  });return runs;
}

export function nativeAirwayMesh(state,first=0){
  const vertices=[],triangles=[],ids=new Map(),pointIds=new WeakMap(),epsilon=1e-6;
  const vertex=p=>{let id=pointIds.get(p);if(id!==undefined)return id;const key=Math.round(p[0]*1e8)+','+Math.round(p[1]*1e8)+','+Math.round(p[2]*1e8);id=ids.get(key);if(id===undefined){id=vertices.length/3;vertices.push(p[0],p[1],p[2]);ids.set(key,id);}pointIds.set(p,id);return id;};
  const face=(a,b,c)=>{
    const ux=b[0]-a[0],uy=b[1]-a[1],uz=b[2]-a[2],vx=c[0]-a[0],vy=c[1]-a[1],vz=c[2]-a[2];
    if(Math.hypot(uy*vz-uz*vy,uz*vx-ux*vz,ux*vy-uy*vx)<1e-12)return;
    triangles.push(vertex(a),vertex(b),vertex(c));
  };
  const rows=state.airway_sections.slice(first).map(sec=>{
    const c=state.centerline[sec.index],valid=sec.upper.map((v,j)=>v!==null&&sec.lower[j]!==null?j:null).filter(j=>j!==null);
    return sec.upper.map((u,j)=>{
      const l=sec.lower[j],ok=u!==null&&l!==null;
      // Outside the sampled wall, close at the nearest valid wall midpoint.
      // Preserve actual contacts inside the profile without interpolating air.
      const nearest=ok?j:valid.reduce((a,b)=>Math.abs(b-j)<Math.abs(a-j)?b:a,valid[0]??j);
      const mid=ok?(u+l)/2:valid.length?(sec.upper[nearest]+sec.lower[nearest])/2:0;
      const gap=ok&&u-l>epsilon?u-l:0;
      const p=d=>[c[0]+c[3]*d,c[1]+c[4]*d,j*7/96-3.5];
      return {gap,top:p(mid+gap/2),bottom:p(mid-gap/2)};
    });
  });
  const cut=(a,b)=>{
    const t=(a.gap-epsilon)/(a.gap-b.gap),mix=k=>a[k].map((v,i)=>v+t*(b[k][i]-v));
    const top=mix('top'),bottom=mix('bottom'),mid=top.map((v,i)=>(v+bottom[i])/2);
    return {gap:0,top:mid,bottom:mid};
  };
  const patch=points=>{
    const poly=[];
    for(let k=0;k<3;k++){
      const a=points[k],b=points[(k+1)%3],inside=a.gap>epsilon,next=b.gap>epsilon;
      if(inside)poly.push(a);if(inside!==next)poly.push(cut(a,b));
    }
    for(let k=1;k<poly.length-1;k++){
      face(poly[0].top,poly[k].top,poly[k+1].top);
      face(poly[0].bottom,poly[k+1].bottom,poly[k].bottom);
    }
  };
  for(let r=1;r<rows.length;r++)for(let j=1;j<96;j++){
    const a=rows[r-1][j-1],b=rows[r-1][j],c=rows[r][j-1],d=rows[r][j];
    patch([a,b,c]);patch([b,d,c]);
  }
  return {vertices,triangles};
}
