import {splitPolygon} from './rigid-contact.mjs';

// The lower cover is the current jaw's inner wall, not the external head skin.
// Project only the reconstructed underside onto this piecewise planar floor.
// Split at the floor's actual triangle edges before projection so a long face
// cannot cut through a ridge between otherwise valid vertices.
const EPS=1e-10;
const dot=(n,p)=>n[0]*p[0]+n[1]*p[1]+n[2]*p[2];
const at=(m,i)=>m.vertices.slice(i*3,i*3+3);
const lerp=(a,b,t)=>a.map((v,k)=>v+(b[k]-v)*t);
let cachedKey='',cachedCells=[];

export function floorCells(mesh,teeth=null){
  const key=JSON.stringify([mesh.vertices,teeth?.vertices]);
  if(key===cachedKey)return cachedCells;
  const cells=[];
  function add(a,b,c){
    const dx=b[0]-a[0],dz=b[2]-a[2],ex=c[0]-a[0],ez=c[2]-a[2],det=dx*ez-dz*ex;
    if(Math.abs(det)<1e-10)return;
    const sx=((b[1]-a[1])*ez-(c[1]-a[1])*dz)/det,sz=(dx*(c[1]-a[1])-ex*(b[1]-a[1]))/det;
    const top={n:[-sx,1,-sz],d:a[1]-sx*a[0]-sz*a[2]},points=[a,b,c],edges=[];
    for(let i=0;i<3;i++){
      const p=points[i],q=points[(i+1)%3],r=points[(i+2)%3];
      let n=[q[2]-p[2],0,p[0]-q[0]],d=dot(n,p);
      if(dot(n,r)>d){n=n.map(v=>-v);d=-d;}
      const length=Math.hypot(...n);edges.push({n:n.map(v=>v/length),d:d/length});
    }
    cells.push({top,edges,points,box:[0,1,2].map(k=>[Math.min(a[k],b[k],c[k]),Math.max(a[k],b[k],c[k])])});
  }
  // Keep the native triangulation. The last rib collapses at the mouth outlet.
  for(let i=0;i<mesh.triangles.length;i+=3){const ids=mesh.triangles.slice(i,i+3);if(Math.min(...ids)>=4*mesh.points)add(...ids.map(j=>at(mesh,j)));}
  const rows=[];
  for(let r=4;r<mesh.ribs-1;r++){
    const row=Array.from({length:mesh.points},(_,j)=>at(mesh,r*mesh.points+j));
    rows.push([[row[0][0],row[0][1],-20],...row,[row.at(-1)[0],row.at(-1)[1],20]]);
  }
  const quad=(a,b,c,d)=>{add(a,b,c);add(b,d,c);};
  // Outside the dental arch continue the rim horizontally. A protruding tip
  // may stay above the outlet, but its return must not descend outside the jaw.
  for(let r=1;r<rows.length;r++)for(const j of [0,rows[r].length-2])quad(rows[r-1][j],rows[r-1][j+1],rows[r][j],rows[r][j+1]);
  for(const [row,x] of [[rows[0],-20],[rows.at(-1),20]])for(let j=1;j<row.length;j++)quad(row[j-1],row[j],[x,row[j-1][1],row[j-1][2]],[x,row[j][1],row[j][2]]);
  // Include the actual moving lower teeth. Taking all faces preserves their
  // upper envelope, including sloping incisor edges, without a visual mask.
  if(teeth)for(let i=0;i<teeth.triangles.length;i+=3){
    const points=teeth.triangles.slice(i,i+3).map(j=>at(teeth,j));add(...points);
    // Carry the crown's support forward at each transverse position. Ending
    // contact at the front tooth wall made the underside fall back down and
    // wrap around that wall. These display-only planes retain the moving
    // crown height and join the protruding underside without that step.
    for(let j=0;j<3;j++){
      const a=points[j],b=points[(j+1)%3],c=points[(j+2)%3];
      if(Math.abs(b[2]-a[2])<EPS)continue;
      const edgeX=a[0]+(c[2]-a[2])*(b[0]-a[0])/(b[2]-a[2]);
      if(c[0]>=edgeX-EPS)continue;
      quad(a,b,[20,a[1],a[2]],[20,b[1],b[2]]);
    }
  }
  cachedKey=key;cachedCells=cells;return cells;
}

function overlap(poly,cell){
  for(const k of [0,2]){let lo=Infinity,hi=-Infinity;for(const p of poly){lo=Math.min(lo,p[k]);hi=Math.max(hi,p[k]);}if(hi<cell.box[k][0]-EPS||lo>cell.box[k][1]+EPS)return false;}
  return true;
}
function conform(poly,cell){
  if(!overlap(poly,cell)||poly.every(p=>dot(cell.top.n,p)>=cell.top.d-EPS))return [poly];
  let pending=poly;const pieces=[];
  for(const edge of cell.edges){
    const split=splitPolygon(pending,edge,1e-12);if(split.outside.length>=3)pieces.push(split.outside);
    pending=split.inside;if(pending.length<3)return [poly];
  }
  const {inside:below,outside:above}=splitPolygon(pending,cell.top,1e-12);
  if(above.length>=3)pieces.push(above);
  if(below.length>=3)pieces.push(below.map(p=>[p[0],cell.top.d-cell.top.n[0]*p[0]-cell.top.n[2]*p[2],p[2]]));
  return pieces;
}

export function constrainVentralMesh(mesh,floor,protectedVertices,teeth=null){
  const cells=floorCells(floor,teeth),vertices=[...mesh.vertices],triangles=[];
  for(let i=0;i<mesh.triangles.length;i+=3){
    const ids=mesh.triangles.slice(i,i+3);
    if(ids.every(j=>j<protectedVertices)){triangles.push(...ids);continue;}
    const original=ids.map(j=>at(mesh,j)),ymin=Math.min(...original.map(p=>p[1]));let pieces=[original];
    for(const cell of cells){if(ymin>=cell.box[1][1]-EPS||!overlap(original,cell))continue;pieces=pieces.flatMap(poly=>conform(poly,cell));}
    if(pieces.length===1&&pieces[0]===original){triangles.push(...ids);continue;}
    for(const poly of pieces)for(let j=1;j<poly.length-1;j++){
      const [a,b,c]=[poly[0],poly[j],poly[j+1]],u=b.map((v,k)=>v-a[k]),v=c.map((v,k)=>v-a[k]);
      if(Math.hypot(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])<EPS)continue;
      const start=vertices.length/3;vertices.push(...a,...b,...c);triangles.push(start,start+1,start+2);
    }
  }
  return {...mesh,vertices,triangles};
}

// The visible midline uses the identical floor planes. Include every crossing
// rather than clamping a few curve samples and bridging over a floor corner.
export function constrainVentralLine(points,floor,teeth=null){
  const cells=floorCells(floor,teeth),out=[];
  if(points.length===1){const p=[...points[0]];for(const cell of cells)if(cell.edges.every(e=>dot(e.n,p)<=e.d+EPS))p[1]=Math.max(p[1],cell.top.d-cell.top.n[0]*p[0]-cell.top.n[2]*p[2]);return [p];}
  for(let i=1;i<points.length;i++){
    const a=points[i-1],b=points[i],cuts=[0,1],active=[];
    for(const cell of cells){
      let lo=0,hi=1;
      for(const edge of cell.edges){
        const da=dot(edge.n,a)-edge.d,db=dot(edge.n,b)-edge.d;
        if(da>EPS&&db>EPS){hi=-1;break;}
        if(da>EPS)lo=Math.max(lo,da/(da-db));else if(db>EPS)hi=Math.min(hi,da/(da-db));
      }
      if(lo>hi||hi<0)continue;
      active.push({cell,lo,hi});cuts.push(lo,hi);
      const da=dot(cell.top.n,a)-cell.top.d,db=dot(cell.top.n,b)-cell.top.d;
      if(da*db<0){const t=da/(da-db);if(t>lo&&t<hi)cuts.push(t);}
    }
    const sorted=[...new Set(cuts)].sort((a,b)=>a-b);
    for(let k=1;k<sorted.length;k++){
      const lo=sorted[k-1],hi=sorted[k];if(hi-lo<1e-12)continue;
      const middle=(lo+hi)/2,local=active.filter(q=>middle>=q.lo&&middle<=q.hi);
      // Include both sides of each cell boundary. The forward crown support
      // makes the anterior contact continuous without bridging floor corners.
      for(const t of [lo,hi]){
        const p=lerp(a,b,t);
        for(const {cell} of local)p[1]=Math.max(p[1],cell.top.d-cell.top.n[0]*p[0]-cell.top.n[2]*p[2]);
        if(!out.length||Math.hypot(...p.map((v,k)=>v-out.at(-1)[k]))>EPS)out.push(p);
      }
    }
  }
  return out;
}
