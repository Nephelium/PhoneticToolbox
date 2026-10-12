import fs from 'node:fs';
import assert from 'node:assert/strict';
import {tongueEnvelope,velumModel} from '../frontend/public/vocal-tract/anatomy.mjs';
import {floorCells,constrainVentralLine} from '../frontend/public/vocal-tract/floor-contact.mjs';
import {splitPolygon} from '../frontend/public/vocal-tract/rigid-contact.mjs';
import {displayLips} from '../frontend/public/vocal-tract/geometry.mjs';
const states=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),rows=[];
const dot=(n,p)=>n.reduce((s,v,k)=>s+v*p[k],0);
const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
const crosses=(a,b,c,d)=>cross(a,b,c)*cross(a,b,d)<-1e-10&&cross(c,d,a)*cross(c,d,b)<-1e-10;
const inPolygon=(p,poly)=>{let yes=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const a=poly[i],b=poly[j],length=Math.hypot(b[0]-a[0],b[1]-a[1]);if(length>0&&Math.abs(cross(a,b,p))/length<1e-7&&p[0]>=Math.min(a[0],b[0])-1e-7&&p[0]<=Math.max(a[0],b[0])+1e-7&&p[1]>=Math.min(a[1],b[1])-1e-7&&p[1]<=Math.max(a[1],b[1])+1e-7)return true;if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])yes=!yes;}return yes;};
for(const [name,s] of Object.entries(states)){
  const start=performance.now(),env=tongueEnvelope(s),ms=performance.now()-start;
  const floor=s.meshes.find(m=>m.name==='lower_cover'),teeth=s.meshes.find(m=>m.name==='lower_teeth'),cells=floorCells(floor,teeth),m=env.mesh,P=m.points,D=34;
  const native=s.meshes.find(m=>m.name==='tongue'),at=(m,i)=>m.vertices.slice(i*3,i*3+3);
  assert.deepEqual(m.vertices.slice(0,D*P*3),native.vertices.slice(0,D*P*3),name+' native dorsal vertices unchanged');
  let checked=0;
  for(let i=0;i<m.triangles.length;i+=3){
    const ids=m.triangles.slice(i,i+3);if(ids.every(j=>j<D*P))continue;
    const tri=ids.map(j=>at(m,j));
    for(const cell of cells){
      if([0,2].some(k=>Math.max(...tri.map(p=>p[k]))<cell.box[k][0]-1e-8||Math.min(...tri.map(p=>p[k]))>cell.box[k][1]+1e-8))continue;
      let poly=tri;
      for(const plane of cell.edges){poly=splitPolygon(poly,plane,1e-12).inside;if(poly.length<3)break;}
      if(poly.length<3)continue;
      const footprint=Math.abs(poly.reduce((sum,p,i)=>sum+p[0]*poly[(i+1)%poly.length][2]-p[2]*poly[(i+1)%poly.length][0],0));
      // A zero-area shared cell edge is contact, not penetration into a cell.
      if(footprint<1e-12)continue;
      // At every overlap corner the linear difference to the native floor
      // must be nonnegative. This also bounds the entire triangle interior.
      for(const p of poly)assert.ok(dot(cell.top.n,p)-cell.top.d>=-2e-7,name+' underside penetrates jaw: '+JSON.stringify({poly,area:poly.reduce((s,p,i)=>s+p[0]*poly[(i+1)%poly.length][2]-p[2]*poly[(i+1)%poly.length][0],0),p,gap:dot(cell.top.n,p)-cell.top.d,cell}));
      checked++;
    }
  }
  for(let j=0;j<P;j++){
    const original=Array.from({length:m.ribs},(_,r)=>{const p=at(m,r*P+j);if(j===Math.floor(P/2))p[2]=0;return p;});
    const ventral=constrainVentralLine(original.slice(D-1),floor,teeth);
    // Only the drawn midline is projected into x/y. Lateral material lines
    // change z and can overlap in this projection without a 3-D intersection.
    if(j===Math.floor(P/2))for(let r=1;r<ventral.length;r++)for(let k=1;k<D-1;k++){
      const a=ventral[r-1],b=ventral[r],c=original[k-1],d=original[k];if(!crosses(a,b,c,d))continue;
      const da=cross(c,d,a),db=cross(c,d,b),t=da/(da-db),hit=a.map((v,k)=>v+t*(b[k]-v));
      // The complete 3-D tongue is now subtracted from both tooth solids and
      // the SVG uses their same sections. Intersections inside a tooth are
      // removed material, not a self intersection of the visible tongue.
      assert.ok(['upper_teeth','lower_teeth'].some(n=>inPolygon(hit,s.contours[n])),name+' visible underside crosses dorsum '+JSON.stringify(hit));
    }
    // The fixed reference is the midline point. Lateral soft tissue can meet
    // the teeth above the floor rather than being pinned inside a tooth.
    if(j===Math.floor(P/2))assert.ok(Math.hypot(...ventral.at(-1).map((v,k)=>v-original.at(-1)[k]))<2e-7,name+' fixed attachment moved');
  }
  const displayed=displayLips(s),section=displayLips(s,{surface:false}),tongue=displayed.meshes.find(m=>m.name==='tongue');let dentalChecks=0;
  assert.deepEqual(section.tongueOutline,displayed.tongueOutline,name+' lazy surface changes 2-D outline');
  const sectionPalate=velumModel(section),surfacePalate=velumModel(displayed);
  for(const key of ['vertices','triangles','profile','exposedProfile','gate','area'])assert.deepEqual(sectionPalate[key],surfacePalate[key],name+' lazy surface changes palate '+key);
  for(let i=0;i<tongue.triangles.length;i+=3){const ps=tongue.triangles.slice(i,i+3).map(j=>at(tongue,j));
    for(const solid of displayed.dentalSolids){
      if([0,1,2].some(k=>Math.max(...ps.map(p=>p[k]))<solid.box[k][0]-1e-8||Math.min(...ps.map(p=>p[k]))>solid.box[k][1]+1e-8))continue;
      assert.ok(solid.planes.some(p=>ps.every(v=>dot(p.n,v)>=p.d-2e-7)),name+' displayed tongue enters '+solid.name);dentalChecks++;
    }
  }
  rows.push({name,ms,checked,dentalChecks,faces:m.triangles.length/3,anchor:env.end});
}
fs.writeFileSync(process.argv[3],JSON.stringify({passed:true,rows},null,2));
console.log(JSON.stringify({passed:true,poses:rows.length,max_ms:Math.max(...rows.map(r=>r.ms))}));
