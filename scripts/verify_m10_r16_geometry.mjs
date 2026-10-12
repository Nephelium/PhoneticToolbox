import fs from 'node:fs';
import assert from 'node:assert/strict';
import {displayLips,oralMesh,connectedAirway,nasalOutlet} from '../frontend/public/vocal-tract/geometry.mjs';
import {registerNose,tongueEnvelope} from '../frontend/public/vocal-tract/anatomy.mjs';
const states=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const nose=registerNose(JSON.parse(fs.readFileSync(new URL('../frontend/public/vocal-tract/assets/nasal.json',import.meta.url),'utf8'))),outlet=nasalOutlet(nose);
const rows=[];
for(const [name,source] of Object.entries(states)){
  const start=performance.now(),s=displayLips(source),oral=oralMesh(s),air=connectedAirway(s,nose,outlet);
  const env=tongueEnvelope(source),orient=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const crosses=(a,b,c,d)=>orient(a,b,c)*orient(a,b,d)<-1e-10&&orient(c,d,a)*orient(c,d,b)<-1e-10;
  // The new fixed floor attachment must not send the ventral return through
  // the native dorsal tongue, including the anterior curled tongue columns.
  for(let j=0;j<env.mesh.points;j++){
    const p=r=>env.mesh.vertices.slice((r*env.mesh.points+j)*3,(r*env.mesh.points+j)*3+2);
    for(let r=35;r<env.mesh.ribs;r++)for(let k=1;k<33;k++)assert.ok(!crosses(p(r-1),p(r),p(k-1),p(k)),name+' ventral self intersection');
  }
  const teethBox=[0,1,2].map(k=>[Math.min(...s.dentalSolids.map(t=>t.box[k][0])),Math.max(...s.dentalSolids.map(t=>t.box[k][1]))]);
  for(const m of [oral,air]){
    assert.ok(m.vertices.every(Number.isFinite),name);
    assert.ok(m.triangles.every(i=>Number.isInteger(i)&&i>=0&&i<m.vertices.length/3),name);
    // Every triangle lies outside each convex tooth solid, including interiors.
    for(let i=0;i<m.triangles.length;i+=3){
      const ps=m.triangles.slice(i,i+3).map(k=>m.vertices.slice(k*3,k*3+3));
      const box=[0,1,2].map(k=>[Math.min(...ps.map(p=>p[k])),Math.max(...ps.map(p=>p[k]))]);
      if(box.some((b,k)=>b[1]<teethBox[k][0]-1e-8||b[0]>teethBox[k][1]+1e-8))continue;
      for(const solid of s.dentalSolids){
        if(solid.box.some((b,k)=>box[k][1]<b[0]-1e-8||box[k][0]>b[1]+1e-8))continue;
        assert.ok(solid.planes.some(p=>ps.every(v=>v.reduce((sum,x,k)=>sum+x*p.n[k],0)>=p.d-2e-7)),name+' dental penetration');
      }
    }
  }
  const anchor=tongueEnvelope(source).end;
  rows.push({name,anchor,oralTriangles:oral.triangles.length/3,connectedTriangles:air.triangles.length/3,total_ms:performance.now()-start});
}
const anchors=rows.filter(r=>r.name.startsWith('anchor-'));
assert.ok(anchors.length>=3);for(const r of anchors)assert.deepEqual(r.anchor,anchors[0].anchor);
fs.writeFileSync(process.argv[3],JSON.stringify({passed:true,rows},null,2));
console.log(JSON.stringify({poses:rows.length,passed:true}));
