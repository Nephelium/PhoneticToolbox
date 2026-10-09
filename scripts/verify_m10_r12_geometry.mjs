import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {displayLips} from '../frontend/public/vocal-tract/geometry.mjs';
import {velumModel} from '../frontend/public/vocal-tract/anatomy.mjs';
const root=new URL(process.argv[2]??'../output/validation/m10-r12/',import.meta.url);
const cases=JSON.parse(readFileSync(new URL('current.json',root),'utf8'));
const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
const weights=[[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[.5,0,.5],[0,.5,.5],[1/3,1/3,1/3]];
const rows=[];
for(const {name,state} of cases){
  const original=JSON.stringify(state),start=performance.now(),shown=displayLips(state),duration=performance.now()-start;
  const tongue=shown.meshes.find(m=>m.name==='tongue'),native=state.meshes.find(m=>m.name==='tongue');
  assert.deepEqual(tongue.vertices.slice(0,34*tongue.points*3),native.vertices.slice(0,34*native.points*3),name+' native dorsal points');
  for(const tooth of ['upper_teeth','lower_teeth'])assert.deepEqual(shown.meshes.find(m=>m.name===tooth),state.meshes.find(m=>m.name===tooth),name+' fixed teeth');
  const ps=shown.tongueOutline;
  for(let i=0;i<ps.length;i++)for(let j=i+2;j<ps.length;j++){
    if(i===0&&j===ps.length-1)continue;
    const [a,b,c,d]=[ps[i],ps[(i+1)%ps.length],ps[j],ps[(j+1)%ps.length]];
    assert.ok(!(cross(a,b,c)*cross(a,b,d)<-1e-10&&cross(c,d,a)*cross(c,d,b)<-1e-10),`${name} self intersection ${i},${j}`);
  }
  let samples=0;
  for(let i=0;i<tongue.triangles.length;i+=3){
    const ids=tongue.triangles.slice(i,i+3);if(ids.every(j=>j<34*tongue.points))continue;
    const pts=ids.map(j=>tongue.vertices.slice(j*3,j*3+3));
    for(const w of weights){
      const p=[0,1,2].map(k=>w.reduce((s,v,j)=>s+v*pts[j][k],0));samples++;
      assert.ok(!shown.dentalSolids.some(d=>d.planes.every(q=>q.n.reduce((s,n,k)=>s+n*p[k],0)-q.d< -1e-6)),name+' tongue inside rigid tooth');
    }
  }
  const velum=velumModel(shown);
  assert.deepEqual(velum.exposedProfile,velum.profile.slice(0,11));
  assert.equal(JSON.stringify(state),original,name+' immutable native snapshot');
  rows.push({name,samples,duration_ms:duration,triangles:tongue.triangles.length/3});
}
const baseline=JSON.parse(readFileSync(new URL('baseline.json',root),'utf8')),comparisons=[];
for(const name of ['a','i','u','n']){
  const before=baseline.find(c=>c.name===name+'-None-0').state,after=cases.find(c=>c.name===name).state;
  const max=(a,b)=>Math.max(...a.map((x,i)=>Math.abs(x-b[i])));
  comparisons.push({name,parameters:max(before.limited,after.limited),areas:max(before.tube_areas,after.tube_areas),teeth:max(before.meshes[3].vertices,after.meshes[3].vertices)});
}
const report={success:true,poses:rows.length,samples:rows.reduce((s,r)=>s+r.samples,0),mean_display_ms:rows.reduce((s,r)=>s+r.duration_ms,0)/rows.length,comparisons,rows};
writeFileSync(new URL('geometry-report.json',root),JSON.stringify(report,null,2));console.log(JSON.stringify({...report,rows:undefined}));
