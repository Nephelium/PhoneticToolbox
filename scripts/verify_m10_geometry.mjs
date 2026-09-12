import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {dirname,join} from 'node:path';
import {displayLips,connectedAirway,point,sectionPaths} from '../frontend/public/vocal-tract/geometry.mjs';
import {registerNose} from '../frontend/public/vocal-tract/anatomy.mjs';
const cases=JSON.parse(readFileSync(process.argv[2],'utf8'));
const nose=registerNose(JSON.parse(readFileSync(new URL('../frontend/public/vocal-tract/assets/nasal.json',import.meta.url),'utf8'))),results=[];
for(const {name,state} of cases){
  const before=JSON.stringify(state),display=displayLips(state),up=display.meshes.find(m=>m.name==='upper_lip'),lo=display.meshes.find(m=>m.name==='lower_lip');
  let minGap=Infinity;
  for(let r=0;r<up.ribs;r++){
    const upper=Array.from({length:up.points},(_,j)=>point(up,r*up.points+j));
    const lower=Array.from({length:lo.points},(_,j)=>point(lo,r*lo.points+j));
    assert.ok(Math.abs(upper[0][2]-lower[0][2])<1e-8,name+' transverse lip correspondence');
    const gap=Math.min(...upper.map(p=>p[1]))-Math.max(...lower.map(p=>p[1]));
    minGap=Math.min(gap,minGap);assert.ok(gap>=-1e-8,name+' upper/lower lip intersection');
  }
  const air=connectedAirway(state,nose),ring=air.junction.throat;
  const gateArea=Math.abs(ring.reduce((sum,p,i)=>{const q=ring[(i+1)%ring.length];return sum+p[0]*q[2]-q[0]*p[2];},0)/2);
  assert.equal(air.junction.open,state.nasal.port_area>0);
  if(air.junction.open)assert.ok(Math.abs(gateArea-state.nasal.port_area)<1e-7,name+' valve opening area');
  assert.ok(sectionPaths(air.connector).length>0,name+' missing sagittal nasopharyngeal connector');
  for(const mesh of [up,lo,air]){
    assert.ok(mesh.vertices.every(Number.isFinite),name+' nonfinite vertex');
    assert.ok(mesh.triangles.every(i=>Number.isInteger(i)&&i>=0&&i<mesh.vertices.length/3),name+' bad triangle index');
  }
  assert.equal(JSON.stringify(state),before,name+' native acoustic geometry mutated');
  results.push({name,lipGapCm:minGap,valveAreaCm2:state.nasal.port_area,connector:true});
}
writeFileSync(join(dirname(process.argv[2]),'result.json'),JSON.stringify({passed:true,cases:results},null,2));
console.log('M10 shared display geometry: '+results.length+' native cases passed');
