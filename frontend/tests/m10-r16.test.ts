import test from 'node:test';
import assert from 'node:assert/strict';
import {nativeAirwayMesh,openRuns} from '../public/vocal-tract/airway-mesh.mjs';
import {dentalSolids,subtractTeeth} from '../public/vocal-tract/rigid-contact.mjs';

const state=(gap:(j:number)=>number)=>({
  centerline:Array.from({length:4},(_,i)=>[i,0,i,0,1]),
  airway_sections:Array.from({length:4},(_,index)=>({index,upper:Array.from({length:96},(_,j)=>gap(j)>0?1:0),lower:Array(96).fill(0)})),
});

function components(m:any){
  const edges=new Map<number,Set<number>>();
  for(let i=0;i<m.triangles.length;i+=3){const tri=m.triangles.slice(i,i+3);for(const a of tri){if(!edges.has(a))edges.set(a,new Set());for(const b of tri)edges.get(a)!.add(b);}}
  const seen=new Set<number>();let count=0;
  for(const a of edges.keys())if(!seen.has(a)){count++;const q=[a];while(q.length){const p=q.pop()!;if(seen.has(p))continue;seen.add(p);q.push(...edges.get(p)!);}}
  return count;
}

test('R16 lateral air remains two disconnected channels at central contact',()=>{
  const s=state(j=>j>=30&&j<=42||j>=54&&j<=66?1:0),copy=structuredClone(s),m=nativeAirwayMesh(s);
  assert.equal(openRuns(s.airway_sections[0]).length,2);
  assert.equal(components(m),2);
  for(let i=0;i<m.triangles.length;i+=3){const zs=m.triangles.slice(i,i+3).map((k:number)=>m.vertices[k*3+2]);assert.ok(Math.max(...zs)<0||Math.min(...zs)>0);}
  assert.deepEqual(s,copy);
});

test('R16 central groove is one channel and complete contact has no air mesh',()=>{
  assert.equal(components(nativeAirwayMesh(state(j=>j>=45&&j<=51?1:0))),1);
  assert.equal(nativeAirwayMesh(state(()=>0)).triangles.length,0);
});

test('R16 channel split and reunion does not span the central closed region',()=>{
  const s=state(j=>j>=30&&j<=66?1:0);
  s.airway_sections[1].upper.fill(0,43,54);s.airway_sections[2].upper.fill(0,43,54);
  const m=nativeAirwayMesh(s);assert.equal(components(m),1);
  for(let i=0;i<m.triangles.length;i+=3){const ps=m.triangles.slice(i,i+3).map((k:number)=>m.vertices.slice(k*3,k*3+3));
    if(ps.every((p:number[])=>p[0]>=1&&p[0]<=2))assert.ok(ps.every((p:number[])=>p[2]<0)||ps.every((p:number[])=>p[2]>0));
  }
});

test('R16 dental subtraction clips triangle interiors even when all vertices are outside',()=>{
  const solid={box:[[-.5,.5],[-.5,.5],[-.5,.5]],planes:[
    {n:[1,0,0],d:.5},{n:[-1,0,0],d:.5},{n:[0,1,0],d:.5},{n:[0,-1,0],d:.5},{n:[0,0,1],d:.5},{n:[0,0,-1],d:.5},
  ]};
  const mesh={vertices:[-2,-2,0,2,-2,0,0,2,0],triangles:[0,1,2]},result=subtractTeeth(mesh,[solid]);
  assert.ok(result.triangles.length>3);
  for(let i=0;i<result.triangles.length;i+=3){const ps=result.triangles.slice(i,i+3).map((k:number)=>result.vertices.slice(k*3,k*3+3));
    assert.ok(solid.planes.some(p=>ps.every((v:number[])=>v.reduce((sum,x,k)=>sum+x*p.n[k],0)>=p.d-1e-7)));
  }
  assert.equal(mesh.triangles.length,3);
});
