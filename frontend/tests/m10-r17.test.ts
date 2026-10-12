import test from 'node:test';
import assert from 'node:assert/strict';
import {zoomMonitor,monitorRange,monitorTimeLabel,waveformLimit} from '../public/vocal-tract/monitor-view.mjs';
import {constrainVentralLine,constrainVentralMesh} from '../public/vocal-tract/floor-contact.mjs';

test('live monitor zoom keeps the pointed time and follows subsequent audio',()=>{
  const original={span:3,lag:0},before=monitorRange(original,48),fraction=.3;
  const view=zoomMonitor(original,-120,fraction),after=monitorRange(view,48);
  assert.ok(view.span<3);
  assert.ok(Math.abs(before.start+3*fraction-after.start-view.span*fraction)<1e-12);
  assert.equal(monitorRange(view,49).start,after.start+1);
  const restored=zoomMonitor(view,120,fraction);
  assert.ok(Math.abs(restored.span-3)<1e-12&&Math.abs(restored.lag)<1e-12);
});

function flatFloor(){
  const vertices:number[]=[],triangles:number[]=[];
  for(let r=0;r<7;r++)for(const z of [-2,0,2])vertices.push(Math.max(0,Math.min(2,(r-4)*2)),-1,z);
  for(let j=0;j<2;j++){const a=12+j,b=a+1,c=15+j,d=c+1;triangles.push(a,b,c,b,d,c);}
  return {vertices,triangles,ribs:7,points:3};
}
const lowerTooth={vertices:[-.5,1,-.5,.5,1,-.5,-.5,1,.5,.5,1,.5],triangles:[0,1,2,1,3,2]};

test('lower incisor support continues forward without a descending step',()=>{
  const line=constrainVentralLine([[1.2,-.4,0],[-.2,-.4,0]],flatFloor(),lowerTooth);
  const boundary=line.filter((p:number[])=>Math.abs(p[0]-.5)<1e-9);
  assert.ok(boundary.length>0);
  assert.ok(boundary.some((p:number[])=>Math.abs(p[1]-1)<1e-9));
  assert.ok(line.every((p:number[])=>Math.abs(p[1]-1)<1e-9));
  assert.deepEqual(constrainVentralLine([[1.2,1.4,0]],flatFloor(),lowerTooth),[[1.2,1.4,0]]);
});

test('forward crown support follows jaw height and stays within its transverse span',()=>{
  const moved={...lowerTooth,vertices:lowerTooth.vertices.map((v,i)=>i%3===0?v+3:i%3===1?v-2:v)};
  const low=flatFloor();low.vertices=low.vertices.map((v,i)=>i%3===1?v-2:v);
  const line=constrainVentralLine([[4.5,-2,0],[3,-2,0]],low,moved);
  assert.ok(line.every((p:number[])=>Math.abs(p[1]+1)<1e-9));
  assert.deepEqual(constrainVentralLine([[4.5,-2,1]],low,moved),[[4.5,-2,1]]);
  const mesh=constrainVentralMesh({vertices:[4,-2,-.2,5,-2,-.2,4.5,-2,.2],triangles:[0,1,2]},low,0,moved);
  assert.ok(mesh.triangles.length>0);
  assert.ok(mesh.triangles.every((i:number)=>Math.abs(mesh.vertices[i*3+1]+1)<1e-9));
});

test('jaw contact checks face interiors and preserves input geometry',()=>{
  const mesh={vertices:[-1,.5,-1,1,.5,-1,0,.5,1],triangles:[0,1,2]},copy=structuredClone(mesh);
  const result=constrainVentralMesh(mesh,flatFloor(),0,lowerTooth);
  assert.deepEqual(mesh,copy);assert.ok(result.triangles.length>3);
  for(let i=0;i<result.triangles.length;i+=3){const ps=result.triangles.slice(i,i+3).map((j:number)=>result.vertices.slice(j*3,j*3+3));
    assert.ok(ps.every((p:number[])=>p[0]<=-.5+1e-8)||ps.every((p:number[])=>p[0]>=.5-1e-8)||ps.every((p:number[])=>p[2]<=-.5+1e-8)||ps.every((p:number[])=>p[2]>=.5-1e-8)||ps.every((p:number[])=>p[1]>=1-1e-8));
  }
});
test('monitor zoom stays inside available history and remains finite',()=>{
  for(const direction of [-10000,10000]){
    let view={span:3,lag:0};for(let i=0;i<100;i++)view=zoomMonitor(view,direction,.5);
    assert.ok(view.span>=.001&&view.span<=12&&view.lag>=0&&view.span+view.lag<=12);
    const r=monitorRange(view,0);assert.ok(r.start>=0&&Number.isFinite(r.end));
  }
});
test('millisecond zoom has distinct tick labels and an honest weak-signal scale',()=>{
  for(const span of [.1,.02,.001]){
    const ticks=Array.from({length:5},(_,i)=>monitorTimeLabel(48+span*i/4,span));
    assert.equal(new Set(ticks).size,5);
  }
  assert.equal(waveformLimit(.004),.005);
  assert.equal(waveformLimit(.004,false),1);
  assert.equal(waveformLimit(0),1);
  assert.equal(waveformLimit(NaN),1);
});
