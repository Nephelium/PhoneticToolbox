import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import zlib from 'node:zlib';
import {lipMetrics,type Point} from '../src/modules/lip-extraction/metrics.ts';
import {LandmarkStabilizer} from '../src/modules/lip-extraction/stabilizer.ts';
import {offsetChoice,verifyLocalBudget,frameRate} from '../src/modules/lip-extraction/state.ts';
import {meshEdges} from '../src/modules/lip-extraction/overlay.ts';
const oracle=JSON.parse(zlib.gunzipSync(fs.readFileSync(new URL('../../tests/fixtures/m05/v2.json.gz',import.meta.url))).toString());
test('M05 complete overlay topology equals independently captured V2 mesh, not metric contour subsets',()=>{
 const expected=oracle.spec.neighbors.flatMap((ns:number[],i:number)=>ns.filter(j=>i<j).map(j=>[i,j]));
 assert.deepEqual(meshEdges,expected);assert(meshEdges.length>1000);
});
function compare(actual:number|null,expected:number|null,label:string){if(expected===null){assert.equal(actual,null,label);return;}assert.notEqual(actual,null,label);assert(Math.abs(actual!-expected)<=Math.max(1e-6,Math.abs(expected)*2e-6),`${label}: ${actual} != ${expected}`);}
test('M05 same input legacy video metrics: frozen arithmetic gate',()=>{
 for(const c of oracle.scientific)for(let i=0;i<c.raw.length;i++)if(c.raw[i]){
  const values=lipMetrics(c.landmarks_full[i]);for(const [key,value] of Object.entries(values))compare(value,c.metrics[key][i],`${c.case}:${i}:${key}`);
 }
});
test('M05 frozen filter dynamics and cross-language points',()=>{
 for(const c of oracle.analytic){const filter=new LandmarkStabilizer(c.cutoff);for(let i=0;i<oracle.input_points.length;i++){
  const actual=filter.filter(oracle.input_points[i] as Point[],i*.043);for(let n=0;n<478;n++)for(let axis=0;axis<2;axis++)compare(actual[n][axis],c.filtered[i][n][axis],`filter ${c.cutoff}:${i}:${n}:${axis}`);
 }}
});
test('M05 offset, missing denominators and local storage budget',()=>{
 assert.equal(offsetChoice('apply',-.05),-.05);assert.equal(offsetChoice('save_without_offset',-.05),0);assert.equal(offsetChoice('cancel',.1),null);
 assert.throws(()=>offsetChoice('apply',NaN));assert.throws(()=>verifyLocalBudget(128_000_000,1));assert.throws(()=>verifyLocalBudget(0,-1));
 const values=lipMetrics(Array.from({length:478},()=>[0,0]));assert.equal(values.area,null);assert.equal(values.open,null);assert.equal(values.open_px,0);
 assert.equal(frameRate(4,0,150),20);assert.equal(frameRate(1,0,0),null);
});
