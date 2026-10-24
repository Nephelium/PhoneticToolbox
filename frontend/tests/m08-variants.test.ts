import test from 'node:test';
import assert from 'node:assert/strict';
import {variants,shiftedF0,variantLabel} from '../src/modules/pitch-manipulation/state.ts';
test('multiple speeds cross every selected pitch, deduplicating equivalent values',()=>{
 const out=variants({speed:'0.8，1;1.0 1.2',pitch_ratio:'1,1.25',pitch_hz:'incomplete -'},'ratio');
 assert.equal(out.length,6);assert.deepEqual(out.map(v=>[v.speed,v.pitch_ratio,v.pitch_hz]),[[.8,1,0],[.8,1.25,0],[1,1,0],[1,1.25,0],[1.2,1,0],[1.2,1.25,0]]);
 assert.deepEqual(variants({speed:'1',pitch_ratio:'-',pitch_hz:'-20 0 +20'},'hz').map(v=>v.pitch_ratio),[1,1,1]);
});
test('variant validation rejects incomplete, nonfinite, nonpositive or excessive choices',()=>{
 for(const speed of ['', '-', '0', '-1', 'Infinity', '1 2x'])assert.throws(()=>variants({speed,pitch_ratio:'1',pitch_hz:'0'},'ratio'));
 assert.throws(()=>variants({speed:'1',pitch_ratio:'0',pitch_hz:'0'},'ratio'));
 assert.throws(()=>variants({speed:Array.from({length:257},(_,i)=>String(i+1)).join(','),pitch_ratio:'1',pitch_hz:'0'},'ratio'));
});
test('single-file pitch operates on voiced edited F0 and preserves gaps and original array',()=>{
 const curve=[100,0,120];assert.deepEqual(shiftedF0(curve,1.2,0),[120,0,144]);assert.deepEqual(shiftedF0(curve,1,-20),[80,0,100]);assert.deepEqual(curve,[100,0,120]);assert.throws(()=>shiftedF0(curve,1,-100));
 assert.deepEqual(shiftedF0([50,150,50],1,-100,{times:[0,.5,1],start:.4,end:.6}),[50,50,50]);
 assert.match(variantLabel({speed:.8,pitch_ratio:1,pitch_hz:-20}),/语速 0.8.*-20 Hz/);
});
