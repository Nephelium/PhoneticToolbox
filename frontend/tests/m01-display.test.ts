import {test} from 'node:test';import assert from 'node:assert/strict';
import {envelope,peakIndex} from '../src/platform/wav.ts';
test('long-wave display preserves extrema, bounds vertices, and never mutates samples',()=>{
 const samples=new Float32Array(2_000_000);samples[731_117]=.875;samples[731_118]=-.75;const index=peakIndex(samples);
 const points=envelope(samples,0,samples.length,480,index);
 assert(points.length<=480);assert.equal(Math.max(...points.map(p=>p[1])),.875);assert.equal(Math.min(...points.map(p=>p[0])),-.75);
 assert.equal(samples[731_117],.875);assert(index.min.length<samples.length/100);
 for(const [start,end,bins] of [[731100,731200,20],[100,9218,71],[254,8193,11]])assert.deepEqual(envelope(samples,start,end,bins,index),envelope(samples,start,end,bins));
});
