import test from 'node:test';import assert from 'node:assert/strict';
import {samplePeak,peakGain,AUDITION_PEAK} from '../src/platform/playback-gain.ts';
test('audition gain reaches -1 dBFS without clipping, links channels and leaves source samples untouched',()=>{
 const left=new Float32Array([-.1,.05]),right=new Float32Array([.2,-.02]),before=[...left,...right],peak=samplePeak([left,right]),gain=peakGain(peak);
 assert(Math.abs(peak*gain-AUDITION_PEAK)<1e-12);assert(Math.abs(left[1]!*gain/(left[0]!*gain)+.5)<1e-12);assert.deepEqual([...left,...right],before);
});
test('silence, extreme low level, loud float PCM and disabled playback have bounded gain',()=>{
 assert.equal(peakGain(0),1);assert.equal(peakGain(1e-6),1);assert.equal(peakGain(.001),32);assert.equal(peakGain(.01,false),1);assert.equal(peakGain(Infinity),1);assert(peakGain(2)*2<1);
});
