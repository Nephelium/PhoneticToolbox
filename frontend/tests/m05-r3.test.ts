import test from 'node:test';
import assert from 'node:assert/strict';
import {checkedOffsetMs,normalizedParameter,waveformEnvelope} from '../src/modules/lip-extraction/alignment.ts';

test('M05-R3 offset is bounded, signed and rounded to a millisecond',()=>{
 assert.equal(checkedOffsetMs(-123),-.123);assert.equal(checkedOffsetMs(2.3),.002);
 for(const invalid of [NaN,Infinity,2001,-2001])assert.throws(()=>checkedOffsetMs(invalid));
});
test('M05-R3 display normalization preserves gaps and original values; positive offset moves right',()=>{
 const rows=[0,1,2,3].map((time_s,i)=>({index:i,time_s,detected:true,points:null,metrics:{open:[10,null,30,20][i]}}));
 const copy=structuredClone(rows),curve=normalizedParameter(rows,'open',.15);
 assert.deepEqual(curve.times,[.15,1.15,2.15,3.15]);assert.deepEqual(curve.values,[0,null,1,.5]);assert.deepEqual(rows,copy);
});
test('M05-R3 waveform peak bins retain impulses without cancelling stereo',()=>{
 const data=Float32Array.from([0,-.8,0,.4,0,0,.9,0]);
 const wave=waveformEnvelope({length:8,sampleRate:8,duration:1,numberOfChannels:2,getChannelData:()=>data} as unknown as AudioBuffer,2);
 assert.deepEqual(wave.values,[-.800000011920929,.4000000059604645,0,.8999999761581421]);assert.deepEqual(wave.times,[.125,.375,.5,.75]);
});
