import {test} from 'node:test';
import assert from 'node:assert/strict';
import {floatWave} from '../src/platform/recording-wav.ts';
test('browser PCM WAV preserves interleaving, rate and unclipped float samples',()=>{
 const bytes=floatWave([[new Float32Array([.25,-.5]),new Float32Array([1.2,-1.2])],[new Float32Array([0]),new Float32Array([.75])]],48000,2),v=new DataView(bytes);
 assert.equal(bytes.byteLength,68);assert.equal(v.getUint16(20,true),3);assert.equal(v.getUint16(22,true),2);assert.equal(v.getUint32(24,true),48000);assert.equal(v.getUint32(40,true),24);
 assert.deepEqual(Array.from({length:6},(_,i)=>v.getFloat32(44+4*i,true)),Array.from(new Float32Array([.25,1.2,-.5,-1.2,0,.75])));
});
test('invalid recording dimensions and nonfinite samples fail before upload',()=>{
 assert.throws(()=>floatWave([],48000,1));assert.throws(()=>floatWave([[new Float32Array([NaN])]],48000,1));
 assert.throws(()=>floatWave([[new Float32Array(1),new Float32Array(2)]],48000,2));assert.throws(()=>floatWave([[new Float32Array(1)]],0,1));
});
import {recordingMeter} from '../src/platform/browser-recording.ts';
test('browser input meters retain clipping/DC thresholds and stereo wiring correlation',()=>{
 const x=new Float32Array([-.5,0,.5,1]),y=new Float32Array([1,0,-1,-2]),meter=recordingMeter([[x,y]],6,true);
 assert.equal(meter.correlation,-1);assert.deepEqual(meter.raw_clip,[1,3]);assert.deepEqual(meter.digital_clip,[1,3]);assert.deepEqual(meter.near_full_scale,[1,3]);assert.equal(meter.dc[0],.25);assert.equal(recordingMeter([[new Float32Array(4),new Float32Array(4)]],0).correlation,null);
});
