import test from 'node:test';
import assert from 'node:assert/strict';
import {preferredMicrophone,isLoopback,signalDb} from '../src/modules/lip-extraction/audio-input.ts';
import {faceBounds,fitFace} from '../src/modules/lip-extraction/overlay.ts';
import type {Point} from '../src/modules/lip-extraction/metrics.ts';

test('M05-R2 automatic input prefers a physical microphone over default stereo mix and aliases',()=>{
 const device=(deviceId:string,label:string,kind='audioinput')=>({deviceId,label,kind} as MediaDeviceInfo);
 const inputs=[device('default','默认 - 立体声混音'),device('mix','立体声混音 (Realtek)'),device('communications','通讯 - Microphone USB'),device('usb','Microphone USB'),device('speaker','Microphone fake output','audiooutput')];
 assert.equal(preferredMicrophone(inputs)?.deviceId,'usb');
 assert.equal(preferredMicrophone([device('mix','Stereo Mix')]),undefined);
 assert.equal(preferredMicrophone([device('hidden','')]),undefined);
 assert.equal(preferredMicrophone([device('internal','麦克风阵列 (Realtek)')])?.deviceId,'internal');
 assert(isLoopback('Stereo Mix (Realtek)'));assert(!isLoopback('Microphone (USB)'));
});
test('M05-R2 input meter distinguishes silence, very low digital noise and a known signal',()=>{
 assert.equal(signalDb([0,0]).peakDb,-120);
 assert(signalDb([.0003,-.0003]).peakDb<-60);
 const input=[.5,-.5,.5,-.5],copy=[...input],level=signalDb(input);
 assert(Math.abs(level.peakDb+6.020599913)<1e-8);assert.equal(level.peakDb,level.rmsDb);assert.deepEqual(input,copy);
});
test('M05-R2 animation fits the whole face at a fixed scale without changing measurement coordinates',()=>{
 const points:Point[]=[[600,100],[800,100],[800,400],[600,400]],copy=structuredClone(points);
 const later:Point[]=points.map(([x,y])=>[x+10,y+10]);
 const bounds=faceBounds([{points},{points:later}])!;
 const fitted=fitFace(points,1280,720,bounds),second=fitFace(later,1280,720,bounds);
 const size=faceBounds([{points:fitted},{points:second}])!;
 assert(Math.abs(size.bottom-size.top-720*.82)<1e-8);
 assert(Math.abs((size.left+size.right)/2-640)<1e-8);
 assert(Math.abs((size.top+size.bottom)/2-360)<1e-8);
 assert(size.left>0&&size.right<1280&&size.top>0&&size.bottom<720);
 // Subtraction of translated floating-point coordinates rounds independently.
 assert(Math.abs((fitted[1][0]-fitted[0][0])-(second[1][0]-second[0][0]))<=4*Number.EPSILON*1280);
 assert.deepEqual(points,copy);assert.equal(faceBounds([{points:null}]),null);
});
