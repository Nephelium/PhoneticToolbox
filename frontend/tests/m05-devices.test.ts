import test from 'node:test';
import assert from 'node:assert/strict';
import {audioDevicesHidden,readCaptureDevices,deviceAccessError} from '../src/modules/lip-extraction/devices.ts';

const device=(deviceId:string,label:string,kind='audioinput')=>({deviceId,label,kind} as MediaDeviceInfo);
const hidden=[device('','')];
const visible=[device('camera','Camera','videoinput'),device('mic','麦克风阵列 (Realtek)'),device('voice','Voice (Steinberg UR22C)'),device('streaming','Streaming (Steinberg UR22C)'),device('mix','立体声混音'),device('line','Input 1/2 (Steinberg UR22C)'),device('speaker','Speaker','audiooutput')];

test('M05 devices: mounting and ordinary refresh never request permission or open audio',async()=>{
 const media={enumerateDevices:async()=>hidden,getUserMedia:async()=>{throw Error('Unexpected capture');}};
 assert(audioDevicesHidden(hidden));assert(audioDevicesHidden([]));assert(!audioDevicesHidden(visible));
 assert.equal(await readCaptureDevices(media,{prepareAudio:async()=>{throw Error('Unexpected permission');}}),hidden);
});

test('M05 devices: explicit authorisation enumerates while audio is active and immediately stops every track',async()=>{
 const calls:string[]=[];let active=false;
 const media={enumerateDevices:async()=>{calls.push('enumerate');return active?visible:hidden;},getUserMedia:async(constraints:MediaStreamConstraints)=>{
  assert.deepEqual(constraints,{audio:true,video:false});calls.push('audio');active=true;
  return {getTracks:()=>[1,2].map(i=>({stop:()=>{calls.push('stop'+i);active=false;}}))} as MediaStream;
 }};
 assert.equal(await readCaptureDevices(media,{authorizeAudio:true,prepareAudio:async()=>{calls.push('prepare');}}),visible);
 assert.deepEqual(calls,['enumerate','prepare','audio','enumerate','stop1','stop2']);assert.equal(active,false);
 // Refresh of the now-authorised list preserves native browser IDs and opens nothing.
 media.enumerateDevices=async()=>visible;
 assert.equal(await readCaptureDevices(media,{authorizeAudio:true,prepareAudio:async()=>{throw Error('Unexpected re-arm');}}),visible);
 assert.equal(calls.length,6);
});

test('M05 devices: failed enumeration and late cancellation still release the temporary stream',async()=>{
 for(const abort of [false,true]){
  let stopped=0,count=0;const controller=new AbortController();
  const media={enumerateDevices:async()=>{if(count++)throw Error('Enumeration unavailable');return hidden;},getUserMedia:async()=>{
   if(abort)controller.abort();return {getTracks:()=>[{stop:()=>stopped++}]} as MediaStream;
  }};
  await assert.rejects(readCaptureDevices(media,{authorizeAudio:true,signal:controller.signal}),abort?{name:'AbortError'}:/Enumeration unavailable/);
  assert.equal(stopped,1);
 }
});

test('M05 devices: cancelled or refused preparation cannot open a microphone',async()=>{
 const controller=new AbortController();controller.abort();
 const media={enumerateDevices:async()=>hidden,getUserMedia:async()=>{throw Error('Unexpected capture');}};
 await assert.rejects(readCaptureDevices(media,{authorizeAudio:true,signal:controller.signal}),{name:'AbortError'});
 await assert.rejects(readCaptureDevices(media,{authorizeAudio:true,prepareAudio:async()=>{throw Error('Recorder owns input');}}),/Recorder owns input/);
});

test('M05 devices: permission rejection is retryable and has an actionable audio-specific message',async()=>{
 let attempts=0;
 const media={enumerateDevices:async()=>hidden,getUserMedia:async()=>{attempts++;throw new DOMException('Denied','NotAllowedError');}};
 for(let i=0;i<2;i++)await assert.rejects(readCaptureDevices(media,{authorizeAudio:true}),e=>deviceAccessError(e).includes('麦克风权限'));
 assert.equal(attempts,2);
 assert.match(deviceAccessError(new DOMException('Missing','NotFoundError')),/连接麦克风或声卡/);
 assert.match(deviceAccessError(new DOMException('Busy','NotReadableError')),/其他程序占用/);
 assert.match(deviceAccessError(new DOMException('Invalid state','AbortError')),/重新点击授权并刷新设备/);
});
