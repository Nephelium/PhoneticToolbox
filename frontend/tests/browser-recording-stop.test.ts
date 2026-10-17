import test from 'node:test';
import assert from 'node:assert/strict';
import {browserRecordingPort} from '../src/platform/browser-recording.ts';
import {recordingDefaults} from '../src/modules/recording/defaults.ts';

test('stop reaches the audio device while a server spectrum is pending, then finalizes the project in order',async()=>{
 const originals=new Map(['window','navigator','AudioContext','AudioWorkletNode','document'].map(k=>[k,Object.getOwnPropertyDescriptor(globalThis,k)]));
 let worklet:any,finishSpectrum!:()=>void,spectrumStarted!:()=>void,deviceStops=0;
 const started=new Promise<void>(r=>spectrumStarted=r),pending=new Promise<void>(r=>finishSpectrum=r);
 class AudioStub{sampleRate=48000;state='running';destination={};audioWorklet={addModule:async()=>{}};async resume(){}async close(){this.state='closed';}createMediaStreamSource(){return {connect(){},disconnect(){}};}}
 class WorkletStub{port={onmessage:null as any,postMessage:(message:string)=>{if(message==='stop'){deviceStops++;this.port.onmessage?.({data:{done:true}});}}};constructor(){worklet=this;}connect(){}disconnect(){}}
 const track={label:'synthetic',getSettings:()=>({channelCount:2}),stop(){},onended:null};
 const globals={window:{isSecureContext:true},navigator:{mediaDevices:{getUserMedia:async()=>({getAudioTracks:()=>[track],getTracks:()=>[track]})}},AudioContext:AudioStub,AudioWorkletNode:WorkletStub,document:{baseURI:'https://example.test/app/'}};
 for(const [k,value] of Object.entries(globals))Object.defineProperty(globalThis,k,{configurable:true,value});
 const server:any={load:async()=>null,live:async()=>{spectrumStarted();await pending;return {};},upload:async(file:File)=>({id:crypto.randomUUID(),sha256:'0'.repeat(64),size_bytes:file.size}),save:async()=>crypto.randomUUID(),remove:async()=>{}};
 const port=browserRecordingPort({label:'owned synthetic',files:{recording:server}} as any);
 try{
  await port.request({op:'start',config:{...recordingDefaults(),device:'default'}});
  worklet.port.onmessage({data:{block:[new Float32Array([.1,.2]),new Float32Array([.3,.4])]}});
  const status=port.request({op:'status',spectrum:true});await started;
  const first=await Promise.race([status,new Promise((_,reject)=>setTimeout(()=>reject(Error('spectrum blocked status')),200))]);assert(first);
  const second:any=await port.request({op:'status',spectrum:true});assert.equal(second.frames,2);
  const stopped=port.request<any>({op:'stop'});assert.equal(deviceStops,1);
  const result=await Promise.race([stopped,new Promise<any>((_,reject)=>setTimeout(()=>reject(Error('spectrum blocked stop')),200))]);
  finishSpectrum();await status;
  assert.equal(result.frames,2);assert.equal(result.project.takes[0].frames,2);assert.equal(result.error,'');
 }finally{finishSpectrum();port.dispose?.();for(const [k,descriptor] of originals)if(descriptor)Object.defineProperty(globalThis,k,descriptor);else Reflect.deleteProperty(globalThis,k);}
});
