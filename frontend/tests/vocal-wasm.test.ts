import test from 'node:test';
import assert from 'node:assert/strict';
import {BrowserVocal} from '../src/platform/browser-vocal.ts';
import {webVocalPort as makeVocalPort} from '../src/platform/vocal.ts';
import {LocalAssets,MemoryAssetDatabase} from '../src/platform/local-assets.ts';
import {vocalProfileStore} from '../src/platform/vocal-profile.ts';
const webVocalPort=(...args:Parameters<typeof makeVocalPort>)=>makeVocalPort(args[0],args[1],args[2],args[3],vocalProfileStore(new LocalAssets('test',args[1],new MemoryAssetDatabase())));

const originals=new Map(['window','AudioContext','sessionStorage'].map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
test.before(()=>{
 Object.defineProperty(globalThis,'window',{configurable:true,value:{addEventListener(){},removeEventListener(){}}});
 Object.defineProperty(globalThis,'AudioContext',{configurable:true,value:class {}});
 Object.defineProperty(globalThis,'sessionStorage',{configurable:true,value:{getItem(){return null;},setItem(){},removeItem(){}}});
});
test.after(()=>{for(const [key,descriptor] of originals)if(descriptor)Object.defineProperty(globalThis,key,descriptor);else Reflect.deleteProperty(globalThis,key);});

class FakeWorker {
 messages:any[]=[];terminated=false;onmessage:any;onerror:any;
 postMessage(value:any){this.messages.push(value);}
 terminate(){this.terminated=true;}
 finish(value:any,checkpoint?:any){const message=this.messages.at(-1);this.onmessage({data:{id:message.id,value,checkpoint}});}
}
test('cancel terminates native work, restores acknowledged pose and keeps playback generations distinct',async()=>{
 const workers:FakeWorker[]=[],port=new BrowserVocal(()=>{const w=new FakeWorker();workers.push(w);return w as unknown as Worker;});
 try{
  const pose=port.request('pose',{revision:8});workers[0]!.finish({revision:8},{pose:{revision:8}});await pose;
  const preview=port.request('preview');workers[0]!.finish({playback_generation:3});assert.equal((await preview).playback_generation,3);
  const render=port.request('animation/prepare');const rejected=assert.rejects(render,{name:'AbortError'});
  assert.equal((await port.request('animation/stop')).cancelled,true);await rejected;assert(workers[0]!.terminated);
  const next=port.request('meta');assert.deepEqual(workers[1]!.messages[0].checkpoint,{pose:{revision:8}});workers[1]!.finish({execution:'browser-wasm'});await next;
  const play=port.request('preview');workers[1]!.finish({playback_generation:1});assert((await play).playback_generation>3);
  const heartbeat=port.request('heartbeat',{playback_generation:5});assert.equal(workers[1]!.messages.at(-1).body.playback_generation,1);workers[1]!.finish({});await heartbeat;
 }finally{port.close();}
});
test('native Worker crashes reject the operation and do not leak state to another executor',async()=>{
 const one=new FakeWorker(),two=new FakeWorker(),a=new BrowserVocal(()=>one as unknown as Worker),b=new BrowserVocal(()=>two as unknown as Worker);
 try{const pending=a.request('pose');one.onerror();await assert.rejects(pending,/中断/);const p=b.request('meta');assert.equal(two.messages[0].checkpoint,undefined);two.finish({});await p;}
 finally{a.close();b.close();}
});
test('local science avoids the server session, while original video encoding remains available',async()=>{
 const operations:string[]=[],server:any[]=[];
 const port=webVocalPort(async(method,path,body:any)=>{server.push({method,path,body});return path.startsWith('vocal/sessions?')?{id:'video-only'}:{id:'video'};},'p','wasm-test',()=>({async request(op){operations.push(op);return op==='meta'?{execution:'browser-wasm'}:{};},close(){}}));
 try{assert.equal((await port.request('meta')).execution,'browser-wasm');await port.request('pose');assert.equal(server.length,0);await port.request('video/begin',{duration:.5});assert.equal(server.length,2);assert.equal(server[1].body.op,'video/begin');assert.deepEqual(operations,['meta','meta','pose']);}
 finally{await port.close();}
});
test('unavailable local runtime chooses the original server before any scientific edit',async()=>{
 const operations:string[]=[];let closed=false;
 const port=webVocalPort(async(_method,path,body:any)=>path.startsWith('vocal/sessions?')?{id:'fallback'}:(operations.push(body.op),{}),'p','wasm-fallback',()=>({async request(){throw Error('no runtime');},close(){closed=true;}}));
 try{await port.request('meta');await port.request('pose');assert(closed);assert.deepEqual(operations,['meta','pose']);}finally{await port.close();}
});
test('a local computation error is reported without silently switching to a fresh server state',async()=>{
 let scientific=0;
 const port=webVocalPort(async()=>{scientific++;return {};},'p','wasm-error',()=>({async request(op){if(op==='pose')throw Error('native failure');return {};},close(){}}));
 try{await port.request('meta');await assert.rejects(port.request('pose'),/native failure/);assert.equal(scientific,0);}finally{await port.close();}
});

test('streaming PCM outranks queued display work without reordering scientific writes',async()=>{
 const worker=new FakeWorker(),port=new BrowserVocal(()=>worker as unknown as Worker);
 try{
  const first=port.request('pose',{revision:1}),monitor=port.request('audio/monitor'),second=port.request('pose',{revision:2}),audio=port.request('audio/pull');
  worker.finish({revision:1});await first;assert.equal(worker.messages.at(-1).op,'audio/pull');
  worker.finish({samples:0});await audio;assert.equal(worker.messages.at(-1).op,'audio/monitor');
  worker.finish({});await monitor;assert.equal(worker.messages.at(-1).body.revision,2);worker.finish({revision:2});await second;
 }finally{port.close();}
});

test('tab deactivation stops after the current PCM block and retains the initialized engine',async()=>{
 const worker=new FakeWorker(),port=new BrowserVocal(()=>worker as unknown as Worker);
 try{
  const audio=port.request('audio/pull'),monitor=port.request('audio/monitor'),stop=port.request('deactivate');
  assert.equal(worker.terminated,false);worker.finish({samples:7680});await audio;
  assert.equal(worker.messages.at(-1).op,'deactivate');worker.finish({active:false});await stop;
  assert.equal(worker.terminated,false);assert.equal(worker.messages.at(-1).op,'audio/monitor');worker.finish({});await monitor;
 }finally{port.close();}
});
