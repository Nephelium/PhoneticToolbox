import test from 'node:test';
import assert from 'node:assert/strict';
import {VocalPlayback,webVocalPort} from '../src/platform/vocal.ts';

test('prepared audio buffers before playback across slow HTTPS pulls without changing PCM',async()=>{
 const pending:((packet:any)=>void)[]=[],starts:number[]=[],copied:Float32Array[]=[];
 const clock={state:'running',currentTime:0,destination:{},async resume(){},createBuffer(_channels:number,n:number,rate:number){return {duration:n/rate,copyToChannel(pcm:Float32Array){copied.push(pcm.slice());}};},createBufferSource(){return {buffer:null,connect(){},disconnect(){},stop(){},start(at:number){starts.push(at);},onended:null};}};
 const player=new VocalPlayback(()=>new Promise(resolve=>pending.push(resolve)));player.context=clock as unknown as AudioContext;
 const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
 const packet=(n:number,value:number,eof=false)=>{const pcm=new Float32Array(n).fill(value);return {samples:n,sample_rate:48000,base64:Buffer.from(pcm.buffer).toString('base64'),eof};};
 try{await player.play('preview');await tick();
 clock.currentTime=.6;pending.shift()!(packet(23040,.5));await tick();assert.equal(starts.length,0);
 clock.currentTime=1.2;pending.shift()!(packet(23040,.25));await tick();assert.equal(starts.length,0);
 clock.currentTime=1.8;pending.shift()!(packet(1920,-.25,true));await tick();
 assert.equal(starts.length,3);assert(Math.abs(starts[0]-1.88)<1e-12);assert.equal(player.underflows,0);
 assert.equal(copied.reduce((n,p)=>n+p.length,0),48000);assert.equal(copied[0][0],.5);assert.equal(copied[1][0],.25);assert.equal(copied[2][0],-.25);
 assert.deepEqual(player.levels(),{rms:0,peak:0});clock.currentTime=2;assert.deepEqual(player.levels(),{rms:.5,peak:.5});
 }finally{player.stop();await tick();}
});

test('browser animation clock counts PCM and excludes network gaps',()=>{
 const player=new VocalPlayback(async()=>{});const clock={currentTime:10};player.context=clock as AudioContext;player.active=true;
 player.segments=[{time:10,offset:0,duration:.16},{time:10.4,offset:.16,duration:.16}];
 assert.equal(player.position(),0);clock.currentTime=10.08;assert(Math.abs(player.position()-.08)<1e-10);
 clock.currentTime=10.3;assert.equal(player.position(),.16);
 clock.currentTime=10.48;assert(Math.abs(player.position()-.24)<1e-10);
 clock.currentTime=11;assert.equal(player.position(),.32);
});

test('closing while a preview is being calculated cannot restart audio',async()=>{
 const original={AudioContext:globalThis.AudioContext,window:globalThis.window,sessionStorage:globalThis.sessionStorage};
 let resolvePreview:(value:any)=>void=()=>{},closed=0,released=0;const values=new Map();
 class Audio {state='running';currentTime=0;async resume(){}async close(){closed++;}createBuffer(){throw Error('closed preview must never play');}}
 Object.assign(globalThis,{AudioContext:Audio,window:{addEventListener(){},removeEventListener(){}},sessionStorage:{getItem:(k:string)=>values.get(k),setItem:(k:string,v:string)=>values.set(k,v),removeItem:(k:string)=>values.delete(k)}});
 try{
  const port=webVocalPort(async(method,path,body:any)=>{
   if(method==='DELETE'){released++;return {};}
   if(path.startsWith('vocal/sessions?'))return {id:'owned'};
   if(body.op==='preview')return new Promise(resolve=>{resolvePreview=resolve;});
   return {};
  },'project','server:alice:project:M10');
  const pending=port.request('preview');const check=assert.rejects(pending,/已关闭/);
  await new Promise(resolve=>setTimeout(resolve,0));await port.close();resolvePreview({active:true});await check;
  assert.equal(released,1);assert.equal(closed,1);
 }finally{Object.assign(globalThis,original);}
});
