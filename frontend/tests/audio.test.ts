import {test} from 'node:test';
import assert from 'node:assert/strict';
import type {AudioAsset} from '../src/platform/types.ts';

// Deterministic device scheduling; real Web Audio is exercised separately in Chrome.
Object.assign(globalThis,{BroadcastChannel:undefined,requestAnimationFrame:()=>1,cancelAnimationFrame:()=>{}});
const pending:{resolve:()=>void;reject:(error:Error)=>void}[]=[];
const sources:{stopped:boolean;starts:number[][];onended:(()=>void)|null}[]=[];
class Device {
  state='running';currentTime=0;
  resume(){return new Promise<void>((resolve,reject)=>pending.push({resolve,reject}));}
  createBuffer(){return {copyToChannel:()=>{}};}
  createGain(){return {gain:{value:0},connect:()=>{},disconnect:()=>{}};}
  createBufferSource(){const s={buffer:null,onended:null,stopped:false,starts:[] as number[][],connect:(gain:unknown)=>gain,disconnect:()=>{},stop(){this.stopped=true;},start(...v:number[]){this.starts.push(v);}};sources.push(s);return s;}
}
Object.assign(globalThis,{AudioContext:Device});
const audio=await import('../src/state/audio.ts');
const asset=(name:string):AudioAsset=>({name,sampleRate:8000,channels:[new Float32Array(8000),new Float32Array(8000)],frames:8000,duration:1});

test('late resume rejection cannot stop a newer successful playback',async()=>{
  const a=audio.play(asset('old'),0,.8,0),b=audio.play(asset('new'),.2,.7,0);
  pending[1].resolve();await b;const current=sources.at(-1)!;
  pending[0].reject(Error('Old device request failed'));await a;
  assert.equal(audio.playback.playing,true);assert.equal(current.stopped,false);assert.equal(audio.playback.error,'');audio.stop();
});
test('late resume success cannot restart audio after stop',async()=>{
  const p=audio.play(asset('stopped'),0,.5,0),last=pending.at(-1)!;audio.stop();const count=sources.length;last.resolve();await p;assert.equal(sources.length,count);assert.equal(audio.playback.playing,false);
});
test('current resume failure reports error and leaves no playback',async()=>{
  const p=audio.play(asset('failure'),0,.5,0);pending.at(-1)!.reject(Error('Current device failed'));await p;assert.equal(audio.playback.playing,false);assert.equal(audio.playback.error,'Current device failed');
});
test('playback ownership includes asset identity and selected channel',async()=>{
  const a=asset('same name'),b=asset('same name');const p=audio.play(a,.2,.8,1);pending.at(-1)!.resolve();await p;
  assert(audio.isCurrentAudio(a,1));assert(!audio.isCurrentAudio(a,0));assert(!audio.isCurrentAudio(b,1));
  audio.pause();assert(audio.isCurrentAudio(a,1));assert.equal(audio.playback.position,.2);audio.stop();assert(!audio.isCurrentAudio(a,1));
});
test('seeking another asset does not auto-start from the old playing state',async()=>{
  const a=asset('original'),b=asset('inverse');const p=audio.play(a,0,.8,0);pending.at(-1)!.resolve();await p;const count=pending.length;
  audio.seek(b,.3,0,.8,0);assert.equal(pending.length,count);assert.equal(audio.playback.playing,false);assert(audio.isCurrentAudio(b,0));assert.equal(audio.playback.position,.3);audio.stop();
});
test('seeking the playing asset resumes at the bounded selected position',async()=>{
  const a=asset('original');const p=audio.play(a,0,.8,0);pending.at(-1)!.resolve();await p;
  audio.seek(a,.4,.2,.8,0);pending.at(-1)!.resolve();await Promise.resolve();assert.equal(audio.playback.playing,true);assert.deepEqual(sources.at(-1)!.starts,[[0,.4,.4]]);audio.stop();
});
test('pending switch displays its own start instead of the old playhead',async()=>{
  const a=asset('previous'),b=asset('new');const p=audio.play(a,.6,.8,0);pending.at(-1)!.resolve();await p;
  const next=audio.play(b,.1,.7,0);assert(audio.isCurrentAudio(b,0));assert.equal(audio.playback.position,.1);assert.equal(audio.playback.playing,false);
  audio.stop();pending.at(-1)!.resolve();await next;
});
