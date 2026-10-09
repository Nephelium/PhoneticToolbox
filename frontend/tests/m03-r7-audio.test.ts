import {test} from 'node:test';
import assert from 'node:assert/strict';
import type {AudioAsset} from '../src/platform/types.ts';

test('M03-R7 selection playback uses buffer offset and retains global position',async()=>{
 const starts:number[][]=[];let tick:()=>void=()=>{};
 Object.assign(globalThis,{BroadcastChannel:undefined,requestAnimationFrame:(fn:()=>void)=>{tick=fn;return 1;},cancelAnimationFrame:()=>{}});
 class Context {
  state='running';currentTime=10;destination={};
  async resume(){}
  createBuffer(){return {copyToChannel(){}};}
  createGain(){return {gain:{value:1},connect(){return this;},disconnect(){}};}
  createBufferSource(){return {buffer:null,onended:null,connect(){return this;},disconnect(){},stop(){},start(...args:number[]){starts.push(args);}};}
 }
 const context=new Context();Object.assign(globalThis,{AudioContext:class {constructor(){return context;}}});
 const {play,playback,pause}=await import('../src/state/audio.ts');
 const asset={name:'tail',sampleRate:48000,frames:480000,duration:1800,originSeconds:1790,channels:[new Float32Array(480000)]} as AudioAsset;
 await play(asset,1795,1799,0);assert.deepEqual(starts[0],[0,5,4]);assert.equal(playback.position,1795);
 context.currentTime=12;tick();assert.equal(playback.position,1797);pause();
 await play({...asset,originSeconds:0,duration:10},2,4,0);assert.deepEqual(starts[1],[0,2,2]);pause();
});
