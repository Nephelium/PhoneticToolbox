import {test} from 'node:test';
import assert from 'node:assert/strict';
import {reactive,nextTick} from 'vue';
import {createAudioSelectionGroup} from '../src/state/audio-selection.ts';
import type {Workspace} from '../src/state/workspace.ts';
const state=()=>reactive({asset:null,start:0,end:0,channel:0,parameters:[],dirty:false,error:'',loading:false,zoom:1,offset:0}) as Workspace;
const asset=(name:string)=>({name,duration:2,sampleRate:8000,channels:[new Float32Array(16000)]});
test('multi-audio loads are unselected; gestures transfer sole ownership and keyboard target',async()=>{
 const group=createAudioSelectionGroup(),a=state(),b=state();group.register(()=>a);group.register(()=>b);
 a.asset=asset('source');a.end=2;b.asset=asset('target');b.end=2;await nextTick();
 assert.deepEqual([a.start,a.end,b.start,b.end],[0,0,0,0]);assert(!group.keyboard(a)&&!group.keyboard(b));
 group.select(a);a.start=.123456789;a.end=.7654321;assert(group.keyboard(a));assert(!group.keyboard(b));
 group.select(b);b.start=.3;b.end=.5;assert.deepEqual([a.start,a.end],[0,0]);assert(!group.keyboard(a)&&group.keyboard(b));
 a.asset=asset('generated');a.end=2;await nextTick();assert.equal(a.end,0);assert.deepEqual([b.start,b.end],[.3,.5]);assert(group.keyboard(b));
 group.dispose();
});
test('same workspace views count once; module groups and single-audio behavior stay independent',async()=>{
 const first=createAudioSelectionGroup(),second=createAudioSelectionGroup(),a=state(),b=state();
 first.register(()=>a);const remove=first.register(()=>a);second.register(()=>b);
 a.asset=asset('a');a.end=2;b.asset=asset('b');b.end=2;await nextTick();assert.equal(a.end,2);assert(first.keyboard(a)&&second.keyboard(b));
 first.select(a);a.start=.123456789;remove();await nextTick();assert.equal(a.start,.123456789);assert.equal(b.end,2);
 first.dispose();second.dispose();
});
