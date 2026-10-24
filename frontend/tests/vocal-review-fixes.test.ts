import test from 'node:test';
import assert from 'node:assert/strict';
import {builtinPose} from '../public/vocal-tract/presets.js';
import {timelinePitch} from '../public/vocal-tract/video-timeline.mjs';
import {commitSequenceImport} from '../public/vocal-tract/sequence-import.mjs';

for(const failure of ['save','apply','none'])test(`sequence import ${failure} preserves or commits the complete sequence and active source`,async()=>{
 const old={frames:[{f0:120}],pitch_curve:[[0,120]],features:[{id:'old'}],global_controls:{f0:{value:120}},perturbation:{jitter:0,shimmer:0,seed:1}};
 const incoming={frames:[{f0:240}],pitch_curve:[],features:[],global_controls:{},perturbation:{jitter:80,shimmer:70,seed:2}};
 let saved=structuredClone(old),active=structuredClone(old.perturbation),displayed=structuredClone(old.perturbation),calls:string[]=[];
 const post=async(op:string,body:any)=>{assert.equal(op,'keyframes');calls.push('save');if(failure==='save')throw Error('quota');saved=structuredClone(body);};
 const apply=async(p:any)=>{calls.push('apply');active=structuredClone(p);if(failure==='apply'&&p.seed===2)throw Error('source failed after accepting settings');displayed=structuredClone(p);};
 const run=commitSequenceImport(post,incoming,old,apply,old.perturbation);
 if(failure==='none'){await run;assert.deepEqual(saved,incoming);assert.deepEqual(active,incoming.perturbation);assert.deepEqual(displayed,active);}
 else{await assert.rejects(run,failure==='save'?/quota/:/source failed/);assert.deepEqual(saved,old);assert.deepEqual(active,old.perturbation);assert.deepEqual(displayed,old.perturbation);}
 if(failure==='save')assert.deepEqual(calls,['save']);
});
test('sequence import attempts both restorations and reports an incomplete rollback honestly',async()=>{
 const old={frames:[],perturbation:null},next={frames:[{}],perturbation:{seed:2}};let sourceRestored=false,writes=0;
 await assert.rejects(commitSequenceImport(async()=>{if(++writes===2)throw Error('disk unavailable');},next,old,async p=>{if(p)throw Error('source unavailable');sourceRestored=true;},null),e=>e.sequenceRestored===false&&/恢复未完成/.test(e.message));
 assert.equal(sourceRestored,true);
});

for(const [speaker,f0] of [['JD2',120],['W02',200]] as const){
 test(`${speaker} switching from a voiceless consonant to a vowel restores the complete builtin pose`,()=>{
  const metadata={speaker,default_f0:f0,presets:{a:[1,2],t:[3,4]},default_poses:{},
   source_presets:{voiced:{mode:'voiced',vibration:1,pressure_pa:800}},
   consonants:{t:{params:[3,4],source:{mode:'voiceless',vibration:0,pressure_pa:800}}}};
  const before=structuredClone(metadata),stop=builtinPose(metadata,'t'),vowel=builtinPose(metadata,'a');
  assert.equal(stop.source.vibration,0);assert.equal(vowel.source.vibration,1);assert.equal(vowel.f0,f0);
  assert.equal(vowel.manual_root,false);assert.deepEqual(vowel.lip_shape,[0,0,0,0]);
  vowel.params[0]=9;vowel.source.pressure_pa=400;assert.deepEqual(metadata,before);
 });
}

test('authored builtin geometry and source override defaults without sharing mutable arrays',()=>{
 const meta={default_f0:120,presets:{a:[1,2]},source_presets:{voiced:{vibration:1}},
  default_poses:{a:{params:[5,6],lip_shape:[.1,0,0,0],source:{vibration:.7}}}};
 const p=builtinPose(meta,'a');assert.deepEqual(p.params,[5,6]);assert.equal(p.source.vibration,.7);
 p.lip_shape[0]=.4;assert.equal(meta.default_poses.a.lip_shape[0],.1);
});

test('video pitch comes from the rendered timeline including a feature edge and unvoiced spans',()=>{
 const points=[{time:0,f0:240},{time:.1-Number.EPSILON,f0:240},{time:.1,f0:280},
  {time:.2-Number.EPSILON,f0:280},{time:.2,f0:null},{time:.3,f0:null}];
 assert.equal(timelinePitch(points,.05),240);assert.equal(timelinePitch(points,.1),280);
 assert.equal(timelinePitch(points,.15),280);assert.equal(timelinePitch(points,.2),null);
 assert.equal(timelinePitch(points,.25),null);assert.equal(timelinePitch([],0),null);
});
