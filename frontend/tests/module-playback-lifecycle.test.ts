import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse,compileScript} from '@vue/compiler-sfc';
import ts from 'typescript';
import * as vue from 'vue';

const noop=()=>{};
const deferred=()=>{let resolve!:(value?:any)=>void;return {promise:new Promise<any>(r=>resolve=r),resolve};};
// The real page setup and Vue watchers run; devices, media and native transport
// are synthetic. These checks do not represent a browser or physical device.
Object.assign(globalThis,{window:{addEventListener:noop,removeEventListener:noop},document:{hidden:false,addEventListener:noop,removeEventListener:noop},cancelAnimationFrame:noop,requestAnimationFrame:()=>1,ResizeObserver:class{observe(){}disconnect(){}}});
function page(relative:string,props:any,overrides:Record<string,any>){
 const source=readFileSync(new URL('../src/'+relative,import.meta.url),'utf8');
 const compiled=compileScript(parse(source).descriptor,{id:relative,inlineTemplate:false});
 const js=ts.transpileModule(compiled.content,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
 const module={exports:{} as any},unmount:(()=>void)[]=[],scope=vue.effectScope();
 const require=(name:string)=>{
  if(name==='vue')return {...vue,onMounted:noop,onUnmounted:(fn:()=>void)=>unmount.push(fn),onBeforeUnmount:(fn:()=>void)=>unmount.push(fn)};
  if(Object.hasOwn(overrides,name))return overrides[name];
  if(name.endsWith('.vue'))return {default:{}};
  if(name.endsWith('/web-settings.ts'))return {useWebSettings:noop,settingsFields:()=>({}),applyingWebSettings:vue.ref(false)};
  return new Proxy({},{get:(_target,key)=>key==='__esModule'?true:noop});
 };
 new Function('require','module','exports',js)(require,module,module.exports);
 const model=scope.run(()=>module.exports.default.setup(props,{emit:noop,expose:noop}));
 return {model,close(){for(const fn of unmount)fn();scope.stop();}};
}
function recordingFixture(stopWait?:ReturnType<typeof deferred>){
 const wait=deferred(),calls:string[]=[],props=vue.reactive({active:true,stateKey:'test-M16'});
 const port={kind:'desktop',async request(body:any){calls.push(body.op);if(body.op==='play'&&calls.filter(op=>op==='play').length===1)await wait.promise;if(body.op==='play_stop'&&stopWait)await stopWait.promise;return body.op==='status'?{playback:{playing:false},capture_active:false}:{};},dispose:noop};
 const instance=page('modules/recording/RecordingPage.vue',props,{
  '../../platform/recording.ts':{getRecordingPort:()=>port},'../../state/audition.ts':{automaticAudioGain:vue.ref(true)},'../../state/audio.ts':{stop:noop},
  './defaults.ts':{recordingDefaults:()=>({device:'default',channels:1,roles:['microphone'],sample_rate:48000}),recordingSettings:()=>({}),bindInput:noop},
  './task-import.ts':{autoMapping:()=>({})},'./state.ts':{currentFrames:(take:any)=>take?.versions[take.head].frames??0,projectSelection:()=>({taskId:'',takeId:'take'})}
 });
 const m=instance.model;m.project.value={tasks:[],selected:{},takes:[{id:'take',head:0,config:{sample_rate:48000,channels:1,roles:['microphone']},versions:[{frames:4800}]}]};m.takeId.value='take';
 return {...instance,props,wait,calls};
}
test('M16 a play reply after hide is stopped before a fresh explicit audition',async()=>{
 const f=recordingFixture(),m=f.model;
 try{
  const pending=m.play();await vue.nextTick();f.props.active=false;await vue.nextTick();
  f.wait.resolve();await pending;assert.equal(m.playing.value,false);assert.deepEqual(f.calls,['play','play_stop']);
  f.props.active=true;await vue.nextTick();await m.play();assert.equal(m.playing.value,true);assert.equal(f.calls.filter(op=>op==='play_stop').length,1);
  f.props.active=false;await new Promise(resolve=>setImmediate(resolve));assert.equal(m.playing.value,false);
 }finally{f.close();}
});
test('M16 hide and return before the reply still cancels the old audition without stopping capture',async()=>{
 const f=recordingFixture(),m=f.model;
 try{
  const pending=m.play();f.props.active=false;await vue.nextTick();f.props.active=true;await vue.nextTick();f.wait.resolve();await pending;
  assert.equal(m.playing.value,false);assert.equal(f.calls.filter(op=>op==='play_stop').length,1);assert(!f.calls.includes('stop'));
  m.capture.value=true;f.props.active=false;await vue.nextTick();assert.equal(m.capture.value,true);assert(!f.calls.includes('stop'));
 }finally{f.close();}
});
test('M16 finishes its compensating stop before permitting a new audition',async()=>{
 const stopWait=deferred(),f=recordingFixture(stopWait),m=f.model;
 try{
  const pending=m.play();f.props.active=false;await vue.nextTick();f.wait.resolve();await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(f.calls,['play','play_stop']);assert.equal(m.busy.value,true);
  f.props.active=true;await vue.nextTick();await m.play();assert.equal(f.calls.filter(op=>op==='play').length,1);
  stopWait.resolve();await pending;await m.play();assert.equal(f.calls.filter(op=>op==='play').length,2);assert.equal(m.playing.value,true);
 }finally{stopWait.resolve();f.close();}
});
function lipFixture(){
 const wait=deferred(),props=vue.reactive({active:true,stateKey:'test-M05',port:{available:true,kind:'desktop',playbackVideo:()=>wait.promise}});
 const row={index:0,time_s:0,detected:false,points:null,metrics:null};let plays=0;
 class Replay{async frame(){return row;}async time(){return row;}prefetch(){}}
 const instance=page('modules/lip-extraction/LipExtractionPage.vue',props,{
  '../../state/media-audition.ts':{useMediaAudition:()=>vue.ref('')},'./replay.ts':{LipReplay:Replay},
  './overlay.ts':{faceBounds:()=>null,drawLegacyOverlay:noop,fitFace:noop},'./devices.ts':{audioDevicesHidden:()=>false},'../../../../resources/m05/resources.json':{default:{files:[]}}
 });
 const m=instance.model;m.results.value=[{id:'result',rows:[row],metadata:{timing:{decoded_frames:1},audio:{present:true}}}];m.sourceUrls.value.result='blob:synthetic-input';
 m.replayVideo.value={videoWidth:1,currentTime:0,paused:true,pause:noop,play:async()=>{plays++;}};
 return {...instance,props,wait,plays:()=>plays};
}
test('M05 hidden media preparation and loadedmetadata cannot resume an old audition',async()=>{
 const f=lipFixture(),m=f.model;
 try{
  await vue.nextTick();m.mediaFailure.value=true;const pending=m.play();await vue.nextTick();assert.equal(m.mediaPreparing.value,true);
  f.props.active=false;await vue.nextTick();f.props.active=true;await vue.nextTick();
  f.wait.resolve({origin_s:0,blob:new Blob(['synthetic'],{type:'video/webm'})});await pending;m.mediaLoaded();await new Promise(resolve=>setImmediate(resolve));assert.equal(f.plays(),0);
  await m.play();assert.equal(f.plays(),1);f.props.active=false;await vue.nextTick();await m.play();assert.equal(f.plays(),1);
 }finally{f.close();}
});
test('M05 hide cancels a pending frame seek and a native media play event',async()=>{
 const f=lipFixture(),m=f.model;
 try{
  await vue.nextTick();const pending=m.play();f.props.active=false;await vue.nextTick();await pending;assert.equal(f.plays(),0);
  m.mediaPlaying();assert.equal(m.replaying.value,false);
 }finally{f.close();}
});
test('M07 a hidden then restored result read must not revive its old audition intent',async()=>{
 const wait=deferred(),props=vue.reactive({active:true,stateKey:'test-M07',context:{files:{m07:{read:()=>wait.promise}}}}),plays:any[]=[];
 const f=page('modules/phonation-synthesis/PhonationSynthesisPage.vue',props,{
  '../../state/workspace.ts':{workspace:()=>vue.reactive({asset:null,start:0,end:0,zoom:1,offset:0,channel:0}),host:{projects:{read:()=>null,write:()=>true}}},
  '../../state/audio.ts':{stop:noop,playback:vue.reactive({playing:false}),isCurrentAudio:()=>false},
  '../../platform/decode.ts':{decodeWav:async()=>({frames:4800,sampleRate:48000,duration:.1,channels:[new Float32Array(4800)]})},
  './state.ts':{defaults:()=>({}),generationDefaults:()=>({step_count:9}),labels:{},clone:structuredClone},
 });
 const transport=page('components/AudioTransport.vue',vue.reactive({get active(){return props.active;},state:f.model.wave}),{
  '../state/audio-selection.ts':{useAudioSelection:()=>undefined},
  '../state/audio.ts':{stop:noop,play:async(...args:any[])=>plays.push(args),playback:vue.reactive({playing:false}),isCurrentAudio:()=>false}
 });
 try{
  f.model.resultTransport.value={selectAll:transport.model.selectAll};
  const pending=f.model.play({job:'result',metadata:{generation:{step_count:9}}},'combined_steps.wav',true);
  props.active=false;await vue.nextTick();props.active=true;await vue.nextTick();wait.resolve(new ArrayBuffer(0));await pending;
  assert.equal(plays.length,0);assert(f.model.wave.asset);assert.equal(f.model.wave.end,.1);
  await f.model.play({job:'result',metadata:{generation:{step_count:9}}},'combined_steps.wav',true);assert.equal(plays.length,1);
 }finally{transport.close();f.close();}
});
