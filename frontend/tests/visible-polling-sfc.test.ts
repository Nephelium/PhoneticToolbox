import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse,compileScript} from '@vue/compiler-sfc';
import ts from 'typescript';
import * as vue from 'vue';

// Exercise actual component lifecycles with a controllable document and transport.
function mount(file:string,hidden=false){
 const compiled=compileScript(parse(readFileSync(new URL('../src/'+file,import.meta.url),'utf8')).descriptor,{id:'visible-poll',inlineTemplate:false});
 const js=ts.transpileModule(compiled.content,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
 const mounted:Array<()=>void>=[],unmounted:Array<()=>void>=[],timers=new Map<number,()=>void>(),events=new Map<string,Set<()=>void>>(),requests:string[]=[];
 let next=0,hold=false,lists=0;const pending:Array<()=>void>=[];
 const document={hidden,addEventListener:(name:string,fn:()=>void)=>{if(!events.has(name))events.set(name,new Set());events.get(name)!.add(fn);},removeEventListener:(name:string,fn:()=>void)=>events.get(name)?.delete(fn)};
 const response=(name:string)=>name.includes('capabilities')?{task_operations:[]}:name.includes('assets?')?{assets:[],available:true}:name.includes('references')?[]:name.includes('capacity-request')?null:{available_bytes:100,quota_bytes:1000,used_bytes:0,reserved_bytes:0,policy_version:2};
 const fetch=async(name:string)=>{requests.push(name);if(hold)await new Promise<void>(resolve=>pending.push(resolve));return {ok:true,json:async()=>response(name)};};
 const require=(name:string)=>{
  if(name==='vue')return {...vue,onMounted:(fn:()=>void)=>mounted.push(fn),onUnmounted:(fn:()=>void)=>unmounted.push(fn)};
  if(name.endsWith('feedback.ts'))return {feedbackCall:async()=>{requests.push('unread');if(hold)await new Promise<void>(resolve=>pending.push(resolve));return {unread:3};}};
  if(name.endsWith('local-assets.ts'))return {LocalStorageError:class extends Error{},LocalAssets:class{async list(){lists++;return []}}};
  if(name.endsWith('local-asset-transport.ts'))return {LocalAssetTransport:class{fetch=fetch;view=(v:unknown)=>v}};
  return {default:{}};
 };
 const module={exports:{} as any},scope=vue.effectScope();new Function('require','module','exports','document','fetch','setInterval','clearInterval',js)(require,module,module.exports,document,fetch,(fn:()=>void)=>{timers.set(++next,fn);return next;},(id:number)=>timers.delete(id));
 const m=scope.run(()=>module.exports.default.setup({ownerId:'owned',csrfToken:'fixture',projectId:'project'},{emit:()=>{},expose:()=>{}}));
 const flush=async()=>{for(let i=0;i<12;i++)await Promise.resolve();await vue.nextTick();};
 return {m,requests,document,events,timers,get lists(){return lists;},async start(){mounted.forEach(fn=>fn());await flush();},async tick(){timers.forEach(fn=>fn());await flush();},async visible(value:boolean){document.hidden=!value;events.get('visibilitychange')?.forEach(fn=>fn());await flush();},hold(){hold=true;},async release(){hold=false;pending.splice(0).forEach(fn=>fn());await flush();},close(){unmounted.forEach(fn=>fn());scope.stop();},flush};
}

test('server capacity/load polling sleeps while hidden, resumes once, and does not overlap slow reads',async()=>{
 const f=mount('account/WebCapacityStatus.vue',true);try{
  await f.start();await f.tick();assert.equal(f.requests.length,0);
  f.hold();await f.visible(true);assert.equal(f.requests.length,4);await f.tick();await f.visible(true);assert.equal(f.requests.length,4);
  await f.visible(false);await f.release();await f.tick();assert.equal(f.requests.length,4);
  await f.visible(true);assert.equal(f.requests.length,8);assert(f.m.server.value);
 }finally{f.close();}assert.equal(f.timers.size,0);assert.equal(f.events.get('visibilitychange')?.size,0);await f.visible(true);await f.tick();assert.equal(f.requests.length,8);
});

test('project file polling skips hidden IndexedDB and network reads but refreshes when visible again',async()=>{
 const f=mount('account/ProjectStorage.vue');try{
  await f.start();assert.equal(f.lists,1);const initial=f.requests.length;assert(initial>=3);
  await f.visible(false);await f.tick();await f.tick();assert.equal(f.lists,1);assert.equal(f.requests.length,initial);
  await f.visible(true);assert.equal(f.lists,2);assert(f.requests.length>initial);
 }finally{f.close();}assert.equal(f.timers.size,0);assert.equal(f.events.get('visibilitychange')?.size,0);const n=f.requests.length;await f.visible(true);await f.tick();assert.equal(f.requests.length,n);
});

test('feedback unread polling coalesces slow checks and ignores a late reply after teardown',async()=>{
 const f=mount('components/FeedbackNotice.vue',true);try{
  await f.start();await f.tick();assert.equal(f.requests.length,0);
  f.hold();await f.visible(true);await f.tick();await f.visible(true);assert.equal(f.requests.length,1);
  await f.visible(false);await f.release();assert.equal(f.m.unread.value,3);await f.tick();assert.equal(f.requests.length,1);
  f.hold();await f.visible(true);assert.equal(f.requests.length,2);f.m.unread.value=4;f.close();await f.release();assert.equal(f.m.unread.value,4);
 }finally{f.close();}assert.equal(f.timers.size,0);assert.equal(f.events.get('visibilitychange')?.size,0);
});
