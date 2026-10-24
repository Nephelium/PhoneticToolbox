import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
import {ref,reactive,computed,watch,nextTick,createRenderer,h,onUnmounted} from 'vue';
import {parse} from '@vue/compiler-sfc';
import {baseParse,NodeTypes} from '@vue/compiler-dom';
import {browserM06} from '../src/platform/browser-m06.ts';
import {defaults,valid,clone} from '../src/modules/speech-synthesis/state.ts';
import {MediaBank} from '../src/modules/perception/media.ts';
import type {M06Port} from '../src/modules/speech-synthesis/port.ts';

const source=(path:string)=>readFileSync(new URL('../src/'+path,import.meta.url),'utf8');
// Run actual SFC declarations with only storage/media/host boundaries replaced.
function component(path:string,names:string[],env:Record<string,any>,state:string[]=[],effects:RegExp[]=[]){
 const script=parse(source(path)).descriptor.scriptSetup!.content;
 const file=ts.createSourceFile(path,script,ts.ScriptTarget.Latest,true,ts.ScriptKind.TS);
 const functions=new Map<string,string>(),variables=new Map<string,string>();
 for(const node of file.statements){
  if(ts.isFunctionDeclaration(node)&&node.name)functions.set(node.name.text,node.getText(file));
  if(ts.isVariableStatement(node))for(const d of node.declarationList.declarations)if(ts.isIdentifier(d.name))variables.set(d.name.text,`let ${d.getText(file)};`);
 }
 const get=(map:Map<string,string>,key:string)=>{assert(map.has(key),`Missing ${path}.${key}`);return map.get(key)!;};
 const statements=effects.map(pattern=>{const matches=file.statements.filter(n=>pattern.test(n.getText(file)));assert.equal(matches.length,1,String(pattern));return matches[0].getText(file);});
 const body=[...state.map(key=>get(variables,key)),...names.map(key=>get(functions,key)),...statements,`return {${[...names,...state].join(',')}};`].join('\n');
 return new Function(...Object.keys(env),ts.transpileModule(body,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText)(...Object.values(env));
}
function deferred<T>(){let resolve!:(v:T)=>void;return {promise:new Promise<T>(r=>resolve=r),resolve:(v:T)=>resolve(v)};}
const turn=()=>new Promise(r=>setImmediate(r));

test('AppShell does not leave a project after M13 save fails, and awaits M06 save',async()=>{
 const names=['m16Page','m17Page','settingsPage','m07Page','m05Page','m11Page','m15Page','m06Page','m14Page','m08Page','m13Page','m09Page','m12Page'];
 const env:Record<string,any>=Object.fromEntries(names.map(name=>[name,ref()]));
 let leaves=0,m13Saves=0;const dirty=new Set(['M13']);
 Object.assign(env,{closingBusy:ref(false),m15Focus:ref(false),settingsDirty:ref(false),moduleDirty:(id:string)=>dirty.has(id),emit:()=>leaves++});
 env.m13Page.value={save:()=>{m13Saves++;return false;}};
 const api=component('app/AppShell.vue',['leaveProject'],env);
 await api.leaveProject();assert.equal(m13Saves,1);assert.equal(leaves,0);assert.equal(env.closingBusy.value,false);
 env.m13Page.value.save=()=>true;const pending=deferred<boolean>();dirty.add('M06');env.m06Page.value={save:()=>pending.promise};
 const leaving=api.leaveProject();await turn();assert.equal(leaves,0);pending.resolve(false);await leaving;assert.equal(leaves,0);
 env.m06Page.value.save=async()=>true;await api.leaveProject();assert.equal(leaves,1);
});

test('M15 hiding or disposing during decode discards late media and never revives on return',async()=>{
 for(const transition of ['hide','dispose','blob']){
  const decode=deferred<any>(),blob=deferred<Blob>(),props=reactive({active:true}),preview=ref<any>(null);let discarded=0,created=0,paused=0;
  const asset={hash:'sha'},prepared={asset,buffer:{},bytes:12,url:transition==='blob'?undefined:'blob:image'};
  const media={unlock:async()=>{},preview:()=>decode.promise,discardPreview:(p:any)=>{assert.equal(p,prepared);discarded++;},dispose:()=>{}};
  let unmount=()=>{};const noop=()=>{};
  const env={props,ref,watch,preview,previewAudio:ref({pause:()=>paused++}),bank:ref(media),store:{blob:()=>blob.promise,close:noop},MediaBank,checked:ref(true),heard:ref(true),runner:ref(null),busy:ref(false),error:ref(''),onBeforeUnmount:(fn:()=>void)=>unmount=fn,monitor:null,window:{removeEventListener:noop},document:{removeEventListener:noop},navigator:{},down:noop,up:noop,blur:noop,pagehide:noop,beforeUnload:noop,visibility:noop,fullscreen:noop,freeze:noop,resumeEvent:noop,device:noop,emit:noop,URL:{revokeObjectURL:()=>{},createObjectURL:()=>{created++;return 'blob:late';}}};
  const api=component('modules/perception/PerceptionPage.vue',['action','stopPreview','audition'],env,['previewUrl','previewEpoch','disposed'],[/^watch\(\(\)=>props.active/,/^onBeforeUnmount\(/]);
  const audition=api.audition(asset);await turn();
  if(transition==='blob'){decode.resolve(prepared);await turn();props.active=false;await nextTick();blob.resolve(new Blob(['wav']));}
  else{if(transition==='dispose')unmount();else{props.active=false;await nextTick();}decode.resolve(prepared);}
  await audition;assert.equal(preview.value,null);assert.equal(discarded,1);assert.equal(created,0);assert(paused>0);
  props.active=true;await nextTick();assert.equal(preview.value,null);
 }
});

test('M15 discarding an obsolete preview releases its allocation without evicting a newer stimulus',t=>{
 const revoked:string[]=[];t.mock.method(URL,'revokeObjectURL',(url:string)=>revoked.push(url));
 const bank=Object.create(MediaBank.prototype) as MediaBank;
 const old:any={asset:{hash:'same'},bytes:12,url:'blob:old'},fresh:any={asset:{hash:'same'},bytes:20,url:'blob:fresh'};
 bank.cache=new Map([['same',fresh]]);bank.used=20;bank.discardPreview(old);
 assert.equal(bank.cache.get('same'),fresh);assert.equal(bank.used,20);assert.deepEqual(revoked,['blob:old']);
 bank.discardPreview(fresh);assert.equal(bank.cache.size,0);assert.equal(bank.used,0);
});

test('M15 current active audition still publishes its audio and stop revokes the URL',async()=>{
 const asset={hash:'sha'},prepared={asset,buffer:{},bytes:12},preview=ref<any>(null),revoked:string[]=[];
 const api=component('modules/perception/PerceptionPage.vue',['action','stopPreview','audition'],{props:{active:true},preview,previewAudio:ref(),bank:ref({unlock:async()=>{},preview:async()=>prepared}),store:{blob:async()=>new Blob(['wav'])},MediaBank,checked:ref(true),heard:ref(true),busy:ref(false),error:ref(''),URL:{createObjectURL:()=> 'blob:current',revokeObjectURL:(url:string)=>revoked.push(url)}},['previewUrl','previewEpoch','disposed']);
 await api.audition(asset);assert.equal(preview.value.url,'blob:current');assert(preview.value.buffer);api.stopPreview();assert.equal(preview.value,null);assert.deepEqual(revoked,['blob:current']);
});

test('M18 actual reader key changes with root, document version and context while unchanged refresh stays stable',()=>{
 assert.match(source('modules/paper-reading/PaperReadingPage.vue'),/<PdfReader[^>]*:key="readerKey"/);
 const env={ref,computed,props:reactive({stateKey:'project:A'}),source:ref('local'),local:ref<any>({root:'',documents:[]}),localRevision:ref(0),localSelected:ref(''),serverSelected:ref(''),availableNotice:ref(),serverPapers:ref([]),error:ref('')};
 const api=component('modules/paper-reading/PaperReadingPage.vue',['applyLocal'],env,['current','readerKey']);
 const library=(root:string,version?:string,size=12)=>({root,folders:[],documents:[{id:'local_same',title:'same',titleZh:'same',relative:'same.pdf',size,publishedAt:'2026-10-24',contentVersion:version}]});
 api.applyLocal(library('A','v1'));const first=api.readerKey.value;
 api.applyLocal(library('A','v1'));assert.equal(api.readerKey.value,first);
 api.applyLocal(library('B','v2',24));assert.notEqual(api.readerKey.value,first);assert.equal(api.current.value.original.size,24);
 api.applyLocal(library('A','v1'));assert.equal(api.readerKey.value,first);
 api.applyLocal(library('A','v3',30));assert.notEqual(api.readerKey.value,first);
 const changed=api.readerKey.value;env.props.stateKey='project:B';assert.notEqual(api.readerKey.value,changed);
 api.applyLocal(library('old-server'));const legacy=api.readerKey.value;api.applyLocal(library('old-server'));assert.notEqual(api.readerKey.value,legacy);
});

test('AppShell M18 template key remounts the reader port when project context changes',()=>{
 const template=parse(source('app/AppShell.vue')).descriptor.template!.content;
 const tree=baseParse(template);let page:any;
 function visit(node:any){if(node.type===NodeTypes.ELEMENT&&node.tag==='PaperReadingPage')page=node;for(const child of node.children??[])visit(child);}
 visit(tree);assert(page);const key=page.props.find((p:any)=>p.type===NodeTypes.DIRECTIVE&&p.name==='bind'&&p.arg?.content==='key');assert(key);
 const binding=new Function('researchContext','return '+key.exp.content);const ports:string[]=[],disposed:string[]=[];
 const Child={props:['port'],setup(props:any){ports.push(props.port);onUnmounted(()=>disposed.push(props.port));return()=>h('reader');}};
 const renderer=createRenderer<any,any>({createElement:()=>({children:[]}),insert:(node,parent)=>parent.children.push(node),remove:()=>{},patchProp:()=>{},createText:()=>({}),createComment:()=>({}),setText:()=>{},setElementText:()=>{},parentNode:()=>null,nextSibling:()=>null});
 const root={children:[]},a={key:'project:A',files:{papers:'port:A'}},b={key:'project:B',files:{papers:'port:B'}};
 renderer.render(h(Child,{key:binding(a),port:a.files.papers}),root);renderer.render(h(Child,{key:binding(b),port:b.files.papers}),root);
 assert.deepEqual(ports,['port:A','port:B']);assert.deepEqual(disposed,['port:A']);renderer.render(null,root);
});

function failedPort(){let blocked=true,computes=0;const bytes={files:[{name:'m06.ptb.json',data:btoa(JSON.stringify({config:defaults(),action:'generate',spectrograms:{}}))},{name:'parameters.csv',data:btoa('csv')}]};
 const port=browserM06({run:async()=>{throw Error('unexpected remote');}} as unknown as M06Port,{project:'p',compute:async()=>{computes++;return bytes;},save:async(name,raw)=>{if(blocked)throw Error('disk full');return {id:name,sha256:'sha',size_bytes:raw.byteLength};}});
 return {port,unblock:()=>{blocked=false;},computes:()=>computes};
}

test('M06 port retains failed bytes across new task attempts, cancellation and explicit reopen',async()=>{
 const f=failedPort();let id='';await assert.rejects(f.port.run('generate',defaults(),undefined,new AbortController().signal,j=>id=j.id),/disk full/);
 for(const action of ['generate','synthesize','extract'] as const)await assert.rejects(f.port.run(action,defaults(),undefined,new AbortController().signal,()=>{}),/待保存/);
 assert.equal(f.computes(),1);assert.equal(f.port.pendingSave!()!.job.id,id);
 const aborted=new AbortController();aborted.abort();await assert.rejects(f.port.retrySave!(id,aborted.signal,()=>{}),{name:'AbortError'});assert(f.port.canRetrySave!(id));
 f.unblock();const result=await f.port.retrySave!(id,new AbortController().signal,()=>{});assert.equal(result.files.length,2);assert.equal(f.computes(),1);assert.equal(f.port.pendingSave!(),undefined);
 const discarded=failedPort();await assert.rejects(discarded.port.run('generate',defaults(),undefined,new AbortController().signal,()=>{}));const pending=discarded.port.pendingSave!()!;discarded.port.discardSave!(pending.job.id);assert.equal(discarded.port.pendingSave!(),undefined);
 discarded.unblock();await discarded.port.run('generate',defaults(),undefined,new AbortController().signal,()=>{});assert.equal(discarded.computes(),2);
});

test('M06 actual SFC restores pending retry, guards new work and blocks close until results and draft save',async()=>{
 const f=failedPort();await assert.rejects(f.port.run('generate',defaults(),undefined,new AbortController().signal,()=>{}));
 const env:any={ref,computed,port:ref(f.port),job:ref(),retryContext:ref(),config:defaults(),revision:ref(7),wave:{dirty:false},notice:ref(''),error:ref(''),busy:ref(false),durationText:ref('2'),low:ref('50'),high:ref('500'),overrideText:ref(''),name:ref('F0'),props:{stateKey:'server:test'},valid,clone,appliedOverride:()=>'',override:()=>{},replace:()=>{},host:{projects:{write:()=>true}},requireApplied:()=>{throw Error('new task reached validation');},epoch:0,disposed:false,abort:undefined,applyCompleted:async()=>{}};
 const api=component('modules/speech-synthesis/SpeechSynthesisPage.vue',['restorePendingSave','discardPendingSave','save','attempt','retrySave','run'],env,['canRetrySave']);
 api.restorePendingSave();assert.equal(env.wave.dirty,true);assert.equal(env.retryContext.value.revision,7);assert(api.canRetrySave.value);
 await api.run('generate');assert.match(env.error.value,/待保存/);assert.equal(f.computes(),1);
 assert.equal(await api.save(),false);assert(f.port.pendingSave!());assert.equal(env.wave.dirty,true);
 f.unblock();assert.equal(await api.save(),true);assert.equal(f.port.pendingSave!(),undefined);assert.equal(env.wave.dirty,false);assert.equal(f.computes(),1);
});

test('M06 pending first write already participates in dirty and close negotiation',async()=>{
 const write=deferred<void>(),wave={dirty:false},job=ref();
 const port=browserM06({} as M06Port,{project:'p',compute:async()=>({files:[{name:'m06.ptb.json',data:btoa('{}')},{name:'parameters.csv',data:btoa('csv')}]}),save:async(name,raw)=>{await write.promise;return {id:name,sha256:'sha',size_bytes:raw.byteLength};}});
 const api=component('modules/speech-synthesis/SpeechSynthesisPage.vue',[],{computed,port:ref(port),job},['canRetrySave']);
 const controller=new AbortController(),running=port.run('generate',defaults(),undefined,controller.signal,j=>job.value=j);await turn();
 assert.equal(api.canRetrySave.value,true);assert.equal(wave.dirty,false);
 const shell=component('app/AppShell.vue',[],{states:{M06:wave},previewKey:(id:string)=>id,m06Page:ref({hasPendingSave:()=>api.canRetrySave.value})},['moduleDirty']);
 assert.equal(shell.moduleDirty('M06'),true);
 assert.throws(()=>port.discardSave!(job.value!.id),/正在保存/);
 write.resolve();await running;assert.equal(shell.moduleDirty('M06'),false);assert.equal(wave.dirty,false);
});

test('AppShell M06 close saves asynchronously and releases pending bytes only after explicit discard',async()=>{
 const waiting=deferred<boolean>();let removed=0,discarded=0;
 const env={closing:ref('M06'),closingBusy:ref(false),m06Page:ref({save:()=>waiting.promise,discardPendingSave:()=>{discarded++;}}),notice:ref(''),remove:()=>removed++,finishRecordingClose:async()=>{}};
 const api=component('app/AppShell.vue',['saveClose','discardClose'],env);
 const saving=api.saveClose();assert.equal(removed,0);assert.equal(env.closingBusy.value,true);waiting.resolve(false);await saving;assert.equal(removed,0);assert.equal(discarded,0);
 const discard=env.m06Page.value.discardPendingSave;env.m06Page.value.discardPendingSave=()=>{throw Error('请先取消当前任务');};await api.discardClose();assert.equal(removed,0);assert.match(env.notice.value,/取消/);env.m06Page.value.discardPendingSave=discard;
 await api.discardClose();assert.equal(discarded,1);assert.equal(removed,1);
});
