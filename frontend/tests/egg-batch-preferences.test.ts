import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse} from '@vue/compiler-sfc';
import ts from 'typescript';
import {ref} from 'vue';
import {taskConfig,validateParameters,batchDefaults,eggErrorMessage} from '../src/modules/egg-analysis/state.ts';
import {preflightExportFonts} from '../src/modules/egg-analysis/fonts.ts';
import {uiMessage,uiErrorMessage,joinMessages} from '../src/modules/module-localization.ts';
import {AccountPreferences,mergePreferences} from '../src/platform/account-preferences.ts';

const source=parse(readFileSync(new URL('../src/modules/egg-analysis/EggAnalysisPage.vue',import.meta.url),'utf8')).descriptor.scriptSetup!.content;
const ast=ts.createSourceFile('EggAnalysisPage.ts',source,ts.ScriptTarget.ES2022,true,ts.ScriptKind.TS);
function evaluate(source:string,bindings:Record<string,unknown>,result:string){
 const js=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
 return new Function(...Object.keys(bindings),js+';return '+result)(...Object.values(bindings));
}
const font={schema_version:'font/1',zh:'SimSun',latin:'Times New Roman',ipa:'Doulos SIL',size_px:12,fallback_policy:'portable'};

test('actual M03 batch submission retains task fonts while persisting scientific parameters only',async()=>{
 const fn=ast.statements.find(node=>ts.isFunctionDeclaration(node)&&node.name?.text==='batchSubmit')!;assert(fn);
 const batchConfig=ref<any>(batchDefaults()),writes:any[]=[],submitted:any[]=[],batchIds=ref<string[]>([]),busy=ref(false);
 const parameters={...batchDefaults(),highpass_cutoff:31,lowpass_cutoff:2200,keep_gci_f0:false};
 // Only acceptance/transport is substituted: no scientific result is simulated.
 const run=evaluate(fn.getText(ast),{taskConfig,validateParameters,preflightExportFonts,exportFontSnapshot:()=>font,tasks:ref({egg:async(file:any,config:any)=>{submitted.push({file,config});return {id:'queued-job',state:'queued'};}}),batchBusy:busy,batchEpoch:0,batchError:ref(''),submittingBatchIds:[],disposed:false,batchIds,batchConfig,host:{projects:{write:(key:string,value:any)=>{writes.push({key,value});return true;}}},props:{stateKey:'owned:batch'},labelJob:()=>{},poll:async()=>{},batchOpen:ref(true),notice:ref<any>(''),joinMessages,uiMessage,uiErrorMessage,eggErrorMessage},'batchSubmit');
 await run([{id:'original-file-id',name:'original.wav'}],parameters);
 assert.equal(submitted.length,1);assert.deepEqual(submitted[0].config.font,font);assert.equal(submitted[0].file.id,'original-file-id');
 const persisted=writes.find(row=>row.key==='egg.batch-config.owned:batch').value;
 assert(!Object.hasOwn(persisted,'font'));assert(!Object.hasOwn(batchConfig.value,'font'));
 assert.equal(persisted.highpass_cutoff,31);assert.equal(persisted.lowpass_cutoff,2200);assert.equal(persisted.keep_gci_f0,false);
 assert.equal(submitted[0].config.highpass_cutoff,31);assert.equal(busy.value,false);assert.deepEqual(batchIds.value,['queued-job']);
 assert.deepEqual(parameters,{...batchDefaults(),highpass_cutoff:31,lowpass_cutoff:2200,keep_gci_f0:false});
});

test('actual old local M03 configuration retains parameters and drops the transient font projection',()=>{
 const from=ast.statements.find(node=>ts.isVariableStatement(node)&&node.declarationList.declarations.some(d=>d.name.getText(ast)==='savedBatchConfig'))!.getStart(ast),last=ast.statements.find(node=>ts.isVariableStatement(node)&&node.declarationList.declarations.some(d=>d.name.getText(ast)==='batchConfig'))!;
 const saved={...batchDefaults(),font,highpass_cutoff:37,keep_gci_f0:false};
 const restored=evaluate(source.slice(from,last.end),{host:{projects:{read:()=>saved}},props:{stateKey:'existing-local'},batchDefaults,ref},'batchConfig.value');
 assert(!Object.hasOwn(restored,'font'));assert.equal(restored.highpass_cutoff,37);assert.equal(restored.keep_gci_f0,false);assert.equal(saved.font,font);
});

function memory(values:Record<string,string>={}){const rows=new Map(Object.entries(values));return {getItem:(key:string)=>rows.get(key)??null,setItem:(key:string,value:string)=>rows.set(key,value),removeItem:(key:string)=>rows.delete(key),rows};}
function server(){
 const values:any={};const writes:any[]=[];let revision=0;
 const fetcher:typeof fetch=async(url,init)=>{
  if(!init?.body)return Response.json({allowed_fields:{M03:['batchConfig','order'],M04:['draft']},modules:[]});
  const module=String(url).split('/').at(-1)!,body=JSON.parse(String(init.body));writes.push({module,...body});
  if(body.changes.batchConfig&&Object.keys(body.changes.batchConfig).some(key=>['font','unknown_control'].includes(key)))return Response.json({detail:'invalid_settings_field'},{status:422});
  values[module]=mergePreferences(values[module]??{},body.changes);return Response.json({module,revision:++revision,values:values[module],updated_at:'2026-10-30T00:00:00Z'});
 };
 return {values,writes,fetcher};
}
async function withStorage(session:ReturnType<typeof memory>,local:ReturnType<typeof memory>,run:()=>Promise<void>){
 const priorSession=globalThis.sessionStorage,priorLocal=globalThis.localStorage;
 Object.defineProperty(globalThis,'sessionStorage',{configurable:true,value:session});Object.defineProperty(globalThis,'localStorage',{configurable:true,value:local});
 try{await run();}finally{Object.defineProperty(globalThis,'sessionStorage',{configurable:true,value:priorSession});Object.defineProperty(globalThis,'localStorage',{configurable:true,value:priorLocal});}
}

test('old pending M03 font is recovered before replay while other scientific edits survive',async()=>{
 const key='ptb.v3.settings-pending.v2.alice',session=memory({[key]:JSON.stringify({M03:{batchConfig:{font,highpass_cutoff:31,keep_gci_f0:false}},M04:{draft:{order:'72'}}})});
 await withStorage(session,memory(),async()=>{
  const backend=server(),preferences=new AccountPreferences('alice',()=> 'csrf',backend.fetcher);
  try{await preferences.open();assert.deepEqual(backend.values.M03,{batchConfig:{highpass_cutoff:31,keep_gci_f0:false}});assert.equal(backend.values.M04.draft.order,'72');assert.equal(preferences.pending.size,0);assert.equal(session.getItem(key),null);}
  finally{preferences.close();}
 });
});

test('old local account configuration recovers only the known font field; unknown fields still fail',async()=>{
 const key='ptb.v3.settings-pending.v2.alice',session=memory(),local=memory({'ptb.v3.web-settings.v1.alice.M03':JSON.stringify({batchConfig:{font,lowpass_cutoff:2200},order:64})});
 await withStorage(session,local,async()=>{
  const backend=server(),preferences=new AccountPreferences('alice',()=> 'csrf',backend.fetcher);
  try{await preferences.open();assert.deepEqual(backend.values.M03,{batchConfig:{lowpass_cutoff:2200},order:64});}finally{preferences.close();}
  local.rows.clear();session.setItem(key,JSON.stringify({M03:{batchConfig:{font,unknown_control:'retain error'}}}));
  const failed=new AccountPreferences('alice',()=> 'csrf',server().fetcher);
  try{await assert.rejects(failed.open(),/invalid_settings_field/);assert.equal(JSON.parse(session.getItem(key)!).M03.batchConfig.unknown_control,'retain error');assert(!Object.hasOwn(JSON.parse(session.getItem(key)!).M03.batchConfig,'font'));}finally{failed.close();}
 });
});
