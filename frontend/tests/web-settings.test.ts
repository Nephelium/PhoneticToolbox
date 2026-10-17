import test from 'node:test';
import assert from 'node:assert/strict';
import {ref,reactive,nextTick,watch,computed} from 'vue';
import {recordingDefaults,recordingSettings} from '../src/modules/recording/defaults.ts';
import {readFileSync} from 'node:fs';
import {connectWebSettings,webSettingsKey,settingsFields,webSettingsError,readWebSettings,writeWebSettings} from '../src/state/web-settings.ts';
import {serverPapers} from '../src/platform/papers-server.ts';

function storage(){const values=new Map<string,unknown>();return {values,read<T>(key:string,fallback:T):T{return structuredClone(values.get(key)??fallback) as T;},write(key:string,value:unknown){values.set(key,structuredClone(value));return true;}};}

test('recording preferences retain editable input settings after the browser supplies device metadata',()=>{
 const store=storage(),scope='server:recording:p:M16',config=ref({...recordingDefaults(),device_name:'Synthetic input',hostapi:'Browser'});
 const persisted=computed({get:()=>recordingSettings(config.value),set:value=>{config.value={...config.value,...value};}});
 const connection=connectWebSettings(scope,{config:persisted},store);
 config.value={...config.value,device:'input-2',sample_rate:44100,channels:1,roles:['egg'],gain_db:3};connection.flush();connection.stop();
 assert.deepEqual(readWebSettings(scope,store),{config:recordingSettings(config.value)});
 assert.equal(config.value.device_name,'Synthetic input');assert.equal(config.value.roles[0],'egg');
 assert.equal('device_name' in (readWebSettings(scope,store).config as object),false);
});
test('settings survive logout/new workspace and remain isolated by account and module',()=>{
 const store=storage(),order=ref('50'),first=connectWebSettings('server:alice:old-project:M04',{order},store);
 order.value='72';first.stop();
 const restored=ref('50'),second=connectWebSettings('server:alice:new-project:M04',{order:restored},store);
 assert.equal(restored.value,'72');second.stop();
 for(const scope of ['server:bob:new-project:M04','server:alice:new-project:M03']){const other=ref('50'),c=connectWebSettings(scope,{order:other},store);assert.equal(other.value,'50');c.stop();}
 assert.equal(webSettingsKey('M04'),null);assert.equal(webSettingsKey('local:M04'),null);
 const desktop=ref('50'),c=connectWebSettings('M04',{order:desktop},store);desktop.value='90';c.stop();assert.equal(store.values.size,1);
});
test('automatic save is debounced, flushes on stop, and stores only declared settings',async()=>{
 const store=storage(),state=reactive({order:'50',dynamic:false,asset:{samples:[1,2,3]},result:'private result'});
 const c=connectWebSettings('server:a:p:M04',settingsFields(state,['order','dynamic']),store);
 state.order='88';state.dynamic=true;await new Promise(resolve=>setTimeout(resolve,240));
 assert.deepEqual([...store.values.values()],[{order:'88',dynamic:true}]);
 state.order='89';c.stop();assert.equal((store.values.values().next().value as any).order,'89');
});
test('malformed values keep defaults and unknown fields are never hydrated',()=>{
 const store=storage(),scope='server:a:p:M04';store.values.set(webSettingsKey(scope)!,{order:{injected:true},dynamic:'true',asset:{samples:[99]}});
 const state=reactive({order:'50',dynamic:false,asset:null}),c=connectWebSettings(scope,settingsFields(state,['order','dynamic']),store);
 assert.deepEqual(state,{order:'50',dynamic:false,asset:null});c.stop();
});
test('Vue numeric inputs restore the declared editable string or number representation',()=>{
 const store=storage(),scope='server:number:p:M04';writeWebSettings(scope,{order:72,volume:'0.5'},store);
 const order=ref('50'),volume=ref(0.7),c=connectWebSettings(scope,{order,volume},store);
 assert.equal(order.value,'72');assert.equal(volume.value,0.5);c.stop();
});
test('storage failure stays visible and a successful retry clears it',()=>{
 const store=storage(),scope='server:quota:p:M04';let blocked=true;
 const failing={...store,write:(key:string,value:unknown)=>blocked?false:store.write(key,value)};
 const order=ref('50'),c=connectWebSettings(scope,{order},failing);order.value='64';c.flush();assert.match(webSettingsError.value,/暂未保存/);
 blocked=false;c.flush();assert.equal(webSettingsError.value,'');assert.equal(readWebSettings(scope,store).order,'64');c.stop();
});
test('parameter-only pages clear their saved marker after hydration watchers and successful write',async()=>{
 const store=storage(),order=ref('50');let dirty=true;
 writeWebSettings('server:marker:p:M04',{order:72},store);
 const unwatch=watch(order,()=>dirty=true);
 const c=connectWebSettings('server:marker:p:M04',{order},store,()=>dirty=false);
 await nextTick();assert.equal(order.value,'72');assert.equal(dirty,false);
 dirty=true;order.value='74';await nextTick();c.flush();assert.equal(dirty,false);c.stop();unwatch();
});
test('independent preference sections merge without replacing reader or server preferences',()=>{
 const store=storage(),scope='server:a:p:M18';writeWebSettings(scope,{paperInterval:12,paperAutoReceive:false},store);
 const zoom=ref(100),c=connectWebSettings(scope,{readerZoom:zoom},store);zoom.value=150;c.stop();
 assert.deepEqual(readWebSettings(scope,store),{paperInterval:12,paperAutoReceive:false,readerZoom:150});
});

test('repeated edits from an open page preserve a newer device value in an unchanged control',()=>{
 const store=storage(),scope='server:concurrent:p:M04';writeWebSettings(scope,{draft:{order:72,amp_max_db:40}},store);
 const draft=ref({order:'50',amp_max_db:'35'}),c=connectWebSettings(scope,{draft},store);
 // A revision-conflict refresh can update the transport cache while this page
 // still displays the old order. Only actual edits from the page are submitted.
 writeWebSettings(scope,{draft:{order:85}},store);
 draft.value.amp_max_db='45';c.flush();draft.value.amp_max_db='46';c.flush();
 assert.deepEqual(readWebSettings(scope,store),{draft:{order:85,amp_max_db:'46'}});c.stop();
});
test('paper auto-receive and the supported 15-minute interval are reapplied in a new workspace',async()=>{
 const oldStorage=globalThis.localStorage,oldFetch=globalThis.fetch,values=new Map<string,string>(),operations:string[]=[];
 Object.defineProperty(globalThis,'localStorage',{configurable:true,value:{getItem:(k:string)=>values.get(k)??null,setItem:(k:string,v:string)=>values.set(k,v)}});
 let autoReceive=true,checkIntervalHours=6;
 globalThis.fetch=async(_url,init)=>{
  const {operation,args}=JSON.parse(String(init?.body));operations.push(operation);
  if(operation==='autoReceive')autoReceive=args.enabled;if(operation==='interval')checkIntervalHours=args.hours;
  return new Response(JSON.stringify({ok:true,value:{firstLaunch:'2026-10-16',warning:'',papers:[],autoReceive,checkIntervalHours}})+'\n');
 };
 try{
  const signal=new AbortController().signal,first=serverPapers('a','old',()=>'',()=>{},signal);
  await first.setAutoReceive(false);await first.setInterval(0.25);autoReceive=true;checkIntervalHours=6;operations.length=0;
  const second=serverPapers('a','new',()=>'',()=>{},signal),status=await second.status();
  assert.equal(status.autoReceive,false);assert.equal(status.checkIntervalHours,0.25);assert.deepEqual(operations,['status','autoReceive','interval']);
 }finally{globalThis.fetch=oldFetch;Object.defineProperty(globalThis,'localStorage',{configurable:true,value:oldStorage});}
});
test('every registered module has an explicit persistence adapter',()=>{
 const pages=['parameter-estimation/ParameterEstimationPage','parameter-display/ParameterDisplayPage','egg-analysis/EggAnalysisPage','lpc-spectrum/LpcSpectrumPage','lip-extraction/LipExtractionPage','speech-synthesis/SpeechSynthesisPage','phonation-synthesis/PhonationSynthesisPage','pitch-manipulation/PitchManipulationPage','spectrogram-to-audio/Spec2WavPage','mfa/MfaAlignmentPage','annotation/AnnotationPage','mandarin-ipa/MandarinIpaPage','phonology-induction/PhonologyInductionPage','perception/PerceptionPage','recording/RecordingPage','ipa-plus/IpaPlusPage','paper-reading/PaperReadingPage'];
 for(const page of pages)assert.match(readFileSync(new URL('../src/modules/'+page+'.vue',import.meta.url),'utf8'),/useWebSettings\(props\./,page);
 assert.match(readFileSync(new URL('../src/modules/vocal-tract/VocalTractPage.vue',import.meta.url),'utf8'),/webSettingsKey\(props.stateKey\)/);
});
