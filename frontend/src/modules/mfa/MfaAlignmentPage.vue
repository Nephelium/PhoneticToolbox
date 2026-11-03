<script setup lang="ts">
import {uiMessage as localizedMessage} from '../module-localization.ts';
import {uiError as displayError,uiMessage as displayMessage,uiErrorMessage as displayErrorValue} from '../module-localization.ts';

import {tr,messageRef} from '../../i18n/index.ts';
import {useWebSettings,settingsFields} from '../../state/web-settings.ts';
import {syncNumberInput} from '../../design/numeric-input.ts';
import AppIcon from '../../components/AppIcon.vue';
import {computed,onMounted,onUnmounted,ref,watch} from 'vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import ModuleSection from '../../components/ModuleSection.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import ModalDialog from '../../components/ModalDialog.vue';
import type {ResearchContext,JobView} from '../../platform/research.ts';
import {projects} from '../../platform/browser.ts';
import {m11Message} from '../../platform/m11.ts';
import {defaults,beamChanged,validate,type TranscriptSource} from './state.ts';
import type {Catalog,CorpusItem,Grant} from './port.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active?:boolean;presentationOnly?:boolean}>();
const emit=defineEmits<{references:[];dirty:[boolean]}>();
const port=computed(()=>props.presentationOnly?undefined:props.context.files.m11);
const key='m11:'+props.stateKey;
const saved=projects.read<{beam:number;retry_beam:number;runtime:string;model:string;transcriptSource?:TranscriptSource}>(key,{...defaults(),runtime:'',model:''});
const config=ref({beam:saved.beam,retry_beam:saved.retry_beam}),runtime=ref(saved.runtime),model=ref(saved.model);
const transcriptSource=ref<TranscriptSource>(saved.transcriptSource??'auto'),loadedSource=ref<TranscriptSource>();
const catalog=ref<Catalog>(),corpus=ref<CorpusItem[]>([]);
const importedDictionaries=ref<{id:string;name:string;asset:{asset_id:string;sha256:string};grant?:Grant}[]>([]),customDictionaryId=ref('');
const dictionary=computed(()=>importedDictionaries.value.find(d=>d.id===customDictionaryId.value)?.asset);
const output=ref<Grant>(),grants=ref<Record<string,Grant>>({});
const installing=ref(false);
const bytesLabel=(bytes:number)=>bytes>=1_000_000_000?`${(bytes/1_000_000_000).toFixed(2)} GB`:bytes>=1_000_000?`${(bytes/1_000_000).toFixed(1)} MB`:`${Math.round(bytes/1000)} KB`;
const error=messageRef(),message=messageRef(),busy=ref(false),dirty=ref(false),help=ref(false),componentOpen=ref(false);
const fileInput=ref<HTMLInputElement>(),dictInput=ref<HTMLInputElement>();
const log=ref({text:'',truncated:false});
const jobs=ref<JobView[]>([]),selected=ref<JobView>(),events=ref<{sequence:number;code:string;created_at:number}[]>([]);
const running=computed(()=>!!selected.value&&['queued','running','cancel_requested'].includes(selected.value.state));
const runtimeInfo=computed(()=>catalog.value?.runtimes.find(r=>r.id===runtime.value));
const modelInfo=computed(()=>catalog.value?.models.find(m=>m.id===model.value));
const models=computed(()=>catalog.value?.models.filter(m=>m.validated_runtime===runtime.value)??[]);
const acousticModels=computed(()=>models.value.filter((m,i,all)=>all.findIndex(other=>other.model_sha256===m.model_sha256)===i));
const acousticSelection=computed(()=>grants.value.model?'pending-model':acousticModels.value.find(m=>m.model_sha256===modelInfo.value?.model_sha256)?.id??'');
const registeredDictionaries=computed(()=>models.value.filter(m=>m.model_sha256===modelInfo.value?.model_sha256));
const dictionarySelection=computed(()=>customDictionaryId.value||model.value);
const pendingResources=computed(()=>Object.keys(grants.value).length>0);
const canCheck=computed(()=>!!(port.value?.component&&!grants.value.archive&&!grants.value.manifest&&(grants.value.runtime||runtimeInfo.value?.validated)&&(grants.value.model||modelInfo.value)&&(grants.value.dictionary||(!customDictionaryId.value&&modelInfo.value))));
const componentProgress=computed(()=>{const s=catalog.value?.component_status;const label=({downloading:tr('modules.mfa.text.62a6ec88f598'),verifying:tr('modules.mfa.text.efc4740c05c0'),extracting:tr('modules.mfa.text.15b8afe1931c'),checking:tr('modules.mfa.text.4deb4e43a1db'),complete:tr('modules.mfa.text.57cf47f232a8'),failed:tr('modules.mfa.text.24ae4005170c')} as Record<string,string>)[s?.stage??''];return installing.value?(label||tr('modules.mfa.text.741544b351d6'))+(s?.total?` · ${Math.floor(100*s.completed/s.total)}%`:''):tr('modules.mfa.text.d6ce2757fae4');});
const selectedDictionaryName=computed(()=>importedDictionaries.value.find(d=>d.id===customDictionaryId.value)?.name||modelInfo.value?.dictionary_name||tr('modules.mfa.text.8bd40b830c9a'));
const sourceChanged=computed(()=>!!corpus.value.length&&loadedSource.value!==transcriptSource.value);
const resultFiles=computed(()=>selected.value?.result_manifest&&'files'in selected.value.result_manifest?selected.value.result_manifest.files:[]);
let timer:ReturnType<typeof setInterval>|undefined,alive=true,refreshing=false;
const controller=new AbortController();
watch([config,runtime,model,transcriptSource,customDictionaryId,grants],()=>{dirty.value=true;},{deep:true});
watch(()=>dirty.value||busy.value||running.value,value=>emit('dirty',value),{immediate:true});
function save(){if(busy.value){error.value=localizedMessage('modules.mfa.text.26f4a2553cf4');return false;}try{const values=validate(config.value);if(!projects.write(key,{...values,runtime:runtime.value,model:model.value,transcriptSource:transcriptSource.value}))throw displayError(displayMessage('modules.mfa.text.9e1a3be429f1'));dirty.value=false;message.value=localizedMessage('modules.mfa.text.448b446c969f');emit('dirty',running.value);return true;}catch(e){error.value=displayErrorValue(e);return false;}}
defineExpose({save});
function mfaDisplayError(cause:unknown){const value=displayErrorValue(cause);return typeof value==='string'?{key:'modules.display.composed',values:{get text(){return m11Message(value);}}}:value;}
async function operation(fn:()=>Promise<void>){if(busy.value)return;busy.value=true;error.value='';message.value='';try{await fn();}catch(e){if(alive)error.value=mfaDisplayError(e);}finally{if(alive)busy.value=false;}}
async function refresh(){
 if(!port.value||refreshing)return;refreshing=true;const selectedId=selected.value?.id;
 try{
  const [c,j]=await Promise.all([port.value.catalog(),port.value.history()]);if(!alive)return;
  catalog.value=c;jobs.value=j;
  if(!runtime.value&&c.runtimes.length)runtime.value=c.runtimes[0].id;
  if(!model.value&&models.value.length)model.value=models.value[0].id;
  if(selectedId&&selected.value?.id===selectedId){
   const updated=j.find(v=>v.id===selectedId);if(updated)selected.value=updated;
   const [eventList,nativeLog]=await Promise.all([port.value.events(selectedId),port.value.log?.(selectedId)]);
   if(alive&&selected.value?.id===selectedId){events.value=eventList.events;if(nativeLog)log.value=nativeLog;}
  }
 }catch(e){if(alive&&selected.value?.id===selectedId)error.value=mfaDisplayError(e);}finally{refreshing=false;}
}
watch(runtime,()=>{if(catalog.value&&!models.value.some(m=>m.id===model.value))model.value=models.value[0]?.id??'';});

onMounted(()=>{if(props.presentationOnly)return;void refresh();timer=setInterval(()=>{if(installing.value||(!busy.value&&props.active!==false))void refresh();},1000);});
onUnmounted(()=>{alive=false;controller.abort();if(timer)clearInterval(timer);});
const beamInput=computed({get:()=>config.value.beam,set:(value:number|string)=>{try{config.value=beamChanged(config.value,Number(value));error.value='';}catch(e){error.value=displayErrorValue(e);}}});
async function imported(event:Event){const input=event.target as HTMLInputElement,files=Array.from(input.files??[]);input.value='';if(!files.length||!port.value)return;await operation(async()=>{corpus.value=await port.value!.import(files,controller.signal,transcriptSource.value);loadedSource.value=transcriptSource.value;dirty.value=true;message.value=localizedMessage('modules.mfa.text.7cc9c3777bb0',{v0:corpus.value.length});});}
function chooseAcoustic(event:Event){const select=event.target as HTMLSelectElement,id=select.value;if(id==='import'){select.value=acousticSelection.value;void pick('model');}else if(id!=='pending-model'){model.value=id;customDictionaryId.value='';delete grants.value.model;delete grants.value.dictionary;}}
function chooseDictionary(event:Event){const select=event.target as HTMLSelectElement,id=select.value;if(id==='import'){select.value=dictionarySelection.value;if(port.value?.local&&port.value.pick&&port.value.dictionaryFromGrant)void importNativeDictionary();else dictInput.value?.click();}else{const imported=importedDictionaries.value.find(d=>d.id===id);if(imported){customDictionaryId.value=id;if(imported.grant)grants.value.dictionary=imported.grant;}else{customDictionaryId.value='';model.value=id;delete grants.value.dictionary;}}}
async function importNativeDictionary(){await operation(async()=>{const grant=await port.value!.pick!('dictionary');if(!grant)return;const asset=await port.value!.dictionaryFromGrant!(grant),id='custom-'+asset.asset_id;importedDictionaries.value.push({id,name:grant.label,asset,grant});customDictionaryId.value=id;grants.value.dictionary=grant;message.value=localizedMessage('modules.mfa.text.7c49f463bef8');});}
async function importDictionary(event:Event){const input=event.target as HTMLInputElement,file=input.files?.[0];input.value='';if(!file||!port.value)return;await operation(async()=>{const asset=await port.value!.dictionary(file,controller.signal),id='custom-'+asset.asset_id;importedDictionaries.value.push({id,name:file.name,asset});customDictionaryId.value=id;dirty.value=true;});}
async function chooseCorpus(){await operation(async()=>{const grant=await port.value!.pick!('corpus');if(!grant)return;corpus.value=await port.value!.corpus!(grant,transcriptSource.value);loadedSource.value=transcriptSource.value;dirty.value=true;message.value=localizedMessage('modules.mfa.text.3331e874dae1',{v0:corpus.value.length});});}
async function pick(purpose:string){await operation(async()=>{const value=await port.value!.pick!(purpose);if(value){grants.value[purpose]=value;message.value=localizedMessage('modules.mfa.text.3ea7ab958b5a');}});}
function discardResources(){grants.value={};customDictionaryId.value='';error.value='';message.value=localizedMessage('modules.mfa.text.e80986fe2e4c');}
async function applyComponent(values:Record<string,string>){const result=await port.value!.component!(values) as {runtime_id?:string;model_id?:string};if(!alive)return;const next=await port.value!.catalog();if(!alive)return;const applied=next.models.find(m=>m.id===result.model_id&&m.validated_runtime===result.runtime_id);if(!applied||!next.runtimes.some(r=>r.id===result.runtime_id&&r.validated))throw Error('m11_component_not_registered');catalog.value=next;runtime.value=result.runtime_id!;model.value=result.model_id!;customDictionaryId.value='';grants.value={};message.value=localizedMessage('modules.mfa.text.ef8e20656365',{v0:applied.name,v1:applied.dictionary_name||localizedMessage('modules.mfa.text.8bd40b830c9a')});const stored=projects.read(key,{...defaults(),runtime:'',model:''});if(!projects.write(key,{...stored,runtime:runtime.value,model:model.value}))message.value=localizedMessage('modules.mfa.applied.rememberFailed',{model:applied.name});}
async function check(action:'check'){await operation(()=>applyComponent({...Object.fromEntries(Object.entries(grants.value).map(([k,v])=>[k,v.id])),action,runtime_id:runtime.value,model_id:model.value}));}
async function installComponent(action:'download'|'install_zip'|'install_folder'){await operation(async()=>{const values:Record<string,string>={action};if(action!=='download'){const purpose=action==='install_zip'?'archive':'runtime';const grant=await port.value!.pick!(purpose);if(!grant)return;values[purpose]=grant.id;}installing.value=true;try{await applyComponent(values);}finally{installing.value=false;}});}
async function start(){await operation(async()=>{if(pendingResources.value)throw displayError(displayMessage('modules.mfa.text.4fe8b1190d78'));if(sourceChanged.value)throw displayError(displayMessage('modules.mfa.text.7c8eaaba2c18'));const settings=validate(config.value);selected.value=await port.value!.create({runtime_id:runtime.value,model_id:model.value,corpus:corpus.value,dictionary:dictionary.value??null,config:settings});events.value=[];await refresh();});}
async function cancel(){if(selected.value)await operation(async()=>{await port.value!.cancel(selected.value!.id);await refresh();});}
async function show(job:JobView){selected.value=job;events.value=[];log.value={text:'',truncated:false};await refresh();}
async function saveResults(){await operation(async()=>{if(!output.value)output.value=(await port.value!.chooseOutput!())??undefined;if(!output.value)return;const result=await port.value!.save!(selected.value!,output.value);message.value=localizedMessage('modules.mfa.text.2e8df72c3988',{v0:result.count,v1:output.value.label,v2:result.directory});});}
useWebSettings(props.stateKey,{config,transcriptSource});
</script>

<template>
<ModuleFrame unified fit class="mfa-page" :label="tr('modules.mfa.text.30e7735369d0')" :aria-busy="busy">
 <template #toolbar><ModuleToolbar>
  <button class="primary" v-if="presentationOnly||port?.local" :disabled="busy" @click="chooseCorpus"><AppIcon name="folder"/>{{tr('modules.mfa.text.05f4f621d83d')}}</button>
  <button class="primary" :disabled="busy||!port" @click="fileInput?.click()">{{tr('modules.mfa.text.e30de71c4122')}}</button>
  <input ref="fileInput" hidden type="file" multiple accept=".wav,.lab,.txt,.TextGrid" @change="imported"/>

  <template #actions><button :disabled="busy" @click="save">{{tr('modules.mfa.text.9a89fe66cf44')}}</button><button @click="help=true">{{tr('modules.mfa.text.a57cfcb8428d')}}</button><button @click="emit('references')"><AppIcon name="book"/>{{tr('modules.mfa.text.7eaa43419522')}}</button></template>
 </ModuleToolbar></template>
 <template #status><ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-else-if="busy" kind="loading" :message="componentProgress"/><ModuleStatus v-if="!port" kind="empty" :message="tr('modules.mfa.text.fc412d7f1a36')"/></template>
 <ModuleWorkbench unified :state-key="stateKey" :left-label="tr('modules.mfa.text.1918aeabe159')" :right-label="tr('modules.mfa.text.e9cacad71717')">
 <template #left>   <ModuleSection :label="tr('modules.mfa.text.9e395cabc099')" :title="tr('modules.mfa.text.9e395cabc099')">
    <div class="mfa-form"><label>{{tr('modules.mfa.text.423f51a28678')}}<select v-model="runtime" :disabled="busy||running"><option value="">{{tr('modules.mfa.text.98290d1e2221')}}</option><option v-for="r in catalog?.runtimes" :key="r.id" :value="r.id">MFA {{r.version}} · {{r.platform}} / {{r.arch}}</option></select></label>
     <label>{{tr('modules.mfa.text.e27f85d2b4ee')}}<select :value="acousticSelection" :disabled="busy||running" @change="chooseAcoustic"><option value="">{{tr('modules.mfa.text.251a1ff6b764')}}</option><option v-for="m in acousticModels" :key="m.id" :value="m.id">{{m.name}}</option><option v-if="grants.model" value="pending-model">{{grants.model.label}} {{tr('modules.mfa.text.64fedb0ffdd9')}}</option><option v-if="presentationOnly||(port?.local&&port.pick)" value="import">{{tr('modules.mfa.text.67a7f8785aae')}}</option></select></label>
     <label>{{tr('modules.mfa.text.4a8e0e20dfd3')}}<select :value="dictionarySelection" :disabled="busy||!port||running" @change="chooseDictionary"><option v-if="!model" value="">{{tr('modules.mfa.text.6fdf0fda72ac')}}</option><option v-for="m in registeredDictionaries" :key="m.id" :value="m.id">{{m.dictionary_name||tr('modules.mfa.text.8bd40b830c9a')}}</option><option v-for="d in importedDictionaries" :key="d.id" :value="d.id">{{d.name}} · {{grants.dictionary?.id===d.grant?.id&&d.grant?tr('modules.mfa.text.2329bc02cd04'):tr('modules.mfa.text.f62f58be54a1')}}</option><option value="import">{{tr('modules.mfa.text.680d1a77c0e2')}}</option></select></label>
     <input ref="dictInput" hidden type="file" accept=".dict,.txt" @change="importDictionary"/>
       <div class="official-resources"><a class="button" href="https://mfa-models.readthedocs.io/en/latest/acoustic/index.html" target="_blank" rel="noopener noreferrer">{{tr('modules.mfa.text.61f65054b398')}}</a><a class="button" href="https://mfa-models.readthedocs.io/en/latest/dictionary/index.html" target="_blank" rel="noopener noreferrer">{{tr('modules.mfa.text.e623893a006f')}}</a></div>
       <p class="muted">{{tr('modules.mfa.text.60059266a8d3')}}</p>
    </div>
    <p v-if="modelInfo" class="muted applied-resources">{{tr('modules.mfa.appliedResources',{model:modelInfo.name,dictionary:pendingResources?modelInfo.dictionary_name||tr('modules.mfa.text.8bd40b830c9a'):selectedDictionaryName})}}</p>
    <ModuleStatus v-if="pendingResources" kind="info" :message="tr('modules.mfa.text.cc6654941eec')"/>
    <div v-if="pendingResources" class="actions resource-actions"><button class="primary" :disabled="busy||running||!canCheck" @click="check('check')">{{tr('modules.mfa.text.855a4c77c959')}}</button><button :disabled="busy||running" @click="discardResources">{{tr('modules.mfa.text.6f7afe802103')}}</button></div>
    <details v-if="runtimeInfo" class="component-details"><summary>{{tr('modules.mfa.text.ab130334c2f5')}}</summary><p>{{tr('modules.mfa.text.6181ed6f9734')}}{{runtimeInfo.source||tr('modules.mfa.text.776e3102744e')}}</p><p>{{tr('modules.mfa.text.ff4a854b2170')}}{{runtimeInfo.download_bytes?.toLocaleString()??tr('modules.mfa.text.fa9ace7682e6')}} {{tr('modules.mfa.text.d520b073436e')}}{{runtimeInfo.installed_bytes?.toLocaleString()??tr('modules.mfa.text.e55f2ab3e631')}} B</p><p>{{tr('modules.mfa.text.2a845bd8e225')}}{{modelInfo?.model_bytes?.toLocaleString()??tr('modules.mfa.text.e55f2ab3e631')}} {{tr('modules.mfa.text.bdca98cc5314')}}{{modelInfo?.dictionary_bytes?.toLocaleString()??tr('modules.mfa.text.e55f2ab3e631')}} B</p><p>{{tr('modules.mfa.text.29a4b865173f')}}</p><dl><template v-for="(v,k) in runtimeInfo.versions" :key="k"><dt>{{k}}</dt><dd>{{v}}</dd></template></dl><p class="digest">{{tr('modules.mfa.text.9f60dde05850')}}{{runtimeInfo.fingerprint}}</p><p class="digest">{{tr('modules.mfa.text.f36b5fccb0c5')}}{{modelInfo?.model_sha256}}</p><p class="digest">{{tr('modules.mfa.text.78ff57ba7789')}}{{pendingResources?modelInfo?.dictionary_sha256:dictionary?.sha256??modelInfo?.dictionary_sha256}}</p><p v-if="runtimeInfo.archive_sha256" class="digest">{{tr('modules.mfa.text.724a584e7abb')}}{{runtimeInfo.archive_sha256}}</p></details>
    <ModuleStatus v-if="catalog?.waiting_reason" kind="info" :message="m11Message(catalog.waiting_reason)"/>
    <button v-if="presentationOnly||port?.local" class="component-toggle" :aria-expanded="componentOpen" @click="componentOpen=!componentOpen">{{componentOpen?tr('modules.mfa.text.436b893d7db9'):tr('modules.mfa.text.3faf5576b6e8')}}</button>
    <div v-if="componentOpen&&(presentationOnly||port?.local)" class="component-manager">
     <p class="muted">{{tr('modules.mfa.text.cf09fd1ff890')}}</p>
     <button class="primary" :disabled="busy||running||!catalog?.download_available" @click="installComponent('download')">{{tr('modules.mfa.text.0bbcd2c8d883')}}</button>
     <div v-if="installing" class="component-progress" aria-live="polite"><p>{{componentProgress}}</p><progress :value="catalog?.component_status?.total?catalog.component_status.completed:undefined" :max="catalog?.component_status?.total||1" :aria-label="tr('modules.mfa.text.e7a6345c63fa')"/><p v-if="catalog?.component_status?.stage==='downloading'" class="download-metrics"><span>{{bytesLabel(catalog.component_status.completed)}} / {{bytesLabel(catalog.component_status.total)}}</span><span>{{bytesLabel(catalog.component_status.bytes_per_second||0)}}/s</span></p></div>
     <p class="muted">{{tr('modules.mfa.text.81bd17dca323')}}</p>
     <button :disabled="busy||running" @click="installComponent('install_zip')">{{tr('modules.mfa.text.4d9ff797e182')}}</button>
     <button :disabled="busy||running" @click="installComponent('install_folder')">{{tr('modules.mfa.text.69a9b2a0f2a5')}}</button>
     <p class="muted">{{tr('modules.mfa.text.332edc9b147a')}}</p>
     <details><summary>{{tr('modules.mfa.text.db8bdb190756')}}</summary><p class="muted">{{tr('modules.mfa.text.3ae7883755d0')}}</p><div class="resource-line"><span>{{grants.runtime?.label||tr('modules.mfa.text.cd3f244871ff')}}</span><button :disabled="busy||running" @click="pick('runtime')">{{tr('modules.mfa.text.8ed03f1121f8')}}</button></div></details>
    </div>
   </ModuleSection>
<ModuleSection :label="tr('modules.mfa.text.7d6ad12d871c')" :title="tr('modules.mfa.text.7d6ad12d871c')">
    <div class="parameter-row"><label>Beam<input type="number" min="1" max="10000" v-model.lazy.number="beamInput" @change="syncNumberInput($event,beamInput)" :disabled="busy"/></label><label>Retry beam<input v-model.number="config.retry_beam" type="number" min="1" max="40000" :disabled="busy"/></label></div>
    <p class="muted">{{tr('modules.mfa.text.406a9e461c9a')}}</p>

   </ModuleSection>
</template>
   <ModuleSection :label="tr('modules.mfa.text.8ca3f45d00ba')" :title="tr('modules.mfa.text.8ca3f45d00ba')">
    <label>{{tr('modules.mfa.text.d5b27dc9b29a')}}<select v-model="transcriptSource" :disabled="busy"><option value="auto">{{tr('modules.mfa.text.16aae713a341')}}</option><option value=".lab">LAB</option><option value=".txt">TXT</option><option value=".TextGrid">TextGrid</option></select></label>
    <p class="muted">{{tr('modules.mfa.text.7f1f889a8fdd')}}</p>
    <ModuleStatus v-if="sourceChanged" kind="info" :message="tr('modules.mfa.text.7c8eaaba2c18')"/>
    <ModuleStatus v-if="!corpus.length" kind="empty" :message="tr('modules.mfa.text.101f6ccc62cc')"/>
    <ul v-else class="corpus-list"><li v-for="c in corpus" :key="c.name"><span>{{c.name}}</span><span class="muted">{{c.transcript_format}}</span></li></ul>
    <p class="muted">{{tr('modules.mfa.selectedPairsLimits',{count:corpus.length})}}</p>
    <div v-if="port?.chooseOutput" class="resource-line"><span>{{tr('modules.mfa.outputFolder',{folder:output?.label||tr('modules.mfa.text.83724296f785')})}}</span><button :disabled="busy" @click="operation(async()=>{output=(await port!.chooseOutput!())??undefined;})">{{tr('modules.mfa.text.537bdbd1e894')}}</button></div>
   </ModuleSection>

 <template #right><ModuleSection :label="tr('modules.mfa.text.38221b413e51')" :title="tr('modules.mfa.text.2c0643fcf211')"><div class="actions"><button class="primary" :disabled="busy||!port||!corpus.length||sourceChanged||pendingResources||!runtime||!model||running" @click="start">{{tr('modules.mfa.text.ffee4fd9dd65')}}</button><button :disabled="busy||!running||selected?.state==='cancel_requested'" @click="cancel">{{tr('modules.mfa.text.537d17f1c531')}}</button></div></ModuleSection>
 <ModuleStatus v-if="message" kind="info" :message="message"/>
 <button :disabled="busy||!port" @click="refresh">{{tr('modules.mfa.text.e4c9549fe2f4')}}</button>

   <ModuleSection :label="tr('modules.mfa.text.c9a4796852d2')" :title="tr('modules.mfa.text.c9a4796852d2')">
    <label>{{tr('modules.mfa.text.c542f21d8c90')}}<select :value="selected?.id??''" @change="show(jobs.find(j=>j.id===($event.target as HTMLSelectElement).value)!)"><option value="" disabled>{{tr('modules.mfa.text.8ca62b571eea')}}</option><option v-for="j in jobs" :key="j.id" :value="j.id">{{new Date(j.created_at*1000).toLocaleString()}} · {{m11Message(j.state)}}</option></select></label>
    <ModuleStatus v-if="!selected" kind="empty" :message="tr('modules.mfa.text.c1ddffcccdff')"/>
    <template v-else>
     <div class="job-state"><strong>{{m11Message(selected.state)}}</strong><span>{{Math.round(selected.progress*100)}}%</span></div><progress :value="selected.progress" max="1" :aria-label="tr('modules.mfa.text.0021f58940cd')"/>
     <ModuleStatus v-if="selected.waiting_reason&&selected.state==='queued'" kind="info" :message="m11Message(selected.waiting_reason)"/>
     <ModuleStatus v-if="selected.error_code" kind="error" :message="m11Message(selected.error_code)"/>
     <ol class="event-log" :aria-label="tr('modules.mfa.text.e13d8b56603b')"><li v-for="e in events" :key="e.sequence"><time>{{new Date(e.created_at*1000).toLocaleTimeString()}}</time><span>{{m11Message(e.code)}}</span></li></ol>
     <details v-if="log.text"><summary>{{tr('modules.mfa.text.273d15d79457')}}</summary><p v-if="log.truncated" class="muted">{{tr('modules.mfa.text.f1e2e5e51469')}}</p><pre class="native-log">{{log.text}}</pre></details>
     <p v-if="running" class="muted">{{tr('modules.mfa.text.ff16aaa741da')}}</p>
    </template>
   </ModuleSection>
   <ModuleSection :label="tr('modules.mfa.text.6014550816c4')" :title="tr('modules.mfa.text.6014550816c4')">
    <ModuleStatus v-if="!resultFiles.length" kind="empty" :message="tr('modules.mfa.text.8626e30d4a24')"/>
    <div v-for="f in resultFiles" :key="f.id" class="result-row"><span>{{f.name}}</span><span class="muted">{{f.size_bytes.toLocaleString()}} B</span><button :disabled="busy" @click="operation(()=>port!.download(selected!,f.id))">{{tr('modules.mfa.text.4673a2306165')}}</button></div>
    <button class="primary" v-if="resultFiles.length&&port?.save" :disabled="busy" @click="saveResults">{{tr('modules.mfa.text.0febea99e04f')}}</button>
    <p class="muted">{{tr('modules.mfa.text.d68ddd4b68bf')}}</p>
   </ModuleSection>
 </template>
 </ModuleWorkbench>
 <ModalDialog v-if="help" :title="tr('modules.mfa.text.eeadbdd2a7e7')" @close="help=false"><p>{{tr('modules.mfa.text.ab73a66043d8')}}</p><p>{{tr('modules.mfa.text.559bfe6ea81d')}}</p><p>{{tr('modules.mfa.text.e2492d1390d0')}}</p><p>{{tr('modules.mfa.text.a866cccc7e90')}}</p><p>{{tr('modules.mfa.text.72fe08c72fdc')}}</p></ModalDialog>
</ModuleFrame>
</template>

<style scoped>
.official-resources{display:grid;gap:var(--control-gap)}.official-resources a{display:flex;justify-content:center;align-items:center;min-height:34px;border:1px solid var(--border);border-radius:6px;background:var(--panel);padding:6px 12px;color:inherit;text-decoration:none;font-size:var(--control-size)}.official-resources a:hover{background:var(--selected);border-color:var(--accent)}
.component-progress p{margin:4px 0;font-size:var(--support-size)}.download-metrics{display:flex;justify-content:space-between;gap:8px;font-variant-numeric:tabular-nums}
.component-details{margin-top:10px;font-size:var(--support-size);overflow-wrap:anywhere}.component-details dl{display:grid;grid-template-columns:1fr 1fr;gap:4px}.component-details dd{margin:0}.native-log{white-space:pre-wrap;overflow-wrap:anywhere;max-height:320px;overflow:auto;font-family:var(--font-mono);font-size:var(--support-size)}.mfa-columns{display:grid;grid-template-columns:var(--panel-left,400px) minmax(300px,1fr);gap:var(--module-gap);align-items:start}.mfa-controls,.mfa-results{display:grid;gap:var(--module-gap);min-width:0}.mfa-form,.component-manager{display:grid;gap:var(--control-gap)}label{display:grid;gap:6px;font-size:var(--control-size)}select,input{width:100%;min-width:0}.resource-line,.actions,.parameter-row,.result-row,.job-state{display:flex;align-items:center;gap:var(--control-gap);flex-wrap:wrap}.resource-line{justify-content:space-between}.resource-line span,.result-row>span:first-child{overflow-wrap:anywhere;flex:1;min-width:140px}.parameter-row label{flex:1;min-width:100px}.actions{margin-top:12px}.component-toggle{margin-top:12px}.component-manager{margin-top:12px;padding-top:12px;border-top:1px solid var(--border)}.digest,.event-log{font-family:var(--font-mono)}.muted{color:var(--muted);font-size:var(--support-size);line-height:1.6}.corpus-list{list-style:none;padding:0;max-height:240px;overflow:auto}.corpus-list li{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid var(--border);overflow-wrap:anywhere}.job-state{justify-content:space-between;margin-top:12px}progress{width:100%;height:8px;accent-color:var(--accent)}.event-log{padding:0;list-style:none;min-height:100px;max-height:300px;overflow:auto;background:var(--bg);border:1px solid var(--border);border-radius:6px}.event-log li{padding:7px 10px;display:flex;gap:12px;font-size:var(--support-size)}.event-log time{color:var(--muted);white-space:nowrap}.result-row{padding:8px 0;border-bottom:1px solid var(--border)}@container module (max-width:800px){.mfa-columns{grid-template-columns:1fr}}

.mfa-columns{flex:1;min-height:0;align-items:stretch}
.mfa-controls,.mfa-results{min-height:0;overflow:auto;align-content:start;overscroll-behavior:contain}
@container module (max-width:800px){.mfa-columns{flex:none}.mfa-controls,.mfa-results{overflow:visible}}
</style>
