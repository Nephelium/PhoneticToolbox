<script setup lang="ts">
import {uiError,uiErrorMessage,setMessageProperty,getMessageProperty} from '../module-localization.ts';
import {uiMessage,joinMessages} from '../module-localization.ts';
import {tr,messageRef} from '../../i18n/index.ts';
import {useWebSettings,settingsFields} from '../../state/web-settings.ts';
import ResearchUpload from '../../components/ResearchUpload.vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import {computed,ref,onMounted,onUnmounted,watch,markRaw} from 'vue';
import type {ResearchContext,ResearchFile,DirectoryGrant,SpectrogramView,BatchView,JobView,BatchSelection,BatchConfig} from '../../platform/research.ts';
import {researchAudio} from '../../platform/research.ts';
import {m01State,saveM01} from './store.ts';
import {association,matchAssociation,associateLips,isLipFile,directoryResources,resetAssociations,reconcile,applyParameters,applySettings,dirty,effectiveOutput,eggDefaults,type Settings} from './state.ts';
import {stop} from '../../state/audio.ts';
import WaveformViewport from '../../components/WaveformViewport.vue';
import TextGridTimeline from '../../components/TextGridTimeline.vue';
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import ParameterDrawer from '../../components/ParameterDrawer.vue';
import SettingsDrawer from '../../components/SettingsDrawer.vue';
import AppIcon from '../../components/AppIcon.vue';
import BatchResults from './BatchResults.vue';
import JointEggSettings from './JointEggSettings.vue';
const props=defineProps<{context:ResearchContext;active?:boolean}>();const emit=defineEmits<{references:[];help:[]}>();
const state=m01State(props.context.key),picker=ref<HTMLInputElement>(),busy=ref(false),notice=messageRef('');
const previewNote=ref('');
const audioFiles=computed(()=>state.files.filter(f=>f.kind==='audio'));
const markedFiles=computed(()=>audioFiles.value.filter(f=>state.marked.includes(f.id)));
function listKey(event:KeyboardEvent){if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='a'){event.preventDefault();state.marked=audioFiles.value.map(f=>f.id);}}
const selected=computed(()=>audioFiles.value.find(f=>f.id===state.selected));
const spectrogramLoader=computed(()=>{const file=selected.value;return props.context.files.spectrogram&&file?(view:SpectrogramView)=>props.context.files.spectrogram!(file,view):undefined;});
const linked=computed(()=>association(state));
const intervals=computed(()=>linked.value.tiers[linked.value.layer]?.intervals??[]);
const segments=computed(()=>intervals.value.filter(i=>i.xmax>i.xmin&&!['','sil','eps','<sil>','<eps>'].includes(i.text.trim().toLowerCase())).length);
const tasks=computed(()=>props.context.files.tasks),taskBusy=ref(false),taskReady=ref(false),batchList=ref<BatchView[]>([]),activeBatch=ref<BatchView|null>(null),jobResults=ref<JobView[]>([]),sliceParameters=ref(false);
const resultCache=new Map<string,JobView>();let taskTimer:ReturnType<typeof setTimeout>|undefined;
let loadingBatch=false,batchGeneration=0,pendingBatchId:string|undefined;
const parameterSource=ref<'recent'|'legacy'>('recent');
const lipFiles=computed(()=>state.files.filter(isLipFile));
const associationError=computed(()=>[linked.value.error,linked.value.lipError].filter(Boolean).join(' '));
let autoSaveBatch:{id:string;directory:string;besideSources:boolean}|null=null;
function scheduleBatch(){clearTimeout(taskTimer);if(!disposed&&tasks.value&&(props.active!==false||taskBusy.value||activeBatch.value&&!activeBatch.value.summary.closed))taskTimer=setTimeout(()=>void loadBatch(),activeBatch.value&&!activeBatch.value.summary.closed?1500:6000);}
async function loadBatch(id?:string){
 if(!tasks.value||disposed)return;if(id)pendingBatchId=id;
 if(taskBusy.value){scheduleBatch();return;}if(loadingBatch)return;
 const ticket=++batchGeneration,requested=pendingBatchId;pendingBatchId=undefined;loadingBatch=true;
 const stale=()=>disposed||ticket!==batchGeneration||pendingBatchId!==undefined;
 try{const list=await tasks.value.list();if(stale())return;batchList.value=list;taskReady.value=true;const chosen=requested??activeBatch.value?.id??batchList.value[0]?.id;
  if(chosen){const batch=await tasks.value.get(chosen);if(stale())return;activeBatch.value=batch;for(const item of batch.summary.items){if(item.job_id&&['succeeded','failed','cancelled','interrupted'].includes(item.state)&&!resultCache.has(item.job_id)){const job=await tasks.value.job(item.job_id);if(stale())return;resultCache.set(item.job_id,job);}}jobResults.value=batch.summary.items.flatMap(item=>item.job_id&&resultCache.has(item.job_id)?[resultCache.get(item.job_id)!]:[]);}
 }catch(e){if(!stale())notice.value=e instanceof Error?uiErrorMessage(e):uiMessage('modules.parameter-estimation.failed_to_read_tasks');}
 finally{loadingBatch=false;if(!disposed){if(pendingBatchId)queueMicrotask(()=>void loadBatch());else scheduleBatch();}}
 if(disposed)return;
 if(autoSaveBatch?.id===activeBatch.value?.id&&activeBatch.value?.summary.closed){const {directory,besideSources}=autoSaveBatch!;autoSaveBatch=null;if(activeBatch.value.summary.counts.succeeded&&tasks.value.save)await saveResults(directory,besideSources);}
 scheduleBatch();
}
async function startBatch(operation:'acoustic_analysis'|'textgrid_segment',scope:'all'|'marked'='all'){
 if(!tasks.value||taskBusy.value)return;
 taskBusy.value=true;batchGeneration++;notice.value='';clearTimeout(taskTimer);
 try{const chosen=operation==='acoustic_analysis'?(scope==='marked'?markedFiles.value:audioFiles.value):audioFiles.value.filter(f=>state.marked.length?state.marked.includes(f.id):f.id===state.selected);
  if(!chosen.length)throw uiError(scope==='marked'?uiMessage('modules.parameter-estimation.select_the_audio_files_to_analyse_first'):uiMessage('modules.parameter-estimation.select_an_audio_file_first'));if(operation==='acoustic_analysis'&&!state.wave.parameters.length)throw uiError(uiMessage('modules.parameter-estimation.select_output_parameters'));
  const layer=operation==='textgrid_segment'?linked.value.tiers[linked.value.layer]?.name??null:null;
  if(operation==='textgrid_segment'&&!layer)throw uiError(uiMessage('modules.parameter-estimation.select_a_textgrid_segmentation_tier'));
  if(props.context.files.kind==='desktop'&&!effectiveOutput(state))throw uiError(uiMessage('modules.parameter-estimation.select_a_results_folder_first'));
  const inputs:BatchSelection[]=[];let noParent=0;
  for(const file of chosen){const a=association(state,file.id);if(a.error||(operation==='acoustic_analysis'&&a.lipError))throw uiError(joinMessages(file.name,'：',getMessageProperty(a,a.error?'error':'lipError')));if(operation==='textgrid_segment'&&!a.textgrid)throw uiError(joinMessages(file.name,uiMessage('modules.parameter-estimation.has_no_associated_textgrid')));
   const item:BatchSelection={audio:file,textgrid:a.textgrid,lip:operation==='acoustic_analysis'?a.lip:null};
   if(operation==='textgrid_segment'&&sliceParameters.value){
    if(parameterSource.value==='legacy'){if(!a.legacy)throw uiError(joinMessages(file.name,uiMessage('modules.parameter-estimation.has_no_explicitly_associated_historical_parameter_table_select_one')));item.legacy_result=a.legacy;}
    else{const parent=await tasks.value.parent(file);if(parent)item.parent_result=parent;else noParent++;}
   }inputs.push(item);}
  const config:BatchConfig|null=operation==='acoustic_analysis'?{extended:null,settings:{...state.settings},selection:{mode:'catalog',keys:[...state.wave.parameters] as NonNullable<NonNullable<BatchConfig['selection']>['keys']>},backend_policy:{reaper:'native_required',wm_f0:'irapt_then_praat'}}:null;
  if(config&&props.context.files.kind==='desktop')config.extended={...state.extended,channel_overrides:Object.fromEntries(chosen.flatMap((f,i)=>state.channelOverrides[f.id]===undefined?[]:[[String(i),state.channelOverrides[f.id]]]))};
  const directory=effectiveOutput(state)?.id;
  const batch=await tasks.value.submit(operation,inputs,config,layer,crypto.randomUUID());if(disposed)return;activeBatch.value=batch;autoSaveBatch=props.context.files.kind==='desktop'&&directory?{id:batch.id,directory,besideSources:state.sameDirectory&&state.recursive}:null;
  notice.value=noParent?uiMessage('modules.parameter-estimation.value_audio_files_have_no_usable_parameter_results_only',{v0:(noParent)}):uiMessage('modules.parameter-estimation.batch_submitted_parameters_and_file_associations_are_fixed_further');
 }catch(e){notice.value=e instanceof Error?uiErrorMessage(e):uiMessage('modules.parameter-estimation.submission_failed');}finally{taskBusy.value=false;void loadBatch();}
}
async function cancelBatch(){if(!tasks.value||!activeBatch.value||taskBusy.value)return;taskBusy.value=true;batchGeneration++;try{activeBatch.value=await tasks.value.cancel(activeBatch.value.id);}catch(e){notice.value=uiErrorMessage(e);}finally{taskBusy.value=false;void loadBatch();}}
async function retryJob(id:string){if(!tasks.value||taskBusy.value)return;taskBusy.value=true;batchGeneration++;try{await tasks.value.retry(id,crypto.randomUUID());}catch(e){notice.value=uiErrorMessage(e);}finally{taskBusy.value=false;void loadBatch();}}
async function saveResults(target?:string,besideSources=state.sameDirectory&&state.recursive){const directory=target??effectiveOutput(state)?.id;if(!tasks.value?.save||!activeBatch.value||taskBusy.value)return;if(!directory){notice.value=uiMessage('modules.parameter-estimation.after_reopening_select_a_results_folder_before_saving_completed');return;}taskBusy.value=true;try{const result=await tasks.value.save(activeBatch.value.id,directory,besideSources);if(disposed)return;await refresh();notice.value=uiMessage('modules.parameter-estimation.saved_value_result_files_existing_files_with_the_same',{v0:(result.count)});}catch(e){if(!disposed)notice.value=uiErrorMessage(e);}finally{taskBusy.value=false;scheduleBatch();}}
async function downloadResult(id:string,name:string){if(!tasks.value?.download||taskBusy.value)return;taskBusy.value=true;try{await tasks.value.download(id,name);}catch(e){notice.value=uiErrorMessage(e);}finally{taskBusy.value=false;scheduleBatch();}}
let disposed=false,previewAbort=new AbortController();
watch(()=>dirty(state),value=>state.wave.dirty=value,{immediate:true});
async function refresh(){
 const version=++state.listVersion;busy.value=true;notice.value='';
 try{if(props.context.files.kind==='desktop'&&!state.input)return;
  const inputs=await props.context.files.list(state.input?.id,state.recursive);
  const directories=new Map<string,ResearchFile[]>();if(state.input)directories.set(state.input.id,inputs);
  for(const directory of [state.associationDirectory,state.lipDirectory])if(directory&&!directories.has(directory.id))directories.set(directory.id,await props.context.files.list(directory.id,state.recursive));
  const values=directoryResources(inputs,state.associationDirectory?directories.get(state.associationDirectory.id):inputs,state.lipDirectory?directories.get(state.lipDirectory.id):inputs);
  if(disposed||version!==state.listVersion)return;
  reconcile(state,[...new Map(values.map(f=>[f.id,f])).values()]);
  for(const file of audioFiles.value){const a=association(state,file.id);try{
   if(!a.manual.textgrid)a.textgrid=matchAssociation(file,state.files,'textgrid');
   a.error='';
  }catch(e){setMessageProperty(a,'error',e instanceof Error?uiErrorMessage(e):uiMessage('modules.parameter-estimation.association_failed'));}}
  associateLips(state);
  if(state.selected&&state.wave.asset)await loadGrid(state.selected,state.loadVersion);
  return true;
 }catch(e){if(!disposed&&version===state.listVersion)notice.value=e instanceof Error?uiErrorMessage(e):uiMessage('modules.parameter-estimation.failed_to_read_folder');}
 finally{if(!disposed&&version===state.listVersion)busy.value=false;}
}
async function choose(purpose:DirectoryGrant['purpose']){
 if(busy.value)return;busy.value=true;notice.value='';
 try{const grant=await props.context.files.choose?.(purpose);if(!grant||disposed)return;
  if(purpose==='input'){state.input=grant;state.loadVersion++;state.selected='';state.wave.asset=null;stop();}
  else if(purpose==='output')state.output=grant;
  if(purpose!=='output')await refresh();
 }catch(e){notice.value=e instanceof Error?uiErrorMessage(e):uiMessage('modules.parameter-estimation.failed_to_select_folder');}finally{busy.value=false;}
}
async function chooseAssociation(kind:'textgrid'|'lip',sameDirectory=false){
 if(busy.value)return;busy.value=true;notice.value='';
 try{
  const grant=sameDirectory?null:await props.context.files.choose?.('association');if(disposed||(!sameDirectory&&!grant))return;
  if(kind==='textgrid')state.associationDirectory=grant??null;else state.lipDirectory=grant??null;
  resetAssociations(state,kind);await refresh();
 }catch(e){if(!disposed)notice.value=e instanceof Error?uiErrorMessage(e):uiMessage('modules.parameter-estimation.failed_to_select_folder');}finally{if(!disposed)busy.value=false;}
}
async function add(event:Event){const input=event.target as HTMLInputElement;props.context.files.add?.(Array.from(input.files??[]));input.value='';await refresh();}
async function loadGrid(fileId:string,version:number){
 const a=association(state,fileId),grid=a.textgrid;if(!grid){a.tiers=[];return;}
 const data=await props.context.files.textgrid(grid);
 if(disposed||state.loadVersion!==version||a.textgrid?.id!==grid.id)return;
 a.textgrid={...grid,sha256:data.sha256};a.gridHash=data.sha256;a.tiers=data.tiers;a.layer=Math.min(a.layer,Math.max(0,data.tiers.length-1));a.error='';
}
async function selectFile(file:ResearchFile){
 previewAbort.abort();previewAbort=new AbortController();
 stop();const version=++state.loadVersion;state.selected=file.id;state.wave.asset=null;state.wave.loading=true;state.wave.error='';previewNote.value='';
 try{const data=await researchAudio(props.context.files,file,previewAbort.signal);if(disposed||version!==state.loadVersion)return;previewNote.value=data.previewNote;
  Object.assign(state.wave,{asset:markRaw(data.asset),start:0,end:data.asset.duration,channel:0,zoom:1,offset:0});file.sha256=data.sha256;
  try{await loadGrid(file.id,version);}catch(e){if(!disposed&&version===state.loadVersion)association(state,file.id).error=e instanceof Error?e.message:tr('modules.parameter-estimation.failed_to_read_textgrid');}
 }catch(e){if(!disposed&&version===state.loadVersion)state.wave.error=e instanceof Error?e.message:tr('modules.parameter-estimation.failed_to_read_audio');}
 finally{if(!disposed&&version===state.loadVersion)state.wave.loading=false;}
}
async function changeAssociation(kind:'textgrid'|'lip',event:Event){
 const id=(event.target as HTMLSelectElement).value,a=linked.value;
 a[kind]=state.files.find(f=>f.id===id&&(kind==='lip'?isLipFile(f):f.kind===kind))??null;a.manual[kind]=true;
 if(kind==='lip')a.lipError='';else a.error='';
 if(kind==='textgrid'){a.tiers=[];a.gridHash='';a.layer=0;const version=++state.loadVersion;try{await loadGrid(state.selected,version);}catch(e){if(version===state.loadVersion)a.error=e instanceof Error?e.message:tr('modules.parameter-estimation.failed_to_read_textgrid');}}
}
async function linkAllLips(){
 if(busy.value||!audioFiles.value.length)return;busy.value=true;notice.value='';
 try{
  if(!await refresh()||disposed)return;
  const result=associateLips(state,true);
  notice.value=uiMessage('modules.parameter-estimation.lip_associations_value_matched_value_without_same_name_data',{v0:(result.matched),v1:(result.missing),v2:(result.ambiguous)});
 }catch(e){if(!disposed)notice.value=e instanceof Error?uiErrorMessage(e):uiMessage('modules.parameter-estimation.failed_to_read_lip_folder');}
 finally{if(!disposed)busy.value=false;}
}
function openParameters(){state.parameterDraft=[...state.wave.parameters];state.drawer='parameters';}
function openSettings(){state.settingsDraft={...state.settings};state.drawer='settings';}
function parameters(keys:string[]){try{applyParameters(state,keys);}catch(e){notice.value=uiErrorMessage(e);}}
function settings(value:Settings){applySettings(state,value);}
function save(){notice.value=saveM01(props.context.key)?uiMessage('modules.parameter-estimation.parameters_and_settings_saved_files_folder_permissions_and_account'):uiMessage('modules.parameter-estimation.failed_to_save_draft_check_local_storage_permissions');}
function intervalSelect(xmin:number,xmax:number){stop();if(!state.wave.asset)return;state.wave.start=Math.max(0,Math.min(xmin,state.wave.asset.duration));state.wave.end=Math.max(state.wave.start,Math.min(xmax,state.wave.asset.duration));}
watch(()=>props.active,active=>{if(active!==false)void loadBatch();else scheduleBatch();});
onMounted(()=>{if(props.context.files.kind!=='desktop'||state.input)void refresh();if(tasks.value)void loadBatch();});
onUnmounted(()=>{disposed=true;clearTimeout(taskTimer);previewAbort.abort();state.loadVersion++;state.listVersion++;state.wave.loading=false;stop();});
useWebSettings(props.context.key,{...settingsFields(state,['settings','sameDirectory','recursive']),...settingsFields(state.wave,['parameters','showBoth','showSpectrogram']),...settingsFields(state.extended,['max_duration_s','audio_channel']),egg:computed({get:()=>({enabled:state.extended.egg!==null,values:state.extended.egg??eggDefaults()}),set:value=>{state.extended.egg=value.enabled?value.values:null;}})});
</script>
<template>
<ModuleFrame unified fit :label="tr('modules.parameter-estimation.parameter_estimation_workspace')" class="workspace-page m01-page">

<template #toolbar><ModuleToolbar :label="tr('modules.parameter-estimation.parameter_estimation_actions')"><ResearchUpload :files="context.files" accept=".wav,.TextGrid,.lip.json,.xlsx,.ptb.sqlite,.ptb.sqlite3" @uploaded="refresh"/><div class="m01-directory-bar">
<template v-if="context.files.kind==='desktop'"><button class="primary" :disabled="busy" @click="choose('input')"><AppIcon name="folder"/>{{tr('modules.parameter-estimation.open_audio_folder')}}</button><span class="directory-label" :title="state.input?.label">{{state.input?.label??tr('modules.parameter-estimation.no_folder_selected')}}</span>
<div class="m01-directory-choice"><button :disabled="busy" @click="chooseAssociation('textgrid')">{{tr('modules.parameter-estimation.select_textgrid_folder')}}</button><span class="directory-label" :title="state.associationDirectory?.label??state.input?.label">{{state.associationDirectory?.label??tr('modules.parameter-estimation.audio_folder')}}</span><button v-if="state.associationDirectory" class="directory-reset" :disabled="busy" :aria-label="tr('modules.parameter-estimation.use_audio_folder_for_textgrid')" :title="tr('modules.parameter-estimation.restore_audio_folder_association')" @click="chooseAssociation('textgrid',true)">{{tr('modules.parameter-estimation.reset')}}</button></div>
<div class="m01-directory-choice"><button :disabled="busy" @click="chooseAssociation('lip')">{{tr('modules.parameter-estimation.select_lip_folder')}}</button><span class="directory-label" :title="state.lipDirectory?.label??state.input?.label">{{state.lipDirectory?.label??tr('modules.parameter-estimation.audio_folder')}}</span><button v-if="state.lipDirectory" class="directory-reset" :disabled="busy" :aria-label="tr('modules.parameter-estimation.use_audio_folder_for_lip_data')" :title="tr('modules.parameter-estimation.restore_audio_folder_association')" @click="chooseAssociation('lip',true)">{{tr('modules.parameter-estimation.reset')}}</button></div></template>
<template v-else-if="context.files.kind==='preview'"><input ref="picker" class="visually-hidden" type="file" accept=".wav" multiple :aria-label="tr('modules.parameter-estimation.select_audio_list')" @change="add"/><button class="primary" @click="picker?.click()">{{tr('modules.parameter-estimation.add_wav_to_list')}}</button><span class="hint">{{tr('modules.parameter-estimation.local_preview_no_upload')}}</span></template>
<template v-else><span>{{tr('modules.parameter-estimation.project_resources')}} {{context.label}}</span><small class="muted">{{tr('modules.parameter-estimation.upload_wav_textgrid_lip_files_and_parameter_tables_directly')}}</small></template>
<label v-if="context.files.kind==='desktop'" class="recursive-option"><input v-model="state.recursive" type="checkbox" :disabled="busy" @change="refresh"/>{{tr('modules.parameter-estimation.include_subfolders')}}</label>
<button :disabled="busy||!audioFiles.length" :title="tr('modules.parameter-estimation.match_all_audio_again_using_the_current_lip_folder')" @click="linkAllLips">{{tr('modules.parameter-estimation.associate_lip_data')}}</button>
<button :disabled="busy||(context.files.kind==='desktop'&&!state.input)" @click="refresh">{{busy?tr('modules.parameter-estimation.reading'):tr('modules.parameter-estimation.refresh_list')}}</button>
<div v-if="context.files.kind==='desktop'" class="m01-output-row"><label><input v-model="state.sameDirectory" type="checkbox"/>{{tr('modules.parameter-estimation.save_results_beside_wav')}}</label><button :disabled="state.sameDirectory||busy" @click="choose('output')">{{tr('modules.parameter-estimation.select_results_folder')}}</button><span class="directory-label" :title="effectiveOutput(state)?.label">{{effectiveOutput(state)?.label??tr('modules.parameter-estimation.no_results_folder_selected')}}</span></div></div><template #actions><button :disabled="!dirty(state)" @click="save">{{tr('modules.parameter-estimation.save_draft')}}</button><button type="button" data-action="module-help" @click="emit('help')">{{tr('modules.parameter-estimation.help')}}</button><button @click="emit('references')"><AppIcon name="book"/>{{tr('modules.parameter-estimation.methods_and_references')}}</button></template></ModuleToolbar></template>
<ModuleStatus v-if="notice" :message="notice"/><ModuleStatus v-if="state.wave.error" kind="error" :message="state.wave.error"/>
<ModuleWorkbench unified :state-key="context.key" :left-label="tr('modules.parameter-estimation.input_and_parameters')" :right-label="tr('modules.parameter-estimation.tasks_and_records')">
<template #left><div class="file-panel m01-input-panel workbench-card" tabindex="0" :aria-label="tr('modules.parameter-estimation.audio_list_scroll_area')" @keydown="listKey"><div class="panel-heading"><h2>{{tr('modules.parameter-estimation.audio_list')}}</h2><small>{{tr('modules.parameter-estimation.fileCount',{count:audioFiles.length})}}</small></div><p class="hint">{{tr('modules.parameter-estimation.click_a_file_to_listen_check_files_for_analysis')}}</p>
<div class="m01-select-all"><label><input :aria-label="tr('modules.parameter-estimation.select_all_audio')" type="checkbox" :checked="audioFiles.length>0&&state.marked.length===audioFiles.length" :indeterminate="state.marked.length>0&&state.marked.length<audioFiles.length" :disabled="!audioFiles.length" @change="state.marked=($event.target as HTMLInputElement).checked?audioFiles.map(f=>f.id):[]"/>{{tr('modules.parameter-estimation.select_all')}}</label><small>{{tr('modules.parameter-estimation.selected')}} {{markedFiles.length}} / {{audioFiles.length}}</small></div><div class="m01-file-list"><div v-for="file in audioFiles" :key="file.id" class="m01-file-entry"><input v-model="state.marked" type="checkbox" :value="file.id" :aria-label="tr('modules.parameter-estimation.select_audio')+file.name"/><button class="file-row" data-audio-selection :class="{selected:state.selected===file.id}" :aria-pressed="state.selected===file.id" @click="selectFile(file)"><AppIcon name="file"/><span>{{file.name}}<small>{{association(state,file.id).textgrid?'TextGrid · ':''}}{{association(state,file.id).lip?tr('modules.parameter-estimation.lip_linked'):''}}{{Math.ceil(file.size/1024)}} KB</small><small v-if="association(state,file.id).error||association(state,file.id).lipError" class="danger-text">{{tr('modules.parameter-estimation.association_needs_confirmation')}}</small></span></button></div></div>
<p v-if="!audioFiles.length" class="empty-small">{{tr('modules.parameter-estimation.select_an_audio_folder_or_refresh_project_resources')}}</p></div><section class="parameter-summary m01-analysis-settings"><h2>{{tr('modules.parameter-estimation.output_parameters')}}</h2><div class="parameter-count"><strong>{{state.wave.parameters.length}}</strong><span>{{tr('modules.parameter-estimation.80_parameters')}}</span></div><button @click="openParameters">{{tr('modules.parameter-estimation.select_output_parameters_178')}}</button><hr/><h2>{{tr('modules.parameter-estimation.analysis_settings')}}</h2><p class="mono muted">{{tr('modules.parameter-estimation.frame_step')}} {{state.settings.frameshift_ms}} ms<br/>{{tr('modules.parameter-estimation.analysis_window')}} {{state.settings.windowsize_ms}} ms</p><button @click="openSettings">{{tr('modules.parameter-estimation.edit_14_settings')}}</button><JointEggSettings v-if="context.files.kind==='desktop'" :value="state.extended" :mono="state.wave.asset?.channels.length===1" :selected="!!selected" :override="state.channelOverrides[state.selected]" @override="value=>value===undefined?delete state.channelOverrides[state.selected]:state.channelOverrides[state.selected]=value"/><span v-if="dirty(state)" class="draft-indicator">{{tr('modules.parameter-estimation.unsaved')}}</span><section v-if="selected&&linked.tiers.length" class="m01-tiers m01-slicing"><label>{{tr('modules.parameter-estimation.textgrid_segmentation_tier')}}<select v-model.number="linked.layer"><option v-for="(tier,i) in linked.tiers" :key="i" :value="i">{{tier.name}}</option></select></label><p class="hint">{{tr('modules.parameter-estimation.candidateSegments',{count:segments})}}</p><label><input v-model="sliceParameters" type="checkbox"/>{{tr('modules.parameter-estimation.also_split_parameter_results')}}</label>
<template v-if="sliceParameters"><div class="legacy-source-row"><label>{{tr('modules.parameter-estimation.parameter_source')}}<select v-model="parameterSource" :aria-label="tr('modules.parameter-estimation.segmentation_parameter_source')"><option value="recent">{{tr('modules.parameter-estimation.latest_complete_results_from_the_same_source_in_this')}}</option><option value="legacy">{{tr('modules.parameter-estimation.choose_historical_parameter_table_source_unverified')}}</option></select></label>
<label v-if="parameterSource==='legacy'">{{tr('modules.parameter-estimation.historical_parameter_table')}}<select :aria-label="tr('modules.parameter-estimation.historical_parameter_table_association')" :value="linked.legacy?.id??''" @change="linked.legacy=state.files.find(f=>f.id===($event.target as HTMLSelectElement).value&&f.kind==='parameter')??null"><option value="">{{tr('modules.parameter-estimation.not_specified')}}</option><option v-for="f in state.files.filter(f=>f.kind==='parameter')" :key="f.id" :value="f.id">{{f.name}}</option></select></label></div>
<p class="hint">{{parameterSource==='recent'?tr('modules.parameter-estimation.matches_the_original_audio_and_preserves_analysis_frames_and'):tr('modules.parameter-estimation.select_a_complete_historical_table_for_each_audio_file')}}</p></template>
<button class="primary" :disabled="!taskReady||taskBusy||busy" @click="startBatch('textgrid_segment')">{{tr('modules.parameter-estimation.save_audio_segments_from_current_tier')}}</button><small v-if="!tasks" class="muted">{{tr('modules.parameter-estimation.audio_preview_and_playback_are_available_use_the_desktop')}}</small></section></section></template>
<template #default><div class="signal-panel"><template v-if="selected"><div class="signal-heading"><h2>{{selected.name}}</h2><p v-if="state.wave.asset" class="mono muted">{{tr('modules.parameter-estimation.audioInfo',{sampleRate:state.wave.asset.sampleRate,channels:state.wave.asset.channels.length,duration:state.wave.asset.duration.toFixed(3)})}}</p></div>
<div class="m01-association-controls"><label>TextGrid<select :aria-label="tr('modules.parameter-estimation.textgrid_association')" :value="linked.textgrid?.id??''" :disabled="state.wave.loading" @change="changeAssociation('textgrid',$event)"><option value="">{{tr('modules.parameter-estimation.no_association')}}</option><option v-for="f in state.files.filter(f=>f.kind==='textgrid')" :key="f.id" :value="f.id">{{f.name}}</option></select></label><label>{{tr('modules.parameter-estimation.lip_data')}}<select :aria-label="tr('modules.parameter-estimation.lip_association')" :value="linked.lip?.id??''" @change="changeAssociation('lip',$event)"><option value="">{{tr('modules.parameter-estimation.no_association')}}</option><option v-for="f in lipFiles" :key="f.id" :value="f.id">{{f.name}}</option></select></label></div>
<p v-if="associationError" role="alert" class="error-banner">{{associationError}}</p>
<p v-if="state.wave.loading" role="status" class="audio-loading"><progress :aria-label="tr('modules.parameter-estimation.audio_reading_and_decoding_progress')"/>{{tr('modules.parameter-estimation.reading_audio_and_associations')}}</p>
<template v-if="state.wave.asset"><p v-if="previewNote" class="hint" role="note">{{previewNote}}</p><WaveformViewport :state="state.wave" :spectrogram-loader="spectrogramLoader"><template #controls><label class="channel-picker">{{tr('modules.parameter-estimation.playback_channel')}}<select v-model.number="state.wave.channel" @change="stop"><option v-for="(_,i) in state.wave.asset.channels" :key="i" :value="i">{{tr('modules.parameter-estimation.channel')}} {{i+1}}</option></select></label></template><template #timeline="{start,end}"><TextGridTimeline v-if="linked.tiers.length" :state="state.wave" :tiers="linked.tiers" :start="start" :end="end" :selected="linked.layer" :selection-start="state.wave.start" :selection-end="state.wave.end" @select="(tier,start,end)=>{linked.layer=tier;intervalSelect(start,end)}"/></template></WaveformViewport>

</template></template><div v-else class="wave-empty"><AppIcon name="wave"/><h2>{{tr('modules.parameter-estimation.select_an_audio_segment')}}</h2><p>{{tr('modules.parameter-estimation.select_a_file_from_the_left_to_view_its')}}</p></div></div></template>
<template #right><div class="m01-batch-bar"><div><strong>{{tr('modules.parameter-estimation.processListCount',{count:audioFiles.length})}}</strong><p class="hint">{{tr('modules.parameter-estimation.batchSelectionHint',{count:markedFiles.length})}}</p></div><button class="primary" :disabled="!taskReady||taskBusy||busy||!audioFiles.length" @click="startBatch('acoustic_analysis')">{{tr('modules.parameter-estimation.analyse_entire_list')}}</button><button class="primary" :disabled="!taskReady||taskBusy||busy||!markedFiles.length" @click="startBatch('acoustic_analysis','marked')">{{tr('modules.parameter-estimation.analyse_selected_files')}}</button><span class="task-operation-status muted" :style="{visibility:taskBusy?'visible':'hidden'}" role="status">{{taskBusy?tr('modules.parameter-estimation.processing_task_action'):'\u00a0'}}</span></div><h2>{{tr('modules.parameter-estimation.analysis_results')}}</h2><BatchResults :batches="batchList" :active="activeBatch" :jobs="jobResults" :busy="taskBusy" :desktop="context.files.kind==='desktop'" @select="loadBatch" @cancel="cancelBatch" @save="saveResults" @retry="retryJob" @download="downloadResult"/></template>
</ModuleWorkbench>

<ParameterDrawer v-if="state.drawer==='parameters'" :selected="state.wave.parameters" :draft="state.parameterDraft" :columns="4" require-selection @draft="state.parameterDraft=$event" @close="state.drawer=''" @apply="parameters"/>
<SettingsDrawer v-if="state.drawer==='settings'" :draft="state.settingsDraft" @draft="state.settingsDraft=$event" @close="state.drawer=''" @apply="settings"/>
</ModuleFrame></template>
<style scoped>
.recursive-option{display:flex;align-items:center;gap:4px;white-space:nowrap}.task-operation-status{display:block;width:100%;min-height:1.5em}
.m01-page .m01-batch-bar>button{width:100%}
.m01-directory-choice{display:flex;align-items:center;flex-wrap:wrap;gap:6px;min-width:0}.m01-directory-choice .directory-reset{color:var(--muted)}
.legacy-source-row{display:flex;flex-wrap:wrap;gap:12px}.legacy-source-row label{display:flex;flex:1 1 100%;min-width:0;flex-direction:column;gap:6px}.legacy-source-row select{min-width:0;max-width:100%}
.m01-input-panel{min-height:0;border-right:1px solid var(--border);display:flex;flex-direction:column}.m01-input-panel .m01-file-list{max-height:28vh;min-height:120px;overflow:auto}.m01-analysis-settings{padding:10px;border:1px solid var(--border);border-radius:var(--radius)}.m01-analysis-settings h2{font-size:0.928571rem;margin:0 0 6px}.m01-analysis-settings .parameter-count{display:inline-flex;gap:6px;margin-right:8px}.m01-analysis-settings .parameter-count strong{font-size:1.428571rem}.m01-analysis-settings button{margin:3px 0}.m01-analysis-settings hr{margin:8px 0}.m01-analysis-settings .hint{margin:6px 0}.m01-analysis-settings .mono{margin:4px 0}.signal-panel{padding:10px;border:1px solid var(--border);border-radius:var(--radius)}
.m01-slicing{margin-top:10px;padding-top:10px}.m01-slicing>label{align-items:stretch;flex-direction:column}.m01-slicing select{min-width:0;max-width:100%;width:100%}.m01-slicing>button{width:100%}.m01-slicing>label:has(input){flex-direction:row;align-items:center}
.m01-page :deep(.wave-track svg){height:max(170px,calc(100dvh - 830px))}.m01-page .wave-empty{min-height:calc(100dvh - 260px)}
.m01-page .signal-panel{flex:1}.m01-page .m01-batch-bar{width:100%;background:var(--panel);margin:0;padding:10px}
</style>
