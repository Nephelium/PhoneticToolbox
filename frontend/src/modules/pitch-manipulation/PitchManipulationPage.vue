<script setup lang="ts">
import {uiError,uiErrorMessage} from '../module-localization.ts';
import {uiMessage,joinMessages} from '../module-localization.ts';
import {localized} from '../module-localization.ts';
import {tr,messageRef,type MessageDescriptor} from '../../i18n/index.ts';
import {useWebSettings,settingsFields} from '../../state/web-settings.ts';
import ResearchUpload from '../../components/ResearchUpload.vue';
import AppIcon from '../../components/AppIcon.vue';
import {vTimePrecision} from '../../design/time-precision.ts';
import PlotPlaceholder from '../../components/PlotPlaceholder.vue';
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import {computed,ref,reactive,markRaw,watch,onMounted,onUnmounted} from 'vue';
import {type ResearchContext,type ResearchFile,audioPreview,sha256} from '../../platform/research.ts';
import {workspace,host,type Workspace} from '../../state/workspace.ts';import {play,stop} from '../../state/audio.ts';import {decodeWav} from '../../platform/decode.ts';
import ModuleFrame from '../../components/ModuleFrame.vue';import ModuleToolbar from '../../components/ModuleToolbar.vue';import ModuleSection from '../../components/ModuleSection.vue';import ModuleStatus from '../../components/ModuleStatus.vue';import ModalDialog from '../../components/ModalDialog.vue';
import WaveformViewport from '../../components/WaveformViewport.vue';import AudioTransport from '../../components/AudioTransport.vue';
import PitchCurve from './PitchCurve.vue';import HistoryPlot from './HistoryPlot.vue';
import VariantSettings from './VariantSettings.vue';
import type {Config,Mode,M08Port,Job,Result} from './port.ts';import {defaults,modes,finite,frequencies,validate,importF0,renamePlan,renamePrefix,visible,errorText,errorMessage,variants,shiftedF0,variantLabel,type PitchMode} from './state.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active:boolean;port?:M08Port}>();const emit=defineEmits<{references:[]}>();
const port=computed(()=>props.port??(props.context.files as typeof props.context.files&{m08?:M08Port}).m08);
const wave=workspace(props.stateKey),resultWave=reactive<Workspace>({...wave,asset:null,dirty:false});
const files=ref<ResearchFile[]>([]),source=ref<ResearchFile>(),directory=ref(''),picker=ref<HTMLInputElement>(),folderPicker=ref<HTMLInputElement>();
const config=reactive(defaults()),times=ref<number[]>([]),original=ref<number[]>([]),modified=ref<number[]>([]),refs=ref<number[]>([]),refInput=ref('200');
const timeRefs=ref<number[]>([]),timeRefInput=ref('0.5'),refDirection=ref('hz');
const numberExports=ref(false);let exportSequence:Record<string,number>|undefined;
watch(numberExports,()=>{exportSequence=undefined;});
// Keep editing text intact (e.g. 0. / -) until an explicit save or task snapshot.
const numericInput=reactive({speed:'1',pitch_ratio:'1',pitch_hz:'0'});
const pitchMode=ref<PitchMode>('ratio');
const minInput=ref('50'),maxInput=ref('350'),ymin=ref(50),ymax=ref(350),error=messageRef(''),notice=messageRef(''),busy=ref(false),tab=ref('single'),help=ref(false),importing=ref(false),importText=ref('');
const jobs=ref<Job[]>([]),history=ref<Result[]>([]),result=ref<Result>(),historyPlot=ref<InstanceType<typeof HistoryPlot>>(),playResult=ref(false),directPlayback=ref(false),batchIds=ref<string[]>([]),cancelSubmit=ref(false);
const loading=ref(false),loadPhase=messageRef('');
const editing=ref(false),rows=ref<{time:string;freqs:string;mode:Mode;selected:boolean}[]>([]),selectedResults=ref<string[]>([]),manage=ref<'delete'|'rename'|'save'|null>(null),manageResults=ref<Result[]>([]),prefix=ref(''),pendingFile=ref<string>();
const removedResults=new Set<string>();
const lastBatchIds=ref<string[]>([]),batchSubmitting=ref(false),batchSavePending=ref(false),pendingResult=ref<Result>();
let followResults=true,viewTicket=0;
const jobLabels=localized(()=>({queued:tr('modules.pitch-manipulation.queued'),running:tr('modules.pitch-manipulation.running'),cancel_requested:tr('modules.pitch-manipulation.cancelling'),cancelled:tr('modules.pitch-manipulation.cancelled'),succeeded:tr('modules.pitch-manipulation.completed'),failed:tr('modules.pitch-manipulation.failed'),interrupted:tr('modules.pitch-manipulation.interrupted')}));
const savedOutput=ref<{directory?:string;destination?:string;names:string[]}>();
let disposed=false,epoch=0,previewAbort:AbortController|undefined,timer:ReturnType<typeof setInterval>,polling=false,resultEpoch=0,historyEpoch=0;
const audioFiles=computed(()=>files.value.filter(f=>f.kind==='audio'));
const range=computed(()=>visible(wave.asset?.duration??0,wave.zoom,wave.offset));

const batchJobs=computed(()=>jobs.value.filter(j=>lastBatchIds.value.includes(j.id)));
const batchResults=computed(()=>batchJobs.value.flatMap(j=>j.results).filter(r=>!removedResults.has(r.id)));
const batchPending=computed(()=>batchSubmitting.value||batchJobs.value.some(j=>['queued','running','cancel_requested'].includes(j.state)));
const variantSummary=computed(()=>{try{const count=variants(numericInput,pitchMode.value).length;return tr('modules.pitch-manipulation.value_versions_per_audio_file_value',{v0:(count),v1:(tab.value==='batch'?tr('modules.pitch-manipulation.value_in_total_for_value_selected_files',{v0:batchIds.value.length,v1:count*batchIds.value.length}):'')});}catch(e){return errorText(e);}});
const currentBatch=computed(()=>history.value.filter(r=>r.source_id===source.value?.id&&r.start.toFixed(2)===range.value[0].toFixed(2)&&r.end.toFixed(2)===range.value[1].toFixed(2)));
const chosen=computed(()=>manageResults.value.filter(r=>selectedResults.value.includes(r.id)));
watch(range,()=>cancelPlayback());watch(config,()=>wave.dirty=true,{deep:true});
watch([numericInput,pitchMode],dirty,{deep:true});
watch(()=>props.active,active=>{if(!active)cancelPlayback();else void poll();});
function dirty(){wave.dirty=true;}
function cancelPlayback(){++resultEpoch;directPlayback.value=false;stop();}
function selectPlayback(isResult:boolean){cancelPlayback();playResult.value=isResult;}
function selectionPlayback(){++resultEpoch;directPlayback.value=false;}
async function playDirect(isResult:boolean){
 cancelPlayback();const state=isResult?resultWave:wave;if(!state.asset||!props.active)return;
 playResult.value=isResult;directPlayback.value=true;const [start,end]=isResult?[0,state.asset.duration]:range.value;
 await play(state.asset,start,end,state.channel);
}
async function refreshHistory(file:ResearchFile){
 if(disposed||file.id!==source.value?.id)return;
 const ticket=++historyEpoch,sourceTicket=epoch;
 const values=await port.value!.history(file);
 if(!disposed&&ticket===historyEpoch&&sourceTicket===epoch&&file.id===source.value?.id)history.value=values.filter(r=>!removedResults.has(r.id));
}
async function attempt(work:()=>Promise<void>|void){error.value='';try{await work();}catch(e){if(!disposed)error.value=errorMessage(e);}}
async function refresh(){const next=await props.context.files.list(directory.value||undefined);if(disposed)return;files.value=next;batchIds.value=batchIds.value.filter(id=>next.some(f=>f.id===id));}
async function choose(){await attempt(async()=>{const grant=await props.context.files.choose?.('input');if(grant){directory.value=grant.id;await refresh();}});}
async function add(event:Event){await attempt(async()=>{const input=event.target as HTMLInputElement;props.context.files.add?.([...input.files??[]]);input.value='';await refresh();});}
function requestLoad(id:string){++viewTicket;pendingResult.value=undefined;if(wave.dirty&&source.value&&id!==source.value.id){pendingFile.value=id;return;}void load(id);}
function selectSource(event:Event){const element=event.target as HTMLSelectElement,id=element.value;element.value=source.value?.id??'';requestLoad(id);}
async function load(id:string){
 const file=files.value.find(f=>f.id===id);if(!file)return;
 const ticket=++epoch;previewAbort?.abort();const controller=new AbortController();previewAbort=controller;
 const current=()=>!disposed&&ticket===epoch&&!controller.signal.aborted;
 loading.value=true;loadPhase.value=uiMessage('modules.pitch-manipulation.reading_original_audio');error.value='';notice.value='';cancelPlayback();++historyEpoch;
 source.value=file;times.value=[];original.value=[];modified.value=[];result.value=undefined;resultWave.asset=null;history.value=[];wave.asset=null;playResult.value=false;
 try{
  // Original samples become usable before the durable scientific preview starts.
  // They retain the recording's sampling rate, channels and timeline.
  let rawError:unknown;
  try{const data=await audioPreview(props.context.files,file,controller.signal);if(!current())return;
   wave.asset=markRaw(data.asset);source.value={...file,sha256:data.sha256};
  }catch(e){if(!current())return;rawError=e;}
  if(wave.asset){wave.zoom=1;wave.offset=0;wave.start=0;wave.end=wave.asset.duration;wave.channel=0;}
  if(port.value){loadPhase.value=wave.asset?uiMessage('modules.pitch-manipulation.original_waveform_and_playback_ready_extracting_f0'):uiMessage('modules.pitch-manipulation.decoding_audio_and_extracting_f0');
   const data=await port.value.preview(source.value!,controller.signal);if(!current())return;
   if(wave.asset&&source.value?.sha256!==data.sha256)throw uiError(uiMessage('modules.pitch-manipulation.source_audio_changed_after_reading_select_again'));
   if(!wave.asset){const asset=await decodeWav(data.wav,file.name,controller.signal);if(!current())return;wave.asset=markRaw(asset);wave.zoom=1;wave.offset=0;wave.start=0;wave.end=asset.duration;wave.channel=0;}
   source.value={...file,sha256:data.sha256};times.value=data.times;original.value=data.original_f0;modified.value=[...data.original_f0];
  }else if(rawError)throw rawError;
  if(!current())return;
  const voiced=times.value.filter((_,i)=>original.value[i]>0);config.points=[{time:Number((voiced[0]??0).toFixed(2)),freqs:[],mode:'order'},{time:Number((voiced.at(-1)??wave.asset!.duration).toFixed(2)),freqs:[],mode:'order'}];
  const saved=host.projects.read<{sha256?:string;modified?:number[];config?:Config;numericInput?:typeof numericInput;pitchMode?:PitchMode;refs?:number[];timeRefs?:number[];ymin?:number;ymax?:number}>('m08.draft.'+props.stateKey+'.'+file.id,{});
  if(saved.sha256===source.value?.sha256&&saved.modified?.length===times.value.length&&saved.modified.every(Number.isFinite)){modified.value=saved.modified;if(saved.config){Object.assign(config,saved.config);Object.assign(numericInput,saved.numericInput??{speed:String(config.speed),pitch_ratio:String(config.pitch_ratio),pitch_hz:String(config.pitch_hz)});pitchMode.value=saved.pitchMode??(config.pitch_hz!==0&&config.pitch_ratio===1?'hz':'ratio');}refs.value=saved.refs??[];timeRefs.value=saved.timeRefs??[];ymin.value=saved.ymin??50;ymax.value=saved.ymax??350;minInput.value=String(ymin.value);maxInput.value=String(ymax.value);}
  await Promise.resolve();if(!current())return;wave.dirty=false;
  if(port.value){loadPhase.value=uiMessage('modules.pitch-manipulation.f0_ready_reading_history');await refreshHistory(source.value!);}
 }catch(e){if(current())error.value=errorMessage(e);}finally{if(ticket===epoch){loading.value=false;loadPhase.value='';}}
}
function save(){try{if(loading.value)throw uiError(uiMessage('modules.pitch-manipulation.wait_for_f0_to_load_before_saving_the_draft'));if(!source.value)throw uiError(uiMessage('modules.pitch-manipulation.select_audio_first'));const values=variants(numericInput,pitchMode.value);if(!host.projects.write('m08.draft.'+props.stateKey+'.'+source.value.id,{sha256:source.value.sha256,modified:modified.value,config:{...config,...values[0]},numericInput:{...numericInput},pitchMode:pitchMode.value,refs:refs.value,timeRefs:timeRefs.value,ymin:ymin.value,ymax:ymax.value}))throw uiError(uiMessage('modules.pitch-manipulation.failed_to_save_draft'));wave.dirty=false;notice.value=uiMessage('modules.pitch-manipulation.f0_and_parameter_draft_saved');return true;}catch(e){error.value=errorMessage(e);return false;}}
defineExpose({save});
function pan(delta:number){wave.offset=Math.max(0,Math.min((wave.asset?.duration??0)-(range.value[1]-range.value[0]),wave.offset+delta));}
function zoom(factor:number,time:number){if(!wave.asset)return;const old=range.value,rel=(time-old[0])/(old[1]-old[0]);wave.zoom=Math.max(1,Math.min(wave.asset.duration/.05,wave.zoom*factor));wave.offset=Math.max(0,Math.min(wave.asset.duration-wave.asset.duration/wave.zoom,time-rel*wave.asset.duration/wave.zoom));}
function addReference(){void attempt(()=>{if(refDirection.value==='time'){const value=finite(timeRefInput.value,tr('modules.pitch-manipulation.reference_line_time'));if(value<0)throw uiError(uiMessage('modules.pitch-manipulation.reference_line_time_cannot_be_below_0_seconds'));timeRefs.value.push(value);}else refs.value.push(finite(refInput.value,tr('modules.pitch-manipulation.reference_line_frequency')));dirty();});}
function applyAxis(){void attempt(()=>{const a=finite(minInput.value,tr('modules.pitch-manipulation.minimum_f0')),b=finite(maxInput.value,tr('modules.pitch-manipulation.maximum_f0'));if(a>=b)throw uiError(uiMessage('modules.pitch-manipulation.maximum_f0_must_exceed_minimum_f0'));ymin.value=a;ymax.value=b;dirty();});}
function applyImport(){void attempt(()=>{modified.value=importF0(times.value,modified.value,...range.value,importText.value);dirty();importing.value=false;});}
function editPoints(){rows.value=config.points.map(p=>({time:String(p.time),freqs:p.freqs.join(','),mode:p.mode,selected:false}));editing.value=true;}
function savePoints(){void attempt(()=>{const points=rows.value.map(r=>({time:finite(r.time,tr('modules.pitch-manipulation.time')),freqs:frequencies(r.freqs),mode:r.mode}));const middle=points.slice(1,-1).sort((a,b)=>a.time-b.time);const sorted=[points[0],...middle,points.at(-1)!];validate(sorted,...range.value);config.points=sorted;editing.value=false;});}
function snapshot(action:Config['action'],whole=false,values={speed:1,pitch_ratio:1,pitch_hz:0}):Config {
 const [a,b]=range.value;if(action==='linear')validate(config.points,a,b);
 const start=whole?0:a,end=whole?(action==='transform'?null:wave.asset!.duration):b;
 return {action,start,end,...values,modified_f0:action==='synthesize'?shiftedF0(modified.value,values.pitch_ratio,values.pitch_hz,{times:times.value,start,end:end!}):null,points:action==='linear'?config.points.map(p=>({...p,freqs:[...p.freqs]})):[],offset:config.offset};
}
async function viewResult(r:Result){
 const ticket=++viewTicket;followResults=false;error.value='';
 let target=files.value.find(f=>f.id===r.source_id);
 if(!target){
  const job=jobs.value.find(j=>j.results.some(v=>v.id===r.id));
  const candidates=audioFiles.value.filter(f=>f.name===job?.source_name||f.sha256===job?.source_sha256&&!!job?.source_sha256);
  try{for(const file of candidates){const read=await props.context.files.read(file);if(ticket!==viewTicket||disposed)return;if(await sha256(read.buffer)===job?.source_sha256){target=file;break;}}}catch(e){if(ticket===viewTicket)error.value=errorMessage(e);return;}
 }
 if(ticket!==viewTicket||disposed)return;
 if(!target){error.value=uiMessage('modules.pitch-manipulation.open_the_original_audio_folder_for_this_result_before');return;}
 const selected={...r,source_id:target.id};
 if(selected.source_id!==source.value?.id){
  if(wave.dirty&&source.value){pendingFile.value=selected.source_id;pendingResult.value=selected;return;}
  await load(selected.source_id);
 }
 if(ticket===viewTicket&&source.value?.id===selected.source_id){tab.value='single';await show(selected);}
}
async function finishSwitch(keep:boolean){const id=pendingFile.value,r=pendingResult.value;if(!id||keep&&!save())return;pendingFile.value=undefined;pendingResult.value=undefined;wave.dirty=false;await load(id);if(r&&source.value?.id===r.source_id){tab.value='single';await show(r);}}
async function show(r:Result,autoplay=false){
 if(removedResults.has(r.id)||r.source_id!==source.value?.id)return;
 if(result.value?.id===r.id&&resultWave.asset){if(autoplay)await playDirect(true);return;}
 const ticket=++resultEpoch,sourceTicket=epoch;
 const current=()=>!disposed&&ticket===resultEpoch&&sourceTicket===epoch&&!removedResults.has(r.id);
 try{
  stop();directPlayback.value=false;
  const bytes=await port.value!.audio(r);if(!current())return;
  const asset=await decodeWav(bytes,r.name);if(!current())return;
  result.value=r;resultWave.asset=markRaw(asset);resultWave.start=0;resultWave.end=0;resultWave.zoom=1;resultWave.offset=0;resultWave.channel=0;
  if(autoplay&&props.active){playResult.value=true;directPlayback.value=true;await play(asset,0,asset.duration,0);}
 }catch(e){if(current())error.value=errorMessage(e);}
}
async function poll(){
 if(polling||disposed||!port.value)return;polling=true;
 try{
  const old=jobs.value,next=await port.value.jobs();if(disposed)return;jobs.value=next;
  const completed=next.filter(j=>j.state==='succeeded'&&j.source_id===source.value?.id&&(old.find(o=>o.id===j.id)?.state!=='succeeded'||old.find(o=>o.id===j.id)?.results.length!==j.results.length));
  if(completed.length&&source.value){const file={...source.value};const latest=completed.at(-1)!;
   if(followResults&&latest.results.length)await show(latest.results[0]);await refreshHistory(file);
  }
  offerBatchSave();
 }catch(e){if(!disposed)error.value=errorMessage(e);}finally{polling=false;}
}
async function submit(action:Config['action'],whole=false){await attempt(async()=>{
 if(!source.value||!port.value)throw uiError(uiMessage('modules.pitch-manipulation.tempo_pitch_computation_is_unavailable_here_use_the_desktop'));
 const configs=action==='linear'?[snapshot(action,whole)]:variants(numericInput,pitchMode.value).map(v=>snapshot(action,whole,v));
 if(action==='synthesize'&&!configs[0].modified_f0?.length)throw uiError(uiMessage('modules.pitch-manipulation.no_actual_f0_is_available_for_synthesis'));
 await submitVersions([{...source.value}],configs,configs.length>1);
});}
async function batch(){await attempt(async()=>{if(!port.value)throw uiError(uiMessage('modules.pitch-manipulation.tempo_pitch_computation_is_unavailable_here_use_the_desktop'));const configs=variants(numericInput,pitchMode.value).map(v=>snapshot('transform',true,v)),inputs=audioFiles.value.filter(f=>batchIds.value.includes(f.id));if(!inputs.length)throw uiError(uiMessage('modules.pitch-manipulation.select_at_least_one_batch_file'));await submitVersions(inputs,configs,true);});}
async function submitVersions(inputs:ResearchFile[],configs:Config[],collect:boolean){
 if(batchPending.value)throw uiError(uiMessage('modules.pitch-manipulation.wait_for_this_batch_to_finish_or_cancel_remaining'));
 if(inputs.length*configs.length>256)throw uiError(uiMessage('modules.pitch-manipulation.this_batch_exceeds_256_versions_reduce_files_or_parameter'));
 busy.value=true;notice.value='';cancelSubmit.value=false;followResults=!collect;batchSubmitting.value=collect;
 if(collect){lastBatchIds.value=[];batchSavePending.value=true;}
 const failures:MessageDescriptor[]=[];let count=0;
 try{for(const file of inputs)for(const cfg of configs){if(cancelSubmit.value||disposed)break;try{
  const job=await port.value!.submit(file,structuredClone(cfg),crypto.randomUUID());count++;
  jobs.value=[...jobs.value.filter(j=>j.id!==job.id),job];if(collect)lastBatchIds.value.push(job.id);
  if(cancelSubmit.value||disposed)await port.value!.cancel(job.id);
 }catch(e){failures.push(joinMessages(file.name,' · ',uiMessage('modules.pitch-manipulation.tempo_value_value',{v0:cfg.speed,v1:cfg.pitch_hz!==0?uiMessage('modules.pitch-manipulation.pitch_value_value_hz',{v0:cfg.pitch_hz>0?'+':'',v1:cfg.pitch_hz}):uiMessage('modules.pitch-manipulation.pitch_value',{v0:cfg.pitch_ratio})}),': ',errorMessage(e)));}}
 notice.value=uiMessage('modules.pitch-manipulation.value_value_tasks_value',{v0:(cancelSubmit.value?uiMessage('modules.pitch-manipulation.further_submission_stopped_submitted'):uiMessage('modules.pitch-manipulation.submitted')),v1:(count),v2:(collect?uiMessage('modules.pitch-manipulation.choose_versions_to_save_when_complete'):'')});
 if(failures.length)error.value=joinMessages(...failures.flatMap((value,index)=>index?['\n',value]:[value]));
 }finally{busy.value=false;batchSubmitting.value=false;await poll();offerBatchSave();}
}
function offerBatchSave(){
 if(!batchSavePending.value||batchPending.value||busy.value||manage.value||editing.value||importing.value||pendingFile.value)return;
 if(lastBatchIds.value.length&&!batchJobs.value.length)return;
 batchSavePending.value=false;
 if(batchResults.value.length){openManage('save',batchResults.value,true);}
}
async function saveAudio(){if(result.value)await exportAudio([result.value]);}
async function exportAudio(selected:Result[],numbered=false){
 await attempt(async()=>{
  if(!port.value?.saveMany)throw uiError(uiMessage('modules.pitch-manipulation.this_entry_does_not_provide_audio_saving'));if(!selected.length)throw uiError(uiMessage('modules.pitch-manipulation.check_the_audio_files_to_save'));
  busy.value=true;notice.value='';
  try{
   if(numbered&&!exportSequence)exportSequence=Object.fromEntries(selected.map((r,i)=>[r.id,i+1]));
   const out=await port.value.saveMany(selected,numbered?exportSequence:undefined);
   if(out.cancelled){notice.value=uiMessage('modules.pitch-manipulation.save_cancelled_generated_results_retained');return;}
   if(out.saved.length)savedOutput.value={directory:out.directory,destination:out.destination,names:out.saved.map(f=>f.name)};
   notice.value=uiMessage('modules.pitch-manipulation.saved_value_audio_files_value',{v0:(out.saved.length),v1:(out.failed.length?uiMessage('modules.pitch-manipulation.value_not_saved',{v0:out.failed.length}):'')});
   if(out.failed.length){selectedResults.value=out.failed.map(f=>f.id);error.value=joinMessages(...out.failed.flatMap((f,index)=>[...(index?['\n']:[]),selected.find(r=>r.id===f.id)?.name??f.id,': ',errorMessage(f.error)]));}
   else manage.value=null;
  }finally{busy.value=false;}
 });
}
function openManage(kind:'delete'|'rename'|'save',values:Result[]=currentBatch.value,preserveError=false){
 if(!preserveError)error.value='';notice.value='';numberExports.value=false;exportSequence=undefined;manageResults.value=[...values];selectedResults.value=manageResults.value.map(r=>r.id);manage.value=kind;
 prefix.value=renamePrefix(manageResults.value);
}
async function manageFiles(){
 if(manage.value==='save'){await exportAudio([...chosen.value],numberExports.value);return;}
 await attempt(async()=>{
  if(!port.value||!chosen.value.length)throw uiError(uiMessage('modules.pitch-manipulation.check_the_results_to_process'));
  const selected=[...chosen.value];busy.value=true;notice.value='';
  try{
   if(manage.value==='delete'){
    const out=await port.value.remove(selected.map(r=>r.id));for(const id of out.removed)removedResults.add(id);
    history.value=history.value.filter(r=>!removedResults.has(r.id));
    if(result.value&&removedResults.has(result.value.id)){cancelPlayback();result.value=undefined;resultWave.asset=null;playResult.value=false;}
    notice.value=uiMessage('modules.pitch-manipulation.deleted_value_results_from_this_batch',{v0:(out.removed.length)});
    if(out.failed.length){manageResults.value=manageResults.value.filter(r=>!removedResults.has(r.id));selectedResults.value=out.failed.map(f=>f.id);throw Error(out.failed.map(f=>f.id+': '+f.error).join('\n'));}
   }else{
    const updated=await port.value.rename(renamePlan(selected,prefix.value,history.value.filter(r=>!selectedResults.value.includes(r.id)).map(r=>r.name)));
    if(result.value)result.value=updated.find(r=>r.id===result.value!.id)??result.value;notice.value=uiMessage('modules.pitch-manipulation.renaming_complete');
   }
   manage.value=null;
  }finally{
   // Failure while refreshing must not leave all actions permanently disabled.
   try{if(source.value)await refreshHistory({...source.value});}finally{busy.value=false;}
  }
 });
}
async function exportPng(){await attempt(async()=>{const blob=await historyPlot.value!.png(),url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='comparison_plot.png';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notice.value=uiMessage('modules.pitch-manipulation.actual_historical_f0_comparison_exported');});}
onMounted(()=>{void attempt(refresh);void poll();timer=setInterval(()=>{if(props.active!==false||jobs.value.some(j=>['queued','running','cancel_requested'].includes(j.state)))void poll();},1000);});onUnmounted(()=>{disposed=true;++epoch;++resultEpoch;previewAbort?.abort();clearInterval(timer);stop();});
useWebSettings(props.stateKey,{...settingsFields(config,['offset']),...settingsFields(numericInput,['speed','pitch_ratio','pitch_hz']),pitchMode,minInput,maxInput,ymin,ymax,refInput,refs,timeRefInput,timeRefs,refDirection,directPlayback});
</script>
<template><ModuleFrame unified fit :label="tr('modules.pitch-manipulation.tempo_and_pitch_workspace')" class="m08-page" :aria-busy="busy||loading">
<template #toolbar><ModuleToolbar><ResearchUpload :files="context.files" accept=".wav,.mp3,.flac" @uploaded="refresh" folders/><button class="primary" v-if="context.files.choose" :disabled="busy" @click="choose"><AppIcon name="folder"/>{{tr('modules.pitch-manipulation.open_audio_folder')}}</button><button class="primary" v-if="context.files.add" :disabled="busy" @click="picker?.click()">{{tr('modules.pitch-manipulation.import_audio')}}</button><button class="primary" v-if="context.files.add" :disabled="busy" @click="folderPicker?.click()">{{tr('modules.pitch-manipulation.import_folder')}}</button><input ref="picker" hidden type="file" multiple accept=".wav,.mp3,.flac" @change="add"/><input ref="folderPicker" hidden type="file" multiple webkitdirectory @change="add"/><button :disabled="busy" @click="attempt(refresh)">{{tr('modules.pitch-manipulation.refresh_files')}}</button><button :aria-pressed="tab==='single'" @click="tab='single'">{{tr('modules.pitch-manipulation.single_file_and_fundamental_frequency')}}</button><button :aria-pressed="tab==='batch'" @click="tab='batch'">{{tr('modules.pitch-manipulation.batch_tempo_and_pitch')}}</button><template #actions><button :disabled="!source||loading" @click="save">{{tr('modules.pitch-manipulation.save_editing_draft')}}</button><button @click="help=true">{{tr('modules.pitch-manipulation.help')}}</button><button @click="emit('references')"><AppIcon name="book"/>{{tr('modules.pitch-manipulation.methods_and_references')}}</button></template></ModuleToolbar></template>

<ModuleWorkbench unified :state-key="stateKey" :left-label="tr('modules.pitch-manipulation.audio_and_parameters')" :center-label="tr('modules.pitch-manipulation.audio_and_fundamental_frequency_plots')" :right-label="tr('modules.pitch-manipulation.tasks_and_history')"><template #left><ModuleSection :label="tr('modules.pitch-manipulation.audio_and_parameters')"><div class="row"><label>{{tr('modules.pitch-manipulation.audio')}}<select :aria-label="tr('modules.pitch-manipulation.tempo_pitch_audio')" :value="source?.id??''" :disabled="busy" @change="selectSource"><option value="">{{tr('modules.pitch-manipulation.select_audio')}}</option><option v-for="file in audioFiles" :key="file.id" :value="file.id">{{file.name}}</option></select></label><span>{{wave.dirty?tr('modules.pitch-manipulation.unsaved_edits'):tr('modules.pitch-manipulation.edits_saved')}}</span></div><VariantSettings v-if="tab==='single'" v-model:mode="pitchMode" :values="numericInput" :summary="variantSummary"/></ModuleSection>
<ModuleSection :label="tr('modules.pitch-manipulation.curves_and_playback')"><div class="row"><label>{{tr('modules.pitch-manipulation.minimum_f0_hz')}}<input v-model="minInput" :aria-label="tr('modules.pitch-manipulation.minimum_f0')" type="text" inputmode="decimal" class="numeric-input"/></label><label>{{tr('modules.pitch-manipulation.maximum_hz')}}<input v-model="maxInput" :aria-label="tr('modules.pitch-manipulation.maximum_f0')" type="text" inputmode="decimal" class="numeric-input"/></label><button @click="applyAxis">{{tr('modules.pitch-manipulation.apply_range')}}</button><label>{{tr('modules.pitch-manipulation.reference_line_orientation')}}<select v-model="refDirection" :aria-label="tr('modules.pitch-manipulation.reference_line_orientation')"><option value="hz">{{tr('modules.pitch-manipulation.horizontal_hz')}}</option><option value="time">{{tr('modules.pitch-manipulation.vertical_seconds')}}</option></select></label><label v-if="refDirection==='time'">{{tr('modules.pitch-manipulation.reference_line_seconds')}}<input v-model="timeRefInput" :aria-label="tr('modules.pitch-manipulation.reference_line_seconds')" type="text" inputmode="decimal" class="numeric-input"/></label><label v-else>{{tr('modules.pitch-manipulation.reference_line_hz')}}<input v-model="refInput" :aria-label="tr('modules.pitch-manipulation.reference_line_hz')" type="text" inputmode="decimal" class="numeric-input"/></label><button @click="addReference">{{tr('modules.pitch-manipulation.add_reference_line')}}</button><button @click="refs=[];timeRefs=[];dirty()">{{tr('modules.pitch-manipulation.clear_reference_lines')}}</button><button :disabled="!times.length" @click="importing=true">{{tr('modules.pitch-manipulation.import_fundamental_frequency_sequence')}}</button></div>
</ModuleSection><details class="advanced" open><summary>{{tr('modules.pitch-manipulation.batch_change_f0_and_turning_points')}}</summary><ModuleSection :label="tr('modules.pitch-manipulation.batch_change_fundamental_frequency')"><div class="row"><button :disabled="!times.length" @click="editPoints">{{tr('modules.pitch-manipulation.edit_turning_point_table')}}</button><span>{{tr('modules.pitch-manipulation.interpolationSummary',{count:Math.max(0,config.points.length-2)})}}</span><label><input v-model="config.offset" type="checkbox"/>{{tr('modules.pitch-manipulation.shift_offset_hz')}}</label><button class="primary" :disabled="busy||loading||batchPending||!port||!times.length" @click="submit('linear')">{{tr('modules.pitch-manipulation.batch_generation')}}</button></div><div class="point-summary"><span v-for="(p,i) in config.points" :key="i">{{p.time}} s：{{p.freqs.join(', ')||tr('modules.pitch-manipulation.not_entered')}} Hz · {{modes.find(m=>m.value===p.mode)?.label}}</span></div></ModuleSection>
</details></template><div v-show="tab==='single'" class="single"><div class="plots"><ModuleSection :label="tr('modules.pitch-manipulation.original_waveform')" :title="tr('modules.pitch-manipulation.original_waveform')"><WaveformViewport :empty-message="tr('modules.pitch-manipulation.select_audio_to_show_its_actual_waveform')" :state="wave" compact-overview auto-amplitude continuous-detail @selection-end="selectPlayback(false)"/></ModuleSection>
<ModuleSection :label="tr('modules.pitch-manipulation.fundamental_frequency_editing')" :title="tr('modules.pitch-manipulation.fundamental_frequency_editing_original_dashed_edited_solid')"><div class="m08-chart-frame"><PitchCurve v-if="times.length" :times="times" :original="original" :modified="modified" :start="range[0]" :end="range[1]" :min="ymin" :max="ymax" :references="refs" :time-references="timeRefs" editable @edit="v=>{modified=v;dirty()}" @pan="pan" @zoom="zoom"/><PlotPlaceholder v-else :message="tr('modules.pitch-manipulation.editing_is_available_after_obtaining_praat_f0')"/></div><div class="plot-footer"><small>{{tr('modules.pitch-manipulation.shift_drag_to_draw_ctrl_drag_to_restore_drag')}}</small></div></ModuleSection>
<ModuleSection :label="tr('modules.pitch-manipulation.synthesized_waveform')" :title="tr('modules.pitch-manipulation.synthesized_waveform')"><WaveformViewport :empty-message="tr('modules.pitch-manipulation.actual_output_appears_after_synthesis')" :state="resultWave" compact-overview auto-amplitude continuous-detail @selection-end="selectPlayback(true)"/><small v-if="result">{{tr('modules.pitch-manipulation.source_interval')}} {{result.start.toFixed(3)}}–{{result.end.toFixed(3)}} {{tr('modules.pitch-manipulation.s_output')}} {{resultWave.asset?.duration.toFixed(3)}} {{tr('modules.pitch-manipulation.s_current_edits_do_not_overwrite_this_result')}}</small></ModuleSection>
<ModuleSection :label="tr('modules.pitch-manipulation.historical_f0_comparison')" :title="tr('modules.pitch-manipulation.historical_f0_comparison_aligned_to_zero_onset')"><div class="m08-chart-frame"><HistoryPlot v-if="history.length" ref="historyPlot" :results="history" :min="ymin" :max="ymax"/><PlotPlaceholder v-else :message="tr('modules.pitch-manipulation.actual_f0_comparison_appears_after_audio_generation')"/></div><div class="plot-footer"><button :disabled="!history.length" @click="exportPng">{{tr('modules.pitch-manipulation.save_comparison_plot')}}</button></div></ModuleSection></div>
</div><ModuleSection v-show="tab==='batch'" :label="tr('modules.pitch-manipulation.folder_batch_processing')" :title="tr('modules.pitch-manipulation.folder_batch_processing_complete_audio')"><VariantSettings v-if="tab==='batch'" v-model:mode="pitchMode" :values="numericInput" :summary="variantSummary"/><div class="row"><button @click="batchIds=audioFiles.map(f=>f.id)">{{tr('modules.pitch-manipulation.select_all_audio')}}</button><button @click="batchIds=[]">{{tr('modules.pitch-manipulation.clear_selection')}}</button><button class="primary" :disabled="busy||batchPending||!port" @click="batch">{{tr('modules.pitch-manipulation.start_batch_processing')}}</button><button v-if="batchSubmitting" @click="cancelSubmit=true">{{tr('modules.pitch-manipulation.stop_submitting_remaining_files')}}</button></div><div class="file-list"><label v-for="f in audioFiles" :key="f.id"><input v-model="batchIds" :value="f.id" type="checkbox"/><span>{{f.name}}</span></label></div><small>{{tr('modules.pitch-manipulation.processes_complete_original_audio_adjusting_tempo_and_pitch_independently')}}</small></ModuleSection>
<template #right><ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-if="notice" :message="notice"/><ModuleStatus v-if="!port" :message="tr('modules.pitch-manipulation.wav_waveform_preview_and_playback_are_available_here_use')"/><ModuleStatus v-if="busy||loading" kind="loading" :message="loading?loadPhase:tr('modules.pitch-manipulation.processing_please_wait')"/><ModuleSection :label="tr('modules.pitch-manipulation.synthesis_and_playback')"><div class="row"><strong>{{tr('modules.pitch-manipulation.current_view')}} {{range[0].toFixed(3)}}–{{range[1].toFixed(3)}} s</strong><button class="primary" :disabled="busy||loading||batchPending||!port||!times.length" @click="submit('synthesize')">{{tr('modules.pitch-manipulation.synthesize_current_view')}}</button><button class="primary" :disabled="busy||loading||batchPending||!port||!times.length" @click="submit('synthesize',true)">{{tr('modules.pitch-manipulation.synthesize_full_audio')}}</button><button class="primary" :disabled="busy||!result||!port?.saveMany" @click="saveAudio">{{tr('modules.pitch-manipulation.save_synthesized_audio')}}</button><button data-audio-selection :disabled="!wave.asset" @click="playDirect(false)">{{tr('modules.pitch-manipulation.play_original_in_current_view')}}</button><button data-audio-selection :disabled="!resultWave.asset" @click="playDirect(true)">{{tr('modules.pitch-manipulation.play_synthesized_audio')}}</button><small v-if="result||history.length" class="generation-location">{{tr('modules.pitch-manipulation.generated_results_are_retained_in_task_history_click_save')}}</small><div v-if="savedOutput" class="save-location"><strong>{{tr('modules.pitch-manipulation.lastSavedCount',{count:savedOutput.names.length})}}</strong><span v-if="savedOutput.directory">{{savedOutput.directory}}</span><span v-else-if="savedOutput.destination">{{tr('modules.pitch-manipulation.saved_to_selected_location')}}{{savedOutput.destination}}</span><span v-else-if="context.files.kind==='desktop'">{{tr('modules.pitch-manipulation.saved_to_the_output_folder_just_selected')}}</span><span v-else>{{tr('modules.pitch-manipulation.browser_download_started_see_the_download_list_for_the')}}</span><details><summary>{{tr('modules.pitch-manipulation.view_saved_file_names')}}</summary><ul><li v-for="name in savedOutput.names" :key="name">{{name}}</li></ul></details></div><span>{{playResult?tr('modules.pitch-manipulation.synthesized_audio_selection'):tr('modules.pitch-manipulation.original_audio_selection')}}</span><AudioTransport :state="playResult?resultWave:wave" :active="active" :external-playback="directPlayback" compact @selection-play="selectionPlayback" @stop="cancelPlayback"/></div></ModuleSection>
<details class="advanced history-details" open><summary>{{tr('modules.pitch-manipulation.historyCount',{count:history.length})}}</summary><ModuleSection :label="tr('modules.pitch-manipulation.current_batch_and_historical_files_131')"><div class="row"><button :disabled="busy||!currentBatch.length||!port" @click="openManage('delete')">{{tr('modules.pitch-manipulation.delete_this_batch_s_audio')}}</button><button :disabled="busy||!currentBatch.length||!port" @click="openManage('rename')">{{tr('modules.pitch-manipulation.batch_rename')}}</button><button class="primary" :disabled="busy||!currentBatch.length||!port?.saveMany" @click="openManage('save')">{{tr('modules.pitch-manipulation.batch_save')}}</button><span>{{tr('modules.pitch-manipulation.batchHistorySummary',{current:currentBatch.length,history:history.length})}}</span></div><ul class="history"><li v-for="(r,i) in history" :key="r.id"><span class="result-name" :title="r.name">{{i+1}} · {{r.name}}</span><button data-audio-selection :aria-pressed="result?.id===r.id" @click="viewResult(r).then(()=>{if(result?.id===r.id)playDirect(true)})">{{tr('modules.pitch-manipulation.play_this_version')}}</button><details><summary>{{tr('modules.pitch-manipulation.generation_parameters')}}</summary><pre>{{JSON.stringify(r.config,null,2)}}</pre></details></li></ul></ModuleSection></details><ModuleSection v-if="lastBatchIds.length" :label="tr('modules.pitch-manipulation.latest_generation_batch')"><strong>{{tr(batchPending?'modules.pitch-manipulation.batchCompletionPending':'modules.pitch-manipulation.batchCompletion',{count:batchResults.length})}}</strong><button class="primary" :disabled="busy||!batchResults.length||!port?.saveMany" @click="batchSavePending=false;openManage('save',batchResults)">{{tr('modules.pitch-manipulation.save_latest_batch')}}</button><small>{{tr('modules.pitch-manipulation.includes_all_originals_from_this_run_select_one_folder')}}</small></ModuleSection>
<details class="advanced task-details" open><summary>{{tr('modules.pitch-manipulation.taskCount',{count:jobs.length})}}</summary><ModuleSection :label="tr('modules.pitch-manipulation.tempo_pitch_tasks_147')"><p v-if="!jobs.length">{{tr('modules.pitch-manipulation.select_and_submit_files_to_view_processing_status_and')}}</p><article v-for="j in jobs" :key="j.id" class="task-row" :data-job-state="j.state"><strong>{{files.find(f=>f.id===j.source_id)?.name??j.source_name??tr('modules.pitch-manipulation.audio')}} · {{j.id.slice(0,8)}}</strong><small v-if="j.config">{{variantLabel(j.config)}}</small><span role="status" :data-job-state="j.state">{{jobLabels[j.state]}}</span><p v-if="j.error" class="error-text">{{errorText(j.error)}}</p><button v-if="['queued','running'].includes(j.state)" @click="attempt(()=>port!.cancel(j.id))">{{tr('modules.pitch-manipulation.cancel_this_task')}}</button><button v-for="r in j.results" :key="r.id" class="task-result" :aria-pressed="result?.id===r.id" @click="viewResult(r)"><span>{{r.name}}</span><small>{{variantLabel(r.config)}}</small><span>{{result?.id===r.id?tr('modules.pitch-manipulation.current_version'):tr('modules.pitch-manipulation.view_this_version')}}</span></button></article></ModuleSection></details>
</template></ModuleWorkbench>
<ModalDialog v-if="importing" :title="tr('modules.pitch-manipulation.import_fundamental_frequency_sequence')" @close="importing=false"><p>{{tr('modules.pitch-manipulation.one_f0_per_line_for_two_columns_the_second')}}</p><textarea v-model="importText" :aria-label="tr('modules.pitch-manipulation.fundamental_frequency_sequence')" rows="8"/><ModuleStatus v-if="error" kind="error" :message="error"/><template #footer><button @click="importing=false">{{tr('modules.pitch-manipulation.cancel')}}</button><button class="primary" @click="applyImport">{{tr('modules.pitch-manipulation.import_sequence')}}</button></template></ModalDialog>
<ModalDialog v-if="editing" :title="tr('modules.pitch-manipulation.edit_turning_point_table')" wide @close="editing=false"><p>{{tr('modules.pitch-manipulation.first_and_last_rows_are_endpoints_and_can_be')}}</p><div class="table-wrap"><table><thead><tr><th>{{tr('modules.pitch-manipulation.select')}}</th><th>{{tr('modules.pitch-manipulation.time_s')}}</th><th>{{tr('modules.pitch-manipulation.frequency_list_hz')}}</th><th>{{tr('modules.pitch-manipulation.connection_type')}}</th></tr></thead><tbody><tr v-for="(r,i) in rows" :key="i"><td><input v-model="r.selected" type="checkbox" :disabled="i===0||i===rows.length-1" :aria-label="tr('modules.pitch-manipulation.select_turning_point')+i"/></td><td><input v-time-precision="'s'" v-model="r.time" type="text" inputmode="decimal" :aria-label="tr('modules.pitch-manipulation.time_162')+i"/></td><td><input v-model="r.freqs" :aria-label="tr('modules.pitch-manipulation.frequency')+i"/></td><td><select v-model="r.mode" :aria-label="tr('modules.pitch-manipulation.connection')+i"><option v-for="m in modes" :key="m.value" :value="m.value">{{m.label}}</option></select></td></tr></tbody></table></div><ModuleStatus v-if="error" kind="error" :message="error"/><template #footer><button @click="rows.splice(rows.length-1,0,{time:'',freqs:'',mode:'order',selected:false})">{{tr('modules.pitch-manipulation.add_row')}}</button><button @click="rows=rows.filter((r,i)=>i===0||i===rows.length-1||!r.selected)">{{tr('modules.pitch-manipulation.delete_selected_turning_points')}}</button><button @click="editing=false">{{tr('modules.pitch-manipulation.cancel')}}</button><button class="primary" @click="savePoints">{{tr('modules.pitch-manipulation.save_changes')}}</button></template></ModalDialog>
<ModalDialog v-if="manage" :title="manage==='delete'?tr('modules.pitch-manipulation.confirm_deletion_of_this_batch_s_audio'):manage==='save'?tr('modules.pitch-manipulation.batch_save'):tr('modules.pitch-manipulation.batch_rename')" :close-disabled="busy" @close="manage=null"><p>{{tr('modules.pitch-manipulation.only_the_checked_generated_results_below_are_affected_original')}}</p><p v-if="manage==='save'">{{context.files.kind==='desktop'?tr('modules.pitch-manipulation.after_save_selected_audio_choose_an_output_folder_once'):tr('modules.pitch-manipulation.save_selected_audio_downloads_wav_for_a_single_version')}}</p><label v-if="manage==='save'" class="export-numbering"><input v-model="numberExports" type="checkbox" :disabled="busy"/>{{tr('modules.pitch-manipulation.add_sequence_numbers_to_file_names_1_2_3')}}</label><label v-if="manage==='rename'">{{tr('modules.pitch-manipulation.new_shared_prefix')}}<input v-model="prefix" :aria-label="tr('modules.pitch-manipulation.new_shared_prefix')"/></label><div class="file-list"><label v-for="r in manageResults" :key="r.id"><input v-model="selectedResults" :value="r.id" type="checkbox" :disabled="busy" @change="exportSequence=undefined"/><span>{{r.name}}</span></label></div><ModuleStatus v-if="error" kind="error" :message="error"/><template #footer><button :disabled="busy" @click="manage=null">{{tr('modules.pitch-manipulation.cancel')}}</button><button :disabled="busy||!chosen.length" @click="manageFiles">{{manage==='delete'?tr('modules.pitch-manipulation.confirm_deletion_of_selected_audio'):manage==='save'?tr('modules.pitch-manipulation.save_selected_audio'):tr('modules.pitch-manipulation.confirm_rename')}}</button></template></ModalDialog>
<ModalDialog v-if="pendingFile" :title="tr('modules.pitch-manipulation.keep_current_edits')" @close="pendingFile=undefined;pendingResult=undefined"><p>{{tr('modules.pitch-manipulation.save_the_current_f0_and_parameter_draft_before_switching')}}</p><template #footer><button @click="pendingFile=undefined;pendingResult=undefined">{{tr('modules.pitch-manipulation.cancel')}}</button><button @click="finishSwitch(false)">{{tr('modules.pitch-manipulation.discard_and_switch')}}</button><button @click="finishSwitch(true)">{{tr('modules.pitch-manipulation.save_and_switch')}}</button></template></ModalDialog>
<ModalDialog v-if="help" :title="tr('modules.pitch-manipulation.tempo_and_pitch_instructions')" wide @close="help=false"><p>{{tr('modules.pitch-manipulation.play_original_in_current_view_plays_the_visible_original')}}</p><p>{{tr('modules.pitch-manipulation.defaults_f0_range_50_350_hz_tempo_1_0')}}</p><p>{{tr('modules.pitch-manipulation.ascending_descending_modes_sort_lists_first_all_combinations_follow')}}</p><p>{{tr('modules.pitch-manipulation.after_drawing_importing_f0_or_successful_batch_synthesis_the')}}</p><p>{{tr('modules.pitch-manipulation.method_src_praat_nonzero_hz_batch_shifts_fix_the')}}</p></ModalDialog>
</ModuleFrame></template>
<style scoped>.m08-page{min-width:0;overflow-wrap:anywhere}.m08-page :deep(.task-row){min-width:0;overflow-wrap:anywhere}.single{display:grid;gap:var(--module-gap)}.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:var(--module-gap)}.row{display:flex;align-items:center;flex-wrap:wrap;gap:8px}.row+.row{margin-top:8px}.row label{display:flex;align-items:center;gap:6px}.row .numeric-input{width:86px}.row select{max-width:400px}.m08-page small{display:block;color:var(--muted);font-size:var(--support-size);line-height:1.6}.m08-page :deep(.wave-track svg){height:max(160px,22vh)}.m08-page :deep(.m08-curve){min-height:170px;height:max(230px,29vh)}.m08-page :deep(.history-plot){min-height:170px;height:max(230px,29vh)}.task-row{display:flex;flex-direction:column;align-items:stretch;gap:6px;margin:10px 0;padding:10px 0;border-bottom:1px solid var(--border)}.task-result{display:flex;flex-direction:column;align-items:flex-start;text-align:left;white-space:normal;overflow-wrap:anywhere;gap:4px}.variant-settings{margin:10px 0}.advanced>summary{cursor:pointer;padding:6px 10px;border:1px solid var(--border);border-radius:var(--radius);background:var(--panel)}.point-summary,.file-list{display:flex;flex-direction:column;gap:8px;margin-top:10px}.history{padding-left:20px;max-height:260px;overflow:auto}.history li{margin:10px 0;padding-bottom:10px;border-bottom:1px solid var(--border);overflow-wrap:anywhere}.result-name{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;line-height:1.5}.history li>button{margin:6px 0}.history li details{color:var(--muted)}.history button{margin-left:8px}.history pre{white-space:pre-wrap}.export-numbering{display:flex;align-items:center;gap:8px;margin-top:12px}.export-numbering input{width:16px;height:16px;flex:0 0 16px}.file-list label{display:flex;align-items:flex-start;gap:8px;min-width:0;overflow-wrap:anywhere}.file-list input[type=checkbox]{flex:0 0 16px;width:16px;min-width:16px;height:16px;margin-top:2px}.file-list label>span{min-width:0;flex:1}.m08-chart-frame{display:flex;min-width:0;min-height:170px;height:max(230px,29vh);background:var(--app);border-radius:var(--radius);overflow:hidden}.m08-chart-frame>:deep(*){flex:1;min-width:0;min-height:0;height:100%}.plot-footer{display:flex;align-items:center;min-height:var(--control-height);margin-top:4px}.plot-footer small{margin:0}.plot-footer button{width:100%}.save-location{display:flex;flex-direction:column;gap:4px;padding:8px;border:1px solid var(--border);border-radius:var(--radius);background:var(--app);font-size:var(--support-size);overflow-wrap:anywhere}.save-location ul{padding-left:18px;margin:4px 0}.save-location details{max-height:160px;overflow:auto}.table-wrap{overflow:auto}table input{max-width:180px}textarea{width:100%;background:var(--panel);color:var(--text);border:1px solid var(--border);font:inherit}button[aria-pressed=true]{background:var(--selected);color:var(--accent)}@media(max-width:1000px){.plots{grid-template-columns:1fr}.row select{max-width:100%}}@media(max-width:600px){.row label{flex-wrap:wrap;max-width:100%}.row select{width:100%}.row input{max-width:100%}}.m08-page :deep(.workbench-left) .row,.m08-page :deep(.workbench-right) .row{align-items:stretch;flex-direction:column}.m08-page :deep(.workbench-left) .row label{justify-content:space-between;min-width:0;flex-wrap:wrap}.m08-page :deep(.workbench-left) select{width:100%;max-width:100%}.m08-page .task-details[open]{max-height:none;overflow:visible}
@container module (min-width:1061px){.single{flex:1;min-height:620px;display:flex}.plots{flex:1;grid-template-rows:repeat(2,minmax(0,1fr))}.plots>.module-section{display:flex;flex-direction:column;min-height:0}.plots .m08-chart-frame{flex:1;height:auto;min-height:170px}.plots .m08-chart-frame :deep(.m08-curve),.plots .m08-chart-frame :deep(.history-plot){height:100%;min-height:0}.plots :deep(.wave-viewport){flex:1;display:flex;flex-direction:column}.plots :deep(.wave-track){flex:1;grid-template-rows:auto minmax(0,1fr) auto}.plots :deep(.wave-track svg){height:100%;min-height:100px}}
</style>
