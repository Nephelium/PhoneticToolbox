<script setup lang="ts">
import {uiError,uiErrorMessage} from '../module-localization.ts';
import {uiMessage,joinMessages} from '../module-localization.ts';
import {tr,messageRef} from '../../i18n/index.ts';
import {useWebSettings,settingsFields} from '../../state/web-settings.ts';
import ResearchUpload from '../../components/ResearchUpload.vue';
import AppIcon from '../../components/AppIcon.vue';
import F0Comparison from './F0Comparison.vue';
import SourceAcknowledgement from './SourceAcknowledgement.vue';
import {taskDescription,taskTime,taskStates,type TaskSnapshot} from './history.ts';
import {stepColor,playingStep,type F0Track} from './f0.ts';
import type {M07F0Display} from '../../platform/m07.ts';
import {vTimePrecision} from '../../design/time-precision.ts';
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import {computed,ref,reactive,watch,onMounted,onUnmounted,markRaw,nextTick} from 'vue';
import type {ResearchContext,ResearchFile,JobView} from '../../platform/research.ts';
import {workspace,host} from '../../state/workspace.ts';
import {stop,playback,isCurrentAudio} from '../../state/audio.ts';
import {decodeWav} from '../../platform/decode.ts';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import ModuleSection from '../../components/ModuleSection.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import ModalDialog from '../../components/ModalDialog.vue';
import WaveformViewport from '../../components/WaveformViewport.vue';
import AudioTransport from '../../components/AudioTransport.vue';
import {defaults,generationDefaults,labels,kinds,designs,clone,validate,validatePoints,table,current,type Points,type Analysis} from './state.ts';
import {downloadM07,messages,m07Message,type M07Result,type Settings} from '../../platform/m07.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active:boolean}>();const emit=defineEmits<{references:[]}>();
const analysis=reactive(defaults()),generation=reactive(generationDefaults()),alignment=ref<'normalize'|'onset'>('normalize'),count=ref(21),kind=ref<1|2|3>(2),reverse=ref(false);
const wave=workspace(props.stateKey),sourceWave=workspace(props.stateKey+'.source'),targetWave=workspace(props.stateKey+'.target');
const files=ref<ResearchFile[]>([]),source=ref<ResearchFile>(),target=ref<ResearchFile>(),directory=ref(''),output=ref(''),outputLabel=ref(''),picker=ref<HTMLInputElement>();
const error=messageRef(''),notice=messageRef(''),busy=ref(false),help=ref(false),applied=ref<M07Result>(),points=ref<Points>({axis:[],source:[],target:[]}),pending=ref(false),analysisIdentity=ref('');
// Keep typed text independent of its numeric value until the table is replaced.
const pointText=ref<{source:string[];target:string[]}>({source:[],target:[]});
watch(points,value=>{pointText.value={source:value.source.map(v=>v===0?'':String(v)),target:value.target.map(v=>v===0?'':String(v))};},{flush:'sync',immediate:true});
const jobs=ref<JobView[]>([]),results=ref<M07Result[]>([]),selected=ref<M07Result>(),playName=ref(''),batchSummary=messageRef('');
const f0Cache=new Map<string,M07F0Display>();
const selectedJob=ref(''),resultTransport=ref<InstanceType<typeof AudioTransport>>(),historyLoading=ref(false);
const taskSnapshots=reactive<Record<string,TaskSnapshot>>({}),f0Tracks=ref<F0Track[]>([]),f0Note=messageRef('');
const selectedResult=computed(()=>results.value.find(result=>result.job===selectedJob.value));
const plotPoints=computed(()=>selected.value?.metadata.controls??points.value);
const plotAlignment=computed(()=>selected.value?.metadata.alignment??alignment.value);
const f0Plot=ref<InstanceType<typeof F0Comparison>>(),exportingF0=ref(false),axisError=messageRef('');
const yMinimum=ref<number|string>(0),yMaximum=ref<number|string>(''),axisRange=ref<{minimum:number;maximum:number|null}>({minimum:0,maximum:null});
const axisKey='m07.f0-axis.'+props.stateKey;
const playingF0=computed(()=>{
 if(!playback.playing||!isCurrentAudio(wave.asset,wave.channel)||!selected.value||!wave.asset)return '';
 return playName.value==='combined_steps.wav'?playingStep(playback.position,wave.asset.sampleRate,wave.asset.frames,selected.value.metadata.generation.step_count):playName.value.replace(/\.wav$/,'');
});
function updateAxis(persist=true){
 const minimum=Number(yMinimum.value),maximum=yMaximum.value===''?null:Number(yMaximum.value);
 if(yMinimum.value===''||!Number.isFinite(minimum)||minimum<0||maximum!==null&&(!Number.isFinite(maximum)||maximum<=minimum)){axisError.value=uiMessage('modules.phonation-synthesis.minimum_must_be_nonnegative_and_maximum_must_exceed_minimum');return;}
 axisRange.value={minimum,maximum};axisError.value='';
 if(persist&&!host.projects.write(axisKey,axisRange.value))axisError.value=uiMessage('modules.phonation-synthesis.display_range_applied_but_local_saving_failed');
}
function resetAxis(){yMinimum.value=0;yMaximum.value='';updateAxis();}
async function exportF0(){
 if(exportingF0.value)return;exportingF0.value=true;
 const filename=selected.value?'M07-'+selected.value.job.slice(0,8)+'-F0.png':'M07-F0.png';
 await attempt(async()=>{const blob=await f0Plot.value!.png();downloadM07(await blob.arrayBuffer(),filename);});
 exportingF0.value=false;
}
const inputEpoch={source:0,target:0};
let restoredDraft:{sourceHash?:string;targetHash?:string;analysisHash?:string;points:Points;alignment:string;count:number}|undefined;
let epoch=0,selectionEpoch=0,auditionEpoch=0,disposed=false,abort:AbortController|undefined,currentJob:string|undefined;
const port=computed(()=>props.context.files.m07);
const identity=computed(()=>JSON.stringify([source.value?.id,source.value?.sha256,target.value?.id,target.value?.sha256,analysis]));
const stale=computed(()=>!!applied.value&&analysisIdentity.value!==identity.value);
const ready=computed(()=>!!applied.value&&!stale.value&&!pending.value&&!busy.value);
const numericKeys=Object.keys(labels) as (keyof Analysis)[];
watch(analysis,()=>{wave.dirty=true;stop();},{deep:true,flush:'sync'});
watch(generation,()=>{wave.dirty=true;stop();},{deep:true,flush:'sync'});
watch([alignment,count],()=>{wave.dirty=true;pending.value=true;stop();void attempt(()=>rebuild());});
watch([kind,reverse],()=>{wave.dirty=true;stop();});
watch(()=>props.active,v=>{if(!v){auditionEpoch++;stop();}});
function report(e:unknown){error.value=m07Message(e);}
async function attempt(work:()=>unknown|Promise<unknown>,valid=()=>true){error.value='';try{await work();}catch(e){if(!disposed&&valid())report(e);}}
function updateJob(job:JobView,snapshot?:TaskSnapshot){if(snapshot)taskSnapshots[job.id]=clone(snapshot);const i=jobs.value.findIndex(j=>j.id===job.id);if(i>=0)jobs.value[i]=job;else jobs.value.unshift(job);currentJob=job.id;}
async function refresh(){files.value=(await props.context.files.list(directory.value||undefined)).filter(f=>/\.wav$/i.test(f.name));}
async function choose(){await attempt(async()=>{const grant=await props.context.files.choose?.('input');if(grant){directory.value=grant.id;await refresh();}});}
async function add(event:Event){await attempt(async()=>{const el=event.target as HTMLInputElement;props.context.files.add?.([...el.files??[]]);el.value='';await refresh();});}
async function chooseOutput(){await attempt(async()=>{const grant=await props.context.files.choose?.('output');if(grant){output.value=grant.id;outputLabel.value=grant.label;}});}
async function select(role:'source'|'target',id:string){
 const file=files.value.find(f=>f.id===id);if(role==='source')source.value=file;else target.value=file;
 wave.dirty=true;stop();const ticket=++inputEpoch[role];selectionEpoch++;epoch++;abort?.abort();busy.value=false;
 const w=role==='source'?sourceWave:targetWave;w.asset=null;
 if(!file)return;
 await attempt(async()=>{const data=await props.context.files.read(file);const asset=await decodeWav(data.buffer,file.name);if(disposed||ticket!==inputEpoch[role]||(role==='source'?source.value:target.value)?.id!==id)return;if(asset.duration>10)throw uiError(uiMessage('modules.phonation-synthesis.each_input_audio_file_is_limited_to_10_seconds'));file.sha256=data.sha256;w.asset=markRaw(asset);w.start=0;w.end=asset.duration;w.zoom=1;w.offset=0;},()=>ticket===inputEpoch[role]);
}
function rebuild(){if(applied.value)points.value=table(applied.value.metadata.source_f0,applied.value.metadata.target_f0,count.value,alignment.value);}
function edit(role:'source'|'target',i:number,event:Event){const text=(event.target as HTMLInputElement).value;points.value[role][i]=text.trim()===''?0:Number(text);pending.value=true;wave.dirty=true;stop();}
function settings(action:Settings['action']):Settings{return {schema_version:'m07/1',batch_group_count:1,batch_group_index:0,action,analysis:clone(analysis),generation:clone(generation),alignment:alignment.value,point_count:count.value,continuum_type:kind.value,reverse_direction:reverse.value,analysis_job_id:action==='analyze'?null:applied.value?.job,controls:action==='apply'?validatePoints(clone(points.value)):null};}
async function run(action:'analyze'|'apply'){
 await attempt(async()=>{
  if(busy.value)return;if(!port.value||!source.value||!target.value)throw uiError(uiMessage('modules.phonation-synthesis.select_source_and_target_audio_first'));validate(analysis,generation);
  if(action==='apply'&&(!applied.value||stale.value))throw uiError(uiMessage('modules.phonation-synthesis.analysis_is_no_longer_valid_extract_f0_again'));
  const config=settings(action),ticket=++epoch,revision=identity.value;selectionEpoch++;f0Tracks.value=[];f0Note.value='';abort=new AbortController();busy.value=true;wave.dirty=true;stop();
  try{const result=await port.value.run(source.value,target.value,config,abort.signal,j=>{if(ticket===epoch&&!disposed)updateJob(j,config);});
   if(disposed||!current(ticket,epoch,revision,identity.value)){if(ticket===epoch)notice.value=uiMessage('modules.phonation-synthesis.input_or_analysis_parameters_changed_late_analysis_was_not');return;}
   taskSnapshots[result.job]=result.metadata;selected.value=undefined;f0Tracks.value=[];f0Note.value='';applied.value=result;analysisIdentity.value=identity.value;points.value=clone(result.metadata.controls);pending.value=false;notice.value=action==='analyze'?uiMessage('modules.phonation-synthesis.analysis_complete_control_points_are_unchanged'):uiMessage('modules.phonation-synthesis.f0_edits_applied_lpc_residuals_and_pulses_are_reused');
   if(action==='analyze'&&restoredDraft&&restoredDraft.sourceHash===source.value?.sha256&&restoredDraft.targetHash===target.value?.sha256&&restoredDraft.alignment===alignment.value&&restoredDraft.count===count.value&&restoredDraft.analysisHash===JSON.stringify(analysis)){points.value=clone(restoredDraft.points);pending.value=true;restoredDraft=undefined;notice.value=uiMessage('modules.phonation-synthesis.same_input_reanalysed_and_draft_control_points_restored_apply');}
  }catch(e){if(ticket===epoch&&!disposed)throw e;}finally{if(ticket===epoch)busy.value=false;}
 });
}
async function generate(all=false){
 await attempt(async()=>{
  if(!ready.value||!port.value||!source.value||!target.value)throw uiError(pending.value?uiMessage('modules.phonation-synthesis.apply_control_point_edits_first'):stale.value?uiMessage('modules.phonation-synthesis.analysis_is_no_longer_valid_extract_f0_again'):uiMessage('modules.phonation-synthesis.complete_analysis_first'));validate(analysis,generation);
  const snapshot=settings('generate'),sourceFile=clone(source.value),targetFile=clone(target.value),plan=all?designs():[{kind:kind.value,reverse:reverse.value}],batch=crypto.randomUUID();
  const ticket=++epoch;abort=new AbortController();const signal=abort.signal;busy.value=true;wave.dirty=true;stop();let complete=0;batchSummary.value=uiMessage('modules.phonation-synthesis.0_value_groups_completed',{v0:(plan.length)});
  try{for(const design of plan){
   if(signal.aborted||ticket!==epoch)break;
   const config={...snapshot,batch_id:batch,batch_group_count:plan.length===6?6 as const:1 as const,batch_group_index:complete,continuum_type:design.kind,reverse_direction:design.reverse};
   const result=await port.value.run(sourceFile,targetFile,config,signal,j=>{if(ticket===epoch&&!disposed)updateJob(j,config);});
   if(disposed||ticket!==epoch)return;
   taskSnapshots[result.job]=result.metadata;if(!results.value.some(r=>r.job===result.job))results.value.unshift(result);await selectTask(result.job);complete++;batchSummary.value=uiMessage('modules.phonation-synthesis.value_value_groups_completed',{v0:(complete),v1:(plan.length)});
  }notice.value=complete===plan.length?'':uiMessage('modules.phonation-synthesis.generation_stopped_complete_groups_are_retained');
  }catch(e){if(ticket===epoch&&!disposed)throw e;}finally{if(ticket===epoch){busy.value=false;if(complete<plan.length)batchSummary.value=uiMessage('modules.phonation-synthesis.partially_completed_value_value_groups_others_are_incomplete_or',{v0:(complete),v1:(plan.length)});}}
 });
}
async function cancel(id=currentJob){abort?.abort();if(id&&port.value)await attempt(()=>port.value!.cancel(id));}
async function loadHistory(){
 if(historyLoading.value)return;historyLoading.value=true;
 await attempt(async()=>{
  if(!port.value)return;const ticket=epoch,selection=selectionEpoch;
  const history=await port.value.list();if(disposed||ticket!==epoch)return;jobs.value=history;
  for(const job of history.filter(j=>j.state==='succeeded')){
   const result=results.value.find(result=>result.job===job.id)??await port.value.result(job);
   if(disposed||ticket!==epoch)return;taskSnapshots[job.id]=result.metadata;
   if(result.metadata.action==='generate'&&!results.value.some(r=>r.job===job.id))results.value.push(result);
  }
  if(!selectedJob.value&&selection===selectionEpoch){const first=history.find(job=>results.value.some(result=>result.job===job.id));if(first)await selectTask(first.id);}
  notice.value='';
 });historyLoading.value=false;
}
async function retry(job:JobView){await attempt(async()=>{if(busy.value||!port.value)return;updateJob(await port.value.retry(job.id),taskSnapshots[job.id]);notice.value=uiMessage('modules.phonation-synthesis.retry_submitted_refresh_history_to_view_results');});}
async function selectTask(id:string){
 selectedJob.value=id;const result=selectedResult.value;
 if(result){await play(result,'combined_steps.wav',false);return;}
 selectionEpoch++;stop();selected.value=undefined;playName.value='';wave.asset=null;f0Tracks.value=[];f0Note.value='';
}
async function play(result:M07Result,name:string,autoplay=true){
 const ticket=++selectionEpoch,audition=++auditionEpoch;selectedJob.value=result.job;selected.value=result;playName.value=name;wave.asset=null;f0Tracks.value=[];f0Note.value='';stop();
 await attempt(async()=>{
  if(!port.value)throw uiError(uiMessage('modules.phonation-synthesis.synthesis_results_cannot_be_read_here_use_the_desktop'));
  const raw=await port.value.read(result,name);const asset=await decodeWav(raw,name);
  if(disposed||ticket!==selectionEpoch)return;
  wave.asset=markRaw(asset);wave.start=0;wave.end=0;wave.zoom=1;wave.offset=0;wave.channel=0;
  await nextTick();if(disposed||ticket!==selectionEpoch)return;
  await resultTransport.value?.selectAll(autoplay&&props.active&&audition===auditionEpoch);
  if(disposed||ticket!==selectionEpoch)return;
  notice.value='';
  void showF0(result,name,ticket);
 },()=>ticket===selectionEpoch);
}
async function showF0(result:M07Result,name:string,ticket:number){
 const valid=()=>!disposed&&ticket===selectionEpoch;
 if(!port.value?.f0){if(valid())f0Note.value=uiMessage('modules.phonation-synthesis.this_entry_does_not_yet_provide_synthesized_f0_curves');return;}
 if(valid())f0Note.value=uiMessage('modules.phonation-synthesis.reading_this_group_s_synthesized_f0');
 try{
  const display=f0Cache.get(result.job)??await port.value.f0(result);if(!valid())return;f0Cache.set(result.job,display);
  f0Tracks.value=display.curves.flatMap((curve,index)=>name==='combined_steps.wav'||curve.name+'.wav'===name?[{...curve,color:stepColor(index)}]:[]);
  f0Note.value=uiMessage('modules.phonation-synthesis.synthesized_f0_uses_this_task_s_generated_trajectory_each');
 }catch(e){if(valid())f0Note.value=joinMessages(uiMessage('modules.phonation-synthesis.failed_to_read_synthesized_f0'),m07Message(e));}
}
async function download(result:M07Result,name:string){const ticket=epoch;await attempt(async()=>{downloadM07(await port.value!.read(result,name),name);if(ticket===epoch)notice.value=uiMessage('modules.phonation-synthesis.file_sent_to_browser_for_download');},()=>ticket===epoch);}
async function saveResult(result:M07Result){const ticket=epoch;await attempt(async()=>{if(props.context.files.kind==='desktop'&&!output.value){await chooseOutput();if(!output.value)return;}await port.value!.save(result,output.value);if(ticket===epoch)notice.value=uiMessage('modules.phonation-synthesis.saved_all_files_and_parameter_manifest_for_this_group');},()=>ticket===epoch);}
function save(){try{if(busy.value)throw uiError(uiMessage('modules.phonation-synthesis.task_is_still_running_finish_or_cancel_it_first'));validate(analysis,generation);if(points.value.axis.length)validatePoints(points.value);const draft={sourceHash:applied.value?.metadata.inputs.source.sha256??restoredDraft?.sourceHash,targetHash:applied.value?.metadata.inputs.target.sha256??restoredDraft?.targetHash,analysisHash:applied.value?JSON.stringify(applied.value.metadata.analysis):restoredDraft?.analysisHash,analysis:clone(analysis),generation:clone(generation),alignment:alignment.value,count:count.value,kind:kind.value,reverse:reverse.value,points:clone(points.value),pending:pending.value};if(!host.projects.write('m07.draft.'+props.stateKey,draft))throw uiError(uiMessage('modules.phonation-synthesis.failed_to_save_draft'));wave.dirty=false;notice.value=uiMessage('modules.phonation-synthesis.parameters_and_unapplied_control_points_saved_as_draft');return true;}catch(e){report(e);return false;}}
defineExpose({save});
onMounted(()=>{const saved=host.projects.read<{minimum:number;maximum:number|null}|null>(axisKey,null);if(saved){yMinimum.value=saved.minimum;yMaximum.value=saved.maximum??'';updateAxis(false);}});
onMounted(()=>{const draft=host.projects.read<{sourceHash?:string;targetHash?:string;analysisHash?:string;analysis:Analysis;generation:ReturnType<typeof generationDefaults>;alignment:'normalize'|'onset';count:number;kind:1|2|3;reverse:boolean;points:Points;pending:boolean}|null>('m07.draft.'+props.stateKey,null);if(draft){try{validate(draft.analysis,draft.generation);Object.assign(analysis,draft.analysis);Object.assign(generation,draft.generation);alignment.value=draft.alignment;count.value=draft.count;kind.value=draft.kind;reverse.value=draft.reverse;points.value=draft.points;restoredDraft=draft;pending.value=!!draft.points.axis.length;notice.value=uiMessage('modules.phonation-synthesis.parameter_draft_restored_reselect_and_analyse_input_before_applying');wave.dirty=false;}catch(e){report(e);}}void attempt(refresh);});
onUnmounted(()=>{disposed=true;epoch++;selectionEpoch++;abort?.abort();stop();sourceWave.asset=null;targetWave.asset=null;});
useWebSettings(props.stateKey,{...settingsFields(analysis,Object.keys(defaults()) as (keyof Analysis)[]),...settingsFields(generation,Object.keys(generationDefaults()) as (keyof typeof generation)[]),alignment,count,kind,reverse,axisRange});
onMounted(()=>{if(props.stateKey.startsWith('server:')){yMinimum.value=axisRange.value.minimum;yMaximum.value=axisRange.value.maximum??'';}});
</script>
<template><ModuleFrame unified fit :label="tr('modules.phonation-synthesis.phonation_type_continuum_workspace')" class="m07-page" :aria-busy="busy">
<template #toolbar><ModuleToolbar><ResearchUpload :files="context.files" accept=".wav" @uploaded="refresh"/><button class="primary" v-if="context.files.choose" @click="choose"><AppIcon name="folder"/>{{tr('modules.phonation-synthesis.open_audio_folder')}}</button><button class="primary" v-else-if="context.files.add" @click="picker?.click()">{{tr('modules.phonation-synthesis.add_wav')}}</button><button @click="attempt(refresh)">{{tr('modules.phonation-synthesis.refresh_files')}}</button><input ref="picker" hidden type="file" accept=".wav" multiple @change="add"/><button v-if="context.files.choose" @click="chooseOutput">{{tr('modules.phonation-synthesis.output_location')}}</button><span>{{outputLabel||(context.files.kind==='server'?tr('modules.phonation-synthesis.project_managed_results_download_to_save'):tr('modules.phonation-synthesis.no_output_location_selected'))}}</span><template #actions><button @click="save">{{tr('modules.phonation-synthesis.save_parameter_draft')}}</button><button @click="help=true">{{tr('modules.phonation-synthesis.help')}}</button><button @click="emit('references')"><AppIcon name="book"/>{{tr('modules.phonation-synthesis.methods_and_references')}}</button></template></ModuleToolbar></template>

<div class="m07-workspace"><ModuleWorkbench unified :state-key="stateKey" :left-label="tr('modules.phonation-synthesis.f0_value_table')" :center-label="tr('modules.phonation-synthesis.audio_and_fundamental_frequency_plots')" :right-label="tr('modules.phonation-synthesis.continuum_and_tasks')"><template #left><ModuleSection :label="tr('modules.phonation-synthesis.input_audio')" class="source-selectors"><div v-for="role in ['source','target'] as const" :key="role"><label>{{role==='source'?tr('modules.phonation-synthesis.source_audio'):tr('modules.phonation-synthesis.target_audio')}}<select :aria-label="role==='source'?tr('modules.phonation-synthesis.source_audio'):tr('modules.phonation-synthesis.target_audio')" :value="(role==='source'?source:target)?.id??''" @change="select(role,($event.target as HTMLSelectElement).value)"><option value="">{{tr('modules.phonation-synthesis.select_wav')}}</option><option v-for="file in files" :key="file.id" :value="file.id">{{file.name}}</option></select></label></div></ModuleSection><ModuleSection :label="tr('modules.phonation-synthesis.analysis_parameters')"><div class="controls"><label>{{tr('modules.phonation-synthesis.f0_backend')}}<select v-model="analysis.f0_backend" :aria-label="tr('modules.phonation-synthesis.f0_backend')"><option value="parselmouth">Parselmouth / Praat</option><option value="reaper">REAPER</option></select></label><label v-for="key in numericKeys.slice(0,3)" :key="key">{{labels[key]}}<input v-time-precision="key.endsWith('_ms')?'ms':undefined" v-model.number="analysis[key]" type="number" step="any" :aria-label="labels[key]"/></label><button class="primary" :disabled="busy||!port||!source||!target||!sourceWave.asset||!targetWave.asset" @click="run('analyze')">{{tr('modules.phonation-synthesis.extract_f0')}}</button><span>{{tr('modules.phonation-synthesis.target_sample_rate_11025_hz')}}</span></div><details><summary>{{tr('modules.phonation-synthesis.advanced_analysis_parameters')}}</summary><div class="controls"><label v-for="key in numericKeys.slice(3)" :key="key">{{labels[key]}}<input v-time-precision="key.endsWith('_ms')?'ms':undefined" v-model.number="analysis[key]" type="number" step="any" :aria-label="labels[key]"/></label><label>{{tr('modules.phonation-synthesis.window_function')}}<select v-model="analysis.window_name"><option v-for="name in ['hamming','hann','blackman','rectangular']" :key="name">{{name}}</option></select></label><label class="check"><input v-model="analysis.trim_silence" type="checkbox"/>{{tr('modules.phonation-synthesis.trim_silence')}}</label></div></details></ModuleSection>
<div class="controls"><label>{{tr('modules.phonation-synthesis.f0_alignment')}}<select v-model="alignment" :aria-label="tr('modules.phonation-synthesis.f0_alignment')" :disabled="busy"><option value="normalize">{{tr('modules.phonation-synthesis.normalize_voiced_duration')}}</option><option value="onset">{{tr('modules.phonation-synthesis.align_onsets')}}</option></select></label><label>{{tr('modules.phonation-synthesis.number_of_f0_control_points')}}<input v-model.number="count" type="number" min="20" max="200" :aria-label="tr('modules.phonation-synthesis.number_of_f0_control_points')" :disabled="busy"/></label><button class="primary" :disabled="busy||!applied||stale" @click="run('apply')">{{tr('modules.phonation-synthesis.apply_edits')}}</button><button :disabled="!applied||pending||stale" @click="download(applied!,'edited_f0.csv')">{{tr('modules.phonation-synthesis.save_f0_csv')}}</button></div><ModuleSection :label="tr('modules.phonation-synthesis.f0_plot_vertical_axis')" class="f0-axis-controls"><strong>{{tr('modules.phonation-synthesis.f0_plot_vertical_axis_hz')}}</strong><div class="axis-inputs"><label>{{tr('modules.phonation-synthesis.minimum')}}<input v-model.number="yMinimum" type="number" min="0" step="any" :aria-label="tr('modules.phonation-synthesis.f0_plot_minimum')" @input="updateAxis()"/></label><label>{{tr('modules.phonation-synthesis.maximum')}}<input v-model.number="yMaximum" type="number" min="0" step="any" :placeholder="tr('modules.phonation-synthesis.auto')" :aria-label="tr('modules.phonation-synthesis.f0_plot_maximum')" @input="updateAxis()"/></label><button @click="resetAxis">{{tr('modules.phonation-synthesis.automatic_range')}}</button></div><small v-if="axisError" role="alert" class="error-text">{{axisError}}</small></ModuleSection><div class="table-scroll"><table v-if="points.axis.length"><thead><tr><th>{{alignment==='normalize'?tr('modules.phonation-synthesis.aligned_time'):tr('modules.phonation-synthesis.aligned_time_ms')}}</th><th>{{tr('modules.phonation-synthesis.source_f0_hz')}}</th><th>{{tr('modules.phonation-synthesis.target_f0_hz')}}</th></tr></thead><tbody><tr v-for="(x,i) in points.axis" :key="i"><td>{{x.toFixed(3)}}</td><td v-for="role in ['source','target'] as const" :key="role"><input v-model="pointText[role][i]" type="text" inputmode="decimal" :aria-label="tr('modules.phonation-synthesis.value_f0_point_value',{v0:(role==='source'?tr('modules.phonation-synthesis.source'):tr('modules.phonation-synthesis.target')),v1:(i+1)})" :disabled="busy" @input="edit(role,i,$event)"/></td></tr></tbody></table><ModuleStatus v-else kind="empty" :message="tr('modules.phonation-synthesis.f0_appears_after_analysis_edits_must_be_applied_explicitly')"/></div></template>
<div class="m07-plots">
 <ModuleSection v-for="role in ['source','target'] as const" :key="role" :label="role==='source'?tr('modules.phonation-synthesis.source_audio'):tr('modules.phonation-synthesis.target_audio')" :title="role==='source'?tr('modules.phonation-synthesis.source_audio'):tr('modules.phonation-synthesis.target_audio')" class="audio-plot" :class="role+'-plot'">
  <WaveformViewport :empty-message="tr('modules.phonation-synthesis.select_audio_to_show_its_actual_waveform')" :state="role==='source'?sourceWave:targetWave" auto-amplitude compact-overview continuous-detail/>
  <AudioTransport :state="role==='source'?sourceWave:targetWave" :active="active" compact/>
 </ModuleSection>
 <ModuleSection :label="tr('modules.phonation-synthesis.fundamental_frequency_curve_comparison')" :title="tr('modules.phonation-synthesis.f0_curve_comparison')" class="main-f0">
  <template #actions><button :disabled="exportingF0||(!plotPoints.axis.length&&!f0Tracks.length)" @click="exportF0">{{exportingF0?tr('modules.phonation-synthesis.exporting'):tr('modules.phonation-synthesis.export_image')}}</button></template><F0Comparison ref="f0Plot" :minimum="axisRange.minimum" :maximum="axisRange.maximum" :active-step="playingF0" :points="plotPoints" :alignment="plotAlignment" :tracks="f0Tracks" :note="f0Note||(selected?tr('modules.phonation-synthesis.task')+selected.job.slice(0,8)+tr('modules.phonation-synthesis.f0_snapshot'):tr('modules.phonation-synthesis.current_f0_control_points'))"/>
 </ModuleSection>
 <ModuleSection :label="tr('modules.phonation-synthesis.synthesized_audio')" :title="tr('modules.phonation-synthesis.synthesized_audio')" class="audio-plot synthesized-plot">
  <WaveformViewport :empty-message="tr('modules.phonation-synthesis.full_group_audio_shown_by_default_after_generation')" :state="wave" auto-amplitude compact-overview continuous-detail/>
  <AudioTransport ref="resultTransport" :state="wave" :active="active" compact/>
  <small v-if="selected" class="current-audio">{{tr('modules.phonation-synthesis.nowPlayingSummary',{description:taskDescription(selected.metadata),name:playName==='combined_steps.wav'?tr('modules.phonation-synthesis.whole_group'):playName,job:selected.job.slice(0,8)})}}</small>
 </ModuleSection>
</div>
<template #right><ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-if="stale" kind="info" :message="tr('modules.phonation-synthesis.input_or_analysis_parameters_changed_invalidating_the_analysis_extract')"/><ModuleStatus v-if="pending" kind="info" :message="tr('modules.phonation-synthesis.control_points_or_alignment_have_unapplied_changes_click_apply')"/><ModuleStatus v-if="notice" kind="info" :message="notice"/><ModuleStatus v-if="!port" kind="info" :message="tr('modules.phonation-synthesis.phonation_type_synthesis_is_unavailable_here_use_the_desktop')"/><ModuleSection :label="tr('modules.phonation-synthesis.continuum_and_amplitude')" class="continuum-controls"><div class="parameter-stack"><label>{{tr('modules.phonation-synthesis.continuum_type')}}<select v-model.number="kind" :aria-label="tr('modules.phonation-synthesis.continuum_type')"><option v-for="(label,id) in kinds" :key="id" :value="Number(id)">{{label}}</option></select></label><label>{{tr('modules.phonation-synthesis.synthesis_direction')}}<select v-model="reverse" :aria-label="tr('modules.phonation-synthesis.synthesis_direction')"><option :value="false">{{tr('modules.phonation-synthesis.source_to_target')}}</option><option :value="true">{{tr('modules.phonation-synthesis.target_to_source')}}</option></select></label><label>{{tr('modules.phonation-synthesis.continuum_steps')}}<input v-model.number="generation.step_count" type="number" min="2" max="50" :aria-label="tr('modules.phonation-synthesis.continuum_steps')"/></label><label class="check"><input v-model="generation.energy_match" type="checkbox"/>{{tr('modules.phonation-synthesis.match_period_energy')}}</label><label class="check"><input v-model="generation.normalize_to_source" type="checkbox"/>{{tr('modules.phonation-synthesis.match_output_amplitude_to_source_audio')}}</label><label>{{tr('modules.phonation-synthesis.output_peak_limit')}}<input v-model.number="generation.output_peak_limit" type="number" step="0.01" min="0.01" max="1"/></label><div class="generate-actions"><button class="primary" :disabled="!ready||!port" @click="generate(false)">{{tr('modules.phonation-synthesis.generate_current')}}</button><button class="primary" :disabled="!ready||!port" @click="generate(true)">{{tr('modules.phonation-synthesis.generate_all_six_groups')}}</button><button v-if="busy" class="cancel-generation" @click="cancel()">{{tr('modules.phonation-synthesis.cancel_task')}}</button></div><small>{{tr('modules.phonation-synthesis.sixGroupsSummary',{steps:generation.step_count})}}</small><small v-if="batchSummary">{{batchSummary}}</small></div></ModuleSection><ModuleSection :label="tr('modules.phonation-synthesis.tasks_and_results')" class="task-results">
 <div class="history-heading"><strong>{{tr('modules.phonation-synthesis.taskHistoryCount',{count:results.length})}}</strong><button :disabled="historyLoading" @click="loadHistory">{{historyLoading?tr('modules.phonation-synthesis.reading'):tr('modules.phonation-synthesis.refresh_task_history')}}</button></div>
 <div class="task-history" :aria-label="tr('modules.phonation-synthesis.phonation_type_task_history')" tabindex="0">
  <ModuleStatus v-if="!jobs.length" kind="empty" :message="tr('modules.phonation-synthesis.no_tasks_yet_analysis_and_generation_are_recorded_separately')"/>
  <article v-for="job in jobs" :key="job.id" class="history-row">
   <button class="history-select" :aria-pressed="selectedJob===job.id" @click="selectTask(job.id)"><strong>{{tr('modules.phonation-synthesis.taskHeading',{id:job.id.slice(0,8),description:taskDescription(taskSnapshots[job.id])})}}</strong><span>{{taskTime(job.created_at)}} · {{taskStates[job.state]}}</span></button>
   <progress v-if="job.state==='running'" :value="job.progress" max="1" :aria-label="job.id.slice(0,8)+tr('modules.phonation-synthesis.progress')"/>
   <small v-if="job.error_code" class="error-text">{{messages[job.error_code]??job.error_code}}</small>
   <button v-if="['queued','running','cancel_requested'].includes(job.state)" @click="cancel(job.id)">{{tr('modules.phonation-synthesis.cancel_this_task')}}</button>
   <button v-if="['failed','cancelled','interrupted'].includes(job.state)" :disabled="busy" @click="retry(job)">{{tr('modules.phonation-synthesis.retry')}} {{job.id.slice(0,8)}}</button>
  </article>
 </div>
 <div v-if="selectedResult" :key="selectedResult.job" class="result-row">
  <strong>{{taskDescription(selectedResult.metadata)}}</strong>
  <small>{{tr('modules.phonation-synthesis.resultSummary',{job:selectedResult.job.slice(0,8),batch:selectedResult.metadata.batch_id?.slice(0,8)??'',index:(selectedResult.metadata.batch_group_index??0)+1,total:selectedResult.metadata.batch_group_count??1})}}</small>
  <div class="result-audios"><button data-audio-selection v-for="f in selectedResult.files.filter(f=>f.name.endsWith('.wav')).sort((a,b)=>a.name.localeCompare(b.name))" :key="f.id" :aria-pressed="selected?.job===selectedResult.job&&playName===f.name" @click="play(selectedResult,f.name)">{{f.name==='combined_steps.wav'?tr('modules.phonation-synthesis.play_whole_group'):f.name.replace('.wav','')}}</button></div>
  <button class="primary" @click="saveResult(selectedResult)">{{context.files.kind==='desktop'?tr('modules.phonation-synthesis.save_complete_group'):tr('modules.phonation-synthesis.download_complete_group')}}</button>
  <button @click="download(selectedResult,'m07.ptb.json')">{{tr('modules.phonation-synthesis.parameter_and_result_manifest')}}</button>
 </div>
 <small v-else-if="selectedJob">{{tr('modules.phonation-synthesis.this_task_has_no_playable_synthesis_group')}}</small>
</ModuleSection></template></ModuleWorkbench></div><SourceAcknowledgement @references="emit('references')"/><ModalDialog v-if="help" :title="tr('modules.phonation-synthesis.phonation_type_continuum_parameter_help')" @close="help=false"><p>{{tr('modules.phonation-synthesis.source_and_target_wav_extract_f0_edit_and_apply')}}</p><p>{{tr('modules.phonation-synthesis.original_audio_f0_determines_lpc_analysis_intervals_residual_f0')}}</p><p>{{tr('modules.phonation-synthesis.normalized_alignment_maps_voiced_duration_to_0_100_onset')}}</p><p>{{tr('modules.phonation-synthesis.period_energy_matching_actually_matches_the_target_residual_period')}}</p><p>{{tr('modules.phonation-synthesis.after_each_group_finishes_computing_all_its_files_are')}}</p><p>{{tr('modules.phonation-synthesis.this_tool_creates_experimental_stimuli_by_manipulating_lpc_residuals')}} <span class="ipa-text">a̤ a̰</span> {{tr('modules.phonation-synthesis.use_the_shared_doulos_sil_font')}}</p></ModalDialog>
</ModuleFrame></template>
<style scoped>
/* Keep plot widths independent of overflow and preview-toolbar wrapping. */
.m07-workspace,.m07-page :deep(.workbench-center){scrollbar-gutter:stable}
.m07-page{overflow:hidden}.m07-page :deep(.workbench-right-body>.module-status){padding:4px 6px}.m07-page :deep(.workbench-right-body>.module-status p){line-height:1.25}.m07-workspace{display:flex;flex-direction:column;flex:1;min-width:0;min-height:0;overflow:auto;overscroll-behavior:contain;scrollbar-width:thin}
.controls{display:flex;flex-wrap:wrap;align-items:end;gap:var(--control-gap);margin-bottom:8px}label{display:grid;gap:4px}label input{width:120px}label select{max-width:100%}.check{display:flex;align-items:center}.check input{width:auto}.continuum-controls{container:m07-controls / inline-size}.parameter-stack{display:grid;gap:3px;font-size:max(12px,calc(var(--control-size)*.93));line-height:1.2}.parameter-stack>label{font-size:inherit;line-height:inherit;display:flex;align-items:center;justify-content:space-between;gap:6px}.parameter-stack>.check{justify-content:flex-start}.parameter-stack>label select{flex:1;max-width:180px;min-width:0}.parameter-stack>label input[type=number]{width:76px}.parameter-stack input,.parameter-stack select,.parameter-stack button{font-size:inherit;min-height:28px;padding:3px 6px}.parameter-stack .check input{padding:0;min-height:16px;width:16px;height:16px;flex:none}.parameter-stack small{color:var(--muted);font-size:var(--support-size)}.generate-actions{display:grid;gap:4px}.generate-actions .cancel-generation{grid-column:1/-1}.f0-axis-controls{display:grid;gap:5px;font-size:var(--support-size)}.axis-inputs{display:grid;grid-template-columns:1fr 1fr;gap:5px}.axis-inputs input{width:100%;min-width:0}.axis-inputs button{grid-column:1/-1;min-height:26px;padding:2px 6px}.f0-axis-controls small{line-height:1.3}.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:4px;border-bottom:1px solid var(--border)}td input{width:100%;min-width:60px}.ipa-text{font-family:var(--font-ipa)}
.m07-plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(2,minmax(290px,1fr));gap:var(--module-gap);flex:1;min-height:600px}.m07-plots>.module-section{display:flex;flex-direction:column;min-height:0;overflow:hidden}.m07-plots :deep(.wave-viewport){flex:1;display:flex;flex-direction:column;min-height:0}.m07-plots :deep(.wave-track){flex:1;min-height:0;grid-template-rows:auto minmax(90px,1fr) auto}.m07-plots :deep(.wave-track svg){height:100%;min-height:90px}.m07-plots :deep(.wave-toolbar){flex:none}.m07-plots :deep(.transport-controls){flex:0 0 auto;min-width:0;margin-top:6px}.m07-plots :deep(.audio-transport){flex-wrap:wrap}.current-audio{color:var(--muted);font-size:var(--support-size);margin-top:6px;overflow-wrap:anywhere}
.history-heading{display:flex;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:5px}.history-heading button{font-size:var(--support-size);min-height:28px;padding:3px 6px}.task-history{max-height:84px;overflow:auto;overscroll-behavior:contain;scrollbar-width:thin;display:grid;gap:3px}.history-row{display:grid;gap:3px}.history-select{display:grid;gap:2px;text-align:left;width:100%;font-size:var(--support-size);line-height:1.2;padding:4px 6px;min-height:0}.history-select strong{font-weight:500}.history-select span,.result-row small{color:var(--muted)}.history-select[aria-pressed=true],.result-audios button[aria-pressed=true]{border-color:var(--accent);background:var(--selected)}.history-row progress{width:100%}.result-row{display:grid;gap:5px;padding-top:6px;margin-top:5px;border-top:1px solid var(--border);font-size:var(--support-size)}.result-row>button{font-size:inherit;min-height:28px;padding:3px 6px}.result-row>strong,.result-row>small{line-height:1.25}.result-audios{display:grid;grid-template-columns:repeat(auto-fit,minmax(max(60px,calc(var(--support-size)*4 + 12px)),1fr));gap:4px;max-height:130px;overflow:auto}.result-audios button{min-width:0;min-height:28px;padding:3px 5px;font-size:inherit}
.m07-page :deep(.workbench-left)>.table-scroll{max-height:none;min-height:180px;flex:1}.m07-page :deep(.workbench-center)>.m07-plots{flex:1;flex-shrink:0}.source-selectors{display:grid;gap:6px}.source-selectors label{display:flex;gap:6px;align-items:center}.source-selectors select{flex:1;min-width:0}.m07-page :deep(.workbench-left) .controls{display:grid;grid-template-columns:1fr 1fr;gap:6px}.m07-page :deep(.workbench-left) .controls label{display:grid;min-width:0;gap:3px}.m07-page :deep(.workbench-left) .controls input,.m07-page :deep(.workbench-left) .controls select{width:100%;min-width:0;max-width:100%}.m07-page :deep(.workbench-left) .controls .check{display:flex}
@container m07-controls (min-width:240px){.generate-actions{grid-template-columns:1fr 1fr}}
@container module (max-width:720px){.m07-plots{grid-template-columns:1fr;grid-template-rows:repeat(4,minmax(280px,auto));min-height:0}.m07-plots>.module-section{min-height:280px}}
</style>
