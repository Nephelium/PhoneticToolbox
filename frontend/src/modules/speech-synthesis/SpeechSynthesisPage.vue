<script setup lang="ts">
import {uiError,uiErrorMessage} from '../module-localization.ts';
import {uiMessage,joinMessages} from '../module-localization.ts';
import {tr,messageRef,type MessageDescriptor} from '../../i18n/index.ts';
import {useWebSettings,settingsFields} from '../../state/web-settings.ts';
import ResearchUpload from '../../components/ResearchUpload.vue';
import AppIcon from '../../components/AppIcon.vue';
import PlotPlaceholder from '../../components/PlotPlaceholder.vue';
import {vTimePrecision} from '../../design/time-precision.ts';
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import {computed,ref,reactive,watch,onMounted,onUnmounted,markRaw} from 'vue';
import type {ResearchContext,ResearchFile,JobView} from '../../platform/research.ts';
import {workspace,host} from '../../state/workspace.ts';import {stop} from '../../state/audio.ts';import {decodeWav} from '../../platform/decode.ts';
import ModuleFrame from '../../components/ModuleFrame.vue';import ModuleToolbar from '../../components/ModuleToolbar.vue';import ModuleSection from '../../components/ModuleSection.vue';import ModuleStatus from '../../components/ModuleStatus.vue';import ModalDialog from '../../components/ModalDialog.vue';import WaveformViewport from '../../components/WaveformViewport.vue';import AudioTransport from '../../components/AudioTransport.vue';import TaskPanel from '../../components/TaskPanel.vue';
import CurveEditor from './CurveEditor.vue';import SynthesisSpectrum from './SynthesisSpectrum.vue';
import {defaults,parameters,vowels,presets,presetLabel,presetMessage,f0Methods,clone,valid,number,ipa,override,resize,f0Range,preset,exportParams,importParams,isCurrent,configPreferences,restorePreferences,preferenceKeys,unavailableParameters,type Config} from './state.ts';
import {copyText} from '../../platform/clipboard.ts';
import type {Action,Result,Raster} from './port.ts';import {downloadBytes} from '../../platform/m06.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active:boolean}>();const emit=defineEmits<{references:[]}>();
const wave=workspace(props.stateKey),config=reactive(defaults()),name=ref('F0'),overrideText=ref(''),durationText=ref('2'),low=ref('50'),high=ref('500');
const error=messageRef(''),notice=messageRef(''),busy=ref(false),revision=ref(0),resultRevision=ref(-1),result=ref<Result>(),job=ref<JobView>();
const help=ref(false),rules=ref(false),pending=ref<(()=>void)|null>(null),pendingLabel=messageRef(''),selectedPreset=ref('常态浊声');
const files=ref<ResearchFile[]>([]),source=ref<ResearchFile>(),directory=ref(''),audioPicker=ref<HTMLInputElement>(),paramPicker=ref<HTMLInputElement>();
const view=ref('wave'),windowMs=ref(20);let epoch=0,disposed=false,abort:AbortController|undefined;
const sourceWave=workspace(props.stateKey+'.m06-source'),previewTarget=ref('result'),sourceSpectra=ref<Record<string,Raster>>({});
const previewWave=computed(()=>previewTarget.value==='source'?sourceWave:wave),spectra=computed(()=>previewTarget.value==='source'?sourceSpectra.value:result.value?.metadata.spectrograms);
const port=computed(()=>props.context.files.m06),stale=computed(()=>!!result.value&&resultRevision.value!==revision.value);
const timelineDuration=computed(()=>previewTarget.value==='source'&&sourceWave.asset?sourceWave.asset.duration:config.duration);
const activeRange=computed<[number,number]>(()=>{const p=previewWave.value,length=timelineDuration.value/p.zoom,start=Math.max(0,Math.min(p.offset,timelineDuration.value-length));return [start,start+length];});
const preferences=reactive(configPreferences(config));
const retryContext=ref<{id:string;revision:number;action:Action;input?:ResearchFile}>();
const canRetrySave=computed(()=>!!job.value&&!!port.value?.canRetrySave?.(job.value.id));
const tasks=computed(()=>job.value?[{id:job.value.id,title:tr('modules.speech-synthesis.acoustic_parameter_synthesis')+job.value.id.slice(0,8),status:job.value.state,progress:job.value.progress,error:job.value.error_code??undefined,canCancel:true}]:[]);
watch(config,()=>{revision.value++;wave.dirty=true;stop();},{deep:true,flush:'sync'});
watch(config,()=>Object.assign(preferences,configPreferences(config)),{deep:true,flush:'sync'});
watch(()=>props.active,v=>{if(!v)stop();});
watch(name,()=>syncOverride());
watch(source,()=>editPending(),{flush:'sync'});
function editPending(){revision.value++;wave.dirty=true;stop();}
function requireApplied(){if(Number(durationText.value)!==config.duration||Number(low.value)!==config.f0_range[0]||Number(high.value)!==config.f0_range[1])throw uiError(uiMessage('modules.speech-synthesis.apply_duration_or_f0_range_first'));const v=config.curves[name.value].override;const expected=v===null?'':String(v*(name.value==='Shimmer'?100:1));if(overrideText.value!==expected)throw uiError(uiMessage('modules.speech-synthesis.apply_the_curve_override_first'));}
const copyNotice=messageRef(''),copyError=messageRef(''),copyValue=ref('');
async function copySymbol(s:string){copyValue.value=s;copyNotice.value='';copyError.value='';try{await copyText(s);copyNotice.value=joinMessages(uiMessage('modules.speech-synthesis.copied'),s);}catch{copyError.value=uiMessage('modules.speech-synthesis.copy_failed_select_the_ipa_symbol_and_press_ctrl');}}
function appliedOverride(){const v=config.curves[name.value].override;return v===null?'':String(v*(name.value==='Shimmer'?100:1));}
function syncOverride(){overrideText.value=appliedOverride();}
function selectParameter(next:string){void attempt(()=>{if(overrideText.value!==appliedOverride())override(config,name.value,overrideText.value);name.value=next;syncOverride();});}
function syncInputs(){syncOverride();durationText.value=String(config.duration);low.value=String(config.f0_range[0]);high.value=String(config.f0_range[1]);}
function attempt(work:()=>void|Promise<void>){error.value='';return Promise.resolve().then(work).catch(e=>{if(!disposed)error.value=uiErrorMessage(e);});}
function replace(c:Config){Object.assign(config,valid(c));syncInputs();resetRange();}
function resetRange(){wave.zoom=1;wave.offset=0;sourceWave.zoom=1;sourceWave.offset=0;}
function setRange(a:number,b:number){if(b<=a)return;const p=previewWave.value;p.zoom=timelineDuration.value/(b-a);p.offset=a;}
function applyOverride(){return attempt(()=>{override(config,name.value,overrideText.value);syncOverride();});}
function applyDuration(){return attempt(()=>{resize(config,number(durationText.value,uiMessage('modules.speech-synthesis.duration')));durationText.value=String(config.duration);resetRange();});}
function applyRange(){return attempt(()=>{f0Range(config,number(low.value,uiMessage('modules.speech-synthesis.minimum_f0')),number(high.value,uiMessage('modules.speech-synthesis.maximum_f0')));low.value=String(config.f0_range[0]);high.value=String(config.f0_range[1]);});}
function confirm(label:string|MessageDescriptor,work:()=>void){pendingLabel.value=label;pending.value=work;}
function commitPending(){const work=pending.value;pending.value=null;if(work)void attempt(work);}
function clear(){confirm(uiMessage('modules.speech-synthesis.restore_all_parameters_and_clear_curve_overrides_and_silent'),()=>{const c=defaults();resize(c,config.duration);c.sample_rate=config.sample_rate;c.sequence=config.sequence;c.fade_in=config.fade_in;c.fade_out=config.fade_out;c.smooth=config.smooth;replace(c);});}
function applyPreset(){confirm(uiMessage('modules.speech-synthesis.confirmApplyPhonation',{preset:presetMessage(selectedPreset.value)}),()=>{requireApplied();preset(config,selectedPreset.value);syncInputs();const shift=config.f0_transform.offset_hz;notice.value=shift===0?uiMessage('modules.speech-synthesis.phonation_preset_applied_f0_uses_the_base_curve'):uiMessage(shift>0?'modules.speech-synthesis.phonationShiftUp':'modules.speech-synthesis.phonationShiftDown',{hz:Math.abs(shift).toFixed(1)});});}
 async function save(){try{if(busy.value)throw uiError(uiMessage('modules.speech-synthesis.wait_for_the_current_task_to_finish_or_cancel'));if(canRetrySave.value){await retrySave();if(canRetrySave.value)return false;}valid(config);if(Number(durationText.value)!==config.duration||Number(low.value)!==config.f0_range[0]||Number(high.value)!==config.f0_range[1])throw uiError(uiMessage('modules.speech-synthesis.apply_duration_or_f0_range_first'));const c=clone(config);if(overrideText.value!==appliedOverride())override(c,name.value,overrideText.value);valid(c);if(!host.projects.write('m06.draft.'+props.stateKey,c))throw uiError(uiMessage('modules.speech-synthesis.failed_to_save_draft'));if(JSON.stringify(c)!==JSON.stringify(config))replace(c);wave.dirty=false;notice.value=uiMessage('modules.speech-synthesis.complete_parameter_draft_saved');return true;}catch(e){error.value=uiErrorMessage(e);return false;}}
function restorePendingSave(){const saved=port.value?.pendingSave?.();if(!saved)return;job.value=saved.job;retryContext.value={id:saved.job.id,revision:JSON.stringify(saved.config)===JSON.stringify(config)?revision.value:-1,action:saved.action};wave.dirty=true;notice.value=uiMessage('modules.speech-synthesis.the_previous_synthesis_result_is_unsaved_retry_saving_without');}
function discardPendingSave(){if(busy.value)throw uiError(uiMessage('modules.speech-synthesis.cancel_the_current_task_before_discarding_results_awaiting_save'));if(job.value)port.value?.discardSave?.(job.value.id);retryContext.value=undefined;}
defineExpose({save,hasPendingSave:()=>canRetrySave.value,discardPendingSave});
async function refresh(){const next=(await props.context.files.list(directory.value||undefined)).filter(f=>/\.wav$/i.test(f.name));files.value=next;if(sourceId.value&&!next.some(f=>f.id===sourceId.value&&(!source.value||f.sha256===source.value.sha256)))await selectSource('');}
async function choose(){await attempt(async()=>{const grant=await props.context.files.choose?.('input');if(grant){directory.value=grant.id;await refresh();}});}
async function addAudio(event:Event){await attempt(async()=>{const target=event.target as HTMLInputElement;props.context.files.add?.([...target.files??[]]);target.value='';await refresh();});}
const sourceId=ref(''),sourceLoading=ref(false);let loadEpoch=0,loadAbort:AbortController|undefined;
async function loadSource(event:Event){await selectSource((event.target as HTMLSelectElement).value);}
async function selectSource(id:string){
 const ticket=++loadEpoch;loadAbort?.abort();loadAbort=new AbortController();sourceId.value=id;
 source.value=undefined;sourceWave.asset=null;sourceSpectra.value={};sourceLoading.value=!!id;error.value='';notice.value='';stop();
 const f=files.value.find(f=>f.id===id);if(!f){sourceLoading.value=false;previewTarget.value='result';return;}
 try{const raw=await props.context.files.read(f,loadAbort.signal);if(f.sha256&&raw.sha256!==f.sha256)throw uiError(uiMessage('modules.speech-synthesis.audio_changed_refresh_files_and_select_it_again'));const asset=await decodeWav(raw.buffer,f.name,loadAbort.signal);
  if(disposed||ticket!==loadEpoch)return;
  source.value=f;sourceWave.asset=markRaw(asset);sourceWave.start=0;sourceWave.end=asset.duration;sourceWave.channel=0;sourceWave.zoom=1;sourceWave.offset=0;
  previewTarget.value='source';view.value='wave';notice.value=uiMessage('modules.speech-synthesis.audio_loaded_click_extract_parameters_to_begin_extraction');
 }catch(e){if(!disposed&&ticket===loadEpoch){sourceId.value='';error.value=uiErrorMessage(e);}}
 finally{if(ticket===loadEpoch)sourceLoading.value=false;}
}
async function run(action:Action){await attempt(async()=>{
 if(!port.value)throw uiError(uiMessage('modules.speech-synthesis.acoustic_parameter_synthesis_is_unavailable_here_use_the_desktop'));if(busy.value)throw uiError(uiMessage('modules.speech-synthesis.wait_for_the_current_task_to_finish_or_cancel'));
 if(canRetrySave.value)throw uiError(uiMessage('modules.speech-synthesis.synthesis_results_await_saving_retry_saving_or_confirm_discard'));
 requireApplied();const submitted=clone(valid(config));if(action==='generate')ipa(submitted.sequence);if(action==='extract'&&(!source.value||sourceLoading.value))throw uiError(uiMessage('modules.speech-synthesis.load_wav_audio_first'));
 if(submitted.duration>10||submitted.duration*submitted.sample_rate>480000)throw uiError(uiMessage('modules.speech-synthesis.the_validated_task_budget_is_10_seconds_and_480'));
 const input=source.value;const rev=revision.value,ticket=++epoch;abort=new AbortController();busy.value=true;notice.value='';retryContext.value=undefined;
 try{const completed=await port.value.run(action,submitted,input,abort.signal,j=>{if(!disposed&&ticket===epoch)job.value=j;});
  await applyCompleted(action,completed,input,rev,ticket);
 }finally{if(ticket===epoch){busy.value=false;if(job.value&&port.value?.canRetrySave?.(job.value.id))retryContext.value={id:job.value.id,revision:rev,action,input};}}
 });}
async function applyCompleted(action:Action,completed:Result,input:ResearchFile|undefined,rev:number,ticket:number){
  abort?.signal.throwIfAborted();
  if(disposed||!isCurrent(rev,revision.value,ticket,epoch)){notice.value=uiMessage('modules.speech-synthesis.configuration_changed_while_the_task_finished_late_results_were');return;}
  if(action==='synthesize'){
   const asset=await decodeWav(completed.wav!,'合成结果.wav',abort?.signal);if(disposed||!isCurrent(rev,revision.value,ticket,epoch))return;
   previewTarget.value='result';result.value=completed;resultRevision.value=rev;wave.asset=markRaw(asset);wave.start=0;wave.end=asset.duration;wave.channel=0;resetRange();notice.value=joinMessages(joinMessages(uiMessage('modules.speech-synthesis.synthesis_complete_playback_and_export_use_this_parameter_snapshot'),((completed.metadata.diagnostics?.output_gain??1)<1?joinMessages(joinMessages(uiMessage('modules.speech-synthesis.output_exceeded_safe_amplitude_and_was_attenuated_by'),(-20*Math.log10(completed.metadata.diagnostics!.output_gain!)).toFixed(1)),' dB'):'')),(completed.metadata.diagnostics?.voiced_frames===0?uiMessage('modules.speech-synthesis.the_selected_algorithm_found_no_voiced_frames_listen_and'):''));
  }else{if(action==='extract'&&input){const raw=await props.context.files.read(input,abort?.signal);if(raw.sha256!==completed.metadata.input_sha256)throw uiError(uiMessage('modules.speech-synthesis.source_audio_changed_after_extraction_parameters_were_not_applied'));const asset=await decodeWav(raw.buffer,input.name,abort?.signal);if(disposed||!isCurrent(rev,revision.value,ticket,epoch))return;sourceWave.asset=markRaw(asset);sourceWave.start=0;sourceWave.end=asset.duration;sourceWave.channel=0;sourceSpectra.value=completed.metadata.spectrograms;previewTarget.value='source';}replace(completed.metadata.config);notice.value=action==='generate'?uiMessage('modules.speech-synthesis.vowel_curves_generated_click_synthesize_audio'):joinMessages(joinMessages(joinMessages(uiMessage('modules.speech-synthesis.parameter_extraction_complete'),f0Methods[completed.metadata.config.f0_method]),uiMessage('modules.speech-synthesis.av_was_initialized_from_relative_energy_ah_is_0')),(completed.metadata.diagnostics?.voiced_mask?.some(Boolean)===false?uiMessage('modules.speech-synthesis.no_voiced_frames_detected_av_is_0_throughout_adjust'):''));}
}
async function retrySave(){await attempt(async()=>{
 const retry=retryContext.value;if(!retry||!port.value?.retrySave||busy.value)throw uiError(uiMessage('modules.speech-synthesis.no_synthesis_result_awaits_saving'));
 const ticket=++epoch;abort=new AbortController();busy.value=true;notice.value=uiMessage('modules.speech-synthesis.retrying_save_without_recomputing');
 try{const completed=await port.value.retrySave(retry.id,abort.signal,j=>{if(!disposed&&ticket===epoch)job.value=j;});await applyCompleted(retry.action,completed,retry.input,retry.revision,ticket);}
 finally{if(ticket===epoch){busy.value=false;if(!port.value?.canRetrySave?.(retry.id))retryContext.value=undefined;}}
 });}
async function cancel(){abort?.abort();if(job.value&&port.value)await attempt(async()=>{await port.value!.cancel(job.value!.id);});}
async function exportAudio(){await attempt(async()=>{if(!result.value)throw uiError(uiMessage('modules.speech-synthesis.no_synthesis_results_yet'));await port.value!.download(result.value,'synthesis.wav');notice.value=stale.value?uiMessage('modules.speech-synthesis.exported_the_previous_result_and_its_actual_synthesis_snapshot'):uiMessage('modules.speech-synthesis.synthesis_result_exported');});}
function exportCurrent(){void attempt(()=>{requireApplied();downloadBytes(exportParams(config),'声学参数合成参数.csv');});}
async function importFile(event:Event){await attempt(async()=>{const target=event.target as HTMLInputElement,file=target.files?.[0];target.value='';if(!file)return;if(file.size>8_000_000)throw uiError(uiMessage('modules.speech-synthesis.parameter_file_exceeds_8_mb'));const c=importParams(await file.text());confirm(uiMessage('modules.speech-synthesis.importing_parameters_will_replace_the_complete_current_configuration'),()=>replace(c));});}
onMounted(()=>{const draft=host.projects.read<Config|null>('m06.draft.'+props.stateKey,null);if(draft){try{replace(draft);wave.dirty=false;}catch(e){error.value=joinMessages((e instanceof Error?uiErrorMessage(e):uiMessage('modules.speech-synthesis.previous_draft_is_invalid')),uiMessage('modules.speech-synthesis.stored_content_was_retained_and_default_parameters_are_in'));}}void attempt(refresh);});
onUnmounted(()=>{loadEpoch++;loadAbort?.abort();sourceWave.asset=null;disposed=true;epoch++;abort?.abort();stop();});
useWebSettings(props.stateKey,{...settingsFields(preferences,[...preferenceKeys,'curveOverrides']),selectedPreset,windowMs});
onMounted(()=>{if(props.stateKey.startsWith('server:')){try{replace(restorePreferences(config,preferences));wave.dirty=false;}catch(e){Object.assign(preferences,configPreferences(config));syncInputs();error.value=joinMessages(uiMessage('modules.speech-synthesis.settings_could_not_be_restored_current_parameters_retained'),uiErrorMessage(e));}}restorePendingSave();});
</script>
<template>
<ModuleFrame unified fit :label="tr('modules.speech-synthesis.acoustic_parameter_synthesis_workspace')" class="m06-page" :aria-busy="busy">
 <template #toolbar><ModuleToolbar><ResearchUpload :files="context.files" accept=".wav,.mp3,.flac" @uploaded="refresh"/>
  <button class="primary" v-if="context.files.choose" @click="choose"><AppIcon name="folder"/>{{tr('modules.speech-synthesis.open_audio_folder')}}</button><button class="primary" v-else-if="context.files.add" @click="audioPicker?.click()">{{tr('modules.speech-synthesis.load_audio')}}</button><button @click="attempt(refresh)">{{tr('modules.speech-synthesis.refresh_files')}}</button>
  <input ref="audioPicker" hidden type="file" accept=".wav" @change="addAudio"/>
  <label class="preset-picker">{{tr('modules.speech-synthesis.f0_algorithm')}} <select v-model="config.f0_method" :aria-label="tr('modules.speech-synthesis.f0_extraction_algorithm')" :title="tr('modules.speech-synthesis.used_for_the_next_extraction_with_applied_f0_bounds')"><option v-for="(label,key) in f0Methods" :key="key" :value="key">{{label}}</option></select></label>
  <button class="primary" :disabled="busy||sourceLoading||!source||!port" @click="run('extract')">{{tr('modules.speech-synthesis.extract_parameters')}}</button><button @click="paramPicker?.click()">{{tr('modules.speech-synthesis.import_parameters')}}</button>
  <input ref="paramPicker" hidden type="file" accept=".csv,.json" @change="importFile"/><button @click="exportCurrent">{{tr('modules.speech-synthesis.export_parameters')}}</button>
  <label class="preset-picker">{{tr('modules.speech-synthesis.phonation_type')}} <select v-model="selectedPreset" :aria-label="tr('modules.speech-synthesis.phonation_type_preset')"><option v-for="p in Object.keys(presets)" :key="p" :value="p">{{presetLabel(p)}}</option></select></label><button @click="applyPreset">{{tr('modules.speech-synthesis.apply_preset')}}</button>
  <template #actions><button @click="save">{{tr('modules.speech-synthesis.save_parameter_draft')}}</button><button @click="help=true">{{tr('modules.speech-synthesis.help')}}</button><button @click="emit('references')"><AppIcon name="book"/>{{tr('modules.speech-synthesis.methods_and_references')}}</button></template>
 </ModuleToolbar></template>
 <ModuleWorkbench unified :state-key="stateKey" :left-label="tr('modules.speech-synthesis.synthesis_parameters')" :center-label="tr('modules.speech-synthesis.parameter_curves_and_audio')" :right-label="tr('modules.speech-synthesis.synthesis_and_tasks')">
  <template #left>
   <ModuleSection :label="tr('modules.speech-synthesis.audio_files')" class="source-section"><label>{{tr('modules.speech-synthesis.audio_files')}}<select class="source-picker" :aria-label="tr('modules.speech-synthesis.audio_files')" :value="sourceId" @change="loadSource"><option value="">{{tr('modules.speech-synthesis.select_an_audio_file')}}</option><option v-for="f in files" :key="f.id" :value="f.id">{{f.name}}</option></select></label><p v-if="sourceLoading" role="status">{{tr('modules.speech-synthesis.loading_audio')}}</p></ModuleSection>
   <ModuleSection :label="tr('modules.speech-synthesis.basic_synthesis_settings')" class="base-settings"><div class="controls">
    <label>{{tr('modules.speech-synthesis.total_duration_s')}}<input v-time-precision="'s'" v-model="durationText" :aria-label="tr('modules.speech-synthesis.total_duration')" @input="editPending"/></label><button @click="applyDuration">{{tr('modules.speech-synthesis.apply_duration')}}</button>
    <label>{{tr('modules.speech-synthesis.fade_in_ms')}}<input v-time-precision="'ms'" v-model.number="config.fade_in" type="number" min="0" max="1000"/></label><label>{{tr('modules.speech-synthesis.fade_out_ms')}}<input v-time-precision="'ms'" v-model.number="config.fade_out" type="number" min="0" max="1000"/></label>
    <label>{{tr('modules.speech-synthesis.smoothing_points')}}<input v-model.number="config.smooth" type="number" min="1" max="50"/></label>
    <label>{{tr('modules.speech-synthesis.minimum_f0_hz')}}<input v-model="low" :aria-label="tr('modules.speech-synthesis.minimum_f0')" @input="editPending"/></label><label>{{tr('modules.speech-synthesis.maximum_f0_hz')}}<input v-model="high" :aria-label="tr('modules.speech-synthesis.maximum_f0')" @input="editPending"/></label><button @click="applyRange">{{tr('modules.speech-synthesis.apply_f0_range')}}</button>
   </div></ModuleSection>
   <ModuleSection :label="tr('modules.speech-synthesis.parameter_list')" class="parameters-section"><div class="parameter-list"><button v-for="(definition,key) in parameters" :key="key" :disabled="unavailableParameters.has(String(key))" :title="unavailableParameters.has(String(key))?tr('modules.speech-synthesis.unused_by_current_cascade_synthesis_imported_values_are_retained'):undefined" :aria-pressed="name===key" @click="selectParameter(String(key))">{{key}} <small>{{unavailableParameters.has(String(key))?tr('modules.speech-synthesis.disabled'):definition[3]}}</small></button></div></ModuleSection>
  </template>
  <ModuleSection :label="tr('modules.speech-synthesis.parameter_curve_92')" class="curve-section">
   <div class="controls curve-controls"><label>{{tr('modules.speech-synthesis.curve_override')}}<input v-model="overrideText" :aria-label="tr('modules.speech-synthesis.curve_override')" :placeholder="tr('modules.speech-synthesis.enter_a_value_or_segmented_sequence')" :title="tr('modules.speech-synthesis.a_single_value_fixes_the_entire_duration_commas_join')" @input="editPending"/></label><button @click="applyOverride">{{tr('modules.speech-synthesis.apply_override')}}</button><button @click="overrideText='';applyOverride()">{{tr('modules.speech-synthesis.clear_override')}}</button><button @click="resetRange">{{tr('modules.speech-synthesis.reset_range')}}</button><button @click="clear">{{tr('modules.speech-synthesis.reset_parameters')}}</button></div>
    <p v-if="name==='HNR'" class="parameter-hint">{{tr('modules.speech-synthesis.hnr_editing_range_is_0_60_db_old_values')}}</p>
    <CurveEditor :config="config" :name="name" :start="activeRange[0]" :end="activeRange[1]" @range="setRange" @error="error=$event"/>
  </ModuleSection>
  <ModuleSection :label="tr('modules.speech-synthesis.audio_preview')" class="preview-section">
   <div class="controls preview-controls"><button :aria-pressed="previewTarget==='result'" @click="previewTarget='result';stop()">{{tr('modules.speech-synthesis.synthesis_result')}}</button><button v-if="sourceWave.asset" :aria-pressed="previewTarget==='source'" @click="previewTarget='source';stop()">{{tr('modules.speech-synthesis.source_audio')}}</button><button :aria-pressed="view==='wave'" @click="view='wave'">{{tr('modules.speech-synthesis.waveform')}}</button><button :aria-pressed="view==='spectrum'" @click="view='spectrum'">{{tr('modules.speech-synthesis.spectrogram')}}</button><label v-if="view==='spectrum'">{{tr('modules.speech-synthesis.window_length')}}<select v-model.number="windowMs" :aria-label="tr('modules.speech-synthesis.spectrogram_window_length')"><option v-for="ms in [5,10,20,40]" :key="ms" :value="ms">{{ms}} ms</option></select></label></div>
   <WaveformViewport v-if="previewWave.asset&&view==='wave'" :state="previewWave" :timeline-duration="timelineDuration" class="synthesis-wave" compact-overview hide-overview-controls auto-amplitude continuous-detail>
    <template #annotations="{x,start,end}"><path v-for="t in config.boundaries.filter(t=>t>=start&&t<=end)" :key="t" class="vowel-boundary" :d="`M${x(t)},0V90`"/></template>
   </WaveformViewport>
   <SynthesisSpectrum v-else-if="spectra?.[String(windowMs)]&&view==='spectrum'" :data="spectra[String(windowMs)]" :start="activeRange[0]" :end="activeRange[1]" :window-ms="windowMs" :boundaries="config.boundaries"/>
   <PlotPlaceholder v-else :message="previewTarget==='source'&&sourceWave.asset?tr('modules.speech-synthesis.source_audio_loaded_waveform_playback_is_available_extract_parameters'):tr('modules.speech-synthesis.no_synthesized_audio_yet_after_generating_vowels_click_synthesize')"/>
  </ModuleSection>
  <template #right>
   <ModuleSection :label="tr('modules.speech-synthesis.synthesis_actions')"><div class="controls synthesis-actions">
    <label class="ipa-input">{{tr('modules.speech-synthesis.ipa_vowel_sequence')}}<input v-model="config.sequence" class="ipa-text" :aria-label="tr('modules.speech-synthesis.ipa_vowel_sequence')" placeholder="a-i/- ///u+/e++o/-"/></label>
    <button @click="rules=true">{{tr('modules.speech-synthesis.vowel_rules')}}</button><button class="primary" :disabled="busy||!port" @click="run('generate')">{{tr('modules.speech-synthesis.generate_vowels')}}</button>
    <button class="primary" :disabled="busy||!port" @click="run('synthesize')">{{tr('modules.speech-synthesis.synthesize_audio')}}</button><button class="primary" :disabled="!result" @click="exportAudio">{{tr('modules.speech-synthesis.export_audio')}}</button><button v-if="canRetrySave" :disabled="busy" @click="retrySave">{{tr('modules.speech-synthesis.retry_saving_result')}}</button><button v-if="busy" @click="cancel">{{tr('modules.speech-synthesis.cancel_task')}}</button>
   </div></ModuleSection>
   <ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-if="stale" kind="info" :message="tr('modules.speech-synthesis.parameters_changed_resynthesis_is_required_playback_and_audio_export')"/><ModuleStatus v-if="notice" kind="info" :message="notice"/><ModuleStatus v-if="!port" kind="info" :message="tr('modules.speech-synthesis.parameter_editing_is_available_use_the_desktop_app_for')"/>
    <details :open="busy||canRetrySave" class="task-details"><summary>{{tr('modules.speech-synthesis.synthesis_task_status')}}</summary><span v-if="result">{{tr('modules.speech-synthesis.result')}} {{result.job.slice(0,8)}} {{tr('modules.speech-synthesis.random_seed')}} {{result.metadata.seed}}</span><TaskPanel :tasks="tasks" :empty-title="tr('modules.speech-synthesis.no_synthesis_task_submitted')" :empty-text="tr('modules.speech-synthesis.editing_parameters_does_not_trigger_audio_synthesis_automatically')" @cancel="cancel"/></details>
  </template>
 </ModuleWorkbench>
 <div class="m06-transport global-transport"><AudioTransport :state="previewWave" :active="active"/></div>
 <ModalDialog v-if="pending" :title="tr('modules.speech-synthesis.confirm_parameter_replacement')" @close="pending=null"><p>{{pendingLabel}}</p><p>{{tr('modules.speech-synthesis.existing_synthesized_audio_is_retained_and_marked_as_an')}}</p><template #footer><button @click="pending=null">{{tr('modules.speech-synthesis.cancel')}}</button><button @click="commitPending">{{tr('modules.speech-synthesis.apply_and_replace')}}</button></template></ModalDialog>
 <ModalDialog v-if="rules" :title="tr('modules.speech-synthesis.vowel_rules')" @close="rules=false"><p class="copy-feedback" role="status">{{copyNotice}}</p><div v-if="copyError" class="copy-fallback"><p role="alert">{{copyError}}</p><input :value="copyValue" readonly :aria-label="tr('modules.speech-synthesis.ipa_symbol_to_copy')" @focus="($event.target as HTMLInputElement).select()"/></div><table class="vowel-rules"><thead><tr><th>IPA</th><th>F1 Hz</th><th>F2 Hz</th><th>F3 Hz</th></tr></thead><tbody><tr v-for="(f,s) in vowels" :key="s"><td><button class="ipa-text" @click="copySymbol(s)">{{s}}</button></td><td v-for="(v,i) in f" :key="i">{{v}}</td></tr></tbody></table><p>{{tr('modules.speech-synthesis.click_an_ipa_symbol_to_copy_it_spaces_are')}}</p></ModalDialog>
 <ModalDialog v-if="help" :title="tr('modules.speech-synthesis.acoustic_parameter_synthesis_help')" @close="help=false"><p>{{tr('modules.speech-synthesis.set_and_apply_duration_enter_ipa_and_generate_vowel')}}</p><p>{{tr('modules.speech-synthesis.shift_drag_to_draw_ctrl_drag_to_restore_defaults')}}</p><p>{{tr('modules.speech-synthesis.applying_duration_immediately_updates_both_plots_time_axes_old')}}</p><p>{{tr('modules.speech-synthesis.export_parameters_saves_the_current_configuration_export_audio_saves')}}</p><p>{{tr('modules.speech-synthesis.select_an_f0_extraction_algorithm_in_the_toolbar_before')}}</p><p>{{tr('modules.speech-synthesis.the_bottom_time_selection_playback_progress_and_volume_follow')}}</p><p>{{tr('modules.speech-synthesis.selecting_wav_on_the_left_loads_a_preview_only')}}</p></ModalDialog>
</ModuleFrame>
</template>
<style scoped>
.controls{display:flex;align-items:end;gap:var(--control-gap);flex-wrap:wrap}.controls label{display:grid;gap:4px;min-width:0}.controls input{width:100px;min-width:0}.preset-picker{display:flex;align-items:center;gap:6px;white-space:nowrap}.source-section{flex:none}.source-section label{display:grid;gap:6px}.source-picker{width:100%;min-width:0}.source-section p{margin:4px 0 0}.curve-controls{flex:none}.curve-controls label{display:flex;align-items:center;gap:6px}.curve-controls input{width:160px}.base-settings .controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));align-items:end}.base-settings input{width:100%}.base-settings button{padding-inline:4px}.parameter-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:3px}.parameter-list button{min-width:0;min-height:28px;padding:3px 5px;white-space:nowrap;text-align:center}.parameter-list button[aria-pressed=true]{color:var(--accent);background:var(--selected)}.parameter-list small{margin-left:4px;color:var(--muted)}.synthesis-actions{display:grid;grid-template-columns:minmax(0,1fr)}.ipa-input input{width:100%}.ipa-text{font-family:var(--font-ipa)}.curve-section,.preview-section{display:flex;flex-direction:column;gap:8px;min-height:0}.preview-section{overflow:hidden}.preview-controls{flex:none}.task-details{overflow-wrap:anywhere}.synthesis-wave{border:1px solid var(--border);border-radius:8px;--wave-axis-width:72px;padding-right:24px;overflow:hidden}.vowel-boundary{stroke:var(--warning);stroke-dasharray:4 4;vector-effect:non-scaling-stroke;pointer-events:none}table{width:100%;border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid var(--border)}
.m06-page :deep(.workbench-center){display:grid;grid-template-rows:minmax(250px,1.1fr) minmax(220px,1fr);overflow:auto}
.preview-controls{align-items:center}.preview-controls label{display:flex;align-items:center;gap:6px;white-space:nowrap}.m06-page{overflow:hidden}.m06-page :deep(.module-workbench){flex:1;min-height:0;overflow:auto}.m06-transport{flex:none}
.parameter-list{min-height:0}.m06-page :deep(.workbench-left .parameter-list button){white-space:nowrap;overflow-wrap:normal;gap:3px}
.vowel-rules td,.vowel-rules th{padding:2px 8px;line-height:1.25}.vowel-rules button{min-height:24px;padding:1px 8px;line-height:1.25;min-width:28px;box-shadow:none}.copy-feedback{min-height:1.25em;margin:6px 0;color:var(--accent)}
.m06-page :deep(.curve-editor){display:flex;flex-direction:column;flex:1;min-height:0}.m06-page :deep(.curve-editor svg){flex:1;height:0;min-height:100px}.m06-page :deep(.curve-editor .track-label){flex:none;font-size:var(--figure-size)}
.m06-page :deep(.synthesis-wave),.m06-page :deep(.synthesis-spectrum){flex:1;min-height:0}.m06-page :deep(.synthesis-wave){display:flex;flex-direction:column}.m06-page :deep(.wave-track){flex:1;min-height:0;grid-template-rows:auto minmax(60px,1fr) auto}.m06-page :deep(.wave-track svg){height:100%;min-height:60px}.m06-page :deep(.pan-label){margin:0}.m06-page :deep(.transport-compact){flex:initial;min-width:0}
@container module (max-width:1060px){.m06-page :deep(.workbench-center){grid-template-rows:340px 300px}.m06-page :deep(.module-workbench){grid-auto-rows:max-content;align-content:start}}
@container module (min-width:1061px){.parameters-section{flex:1;min-height:0;display:flex;flex-direction:column}.parameter-list{flex:1;grid-template-rows:repeat(12,minmax(24px,1fr))}.m06-page :deep(.workbench-left){gap:8px}.base-settings,.parameters-section{padding:8px}}
</style>
