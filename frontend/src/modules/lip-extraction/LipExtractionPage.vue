<script setup lang="ts">
import {uiError,uiErrorMessage} from '../module-localization.ts';
import {uiMessage,joinMessages} from '../module-localization.ts';
import {tr,messageRef,messageText,type MessageDescriptor} from '../../i18n/index.ts';
import {useMediaAudition} from '../../state/media-audition.ts';
import {useWebSettings,settingsFields} from '../../state/web-settings.ts';
import AppIcon from '../../components/AppIcon.vue';
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import {computed,onMounted,onUnmounted,ref,shallowRef,watch,nextTick,markRaw} from 'vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleSection from '../../components/ModuleSection.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import {LipCapture,type CaptureSettings} from './capture.ts';
import {drawLegacyOverlay,faceBounds,fitFace} from './overlay.ts';
import {isLoopback} from './audio-input.ts';
import {audioDevicesHidden,readCaptureDevices,deviceAccessErrorMessage} from './devices.ts';
import type {CaptureMode} from './state.ts';
import type {LipPort,LipResult,LipRow} from './port.ts';
import {unavailableLipPort} from '../../platform/m05.ts';
import resources from '../../../../resources/m05/resources.json';
import {LipReplay} from './replay.ts';
import ModalDialog from '../../components/ModalDialog.vue';
import OffsetDialog from './OffsetDialog.vue';
import {waveformEnvelope,type AlignmentWaveform} from './alignment.ts';
import {claimCapture,releaseCapture} from '../../platform/capture-lease.ts';
const props=defineProps<{stateKey?:string;active?:boolean;port?:LipPort}>();
const emit=defineEmits<{references:[];dirty:[boolean]}>();
const video=ref<HTMLVideoElement|null>(null),canvas=ref<HTMLCanvasElement|null>(null),revision=ref(0);
const devices=ref<MediaDeviceInfo[]>([]),camera=ref(''),microphone=ref(''),mode=ref<CaptureMode|'offline'>('preview'),filter=ref(true),cutoff=ref(15),fps=ref(30),delegate=ref<'CPU'|'GPU'>('CPU'),mirror=ref(true);
const deviceBusy=ref(false),audioNeedsPermission=computed(()=>audioDevicesHidden(devices.value));
let deviceAbort:AbortController|undefined;
const error=messageRef(''),notice=messageRef(''),help=ref(false),files=ref<File[]>([]),results=ref<LipResult[]>([]),selection=ref(0),offset=ref(0),busy=ref(false),progress=messageRef(''),replaying=ref(false),replayIndex=ref(0),quality=ref<'high'|'standard'|'small'>('high');
const fileInput=ref<HTMLInputElement|null>(null),batchIds=ref<string[]>([]),saveAll=ref(true);
const localSaved=ref(false),confirmDiscard=ref(false),resultDirty=ref(false);
const saveDialog=ref(false),saveVideo=ref(true),saveAnimation=ref(true),offsetDialog=ref(false),offsetPreview=ref<number|null>(null),alignmentError=messageRef(''),alignmentLoading=ref(false),waveform=shallowRef<AlignmentWaveform|null>(null),inspection=shallowRef<Record<string,any>>(),captureOffset=ref(0);
let saveDecision:((value:boolean)=>void)|undefined,canvasObserver:ResizeObserver|undefined;
const effectiveOffset=computed(()=>offsetPreview.value??offset.value);
const recordingUrl=ref('');
function clearRecordingUrl(){if(recordingUrl.value)URL.revokeObjectURL(recordingUrl.value);recordingUrl.value='';}
const mediaFailure=ref(false),offsets=ref<Record<string,number>>({});let restoringOffset=false;
const replayVideo=ref<HTMLVideoElement|null>(null),sourceUrls=ref<Record<string,string>>({});const unsaved=new Set<string>();
const replayGainMessage=useMediaAudition(replayVideo);
const sourceOrigins=ref<Record<string,number>>({}),mediaPreparing=ref(false),mediaMessage=messageRef('');
const compatibleSources=new Set<string>();let mediaGeneration=0,playGeneration=0,resumeMedia=false,pendingMediaTime:number|null=null;
let capture:LipCapture|undefined,abort:AbortController|undefined,replayTimer=0,replayFrameCallback=0,replayRequest=0;
let replayStore:LipReplay|undefined,disposed=false,historySourceId='';
const replayRow=shallowRef<LipRow>(),exportInfo=ref<Record<string,any>>();
const cloudOptIn=ref(false),history=ref<{id:string;name:string}[]>([]),historyId=ref('');
async function refreshHistory(){try{const stored=await port.value.history?.()??[];const current=results.value.map(r=>({id:r.id,name:r.name}));history.value=[...current.slice().reverse(),...stored.filter(h=>!current.some(r=>r.id===h.id))];if(!history.value.some(h=>h.id===historyId.value))historyId.value=history.value[0]?.id??'';}catch(e){error.value=uiErrorMessage(e);}}
async function loadHistory(){if(!historyId.value||!port.value.load||busy.value||recording.value)return;const old=results.value.findIndex(r=>r.id===historyId.value);if(old>=0){selection.value=old;batchIds.value=[historyId.value];return;}if(results.value.length>=100){error.value=uiMessage('modules.lip-extraction.this_page_retains_up_to_100_results_save_and');return;}busy.value=true;try{const result=await port.value.load(historyId.value);if(disposed)return;results.value.forEach(r=>r.rows=[]);result.rows=markRaw(result.rows);results.value.push(result);selection.value=results.value.length-1;batchIds.value=[result.id];}catch(e){error.value=uiErrorMessage(e);}finally{busy.value=false;}}
const port=computed(()=>props.port??unavailableLipPort),state=computed(()=>{revision.value;return capture?{...capture.state,stats:{...capture.state.stats}}:undefined;}),recording=computed(()=>!!state.value&&['opening','previewing','recording','stopping','finalizing'].includes(state.value.phase));
const canAnalyze=computed(()=>port.value.available&&(port.value.kind==='desktop'||cloudOptIn.value));
const selected=computed(()=>results.value[selection.value]);
const saveTargets=computed(()=>saveAll.value&&batchIds.value.length>1?results.value.filter(r=>batchIds.value.includes(r.id)):selected.value?[selected.value]:[]);
const recordedRows=computed<LipRow[]>(()=>{revision.value;const anchor=inspection.value?.audio?.first_decoded_pts_s??0;return (capture?.frames??[]).map(r=>({...r,time_s:r.time_s-anchor}));});
const rows=computed<LipRow[]>(()=>selected.value?.rows??recordedRows.value);
const shown=computed(()=>selected.value||recordingUrl.value?replayRow.value:state.value?.latest);
const animationBounds=computed(()=>recording.value?null:faceBounds(rows.value));
const playbackUrl=computed(()=>selected.value?sourceUrls.value[selected.value.id]??'':recordingUrl.value);
const recordedMode=computed(()=>{revision.value;return capture?.metadata().settings?.mode??mode.value;});
const phaseText=computed(()=>({idle:tr('modules.lip-extraction.not_started'),opening:tr('modules.lip-extraction.preparing'),previewing:tr('modules.lip-extraction.previewing'),recording:tr('modules.lip-extraction.recording'),stopping:tr('modules.lip-extraction.stopping'),finalizing:tr('modules.lip-extraction.finalizing'),ready:tr('modules.lip-extraction.ended'),failed:tr('modules.lip-extraction.capture_failed')}[state.value?.phase??'idle']));
const canSaveRecording=computed(()=>{revision.value;return !busy.value&&['ready','failed'].includes(state.value?.phase??'')&&!!capture?.chunks.length;});
const inputWarning=computed(()=>{const audio=state.value?.audio;if(mode.value==='preview'&&!audio?.label)return '';if(isLoopback(audio?.label||devices.value.find(d=>d.deviceId===microphone.value)?.label||''))return tr('modules.lip-extraction.current_input_is_stereo_mix_system_playback_to_record');if(audio?.muted)return tr('modules.lip-extraction.the_microphone_track_is_receiving_no_data_check_device');if(audio?.lowSignal||exportInfo.value?.audio?.signal?.low_signal)return tr('modules.lip-extraction.input_is_very_low_peak_below_60_dbfs_speak');return audio?.monitorError??'';});
const microphoneLevelText=computed(()=>{const peak=state.value?.audio.peakDb;const level=peak==null?tr('modules.lip-extraction.level_not_ready'):`${peak.toFixed(1)} dBFS`;return recording.value?level:tr('modules.lip-extraction.stoppedInputLevel',{level});});
const frameCount=computed(()=>selected.value?.metadata.timing?.decoded_frames??rows.value.length);

const curveKeys=['area','outer_width','open','circularity'];
const curves=computed(()=>Object.fromEntries(curveKeys.map(key=>[key,curve(key)])));
const dirty=computed(()=>{revision.value;return !!capture?.state.dirty||resultDirty.value||recording.value||busy.value;});
watch(dirty,value=>emit('dirty',value));
watch(recording,value=>{if(!value)releaseCapture('M05');});
onUnmounted(()=>{if(!deviceBusy.value)releaseCapture('M05');});
watch(selected,async result=>{mediaGeneration++;mediaPreparing.value=false;mediaMessage.value='';resumeMedia=false;pendingMediaTime=null;mediaFailure.value=false;stopReplay();replayStore=undefined;replayRow.value=undefined;replayIndex.value=0;restoringOffset=true;offset.value=result?(offsets.value[result.id]??result.metadata.timing?.lip_manual_offset??0):captureOffset.value;restoringOffset=false;if(!result)return;
 try{if(!result.rows.length&&port.value.load){const loaded=await port.value.load(result.id);if(disposed||selected.value?.id!==result.id)return;results.value.forEach(r=>r.rows=[]);result.rows=markRaw(loaded.rows);}if(selected.value?.id!==result.id)return;replayStore=new LipReplay(result,port.value);await seekFrame(0,false);if(!sourceUrls.value[result.id]&&port.value.sourceVideo&&result.metadata.input_asset){const blob=await port.value.sourceVideo(result);if(disposed||selected.value?.id!==result.id)return;if(historySourceId){URL.revokeObjectURL(sourceUrls.value[historySourceId]);delete sourceUrls.value[historySourceId];delete sourceOrigins.value[historySourceId];compatibleSources.delete(historySourceId);}historySourceId=result.id;sourceOrigins.value[result.id]=0;compatibleSources.delete(result.id);sourceUrls.value[result.id]=URL.createObjectURL(blob);}}catch(e){error.value=uiErrorMessage(e);}});
watch(offset,()=>{if(restoringOffset)return;if(selected.value){offsets.value[selected.value.id]=offset.value;unsaved.add(selected.value.id);resultDirty.value=true;}else if(recordingUrl.value&&capture){captureOffset.value=offset.value;capture.state.dirty=true;update();}syncReplay();},{flush:'sync'});
watch(effectiveOffset,()=>syncReplay());
watch(()=>props.active,active=>{if(active===false){stopReplay();capture?.hidden();}});
watch(shown,()=>draw(),{deep:false});
watch(mirror,()=>draw());
function update(){revision.value++;if(!selected.value)draw();}
async function refresh(authorizeAudio=false,announce=false){
 if(deviceBusy.value||disposed)return;deviceBusy.value=true;deviceAbort=new AbortController();let leased=false,prepared=false;
 if(announce){error.value='';notice.value='';}
 try{
  if(!navigator.mediaDevices?.enumerateDevices)throw uiError(uiMessage('modules.lip-extraction.cannot_read_devices_open_via_https_or_the_desktop'));
  const listed=await readCaptureDevices(navigator.mediaDevices,{authorizeAudio,signal:deviceAbort.signal,prepareAudio:async()=>{claimCapture('M05');leased=true;await port.value.prepareCapture?.('devices');prepared=true;}});
  if(disposed)return;devices.value=listed;
  if(announce)notice.value=audioDevicesHidden(listed)?uiMessage('modules.lip-extraction.audio_device_list_is_not_available_yet_check_microphone'):uiMessage('modules.lip-extraction.device_list_refreshed_choose_a_microphone_or_sound_card');
 }catch(e){if(!disposed){const display=deviceAccessErrorMessage(e),raw=messageText(display);const message=prepared?await port.value.captureError?.(raw,'devices').catch(()=>raw)??raw:raw;error.value=message===raw?display:uiErrorMessage(uiError(message));}}
 finally{if(leased)releaseCapture('M05');deviceBusy.value=false;deviceAbort=undefined;}
}
async function start(){if(mode.value==='offline'||recording.value||busy.value||deviceBusy.value)return;error.value='';notice.value='';stopReplay();if(dirty.value&&!await save())return;clearRecordingUrl();results.value=[];selection.value=0;Object.values(sourceUrls.value).forEach(URL.revokeObjectURL);sourceUrls.value={};files.value=[];offsets.value={};exportInfo.value=undefined;localSaved.value=false;inspection.value=undefined;waveform.value=null;captureOffset.value=0;offset.value=0;replayRow.value=undefined;replayStore=undefined;
 try{claimCapture('M05');await port.value.prepareCapture?.();await capture!.start({camera:camera.value,microphone:microphone.value,mode:mode.value,filter:filter.value,cutoff:cutoff.value,delegate:delegate.value,requestedFps:fps.value,preferMp4:!port.value.saveRecording} as CaptureSettings);await refresh();}catch(e){if(!recording.value)releaseCapture('M05');const display=uiErrorMessage(e),raw=messageText(display);const message=await port.value.captureError?.(raw).catch(()=>raw)??raw;error.value=message===raw?display:uiErrorMessage(uiError(message));}}
async function stop(){try{await capture?.stop();if(capture?.chunks.length){
 clearRecordingUrl();recordingUrl.value=URL.createObjectURL(capture.blob());
 busy.value=true;alignmentError.value='';progress.value=uiMessage('modules.lip-extraction.checking_audio_timeline');
 try{if(port.value.inspectRecording){inspection.value=await port.value.inspectRecording(capture.blob());waveform.value=inspection.value.waveform;}else await decodeWaveform(capture.blob());}catch(e){alignmentError.value=uiErrorMessage(e);notice.value=uiMessage('modules.lip-extraction.audio_timeline_check_failed_recording_is_retained_retry_the');}finally{busy.value=false;progress.value='';}
 setupCaptureReplay();if(!alignmentError.value)notice.value=recordedMode.value==='record_then_analyze'?uiMessage('modules.lip-extraction.recording_ended_save_video_and_audio_then_compute_lip'):uiMessage('modules.lip-extraction.recording_ended_play_video_and_animation_in_sync_or');
 }else notice.value=uiMessage('modules.lip-extraction.preview_ended_no_video_audio_or_lip_records_were');}catch(e){error.value=uiErrorMessage(e);}}
function setupCaptureReplay(){if(!rows.value.length)return;replayStore=new LipReplay({id:'capture',name:tr('modules.lip-extraction.current_recording'),backend:'candidate',rows:recordedRows.value,metadata:{},files:[]},{...port.value,replay:undefined});void seekFrame(0,false);}
async function decodeWaveform(blob:Blob){const context=new AudioContext();try{const buffer=await context.decodeAudioData(await blob.arrayBuffer());if(buffer.length>60000000)throw uiError(uiMessage('modules.lip-extraction.audio_exceeds_waveform_preview_limits'));waveform.value=waveformEnvelope(buffer);}finally{await context.close();}}
async function openOffset(){stopReplay();offsetDialog.value=true;offsetPreview.value=offset.value;alignmentError.value='';alignmentLoading.value=true;
 try{if(selected.value){waveform.value=null;if(!port.value.audioPreview)throw uiError(uiMessage('modules.lip-extraction.result_audio_cannot_be_read_now'));const data=await port.value.audioPreview(selected.value);waveform.value=data.waveform;}
 else if(!inspection.value&&port.value.inspectRecording){inspection.value=await port.value.inspectRecording(capture!.blob());waveform.value=inspection.value.waveform;setupCaptureReplay();}
 else if(!waveform.value)await decodeWaveform(capture!.blob());}
 catch(e){alignmentError.value=uiErrorMessage(e);}finally{alignmentLoading.value=false;}}
function closeOffset(){offsetDialog.value=false;offsetPreview.value=null;}
function applyOffset(value:number){offset.value=value;closeOffset();notice.value=uiMessage('modules.lip-extraction.offset_applied_to_playback_save_the_recording_separately_or');}

function choose(event:Event){error.value='';notice.value='';files.value=Array.from((event.target as HTMLInputElement).files??[]);if(files.value.length>100){files.value=[];error.value=uiMessage('modules.lip-extraction.up_to_100_videos_per_batch');}}
async function analyze(){if(!files.value.length||recording.value||busy.value||!canAnalyze.value)return;if(results.value.length+files.value.length>100){error.value=uiMessage('modules.lip-extraction.this_page_processes_up_to_100_results_work_in');return;}error.value='';notice.value='';busy.value=true;abort=new AbortController();batchIds.value=[];saveAll.value=true;saveVideo.value=saveAnimation.value=true;const failures:(string|MessageDescriptor)[]=[];
 try{stopReplay();for(const file of files.value){if(abort.signal.aborted)break;try{progress.value=uiMessage('modules.lip-extraction.preparing_value',{v0:(file.name)});const result=await port.value.analyze(file,{filter_enabled:filter.value,cutoff_hz:cutoff.value},abort.signal,text=>progress.value=joinMessages(file.name,'：',text));if(disposed)return;results.value.forEach(r=>r.rows=[]);result.rows=markRaw(result.rows);sourceUrls.value[result.id]=URL.createObjectURL(file);results.value.push(result);selection.value=results.value.length-1;batchIds.value.push(result.id);unsaved.add(result.id);resultDirty.value=true;}catch(e){failures.push(joinMessages(file.name,'：',uiErrorMessage(e)));if(abort.signal.aborted)break;}}notice.value=uiMessage('modules.lip-extraction.completed_value_value_files_results_await_saving',{v0:(batchIds.value.length),v1:(files.value.length)});if(failures.length)error.value=joinMessages(...failures.flatMap((item,index)=>index?['；',item]:[item]));}
 finally{progress.value='';abort=undefined;try{if(!disposed)await refreshHistory();}finally{busy.value=false;}}}
function captureMetadata(){const meta=capture!.metadata();return new Blob([JSON.stringify({...meta,resources:meta.inference_requested?resources.files:null,lip_manual_offset:captureOffset.value,inspection:inspection.value?{audio:inspection.value.audio,first_video_pts_s:inspection.value.first_video_pts_s}:null,save_options:recordedMode.value==='record_then_analyze'?{video:true,animation:false}:{video:saveVideo.value,animation:saveAnimation.value},frames:capture!.frames})],{type:'application/json'});}
function requestSave():Promise<boolean>{if(saveDialog.value)return Promise.resolve(false);if(busy.value||!canSaveRecording.value)return Promise.resolve(false);if(recordedMode.value==='record_then_analyze')return saveRecording();saveVideo.value=saveAnimation.value=true;saveDialog.value=true;return new Promise(resolve=>{saveDecision=resolve;});}
function cancelSave(){saveDialog.value=false;saveDecision?.(false);saveDecision=undefined;}
async function confirmSave(){const ok=await saveRecording();if(ok){saveDialog.value=false;saveDecision?.(true);saveDecision=undefined;}}
async function saveRecording(){if(busy.value)return false;stopReplay();error.value='';busy.value=true;try{const blob=capture!.blob();
 if(!port.value.saveRecording)throw uiError(uiMessage('modules.lip-extraction.recording_retained_use_the_desktop_app_to_export'));
 progress.value=uiMessage('modules.lip-extraction.saving_selected_media_audio_and_lip_data');const saved=await port.value.saveRecording('唇形原始录制.'+(blob.type.includes('mp4')?'mp4':'webm'),blob,captureMetadata());
 if(saved.saved){localSaved.value=true;exportInfo.value=saved.recording;capture?.markSaved();notice.value=joinMessages(uiMessage('modules.lip-extraction.selected_recording_files_saved_and_verified'),(saved.recording?.video_timing_warning??''));return true;}
 notice.value=uiMessage('modules.lip-extraction.save_cancelled_recording_retained');return false;
 }catch(e){error.value=joinMessages(uiMessage('modules.lip-extraction.save_failed'),uiErrorMessage(e));return false;}finally{busy.value=false;progress.value='';}}
async function saveResults(targets:LipResult[],action:'apply'|'save_without_offset'){if(busy.value||!targets.length)return false;stopReplay();error.value='';busy.value=true;let savedCount=0;
 const onSaved=(id:string)=>{unsaved.delete(id);resultDirty.value=unsaved.size>0;savedCount++;};
 const items=targets.map(result=>({result,offset:action==='apply'?(result.id===selected.value?.id?offset.value:offsets.value[result.id]??result.metadata.timing?.lip_manual_offset??0):0}));
 try{progress.value=uiMessage('modules.lip-extraction.saving_value_results',{v0:(items.length)});const ok=port.value.saveBatch?await port.value.saveBatch(items,action,{video:saveVideo.value,animation:saveAnimation.value,quality:quality.value},onSaved):await (async()=>{for(const item of items){if(!await port.value.save(item.result,item.offset,action))return false;onSaved(item.result.id);}return true;})();notice.value=ok?uiMessage('modules.lip-extraction.saved_value_results',{v0:(savedCount)}):uiMessage('modules.lip-extraction.saved_value_results_remaining_results_are_retained_and_can',{v0:(savedCount)});return ok;}catch(e){error.value=uiErrorMessage(e);notice.value=uiMessage('modules.lip-extraction.saved_value_results_remaining_results_are_retained_you_can',{v0:(savedCount)});return false;}finally{busy.value=false;progress.value='';}}
async function saveResult(action:'apply'|'save_without_offset'){return saveResults(saveTargets.value,action);}
async function save(){if(recording.value||busy.value){error.value=uiMessage('modules.lip-extraction.capture_or_analysis_is_not_finished_stop_it_and');return false;}const pending=results.value.filter(r=>unsaved.has(r.id));if(pending.length&&!await saveResults(pending,'apply'))return false;if(capture?.state.dirty&&!await requestSave())return false;return !dirty.value;}

defineExpose({save,dispose:()=>capture?.dispose()});
function discard(){stopReplay();clearRecordingUrl();capture?.discard();results.value=[];batchIds.value=[];unsaved.clear();Object.values(sourceUrls.value).forEach(URL.revokeObjectURL);sourceUrls.value={};resultDirty.value=false;confirmDiscard.value=false;inspection.value=undefined;waveform.value=null;exportInfo.value=undefined;replayRow.value=undefined;replayStore=undefined;notice.value=uiMessage('modules.lip-extraction.unsaved_content_on_this_page_discarded_original_disk_files');}
function cancelReplayClock(){if(replayTimer)cancelAnimationFrame(replayTimer);if(replayFrameCallback)replayVideo.value?.cancelVideoFrameCallback(replayFrameCallback);replayTimer=replayFrameCallback=0;}
function stopReplay(cancelIntent=true){if(cancelIntent){playGeneration++;resumeMedia=false;}replaying.value=false;cancelReplayClock();replayVideo.value?.pause();replayRequest++;}
function replayOrigin(){return selected.value?(sourceOrigins.value[selected.value.id]??0)-(selected.value.metadata.timing?.anchor_s??0):-(inspection.value?.audio?.first_decoded_pts_s??0);}
async function preparePlayback(){
 const result=selected.value;if(!result||!port.value.playbackVideo||mediaPreparing.value)return;
 const generation=++mediaGeneration,position=(replayVideo.value?.currentTime??0)+replayOrigin();
 mediaPreparing.value=true;mediaMessage.value=uiMessage('modules.lip-extraction.preparing_audio_video_playback');stopReplay(false);
 try{const media=await port.value.playbackVideo(result);if(disposed||generation!==mediaGeneration||selected.value?.id!==result.id)return;
  const previous=sourceUrls.value[result.id];sourceOrigins.value[result.id]=media.origin_s;pendingMediaTime=Math.max(0,position-replayOrigin());compatibleSources.add(result.id);mediaFailure.value=false;mediaMessage.value='';sourceUrls.value[result.id]=URL.createObjectURL(media.blob);if(previous)URL.revokeObjectURL(previous);
 }catch(e){if(generation===mediaGeneration){mediaFailure.value=true;mediaMessage.value=joinMessages(uiMessage('modules.lip-extraction.failed_to_prepare_audio_video_playback'),uiErrorMessage(e));resumeMedia=false;}}
 finally{if(generation===mediaGeneration)mediaPreparing.value=false;}
}
function mediaError(){
 mediaFailure.value=true;stopReplay(false);
 if(selected.value&&port.value.playbackVideo&&!compatibleSources.has(selected.value.id)){void preparePlayback();return;}
 mediaMessage.value=uiMessage('modules.lip-extraction.this_video_cannot_play_now_retry_preparing_audio_video');
}
function mediaLoaded(){
 if(replayVideo.value&&!replayVideo.value.videoWidth){mediaError();return;}
 if(pendingMediaTime!==null&&replayVideo.value){replayVideo.value.currentTime=pendingMediaTime;pendingMediaTime=null;}
 if(resumeMedia){resumeMedia=false;if(!disposed&&props.active!==false)void play();}
}
function missingAudio(time:number){
 const media=replayVideo.value as (HTMLVideoElement&{webkitAudioDecodedByteCount?:number})|null;
 return !!selected.value?.metadata.audio?.present&&time+replayOrigin()>.2&&media?.webkitAudioDecodedByteCount===0;
}
async function seekFrame(index:number,moveVideo=true){
 const store=replayStore,request=++replayRequest;if(!store)return;
 try{const row=await store.frame(Number(index));if(disposed||request!==replayRequest||store!==replayStore||!row)return;replayRow.value=row;replayIndex.value=row.index;store.prefetch(row.index);if(moveVideo&&replayVideo.value)replayVideo.value.currentTime=Math.max(0,row.time_s-replayOrigin()+effectiveOffset.value);draw();}catch(e){if(request===replayRequest)error.value=uiErrorMessage(e);}
}
async function showTime(t:number){
 const store=replayStore,request=++replayRequest;if(!store)return;
 try{const row=await store.time(t);if(disposed||request!==replayRequest||store!==replayStore||!row)return;replayRow.value=t<(rows.value[0]?.time_s??0)||t>(rows.value.at(-1)?.time_s??0)+.15?{...row,detected:false,points:null,metrics:null}:row;replayIndex.value=row.index;store.prefetch(row.index);}catch(e){if(request===replayRequest){stopReplay();error.value=uiErrorMessage(e);}}
}
function replayAllowed(){return !disposed&&props.active!==false;}
async function play(){if(!replayAllowed()||!rows.value.length)return;const ticket=++playGeneration;if(mediaPreparing.value){resumeMedia=true;return;}if(mediaFailure.value&&playbackUrl.value){resumeMedia=true;await preparePlayback();return;}if(replayIndex.value>=frameCount.value-1)await seekFrame(0);if(!replayAllowed()||ticket!==playGeneration)return;const media=replayVideo.value;
 if(!mediaFailure.value&&playbackUrl.value&&media){try{await media.play();}catch(e){error.value=uiErrorMessage(e);stopReplay();}return;}
 replaying.value=true;const first=performance.now(),origin=replayRow.value?.time_s??rows.value[0].time_s;
 const tick=()=>{if(!replaying.value)return;const now=origin+(performance.now()-first)/1000;void showTime(now);if(now>=(rows.value.at(-1)?.time_s??0)){replaying.value=false;return;}replayTimer=requestAnimationFrame(tick);};tick();
}
function mediaPlaying(){if(disposed||props.active===false){stopReplay();return;}cancelReplayClock();replaying.value=true;const media=replayVideo.value;if(!media)return;
 const tick=(_now:number,meta:VideoFrameCallbackMetadata)=>{if(media!==replayVideo.value||media.paused)return;if(missingAudio(meta.mediaTime)){resumeMedia=true;mediaError();return;}void showTime(meta.mediaTime+replayOrigin()-effectiveOffset.value);replayFrameCallback=media.requestVideoFrameCallback(tick);};
 if(media.requestVideoFrameCallback)replayFrameCallback=media.requestVideoFrameCallback(tick);
 else{const fallback=()=>{if(media.paused)return;if(missingAudio(media.currentTime)){resumeMedia=true;mediaError();return;}syncReplay();replayTimer=requestAnimationFrame(fallback);};fallback();}
 syncReplay();
}
function mediaPaused(){replaying.value=false;cancelReplayClock();syncReplay();}
function syncReplay(){if(replayVideo.value)void showTime(replayVideo.value.currentTime+replayOrigin()-effectiveOffset.value);}
function draw(){const el=canvas.value;if(!el)return;const row=shown.value;const box=el.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 const width=Math.max(2,Math.round(box.width*dpr)),height=Math.max(2,Math.round(box.height*dpr));if(el.width!==width)el.width=width;if(el.height!==height)el.height=height;
 const context=el.getContext('2d');if(!context)return;context.clearRect(0,0,width,height);
 if(row?.points)drawLegacyOverlay(context,fitFace(row.points,width,height,animationBounds.value??undefined,.94),width,height);}

function curve(key:string){const data=selected.value?selected.value.rows:rows.value.slice(-300);if(!data.length)return '';const values=data.map(r=>r.metrics?.[key]).filter((x):x is number=>typeof x==='number'&&Number.isFinite(x));if(!values.length)return '';const low=Math.min(...values),high=Math.max(...values),start=data[0].time_s,end=data.at(-1)!.time_s;let path='',connected=false;for(const r of data){const value=r.metrics?.[key];if(typeof value!=='number'||!Number.isFinite(value)){connected=false;continue;}const x=10+(r.time_s-start)/Math.max(.001,end-start)*580,y=110-(value-low)/Math.max(1e-9,high-low)*90;path+=`${connected?'L':'M'}${x.toFixed(2)},${y.toFixed(2)} `;connected=true;}return path;}

async function exportAnimation(format:'mp4'|'gif'){if(busy.value||!selected.value||!port.value.exportAnimation)return;error.value='';busy.value=true;progress.value=uiMessage('modules.lip-extraction.exporting_animation');abort=new AbortController();try{const ok=await port.value.exportAnimation({...selected.value,metadata:{...selected.value.metadata,timing:{...selected.value.metadata.timing,lip_manual_offset:offset.value}}},format,quality.value,abort.signal);notice.value=ok?uiMessage('modules.lip-extraction.animation_written'):uiMessage('modules.lip-extraction.animation_export_cancelled');}catch(e){error.value=uiErrorMessage(e);}finally{busy.value=false;progress.value='';abort=undefined;}}
function beforeUnload(event:BeforeUnloadEvent){if(dirty.value){event.preventDefault();event.returnValue='';}}
function hidden(){if(document.hidden)capture?.hidden();}
onMounted(()=>{capture=new LipCapture(video.value!,update);canvasObserver=new ResizeObserver(()=>draw());if(canvas.value)canvasObserver.observe(canvas.value);void refresh();void refreshHistory();window.addEventListener('beforeunload',beforeUnload);document.addEventListener('visibilitychange',hidden);});
onUnmounted(()=>{disposed=true;deviceAbort?.abort();mediaGeneration++;canvasObserver?.disconnect();saveDecision?.(false);clearRecordingUrl();Object.values(sourceUrls.value).forEach(URL.revokeObjectURL);abort?.abort();stopReplay();capture?.dispose();window.removeEventListener('beforeunload',beforeUnload);document.removeEventListener('visibilitychange',hidden);});
useWebSettings(props.stateKey,{camera,microphone,filter,cutoff,fps,delegate,mirror,quality,saveAll,saveVideo,saveAnimation});
</script>
<template>
<ModuleFrame unified fit :label="tr('modules.lip-extraction.lip_extraction_workspace')" class="lip-page" :aria-busy="busy">
 <template #toolbar><ModuleToolbar>
  <label>{{tr('modules.lip-extraction.mode')}} <select v-model="mode" :aria-label="tr('modules.lip-extraction.capture_mode')" :disabled="recording||busy||deviceBusy"><option value="preview">{{tr('modules.lip-extraction.preview')}}</option><option value="realtime">{{tr('modules.lip-extraction.live_computation')}}</option><option value="record_then_analyze">{{tr('modules.lip-extraction.video_recording')}}</option><option value="offline">{{tr('modules.lip-extraction.offline_analysis')}}</option></select></label>
  <input ref="fileInput" class="lip-file-input" type="file" accept=".mp4,.mov,.avi,.mkv,.wmv,.m4v,.webm" multiple :disabled="busy||recording" :aria-label="tr('modules.lip-extraction.select_offline_videos')" @change="choose"/>
  <button v-if="mode==='offline'" class="primary" :disabled="recording||busy" @click="fileInput?.click()">{{tr('modules.lip-extraction.select_files')}}</button>
  <button v-if="mode==='offline'" class="primary" :disabled="busy||recording||!files.length||!canAnalyze" @click="analyze">{{tr('modules.lip-extraction.analyse_selected_files')}}</button>
  <button v-else class="primary" :disabled="recording||busy||deviceBusy" @click="start">{{dirty?tr('modules.lip-extraction.save_and_start_next'):mode==='preview'?tr('modules.lip-extraction.open_preview'):tr('modules.lip-extraction.start_recording')}}</button>
  <button v-if="mode!=='offline'" class="primary" :disabled="!state||!['opening','previewing','recording'].includes(state.phase)" @click="stop">{{tr('modules.lip-extraction.end_recording')}}</button>
  <button v-if="mode!=='offline'" class="primary" :disabled="!canSaveRecording" @click="requestSave">{{tr('modules.lip-extraction.save_recording_as')}}</button>
  <button v-if="busy&&abort&&progress" @click="abort?.abort()">{{tr('modules.lip-extraction.cancel_this_task')}}</button>
  <button :disabled="recording||busy||!dirty" @click="confirmDiscard=true">{{tr('modules.lip-extraction.discard_unsaved_content')}}</button>
  <button :disabled="recording||busy||!rows.length||(!selected&&!recordingUrl)" @click="openOffset">{{tr('modules.lip-extraction.check_offset')}}</button>
  <template #actions><button @click="help=!help">{{tr('modules.lip-extraction.help')}}</button><button @click="emit('references')"><AppIcon name="book"/>{{tr('modules.lip-extraction.methods_and_references')}}</button></template>
 </ModuleToolbar></template>
 <template #status><ModuleStatus v-if="error||state?.error" kind="error" :message="error||state?.error||''"/><ModuleStatus v-if="notice||progress" kind="info" :message="progress||notice"/></template>
 <ModuleWorkbench unified :state-key="stateKey??'M05'" :left-label="tr('modules.lip-extraction.devices_and_capture')" :right-label="tr('modules.lip-extraction.video_and_records')">
 <template #left>
 <ModuleSection v-if="mode==='offline'" :label="tr('modules.lip-extraction.selected_videos')" :title="tr('modules.lip-extraction.selected_files')"><ul v-if="files.length" class="lip-file-list"><li v-for="(file,i) in files" :key="i">{{file.name}} · {{(file.size/1e6).toFixed(1)}} MB</li></ul><p v-else class="lip-save-note">{{tr('modules.lip-extraction.select_one_or_more_videos_to_analyse')}}</p><label v-if="port.kind==='browser'&&port.available"><input v-model="cloudOptIn" type="checkbox"/>{{tr('modules.lip-extraction.upload_selected_videos_to_the_current_account_for_offline')}}</label><ModuleStatus v-if="!port.available" kind="info" :message="port.reason??tr('modules.lip-extraction.offline_analysis_is_not_yet_available_on_this_platform')"/></ModuleSection>
 <ModuleSection :label="tr('modules.lip-extraction.media_devices_and_parameters')" :title="tr('modules.lip-extraction.devices_and_parameters')"><div class="lip-fields">
  <label v-if="mode!=='offline'">{{tr('modules.lip-extraction.camera')}} <select v-model="camera" :disabled="recording||busy||deviceBusy"><option value="">{{tr('modules.lip-extraction.system_default')}}</option><option v-if="camera&&!devices.some(d=>d.kind==='videoinput'&&d.deviceId===camera)" :value="camera">{{tr('modules.lip-extraction.selected_camera_is_unavailable')}}</option><option v-for="d in devices.filter(d=>d.kind==='videoinput')" :key="d.deviceId" :value="d.deviceId">{{d.label||tr('modules.lip-extraction.camera_not_authorized')}}</option></select></label>
  <label v-if="mode!=='offline'">{{tr('modules.lip-extraction.microphone')}} <select v-model="microphone" :aria-label="tr('modules.lip-extraction.microphone')" :disabled="recording||busy||deviceBusy"><option value="">{{tr('modules.lip-extraction.select_microphone_automatically')}}</option><option v-if="microphone&&!devices.some(d=>d.kind==='audioinput'&&d.deviceId===microphone)" :value="microphone">{{audioNeedsPermission?tr('modules.lip-extraction.selected_microphone_needs_permission'):tr('modules.lip-extraction.selected_microphone_is_unavailable')}}</option><option v-for="d in devices.filter(d=>d.kind==='audioinput'&&d.deviceId&&d.label)" :key="d.deviceId" :value="d.deviceId">{{d.label}}</option></select></label>
  <p v-if="mode==='preview'" class="lip-save-note">{{tr('modules.lip-extraction.preview_uses_only_the_camera_the_selected_microphone_is')}}</p>
  <div v-if="state?.audio.label" class="lip-input-level"><span>{{tr('modules.lip-extraction.actual_input')}}{{state.audio.label}}</span><meter :aria-label="tr('modules.lip-extraction.microphone_input_level')" min="-120" max="0" :value="state.audio.peakDb??-120"/><span>{{microphoneLevelText}}</span></div>
  <ModuleStatus v-if="inputWarning" class="lip-input-warning" kind="info" :message="inputWarning"/>
  <p v-if="mode!=='offline'&&audioNeedsPermission" class="lip-save-note">{{tr('modules.lip-extraction.authorization_briefly_enables_the_microphone_to_read_the_full')}}</p>
  <button v-if="mode!=='offline'" :disabled="recording||busy||deviceBusy" @click="refresh(true,true)">{{deviceBusy?tr('modules.lip-extraction.reading_devices'):audioNeedsPermission?tr('modules.lip-extraction.authorize_and_refresh_devices'):tr('modules.lip-extraction.refresh_devices')}}</button>
  <label v-if="mode!=='offline'">{{tr('modules.lip-extraction.requested_frame_rate')}} <span><input v-model.number="fps" type="number" min="1" max="240" :disabled="recording||busy"/> fps</span></label>
  <label><input v-model="filter" type="checkbox" :disabled="recording||busy"/>{{tr('modules.lip-extraction.landmark_stabilization')}}</label>
  <label>{{tr('modules.lip-extraction.cutoff_frequency')}} <span><input v-model.number="cutoff" type="number" min="1" max="240" :disabled="recording||busy||!filter"/> Hz</span></label>
  <label v-if="mode!=='offline'">{{tr('modules.lip-extraction.inference_device')}} <select v-model="delegate" :disabled="recording||busy"><option>CPU</option><option>GPU</option></select></label>
  <label><input v-model="mirror" type="checkbox"/>{{tr('modules.lip-extraction.mirror_preview')}}</label>
 </div></ModuleSection>
 <ModuleSection v-if="selected" :label="tr('modules.lip-extraction.offline_analysis_statistics')" :title="tr('modules.lip-extraction.analysis_status')"><div class="lip-state">
  <span class="lip-state-wide">{{selected.name}}</span><span>{{tr('modules.lip-extraction.decoded')}} {{frameCount}}</span><span>{{tr('modules.lip-extraction.inferred')}} {{frameCount}}</span><span>{{tr('modules.lip-extraction.detected')}} {{selected.metadata.validity?.detected??'—'}}</span><span>{{tr('modules.lip-extraction.not_detected')}} {{selected.metadata.validity?.missing??'—'}}</span><span>{{tr('modules.lip-extraction.skipped_inference_0')}}</span>
  <span class="lip-state-wide">{{tr('modules.lip-extraction.file')}} {{selected.metadata.timing?.decoded_observed_fps?.toFixed(1)??'—'}} fps · {{selected.metadata.audio?.present?tr('modules.lip-extraction.with_audio'):tr('modules.lip-extraction.no_audio')}}</span>
 </div><p class="lip-save-note">{{tr('modules.lip-extraction.offline_analysis_computes_each_decoded_frame_see_capture_records')}}</p></ModuleSection>
 <ModuleSection v-if="mode!=='offline'||(state?.stats.presented??0)>0" :label="tr('modules.lip-extraction.capture_and_save_status')" :title="selected?tr('modules.lip-extraction.previous_capture_on_this_page'):tr('modules.lip-extraction.capture_status')"><div class="lip-state" :data-phase="state?.phase??'idle'">
  <span class="lip-state-wide">{{tr('modules.lip-extraction.stage')}} <strong>{{phaseText}}</strong></span><span>{{tr('modules.lip-extraction.presented')}} {{state?.stats.presented??0}}</span><span>{{tr('modules.lip-extraction.inferred')}} {{state?.stats.processed??0}}</span><span>{{tr('modules.lip-extraction.detected')}} {{state?.stats.detected??0}}</span><span>{{tr('modules.lip-extraction.skipped_inference')}} {{state?.stats.skippedInference??0}}</span><span>{{tr('modules.lip-extraction.presentation_gaps')}} {{state?.stats.observedPresentationGaps??0}}</span><span>{{tr('modules.lip-extraction.encoded')}} {{((state?.stats.encodedBytes??0)/1e6).toFixed(1)}} MB</span>
  <span class="lip-state-wide">{{tr('modules.lip-extraction.device')}} {{state?.trackSettings?.width??'—'}}×{{state?.trackSettings?.height??'—'}} · {{state?.trackSettings?.frameRate??'—'}} fps</span>
  <span v-if="exportInfo" class="lip-state-wide">{{tr('modules.lip-extraction.file')}} {{exportInfo.video_frames}} {{tr('modules.lip-extraction.frames')}} {{exportInfo.observed_video_fps?.toFixed(1)??'—'}} fps · {{exportInfo.audio?.present?tr('modules.lip-extraction.with_audio'):tr('modules.lip-extraction.no_audio')}}</span>
  <span v-if="state?.stats.processed" class="lip-state-wide">{{tr('modules.lip-extraction.live')}} {{state.stats.processed}} {{tr('modules.lip-extraction.frames')}} {{capture?.metadata().overlay_display_fps?.toFixed(1)??'—'}} fps</span>
 </div>
 <p class="lip-save-note">{{recordedMode==='preview'?tr('modules.lip-extraction.preview_displays_video_and_lip_landmarks_only'):recordedMode==='record_then_analyze'?tr('modules.lip-extraction.saves_video_and_audio_compute_lip_data_separately_in'):tr('modules.lip-extraction.choose_video_and_animation_when_saving_separately_audio_and')}}</p>
 <details class="lip-stat-help"><summary>{{tr('modules.lip-extraction.statistics_explained')}}</summary><dl><dt>{{tr('modules.lip-extraction.presented')}}</dt><dd>{{tr('modules.lip-extraction.number_of_new_images_observed_through_video_frame_callbacks')}}</dd><dt>{{tr('modules.lip-extraction.inferred')}}</dt><dd>{{tr('modules.lip-extraction.images_processed_by_the_lip_model_including_images_without')}}</dd><dt>{{tr('modules.lip-extraction.detected')}}</dt><dd>{{tr('modules.lip-extraction.inferred_images_with_successfully_detected_facial_landmarks')}}</dd><dt>{{tr('modules.lip-extraction.skipped_inference')}}</dt><dd>{{tr('modules.lip-extraction.times_the_current_image_was_not_sent_to_the')}}</dd><dt>{{tr('modules.lip-extraction.presentation_gaps')}}</dt><dd>{{tr('modules.lip-extraction.accumulated_gaps_in_browser_presented_frame_numbers_these_indicate')}}</dd></dl></details>
 <p v-if="exportInfo" class="lip-save-note">{{tr('modules.lip-extraction.saved')}}{{exportInfo.files?.map((f:any)=>f.name).join('、')}}。</p>
 </ModuleSection>
 </template>
 <ModuleSection class="lip-face-section" :label="tr('modules.lip-extraction.video_and_lip_view')">
  <div class="lip-video" :class="{mirrored:mirror&&recording}"><canvas ref="canvas" width="960" height="540"/><div v-if="!shown?.points" class="lip-face-empty">{{recordedMode==='record_then_analyze'&&recording?tr('modules.lip-extraction.this_mode_records_media_only_lip_data_is_computed'):shown?tr('modules.lip-extraction.no_face_detected_in_this_frame'):tr('modules.lip-extraction.open_preview_record_or_select_a_video_to_show')}}</div></div>
  <div class="lip-metrics"><span v-for="key in ['area','height','outer_width','inner_width','total_width','open','circularity','face_width','face_height']" :key="key"><span>{{key}}</span><strong>{{shown?.metrics?.[key]?.toFixed(4)??'—'}}</strong></span></div>
  <div class="lip-curves"><figure v-for="key in curveKeys" :key="key"><figcaption>{{key}}</figcaption><svg viewBox="0 0 600 125" role="img" :aria-label="key+tr('modules.lip-extraction.parameter_curves')"><path :d="curves[key]" fill="none" stroke="currentColor" stroke-width="1.5"/></svg></figure></div>
  <p class="lip-curve-note">{{tr('modules.lip-extraction.actual_time_on_the_horizontal_axis_each_curve_has')}}<span v-if="!recording&&rows.length"> {{tr('modules.lip-extraction.lip_offset')}} {{Math.round(offset*1000)}} ms</span></p>
 </ModuleSection>
 <template #right>
 <ModuleSection v-if="results.length" :label="tr('modules.lip-extraction.results_offset_and_animation_playback')" :title="tr('modules.lip-extraction.save_results')"><select v-model.number="selection" :disabled="busy||recording" :aria-label="tr('modules.lip-extraction.select_lip_result')"><option v-for="(result,i) in results" :key="result.id" :value="i">{{result.name}}</option></select><p class="lip-save-note">{{selected?.backend}} {{tr('modules.lip-extraction.complete')}} {{frameCount}} {{tr('modules.lip-extraction.frames_curve_preview')}} {{rows.length}} {{tr('modules.lip-extraction.points_missing_detections_stay_blank')}}</p>
  <div class="lip-save-options"><label><input v-model="saveVideo" type="checkbox" :disabled="busy"/>{{tr('modules.lip-extraction.video')}}</label><label><input v-model="saveAnimation" type="checkbox" :disabled="busy"/>{{tr('modules.lip-extraction.animation')}}</label></div>
  <label v-if="batchIds.length>1"><input v-model="saveAll" type="checkbox" :disabled="busy"/>{{tr('modules.lip-extraction.save_entire_batch')}} {{batchIds.length}} {{tr('modules.lip-extraction.results')}}</label>
  <div class="lip-actions"><button class="primary" :disabled="busy" @click="saveResult('apply')">{{tr('modules.lip-extraction.save_apply_offset')}}</button><button class="primary" :disabled="busy" @click="saveResult('save_without_offset')">{{tr('modules.lip-extraction.save_without_offset')}}</button></div>
  <div class="lip-actions"><select v-model="quality" :aria-label="tr('modules.lip-extraction.animation_export_quality')"><option value="high">{{tr('modules.lip-extraction.high_definition_1080')}}</option><option value="standard">{{tr('modules.lip-extraction.standard_720')}}</option><option value="small">{{tr('modules.lip-extraction.small_size_540')}}</option></select><button class="primary" :disabled="busy||!port.exportAnimation" @click="exportAnimation('mp4')">{{tr('modules.lip-extraction.export_animation_mp4')}}</button><button class="primary" :disabled="busy||!port.exportAnimation" @click="exportAnimation('gif')">{{tr('modules.lip-extraction.export_gif')}}</button></div>
 </ModuleSection>
 <ModuleSection class="lip-media-section" :label="tr('modules.lip-extraction.video_playback')" :title="tr('modules.lip-extraction.video_and_synchronized_playback')">
  <video ref="video" class="lip-live-video" playsinline muted :class="{mirrored:mirror}" v-show="!playbackUrl" :aria-label="tr('modules.lip-extraction.live_camera_image')"/>
  <video v-if="playbackUrl" ref="replayVideo" class="lip-recording-preview" :src="playbackUrl" controls preload="metadata" :aria-label="tr('modules.lip-extraction.synchronized_video_and_animation_playback')" @error="mediaError" @loadedmetadata="mediaLoaded" @play="mediaPlaying" @pause="mediaPaused" @seeked="syncReplay" @ended="mediaPaused"/>
  <p v-if="replayGainMessage" role="status">{{replayGainMessage}}</p><p v-if="mediaMessage" class="lip-save-note" role="status">{{mediaMessage}}</p><button v-if="mediaFailure&&!mediaPreparing&&selected&&port.playbackVideo" @click="preparePlayback">{{tr('modules.lip-extraction.retry_audio_video_playback')}}</button>
  <div v-if="rows.length&&!recording" class="lip-replay-controls"><button :disabled="mediaPreparing" @click="replaying?stopReplay():play()">{{replaying?tr('modules.lip-extraction.pause_playback'):tr('modules.lip-extraction.synchronized_playback')}}</button><span>{{(replayRow?.time_s??0).toFixed(3)}} s</span><input v-model.number="replayIndex" type="range" min="0" :max="Math.max(0,frameCount-1)" :aria-label="tr('modules.lip-extraction.playback_frame')" @input="stopReplay();seekFrame(replayIndex)"/></div>
 </ModuleSection>
 <div v-if="help" class="lip-help">{{tr('modules.lip-extraction.live_lip_extraction_uses_face_landmarker_equivalence_to_v2')}}</div>
 <ModuleSection v-if="port.history" :label="tr('modules.lip-extraction.local_task_history')" :title="tr('modules.lip-extraction.historical_results')"><div class="lip-actions"><button :disabled="busy||recording" @click="refreshHistory">{{tr('modules.lip-extraction.refresh_local_task_history')}}</button><select v-model="historyId" :aria-label="tr('modules.lip-extraction.historical_lip_tasks')"><option v-for="job in history" :key="job.id" :value="job.id">{{job.name}}</option></select><button :disabled="busy||recording||!historyId" @click="loadHistory">{{tr('modules.lip-extraction.read_historical_results')}}</button></div></ModuleSection>

 </template>
 </ModuleWorkbench>
 <ModalDialog v-if="saveDialog" :title="tr('modules.lip-extraction.save_recording_as')" :close-disabled="busy" @close="cancelSave"><p>{{tr('modules.lip-extraction.audio_and_lip_data_are_always_saved_select_any')}}</p><div class="lip-save-options"><label><input v-model="saveVideo" type="checkbox" :disabled="busy"/>{{tr('modules.lip-extraction.video')}}</label><label><input v-model="saveAnimation" type="checkbox" :disabled="busy"/>{{tr('modules.lip-extraction.animation')}}</label></div><p class="lip-save-note">{{tr('modules.lip-extraction.video_and_animation_are_both_mp4_if_neither_is')}}</p><p v-if="error" role="alert">{{error}}</p><p v-if="busy" role="status">{{progress}}</p><template #footer><button :disabled="busy" @click="cancelSave">{{tr('modules.lip-extraction.cancel')}}</button><button class="primary" :disabled="busy" @click="confirmSave">{{tr('modules.lip-extraction.select_folder_and_save')}}</button></template></ModalDialog>
 <OffsetDialog v-if="offsetDialog" :rows="rows" :waveform="waveform" :offset="offset" :loading="alignmentLoading" :error="alignmentError" @preview="offsetPreview=$event" @apply="applyOffset" @close="closeOffset"/>
 <ModalDialog v-if="confirmDiscard" :title="tr('modules.lip-extraction.discard_unsaved_content')" @close="confirmDiscard=false"><p>{{tr('modules.lip-extraction.discard_unsaved_recordings_and_results_on_this_page')}}</p><template #footer><button @click="confirmDiscard=false">{{tr('modules.lip-extraction.cancel')}}</button><button @click="discard">{{tr('modules.lip-extraction.discard')}}</button></template></ModalDialog>
</ModuleFrame>
</template>
<style scoped>
.lip-file-input{display:none}.lip-file-list{padding-left:20px;overflow-wrap:anywhere}.lip-input-level{display:grid;gap:4px;overflow-wrap:anywhere}.lip-input-level meter{width:100%}.lip-input-warning{color:var(--warning,var(--accent))}.lip-save-note{font-size:var(--support-size);overflow-wrap:anywhere;color:var(--muted);margin:8px 0 0}
.lip-fields{display:grid;gap:8px}.lip-fields>label{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}.lip-fields>label:has(input[type=checkbox]){justify-content:flex-start}.lip-fields select{flex:1;min-width:90px;max-width:100%}.lip-fields input[type=number]{width:78px}.lip-fields label>span{white-space:nowrap}
.lip-face-section{container:lip-face / inline-size;display:flex;flex-direction:column;flex:1!important;min-height:440px;overflow:hidden}.lip-video{position:relative;width:100%;flex:1;min-height:180px;background:var(--panel);overflow:hidden}.lip-video canvas{position:absolute;inset:0;width:100%;height:100%}.lip-video.mirrored canvas,video.mirrored{transform:scaleX(-1)}.lip-face-empty{position:absolute;inset:0;display:grid;place-items:center;text-align:center;padding:24px;color:var(--muted)}
.lip-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px 16px;font-family:var(--font-figure);font-size:var(--figure-size);padding-top:10px;flex:none}.lip-metrics>span{display:flex;justify-content:space-between;gap:8px;min-width:0;font-variant-numeric:tabular-nums}.lip-metrics strong{font-weight:400}.lip-metrics span span{overflow-wrap:anywhere}
.lip-curves{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:12px;flex:none}.lip-curves figure{margin:0;min-width:0}.lip-curves svg{display:block;width:100%;height:55px;color:var(--accent)}.lip-curves path{vector-effect:non-scaling-stroke}.lip-curves figcaption{font-size:var(--figure-size);font-family:var(--font-figure)}.lip-curve-note{margin:3px 0 0;color:var(--muted);font-size:var(--figure-size);flex:none}
.lip-state{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;font-variant-numeric:tabular-nums}.lip-state-wide{grid-column:1/-1}.lip-state span{overflow-wrap:anywhere}.lip-actions{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:8px}.lip-actions select{width:100%}.lip-help{padding:10px;border:1px solid var(--border);line-height:1.6;overflow-wrap:anywhere}
.lip-stat-help{margin-top:10px;font-size:var(--support-size);color:var(--muted);line-height:1.6}.lip-stat-help summary{cursor:pointer}.lip-stat-help dl{margin:8px 0 0}.lip-stat-help dt{color:var(--text);margin-top:6px}.lip-stat-help dd{margin:0}
.lip-live-video,.lip-recording-preview{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:var(--app);max-height:260px}.lip-replay-controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}.lip-replay-controls input{width:100%}.lip-save-options{display:flex;gap:32px;padding:16px 0}.lip-save-options label{display:flex;gap:8px;align-items:center}.lip-page button,.lip-page input,.lip-page select{font:inherit}.lip-page input[type=file]{max-width:100%}
@container lip-face (max-width:480px){.lip-metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 10px}.lip-curves{grid-template-columns:repeat(2,minmax(0,1fr));margin-top:8px}.lip-curves svg{height:30px}}
@container module (min-width:850px) and (max-width:1060px){
 .lip-page :deep(.module-workbench){grid-template-columns:minmax(180px,min(var(--panel-left,240px),240px)) minmax(0,1fr) minmax(220px,min(var(--panel-right,260px),260px));flex:1;min-height:0;align-items:stretch}
 .lip-page :deep(.module-workbench.right-collapsed){grid-template-columns:minmax(180px,min(var(--panel-left,240px),240px)) minmax(0,1fr) 42px}
 .lip-page :deep(.workbench-left),.lip-page :deep(.workbench-center){height:100%;overflow:auto}
 .lip-page :deep(.workbench-right){grid-column:auto;border:1px solid var(--border);padding:8px}
 .lip-page :deep(.workbench-right-body){max-height:none}
 .lip-face-section{min-height:400px}
}
@container module (max-width:849px){.lip-face-section{height:560px;flex:none!important}}
</style>
