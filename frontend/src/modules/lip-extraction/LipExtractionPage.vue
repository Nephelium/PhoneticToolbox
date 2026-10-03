<script setup lang="ts">
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import {computed,onMounted,onUnmounted,ref,shallowRef,watch,nextTick,markRaw} from 'vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleSection from '../../components/ModuleSection.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import {LipCapture,type CaptureSettings} from './capture.ts';
import {drawLegacyOverlay,faceBounds,fitFace} from './overlay.ts';
import {isLoopback} from './audio-input.ts';
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
const devices=ref<MediaDeviceInfo[]>([]),camera=ref(''),microphone=ref(''),mode=ref<CaptureMode>('preview'),filter=ref(true),cutoff=ref(15),fps=ref(30),delegate=ref<'CPU'|'GPU'>('CPU'),mirror=ref(true);
const error=ref(''),notice=ref(''),help=ref(false),files=ref<File[]>([]),results=ref<LipResult[]>([]),selection=ref(0),offset=ref(0),busy=ref(false),progress=ref(''),replaying=ref(false),replayIndex=ref(0),quality=ref<'high'|'standard'|'small'>('standard');
const localSaved=ref(false),confirmDiscard=ref(false),resultDirty=ref(false);
const saveDialog=ref(false),saveVideo=ref(true),saveAnimation=ref(true),offsetDialog=ref(false),offsetPreview=ref<number|null>(null),alignmentError=ref(''),alignmentLoading=ref(false),waveform=shallowRef<AlignmentWaveform|null>(null),inspection=shallowRef<Record<string,any>>(),captureOffset=ref(0);
let saveDecision:((value:boolean)=>void)|undefined,canvasObserver:ResizeObserver|undefined;
const effectiveOffset=computed(()=>offsetPreview.value??offset.value);
const recordingUrl=ref('');
function clearRecordingUrl(){if(recordingUrl.value)URL.revokeObjectURL(recordingUrl.value);recordingUrl.value='';}
const mediaFailure=ref(false),offsets=ref<Record<string,number>>({});let restoringOffset=false;
const replayVideo=ref<HTMLVideoElement|null>(null),sourceUrls=ref<Record<string,string>>({});const unsaved=new Set<string>();
let capture:LipCapture|undefined,abort:AbortController|undefined,replayTimer=0,replayFrameCallback=0,replayRequest=0;
let replayStore:LipReplay|undefined,disposed=false;
const replayRow=shallowRef<LipRow>(),exportInfo=ref<Record<string,any>>();
const cloudOptIn=ref(false),history=ref<{id:string;name:string}[]>([]),historyId=ref('');
async function refreshHistory(){try{history.value=await port.value.history?.()??[];historyId.value=history.value[0]?.id??'';}catch(e){error.value=(e as Error).message;}}
async function loadHistory(){if(!historyId.value||!port.value.load||busy.value||recording.value)return;const old=results.value.findIndex(r=>r.id===historyId.value);if(old>=0){selection.value=old;return;}if(results.value.length>=100){error.value='本页最多保留 100 项结果，请保存后重开页面；任务记录仍保留。';return;}busy.value=true;try{const result=await port.value.load(historyId.value);if(disposed)return;results.value.forEach(r=>r.rows=[]);result.rows=markRaw(result.rows);results.value.push(result);selection.value=results.value.length-1;}catch(e){error.value=(e as Error).message;}finally{busy.value=false;}}
const port=computed(()=>props.port??unavailableLipPort),state=computed(()=>{revision.value;return capture?{...capture.state,stats:{...capture.state.stats}}:undefined;}),recording=computed(()=>!!state.value&&['opening','previewing','recording','stopping','finalizing'].includes(state.value.phase));
const canAnalyze=computed(()=>port.value.available&&(port.value.kind==='desktop'||cloudOptIn.value));
const selected=computed(()=>results.value[selection.value]);
const recordedRows=computed<LipRow[]>(()=>{revision.value;const anchor=inspection.value?.audio?.first_decoded_pts_s??0;return (capture?.frames??[]).map(r=>({...r,time_s:r.time_s-anchor}));});
const rows=computed<LipRow[]>(()=>selected.value?.rows??recordedRows.value);
const shown=computed(()=>selected.value||recordingUrl.value?replayRow.value:state.value?.latest);
const animationBounds=computed(()=>recording.value?null:faceBounds(rows.value));
const playbackUrl=computed(()=>selected.value?sourceUrls.value[selected.value.id]??'':recordingUrl.value);
const recordedMode=computed(()=>{revision.value;return capture?.metadata().settings?.mode??mode.value;});
const phaseText=computed(()=>({idle:'未开始',opening:'准备中',previewing:'预览中',recording:'录制中',stopping:'正在结束',finalizing:'正在收尾',ready:'已结束',failed:'采集失败'}[state.value?.phase??'idle']));
const canSaveRecording=computed(()=>{revision.value;return !busy.value&&['ready','failed'].includes(state.value?.phase??'')&&!!capture?.chunks.length;});
const inputWarning=computed(()=>{const audio=state.value?.audio;if(mode.value==='preview'&&!audio?.label)return '';if(isLoopback(audio?.label||devices.value.find(d=>d.deviceId===microphone.value)?.label||''))return '当前输入为立体声混音/系统回放。录人声请停止采集并选择实际麦克风。';if(audio?.muted)return '麦克风音轨当前没有送入数据，请检查设备静音和权限。';if(audio?.lowSignal||exportInfo.value?.audio?.signal?.low_signal)return '输入信号很低（峰值低于 −60 dBFS），请对着麦克风说话并检查设备、静音和输入音量。';return audio?.monitorError??'';});
const frameCount=computed(()=>selected.value?.metadata.timing?.decoded_frames??rows.value.length);

const curveKeys=['area','outer_width','open','circularity'];
const curves=computed(()=>Object.fromEntries(curveKeys.map(key=>[key,curve(key)])));
const dirty=computed(()=>{revision.value;return !!capture?.state.dirty||resultDirty.value||recording.value||busy.value;});
watch(dirty,value=>emit('dirty',value));
watch(recording,value=>{if(!value)releaseCapture('M05');});
onUnmounted(()=>releaseCapture('M05'));
watch(selected,async result=>{mediaFailure.value=false;stopReplay();replayStore=undefined;replayRow.value=undefined;replayIndex.value=0;restoringOffset=true;offset.value=result?(offsets.value[result.id]??result.metadata.timing?.lip_manual_offset??0):captureOffset.value;restoringOffset=false;if(!result)return;
 try{if(!result.rows.length&&port.value.load){const loaded=await port.value.load(result.id);if(disposed||selected.value?.id!==result.id)return;results.value.forEach(r=>r.rows=[]);result.rows=markRaw(loaded.rows);}if(selected.value?.id!==result.id)return;replayStore=new LipReplay(result,port.value);await seekFrame(0,false);}catch(e){error.value=(e as Error).message;}});
watch(offset,()=>{if(restoringOffset)return;if(selected.value){offsets.value[selected.value.id]=offset.value;unsaved.add(selected.value.id);resultDirty.value=true;}else if(recordingUrl.value&&capture){captureOffset.value=offset.value;capture.state.dirty=true;update();}syncReplay();},{flush:'sync'});
watch(effectiveOffset,()=>syncReplay());
watch(()=>props.active,active=>{if(active===false){stopReplay();capture?.hidden();}});
watch(shown,()=>draw(),{deep:false});
watch(mirror,()=>draw());
function update(){revision.value++;if(!selected.value)draw();}
async function refresh(){try{if(!navigator.mediaDevices?.enumerateDevices)throw Error('此环境没有媒体设备 API，请检查 HTTPS 或桌面采集能力');devices.value=await navigator.mediaDevices.enumerateDevices();notice.value='设备列表已刷新；现有设备选择保留。';}catch(e){error.value=(e as Error).message;}}
async function start(){if(recording.value||busy.value)return;error.value='';notice.value='';stopReplay();if(dirty.value&&!await save())return;clearRecordingUrl();results.value=[];selection.value=0;Object.values(sourceUrls.value).forEach(URL.revokeObjectURL);sourceUrls.value={};files.value=[];offsets.value={};exportInfo.value=undefined;localSaved.value=false;inspection.value=undefined;waveform.value=null;captureOffset.value=0;offset.value=0;replayRow.value=undefined;replayStore=undefined;
 try{claimCapture('M05');await port.value.prepareCapture?.();await capture!.start({camera:camera.value,microphone:microphone.value,mode:mode.value,filter:filter.value,cutoff:cutoff.value,delegate:delegate.value,requestedFps:fps.value,preferMp4:!port.value.saveRecording} as CaptureSettings);await refresh();}catch(e){if(!recording.value)releaseCapture('M05');const message=(e as Error).message;error.value=await port.value.captureError?.(message).catch(()=>message)??message;}}
async function stop(){try{await capture?.stop();if(capture?.chunks.length){
 clearRecordingUrl();recordingUrl.value=URL.createObjectURL(capture.blob());
 busy.value=true;alignmentError.value='';progress.value='正在核对音频时间轴…';
 try{if(port.value.inspectRecording){inspection.value=await port.value.inspectRecording(capture.blob());waveform.value=inspection.value.waveform;}else await decodeWaveform(capture.blob());}catch(e){alignmentError.value=(e as Error).message;notice.value='音频时间轴核对失败，录制仍保留，可重试检查偏移量。';}finally{busy.value=false;progress.value='';}
 setupCaptureReplay();if(!alignmentError.value)notice.value=recordedMode.value==='record_then_analyze'?'录制已结束，另存视频与音频后可在离线分析中计算唇形。':'录制已结束，可同步回放视频与动画，或另存录制。';
 }else notice.value='预览已结束，未保存视频、音频或唇形记录。';}catch(e){error.value=(e as Error).message;}}
function setupCaptureReplay(){if(!rows.value.length)return;replayStore=new LipReplay({id:'capture',name:'本次录制',backend:'candidate',rows:recordedRows.value,metadata:{},files:[]},{...port.value,replay:undefined});void seekFrame(0,false);}
async function decodeWaveform(blob:Blob){const context=new AudioContext();try{const buffer=await context.decodeAudioData(await blob.arrayBuffer());if(buffer.length>60000000)throw Error('音频超出波形预览预算。');waveform.value=waveformEnvelope(buffer);}finally{await context.close();}}
async function openOffset(){stopReplay();offsetDialog.value=true;offsetPreview.value=offset.value;alignmentError.value='';alignmentLoading.value=true;
 try{if(selected.value){waveform.value=null;if(!port.value.audioPreview)throw Error('当前宿主不支持读取结果音频。');const data=await port.value.audioPreview(selected.value);waveform.value=data.waveform;}
 else if(!inspection.value&&port.value.inspectRecording){inspection.value=await port.value.inspectRecording(capture!.blob());waveform.value=inspection.value.waveform;setupCaptureReplay();}
 else if(!waveform.value)await decodeWaveform(capture!.blob());}
 catch(e){alignmentError.value=(e as Error).message;}finally{alignmentLoading.value=false;}}
function closeOffset(){offsetDialog.value=false;offsetPreview.value=null;}
function applyOffset(value:number){offset.value=value;closeOffset();notice.value='偏移量已应用到回放；请另存录制或保存结果以写入文件。';}

function choose(event:Event){files.value=Array.from((event.target as HTMLInputElement).files??[]);if(files.value.length>100){files.value=[];error.value='每批最多 100 个视频';}}
async function analyze(){if(!files.value.length||recording.value||busy.value||!canAnalyze.value)return;if(results.value.length+files.value.length>100){error.value='本页最多处理 100 项结果，请分批操作；已完成任务仍保留。';return;}error.value='';busy.value=true;abort=new AbortController();
 try{stopReplay();for(const file of files.value){progress.value=`准备 ${file.name}`;const result=await port.value.analyze(file,{filter_enabled:filter.value,cutoff_hz:cutoff.value},abort.signal,text=>progress.value=`${file.name}：${text}`);if(disposed)return;results.value.forEach(r=>r.rows=[]);result.rows=markRaw(result.rows);results.value.push(result);sourceUrls.value[result.id]=URL.createObjectURL(file);selection.value=results.value.length-1;unsaved.add(result.id);resultDirty.value=true;}notice.value=`已完成 ${files.value.length} 个文件，结果等待保存。`;}
 catch(e){error.value=(e as Error).message;}finally{busy.value=false;progress.value='';}}
function captureMetadata(){const meta=capture!.metadata();return new Blob([JSON.stringify({...meta,resources:meta.inference_requested?resources.files:null,lip_manual_offset:captureOffset.value,inspection:inspection.value?{audio:inspection.value.audio,first_video_pts_s:inspection.value.first_video_pts_s}:null,save_options:recordedMode.value==='record_then_analyze'?{video:true,animation:false}:{video:saveVideo.value,animation:saveAnimation.value},frames:capture!.frames})],{type:'application/json'});}
function requestSave():Promise<boolean>{if(saveDialog.value)return Promise.resolve(false);if(busy.value||!canSaveRecording.value)return Promise.resolve(false);if(recordedMode.value==='record_then_analyze')return saveRecording();saveVideo.value=saveAnimation.value=true;saveDialog.value=true;return new Promise(resolve=>{saveDecision=resolve;});}
function cancelSave(){saveDialog.value=false;saveDecision?.(false);saveDecision=undefined;}
async function confirmSave(){const ok=await saveRecording();if(ok){saveDialog.value=false;saveDecision?.(true);saveDecision=undefined;}}
async function saveRecording(){if(busy.value)return false;stopReplay();error.value='';busy.value=true;try{const blob=capture!.blob();
 if(!port.value.saveRecording)throw Error('当前浏览器宿主没有本地录制导出能力，请从桌面工作台保存；录制仍保留。');
 progress.value='正在保存所选媒体、音频与唇形数据…';const saved=await port.value.saveRecording('唇形原始录制.'+(blob.type.includes('mp4')?'mp4':'webm'),blob,captureMetadata());
 if(saved.saved){localSaved.value=true;exportInfo.value=saved.recording;capture?.markSaved();notice.value='所选录制文件已保存并核对。'+(saved.recording?.video_timing_warning??'');return true;}
 notice.value='已取消保存，录制保留。';return false;
 }catch(e){error.value='保存失败：'+(e as Error).message;return false;}finally{busy.value=false;progress.value='';}}
async function saveResult(action:'apply'|'save_without_offset'){if(busy.value)return false;stopReplay();error.value='';const result=selected.value;if(!result)return false;const value=action==='apply'?offset.value:0;busy.value=true;
 try{const ok=await port.value.save(result,value,action);if(ok){unsaved.delete(result.id);resultDirty.value=unsaved.size>0;notice.value='完整结果与偏移写入完成。';}return ok;}catch(e){error.value=(e as Error).message;return false;}finally{busy.value=false;}}

async function save(){if(recording.value||busy.value){error.value='采集或分析尚未结束，请先停止并等待收尾。';return false;}for(let i=0;i<results.value.length;i++)if(unsaved.has(results.value[i].id)){selection.value=i;await nextTick();if(!await saveResult('apply'))return false;}if(capture?.state.dirty&&!await requestSave())return false;return !dirty.value;}
defineExpose({save,dispose:()=>capture?.dispose()});
function discard(){stopReplay();clearRecordingUrl();capture?.discard();results.value=[];unsaved.clear();Object.values(sourceUrls.value).forEach(URL.revokeObjectURL);sourceUrls.value={};resultDirty.value=false;confirmDiscard.value=false;inspection.value=undefined;waveform.value=null;exportInfo.value=undefined;replayRow.value=undefined;replayStore=undefined;notice.value='已放弃本页未保存内容，磁盘原文件未改动。';}
function cancelReplayClock(){if(replayTimer)cancelAnimationFrame(replayTimer);if(replayFrameCallback)replayVideo.value?.cancelVideoFrameCallback(replayFrameCallback);replayTimer=replayFrameCallback=0;}
function stopReplay(){replaying.value=false;cancelReplayClock();replayVideo.value?.pause();replayRequest++;}
function replayOrigin(){return selected.value?-(selected.value.metadata.timing?.anchor_s??0):-(inspection.value?.audio?.first_decoded_pts_s??0);}
async function seekFrame(index:number,moveVideo=true){
 const store=replayStore,request=++replayRequest;if(!store)return;
 try{const row=await store.frame(Number(index));if(disposed||request!==replayRequest||store!==replayStore||!row)return;replayRow.value=row;replayIndex.value=row.index;store.prefetch(row.index);if(moveVideo&&replayVideo.value)replayVideo.value.currentTime=Math.max(0,row.time_s-replayOrigin()+effectiveOffset.value);draw();}catch(e){if(request===replayRequest)error.value=(e as Error).message;}
}
async function showTime(t:number){
 const store=replayStore,request=++replayRequest;if(!store)return;
 try{const row=await store.time(t);if(disposed||request!==replayRequest||store!==replayStore||!row)return;replayRow.value=t<(rows.value[0]?.time_s??0)||t>(rows.value.at(-1)?.time_s??0)+.15?{...row,detected:false,points:null,metrics:null}:row;replayIndex.value=row.index;store.prefetch(row.index);}catch(e){if(request===replayRequest){stopReplay();error.value=(e as Error).message;}}
}
async function play(){if(!rows.value.length)return;if(replayIndex.value>=frameCount.value-1)await seekFrame(0);const media=replayVideo.value;
 if(!mediaFailure.value&&playbackUrl.value&&media){try{await media.play();}catch(e){error.value=(e as Error).message;stopReplay();}return;}
 replaying.value=true;const first=performance.now(),origin=replayRow.value?.time_s??rows.value[0].time_s;
 const tick=()=>{if(!replaying.value)return;const now=origin+(performance.now()-first)/1000;void showTime(now);if(now>=(rows.value.at(-1)?.time_s??0)){replaying.value=false;return;}replayTimer=requestAnimationFrame(tick);};tick();
}
function mediaPlaying(){cancelReplayClock();replaying.value=true;const media=replayVideo.value;if(!media)return;
 const tick=(_now:number,meta:VideoFrameCallbackMetadata)=>{if(media!==replayVideo.value||media.paused)return;void showTime(meta.mediaTime+replayOrigin()-effectiveOffset.value);replayFrameCallback=media.requestVideoFrameCallback(tick);};
 if(media.requestVideoFrameCallback)replayFrameCallback=media.requestVideoFrameCallback(tick);
 else{const fallback=()=>{if(media.paused)return;syncReplay();replayTimer=requestAnimationFrame(fallback);};fallback();}
 syncReplay();
}
function mediaPaused(){replaying.value=false;cancelReplayClock();syncReplay();}
function syncReplay(){if(replayVideo.value)void showTime(replayVideo.value.currentTime+replayOrigin()-effectiveOffset.value);}
function draw(){const el=canvas.value;if(!el)return;const row=shown.value;const box=el.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 const width=Math.max(2,Math.round(box.width*dpr)),height=Math.max(2,Math.round(box.height*dpr));if(el.width!==width)el.width=width;if(el.height!==height)el.height=height;
 const context=el.getContext('2d');if(!context)return;context.clearRect(0,0,width,height);
 if(row?.points)drawLegacyOverlay(context,fitFace(row.points,width,height,animationBounds.value??undefined,.94),width,height);}

function curve(key:string){const data=selected.value?selected.value.rows:rows.value.slice(-300);if(!data.length)return '';const values=data.map(r=>r.metrics?.[key]).filter((x):x is number=>typeof x==='number'&&Number.isFinite(x));if(!values.length)return '';const low=Math.min(...values),high=Math.max(...values),start=data[0].time_s,end=data.at(-1)!.time_s;let path='',connected=false;for(const r of data){const value=r.metrics?.[key];if(typeof value!=='number'||!Number.isFinite(value)){connected=false;continue;}const x=10+(r.time_s-start)/Math.max(.001,end-start)*580,y=110-(value-low)/Math.max(1e-9,high-low)*90;path+=`${connected?'L':'M'}${x.toFixed(2)},${y.toFixed(2)} `;connected=true;}return path;}

async function exportAnimation(format:'mp4'|'gif'){if(!selected.value||!port.value.exportAnimation)return;error.value='';busy.value=true;abort=new AbortController();try{const ok=await port.value.exportAnimation({...selected.value,metadata:{...selected.value.metadata,timing:{...selected.value.metadata.timing,lip_manual_offset:offset.value}}},format,quality.value,abort.signal);notice.value=ok?'动画写入完成。':'动画导出已取消。';}catch(e){error.value=(e as Error).message;}finally{busy.value=false;}}
function beforeUnload(event:BeforeUnloadEvent){if(dirty.value){event.preventDefault();event.returnValue='';}}
function hidden(){if(document.hidden)capture?.hidden();}
onMounted(()=>{capture=new LipCapture(video.value!,update);canvasObserver=new ResizeObserver(()=>draw());if(canvas.value)canvasObserver.observe(canvas.value);void refresh();window.addEventListener('beforeunload',beforeUnload);document.addEventListener('visibilitychange',hidden);});
onUnmounted(()=>{disposed=true;canvasObserver?.disconnect();saveDecision?.(false);clearRecordingUrl();Object.values(sourceUrls.value).forEach(URL.revokeObjectURL);abort?.abort();stopReplay();capture?.dispose();window.removeEventListener('beforeunload',beforeUnload);document.removeEventListener('visibilitychange',hidden);});
</script>
<template>
<ModuleFrame unified fit label="唇形提取工作区" class="lip-page" :aria-busy="busy">
 <template #toolbar><ModuleToolbar>
  <label>模式 <select v-model="mode" aria-label="采集模式" :disabled="recording||busy"><option value="preview">实时预览</option><option value="realtime">实时录制</option><option value="record_then_analyze">高帧率先录后算</option></select></label>
  <button class="primary" :disabled="recording||busy" @click="start">{{dirty?'保存并开始下一次':mode==='preview'?'打开预览':'开始录制'}}</button>
  <button class="primary" :disabled="!state||!['opening','previewing','recording'].includes(state.phase)" @click="stop">结束录制</button>
  <button class="primary" :disabled="!canSaveRecording" @click="requestSave">另存录制</button>
  <button :disabled="recording||busy||!dirty" @click="confirmDiscard=true">放弃未保存内容</button>
  <button :disabled="recording||busy||!rows.length||(!selected&&!recordingUrl)" @click="openOffset">检查偏移量</button>
  <template #actions><button @click="help=!help">使用说明</button><button @click="emit('references')">方法与引用</button></template>
 </ModuleToolbar></template>
 <template #status><ModuleStatus v-if="error||state?.error" kind="error" :message="error||state?.error||''"/><ModuleStatus v-if="notice||progress" kind="info" :message="progress||notice"/></template>
 <ModuleWorkbench unified :state-key="stateKey??'M05'" left-label="设备与采集" right-label="视频与记录">
 <template #left>
 <ModuleSection label="媒体设备与参数" title="设备与参数"><div class="lip-fields">
  <label>摄像头 <select v-model="camera" :disabled="recording||busy"><option value="">系统默认</option><option v-if="camera&&!devices.some(d=>d.kind==='videoinput'&&d.deviceId===camera)" :value="camera">所选摄像头已不可用</option><option v-for="d in devices.filter(d=>d.kind==='videoinput')" :key="d.deviceId" :value="d.deviceId">{{d.label||'未授权摄像头'}}</option></select></label>
  <label>麦克风 <select v-model="microphone" aria-label="麦克风" :disabled="recording||busy||mode==='preview'"><option value="">自动选择麦克风</option><option v-if="microphone&&!devices.some(d=>d.kind==='audioinput'&&d.deviceId===microphone)" :value="microphone">所选麦克风已不可用</option><option v-for="d in devices.filter(d=>d.kind==='audioinput')" :key="d.deviceId" :value="d.deviceId">{{d.label||'未授权麦克风'}}</option></select></label>
  <div v-if="state?.audio.label" class="lip-input-level"><span>实际输入：{{state.audio.label}}</span><meter aria-label="麦克风输入电平" min="-120" max="0" :value="state.audio.peakDb??-120"/><span>{{state.audio.peakDb===null?'电平尚未就绪':state.audio.peakDb.toFixed(1)+' dBFS'}}{{recording?'':'（停止前）'}}</span></div>
  <ModuleStatus v-if="inputWarning" class="lip-input-warning" kind="info" :message="inputWarning"/>
  <button :disabled="recording||busy" @click="refresh">刷新设备</button>
  <label>请求帧率 <span><input v-model.number="fps" type="number" min="1" max="240" :disabled="recording||busy"/> fps</span></label>
  <label><input v-model="filter" type="checkbox" :disabled="recording||busy"/>特征点防抖</label>
  <label>截止频率 <span><input v-model.number="cutoff" type="number" min="1" max="240" :disabled="recording||busy||!filter"/> Hz</span></label>
  <label>推理设备 <select v-model="delegate" :disabled="recording||busy"><option>CPU</option><option>GPU</option></select></label>
  <label><input v-model="mirror" type="checkbox"/>镜像预览</label>
 </div></ModuleSection>
 <ModuleSection label="采集与保存状态" title="采集状态"><div class="lip-state" :data-phase="state?.phase??'idle'">
  <span class="lip-state-wide">阶段 <strong>{{phaseText}}</strong></span><span>呈现 {{state?.stats.presented??0}}</span><span>已推理 {{state?.stats.processed??0}}</span><span>检出 {{state?.stats.detected??0}}</span><span>跳过推理 {{state?.stats.skippedInference??0}}</span><span>呈现缺口 {{state?.stats.observedPresentationGaps??0}}</span><span>编码 {{((state?.stats.encodedBytes??0)/1e6).toFixed(1)}} MB</span>
  <span class="lip-state-wide">设备 {{state?.trackSettings?.width??'—'}}×{{state?.trackSettings?.height??'—'}} · {{state?.trackSettings?.frameRate??'—'}} fps</span>
  <span v-if="exportInfo" class="lip-state-wide">文件 {{exportInfo.video_frames}} 帧 · {{exportInfo.observed_video_fps?.toFixed(1)??'—'}} fps · {{exportInfo.audio?.present?'含音频':'无音频'}}</span>
  <span v-if="state?.stats.processed" class="lip-state-wide">实时 {{state.stats.processed}} 帧 · {{capture?.metadata().overlay_display_fps?.toFixed(1)??'—'}} fps</span>
 </div>
 <p class="lip-save-note">{{mode==='preview'?'实时预览不录制音视频，也不保存唇形数据。':mode==='record_then_analyze'?'保存视频和音频，唇形在离线分析中另行计算。':'另存时选择视频和动画，音频与唇形数据始终保存。'}}</p>
 <p v-if="exportInfo" class="lip-save-note">已保存：{{exportInfo.files?.map((f:any)=>f.name).join('、')}}。</p>
 </ModuleSection>
 </template>
 <ModuleSection class="lip-face-section" label="视频与唇形画面">
  <div class="lip-video" :class="{mirrored:mirror&&recording}"><canvas ref="canvas" width="960" height="540"/><div v-if="!shown?.points" class="lip-face-empty">{{recordedMode==='record_then_analyze'&&recording?'此模式只录制媒体，唇形稍后计算':shown?'此帧未检出面部':'打开预览、录制或选择视频后显示面部特征点'}}</div></div>
  <div class="lip-metrics"><span v-for="key in ['area','height','outer_width','inner_width','total_width','open','circularity','face_width','face_height']" :key="key"><span>{{key}}</span><strong>{{shown?.metrics?.[key]?.toFixed(4)??'—'}}</strong></span></div>
  <div class="lip-curves"><figure v-for="key in curveKeys" :key="key"><figcaption>{{key}}</figcaption><svg viewBox="0 0 600 125" role="img" :aria-label="key+' 参数曲线'"><path :d="curves[key]" fill="none" stroke="currentColor" stroke-width="1.5"/></svg></figure></div>
  <p class="lip-curve-note">横轴为实际时间，各曲线独立纵轴<span v-if="!recording&&rows.length"> · 唇形偏移 {{Math.round(offset*1000)}} ms</span></p>
 </ModuleSection>
 <template #right>
 <ModuleSection class="lip-media-section" label="视频回放" title="视频与同步回放">
  <video ref="video" class="lip-live-video" playsinline muted :class="{mirrored:mirror}" v-show="!playbackUrl" aria-label="摄像头实时画面"/>
  <video v-if="playbackUrl" ref="replayVideo" class="lip-recording-preview" :src="playbackUrl" controls preload="metadata" aria-label="视频与动画同步回放" @error="mediaFailure=true;notice='当前浏览器无法播放此编码，请保存后使用本机播放器核对。'" @play="mediaPlaying" @pause="mediaPaused" @seeked="syncReplay" @ended="mediaPaused"/>
  <div v-if="rows.length&&!recording" class="lip-replay-controls"><button @click="replaying?stopReplay():play()">{{replaying?'暂停回放':'同步回放'}}</button><span>{{(replayRow?.time_s??0).toFixed(3)}} s</span><input v-model.number="replayIndex" type="range" min="0" :max="Math.max(0,frameCount-1)" aria-label="回放帧" @input="stopReplay();seekFrame(replayIndex)"/></div>
 </ModuleSection>
 <div v-if="help" class="lip-help">实时唇形使用 Face Landmarker，与 V2 FaceMesh 尚未通过等价验证。镜像和面部放大只改变显示。高帧率是设备请求值，以实际文件为准。录制暂存上限 128 MB，实时参数上限 32 MB。编码起点估计、设备延迟和滤波滞后均可能造成偏移，请结合已知同步事件检查。偏移调整只平移唇形，不改音频或原视频。</div>
 <ModuleSection label="已有视频与批量分析" title="离线分析"><div class="lip-actions"><input type="file" accept=".mp4,.mov,.avi,.mkv,.wmv,.m4v,.webm" multiple :disabled="busy||recording" aria-label="选择离线视频" @change="choose"/><button class="primary" :disabled="busy||recording||!files.length||!canAnalyze" @click="analyze">分析所选 {{files.length}} 个视频</button><button v-if="busy&&abort" @click="abort?.abort()">取消本次任务</button></div><label v-if="port.kind==='browser'&&port.available"><input v-model="cloudOptIn" type="checkbox"/>上传所选视频至当前账号进行离线分析</label><ModuleStatus v-if="!port.available" kind="info" :message="port.reason??'当前平台离线分析尚未开放'"/><ul v-if="files.length"><li v-for="file in files" :key="file.name">{{file.name}} · {{(file.size/1e6).toFixed(1)}} MB</li></ul></ModuleSection>
 <ModuleSection v-if="port.history" label="本地历史任务" title="历史结果"><div class="lip-actions"><button :disabled="busy||recording" @click="refreshHistory">刷新本地历史任务</button><select v-model="historyId" aria-label="历史唇形任务"><option v-for="job in history" :key="job.id" :value="job.id">{{job.name}}</option></select><button :disabled="busy||recording||!historyId" @click="loadHistory">读取历史结果</button></div></ModuleSection>
 <ModuleSection v-if="results.length" label="结果、偏移与动画回放" title="结果保存"><select v-model.number="selection" :disabled="busy||recording" aria-label="选择唇形结果"><option v-for="(result,i) in results" :key="result.id" :value="i">{{result.name}}</option></select><p class="lip-save-note">{{selected?.backend}} · 完整 {{frameCount}} 帧，曲线预览 {{rows.length}} 点。检测缺失留空。</p>
  <div class="lip-actions"><button class="primary" :disabled="busy" @click="saveResult('apply')">应用偏移并保存</button><button class="primary" :disabled="busy" @click="saveResult('save_without_offset')">保存但不应用偏移</button></div>
  <div class="lip-actions"><select v-model="quality" aria-label="动画导出质量"><option value="high">高清 1080</option><option value="standard">标准 720</option><option value="small">小体积 540</option></select><button class="primary" :disabled="busy||!port.exportAnimation" @click="exportAnimation('mp4')">导出动画 MP4</button><button class="primary" :disabled="busy||!port.exportAnimation" @click="exportAnimation('gif')">导出 GIF</button></div>
 </ModuleSection>
 </template>
 </ModuleWorkbench>
 <ModalDialog v-if="saveDialog" title="另存录制" :close-disabled="busy" @close="cancelSave"><p>音频与唇形数据始终保存，选择需要同时保存的画面。</p><div class="lip-save-options"><label><input v-model="saveVideo" type="checkbox" :disabled="busy"/>视频</label><label><input v-model="saveAnimation" type="checkbox" :disabled="busy"/>动画</label></div><p class="lip-save-note">视频和动画均为 MP4。两项都取消时，只保存音频、唇形数据及其时间信息。</p><p v-if="error" role="alert">{{error}}</p><p v-if="busy" role="status">{{progress}}</p><template #footer><button :disabled="busy" @click="cancelSave">取消</button><button class="primary" :disabled="busy" @click="confirmSave">选择目录并保存</button></template></ModalDialog>
 <OffsetDialog v-if="offsetDialog" :rows="rows" :waveform="waveform" :offset="offset" :loading="alignmentLoading" :error="alignmentError" @preview="offsetPreview=$event" @apply="applyOffset" @close="closeOffset"/>
 <ModalDialog v-if="confirmDiscard" title="放弃未保存内容" @close="confirmDiscard=false"><p>放弃本页尚未保存的录制和结果？</p><template #footer><button @click="confirmDiscard=false">取消</button><button @click="discard">放弃</button></template></ModalDialog>
</ModuleFrame>
</template>
<style scoped>
.lip-input-level{display:grid;gap:4px;overflow-wrap:anywhere}.lip-input-level meter{width:100%}.lip-input-warning{color:var(--warning,var(--accent))}.lip-save-note{font-size:var(--support-size);overflow-wrap:anywhere;color:var(--muted);margin:8px 0 0}
.lip-fields{display:grid;gap:8px}.lip-fields>label{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}.lip-fields>label:has(input[type=checkbox]){justify-content:flex-start}.lip-fields select{flex:1;min-width:90px;max-width:100%}.lip-fields input[type=number]{width:78px}.lip-fields label>span{white-space:nowrap}
.lip-face-section{container:lip-face / inline-size;display:flex;flex-direction:column;flex:1!important;min-height:440px;overflow:hidden}.lip-video{position:relative;width:100%;flex:1;min-height:180px;background:var(--panel);overflow:hidden}.lip-video canvas{position:absolute;inset:0;width:100%;height:100%}.lip-video.mirrored canvas,video.mirrored{transform:scaleX(-1)}.lip-face-empty{position:absolute;inset:0;display:grid;place-items:center;text-align:center;padding:24px;color:var(--muted)}
.lip-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px 16px;font-family:var(--font-figure);font-size:var(--figure-size);padding-top:10px;flex:none}.lip-metrics>span{display:flex;justify-content:space-between;gap:8px;min-width:0;font-variant-numeric:tabular-nums}.lip-metrics strong{font-weight:400}.lip-metrics span span{overflow-wrap:anywhere}
.lip-curves{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:12px;flex:none}.lip-curves figure{margin:0;min-width:0}.lip-curves svg{display:block;width:100%;height:55px;color:var(--accent)}.lip-curves path{vector-effect:non-scaling-stroke}.lip-curves figcaption{font-size:var(--figure-size);font-family:var(--font-figure)}.lip-curve-note{margin:3px 0 0;color:var(--muted);font-size:var(--figure-size);flex:none}
.lip-state{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;font-variant-numeric:tabular-nums}.lip-state-wide{grid-column:1/-1}.lip-state span{overflow-wrap:anywhere}.lip-actions{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:8px}.lip-actions select{width:100%}.lip-help{padding:10px;border:1px solid var(--border);line-height:1.6;overflow-wrap:anywhere}
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
