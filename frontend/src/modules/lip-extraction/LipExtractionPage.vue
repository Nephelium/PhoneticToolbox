<script setup lang="ts">
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import {computed,onMounted,onUnmounted,ref,shallowRef,watch,nextTick,markRaw} from 'vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleSection from '../../components/ModuleSection.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import {LipCapture,type CaptureSettings} from './capture.ts';
import {drawLegacyOverlay} from './overlay.ts';
import type {CaptureMode} from './state.ts';
import type {LipPort,LipResult,LipRow} from './port.ts';
import {saveM05Local,unavailableLipPort} from '../../platform/m05.ts';
import resources from '../../../../resources/m05/resources.json';
import {LipReplay} from './replay.ts';
import {claimCapture,releaseCapture} from '../../platform/capture-lease.ts';
const props=defineProps<{stateKey?:string;active?:boolean;port?:LipPort}>();
const emit=defineEmits<{references:[];dirty:[boolean]}>();
const video=ref<HTMLVideoElement|null>(null),canvas=ref<HTMLCanvasElement|null>(null),revision=ref(0);
const devices=ref<MediaDeviceInfo[]>([]),camera=ref(''),microphone=ref(''),mode=ref<CaptureMode>('preview'),filter=ref(true),cutoff=ref(15),fps=ref(30),delegate=ref<'CPU'|'GPU'>('CPU'),mirror=ref(true);
const error=ref(''),notice=ref(''),help=ref(false),files=ref<File[]>([]),results=ref<LipResult[]>([]),selection=ref(0),offset=ref(0),busy=ref(false),progress=ref(''),replaying=ref(false),replayIndex=ref(0),quality=ref<'high'|'standard'|'small'>('standard');
const localSaved=ref(false),metadataSaved=ref(false),mediaRequested=ref(false),metadataRequested=ref(false),confirmDiscard=ref(false),resultDirty=ref(false);
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
const selected=computed(()=>results.value[selection.value]),rows=computed<LipRow[]>(()=>{revision.value;return selected.value?.rows??capture?.frames??[];});
const shown=computed(()=>selected.value?replayRow.value:state.value?.latest);
const frameCount=computed(()=>selected.value?.metadata.timing?.decoded_frames??rows.value.length);
const videoRatio=computed(()=>{const r:LipRow|undefined=shown.value??undefined,d=selected.value?.metadata.coordinates?.resolutions?.[0];return `${r?.width||r?.input_resolution?.[0]||d?.[0]||state.value?.trackSettings?.width||16} / ${r?.height||r?.input_resolution?.[1]||d?.[1]||state.value?.trackSettings?.height||9}`;});
const curveKeys=['area','outer_width','open','circularity'];
const curves=computed(()=>Object.fromEntries(curveKeys.map(key=>[key,curve(key)])));
const dirty=computed(()=>{revision.value;return !!capture?.state.dirty||resultDirty.value||recording.value||busy.value;});
watch(dirty,value=>emit('dirty',value));
watch(recording,value=>{if(!value)releaseCapture('M05');});
onUnmounted(()=>releaseCapture('M05'));
watch(selected,async result=>{mediaFailure.value=false;stopReplay();replayStore=undefined;replayRow.value=undefined;replayIndex.value=0;restoringOffset=true;offset.value=result?(offsets.value[result.id]??result.metadata.timing?.lip_manual_offset??0):0;restoringOffset=false;if(!result)return;
 try{if(!result.rows.length&&port.value.load){const loaded=await port.value.load(result.id);if(disposed||selected.value?.id!==result.id)return;results.value.forEach(r=>r.rows=[]);result.rows=markRaw(loaded.rows);}if(selected.value?.id!==result.id)return;replayStore=new LipReplay(result,port.value);await seekFrame(0,false);}catch(e){error.value=(e as Error).message;}});
watch(offset,()=>{if(selected.value&&!restoringOffset){offsets.value[selected.value.id]=offset.value;unsaved.add(selected.value.id);resultDirty.value=true;}},{flush:'sync'});
watch(()=>props.active,active=>{if(active===false){stopReplay();capture?.hidden();}});
watch(shown,()=>draw(),{deep:false});
watch(mirror,()=>draw());
function update(){revision.value++;if(!selected.value)draw();}
async function refresh(){try{if(!navigator.mediaDevices?.enumerateDevices)throw Error('此环境没有媒体设备 API，请检查 HTTPS 或桌面采集能力');devices.value=await navigator.mediaDevices.enumerateDevices();notice.value='设备列表已刷新；现有设备选择保留。';}catch(e){error.value=(e as Error).message;}}
async function start(){if(recording.value||busy.value)return;error.value='';notice.value='';stopReplay();if(dirty.value&&!await save())return;results.value=[];selection.value=0;Object.values(sourceUrls.value).forEach(URL.revokeObjectURL);sourceUrls.value={};files.value=[];offsets.value={};exportInfo.value=undefined;localSaved.value=metadataSaved.value=mediaRequested.value=metadataRequested.value=false;
 try{claimCapture('M05');await port.value.prepareCapture?.();await capture!.start({camera:camera.value,microphone:microphone.value,mode:mode.value,filter:filter.value,cutoff:cutoff.value,delegate:delegate.value,requestedFps:fps.value,preferMp4:!port.value.saveRecording} as CaptureSettings);await refresh();}catch(e){if(!recording.value)releaseCapture('M05');const message=(e as Error).message;error.value=await port.value.captureError?.(message).catch(()=>message)??message;}}
async function stop(){try{await capture?.stop();notice.value='采集已停止并收尾，录制仍需保存。';}catch(e){error.value=(e as Error).message;}}
function choose(event:Event){files.value=Array.from((event.target as HTMLInputElement).files??[]);if(files.value.length>100){files.value=[];error.value='每批最多 100 个视频';}}
async function analyze(){if(!files.value.length||recording.value||busy.value||!canAnalyze.value)return;if(results.value.length+files.value.length>100){error.value='本页最多处理 100 项结果，请分批操作；已完成任务仍保留。';return;}error.value='';busy.value=true;abort=new AbortController();
 try{stopReplay();for(const file of files.value){progress.value=`准备 ${file.name}`;const result=await port.value.analyze(file,{filter_enabled:filter.value,cutoff_hz:cutoff.value},abort.signal,text=>progress.value=`${file.name}：${text}`);if(disposed)return;results.value.forEach(r=>r.rows=[]);result.rows=markRaw(result.rows);results.value.push(result);sourceUrls.value[result.id]=URL.createObjectURL(file);selection.value=results.value.length-1;unsaved.add(result.id);resultDirty.value=true;}notice.value=`已完成 ${files.value.length} 个文件，结果等待保存。`;}
 catch(e){error.value=(e as Error).message;}finally{busy.value=false;progress.value='';}}
async function analyzeRecording(){if(busy.value||recording.value)return;try{const blob=capture!.blob();
 if(port.value.analyzeRecording){if(!exportInfo.value?.token){await saveRecording();if(!exportInfo.value?.token)return;}if(results.value.length>=100)throw Error('本页最多保留 100 项结果。');busy.value=true;abort=new AbortController();const result=await port.value.analyzeRecording(exportInfo.value.token,{filter_enabled:filter.value,cutoff_hz:cutoff.value},abort.signal,s=>progress.value=s);if(disposed)return;results.value.forEach(r=>r.rows=[]);result.rows=markRaw(result.rows);results.value.push(result);sourceUrls.value[result.id]=URL.createObjectURL(blob);selection.value=results.value.length-1;unsaved.add(result.id);resultDirty.value=true;notice.value='正式分析完成，结果等待保存。';}
 else{files.value=[new File([blob],'本地原始录制.'+(blob.type.includes('mp4')?'mp4':'webm'),{type:blob.type})];await analyze();}
 }catch(e){error.value=(e as Error).message;}finally{busy.value=false;progress.value='';}}
async function saveBlob(name:string,blob:Blob){if(port.value.saveLocal)return await port.value.saveLocal(name,blob)?'saved':'cancelled';return saveM05Local(name,blob);}
function captureMetadata(){const meta=capture!.metadata();return new Blob([JSON.stringify({...meta,resources:meta.inference_requested?resources.files:null,lip_manual_offset:0,frames:capture!.frames})],{type:'application/json'});}
async function saveRecording(){if(busy.value)return;stopReplay();error.value='';busy.value=true;try{const blob=capture!.blob();
 if(port.value.saveRecording){progress.value='正在保存 MP4、音频与采集记录…';const saved=await port.value.saveRecording('唇形原始录制.'+(blob.type.includes('mp4')?'mp4':'webm'),blob,captureMetadata());if(saved.saved){localSaved.value=metadataSaved.value=true;exportInfo.value=saved.recording;capture?.markSaved();notice.value='MP4、WAV 与采集记录已保存并核对。';}else notice.value='已取消保存，录制保留。';}
 else{if(!blob.type.includes('mp4'))throw Error('当前浏览器编码器不支持 MP4。请使用桌面版保存 MP4 与 WAV，当前录制仍保留。');const status=await saveBlob('唇形原始录制.mp4',blob);if(status==='saved')localSaved.value=true;else if(status==='download_requested')mediaRequested.value=true;notice.value=status==='saved'?'MP4 已保存，请继续保存候选记录。':status==='download_requested'?'MP4 下载已发起，请核对文件。':'已取消保存，录制保留。';if(localSaved.value&&metadataSaved.value)capture?.markSaved();}
 }catch(e){error.value='保存失败：'+(e as Error).message;}finally{busy.value=false;progress.value='';}}
async function saveMetadata(){if(busy.value)return;error.value='';busy.value=true;try{const status=await saveBlob('唇形候选记录.m05-preview.json',captureMetadata());if(status==='saved')metadataSaved.value=true;else if(status==='download_requested')metadataRequested.value=true;notice.value=status==='saved'?'候选参数与时间元数据已保存。':status==='download_requested'?'参数下载已发起，请核对文件。':'已取消保存。';if(localSaved.value&&metadataSaved.value)capture?.markSaved();}catch(e){error.value='保存失败：'+(e as Error).message;}finally{busy.value=false;}}
async function saveResult(action:'apply'|'save_without_offset'){if(busy.value)return false;stopReplay();error.value='';const result=selected.value;if(!result)return false;const value=action==='apply'?offset.value:0;busy.value=true;
 try{const ok=await port.value.save(result,value,action);if(ok){unsaved.delete(result.id);resultDirty.value=unsaved.size>0;notice.value='完整结果与偏移写入完成。';}return ok;}catch(e){error.value=(e as Error).message;return false;}finally{busy.value=false;}}

async function save(){if(recording.value||busy.value){error.value='采集或分析尚未结束，请先停止并等待收尾。';return false;}for(let i=0;i<results.value.length;i++)if(unsaved.has(results.value[i].id)){selection.value=i;await nextTick();if(!await saveResult('apply'))return false;}if(capture?.state.dirty){await saveRecording();if(capture.state.dirty&&localSaved.value&&!port.value.saveRecording)await saveMetadata();}return !dirty.value;}
defineExpose({save,dispose:()=>capture?.dispose()});
function discard(){stopReplay();capture?.discard();results.value=[];unsaved.clear();Object.values(sourceUrls.value).forEach(URL.revokeObjectURL);sourceUrls.value={};resultDirty.value=false;confirmDiscard.value=false;notice.value='已放弃本页未保存内容，磁盘原文件未改动。';}
function cancelReplayClock(){if(replayTimer)cancelAnimationFrame(replayTimer);if(replayFrameCallback)replayVideo.value?.cancelVideoFrameCallback(replayFrameCallback);replayTimer=replayFrameCallback=0;}
function stopReplay(){replaying.value=false;cancelReplayClock();replayVideo.value?.pause();replayRequest++;}
function replayOrigin(){return (selected.value?.metadata.timing?.first_video_pts_s??0)-(selected.value?.metadata.timing?.anchor_s??0);}
async function seekFrame(index:number,moveVideo=true){
 const store=replayStore,request=++replayRequest;if(!store)return;
 try{const row=await store.frame(Number(index));if(disposed||request!==replayRequest||store!==replayStore||!row)return;replayRow.value=row;replayIndex.value=row.index;store.prefetch(row.index);if(moveVideo&&replayVideo.value)replayVideo.value.currentTime=Math.max(0,row.time_s-replayOrigin()+offset.value);draw();}catch(e){if(request===replayRequest)error.value=(e as Error).message;}
}
async function showTime(t:number){
 const store=replayStore,request=++replayRequest;if(!store)return;
 try{const row=await store.time(t);if(disposed||request!==replayRequest||store!==replayStore||!row)return;replayRow.value=t<(rows.value[0]?.time_s??0)?{...row,detected:false,points:null,metrics:null}:row;replayIndex.value=row.index;store.prefetch(row.index);}catch(e){if(request===replayRequest){stopReplay();error.value=(e as Error).message;}}
}
async function play(){if(!rows.value.length)return;if(replayIndex.value>=frameCount.value-1)await seekFrame(0);const media=replayVideo.value;
 if(selected.value&&!mediaFailure.value&&sourceUrls.value[selected.value.id]&&media){try{await media.play();}catch(e){error.value=(e as Error).message;stopReplay();}return;}
 replaying.value=true;const first=performance.now(),origin=replayRow.value?.time_s??rows.value[0].time_s;
 const tick=()=>{if(!replaying.value)return;const now=origin+(performance.now()-first)/1000;void showTime(now);if(now>=(rows.value.at(-1)?.time_s??0)){replaying.value=false;return;}replayTimer=requestAnimationFrame(tick);};tick();
}
function mediaPlaying(){cancelReplayClock();replaying.value=true;const media=replayVideo.value;if(!media)return;
 const tick=(_now:number,meta:VideoFrameCallbackMetadata)=>{if(media!==replayVideo.value||media.paused)return;void showTime(meta.mediaTime+replayOrigin()-offset.value);replayFrameCallback=media.requestVideoFrameCallback(tick);};
 if(media.requestVideoFrameCallback)replayFrameCallback=media.requestVideoFrameCallback(tick);
 else{const fallback=()=>{if(media.paused)return;syncReplay();replayTimer=requestAnimationFrame(fallback);};fallback();}
 syncReplay();
}
function mediaPaused(){replaying.value=false;cancelReplayClock();syncReplay();}
function syncReplay(){if(selected.value&&replayVideo.value)void showTime(replayVideo.value.currentTime+replayOrigin()-offset.value);}
function draw(){const el=canvas.value;if(!el)return;
 const row:LipRow|undefined=selected.value?replayRow.value:capture?.state.latest??undefined;
 const width=row?.width||row?.input_resolution?.[0]||selected.value?.metadata.coordinates?.resolutions?.[0]?.[0]||video.value?.videoWidth||capture?.previewFrame?.width||640,height=row?.height||row?.input_resolution?.[1]||selected.value?.metadata.coordinates?.resolutions?.[0]?.[1]||video.value?.videoHeight||capture?.previewFrame?.height||480;
 if(el.width!==width)el.width=width;if(el.height!==height)el.height=height;
 const context=el.getContext('2d');if(!context)return;context.clearRect(0,0,width,height);
 if(!selected.value&&capture?.previewFrame)context.drawImage(capture.previewFrame,0,0,width,height);
 if(row?.points)drawLegacyOverlay(context,row.points,width,height);}
function curve(key:string){const data=selected.value?selected.value.rows:rows.value.slice(-300);if(!data.length)return '';const values=data.map(r=>r.metrics?.[key]).filter((x):x is number=>typeof x==='number'&&Number.isFinite(x));if(!values.length)return '';const low=Math.min(...values),high=Math.max(...values),start=data[0].time_s,end=data.at(-1)!.time_s;let path='',connected=false;for(const r of data){const value=r.metrics?.[key];if(typeof value!=='number'||!Number.isFinite(value)){connected=false;continue;}const x=10+(r.time_s-start)/Math.max(.001,end-start)*580,y=110-(value-low)/Math.max(1e-9,high-low)*90;path+=`${connected?'L':'M'}${x.toFixed(2)},${y.toFixed(2)} `;connected=true;}return path;}

async function exportAnimation(format:'mp4'|'gif'){if(!selected.value||!port.value.exportAnimation)return;error.value='';busy.value=true;abort=new AbortController();try{const ok=await port.value.exportAnimation({...selected.value,metadata:{...selected.value.metadata,timing:{...selected.value.metadata.timing,lip_manual_offset:offset.value}}},format,quality.value,abort.signal);notice.value=ok?'动画写入完成。':'动画导出已取消。';}catch(e){error.value=(e as Error).message;}finally{busy.value=false;}}
function beforeUnload(event:BeforeUnloadEvent){if(dirty.value){event.preventDefault();event.returnValue='';}}
function hidden(){if(document.hidden)capture?.hidden();}
onMounted(()=>{capture=new LipCapture(video.value!,update);void refresh();window.addEventListener('beforeunload',beforeUnload);document.addEventListener('visibilitychange',hidden);});
onUnmounted(()=>{disposed=true;Object.values(sourceUrls.value).forEach(URL.revokeObjectURL);abort?.abort();stopReplay();capture?.dispose();window.removeEventListener('beforeunload',beforeUnload);document.removeEventListener('visibilitychange',hidden);});
</script>
<template>
<ModuleFrame fit label="唇形提取工作区" class="lip-page" :aria-busy="busy">
 <template #status><ModuleStatus v-if="error||state?.error" kind="error" :message="error||state?.error||''"/><ModuleStatus v-if="notice||progress" kind="info" :message="progress||notice"/></template>
 <ModuleWorkbench :state-key="stateKey??'M05'" left-label="设备与采集" right-label="任务与记录" :left-width="250" :right-width="310"><template #left><div class="lip-actions"><label>模式 <select v-model="mode" :disabled="recording"><option value="preview">实时预览（候选）</option><option value="realtime">实时参数与原始录制</option><option value="raw">原始视频录制</option><option value="record_then_analyze">高帧率先录后算</option></select></label><button :disabled="recording||busy" @click="start">{{dirty?'保存并开始下一次':mode==='preview'?'打开预览':'开始录制'}}</button><button :disabled="!state||!['opening','previewing','recording'].includes(state.phase)" @click="stop">停止并收尾</button></div> <ModuleSection label="媒体设备与参数" title="设备与参数"><div class="lip-fields">
  <label>摄像头 <select v-model="camera" :disabled="recording"><option value="">系统默认</option><option v-if="camera&amp;&amp;!devices.some(d=>d.kind==='videoinput'&amp;&amp;d.deviceId===camera)" :value="camera">所选摄像头已不可用</option><option v-for="d in devices.filter(d=>d.kind==='videoinput')" :key="d.deviceId" :value="d.deviceId">{{d.label||'未授权摄像头'}}</option></select></label>
  <label>麦克风 <select v-model="microphone" :disabled="recording"><option value="">系统默认</option><option v-if="microphone&amp;&amp;!devices.some(d=>d.kind==='audioinput'&amp;&amp;d.deviceId===microphone)" :value="microphone">所选麦克风已不可用</option><option v-for="d in devices.filter(d=>d.kind==='audioinput')" :key="d.deviceId" :value="d.deviceId">{{d.label||'未授权麦克风'}}</option></select></label>
  <button :disabled="recording" @click="refresh">刷新设备</button><label>请求帧率 <input v-model.number="fps" type="number" min="1" max="240" :disabled="recording"/> fps</label>
  <label><input v-model="filter" type="checkbox" :disabled="recording"/>特征点防抖</label><label>截止频率 <input v-model.number="cutoff" type="number" min="1" max="240" :disabled="recording"/> Hz</label>
  <label>推理设备 <select v-model="delegate" :disabled="recording"><option>CPU</option><option>GPU</option></select></label><label><input v-model="mirror" type="checkbox"/>镜像预览</label>
 </div></ModuleSection>
 <ModuleSection label="采集与保存状态" title="采集状态"><div class="lip-state"><span>阶段 {{state?.phase??'idle'}}</span><span>呈现 {{state?.stats.presented??0}}</span><span>已推理 {{state?.stats.processed??0}}</span><span>检出 {{state?.stats.detected??0}}</span><span>跳过推理 {{state?.stats.skippedInference??0}}</span><span>可观测呈现缺口 {{state?.stats.observedPresentationGaps??0}}</span><span>编码 {{((state?.stats.encodedBytes??0)/1e6).toFixed(1)}} MB</span><span>设备协商 {{state?.trackSettings?.width??'—'}}×{{state?.trackSettings?.height??'—'}} · {{state?.trackSettings?.frameRate??'—'}} fps</span><span v-if="exportInfo">文件 {{exportInfo.video_frames}} 帧 · {{exportInfo.observed_video_fps?.toFixed(1)??'—'}} fps · {{exportInfo.audio?.present?'含音频':'无音频'}}</span><span v-else>保存后核对文件帧率与音频</span><span v-if="state?.stats.processed">候选 {{state.stats.processed}} 帧 · {{capture?.metadata().overlay_display_fps?.toFixed(1)??'—'}} fps</span></div>
  <div class="lip-actions"><button :disabled="busy||!['ready','failed'].includes(state?.phase??'')||!state?.dirty" @click="saveRecording">保存录制（MP4 + 音频）</button><button :disabled="busy||!['ready','failed'].includes(state?.phase??'')" @click="saveMetadata">保存候选参数与时间记录</button><button :disabled="state?.phase!=='ready'||!canAnalyze||busy" @click="analyzeRecording">对录制做正式离线分析</button><button v-if="(mediaRequested||localSaved)&amp;&amp;(metadataRequested||metadataSaved)&amp;&amp;state?.dirty" @click="capture?.markSaved();mediaRequested=metadataRequested=false;notice='已按用户确认标记保存。'">已核对录制及参数下载</button><button :disabled="recording||busy||!dirty" @click="confirmDiscard=true">放弃未保存内容</button></div>
  <div v-if="confirmDiscard" role="alertdialog" aria-label="放弃未保存内容"><p>放弃本页尚未保存的录制和结果？</p><button @click="discard">放弃</button><button @click="confirmDiscard=false">取消</button></div>
 </ModuleSection>
</template>
 <ModuleSection label="视频与唇形画面"><div class="lip-video" :style="{aspectRatio:videoRatio}" :class="{mirrored:mirror&&!selected}"><video ref="video" playsinline muted :hidden="!!selected"/><canvas ref="canvas" width="960" height="540"/></div><ModuleStatus v-if="!shown" kind="empty" message="打开预览、录制或选择视频后显示真实结果。"/>
 <div class="lip-metrics"><span v-for="key in ['area','height','outer_width','inner_width','total_width','open','circularity','face_width','face_height']" :key="key">{{key}} {{shown?.metrics?.[key]?.toFixed(4)??'—'}}</span></div>
 <div v-if="shown" class="lip-curves"><figure v-for="key in curveKeys" :key="key"><figcaption>{{key}}</figcaption><svg viewBox="0 0 600 125" role="img" :aria-label="key+' 参数曲线'"><path :d="curves[key]" fill="none" stroke="currentColor" stroke-width="1.5"/></svg></figure></div><p v-if="shown" class="lip-curve-note">横轴为实际时间，各曲线独立纵轴</p>
 </ModuleSection>
 <template #right><div class="lip-actions"><button @click="help=!help">帮助</button><button @click="emit('references')">方法与引用</button></div> <ModuleStatus kind="info" message="浏览器 Face Landmarker 为候选预览，与旧 FaceMesh 不等价。默认仅在本设备处理，正式结果使用 legacy 离线分析。"/>
 <div v-if="help" class="lip-help">相机镜像只改变预览。高帧率是采集请求，实际取决于设备和浏览器。推理可以跳过预览帧，正式离线分析逐帧处理。原始录制最多暂存 128 MB，候选参数最多 32 MB，达到预算会明确提示。音频/视频时钟与漂移需对编码文件离线核对。自动保存、实验同步及不掉帧均不作保证。桌面磁盘录制可使用 scripts/m05_record_desktop.py（先列出麦克风，再以 --start、设备索引和新输出目录显式启动）；操作说明见 docs/manual/lip-extraction.md。</div>

 <ModuleSection label="已有视频与批量分析" title="离线分析"><div class="lip-actions"><input type="file" accept=".mp4,.mov,.avi,.mkv,.wmv,.m4v,.webm" multiple :disabled="busy||recording" aria-label="选择离线视频" @change="choose"/><button :disabled="busy||recording||!files.length||!canAnalyze" @click="analyze">分析所选 {{files.length}} 个视频</button><button v-if="busy" @click="abort?.abort()">取消本次任务</button></div><label v-if="port.kind==='browser' &amp;&amp; port.available"><input v-model="cloudOptIn" type="checkbox"/>明确上传所选视频至当前账号，进行正式离线分析（占用服务器额度）</label><ModuleStatus v-if="!port.available" kind="info" :message="port.reason??'当前平台离线分析尚未开放'"/><ul v-if="files.length"><li v-for="file in files" :key="file.name">{{file.name}} · {{(file.size/1e6).toFixed(1)}} MB</li></ul></ModuleSection>
 <ModuleSection v-if="port.history" label="本地历史任务"><div class="lip-actions"><button :disabled="busy||recording" @click="refreshHistory">刷新本地历史任务</button><select v-model="historyId" aria-label="历史唇形任务"><option v-for="job in history" :key="job.id" :value="job.id">{{job.name}}</option></select><button :disabled="busy||recording||!historyId" @click="loadHistory">读取历史结果</button></div></ModuleSection>
 <ModuleSection v-if="results.length" label="结果、偏移与动画回放" title="结果与回放"><select v-model.number="selection" :disabled="busy" aria-label="选择唇形结果" ><option v-for="(result,i) in results" :key="result.id" :value="i">{{result.name}}</option></select><p>{{selected?.backend}} · 完整 {{frameCount}} 帧，曲线预览 {{rows.length}} 点。动画按完整帧回放，检测缺失留空。</p>
  <video v-if="selected&amp;&amp;sourceUrls[selected.id]" ref="replayVideo" :src="sourceUrls[selected.id]" controls preload="metadata" aria-label="原始视频音频回放" @error="mediaFailure=true;notice='当前浏览器无法播放此容器或编码，仍可回放唇形动画；音频可从正式结果单独读取。'" @play="mediaPlaying" @pause="mediaPaused" @seeked="syncReplay" @ended="mediaPaused" style="max-width:100%;max-height:220px"/>
  <div class="lip-actions"><button @click="replaying?stopReplay():play()">{{replaying?'暂停动画':'播放动画'}}</button><input v-model.number="replayIndex" type="range" min="0" :max="Math.max(0,frameCount-1)" aria-label="回放帧" @input="stopReplay();seekFrame(replayIndex)"/><span>{{(replayRow?.time_s??0).toFixed(3)}} s</span><label>音唇 offset <input v-model.number="offset" :disabled="busy" type="number" min="-2" max="2" step="0.001"/> s</label><button :disabled="busy" @click="saveResult('apply')">应用偏移并保存</button><button :disabled="busy" @click="saveResult('save_without_offset')">保存但不应用偏移</button><button @click="notice='已取消本次偏移保存，原始结果保留。'">取消偏移保存</button></div>
  <p v-if="selected?.metadata.offset_suggestion?.available">V2 互相关建议 {{selected.metadata.offset_suggestion.offset_seconds.toFixed(3)}} s，仅供人工判断。<button :disabled="busy" @click="offset=Number(selected.metadata.offset_suggestion.offset_seconds.toFixed(3))">填入偏移建议</button></p><p v-else-if="selected?.metadata.audio?.present">自动偏移建议仅处理不超过 600 万采样点的单声道音频；当前结果请手动调整并回放核对。</p>
  <div class="lip-actions"><select v-model="quality" aria-label="动画导出质量"><option value="high">高清 1080</option><option value="standard">标准 720</option><option value="small">小体积 540</option></select><button :disabled="busy||!port.exportAnimation" @click="exportAnimation('mp4')">导出视频</button><button :disabled="busy||!port.exportAnimation" @click="exportAnimation('gif')">导出 GIF</button></div>
 </ModuleSection>
</template></ModuleWorkbench></ModuleFrame>
</template>
<style scoped>
.lip-curves{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-top:14px}.lip-curve-note{margin:2px 0 0;color:var(--muted);font-size:var(--figure-size)}.lip-curves figure{margin:0;min-width:0}.lip-curves svg{display:block;width:100%;height:auto;aspect-ratio:600/125;color:var(--accent)}.lip-curves path{vector-effect:non-scaling-stroke}.lip-curves figcaption{font-size:var(--figure-size);font-family:var(--font-figure)}.lip-columns{display:grid;grid-template-columns:var(--panel-left,260px) minmax(320px,1fr);gap:var(--module-gap)}.lip-fields{display:grid;gap:8px}.lip-fields label{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.lip-fields select{max-width:100%}.lip-fields input[type=number]{width:84px}.lip-video{position:relative;width:100%;background:var(--panel);overflow:hidden}.lip-video video,.lip-video canvas{width:100%;height:100%;object-fit:contain}.lip-video canvas{position:absolute;inset:0}.mirrored video,.mirrored canvas{transform:scaleX(-1)}.lip-metrics,.lip-state,.lip-actions{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.lip-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px 20px;font-family:var(--font-figure);font-size:var(--figure-size);padding-top:8px}.lip-metrics span{min-width:0;font-variant-numeric:tabular-nums}.lip-state{display:grid;grid-template-columns:1fr;font-variant-numeric:tabular-nums;margin-bottom:8px}.lip-actions input[type=number]{width:100px}.lip-help{padding:12px;border:1px solid var(--border);line-height:1.6}.lip-page button,.lip-page input,.lip-page select{font:inherit}.lip-page input[type=file]{max-width:100%}@container module (max-width:720px){.lip-columns{grid-template-columns:1fr}.lip-fields{grid-template-columns:repeat(2,minmax(0,1fr))}}@container module (max-width:720px){.lip-curves{grid-template-columns:repeat(2,minmax(0,1fr))}}@container module (max-width:420px){.lip-fields{grid-template-columns:1fr}.lip-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
