<script setup lang="ts">
import {ref,computed,watch,onMounted,onUnmounted,markRaw} from 'vue';
import type {ResearchContext,ResearchFile,JobView,EggTaskConfig} from '../../platform/research.ts';
import type {AudioAsset} from '../../platform/types.ts';
import type {components} from '../../../../contracts/generated/api';
import EggInverseResult from './EggInverseResult.vue';
import {audioPreview} from '../../platform/research.ts';
import {decodeWav} from '../../platform/decode.ts';
import {workspace,host,type Workspace} from '../../state/workspace.ts';
import {stop} from '../../state/audio.ts';
import AudioTransport from '../../components/AudioTransport.vue';
import AppIcon from '../../components/AppIcon.vue';
import WaveformViewport from '../../components/WaveformViewport.vue';
import TaskPanel from '../../components/TaskPanel.vue';
import ModalDialog from '../../components/ModalDialog.vue';
import EggSignalPlots from './EggSignalPlots.vue';
import EggControls from './EggControls.vue';
import EggParameters from './EggParameters.vue';
import EggBatchPanel from './EggBatchPanel.vue';
import {rangeAfterGesture,microAfterGesture} from './navigation.ts';
import {defaults,batchDefaults,signature,taskConfig,validate,errors,type PreviewRecord} from './state.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active:boolean}>();const emit=defineEmits<{references:[];close:[]}>();
const wave=workspace(props.stateKey),config=ref<EggTaskConfig>({...defaults(),...host.projects.read<Partial<EggTaskConfig>>('egg.config.'+props.stateKey,{})});
const batchConfig=ref<EggTaskConfig>({...batchDefaults(),...host.projects.read<Partial<EggTaskConfig>>('egg.batch-config.'+props.stateKey,{})});
const resultNames=ref<Record<string,string>>({}),resultError=ref(''),resultNotice=ref(''),savingResult=ref(false);
const analysisAudio=ref<AudioAsset>();const inverseView=ref<components['schemas']['EggInverseData']>();
const files=ref<ResearchFile[]>([]),source=ref<ResearchFile>(),sourceHash=ref(''),directory=ref(''),picker=ref<HTMLInputElement>();
const jobs=ref<JobView[]>([]),labels=ref<Record<string,string>>(host.projects.read('egg.jobs.'+props.stateKey,{}));
const preview=ref<PreviewRecord>(),psd=ref(''),error=ref(''),notice=ref(''),loading=ref(false),submitting=ref(false),pending=ref('');
const batchOpen=ref(false),batchBusy=ref(false),help=ref(false),order=ref<number|string|null>(null),resultJob=ref<JobView>(),resultImages=ref<{name:string;url:string}[]>([]),inverseWaves=ref<Workspace[]>([]),resultConfig=ref<EggTaskConfig>();
const tasks=computed(()=>props.context.files.tasks),stale=computed(()=>!!preview.value&&(signature(preview.value.config)!==signature(config.value)||!!source.value&&preview.value.input_sha256!==sourceHash.value));
const displayed=computed(()=>stale.value?undefined:preview.value?.preview);
const playable=computed<Workspace>(()=>({...wave,asset:stale.value||!displayed.value?null:analysisAudio.value??null,channel:0}));
const duration=computed(()=>wave.asset?.duration??0);let disposed=false,epoch=0,viewEpoch=0,polling=false,batchEpoch=0;const urls=new Set<string>();const batchIds=ref<string[]>(host.projects.read('egg.batch.'+props.stateKey,[])),savingBatch=ref(false);
const completedBatch=computed(()=>jobs.value.filter(j=>batchIds.value.includes(j.id)&&j.state==='succeeded'));
let pendingSnapshot:{signature:string;hash:string;epoch:number}|undefined;
const jobViews=computed(()=>jobs.value.map(j=>({id:j.id,title:labels.value[j.id]??`EGG · ${new Date(j.created_at*1000).toLocaleString()} · ${j.id.slice(0,8)}`,status:j.state,progress:j.progress,error:j.error_code?(errors[j.error_code]??j.error_code):undefined,canCancel:true,canRetry:true})));
function makeUrl(buffer:ArrayBuffer){const url=URL.createObjectURL(new Blob([buffer],{type:'image/png'}));urls.add(url);return url;}
function release(url:string){if(url){URL.revokeObjectURL(url);urls.delete(url);}}
function clearPreview(){clearTimeout(gestureTimer);stop();analysisAudio.value=undefined;preview.value=undefined;release(psd.value);psd.value='';}
function labelJob(job:JobView,label:string){labels.value[job.id]=label;host.projects.write('egg.jobs.'+props.stateKey,labels.value);jobs.value=[job,...jobs.value.filter(j=>j.id!==job.id)];}
function save(){const okay=host.projects.write('egg.config.'+props.stateKey,config.value);if(okay)wave.dirty=false;else error.value='参数草稿保存失败，请检查本机存储。';return okay;}
defineExpose({save});
watch(config,()=>{wave.dirty=true;stop();},{deep:true});
watch(()=>[wave.start,wave.end],([start,end])=>{if(start!==config.value.roi_start||end!==config.value.roi_end){config.value.roi_start=start;config.value.roi_end=end;config.value.micro_center=(start+end)/2;stop();}});
watch(()=>config.value.flip_channels,flip=>{if(wave.asset)wave.channel=wave.asset.channels.length===1?0:flip?0:1;});
watch(()=>props.active,active=>{if(!active)stop();});
function setRange(start:number,end:number){config.value.roi_start=start;config.value.roi_end=end;config.value.micro_center=(start+end)/2;wave.start=start;wave.end=end;}
async function refresh(){try{const list=await props.context.files.list(directory.value||undefined);if(!disposed)files.value=list.filter(f=>f.kind==='audio');}catch(e){error.value=String(e);}}
async function choose(){try{const grant=await props.context.files.choose?.('input');if(grant){directory.value=grant.id;await refresh();}}catch(e){error.value=String(e);}}
function addFiles(event:Event){const input=event.target as HTMLInputElement;props.context.files.add?.([...input.files??[]]);input.value='';void refresh();}
async function load(file:ResearchFile){if(!file)return;const ticket=++epoch;closeResult();clearPreview();pending.value='';source.value=undefined;sourceHash.value='';wave.asset=null;error.value='';loading.value=true;
  try{const data=await audioPreview(props.context.files,file);if(disposed||ticket!==epoch)return;if(data.asset.channels.length!==2)throw Error(errors.egg_stereo_required);source.value=file;sourceHash.value=data.sha256;wave.asset=markRaw(data.asset);wave.channel=config.value.flip_channels?0:1;wave.zoom=Math.max(1,data.asset.duration/60);wave.offset=0;setRange(0,Math.min(.5,data.asset.duration));notice.value='已读取双声道音频，点击更新分析。试听使用计算后归一化的音频声道。';}
  catch(e){if(ticket===epoch)error.value=String(e);}finally{if(ticket===epoch)loading.value=false;}}
async function readFile(job:JobView,name:string){if(job.result_manifest?.kind!=='managed_egg_files'||!tasks.value?.result)throw Error('结果尚不可读取。');const f=job.result_manifest.files.find(f=>f.name===name);if(!f)throw Error('结果文件不完整。');return tasks.value.result(job.id,f.id,f.sha256);}
async function showPreview(job:JobView,restore=false,viewTicket?:number){const ticket=epoch;const expectation=pendingSnapshot;const meta=JSON.parse(new TextDecoder().decode(await readFile(job,'egg.ptb.json'))) as PreviewRecord;if(meta.config.mode!=='preview'||meta.preview?.schema_version!=='egg-preview/1')throw Error('该任务不包含交互绘图快照。');
  if(!restore&&(!expectation||expectation.epoch!==ticket||expectation.signature!==signature(config.value)||expectation.hash!==sourceHash.value))return;
  const [wav,png]=await Promise.all([readFile(job,'egg_AUDIO.wav'),readFile(job,'egg_PSD.png')]);const asset=await decodeWav(wav,'归一化分析音频.wav');if(disposed||ticket!==epoch||(restore&&viewTicket!==viewEpoch))return;
  if(!restore&&(expectation?.signature!==signature(config.value)||expectation?.hash!==sourceHash.value))return;
  if(restore){source.value=undefined;sourceHash.value='';config.value={...meta.config};wave.start=meta.selection.start_s;wave.end=meta.selection.end_s;notice.value='已恢复该任务的参数与图形。重新选择源文件后可继续计算。';}
  clearPreview();preview.value=markRaw(meta);psd.value=makeUrl(png);analysisAudio.value=markRaw(asset);if(restore){wave.asset=markRaw(asset);wave.channel=0;wave.zoom=1;wave.offset=0;}
}
async function poll(){if(polling||disposed||!tasks.value?.eggJobs)return;polling=true;
  try{const list=await tasks.value.eggJobs();if(disposed)return;jobs.value=list;const current=list.find(j=>j.id===pending.value);
    if(current&&!['queued','running','cancel_requested'].includes(current.state)){pending.value='';if(current.state==='succeeded')await showPreview(current);else error.value=errors[current.error_code??'']??(current.state==='cancelled'?'分析已取消。':current.error_code??'分析未完成。');}}
  catch(e){if(!disposed)error.value=String(e);}finally{polling=false;}}
async function submit(mode:EggTaskConfig['mode']){if(!source.value||!wave.asset||!tasks.value?.egg||submitting.value||pending.value)return;error.value='';notice.value='';submitting.value=true;const ticket=epoch;const file=source.value;
  try{const frozen=taskConfig(config.value,mode,order.value===''||order.value==null?null:Number(order.value));validate(frozen,duration.value,wave.asset.sampleRate);if(mode==='inverse'&&((frozen.roi_end??0)-(frozen.roi_start??0)>1||((frozen.roi_end??0)-(frozen.roi_start??0))*wave.asset.sampleRate>48000))throw Error(errors.egg_inverse_budget);
    const snapshot={signature:signature(config.value),hash:sourceHash.value,epoch:ticket};const job=await tasks.value.egg(file,frozen,crypto.randomUUID());labelJob(job,`${file.name} · ${{preview:'交互分析',single:'CSV + 三张图',batch:'批次',inverse:'逆滤波'}[mode??'single']}`);
    if(disposed||ticket!==epoch)return;if(mode==='preview'){pendingSnapshot=snapshot;pending.value=job.id;}else notice.value='导出任务已提交，完成后在处理记录中查看和保存。';await poll();}
  catch(e){if(ticket===epoch)error.value=String(e);}finally{submitting.value=false;}}
function seek(time:number){if(!displayed.value||pending.value)return;config.value.micro_center=time;void submit('preview');}
let gestureTimer:ReturnType<typeof setTimeout>|undefined;
function gesture(area:'main'|'micro',kind:'zoom'|'pan',value:number){
  if(!source.value||pending.value||submitting.value||loading.value)return;
  const before=signature(config.value);
  if(area==='main'){const [a,b]=rangeAfterGesture(wave.start,wave.end,duration.value,kind,value);setRange(a,b);}
  else{const next=microAfterGesture(config.value.micro_center??(wave.start+wave.end)/2,config.value.micro_width_ms??50,duration.value,kind,value);config.value.micro_center=next.center;config.value.micro_width_ms=next.width;}
  if(before===signature(config.value))return;
  clearTimeout(gestureTimer);gestureTimer=setTimeout(()=>{if(!disposed&&props.active)void submit('preview');},180);
}
async function cancel(id:string){try{await tasks.value?.cancelJob?.(id);await poll();}catch(e){error.value=String(e);}}
async function retry(id:string){try{const job=await tasks.value!.retry(id,crypto.randomUUID());labelJob(job,(labels.value[id]??'EGG')+' · 重试');if(batchIds.value.includes(id)){batchIds.value.push(job.id);host.projects.write('egg.batch.'+props.stateKey,batchIds.value);}await poll();}catch(e){error.value=String(e);}}
async function batchSubmit(selected:ResearchFile[],settings:EggTaskConfig){if(!tasks.value?.egg)return;const ticket=++batchEpoch;batchBusy.value=true;error.value='';batchIds.value=[];host.projects.write('egg.batch.'+props.stateKey,[]);const frozen=taskConfig(settings,'batch');batchConfig.value={...frozen};host.projects.write('egg.batch-config.'+props.stateKey,frozen);let count=0;const failures:string[]=[];
  try{for(const file of selected){if(disposed||ticket!==batchEpoch)break;try{const job=await tasks.value.egg(file,frozen,crypto.randomUUID());batchIds.value.push(job.id);host.projects.write('egg.batch.'+props.stateKey,batchIds.value);labelJob(job,file.name+' · 批次');count++;if(ticket!==batchEpoch)await tasks.value.cancelJob?.(job.id);}catch{failures.push(file.name);}}}
  finally{batchBusy.value=false;batchOpen.value=false;notice.value=`已提交 ${count} 个文件。${failures.length?'未提交：'+failures.join('、'):''}`;await poll();}}
async function cancelBatch(){batchEpoch++;for(const id of batchIds.value)await cancel(id);notice.value='已请求取消本次批量任务，已完成的结果保留。';}
function closeResult(){viewEpoch++;resultJob.value=undefined;resultConfig.value=undefined;inverseView.value=undefined;for(const image of resultImages.value)release(image.url);resultImages.value=[];inverseWaves.value=[];stop();}
async function view(job:JobView){closeResult();const ticket=viewEpoch;error.value='';try{const metadata=JSON.parse(new TextDecoder().decode(await readFile(job,'egg.ptb.json')));if(disposed||ticket!==viewEpoch)return;
    if(metadata.config.mode==='preview'){epoch++;pending.value='';await showPreview(job,true,ticket);return;}
    resultNames.value=metadata.export_names??{};resultError.value='';resultNotice.value='';resultJob.value=job;resultConfig.value=metadata.config;inverseView.value=metadata.inverse_view;
    if(job.result_manifest?.kind==='managed_egg_files')for(const f of job.result_manifest.files){if(!/\.(png|wav)$/.test(f.name))continue;const raw=await readFile(job,f.name);if(disposed||ticket!==viewEpoch)return;if(f.name.endsWith('.png'))resultImages.value.push({name:resultNames.value[f.name]??f.name,url:makeUrl(raw)});else{const asset=await decodeWav(raw,f.name);if(disposed||ticket!==viewEpoch)return;inverseWaves.value.push({asset:markRaw(asset),start:0,end:asset.duration,channel:0,parameters:[],dirty:false,error:'',loading:false,zoom:1,offset:0});}}}
  catch(e){if(!disposed&&ticket===viewEpoch)error.value=String(e);}}
async function saveBatch(){if(savingBatch.value)return;savingBatch.value=true;error.value='';try{const grant=await props.context.files.choose?.('output');if(!grant||!tasks.value?.saveJob)return;const selected=[...completedBatch.value];let count=0;const failed:string[]=[];for(const job of selected){try{await tasks.value.saveJob(job.id,grant.id);count++;}catch{failed.push(labels.value[job.id]??job.id.slice(0,8));}}notice.value=`已保存 ${count} 个批次任务的完整结果。${failed.length?'未保存：'+failed.join('、'):''}`;}catch(e){error.value=String(e);}finally{savingBatch.value=false;}}
async function saveResult(job:JobView){if(savingResult.value)return;savingResult.value=true;resultError.value='';resultNotice.value='';try{const grant=await props.context.files.choose?.('output');if(grant&&tasks.value?.saveJob){const result=await tasks.value.saveJob(job.id,grant.id);resultNotice.value=`已保存 ${result.count} 个结果文件。`;}else resultNotice.value='已取消保存，计算结果仍保留。';}catch(e){resultError.value=String(e);}finally{savingResult.value=false;}}
async function download(id:string,name:string){try{await tasks.value?.download?.(id,resultNames.value[name]??name);}catch(e){resultError.value=String(e);}}
onMounted(()=>{if(props.context.files.kind!=='desktop')void refresh();void poll();});const timer=setInterval(()=>void poll(),1500);
onUnmounted(()=>{disposed=true;clearTimeout(gestureTimer);epoch++;viewEpoch++;batchEpoch++;clearInterval(timer);stop();for(const url of urls)URL.revokeObjectURL(url);});
</script>
<template><section class="egg-page">
<header class="egg-heading"><div><h1>EGG 分析</h1><small>接触商、速度商与声门事件</small></div><div class="egg-actions"><button v-if="context.files.choose" @click="choose" :disabled="loading"><AppIcon name="folder"/>打开 WAV 目录</button><template v-if="context.files.add"><input ref="picker" type="file" accept=".wav" hidden multiple @change="addFiles"/><button @click="picker?.click()">导入 WAV</button></template><button @click="refresh" :disabled="!directory&&context.files.kind==='desktop'">刷新文件</button><button @click="config.flip_channels=!config.flip_channels;clearPreview()" :disabled="!source">交换声道</button><button @click="batchOpen=true" :disabled="!files.length||!tasks?.egg">批量分析</button><button @click="help=true">使用说明</button><button @click="emit('references')">方法与来源</button></div></header>
<div class="egg-source"><select aria-label="EGG 音频文件" :value="source?.id??''" :disabled="loading" @change="load(files.find(f=>f.id===($event.target as HTMLSelectElement).value)!)"><option value="" disabled>选择双声道 WAV</option><option v-for="f in files" :key="f.id" :value="f.id">{{f.name}}</option></select><span>{{config.flip_channels?'左：音频 · 右：EGG':'左：EGG · 右：音频'}}</span><small v-if="wave.asset">{{wave.asset.sampleRate}} Hz · {{duration.toFixed(3)}} s</small><span v-if="loading||submitting||pending" role="status">{{loading?'正在读取音频…':'分析任务处理中…'}}</span></div>
<p v-if="error" role="alert" class="error-banner">{{error}}</p><p v-if="notice" role="status" class="hint">{{notice}}</p><p v-if="stale" class="notice" role="status">参数或选区已变化，旧图已失效。点击更新分析后再试听或保存当前结果。</p><p v-if="!tasks?.egg" class="notice">当前入口仅支持文件预览。EGG 计算请使用已配置任务服务的桌面工作台或网页项目。</p>
<EggSignalPlots :data="displayed" :psd="psd" :config="config" @seek="seek" @gesture="gesture"><template #filter><div class="egg-filter"><label><input v-model="config.signal_mode" type="checkbox" true-value="filtered" false-value="raw"/>滤波</label><label>高通 <input v-model.number="config.highpass_cutoff" aria-label="EGG 高通频率" type="range" min="1" :max="Math.min(100,(config.lowpass_cutoff??1000)-1)" step="1"/>{{config.highpass_cutoff}} Hz</label><label>低通 Hz <input v-model.number="config.lowpass_cutoff" aria-label="EGG 低通频率" type="number" min="1" max="47999"/></label></div></template></EggSignalPlots>
<section class="egg-controls"><EggControls v-model="config" v-model:order="order" :start="wave.start" :end="wave.end" :duration="duration" :can-update="!!source&&!!tasks?.egg&&!loading&&!submitting&&!pending" :can-export="!!displayed&&!!source&&!submitting&&!pending" :pending="!!pending" @range="setRange" @update="submit('preview')" @cancel="cancel(pending)" @save="submit('single')" @inverse="submit('inverse')"><template #playback><AudioTransport :state="playable" :active="active&&!resultJob" compact/></template></EggControls><EggParameters v-model="config"/></section>
<section class="egg-overview"><div class="egg-actions"><h2>音频总览</h2><small>拖动选区 · Ctrl + 滚轮缩放 · 最多 60 秒视窗</small></div><WaveformViewport v-if="wave.asset" :state="wave" :max-window-seconds="60"/><p v-else class="empty-small">选择 WAV 后显示真实音频波形。</p></section>
<details class="egg-history" open><summary>处理记录 · {{jobs.length}} 个任务</summary><p class="hint">已提交任务保存在当前项目。关闭模块后仍可恢复查看、取消、重试或保存已完成结果。</p><div class="egg-actions"><button v-if="batchBusy||batchIds.length" @click="cancelBatch">取消本次批量任务</button><button v-if="tasks?.saveJob&&completedBatch.length" :disabled="savingBatch" @click="saveBatch">保存本次批量结果（{{completedBatch.length}}）</button><button @click="poll">刷新记录</button><button @click="save">保存参数草稿</button></div><TaskPanel :tasks="jobViews" empty-title="尚无 EGG 任务" empty-text="更新分析、单文件导出及批量操作的进度会显示在这里。" @cancel="cancel" @retry="retry"/><div class="egg-result-links"><template v-for="job in jobs.filter(j=>j.state==='succeeded')" :key="job.id"><button @click="view(job)">查看 {{labels[job.id]??job.id.slice(0,8)}}</button></template></div></details>
<EggBatchPanel v-if="batchOpen" :files="files" :config="batchConfig" :busy="batchBusy" @close="batchOpen=false" @submit="batchSubmit"/>
<ModalDialog v-if="help" title="EGG 使用说明" wide @close="help=false"><p>双声道默认左 EGG、右音频，可交换。各声道独立归一化至峰值 0.7。点击更新分析后，左侧查看 CQ/SQ 与语谱图，点击曲线区定位右侧微观中心。四图支持滚轮缩放、拖动平移，聚焦后方向键平移、加减键缩放；手势结束后更新。微观窗口支持 5–5000 ms，宽窗口沿用旧版抽点显示，事件位置保留。底部总览拖动选择分析区间。</p><p>默认 GCI 斜率、GOI 尺度 0.25，高通 25 Hz、低通 1000 Hz。峰显著度自动模式使用旧版局部窗口。修改任何分析参数后需更新；缺失 CQ/SQ 分别留空。</p><p>保存 CSV / 三图会创建当前选区的完整导出任务。完成后从处理记录查看，桌面选择目录保存，网页逐个下载。单文件 CSV 始终保留两类 F0，页面开关只控制显示。批次参数独立保存，默认同时保留两类 F0；完整滤波文件的 20 ms 平均绝对振幅静音遮罩仅作用于 CSV。</p><p>逆滤波适合稳定元音短片段，限 1 秒 / 48000 帧。输出包含归一化分析音频和简化闭相估计。当前单个分析文件限 60 秒 / 288 万帧，长录音可先在参数估计中切分。微观波形、事件与 CQ 的局部滤波范围沿用旧版，悬停四图区域可查看来源：CQ 使用 processed ±100 ms 重复滤波，微观显示使用 raw ±100 ms 滤波裁剪，事件使用 raw ±50 ms。GCI 实线、GOI 虚线。</p></ModalDialog>
<ModalDialog v-if="resultJob" title="EGG 任务结果" wide @close="closeResult"><p v-if="resultError" role="alert" class="error">{{resultError}}</p><p v-if="resultNotice" role="status">{{resultNotice}}</p><p class="hint">任务 {{resultJob.id.slice(0,8)}} · 参数与结果均来自该次计算快照。</p><details><summary>参数快照</summary><pre>{{JSON.stringify(resultConfig,null,2)}}</pre></details><p v-if="resultConfig?.mode==='inverse'">原音频为归一化分析片段；IF 为简化闭相逆滤波估计。</p><EggInverseResult v-if="inverseView" :data="inverseView"/><section v-for="(item,i) in inverseWaves" :key="i"><h3>{{i===0?'归一化分析音频':'IF 估计'}}</h3><WaveformViewport :state="item"/><AudioTransport :state="item" :active="false" compact/></section><figure v-for="image in resultImages" :key="image.name"><figcaption>{{image.name}} · 导出图</figcaption><img :src="image.url" :alt="image.name" class="egg-export-image"/></figure><template #footer><button v-if="tasks?.saveJob" :disabled="savingResult" @click="saveResult(resultJob)">选择目录保存完整结果</button><template v-if="tasks?.download&&resultJob.result_manifest?.kind==='managed_egg_files'"><button v-for="f in resultJob.result_manifest.files" :key="f.id" @click="download(f.id,f.name)">下载 {{resultNames[f.name]??f.name}}</button></template><button @click="closeResult">返回分析</button></template></ModalDialog>
</section></template>
<style scoped>
.egg-page{padding:16px;display:flex;flex-direction:column;gap:10px;min-width:0;overflow:auto;flex:1}.egg-heading{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}.egg-heading h1{font-size:20px}.egg-actions,.egg-source,.egg-filter,.egg-filter label{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.egg-source select{flex:1;max-width:520px;min-width:170px}.egg-source>span{font-size:12px;color:var(--muted)}.egg-filter{font-size:12px}.egg-filter input[type=range]{width:65px}.egg-filter input[type=number]{width:67px;min-height:28px;padding:4px 6px}.egg-controls{background:var(--app);border:1px solid var(--border);border-radius:var(--radius);padding:10px 12px;display:grid;gap:10px}.egg-overview{border:1px solid var(--border);border-radius:var(--radius);padding:8px 12px}.egg-overview :deep(.wave-toolbar)>span{visibility:hidden}.egg-overview :deep(.wave-svg){max-height:110px}.egg-history{border-top:1px solid var(--border);padding:10px 0}.egg-history summary{cursor:pointer;font-weight:600}.egg-history :deep(.task-panel){max-height:280px;overflow:auto}.egg-result-links{display:flex;gap:6px;flex-wrap:wrap;max-height:160px;overflow:auto;margin-top:8px}.egg-result-links button{max-width:100%;overflow:hidden;text-overflow:ellipsis;display:block}.egg-export-image{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}figure{margin:12px 0}@media(max-width:700px){.egg-page{padding:10px}.egg-heading{align-items:flex-start}.egg-source select{max-width:100%;flex-basis:100%}}
</style>
