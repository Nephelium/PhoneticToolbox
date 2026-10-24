<script setup lang="ts">
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
import type {Config,Mode,M08Port,Job,Result} from './port.ts';import {defaults,modes,finite,frequencies,validate,importF0,renamePlan,renamePrefix,visible,errorText,variants,shiftedF0,variantLabel,type PitchMode} from './state.ts';
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
const minInput=ref('50'),maxInput=ref('350'),ymin=ref(50),ymax=ref(350),error=ref(''),notice=ref(''),busy=ref(false),tab=ref('single'),help=ref(false),importing=ref(false),importText=ref('');
const jobs=ref<Job[]>([]),history=ref<Result[]>([]),result=ref<Result>(),historyPlot=ref<InstanceType<typeof HistoryPlot>>(),playResult=ref(false),directPlayback=ref(false),batchIds=ref<string[]>([]),cancelSubmit=ref(false);
const loading=ref(false),loadPhase=ref('');
const editing=ref(false),rows=ref<{time:string;freqs:string;mode:Mode;selected:boolean}[]>([]),selectedResults=ref<string[]>([]),manage=ref<'delete'|'rename'|'save'|null>(null),manageResults=ref<Result[]>([]),prefix=ref(''),pendingFile=ref<string>();
const removedResults=new Set<string>();
const lastBatchIds=ref<string[]>([]),batchSubmitting=ref(false),batchSavePending=ref(false),pendingResult=ref<Result>();
let followResults=true,viewTicket=0;
const jobLabels={queued:'排队中',running:'运行中',cancel_requested:'正在取消',cancelled:'已取消',succeeded:'已完成',failed:'失败',interrupted:'已中断'};
const savedOutput=ref<{directory?:string;destination?:string;names:string[]}>();
let disposed=false,epoch=0,previewAbort:AbortController|undefined,timer:ReturnType<typeof setInterval>,polling=false,resultEpoch=0,historyEpoch=0;
const audioFiles=computed(()=>files.value.filter(f=>f.kind==='audio'));
const range=computed(()=>visible(wave.asset?.duration??0,wave.zoom,wave.offset));

const batchJobs=computed(()=>jobs.value.filter(j=>lastBatchIds.value.includes(j.id)));
const batchResults=computed(()=>batchJobs.value.flatMap(j=>j.results).filter(r=>!removedResults.has(r.id)));
const batchPending=computed(()=>batchSubmitting.value||batchJobs.value.some(j=>['queued','running','cancel_requested'].includes(j.state)));
const variantSummary=computed(()=>{try{const count=variants(numericInput,pitchMode.value).length;return `每份音频 ${count} 个版本${tab.value==='batch'?`，所选 ${batchIds.value.length} 份共 ${count*batchIds.value.length} 个`:''}`;}catch(e){return errorText(e);}});
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
async function attempt(work:()=>Promise<void>|void){error.value='';try{await work();}catch(e){if(!disposed)error.value=errorText(e);}}
async function refresh(){const next=await props.context.files.list(directory.value||undefined);if(disposed)return;files.value=next;batchIds.value=batchIds.value.filter(id=>next.some(f=>f.id===id));}
async function choose(){await attempt(async()=>{const grant=await props.context.files.choose?.('input');if(grant){directory.value=grant.id;await refresh();}});}
async function add(event:Event){await attempt(async()=>{const input=event.target as HTMLInputElement;props.context.files.add?.([...input.files??[]]);input.value='';await refresh();});}
function requestLoad(id:string){++viewTicket;pendingResult.value=undefined;if(wave.dirty&&source.value&&id!==source.value.id){pendingFile.value=id;return;}void load(id);}
function selectSource(event:Event){const element=event.target as HTMLSelectElement,id=element.value;element.value=source.value?.id??'';requestLoad(id);}
async function load(id:string){
 const file=files.value.find(f=>f.id===id);if(!file)return;
 const ticket=++epoch;previewAbort?.abort();const controller=new AbortController();previewAbort=controller;
 const current=()=>!disposed&&ticket===epoch&&!controller.signal.aborted;
 loading.value=true;loadPhase.value='正在读取原始音频…';error.value='';notice.value='';cancelPlayback();++historyEpoch;
 source.value=file;times.value=[];original.value=[];modified.value=[];result.value=undefined;resultWave.asset=null;history.value=[];wave.asset=null;playResult.value=false;
 try{
  // Original samples become usable before the durable scientific preview starts.
  // They retain the recording's sampling rate, channels and timeline.
  let rawError:unknown;
  try{const data=await audioPreview(props.context.files,file,controller.signal);if(!current())return;
   wave.asset=markRaw(data.asset);source.value={...file,sha256:data.sha256};
  }catch(e){if(!current())return;rawError=e;}
  if(wave.asset){wave.zoom=1;wave.offset=0;wave.start=0;wave.end=wave.asset.duration;wave.channel=0;}
  if(port.value){loadPhase.value=wave.asset?'原始波形与试听已就绪，正在提取 F0…':'正在解码音频并提取 F0…';
   const data=await port.value.preview(source.value!,controller.signal);if(!current())return;
   if(wave.asset&&source.value?.sha256!==data.sha256)throw Error('源音频在读取后已变化，请重新选择');
   if(!wave.asset){const asset=await decodeWav(data.wav,file.name,controller.signal);if(!current())return;wave.asset=markRaw(asset);wave.zoom=1;wave.offset=0;wave.start=0;wave.end=asset.duration;wave.channel=0;}
   source.value={...file,sha256:data.sha256};times.value=data.times;original.value=data.original_f0;modified.value=[...data.original_f0];
  }else if(rawError)throw rawError;
  if(!current())return;
  const voiced=times.value.filter((_,i)=>original.value[i]>0);config.points=[{time:Number((voiced[0]??0).toFixed(2)),freqs:[],mode:'order'},{time:Number((voiced.at(-1)??wave.asset!.duration).toFixed(2)),freqs:[],mode:'order'}];
  const saved=host.projects.read<{sha256?:string;modified?:number[];config?:Config;numericInput?:typeof numericInput;pitchMode?:PitchMode;refs?:number[];timeRefs?:number[];ymin?:number;ymax?:number}>('m08.draft.'+props.stateKey+'.'+file.id,{});
  if(saved.sha256===source.value?.sha256&&saved.modified?.length===times.value.length&&saved.modified.every(Number.isFinite)){modified.value=saved.modified;if(saved.config){Object.assign(config,saved.config);Object.assign(numericInput,saved.numericInput??{speed:String(config.speed),pitch_ratio:String(config.pitch_ratio),pitch_hz:String(config.pitch_hz)});pitchMode.value=saved.pitchMode??(config.pitch_hz!==0&&config.pitch_ratio===1?'hz':'ratio');}refs.value=saved.refs??[];timeRefs.value=saved.timeRefs??[];ymin.value=saved.ymin??50;ymax.value=saved.ymax??350;minInput.value=String(ymin.value);maxInput.value=String(ymax.value);}
  await Promise.resolve();if(!current())return;wave.dirty=false;
  if(port.value){loadPhase.value='F0 已就绪，正在读取历史…';await refreshHistory(source.value!);}
 }catch(e){if(current())error.value=errorText(e);}finally{if(ticket===epoch){loading.value=false;loadPhase.value='';}}
}
function save(){try{if(loading.value)throw Error('请等待 F0 加载完成后保存草稿');if(!source.value)throw Error('请先选择音频');const values=variants(numericInput,pitchMode.value);if(!host.projects.write('m08.draft.'+props.stateKey+'.'+source.value.id,{sha256:source.value.sha256,modified:modified.value,config:{...config,...values[0]},numericInput:{...numericInput},pitchMode:pitchMode.value,refs:refs.value,timeRefs:timeRefs.value,ymin:ymin.value,ymax:ymax.value}))throw Error('草稿保存失败');wave.dirty=false;notice.value='F0 与参数草稿已保存';return true;}catch(e){error.value=errorText(e);return false;}}
defineExpose({save});
function pan(delta:number){wave.offset=Math.max(0,Math.min((wave.asset?.duration??0)-(range.value[1]-range.value[0]),wave.offset+delta));}
function zoom(factor:number,time:number){if(!wave.asset)return;const old=range.value,rel=(time-old[0])/(old[1]-old[0]);wave.zoom=Math.max(1,Math.min(wave.asset.duration/.05,wave.zoom*factor));wave.offset=Math.max(0,Math.min(wave.asset.duration-wave.asset.duration/wave.zoom,time-rel*wave.asset.duration/wave.zoom));}
function addReference(){void attempt(()=>{if(refDirection.value==='time'){const value=finite(timeRefInput.value,'参考线时间');if(value<0)throw Error('参考线时间不能小于 0 秒');timeRefs.value.push(value);}else refs.value.push(finite(refInput.value,'参考线频率'));dirty();});}
function applyAxis(){void attempt(()=>{const a=finite(minInput.value,'F0 下限'),b=finite(maxInput.value,'F0 上限');if(a>=b)throw Error('F0 上限必须大于下限');ymin.value=a;ymax.value=b;dirty();});}
function applyImport(){void attempt(()=>{modified.value=importF0(times.value,modified.value,...range.value,importText.value);dirty();importing.value=false;});}
function editPoints(){rows.value=config.points.map(p=>({time:String(p.time),freqs:p.freqs.join(','),mode:p.mode,selected:false}));editing.value=true;}
function savePoints(){void attempt(()=>{const points=rows.value.map(r=>({time:finite(r.time,'时间'),freqs:frequencies(r.freqs),mode:r.mode}));const middle=points.slice(1,-1).sort((a,b)=>a.time-b.time);const sorted=[points[0],...middle,points.at(-1)!];validate(sorted,...range.value);config.points=sorted;editing.value=false;});}
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
  try{for(const file of candidates){const read=await props.context.files.read(file);if(ticket!==viewTicket||disposed)return;if(await sha256(read.buffer)===job?.source_sha256){target=file;break;}}}catch(e){if(ticket===viewTicket)error.value=errorText(e);return;}
 }
 if(ticket!==viewTicket||disposed)return;
 if(!target){error.value='请先打开此结果对应的原音频目录，再查看任务';return;}
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
 }catch(e){if(current())error.value=errorText(e);}
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
 }catch(e){if(!disposed)error.value=errorText(e);}finally{polling=false;}
}
async function submit(action:Config['action'],whole=false){await attempt(async()=>{
 if(!source.value||!port.value)throw Error('当前环境无法进行变速变调计算，请使用桌面版');
 const configs=action==='linear'?[snapshot(action,whole)]:variants(numericInput,pitchMode.value).map(v=>snapshot(action,whole,v));
 if(action==='synthesize'&&!configs[0].modified_f0?.length)throw Error('未取得真实 F0，无法合成');
 await submitVersions([{...source.value}],configs,configs.length>1);
});}
async function batch(){await attempt(async()=>{if(!port.value)throw Error('当前环境无法进行变速变调计算，请使用桌面版');const configs=variants(numericInput,pitchMode.value).map(v=>snapshot('transform',true,v)),inputs=audioFiles.value.filter(f=>batchIds.value.includes(f.id));if(!inputs.length)throw Error('请选择至少一个批处理文件');await submitVersions(inputs,configs,true);});}
async function submitVersions(inputs:ResearchFile[],configs:Config[],collect:boolean){
 if(batchPending.value)throw Error('请等待当前批次完成，或先取消剩余任务');
 if(inputs.length*configs.length>256)throw Error('本批次超过 256 个版本，请减少文件或参数值');
 busy.value=true;notice.value='';cancelSubmit.value=false;followResults=!collect;batchSubmitting.value=collect;
 if(collect){lastBatchIds.value=[];batchSavePending.value=true;}
 const failures:string[]=[];let count=0;
 try{for(const file of inputs)for(const cfg of configs){if(cancelSubmit.value||disposed)break;try{
  const job=await port.value!.submit(file,structuredClone(cfg),crypto.randomUUID());count++;
  jobs.value=[...jobs.value.filter(j=>j.id!==job.id),job];if(collect)lastBatchIds.value.push(job.id);
  if(cancelSubmit.value||disposed)await port.value!.cancel(job.id);
 }catch(e){failures.push(`${file.name} · ${variantLabel(cfg)}：${errorText(e)}`);}}
 notice.value=`${cancelSubmit.value?'已停止后续提交，已提交':'已提交'} ${count} 个任务${collect?'，完成后请选择需要保存的版本':''}`;
 if(failures.length)error.value=failures.join('\n');
 }finally{busy.value=false;batchSubmitting.value=false;await poll();offerBatchSave();}
}
function offerBatchSave(){
 if(!batchSavePending.value||batchPending.value||busy.value||manage.value||editing.value||importing.value||pendingFile.value)return;
 if(lastBatchIds.value.length&&!batchJobs.value.length)return;
 batchSavePending.value=false;
 if(batchResults.value.length){const previousError=error.value;openManage('save',batchResults.value);error.value=previousError;}
}
async function saveAudio(){if(result.value)await exportAudio([result.value]);}
async function exportAudio(selected:Result[],numbered=false){
 await attempt(async()=>{
  if(!port.value?.saveMany)throw Error('当前入口未提供音频保存');if(!selected.length)throw Error('请勾选需要保存的音频');
  busy.value=true;notice.value='';
  try{
   if(numbered&&!exportSequence)exportSequence=Object.fromEntries(selected.map((r,i)=>[r.id,i+1]));
   const out=await port.value.saveMany(selected,numbered?exportSequence:undefined);
   if(out.cancelled){notice.value='已取消保存，生成结果保留';return;}
   if(out.saved.length)savedOutput.value={directory:out.directory,destination:out.destination,names:out.saved.map(f=>f.name)};
   notice.value=`已保存 ${out.saved.length} 个音频${out.failed.length?`，${out.failed.length} 个未保存`:''}`;
   if(out.failed.length){selectedResults.value=out.failed.map(f=>f.id);error.value=out.failed.map(f=>`${selected.find(r=>r.id===f.id)?.name??f.id}：${f.error}`).join('\n');}
   else manage.value=null;
  }finally{busy.value=false;}
 });
}
function openManage(kind:'delete'|'rename'|'save',values:Result[]=currentBatch.value){
 error.value='';notice.value='';numberExports.value=false;exportSequence=undefined;manageResults.value=[...values];selectedResults.value=manageResults.value.map(r=>r.id);manage.value=kind;
 prefix.value=renamePrefix(manageResults.value);
}
async function manageFiles(){
 if(manage.value==='save'){await exportAudio([...chosen.value],numberExports.value);return;}
 await attempt(async()=>{
  if(!port.value||!chosen.value.length)throw Error('请勾选需要处理的结果');
  const selected=[...chosen.value];busy.value=true;notice.value='';
  try{
   if(manage.value==='delete'){
    const out=await port.value.remove(selected.map(r=>r.id));for(const id of out.removed)removedResults.add(id);
    history.value=history.value.filter(r=>!removedResults.has(r.id));
    if(result.value&&removedResults.has(result.value.id)){cancelPlayback();result.value=undefined;resultWave.asset=null;playResult.value=false;}
    notice.value=`已删除 ${out.removed.length} 个本批次结果`;
    if(out.failed.length){manageResults.value=manageResults.value.filter(r=>!removedResults.has(r.id));selectedResults.value=out.failed.map(f=>f.id);throw Error(out.failed.map(f=>f.id+': '+f.error).join('\n'));}
   }else{
    const updated=await port.value.rename(renamePlan(selected,prefix.value,history.value.filter(r=>!selectedResults.value.includes(r.id)).map(r=>r.name)));
    if(result.value)result.value=updated.find(r=>r.id===result.value!.id)??result.value;notice.value='重命名完成';
   }
   manage.value=null;
  }finally{
   // Failure while refreshing must not leave all actions permanently disabled.
   try{if(source.value)await refreshHistory({...source.value});}finally{busy.value=false;}
  }
 });
}
async function exportPng(){await attempt(async()=>{const blob=await historyPlot.value!.png(),url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='comparison_plot.png';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notice.value='已导出实际历史 F0 对比图';});}
onMounted(()=>{void attempt(refresh);void poll();timer=setInterval(()=>{if(props.active!==false||jobs.value.some(j=>['queued','running','cancel_requested'].includes(j.state)))void poll();},1000);});onUnmounted(()=>{disposed=true;++epoch;++resultEpoch;previewAbort?.abort();clearInterval(timer);stop();});
useWebSettings(props.stateKey,{...settingsFields(config,['offset']),...settingsFields(numericInput,['speed','pitch_ratio','pitch_hz']),pitchMode,minInput,maxInput,ymin,ymax,refInput,refs,timeRefInput,timeRefs,refDirection,directPlayback});
</script>
<template><ModuleFrame unified fit label="变速变调工作区" class="m08-page" :aria-busy="busy||loading">
<template #toolbar><ModuleToolbar><ResearchUpload :files="context.files" accept=".wav,.mp3,.flac" @uploaded="refresh" folders/><button class="primary" v-if="context.files.choose" :disabled="busy" @click="choose"><AppIcon name="folder"/>打开音频目录</button><button class="primary" v-if="context.files.add" :disabled="busy" @click="picker?.click()">导入音频</button><button class="primary" v-if="context.files.add" :disabled="busy" @click="folderPicker?.click()">导入文件夹</button><input ref="picker" hidden type="file" multiple accept=".wav,.mp3,.flac" @change="add"/><input ref="folderPicker" hidden type="file" multiple webkitdirectory @change="add"/><button :disabled="busy" @click="attempt(refresh)">刷新文件</button><button :aria-pressed="tab==='single'" @click="tab='single'">单文件与基频</button><button :aria-pressed="tab==='batch'" @click="tab='batch'">批量变速变调</button><template #actions><button :disabled="!source||loading" @click="save">保存编辑草稿</button><button @click="help=true">帮助</button><button @click="emit('references')"><AppIcon name="book"/>方法与引用</button></template></ModuleToolbar></template>

<ModuleWorkbench unified :state-key="stateKey" left-label="音频与参数" center-label="音频与基频图" right-label="任务与历史"><template #left><ModuleSection label="音频与参数"><div class="row"><label>音频<select aria-label="变速变调音频" :value="source?.id??''" :disabled="busy" @change="selectSource"><option value="">选择音频</option><option v-for="file in audioFiles" :key="file.id" :value="file.id">{{file.name}}</option></select></label><span>{{wave.dirty?'有未保存编辑':'编辑已保存'}}</span></div><VariantSettings v-if="tab==='single'" v-model:mode="pitchMode" :values="numericInput" :summary="variantSummary"/></ModuleSection>
<ModuleSection label="曲线与试听"><div class="row"><label>F0 下限 Hz<input v-model="minInput" aria-label="F0 下限" type="text" inputmode="decimal" class="numeric-input"/></label><label>上限 Hz<input v-model="maxInput" aria-label="F0 上限" type="text" inputmode="decimal" class="numeric-input"/></label><button @click="applyAxis">应用范围</button><label>参考线方向<select v-model="refDirection" aria-label="参考线方向"><option value="hz">横向 · Hz</option><option value="time">纵向 · 秒</option></select></label><label v-if="refDirection==='time'">参考线 秒<input v-model="timeRefInput" aria-label="参考线 秒" type="text" inputmode="decimal" class="numeric-input"/></label><label v-else>参考线 Hz<input v-model="refInput" aria-label="参考线 Hz" type="text" inputmode="decimal" class="numeric-input"/></label><button @click="addReference">添加参考线</button><button @click="refs=[];timeRefs=[];dirty()">清除参考线</button><button :disabled="!times.length" @click="importing=true">导入基频序列</button></div>
</ModuleSection><details class="advanced" open><summary>批量改变基频与拐点</summary><ModuleSection label="批量改变基频"><div class="row"><button :disabled="!times.length" @click="editPoints">编辑拐点表</button><span>直线插值 · {{Math.max(0,config.points.length-2)}} 个拐点</span><label><input v-model="config.offset" type="checkbox"/>升降偏移 (Hz)</label><button class="primary" :disabled="busy||loading||batchPending||!port||!times.length" @click="submit('linear')">批量生成</button></div><div class="point-summary"><span v-for="(p,i) in config.points" :key="i">{{p.time}} s：{{p.freqs.join(', ')||'未填写'}} Hz · {{modes.find(m=>m.value===p.mode)?.label}}</span></div></ModuleSection>
</details></template><div v-show="tab==='single'" class="single"><div class="plots"><ModuleSection label="原始波形" title="原始波形"><WaveformViewport empty-message="选择音频后显示真实波形。" :state="wave" compact-overview auto-amplitude continuous-detail @selection-end="selectPlayback(false)"/></ModuleSection>
<ModuleSection label="基频编辑" title="基频编辑 · 原始虚线 / 修改实线"><div class="m08-chart-frame"><PitchCurve v-if="times.length" :times="times" :original="original" :modified="modified" :start="range[0]" :end="range[1]" :min="ymin" :max="ymax" :references="refs" :time-references="timeRefs" editable @edit="v=>{modified=v;dirty()}" @pan="pan" @zoom="zoom"/><PlotPlaceholder v-else message="取得 Praat F0 后可编辑。"/></div><div class="plot-footer"><small>Shift＋拖动绘制，Ctrl＋拖动恢复，普通拖动平移，Ctrl＋滚轮缩放。</small></div></ModuleSection>
<ModuleSection label="合成波形" title="合成波形"><WaveformViewport empty-message="合成后显示实际输出。" :state="resultWave" compact-overview auto-amplitude continuous-detail @selection-end="selectPlayback(true)"/><small v-if="result">源区间 {{result.start.toFixed(3)}}–{{result.end.toFixed(3)}} s；输出 {{resultWave.asset?.duration.toFixed(3)}} s。当前编辑不会覆盖此结果。</small></ModuleSection>
<ModuleSection label="历史 F0 对比" title="历史 F0 对比 · 零起点对齐"><div class="m08-chart-frame"><HistoryPlot v-if="history.length" ref="historyPlot" :results="history" :min="ymin" :max="ymax"/><PlotPlaceholder v-else message="生成音频后显示实际 F0 对比。"/></div><div class="plot-footer"><button :disabled="!history.length" @click="exportPng">保存对比图</button></div></ModuleSection></div>
</div><ModuleSection v-show="tab==='batch'" label="文件夹批量处理" title="文件夹批量处理 · 整段音频"><VariantSettings v-if="tab==='batch'" v-model:mode="pitchMode" :values="numericInput" :summary="variantSummary"/><div class="row"><button @click="batchIds=audioFiles.map(f=>f.id)">全选音频</button><button @click="batchIds=[]">清除选择</button><button class="primary" :disabled="busy||batchPending||!port" @click="batch">开始批量处理</button><button v-if="batchSubmitting" @click="cancelSubmit=true">停止提交剩余文件</button></div><div class="file-list"><label v-for="f in audioFiles" :key="f.id"><input v-model="batchIds" :value="f.id" type="checkbox"/><span>{{f.name}}</span></label></div><small>处理整段原音，语速与音高独立调整。完成后勾选版本并选择保存位置，所有原音的本批结果可一次保存。</small></ModuleSection>
<template #right><ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-if="notice" :message="notice"/><ModuleStatus v-if="!port" message="当前环境支持 WAV 波形预览和试听。计算、保存和批次管理请使用桌面版。"/><ModuleStatus v-if="busy||loading" kind="loading" :message="loading?loadPhase:'正在处理，请稍候…'"/><ModuleSection label="合成与试听"><div class="row"><strong>当前视野 {{range[0].toFixed(3)}}–{{range[1].toFixed(3)}} s</strong><button class="primary" :disabled="busy||loading||batchPending||!port||!times.length" @click="submit('synthesize')">合成当前视野</button><button class="primary" :disabled="busy||loading||batchPending||!port||!times.length" @click="submit('synthesize',true)">合成整段</button><button class="primary" :disabled="busy||!result||!port?.saveMany" @click="saveAudio">保存合成音</button><button data-audio-selection :disabled="!wave.asset" @click="playDirect(false)">播放当前视野原音</button><button data-audio-selection :disabled="!resultWave.asset" @click="playDirect(true)">播放合成音</button><small v-if="result||history.length" class="generation-location">生成结果已保留在任务历史；点击保存合成音或批量保存，选择输出目录。</small><div v-if="savedOutput" class="save-location"><strong>最近保存：{{savedOutput.names.length}} 个音频</strong><span v-if="savedOutput.directory">{{savedOutput.directory}}</span><span v-else-if="savedOutput.destination">已保存至所选位置：{{savedOutput.destination}}</span><span v-else-if="context.files.kind==='desktop'">已保存到刚才选择的输出目录。</span><span v-else>已发起浏览器下载，请在下载列表查看保存位置。</span><details><summary>查看保存文件名</summary><ul><li v-for="name in savedOutput.names" :key="name">{{name}}</li></ul></details></div><span>{{playResult?'合成音选区':'原音选区'}}</span><AudioTransport :state="playResult?resultWave:wave" :active="active" :external-playback="directPlayback" compact @selection-play="selectionPlayback" @stop="cancelPlayback"/></div></ModuleSection>
<details class="advanced history-details" open><summary>本批次与历史文件（{{history.length}}）</summary><ModuleSection label="本批次与历史文件"><div class="row"><button :disabled="busy||!currentBatch.length||!port" @click="openManage('delete')">删除本批次音频</button><button :disabled="busy||!currentBatch.length||!port" @click="openManage('rename')">批量重命名</button><button class="primary" :disabled="busy||!currentBatch.length||!port?.saveMany" @click="openManage('save')">批量保存</button><span>当前视野批次 {{currentBatch.length}} 个；同源历史 {{history.length}} 个</span></div><ul class="history"><li v-for="(r,i) in history" :key="r.id"><span class="result-name" :title="r.name">{{i+1}} · {{r.name}}</span><button data-audio-selection :aria-pressed="result?.id===r.id" @click="viewResult(r).then(()=>{if(result?.id===r.id)playDirect(true)})">试听此版本</button><details><summary>生成参数</summary><pre>{{JSON.stringify(r.config,null,2)}}</pre></details></li></ul></ModuleSection></details><ModuleSection v-if="lastBatchIds.length" label="最近生成批次"><strong>本批 {{batchResults.length}} 个版本已完成{{batchPending?'，其余仍在处理':''}}</strong><button class="primary" :disabled="busy||!batchResults.length||!port?.saveMany" @click="batchSavePending=false;openManage('save',batchResults)">保存最近批次</button><small>包含本次所有原音，可一次选择目录保存。</small></ModuleSection>
<details class="advanced task-details" open><summary>变速变调任务（{{jobs.length}}）</summary><ModuleSection label="变速变调任务"><p v-if="!jobs.length">选择文件并提交后，可查看各版本的处理状态与结果。</p><article v-for="j in jobs" :key="j.id" class="task-row"><strong>{{files.find(f=>f.id===j.source_id)?.name??j.source_name??'音频'}} · {{j.id.slice(0,8)}}</strong><small v-if="j.config">{{variantLabel(j.config)}}</small><span role="status">{{jobLabels[j.state]}}</span><p v-if="j.error" class="error-text">{{errorText(j.error)}}</p><button v-if="['queued','running'].includes(j.state)" @click="attempt(()=>port!.cancel(j.id))">取消此任务</button><button v-for="r in j.results" :key="r.id" class="task-result" :aria-pressed="result?.id===r.id" @click="viewResult(r)"><span>{{r.name}}</span><small>{{variantLabel(r.config)}}</small><span>{{result?.id===r.id?'当前版本':'查看此版本'}}</span></button></article></ModuleSection></details>
</template></ModuleWorkbench>
<ModalDialog v-if="importing" title="导入基频序列" @close="importing=false"><p>每行一个 F0；两列时取第二列，时间列忽略。沿当前视野内唯一连续有声段均匀重采样，不改变视野。</p><textarea v-model="importText" aria-label="基频序列" rows="8"/><ModuleStatus v-if="error" kind="error" :message="error"/><template #footer><button @click="importing=false">取消</button><button class="primary" @click="applyImport">导入序列</button></template></ModalDialog>
<ModalDialog v-if="editing" title="编辑拐点表" wide @close="editing=false"><p>首行与末行是起终点，可编辑、不可删除。中间拐点保存时按时间排序，连接方式随行保留。</p><div class="table-wrap"><table><thead><tr><th>选择</th><th>时间 s</th><th>频率列表 Hz</th><th>连接方式</th></tr></thead><tbody><tr v-for="(r,i) in rows" :key="i"><td><input v-model="r.selected" type="checkbox" :disabled="i===0||i===rows.length-1" :aria-label="'选择拐点 '+i"/></td><td><input v-time-precision="'s'" v-model="r.time" type="text" inputmode="decimal" :aria-label="'时间 '+i"/></td><td><input v-model="r.freqs" :aria-label="'频率 '+i"/></td><td><select v-model="r.mode" :aria-label="'连接 '+i"><option v-for="m in modes" :key="m.value" :value="m.value">{{m.label}}</option></select></td></tr></tbody></table></div><ModuleStatus v-if="error" kind="error" :message="error"/><template #footer><button @click="rows.splice(rows.length-1,0,{time:'',freqs:'',mode:'order',selected:false})">添加行</button><button @click="rows=rows.filter((r,i)=>i===0||i===rows.length-1||!r.selected)">删除选中拐点</button><button @click="editing=false">取消</button><button class="primary" @click="savePoints">保存更改</button></template></ModalDialog>
<ModalDialog v-if="manage" :title="manage==='delete'?'确认删除本批次音频':manage==='save'?'批量保存':'批量重命名'" :close-disabled="busy" @close="manage=null"><p>仅处理下列勾选的已生成结果，原始音频不在此列表。</p><p v-if="manage==='save'">{{context.files.kind==='desktop'?'点击保存所选音频后，选择一次输出目录，保存所有勾选的 WAV。同名文件保留。':'点击保存所选音频后，单个版本下载 WAV，多个版本打包下载 ZIP。支持时弹出保存窗口选择位置，其他浏览器使用下载设置中的保存位置。'}}</p><label v-if="manage==='save'" class="export-numbering"><input v-model="numberExports" type="checkbox" :disabled="busy"/>在文件名前添加序号（1_、2_、3_…）</label><label v-if="manage==='rename'">新的共同前缀<input v-model="prefix" aria-label="新的共同前缀"/></label><div class="file-list"><label v-for="r in manageResults" :key="r.id"><input v-model="selectedResults" :value="r.id" type="checkbox" :disabled="busy" @change="exportSequence=undefined"/><span>{{r.name}}</span></label></div><ModuleStatus v-if="error" kind="error" :message="error"/><template #footer><button :disabled="busy" @click="manage=null">取消</button><button :disabled="busy||!chosen.length" @click="manageFiles">{{manage==='delete'?'确认删除所选音频':manage==='save'?'保存所选音频':'确认重命名'}}</button></template></ModalDialog>
<ModalDialog v-if="pendingFile" title="保留当前编辑" @close="pendingFile=undefined;pendingResult=undefined"><p>切换文件前保存当前 F0 与参数草稿？</p><template #footer><button @click="pendingFile=undefined;pendingResult=undefined">取消</button><button @click="finishSwitch(false)">放弃并切换</button><button @click="finishSwitch(true)">保存并切换</button></template></ModalDialog>
<ModalDialog v-if="help" title="变速变调使用说明" wide @close="help=false"><p>播放当前视野原音直接试听原音可见范围，播放合成音直接试听完整输出。播放选区使用最后鼠标选中的原音或合成音区间，与上方两个按钮独立。合成整段明确处理整个音频，F0 使用完整编辑曲线。变速后合成波形按实际输出时长显示。</p><p>F0 默认范围 50–350 Hz，语速 1.0，音高倍率 1.0。参考线可选择横向 Hz 或纵向秒，纵向位置使用原始音频时间，两类线可并存并随编辑草稿保存。语速与音高都可输入多个值，用逗号或空格分隔。音高倍率和 Hz 偏移二选一，2 个语速与 3 个音高值会为每份原音生成 6 个版本。单文件音高调整应用到当前编辑 F0，拐点批量生成沿用独立参数。Praat 提取保留自身默认参数；Manipulation 为 0.01 s、75–600 Hz。语速及文件夹音高采用原有 0.01 变换阈值；单文件 F0 音高按所填倍率或偏移计算。</p><p>顺序/逆序先将列表排序，全连接按输入组合顺序，常量取首值。拐点必须位于起终点之间。偏移模式的值是 Hz 增量。</p><p>每次手绘、导入基频或批量合成成功后，历史图显示实际输出音频重新提取的 F0，零起点对齐。试听此版本加载并直接播放完整音频。多个版本或文件夹批处理完成后会打开批量保存，包含本次所有原音。取消后可点保存最近批次重新选择。批量保存中的在文件名前添加序号默认不勾选，勾选后按所选顺序添加 1_、2_ 等前缀，历史名称保持不变。点击任务中的具体版本可切换到对应原音和输出；当前编辑未保存时先选择保存或放弃。生成结果保留在任务历史中。保存合成音和批量保存会让你选择输出目录，完成后在右栏显示最近保存的完整目录和实际文件名，不增加历史记录，同名文件保留。</p><p>方法：SRC-PRAAT。批量非零 Hz 偏移修正了旧版向 Praat 传入无效单位 Hz 的错误，使用 Hertz；其他计算顺序保持。</p></ModalDialog>
</ModuleFrame></template>
<style scoped>.m08-page{min-width:0;overflow-wrap:anywhere}.m08-page :deep(.task-row){min-width:0;overflow-wrap:anywhere}.single{display:grid;gap:var(--module-gap)}.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:var(--module-gap)}.row{display:flex;align-items:center;flex-wrap:wrap;gap:8px}.row+.row{margin-top:8px}.row label{display:flex;align-items:center;gap:6px}.row .numeric-input{width:86px}.row select{max-width:400px}.m08-page small{display:block;color:var(--muted);font-size:var(--support-size);line-height:1.6}.m08-page :deep(.wave-track svg){height:max(160px,22vh)}.m08-page :deep(.m08-curve){min-height:170px;height:max(230px,29vh)}.m08-page :deep(.history-plot){min-height:170px;height:max(230px,29vh)}.task-row{display:flex;flex-direction:column;align-items:stretch;gap:6px;margin:10px 0;padding:10px 0;border-bottom:1px solid var(--border)}.task-result{display:flex;flex-direction:column;align-items:flex-start;text-align:left;white-space:normal;overflow-wrap:anywhere;gap:4px}.variant-settings{margin:10px 0}.advanced>summary{cursor:pointer;padding:6px 10px;border:1px solid var(--border);border-radius:var(--radius);background:var(--panel)}.point-summary,.file-list{display:flex;flex-direction:column;gap:8px;margin-top:10px}.history{padding-left:20px;max-height:260px;overflow:auto}.history li{margin:10px 0;padding-bottom:10px;border-bottom:1px solid var(--border);overflow-wrap:anywhere}.result-name{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;line-height:1.5}.history li>button{margin:6px 0}.history li details{color:var(--muted)}.history button{margin-left:8px}.history pre{white-space:pre-wrap}.export-numbering{display:flex;align-items:center;gap:8px;margin-top:12px}.export-numbering input{width:16px;height:16px;flex:0 0 16px}.file-list label{display:flex;align-items:flex-start;gap:8px;min-width:0;overflow-wrap:anywhere}.file-list input[type=checkbox]{flex:0 0 16px;width:16px;min-width:16px;height:16px;margin-top:2px}.file-list label>span{min-width:0;flex:1}.m08-chart-frame{display:flex;min-width:0;min-height:170px;height:max(230px,29vh);background:var(--app);border-radius:var(--radius);overflow:hidden}.m08-chart-frame>:deep(*){flex:1;min-width:0;min-height:0;height:100%}.plot-footer{display:flex;align-items:center;min-height:var(--control-height);margin-top:4px}.plot-footer small{margin:0}.plot-footer button{width:100%}.save-location{display:flex;flex-direction:column;gap:4px;padding:8px;border:1px solid var(--border);border-radius:var(--radius);background:var(--app);font-size:var(--support-size);overflow-wrap:anywhere}.save-location ul{padding-left:18px;margin:4px 0}.save-location details{max-height:160px;overflow:auto}.table-wrap{overflow:auto}table input{max-width:180px}textarea{width:100%;background:var(--panel);color:var(--text);border:1px solid var(--border);font:inherit}button[aria-pressed=true]{background:var(--selected);color:var(--accent)}@media(max-width:1000px){.plots{grid-template-columns:1fr}.row select{max-width:100%}}@media(max-width:600px){.row label{flex-wrap:wrap;max-width:100%}.row select{width:100%}.row input{max-width:100%}}.m08-page :deep(.workbench-left) .row,.m08-page :deep(.workbench-right) .row{align-items:stretch;flex-direction:column}.m08-page :deep(.workbench-left) .row label{justify-content:space-between;min-width:0;flex-wrap:wrap}.m08-page :deep(.workbench-left) select{width:100%;max-width:100%}.m08-page .task-details[open]{max-height:none;overflow:visible}
@container module (min-width:1061px){.single{flex:1;min-height:620px;display:flex}.plots{flex:1;grid-template-rows:repeat(2,minmax(0,1fr))}.plots>.module-section{display:flex;flex-direction:column;min-height:0}.plots .m08-chart-frame{flex:1;height:auto;min-height:170px}.plots .m08-chart-frame :deep(.m08-curve),.plots .m08-chart-frame :deep(.history-plot){height:100%;min-height:0}.plots :deep(.wave-viewport){flex:1;display:flex;flex-direction:column}.plots :deep(.wave-track){flex:1;grid-template-rows:auto minmax(0,1fr) auto}.plots :deep(.wave-track svg){height:100%;min-height:100px}}
</style>
