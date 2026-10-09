<script setup lang="ts">
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
import {downloadM07,messages,type M07Result,type Settings} from '../../platform/m07.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active:boolean}>();const emit=defineEmits<{references:[]}>();
const analysis=reactive(defaults()),generation=reactive(generationDefaults()),alignment=ref<'normalize'|'onset'>('normalize'),count=ref(21),kind=ref<1|2|3>(2),reverse=ref(false);
const wave=workspace(props.stateKey),sourceWave=workspace(props.stateKey+'.source'),targetWave=workspace(props.stateKey+'.target');
const files=ref<ResearchFile[]>([]),source=ref<ResearchFile>(),target=ref<ResearchFile>(),directory=ref(''),output=ref(''),outputLabel=ref(''),picker=ref<HTMLInputElement>();
const error=ref(''),notice=ref(''),busy=ref(false),help=ref(false),applied=ref<M07Result>(),points=ref<Points>({axis:[],source:[],target:[]}),pending=ref(false),analysisIdentity=ref('');
const jobs=ref<JobView[]>([]),results=ref<M07Result[]>([]),selected=ref<M07Result>(),playName=ref(''),batchSummary=ref('');
const f0Cache=new Map<string,M07F0Display>();
const selectedJob=ref(''),resultTransport=ref<InstanceType<typeof AudioTransport>>(),historyLoading=ref(false);
const taskSnapshots=reactive<Record<string,TaskSnapshot>>({}),f0Tracks=ref<F0Track[]>([]),f0Note=ref('');
const selectedResult=computed(()=>results.value.find(result=>result.job===selectedJob.value));
const plotPoints=computed(()=>selected.value?.metadata.controls??points.value);
const plotAlignment=computed(()=>selected.value?.metadata.alignment??alignment.value);
const f0Plot=ref<InstanceType<typeof F0Comparison>>(),exportingF0=ref(false),axisError=ref('');
const yMinimum=ref<number|string>(0),yMaximum=ref<number|string>(''),axisRange=ref<{minimum:number;maximum:number|null}>({minimum:0,maximum:null});
const axisKey='m07.f0-axis.'+props.stateKey;
const playingF0=computed(()=>{
 if(!playback.playing||!isCurrentAudio(wave.asset,wave.channel)||!selected.value||!wave.asset)return '';
 return playName.value==='combined_steps.wav'?playingStep(playback.position,wave.asset.sampleRate,wave.asset.frames,selected.value.metadata.generation.step_count):playName.value.replace(/\.wav$/,'');
});
function updateAxis(persist=true){
 const minimum=Number(yMinimum.value),maximum=yMaximum.value===''?null:Number(yMaximum.value);
 if(yMinimum.value===''||!Number.isFinite(minimum)||minimum<0||maximum!==null&&(!Number.isFinite(maximum)||maximum<=minimum)){axisError.value='下限须为非负数，上限须大于下限。留空上限可自动计算。';return;}
 axisRange.value={minimum,maximum};axisError.value='';
 if(persist&&!host.projects.write(axisKey,axisRange.value))axisError.value='显示范围已应用，本机保存失败。';
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
let epoch=0,selectionEpoch=0,disposed=false,abort:AbortController|undefined,currentJob:string|undefined;
const port=computed(()=>props.context.files.m07);
const identity=computed(()=>JSON.stringify([source.value?.id,source.value?.sha256,target.value?.id,target.value?.sha256,analysis]));
const stale=computed(()=>!!applied.value&&analysisIdentity.value!==identity.value);
const ready=computed(()=>!!applied.value&&!stale.value&&!pending.value&&!busy.value);
const numericKeys=Object.keys(labels) as (keyof Analysis)[];
watch(analysis,()=>{wave.dirty=true;stop();},{deep:true,flush:'sync'});
watch(generation,()=>{wave.dirty=true;stop();},{deep:true,flush:'sync'});
watch([alignment,count],()=>{wave.dirty=true;pending.value=true;stop();void attempt(()=>rebuild());});
watch([kind,reverse],()=>{wave.dirty=true;stop();});
watch(()=>props.active,v=>{if(!v)stop();});
function report(e:unknown){error.value=e instanceof Error?(messages[e.message]??e.message):String(e);}
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
 await attempt(async()=>{const data=await props.context.files.read(file);const asset=await decodeWav(data.buffer,file.name);if(disposed||ticket!==inputEpoch[role]||(role==='source'?source.value:target.value)?.id!==id)return;if(asset.duration>10)throw Error('当前 M07 输入限 10 秒');file.sha256=data.sha256;w.asset=markRaw(asset);w.start=0;w.end=asset.duration;w.zoom=1;w.offset=0;},()=>ticket===inputEpoch[role]);
}
function rebuild(){if(applied.value)points.value=table(applied.value.metadata.source_f0,applied.value.metadata.target_f0,count.value,alignment.value);}
function edit(role:'source'|'target',i:number,event:Event){const text=(event.target as HTMLInputElement).value;points.value[role][i]=text.trim()===''?0:Number(text);pending.value=true;wave.dirty=true;stop();}
function settings(action:Settings['action']):Settings{return {schema_version:'m07/1',batch_group_count:1,batch_group_index:0,action,analysis:clone(analysis),generation:clone(generation),alignment:alignment.value,point_count:count.value,continuum_type:kind.value,reverse_direction:reverse.value,analysis_job_id:action==='analyze'?null:applied.value?.job,controls:action==='apply'?validatePoints(clone(points.value)):null};}
async function run(action:'analyze'|'apply'){
 await attempt(async()=>{
  if(busy.value)return;if(!port.value||!source.value||!target.value)throw Error('请先选择源音频和目标音频，并使用可用的任务宿主');validate(analysis,generation);
  if(action==='apply'&&(!applied.value||stale.value))throw Error('分析已失效，请重新提取 F0');
  const config=settings(action),ticket=++epoch,revision=identity.value;selectionEpoch++;f0Tracks.value=[];f0Note.value='';abort=new AbortController();busy.value=true;wave.dirty=true;stop();
  try{const result=await port.value.run(source.value,target.value,config,abort.signal,j=>{if(ticket===epoch&&!disposed)updateJob(j,config);});
   if(disposed||!current(ticket,epoch,revision,identity.value)){if(ticket===epoch)notice.value='输入或分析参数已改变，迟到分析未应用';return;}
   taskSnapshots[result.job]=result.metadata;selected.value=undefined;f0Tracks.value=[];f0Note.value='';applied.value=result;analysisIdentity.value=identity.value;points.value=clone(result.metadata.controls);pending.value=false;notice.value=action==='analyze'?'分析完成，控制点尚未修改':'F0 编辑已应用，LPC、残差和脉冲继续复用';
   if(action==='analyze'&&restoredDraft&&restoredDraft.sourceHash===source.value?.sha256&&restoredDraft.targetHash===target.value?.sha256&&restoredDraft.alignment===alignment.value&&restoredDraft.count===count.value&&restoredDraft.analysisHash===JSON.stringify(analysis)){points.value=clone(restoredDraft.points);pending.value=true;restoredDraft=undefined;notice.value='已重新分析相同输入，并恢复草稿控制点，请显式应用编辑';}
  }catch(e){if(ticket===epoch&&!disposed)throw e;}finally{if(ticket===epoch)busy.value=false;}
 });
}
async function generate(all=false){
 await attempt(async()=>{
  if(!ready.value||!port.value||!source.value||!target.value)throw Error(pending.value?'请先应用控制点编辑':stale.value?'分析已失效，请重新提取 F0':'请先完成分析');validate(analysis,generation);
  const snapshot=settings('generate'),sourceFile=clone(source.value),targetFile=clone(target.value),plan=all?designs():[{kind:kind.value,reverse:reverse.value}],batch=crypto.randomUUID();
  const ticket=++epoch;abort=new AbortController();const signal=abort.signal;busy.value=true;wave.dirty=true;stop();let complete=0;batchSummary.value=`0 / ${plan.length} 组完成`;
  try{for(const design of plan){
   if(signal.aborted||ticket!==epoch)break;
   const config={...snapshot,batch_id:batch,batch_group_count:plan.length===6?6 as const:1 as const,batch_group_index:complete,continuum_type:design.kind,reverse_direction:design.reverse};
   const result=await port.value.run(sourceFile,targetFile,config,signal,j=>{if(ticket===epoch&&!disposed)updateJob(j,config);});
   if(disposed||ticket!==epoch)return;
   taskSnapshots[result.job]=result.metadata;if(!results.value.some(r=>r.job===result.job))results.value.unshift(result);await selectTask(result.job);complete++;batchSummary.value=`${complete} / ${plan.length} 组完成`;
  }notice.value=complete===plan.length?'':'生成已停止，完整组保留';
  }catch(e){if(ticket===epoch&&!disposed)throw e;}finally{if(ticket===epoch){busy.value=false;if(complete<plan.length)batchSummary.value=`部分完成：${complete} / ${plan.length} 组，其余未完成或未提交`;}}
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
async function retry(job:JobView){await attempt(async()=>{if(busy.value||!port.value)return;updateJob(await port.value.retry(job.id),taskSnapshots[job.id]);notice.value='已提交重试，请刷新历史查看结果';});}
async function selectTask(id:string){
 selectedJob.value=id;const result=selectedResult.value;
 if(result){await play(result,'combined_steps.wav',false);return;}
 selectionEpoch++;stop();selected.value=undefined;playName.value='';wave.asset=null;f0Tracks.value=[];f0Note.value='';
}
async function play(result:M07Result,name:string,autoplay=true){
 const ticket=++selectionEpoch;selectedJob.value=result.job;selected.value=result;playName.value=name;wave.asset=null;f0Tracks.value=[];f0Note.value='';stop();
 await attempt(async()=>{
  if(!port.value)throw Error('当前入口未提供 M07 结果读取');
  const raw=await port.value.read(result,name);const asset=await decodeWav(raw,name);
  if(disposed||ticket!==selectionEpoch)return;
  wave.asset=markRaw(asset);wave.start=0;wave.end=0;wave.zoom=1;wave.offset=0;wave.channel=0;
  await nextTick();if(disposed||ticket!==selectionEpoch)return;
  await resultTransport.value?.selectAll(autoplay);
  if(disposed||ticket!==selectionEpoch)return;
  notice.value='';
  void showF0(result,name,ticket);
 },()=>ticket===selectionEpoch);
}
async function showF0(result:M07Result,name:string,ticket:number){
 const valid=()=>!disposed&&ticket===selectionEpoch;
 if(!port.value?.f0){if(valid())f0Note.value='当前入口尚未提供合成 F0 曲线';return;}
 if(valid())f0Note.value='正在读取本组的合成 F0…';
 try{
  const display=f0Cache.get(result.job)??await port.value.f0(result);if(!valid())return;f0Cache.set(result.job,display);
  f0Tracks.value=display.curves.flatMap((curve,index)=>name==='combined_steps.wav'||curve.name+'.wav'===name?[{...curve,color:stepColor(index)}]:[]);
  f0Note.value='合成 F0 使用该任务生成轨迹；每一步独立对齐。';
 }catch(e){if(valid())f0Note.value='合成 F0 读取失败：'+(e instanceof Error?e.message:String(e));}
}
async function download(result:M07Result,name:string){const ticket=epoch;await attempt(async()=>{downloadM07(await port.value!.read(result,name),name);if(ticket===epoch)notice.value='文件已交给浏览器下载';},()=>ticket===epoch);}
async function saveResult(result:M07Result){const ticket=epoch;await attempt(async()=>{if(props.context.files.kind==='desktop'&&!output.value){await chooseOutput();if(!output.value)return;}await port.value!.save(result,output.value);if(ticket===epoch)notice.value='已保存本组完整文件和参数清单';},()=>ticket===epoch);}
function save(){try{if(busy.value)throw Error('任务仍在进行，请先完成或取消');validate(analysis,generation);if(points.value.axis.length)validatePoints(points.value);const draft={sourceHash:applied.value?.metadata.inputs.source.sha256??restoredDraft?.sourceHash,targetHash:applied.value?.metadata.inputs.target.sha256??restoredDraft?.targetHash,analysisHash:applied.value?JSON.stringify(applied.value.metadata.analysis):restoredDraft?.analysisHash,analysis:clone(analysis),generation:clone(generation),alignment:alignment.value,count:count.value,kind:kind.value,reverse:reverse.value,points:clone(points.value),pending:pending.value};if(!host.projects.write('m07.draft.'+props.stateKey,draft))throw Error('草稿保存失败');wave.dirty=false;notice.value='参数和未应用控制点已保存为草稿';return true;}catch(e){report(e);return false;}}
defineExpose({save});
onMounted(()=>{const saved=host.projects.read<{minimum:number;maximum:number|null}|null>(axisKey,null);if(saved){yMinimum.value=saved.minimum;yMaximum.value=saved.maximum??'';updateAxis(false);}});
onMounted(()=>{const draft=host.projects.read<{sourceHash?:string;targetHash?:string;analysisHash?:string;analysis:Analysis;generation:ReturnType<typeof generationDefaults>;alignment:'normalize'|'onset';count:number;kind:1|2|3;reverse:boolean;points:Points;pending:boolean}|null>('m07.draft.'+props.stateKey,null);if(draft){try{validate(draft.analysis,draft.generation);Object.assign(analysis,draft.analysis);Object.assign(generation,draft.generation);alignment.value=draft.alignment;count.value=draft.count;kind.value=draft.kind;reverse.value=draft.reverse;points.value=draft.points;restoredDraft=draft;pending.value=!!draft.points.axis.length;notice.value='已恢复参数草稿。重新选择输入并分析后才能应用旧控制点';wave.dirty=false;}catch(e){report(e);}}void attempt(refresh);});
onUnmounted(()=>{disposed=true;epoch++;selectionEpoch++;abort?.abort();stop();sourceWave.asset=null;targetWave.asset=null;});
</script>
<template><ModuleFrame unified fit label="发声类型连续统工作区" class="m07-page" :aria-busy="busy">
<template #toolbar><ModuleToolbar><button class="primary" v-if="context.files.choose" @click="choose"><AppIcon name="folder"/>打开音频目录</button><button class="primary" v-else-if="context.files.add" @click="picker?.click()">添加 WAV</button><button @click="attempt(refresh)">刷新文件</button><input ref="picker" hidden type="file" accept=".wav" multiple @change="add"/><button v-if="context.files.choose" @click="chooseOutput">输出位置</button><span>{{outputLabel||(context.files.kind==='server'?'项目受管结果 · 下载保存':'尚未选择输出位置')}}</span><template #actions><button @click="save">保存参数草稿</button><button @click="help=true">帮助</button><button @click="emit('references')"><AppIcon name="book"/>方法与引用</button></template></ModuleToolbar></template>

<div class="m07-workspace"><ModuleWorkbench unified :state-key="stateKey" left-label="F0 数值表" center-label="音频与基频图" right-label="连续统与任务"><template #left><ModuleSection label="输入音频" class="source-selectors"><div v-for="role in ['source','target'] as const" :key="role"><label>{{role==='source'?'源音频':'目标音频'}}<select :aria-label="role==='source'?'源音频':'目标音频'" :value="(role==='source'?source:target)?.id??''" @change="select(role,($event.target as HTMLSelectElement).value)"><option value="">请选择 WAV</option><option v-for="file in files" :key="file.id" :value="file.id">{{file.name}}</option></select></label></div></ModuleSection><ModuleSection label="分析参数"><div class="controls"><label>F0 后端<select v-model="analysis.f0_backend" aria-label="F0 后端"><option value="parselmouth">Parselmouth / Praat</option><option value="reaper">REAPER</option></select></label><label v-for="key in numericKeys.slice(0,3)" :key="key">{{labels[key]}}<input v-time-precision="key.endsWith('_ms')?'ms':undefined" v-model.number="analysis[key]" type="number" step="any" :aria-label="labels[key]"/></label><button class="primary" :disabled="busy||!port||!source||!target||!sourceWave.asset||!targetWave.asset" @click="run('analyze')">提取 F0</button><span>目标采样率 11025 Hz</span></div><details><summary>高级分析参数</summary><div class="controls"><label v-for="key in numericKeys.slice(3)" :key="key">{{labels[key]}}<input v-time-precision="key.endsWith('_ms')?'ms':undefined" v-model.number="analysis[key]" type="number" step="any" :aria-label="labels[key]"/></label><label>窗函数<select v-model="analysis.window_name"><option v-for="name in ['hamming','hann','blackman','rectangular']" :key="name">{{name}}</option></select></label><label class="check"><input v-model="analysis.trim_silence" type="checkbox"/>静音裁剪</label></div></details></ModuleSection>
<div class="controls"><label>F0 对齐<select v-model="alignment" aria-label="F0 对齐" :disabled="busy"><option value="normalize">归一化有声时长</option><option value="onset">起点对齐</option></select></label><label>F0 控制点数量<input v-model.number="count" type="number" min="20" max="200" aria-label="F0 控制点数量" :disabled="busy"/></label><button class="primary" :disabled="busy||!applied||stale" @click="run('apply')">应用编辑</button><button :disabled="!applied||pending||stale" @click="download(applied!,'edited_f0.csv')">保存 F0 CSV</button></div><ModuleSection label="F0 图窗纵轴" class="f0-axis-controls"><strong>F0 图窗纵轴 (Hz)</strong><div class="axis-inputs"><label>下限<input v-model.number="yMinimum" type="number" min="0" step="any" aria-label="F0 图窗纵轴下限" @input="updateAxis()"/></label><label>上限<input v-model.number="yMaximum" type="number" min="0" step="any" placeholder="自动" aria-label="F0 图窗纵轴上限" @input="updateAxis()"/></label><button @click="resetAxis">自动范围</button></div><small v-if="axisError" role="alert" class="error-text">{{axisError}}</small></ModuleSection><div class="table-scroll"><table v-if="points.axis.length"><thead><tr><th>{{alignment==='normalize'?'对齐时间 (%)':'对齐时间 (ms)'}}</th><th>源 F0 (Hz)</th><th>目标 F0 (Hz)</th></tr></thead><tbody><tr v-for="(x,i) in points.axis" :key="i"><td>{{x.toFixed(3)}}</td><td v-for="role in ['source','target'] as const" :key="role"><input :value="points[role][i]===0?'':points[role][i]" :aria-label="`${role==='source'?'源':'目标'} F0 第 ${i+1} 点`" :disabled="busy" @input="edit(role,i,$event)"/></td></tr></tbody></table><ModuleStatus v-else kind="empty" message="完成分析后显示 F0。编辑需显式应用，生成不会自动混入未应用值。"/></div></template>
<div class="m07-plots">
 <ModuleSection v-for="role in ['source','target'] as const" :key="role" :label="role==='source'?'源音频':'目标音频'" :title="role==='source'?'源音频':'目标音频'" class="audio-plot" :class="role+'-plot'">
  <WaveformViewport empty-message="选择音频后显示实际波形" :state="role==='source'?sourceWave:targetWave" auto-amplitude compact-overview continuous-detail/>
  <AudioTransport :state="role==='source'?sourceWave:targetWave" :active="active" compact/>
 </ModuleSection>
 <ModuleSection label="基频曲线对比" title="F0 曲线对比" class="main-f0">
  <template #actions><button :disabled="exportingF0||(!plotPoints.axis.length&&!f0Tracks.length)" @click="exportF0">{{exportingF0?'导出中…':'导出图片'}}</button></template><F0Comparison ref="f0Plot" :minimum="axisRange.minimum" :maximum="axisRange.maximum" :active-step="playingF0" :points="plotPoints" :alignment="plotAlignment" :tracks="f0Tracks" :note="f0Note||(selected?'任务 '+selected.job.slice(0,8)+' 的 F0 快照':'当前 F0 控制点')"/>
 </ModuleSection>
 <ModuleSection label="合成音频" title="合成音频" class="audio-plot synthesized-plot">
  <WaveformViewport empty-message="生成后默认显示整组音频" :state="wave" auto-amplitude compact-overview continuous-detail/>
  <AudioTransport ref="resultTransport" :state="wave" :active="active" compact/>
  <small v-if="selected" class="current-audio">当前试听：{{taskDescription(selected.metadata)}} · {{playName==='combined_steps.wav'?'整组':playName}} · {{selected.job.slice(0,8)}}</small>
 </ModuleSection>
</div>
<template #right><ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-if="stale" kind="info" message="输入或分析参数已改变，原分析失效，请重新提取 F0。已生成结果保持原快照。"/><ModuleStatus v-if="pending" kind="info" message="控制点或对齐方式有未应用修改，请点击应用编辑。"/><ModuleStatus v-if="notice" kind="info" :message="notice"/><ModuleStatus v-if="!port" kind="info" message="当前宿主未提供 M07 任务能力。请选择已验证的 Windows 工作台，网页计算需通过平台准入。"/><ModuleSection label="连续统与幅度" class="continuum-controls"><div class="parameter-stack"><label>连续统类型<select v-model.number="kind" aria-label="连续统类型"><option v-for="(label,id) in kinds" :key="id" :value="Number(id)">{{label}}</option></select></label><label>合成方向<select v-model="reverse" aria-label="合成方向"><option :value="false">源到目标</option><option :value="true">目标到源</option></select></label><label>连续统步数<input v-model.number="generation.step_count" type="number" min="2" max="50" aria-label="连续统步数"/></label><label class="check"><input v-model="generation.energy_match" type="checkbox"/>周期能量匹配</label><label class="check"><input v-model="generation.normalize_to_source" type="checkbox"/>输出幅度匹配源音频</label><label>输出峰值限制<input v-model.number="generation.output_peak_limit" type="number" step="0.01" min="0.01" max="1"/></label><div class="generate-actions"><button class="primary" :disabled="!ready||!port" @click="generate(false)">生成当前</button><button class="primary" :disabled="!ready||!port" @click="generate(true)">生成全部六组</button><button v-if="busy" class="cancel-generation" @click="cancel()">取消任务</button></div><small>六组：三种类型 × 双方向 · 每组 {{generation.step_count}} 步</small><small v-if="batchSummary">{{batchSummary}}</small></div></ModuleSection><ModuleSection label="任务与结果" class="task-results">
 <div class="history-heading"><strong>任务历史 · {{results.length}} 组</strong><button :disabled="historyLoading" @click="loadHistory">{{historyLoading?'读取中…':'刷新任务历史'}}</button></div>
 <div class="task-history" aria-label="发声类型任务历史" tabindex="0">
  <ModuleStatus v-if="!jobs.length" kind="empty" message="暂无任务。分析与生成分别记录。"/>
  <article v-for="job in jobs" :key="job.id" class="history-row">
   <button class="history-select" :aria-pressed="selectedJob===job.id" @click="selectTask(job.id)"><strong>{{job.id.slice(0,8)}}，{{taskDescription(taskSnapshots[job.id])}}</strong><span>{{taskTime(job.created_at)}} · {{taskStates[job.state]}}</span></button>
   <progress v-if="job.state==='running'" :value="job.progress" max="1" :aria-label="job.id.slice(0,8)+' 进度'"/>
   <small v-if="job.error_code" class="error-text">{{messages[job.error_code]??job.error_code}}</small>
   <button v-if="['queued','running','cancel_requested'].includes(job.state)" @click="cancel(job.id)">取消此任务</button>
   <button v-if="['failed','cancelled','interrupted'].includes(job.state)" :disabled="busy" @click="retry(job)">重试 {{job.id.slice(0,8)}}</button>
  </article>
 </div>
 <div v-if="selectedResult" :key="selectedResult.job" class="result-row">
  <strong>{{taskDescription(selectedResult.metadata)}}</strong>
  <small>完整成功 · {{selectedResult.job.slice(0,8)}} · 批次 {{selectedResult.metadata.batch_id?.slice(0,8)}} 第 {{(selectedResult.metadata.batch_group_index??0)+1}} / {{selectedResult.metadata.batch_group_count??1}} 组</small>
  <div class="result-audios"><button v-for="f in selectedResult.files.filter(f=>f.name.endsWith('.wav')).sort((a,b)=>a.name.localeCompare(b.name))" :key="f.id" :aria-pressed="selected?.job===selectedResult.job&&playName===f.name" @click="play(selectedResult,f.name)">{{f.name==='combined_steps.wav'?'整组试听':f.name.replace('.wav','')}}</button></div>
  <button class="primary" @click="saveResult(selectedResult)">{{context.files.kind==='desktop'?'保存完整组':'下载完整组'}}</button>
  <button @click="download(selectedResult,'m07.ptb.json')">参数与结果清单</button>
 </div>
 <small v-else-if="selectedJob">此任务没有可试听的合成组。</small>
</ModuleSection></template></ModuleWorkbench></div><SourceAcknowledgement @references="emit('references')"/><ModalDialog v-if="help" title="发声类型连续统参数帮助" @close="help=false"><p>源与目标 WAV → 提取 F0 → 编辑并应用控制点 → 生成当前或六组 → 试听与保存。输入目前限每份 10 秒、480,000 帧及 8 MB。桌面无需登录，网页输入和结果受项目权限与额度控制。</p><p>原音频 F0 确定 LPC 分析区间，残差 F0 用于脉冲与连续统。F0 提取间隔以毫秒计，内部轨迹为 1 ms 网格，LPC 帧长和帧移以采样点计。编辑 F0 继续使用原 LPC、残差和脉冲。</p><p>归一化对齐把有声时长映射到 0–100%，起点对齐保留各自时长。合成沿用原方法，将目标有效 F0 重采样到源分析区间。CSV 保存完整 1 ms 源/目标轨迹，空白控制点按无声值处理，内部有效点间插值规则与原实现一致。</p><p>周期能量匹配实际匹配目标残差周期的绝对峰值，输出响度匹配实际匹配全段平均绝对振幅，两个开关独立。峰值限制在重合成后逐步应用。结果端点仍受当前方向的源声道滤波和幅度处理影响。</p><p>每组原子发布，全部六组逐组执行。取消或失败后完整组保留，刷新历史可找回。生成后更改参数不会修改已有结果，试听与导出始终读取该组快照。保存时使用独立组目录，不覆盖旧成果。</p><p>该方法是 LPC 残差操纵得到的实验刺激，不表示还原声门生理机制，也未证明发声类别知觉效度或步长知觉等距。来源包括载瓦语相关研究及继承 Python 实现，论文与代码许可分别登记。音标示例 <span class="ipa-text">a̤ a̰</span> 使用公共 Doulos SIL 字体。</p></ModalDialog>
</ModuleFrame></template>
<style scoped>
.m07-page{overflow:hidden}.m07-page :deep(.workbench-right-body>.module-status){padding:4px 6px}.m07-page :deep(.workbench-right-body>.module-status p){line-height:1.25}.m07-workspace{display:flex;flex-direction:column;flex:1;min-width:0;min-height:0;overflow:auto;overscroll-behavior:contain;scrollbar-width:thin}
.controls{display:flex;flex-wrap:wrap;align-items:end;gap:var(--control-gap);margin-bottom:8px}label{display:grid;gap:4px}label input{width:120px}label select{max-width:100%}.check{display:flex;align-items:center}.check input{width:auto}.continuum-controls{container:m07-controls / inline-size}.parameter-stack{display:grid;gap:3px;font-size:max(12px,calc(var(--control-size)*.93));line-height:1.2}.parameter-stack>label{font-size:inherit;line-height:inherit;display:flex;align-items:center;justify-content:space-between;gap:6px}.parameter-stack>.check{justify-content:flex-start}.parameter-stack>label select{flex:1;max-width:180px;min-width:0}.parameter-stack>label input[type=number]{width:76px}.parameter-stack input,.parameter-stack select,.parameter-stack button{font-size:inherit;min-height:28px;padding:3px 6px}.parameter-stack .check input{padding:0;min-height:16px;width:16px;height:16px;flex:none}.parameter-stack small{color:var(--muted);font-size:var(--support-size)}.generate-actions{display:grid;gap:4px}.generate-actions .cancel-generation{grid-column:1/-1}.f0-axis-controls{display:grid;gap:5px;font-size:var(--support-size)}.axis-inputs{display:grid;grid-template-columns:1fr 1fr;gap:5px}.axis-inputs input{width:100%;min-width:0}.axis-inputs button{grid-column:1/-1;min-height:26px;padding:2px 6px}.f0-axis-controls small{line-height:1.3}.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:4px;border-bottom:1px solid var(--border)}td input{width:100%;min-width:60px}.ipa-text{font-family:var(--font-ipa)}
.m07-plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(2,minmax(290px,1fr));gap:var(--module-gap);flex:1;min-height:600px}.m07-plots>.module-section{display:flex;flex-direction:column;min-height:0;overflow:hidden}.m07-plots :deep(.wave-viewport){flex:1;display:flex;flex-direction:column;min-height:0}.m07-plots :deep(.wave-track){flex:1;min-height:0;grid-template-rows:auto minmax(90px,1fr) auto}.m07-plots :deep(.wave-track svg){height:100%;min-height:90px}.m07-plots :deep(.wave-toolbar){flex:none}.m07-plots :deep(.transport-controls){flex:0 0 auto;min-width:0;margin-top:6px}.m07-plots :deep(.audio-transport){flex-wrap:wrap}.current-audio{color:var(--muted);font-size:var(--support-size);margin-top:6px;overflow-wrap:anywhere}
.history-heading{display:flex;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:5px}.history-heading button{font-size:var(--support-size);min-height:28px;padding:3px 6px}.task-history{max-height:84px;overflow:auto;overscroll-behavior:contain;scrollbar-width:thin;display:grid;gap:3px}.history-row{display:grid;gap:3px}.history-select{display:grid;gap:2px;text-align:left;width:100%;font-size:var(--support-size);line-height:1.2;padding:4px 6px;min-height:0}.history-select strong{font-weight:500}.history-select span,.result-row small{color:var(--muted)}.history-select[aria-pressed=true],.result-audios button[aria-pressed=true]{border-color:var(--accent);background:var(--selected)}.history-row progress{width:100%}.result-row{display:grid;gap:5px;padding-top:6px;margin-top:5px;border-top:1px solid var(--border);font-size:var(--support-size)}.result-row>button{font-size:inherit;min-height:28px;padding:3px 6px}.result-row>strong,.result-row>small{line-height:1.25}.result-audios{display:grid;grid-template-columns:repeat(auto-fit,minmax(max(60px,calc(var(--support-size)*4 + 12px)),1fr));gap:4px;max-height:130px;overflow:auto}.result-audios button{min-width:0;min-height:28px;padding:3px 5px;font-size:inherit}
.m07-page :deep(.workbench-left)>.table-scroll{max-height:none;min-height:180px;flex:1}.m07-page :deep(.workbench-center)>.m07-plots{flex:1;flex-shrink:0}.source-selectors{display:grid;gap:6px}.source-selectors label{display:flex;gap:6px;align-items:center}.source-selectors select{flex:1;min-width:0}.m07-page :deep(.workbench-left) .controls{display:grid;grid-template-columns:1fr 1fr;gap:6px}.m07-page :deep(.workbench-left) .controls label{display:grid;min-width:0;gap:3px}.m07-page :deep(.workbench-left) .controls input,.m07-page :deep(.workbench-left) .controls select{width:100%;min-width:0;max-width:100%}.m07-page :deep(.workbench-left) .controls .check{display:flex}
@container m07-controls (min-width:240px){.generate-actions{grid-template-columns:1fr 1fr}}
@container module (max-width:720px){.m07-plots{grid-template-columns:1fr;grid-template-rows:repeat(4,minmax(280px,auto));min-height:0}.m07-plots>.module-section{min-height:280px}}
</style>
