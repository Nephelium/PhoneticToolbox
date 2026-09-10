<script setup lang="ts">
import {computed,ref,onMounted,onUnmounted,watch,markRaw} from 'vue';
import type {ResearchContext,ResearchFile,DirectoryGrant,SpectrogramView,BatchView,JobView,BatchSelection,BatchConfig} from '../../platform/research.ts';
import {audioPreview} from '../../platform/research.ts';
import {m01State,saveM01} from './store.ts';
import {association,matchAssociation,reconcile,applyParameters,applySettings,dirty,effectiveOutput,type Settings} from './state.ts';
import {stop} from '../../state/audio.ts';
import WaveformViewport from '../../components/WaveformViewport.vue';
import WorkbenchColumns from '../../components/WorkbenchColumns.vue';
import ParameterDrawer from '../../components/ParameterDrawer.vue';
import SettingsDrawer from '../../components/SettingsDrawer.vue';
import AppIcon from '../../components/AppIcon.vue';
import BatchResults from './BatchResults.vue';
const props=defineProps<{context:ResearchContext}>();const emit=defineEmits<{references:[]}>();
const state=m01State(props.context.key),picker=ref<HTMLInputElement>(),busy=ref(false),notice=ref('');
const audioFiles=computed(()=>state.files.filter(f=>f.kind==='audio'));
function listKey(event:KeyboardEvent){if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='a'){event.preventDefault();state.marked=audioFiles.value.map(f=>f.id);}}
const selected=computed(()=>audioFiles.value.find(f=>f.id===state.selected));
const spectrogramLoader=computed(()=>{const file=selected.value;return props.context.files.spectrogram&&file?(view:SpectrogramView)=>props.context.files.spectrogram!(file,view):undefined;});
const linked=computed(()=>association(state));
const intervals=computed(()=>linked.value.tiers[linked.value.layer]?.intervals??[]);
const segments=computed(()=>intervals.value.filter(i=>i.xmax>i.xmin&&!['','sil','eps','<sil>','<eps>'].includes(i.text.trim().toLowerCase())).length);
const tasks=computed(()=>props.context.files.tasks),taskBusy=ref(false),taskReady=ref(false),batchList=ref<BatchView[]>([]),activeBatch=ref<BatchView|null>(null),jobResults=ref<JobView[]>([]),sliceParameters=ref(false);
const resultCache=new Map<string,JobView>();let taskTimer:ReturnType<typeof setTimeout>|undefined;
const parameterSource=ref<'recent'|'legacy'>('recent'),legacyOpen=ref(false),pickleId=ref(''),converting=ref(false);
let autoSaveBatch:{id:string;directory:string}|null=null;
function scheduleBatch(){clearTimeout(taskTimer);if(!disposed&&tasks.value)taskTimer=setTimeout(()=>void loadBatch(),activeBatch.value&&!activeBatch.value.summary.closed?1500:6000);}
async function loadBatch(id?:string){
 if(!tasks.value||disposed)return;if(taskBusy.value){scheduleBatch();return;}
 taskBusy.value=true;
 try{const list=await tasks.value.list();if(disposed)return;batchList.value=list;taskReady.value=true;const chosen=id??activeBatch.value?.id??batchList.value[0]?.id;
  if(chosen){const batch=await tasks.value.get(chosen);if(disposed)return;activeBatch.value=batch;for(const item of batch.summary.items){if(item.job_id&&['succeeded','failed','cancelled','interrupted'].includes(item.state)&&!resultCache.has(item.job_id)){const job=await tasks.value.job(item.job_id);if(disposed)return;resultCache.set(item.job_id,job);}}jobResults.value=batch.summary.items.flatMap(item=>item.job_id&&resultCache.has(item.job_id)?[resultCache.get(item.job_id)!]:[]);}
 }catch(e){if(!disposed)notice.value=e instanceof Error?e.message:'任务读取失败。';}
 finally{taskBusy.value=false;}
 if(disposed)return;
 if(autoSaveBatch?.id===activeBatch.value?.id&&activeBatch.value?.summary.closed){const directory=autoSaveBatch!.directory;autoSaveBatch=null;if(activeBatch.value.summary.counts.succeeded&&tasks.value.save)await saveResults(directory);}
 scheduleBatch();
}
async function startBatch(operation:'acoustic_analysis'|'textgrid_segment'){
 if(!tasks.value||taskBusy.value)return;
 taskBusy.value=true;notice.value='';clearTimeout(taskTimer);
 try{const chosen=operation==='acoustic_analysis'?audioFiles.value:audioFiles.value.filter(f=>state.marked.length?state.marked.includes(f.id):f.id===state.selected);
  if(!chosen.length)throw Error('请先选择音频。');if(operation==='acoustic_analysis'&&!state.wave.parameters.length)throw Error('请选择输出参数。');
  const layer=operation==='textgrid_segment'?linked.value.tiers[linked.value.layer]?.name??null:null;
  if(operation==='textgrid_segment'&&!layer)throw Error('请选择TextGrid切分层。');
  if(props.context.files.kind==='desktop'&&!effectiveOutput(state))throw Error('请先选择结果目录。');
  const inputs:BatchSelection[]=[];let noParent=0;
  for(const file of chosen){const a=association(state,file.id);if(a.error)throw Error(file.name+'：'+a.error);if(operation==='textgrid_segment'&&!a.textgrid)throw Error(file.name+' 尚未关联TextGrid。');
   const item:BatchSelection={audio:file,textgrid:a.textgrid,lip:operation==='acoustic_analysis'?a.lip:null};
   if(operation==='textgrid_segment'&&sliceParameters.value){
    if(parameterSource.value==='legacy'){if(!a.legacy)throw Error(file.name+' 尚未明确关联历史参数表，请逐项选择。');item.legacy_result=a.legacy;}
    else{const parent=await tasks.value.parent(file);if(parent)item.parent_result=parent;else noParent++;}
   }inputs.push(item);}
  const config:BatchConfig|null=operation==='acoustic_analysis'?{settings:{...state.settings},selection:{mode:'catalog',keys:[...state.wave.parameters] as NonNullable<NonNullable<BatchConfig['selection']>['keys']>},backend_policy:{reaper:'native_required',wm_f0:'irapt_then_praat'}}:null;
  const directory=effectiveOutput(state)?.id;
  const batch=await tasks.value.submit(operation,inputs,config,layer,crypto.randomUUID());if(disposed)return;activeBatch.value=batch;autoSaveBatch=props.context.files.kind==='desktop'&&directory?{id:batch.id,directory}:null;
  notice.value=noParent?`${noParent}个音频没有可用参数结果，将仅切分音频。`:'批次已提交。参数与文件关联已固定，后续编辑只影响新任务。';
 }catch(e){notice.value=e instanceof Error?e.message:'提交失败。';}finally{taskBusy.value=false;void loadBatch();}
}
async function cancelBatch(){if(!tasks.value||!activeBatch.value||taskBusy.value)return;taskBusy.value=true;try{activeBatch.value=await tasks.value.cancel(activeBatch.value.id);}catch(e){notice.value=String(e);}finally{taskBusy.value=false;void loadBatch();}}
async function retryJob(id:string){if(!tasks.value||taskBusy.value)return;taskBusy.value=true;try{await tasks.value.retry(id,crypto.randomUUID());}catch(e){notice.value=String(e);}finally{taskBusy.value=false;void loadBatch();}}
async function saveResults(target?:string){const directory=target??effectiveOutput(state)?.id;if(!tasks.value?.save||!activeBatch.value||taskBusy.value)return;if(!directory){notice.value='重新打开后，请先选择结果目录，再保存已完成结果。';return;}taskBusy.value=true;try{const result=await tasks.value.save(activeBatch.value.id,directory);if(disposed)return;await refresh();notice.value=`已保存${result.count}个结果文件；已有同名不同内容文件保持原样。`;}catch(e){if(!disposed)notice.value=String(e);}finally{taskBusy.value=false;scheduleBatch();}}
async function downloadResult(id:string,name:string){if(!tasks.value?.download||taskBusy.value)return;taskBusy.value=true;try{await tasks.value.download(id,name);}catch(e){notice.value=String(e);}finally{taskBusy.value=false;scheduleBatch();}}
let disposed=false,previewAbort=new AbortController();
watch(()=>dirty(state),value=>state.wave.dirty=value,{immediate:true});
async function refresh(){
 const version=++state.listVersion;busy.value=true;notice.value='';
 try{if(props.context.files.kind==='desktop'&&!state.input)return;
  const values=await props.context.files.list(state.input?.id);
  if(state.associationDirectory)values.push(...(await props.context.files.list(state.associationDirectory.id)).filter(f=>f.kind!=='audio'));
  if(disposed||version!==state.listVersion)return;
  reconcile(state,[...new Map(values.map(f=>[f.id,f])).values()]);
  for(const file of audioFiles.value){const a=association(state,file.id);try{
   for(const kind of ['textgrid','lip'] as const)if(!a[kind]&&!a.manual[kind])a[kind]=matchAssociation(file,state.files,kind);
   a.error='';
  }catch(e){a.error=e instanceof Error?e.message:'关联失败。';}}
  if(state.selected&&state.wave.asset)await loadGrid(state.selected,state.loadVersion);
 }catch(e){if(!disposed&&version===state.listVersion)notice.value=e instanceof Error?e.message:'目录读取失败。';}
 finally{if(!disposed&&version===state.listVersion)busy.value=false;}
}
async function choose(purpose:DirectoryGrant['purpose']){
 if(busy.value)return;busy.value=true;notice.value='';
 try{const grant=await props.context.files.choose?.(purpose);if(!grant||disposed)return;
  if(purpose==='input'){state.input=grant;state.loadVersion++;state.selected='';state.wave.asset=null;stop();}
  else if(purpose==='output')state.output=grant;else state.associationDirectory=grant;
  if(purpose!=='output')await refresh();
 }catch(e){notice.value=e instanceof Error?e.message:'目录选择失败。';}finally{busy.value=false;}
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
 stop();const version=++state.loadVersion;state.selected=file.id;state.wave.asset=null;state.wave.loading=true;state.wave.error='';
 try{const data=await audioPreview(props.context.files,file,previewAbort.signal);if(disposed||version!==state.loadVersion)return;
  Object.assign(state.wave,{asset:markRaw(data.asset),start:0,end:data.asset.duration,channel:0,zoom:1,offset:0});file.sha256=data.sha256;
  try{await loadGrid(file.id,version);}catch(e){if(!disposed&&version===state.loadVersion)association(state,file.id).error=e instanceof Error?e.message:'TextGrid读取失败。';}
 }catch(e){if(!disposed&&version===state.loadVersion)state.wave.error=e instanceof Error?e.message:'音频读取失败。';}
 finally{if(!disposed&&version===state.loadVersion)state.wave.loading=false;}
}
async function changeAssociation(kind:'textgrid'|'lip',event:Event){
 const id=(event.target as HTMLSelectElement).value,a=linked.value;
 a[kind]=state.files.find(f=>f.id===id&&f.kind===kind)??null;a.manual[kind]=true;a.error='';
 if(kind==='textgrid'){a.tiers=[];a.gridHash='';a.layer=0;const version=++state.loadVersion;try{await loadGrid(state.selected,version);}catch(e){if(version===state.loadVersion)a.error=e instanceof Error?e.message:'TextGrid读取失败。';}}
}
async function convertLip(){
 const file=state.files.find(f=>f.id===pickleId.value&&f.kind==='lip_pickle'),audioId=state.selected;
 if(!file||!props.context.files.convertLip||converting.value)return;converting.value=true;notice.value='';
 try{const result=await props.context.files.convertLip(file);if(disposed)return;await refresh();
  if(state.files.some(f=>f.id===audioId)){const a=association(state,audioId);a.lip=result.file;a.manual.lip=true;}
  legacyOpen.value=false;
  notice.value=`已保存 ${result.file.name} 并关联到所选音频。${result.companion_found?'已读取同名伴随时间戳，音频起点优先采用 PKL 内元数据。':''}原 PKL 保持原样。`;
 }catch(e){if(!disposed)notice.value=e instanceof Error?e.message:'转换失败。';}finally{converting.value=false;}
}
function openParameters(){state.parameterDraft=[...state.wave.parameters];state.drawer='parameters';}
function openSettings(){state.settingsDraft={...state.settings};state.drawer='settings';}
function parameters(keys:string[]){try{applyParameters(state,keys);}catch(e){notice.value=String(e);}}
function settings(value:Settings){applySettings(state,value);}
function save(){notice.value=saveM01(props.context.key)?'参数与设置已保存。文件、目录权限和账号凭据不会写入草稿。':'草稿保存失败，请检查本机存储权限。';}
function intervalSelect(xmin:number,xmax:number){stop();if(!state.wave.asset)return;state.wave.start=Math.max(0,Math.min(xmin,state.wave.asset.duration));state.wave.end=Math.max(state.wave.start,Math.min(xmax,state.wave.asset.duration));}
onMounted(()=>{if(props.context.files.kind!=='desktop'||state.input)void refresh();if(tasks.value)void loadBatch();});
onUnmounted(()=>{disposed=true;clearTimeout(taskTimer);previewAbort.abort();state.loadVersion++;state.listVersion++;state.wave.loading=false;stop();});
</script>
<template>
<section class="workspace-page m01-page" aria-label="参数估计 工作区">
<header class="page-heading"><div><p class="eyebrow">M01 · {{context.label}}</p><h1>参数估计</h1></div><button @click="emit('references')"><AppIcon name="book"/>方法与引用</button></header>
<div class="m01-directory-bar">
<template v-if="context.files.kind==='desktop'"><button class="primary" :disabled="busy" @click="choose('input')">选择音频目录</button><span class="directory-label" :title="state.input?.label">{{state.input?.label??'尚未选择目录'}}</span><button :disabled="busy" @click="choose('association')">选择关联目录</button><span v-if="state.associationDirectory" class="directory-label" :title="state.associationDirectory.label">{{state.associationDirectory.label}}</span></template>
<template v-else-if="context.files.kind==='preview'"><input ref="picker" class="visually-hidden" type="file" accept=".wav" multiple aria-label="选择音频列表" @change="add"/><button class="primary" @click="picker?.click()">添加WAV到列表</button><span class="hint">本机预览 · 不上传</span></template>
<template v-else><span>项目资源 · {{context.label}}</span><small class="muted">在项目文件管理中上传 WAV、TextGrid、.lip.json 或历史 XLSX/SQLite，再刷新。</small></template>
<button :disabled="busy||(context.files.kind==='desktop'&&!state.input)" @click="refresh">{{busy?'正在读取…':'刷新列表'}}</button>
<div v-if="context.files.kind==='desktop'" class="m01-output-row"><label><input v-model="state.sameDirectory" type="checkbox"/>结果与WAV同目录</label><button :disabled="state.sameDirectory||busy" @click="choose('output')">选择结果目录</button><span class="directory-label" :title="effectiveOutput(state)?.label">{{effectiveOutput(state)?.label??'尚未选择结果目录'}}</span></div></div>
<p v-if="notice" role="status" class="notice">{{notice}}</p><p v-if="state.wave.error" role="alert" class="error-banner">{{state.wave.error}}</p>
<WorkbenchColumns @files-keydown="listKey">
<template #files><div class="panel-heading"><h2>音频列表</h2><small>{{audioFiles.length}} 个文件</small></div><p class="hint">点选用于试听；批处理覆盖整个列表。</p>
<div class="m01-select-all"><label><input aria-label="全选音频" type="checkbox" :checked="audioFiles.length>0&&state.marked.length===audioFiles.length" :indeterminate="state.marked.length>0&&state.marked.length<audioFiles.length" :disabled="!audioFiles.length" @change="state.marked=($event.target as HTMLInputElement).checked?audioFiles.map(f=>f.id):[]"/>全选</label><small>切分选择 {{state.marked.length}} / {{audioFiles.length}}</small></div><div class="m01-file-list"><div v-for="file in audioFiles" :key="file.id" class="m01-file-entry"><input v-model="state.marked" type="checkbox" :value="file.id" :aria-label="'选择切分 '+file.name"/><button class="file-row" :class="{selected:state.selected===file.id}" :aria-pressed="state.selected===file.id" @click="selectFile(file)"><AppIcon name="file"/><span>{{file.name}}<small>{{association(state,file.id).textgrid?'TextGrid · ':''}}{{association(state,file.id).lip?'唇形关联 · ':''}}{{Math.ceil(file.size/1024)}} KB</small><small v-if="association(state,file.id).error" class="danger-text">关联待确认</small></span></button></div></div>
<p v-if="!audioFiles.length" class="empty-small">选择音频目录或刷新项目资源。</p></template>
<template #default><template v-if="selected"><div class="signal-heading"><h2>{{selected.name}}</h2><p v-if="state.wave.asset" class="mono muted">{{state.wave.asset.sampleRate}} Hz · {{state.wave.asset.channels.length}} 声道 · {{state.wave.asset.duration.toFixed(3)}} s</p></div>
<div class="m01-association-controls"><label>TextGrid<select aria-label="TextGrid关联" :value="linked.textgrid?.id??''" :disabled="state.wave.loading" @change="changeAssociation('textgrid',$event)"><option value="">不关联</option><option v-for="f in state.files.filter(f=>f.kind==='textgrid')" :key="f.id" :value="f.id">{{f.name}}</option></select></label><label>唇形数据<select aria-label="唇形关联" :value="linked.lip?.id??''" @change="changeAssociation('lip',$event)"><option value="">不关联</option><option v-for="f in state.files.filter(f=>f.kind==='lip')" :key="f.id" :value="f.id">{{f.name}}</option></select></label></div>
<details class="association-help" :open="legacyOpen" @toggle="legacyOpen=($event.target as HTMLDetailsElement).open"><summary>关联说明与旧文件兼容</summary>
 <p class="hint">TextGrid 与 .lip.json 同名自动关联，也可改选。历史参数表在下方切分区选择。</p>
 <template v-if="context.files.convertLip"><label class="legacy-field">旧唇形 PKL<select v-model="pickleId" aria-label="旧唇形PKL"><option value="">请选择旧文件</option><option v-for="f in state.files.filter(f=>f.kind==='lip_pickle'&&!/_timestamps\.pkl$/i.test(f.name))" :key="f.id" :value="f.id">{{f.name}}</option></select></label><button :disabled="converting||!pickleId||busy" @click="convertLip">{{converting?'正在转换…':'转换并保存 .lip.json'}}</button><p class="hint">在 PKL 同目录创建新文件并关联当前音频；已有文件不覆盖。自动读取同名 _timestamps.pkl 作为起点后备。仅转换时间与四项唇形参数，原文件中的 landmarks 等仍保留在 PKL。</p></template>
 <p v-else class="hint">旧唇形 PKL 请在桌面版转换为 .lip.json 后上传。</p>
</details><p v-if="linked.error" role="alert" class="error-banner">{{linked.error}}</p>
<p v-if="state.wave.loading" role="status" class="audio-loading"><progress aria-label="音频读取与解码进度"/>正在读取音频与关联…</p>
<template v-if="state.wave.asset"><WaveformViewport :state="state.wave" :spectrogram-loader="spectrogramLoader"><template #controls><label class="channel-picker">试听声道<select v-model.number="state.wave.channel" @change="stop"><option v-for="(_,i) in state.wave.asset.channels" :key="i" :value="i">声道 {{i+1}}</option></select></label></template></WaveformViewport>
<section v-if="linked.tiers.length" class="m01-tiers"><label>TextGrid切分层<select v-model.number="linked.layer"><option v-for="(tier,i) in linked.tiers" :key="i" :value="i">{{tier.name}}</option></select></label><div class="m01-intervals"><button v-for="(interval,i) in intervals" :key="i" @click="intervalSelect(interval.xmin,interval.xmax)"><span class="ipa-sample">{{interval.text||'（空标签）'}}</span><small>{{interval.xmin.toFixed(3)}}–{{interval.xmax.toFixed(3)}} s</small></button></div><p class="hint">当前层有 {{segments}} 个候选片段。勾选多项可批量切分；未勾选时处理当前音频。</p><label><input v-model="sliceParameters" type="checkbox"/>同时切分参数结果</label>
<template v-if="sliceParameters"><div class="legacy-source-row"><label>参数来源<select v-model="parameterSource" aria-label="切分参数来源"><option value="recent">本应用最近一次同源完整结果</option><option value="legacy">指定历史参数表（来源未核实）</option></select></label>
<label v-if="parameterSource==='legacy'">历史参数表<select aria-label="历史参数表关联" :value="linked.legacy?.id??''" @change="linked.legacy=state.files.find(f=>f.id===($event.target as HTMLSelectElement).value&&f.kind==='parameter')??null"><option value="">未指定</option><option v-for="f in state.files.filter(f=>f.kind==='parameter')" :key="f.id" :value="f.id">{{f.name}}</option></select></label></div>
<p class="hint">{{parameterSource==='recent'?'匹配相同原音频，保留原分析帧与时间，不重新估计。无参数结果时仅保存音频。':'请为每个待切分音频选择对应的完整历史表。旧表无法核实原 WAV 来源，输出会保留此标记；未关联时停止提交。'}}</p></template>
<button :disabled="!taskReady||taskBusy||busy" @click="startBatch('textgrid_segment')">保存当前层切分音频</button><small v-if="!tasks" class="muted">当前入口仅供预览；持久任务入口需本机任务服务或项目服务。</small></section>
</template></template><div v-else class="wave-empty"><AppIcon name="wave"/><h2>选择一段声音</h2><p>从左侧列表选择文件，查看真实波形、标签和试听选区。</p></div></template>
<template #settings><h2>输出参数</h2><div class="parameter-count"><strong>{{state.wave.parameters.length}}</strong><span>/ 80 项</span></div><button @click="openParameters">选择输出参数</button><hr/><h2>分析设置</h2><p class="mono muted">帧移 {{state.settings.frameshift_ms}} ms<br/>分析窗 {{state.settings.windowsize_ms}} ms</p><button @click="openSettings">编辑14项设置</button><button :disabled="!dirty(state)" @click="save">保存草稿</button><span v-if="dirty(state)" class="draft-indicator">● 尚未保存</span><hr/><div class="m01-batch-bar"><div><strong>处理列表中的 {{audioFiles.length}} 个文件</strong><p class="hint">使用全列表及当前参数设置；计算期间可以继续试听。</p></div><button class="primary" :disabled="!taskReady||taskBusy||busy||!audioFiles.length" @click="startBatch('acoustic_analysis')">开始全列表分析</button><span v-if="taskBusy" class="muted" role="status">正在处理任务操作…</span></div><hr/><h2>分析结果</h2><BatchResults :batches="batchList" :active="activeBatch" :jobs="jobResults" :busy="taskBusy" :desktop="context.files.kind==='desktop'" @select="loadBatch" @cancel="cancelBatch" @save="saveResults" @retry="retryJob" @download="downloadResult"/></template>
</WorkbenchColumns>

<ParameterDrawer v-if="state.drawer==='parameters'" :selected="state.wave.parameters" :draft="state.parameterDraft" require-selection @draft="state.parameterDraft=$event" @close="state.drawer=''" @apply="parameters"/>
<SettingsDrawer v-if="state.drawer==='settings'" :draft="state.settingsDraft" @draft="state.settingsDraft=$event" @close="state.drawer=''" @apply="settings"/>
</section></template>
<style scoped>
.legacy-field{display:flex;flex-direction:column;gap:6px;margin:10px 0}.legacy-field select{min-width:0;max-width:100%}.association-help p{overflow-wrap:anywhere}
.legacy-source-row{display:flex;flex-wrap:wrap;gap:12px}.legacy-source-row label{display:flex;flex:1 1 240px;min-width:0;flex-direction:column;gap:6px}.legacy-source-row select{min-width:0;max-width:100%}
</style>
