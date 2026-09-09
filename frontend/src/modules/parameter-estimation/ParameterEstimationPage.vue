<script setup lang="ts">
import {computed,ref,onMounted,onUnmounted,watch,markRaw} from 'vue';
import type {ResearchContext,ResearchFile,DirectoryGrant,SpectrogramView} from '../../platform/research.ts';
import {audioPreview} from '../../platform/research.ts';
import {m01State,saveM01} from './store.ts';
import {association,matchAssociation,reconcile,applyParameters,applySettings,dirty,effectiveOutput,type Settings} from './state.ts';
import {stop} from '../../state/audio.ts';
import WaveformViewport from '../../components/WaveformViewport.vue';
import ParameterDrawer from '../../components/ParameterDrawer.vue';
import SettingsDrawer from '../../components/SettingsDrawer.vue';
import AppIcon from '../../components/AppIcon.vue';
const props=defineProps<{context:ResearchContext}>();const emit=defineEmits<{references:[]}>();
const state=m01State(props.context.key),picker=ref<HTMLInputElement>(),busy=ref(false),notice=ref('');
const audioFiles=computed(()=>state.files.filter(f=>f.kind==='audio'));
const selected=computed(()=>audioFiles.value.find(f=>f.id===state.selected));
const spectrogramLoader=computed(()=>{const file=selected.value;return props.context.files.spectrogram&&file?(view:SpectrogramView)=>props.context.files.spectrogram!(file,view):undefined;});
const linked=computed(()=>association(state));
const intervals=computed(()=>linked.value.tiers[linked.value.layer]?.intervals??[]);
const segments=computed(()=>intervals.value.filter(i=>i.xmax>i.xmin&&!['','sil','eps','<sil>','<eps>'].includes(i.text.trim().toLowerCase())).length);
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
function openParameters(){state.parameterDraft=[...state.wave.parameters];state.drawer='parameters';}
function openSettings(){state.settingsDraft={...state.settings};state.drawer='settings';}
function parameters(keys:string[]){try{applyParameters(state,keys);}catch(e){notice.value=String(e);}}
function settings(value:Settings){applySettings(state,value);}
function save(){notice.value=saveM01(props.context.key)?'参数与设置已保存。文件、目录权限和账号凭据不会写入草稿。':'草稿保存失败，请检查本机存储权限。';}
function intervalSelect(xmin:number,xmax:number){stop();if(!state.wave.asset)return;state.wave.start=Math.max(0,Math.min(xmin,state.wave.asset.duration));state.wave.end=Math.max(state.wave.start,Math.min(xmax,state.wave.asset.duration));}
onMounted(()=>{if(props.context.files.kind!=='desktop'||state.input)void refresh();});
onUnmounted(()=>{disposed=true;previewAbort.abort();state.loadVersion++;state.listVersion++;state.wave.loading=false;stop();});
</script>
<template>
<section class="workspace-page m01-page" aria-label="参数估计 工作区">
<header class="page-heading"><div><p class="eyebrow">M01 · {{context.label}}</p><h1>参数估计</h1><p class="muted">组织音频与关联，设置下一次分析。</p></div><button @click="emit('references')"><AppIcon name="book"/>方法与引用</button></header>
<div class="m01-directory-bar">
<template v-if="context.files.kind==='desktop'"><button class="primary" :disabled="busy" @click="choose('input')">选择音频目录</button><span class="directory-label">{{state.input?.label??'尚未选择目录'}}</span><button :disabled="busy" @click="choose('association')">选择关联目录</button><span v-if="state.associationDirectory" class="directory-label">{{state.associationDirectory.label}}</span></template>
<template v-else-if="context.files.kind==='preview'"><input ref="picker" class="visually-hidden" type="file" accept=".wav" multiple aria-label="选择音频列表" @change="add"/><button class="primary" @click="picker?.click()">添加WAV到列表</button><span class="hint">本机预览 · 不上传</span></template>
<template v-else><span>项目资源 · {{context.label}}</span><small class="muted">在项目文件管理中上传WAV、TextGrid与.lip.json，再刷新。</small></template>
<button :disabled="busy||(context.files.kind==='desktop'&&!state.input)" @click="refresh">{{busy?'正在读取…':'刷新列表'}}</button>
</div>
<div v-if="context.files.kind==='desktop'" class="m01-output-row"><label><input v-model="state.sameDirectory" type="checkbox"/>结果与WAV同目录</label><button :disabled="state.sameDirectory||busy" @click="choose('output')">选择结果目录</button><span class="directory-label">{{effectiveOutput(state)?.label??'尚未选择结果目录'}}</span></div>
<p v-if="notice" role="status" class="notice">{{notice}}</p><p v-if="state.wave.error" role="alert" class="error-banner">{{state.wave.error}}</p>
<div class="workbench-grid">
<aside class="file-panel"><div class="panel-heading"><h2>音频列表</h2><small>{{audioFiles.length}} 个文件</small></div><p class="hint">点选用于试听；批处理覆盖整个列表。</p>
<div class="m01-select-all"><label><input aria-label="全选音频" type="checkbox" :checked="audioFiles.length>0&&state.marked.length===audioFiles.length" :indeterminate="state.marked.length>0&&state.marked.length<audioFiles.length" :disabled="!audioFiles.length" @change="state.marked=($event.target as HTMLInputElement).checked?audioFiles.map(f=>f.id):[]"/>全选</label><small>切分选择 {{state.marked.length}} / {{audioFiles.length}}</small></div><div class="m01-file-list"><div v-for="file in audioFiles" :key="file.id" class="m01-file-entry"><input v-model="state.marked" type="checkbox" :value="file.id" :aria-label="'选择切分 '+file.name"/><button class="file-row" :class="{selected:state.selected===file.id}" :aria-pressed="state.selected===file.id" @click="selectFile(file)"><AppIcon name="file"/><span>{{file.name}}<small>{{association(state,file.id).textgrid?'TextGrid · ':''}}{{association(state,file.id).lip?'唇形关联 · ':''}}{{Math.ceil(file.size/1024)}} KB</small><small v-if="association(state,file.id).error" class="danger-text">关联待确认</small></span></button></div></div>
<p v-if="!audioFiles.length" class="empty-small">选择音频目录或刷新项目资源。</p></aside>
<div class="signal-panel"><template v-if="selected"><div class="signal-heading"><h2>{{selected.name}}</h2><p v-if="state.wave.asset" class="mono muted">{{state.wave.asset.sampleRate}} Hz · {{state.wave.asset.channels.length}} 声道 · {{state.wave.asset.duration.toFixed(3)}} s</p></div>
<div class="m01-association-controls"><label>TextGrid<select aria-label="TextGrid关联" :value="linked.textgrid?.id??''" :disabled="state.wave.loading" @change="changeAssociation('textgrid',$event)"><option value="">不关联</option><option v-for="f in state.files.filter(f=>f.kind==='textgrid')" :key="f.id" :value="f.id">{{f.name}}</option></select></label><label>唇形数据<select aria-label="唇形关联" :value="linked.lip?.id??''" @change="changeAssociation('lip',$event)"><option value="">不关联</option><option v-for="f in state.files.filter(f=>f.kind==='lip')" :key="f.id" :value="f.id">{{f.name}}</option></select></label></div>
<p class="hint">同名自动关联；可明确改选。唇形使用安全.lip.json格式，旧PKL需先经受限转换。</p><p v-if="linked.error" role="alert" class="error-banner">{{linked.error}}</p>
<p v-if="state.wave.loading" role="status">正在读取音频与关联…</p>
<template v-if="state.wave.asset"><label class="channel-picker">试听声道<select v-model.number="state.wave.channel" @change="stop"><option v-for="(_,i) in state.wave.asset.channels" :key="i" :value="i">声道 {{i+1}}</option></select></label><WaveformViewport :state="state.wave" :spectrogram-loader="spectrogramLoader"/>
<section v-if="linked.tiers.length" class="m01-tiers"><label>TextGrid切分层<select v-model.number="linked.layer"><option v-for="(tier,i) in linked.tiers" :key="i" :value="i">{{tier.name}}</option></select></label><div class="m01-intervals"><button v-for="(interval,i) in intervals" :key="i" @click="intervalSelect(interval.xmin,interval.xmax)"><span class="ipa-sample">{{interval.text||'（空标签）'}}</span><small>{{interval.xmin.toFixed(3)}}–{{interval.xmax.toFixed(3)}} s</small></button></div><p class="hint">点选区间同步试听选区。当前层有 {{segments}} 个候选片段（跳过空白、sil、eps标签）。</p><button disabled>保存当前层切分音频</button><small class="muted">切分写入将在任务接入后启用。</small></section>
</template></template><div v-else class="wave-empty"><AppIcon name="wave"/><h2>选择一段声音</h2><p>从左侧列表选择文件，查看真实波形、标签和试听选区。</p></div></div>
<aside class="parameter-summary"><h2>输出参数</h2><div class="parameter-count"><strong>{{state.wave.parameters.length}}</strong><span>/ 80 项</span></div><button @click="openParameters">选择输出参数</button><hr/><h2>分析设置</h2><p class="mono muted">帧移 {{state.settings.frameshift_ms}} ms<br/>分析窗 {{state.settings.windowsize_ms}} ms</p><button @click="openSettings">编辑14项设置</button><button :disabled="!dirty(state)" @click="save">保存草稿</button><span v-if="dirty(state)" class="draft-indicator">● 尚未保存</span><hr/><h2>分析结果</h2><p class="empty-small">暂无计算结果</p></aside>
</div>
<div class="m01-batch-bar"><div><strong>处理列表中的 {{audioFiles.length}} 个文件</strong><p class="hint">与当前试听文件、TextGrid切分层分别操作。</p></div><button disabled>开始全列表分析</button><span class="muted">计算、取消与双格式结果发布将在下一阶段接入。</span></div>
<ParameterDrawer v-if="state.drawer==='parameters'" :selected="state.wave.parameters" :draft="state.parameterDraft" require-selection @draft="state.parameterDraft=$event" @close="state.drawer=''" @apply="parameters"/>
<SettingsDrawer v-if="state.drawer==='settings'" :draft="state.settingsDraft" @draft="state.settingsDraft=$event" @close="state.drawer=''" @apply="settings"/>
</section></template>
