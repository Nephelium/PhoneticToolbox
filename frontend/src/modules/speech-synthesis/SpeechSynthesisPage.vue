<script setup lang="ts">
import AppIcon from '../../components/AppIcon.vue';
import PlotPlaceholder from '../../components/PlotPlaceholder.vue';
import {vTimePrecision} from '../../design/time-precision.ts';
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import {computed,ref,reactive,watch,onMounted,onUnmounted,markRaw} from 'vue';
import type {ResearchContext,ResearchFile,JobView} from '../../platform/research.ts';
import {workspace,host} from '../../state/workspace.ts';import {stop} from '../../state/audio.ts';import {decodeWav} from '../../platform/decode.ts';
import ModuleFrame from '../../components/ModuleFrame.vue';import ModuleToolbar from '../../components/ModuleToolbar.vue';import ModuleSection from '../../components/ModuleSection.vue';import ModuleStatus from '../../components/ModuleStatus.vue';import ModalDialog from '../../components/ModalDialog.vue';import WaveformViewport from '../../components/WaveformViewport.vue';import AudioTransport from '../../components/AudioTransport.vue';import TaskPanel from '../../components/TaskPanel.vue';
import CurveEditor from './CurveEditor.vue';import SynthesisSpectrum from './SynthesisSpectrum.vue';
import {defaults,parameters,vowels,presets,f0Methods,clone,valid,number,ipa,override,resize,f0Range,preset,exportParams,importParams,isCurrent,type Config} from './state.ts';
import {copyText} from '../../platform/clipboard.ts';
import type {Action,Result,Raster} from './port.ts';import {downloadBytes} from '../../platform/m06.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active:boolean}>();const emit=defineEmits<{references:[]}>();
const wave=workspace(props.stateKey),config=reactive(defaults()),name=ref('F0'),overrideText=ref(''),durationText=ref('2'),low=ref('50'),high=ref('500');
const error=ref(''),notice=ref(''),busy=ref(false),revision=ref(0),resultRevision=ref(-1),result=ref<Result>(),job=ref<JobView>();
const help=ref(false),rules=ref(false),pending=ref<(()=>void)|null>(null),pendingLabel=ref(''),selectedPreset=ref('常态浊声');
const files=ref<ResearchFile[]>([]),source=ref<ResearchFile>(),directory=ref(''),audioPicker=ref<HTMLInputElement>(),paramPicker=ref<HTMLInputElement>();
const view=ref('wave'),windowMs=ref(20);let epoch=0,disposed=false,abort:AbortController|undefined;
const sourceWave=workspace(props.stateKey+'.m06-source'),previewTarget=ref('result'),sourceSpectra=ref<Record<string,Raster>>({});
const previewWave=computed(()=>previewTarget.value==='source'?sourceWave:wave),spectra=computed(()=>previewTarget.value==='source'?sourceSpectra.value:result.value?.metadata.spectrograms);
const port=computed(()=>props.context.files.m06),stale=computed(()=>!!result.value&&resultRevision.value!==revision.value);
const timelineDuration=computed(()=>previewTarget.value==='source'&&sourceWave.asset?sourceWave.asset.duration:config.duration);
const activeRange=computed<[number,number]>(()=>{const p=previewWave.value,length=timelineDuration.value/p.zoom,start=Math.max(0,Math.min(p.offset,timelineDuration.value-length));return [start,start+length];});
const tasks=computed(()=>job.value?[{id:job.value.id,title:'声学参数合成 · '+job.value.id.slice(0,8),status:job.value.state,progress:job.value.progress,error:job.value.error_code??undefined,canCancel:true}]:[]);
watch(config,()=>{revision.value++;wave.dirty=true;stop();},{deep:true,flush:'sync'});
watch(()=>props.active,v=>{if(!v)stop();});
watch(name,()=>syncOverride());
watch(source,()=>editPending(),{flush:'sync'});
function editPending(){revision.value++;wave.dirty=true;stop();}
function requireApplied(){if(Number(durationText.value)!==config.duration||Number(low.value)!==config.f0_range[0]||Number(high.value)!==config.f0_range[1])throw Error('请先应用时长或 F0 范围');const v=config.curves[name.value].override;const expected=v===null?'':String(v*(name.value==='Shimmer'?100:1));if(overrideText.value!==expected)throw Error('请先应用曲线覆盖');}
const copyNotice=ref(''),copyError=ref(''),copyValue=ref('');
async function copySymbol(s:string){copyValue.value=s;copyNotice.value='';copyError.value='';try{await copyText(s);copyNotice.value='已复制 '+s;}catch{copyError.value='复制失败，请选中音标后按 Ctrl+C。';}}
function syncOverride(){const v=config.curves[name.value].override;overrideText.value=v===null?'':String(v*(name.value==='Shimmer'?100:1));}
function selectParameter(next:string){void attempt(()=>{override(config,name.value,overrideText.value);name.value=next;syncOverride();});}
function syncInputs(){syncOverride();durationText.value=String(config.duration);low.value=String(config.f0_range[0]);high.value=String(config.f0_range[1]);}
function attempt(work:()=>void|Promise<void>){error.value='';return Promise.resolve().then(work).catch(e=>{if(!disposed)error.value=e instanceof Error?e.message:String(e);});}
function replace(c:Config){Object.assign(config,valid(c));syncInputs();resetRange();}
function resetRange(){wave.zoom=1;wave.offset=0;sourceWave.zoom=1;sourceWave.offset=0;}
function setRange(a:number,b:number){if(b<=a)return;const p=previewWave.value;p.zoom=timelineDuration.value/(b-a);p.offset=a;}
function applyOverride(){return attempt(()=>{override(config,name.value,overrideText.value);syncOverride();});}
function applyDuration(){return attempt(()=>{resize(config,number(durationText.value,'时长'));durationText.value=String(config.duration);resetRange();});}
function applyRange(){return attempt(()=>{f0Range(config,number(low.value,'F0 下限'),number(high.value,'F0 上限'));low.value=String(config.f0_range[0]);high.value=String(config.f0_range[1]);});}
function confirm(label:string,work:()=>void){pendingLabel.value=label;pending.value=work;}
function commitPending(){const work=pending.value;pending.value=null;if(work)void attempt(work);}
function clear(){confirm('恢复全部参数，清除曲线覆盖与静音区间',()=>{const c=defaults();resize(c,config.duration);c.sample_rate=config.sample_rate;c.sequence=config.sequence;c.fade_in=config.fade_in;c.fade_out=config.fade_out;c.smooth=config.smooth;replace(c);});}
function applyPreset(){confirm('应用“'+selectedPreset.value+'”声源参数，保留 F0 曲线形状；假声和嘎裂只作整体平移',()=>{requireApplied();preset(config,selectedPreset.value);syncInputs();const shift=config.f0_transform.offset_hz;notice.value=shift===0?'发声预设已应用，F0 使用基础曲线':'发声预设已应用，F0 整体'+(shift>0?'上移 ':'下移 ')+Math.abs(shift).toFixed(1)+' Hz，可继续编辑';});}
function save(){try{valid(config);if(Number(durationText.value)!==config.duration||Number(low.value)!==config.f0_range[0]||Number(high.value)!==config.f0_range[1])throw Error('请先应用时长或 F0 范围');const c=clone(config);override(c,name.value,overrideText.value);valid(c);if(!host.projects.write('m06.draft.'+props.stateKey,c))throw Error('草稿保存失败');if(JSON.stringify(c)!==JSON.stringify(config))replace(c);wave.dirty=false;notice.value='完整参数草稿已保存';return true;}catch(e){error.value=String(e);return false;}}
defineExpose({save});
async function refresh(){const next=(await props.context.files.list(directory.value||undefined)).filter(f=>/\.wav$/i.test(f.name));files.value=next;if(sourceId.value&&!next.some(f=>f.id===sourceId.value&&(!source.value||f.sha256===source.value.sha256)))await selectSource('');}
async function choose(){await attempt(async()=>{const grant=await props.context.files.choose?.('input');if(grant){directory.value=grant.id;await refresh();}});}
async function addAudio(event:Event){await attempt(async()=>{const target=event.target as HTMLInputElement;props.context.files.add?.([...target.files??[]]);target.value='';await refresh();});}
const sourceId=ref(''),sourceLoading=ref(false);let loadEpoch=0,loadAbort:AbortController|undefined;
async function loadSource(event:Event){await selectSource((event.target as HTMLSelectElement).value);}
async function selectSource(id:string){
 const ticket=++loadEpoch;loadAbort?.abort();loadAbort=new AbortController();sourceId.value=id;
 source.value=undefined;sourceWave.asset=null;sourceSpectra.value={};sourceLoading.value=!!id;error.value='';notice.value='';stop();
 const f=files.value.find(f=>f.id===id);if(!f){sourceLoading.value=false;previewTarget.value='result';return;}
 try{const raw=await props.context.files.read(f,loadAbort.signal);if(f.sha256&&raw.sha256!==f.sha256)throw Error('音频已变化，请刷新文件后重新选择。');const asset=await decodeWav(raw.buffer,f.name,loadAbort.signal);
  if(disposed||ticket!==loadEpoch)return;
  source.value=f;sourceWave.asset=markRaw(asset);sourceWave.start=0;sourceWave.end=asset.duration;sourceWave.channel=0;sourceWave.zoom=1;sourceWave.offset=0;
  previewTarget.value='source';view.value='wave';notice.value='音频已加载，点击提取参数开始提取。';
 }catch(e){if(!disposed&&ticket===loadEpoch){sourceId.value='';error.value=e instanceof Error?e.message:String(e);}}
 finally{if(ticket===loadEpoch)sourceLoading.value=false;}
}
async function run(action:Action){await attempt(async()=>{
 if(!port.value)throw Error('当前宿主尚未提供 M06 科学任务能力');if(busy.value)throw Error('请等待当前任务完成或取消');
 requireApplied();const submitted=clone(valid(config));if(action==='generate')ipa(submitted.sequence);if(action==='extract'&&(!source.value||sourceLoading.value))throw Error('请先加载 WAV 音频');
 if(submitted.duration>10||submitted.duration*submitted.sample_rate>480000)throw Error('当前已验证任务预算为 10 秒且 480,000 样本，请缩短输入');
 const input=source.value;const rev=revision.value,ticket=++epoch;abort=new AbortController();busy.value=true;notice.value='';
 try{const completed=await port.value.run(action,submitted,input,abort.signal,j=>{if(!disposed&&ticket===epoch)job.value=j;});
  if(disposed||!isCurrent(rev,revision.value,ticket,epoch)){notice.value='任务完成期间配置已改变，迟到结果未应用。原结果保留。';return;}
  if(action==='synthesize'){
   const asset=await decodeWav(completed.wav!,'合成结果.wav');if(disposed||!isCurrent(rev,revision.value,ticket,epoch))return;
   previewTarget.value='result';result.value=completed;resultRevision.value=rev;wave.asset=markRaw(asset);wave.start=0;wave.end=asset.duration;wave.channel=0;resetRange();notice.value='合成完成，试听与导出绑定本次参数快照'+((completed.metadata.diagnostics?.output_gain??1)<1?'；输出超过安全幅度，已整体衰减 '+(-20*Math.log10(completed.metadata.diagnostics!.output_gain!)).toFixed(1)+' dB':'')+(completed.metadata.diagnostics?.voiced_frames===0?'；所选算法未检出有声帧，请试听核查，必要时切换算法或调整 F0 范围。':'');
  }else{if(action==='extract'&&input){const raw=await props.context.files.read(input,abort.signal);if(raw.sha256!==completed.metadata.input_sha256)throw Error('源音频在提取后已变化，未应用参数');const asset=await decodeWav(raw.buffer,input.name);if(disposed||!isCurrent(rev,revision.value,ticket,epoch))return;sourceWave.asset=markRaw(asset);sourceWave.start=0;sourceWave.end=asset.duration;sourceWave.channel=0;sourceSpectra.value=completed.metadata.spectrograms;previewTarget.value='source';}replace(completed.metadata.config);notice.value=action==='generate'?'元音曲线已生成，请点击合成音频':'参数提取完成（'+f0Methods[completed.metadata.config.f0_method]+'）。AV 按相对能量初始化，AH 为 0，请按需要调整声源后合成'+(completed.metadata.diagnostics?.voiced_mask?.some(Boolean)===false?'。未检出有声帧，AV 全为 0；可调整 F0 范围或切换算法后重新提取':'');}
 }finally{if(ticket===epoch)busy.value=false;}
 });}
async function cancel(){abort?.abort();if(job.value&&port.value)await attempt(async()=>{await port.value!.cancel(job.value!.id);});}
async function exportAudio(){await attempt(async()=>{if(!result.value)throw Error('暂无合成结果');await port.value!.download(result.value,'synthesis.wav');notice.value=stale.value?'已导出旧结果及其实际合成快照':'已导出合成结果';});}
function exportCurrent(){void attempt(()=>{requireApplied();downloadBytes(exportParams(config),'声学参数合成参数.csv');});}
async function importFile(event:Event){await attempt(async()=>{const target=event.target as HTMLInputElement,file=target.files?.[0];target.value='';if(!file)return;if(file.size>8_000_000)throw Error('参数文件超过 8 MB');const c=importParams(await file.text());confirm('导入参数将覆盖当前完整配置',()=>replace(c));});}
onMounted(()=>{const draft=host.projects.read<Config|null>('m06.draft.'+props.stateKey,null);if(draft){try{replace(draft);wave.dirty=false;}catch(e){error.value=(e instanceof Error?e.message:'旧草稿无效')+' 已保留存储内容并使用默认参数。';}}void attempt(refresh);});
onUnmounted(()=>{loadEpoch++;loadAbort?.abort();sourceWave.asset=null;disposed=true;epoch++;abort?.abort();stop();});
</script>
<template>
<ModuleFrame unified fit label="声学参数合成工作区" class="m06-page" :aria-busy="busy">
 <template #toolbar><ModuleToolbar>
  <button class="primary" v-if="context.files.choose" @click="choose"><AppIcon name="folder"/>打开音频目录</button><button class="primary" v-else @click="audioPicker?.click()">加载音频</button><button @click="attempt(refresh)">刷新文件</button>
  <input ref="audioPicker" hidden type="file" accept=".wav" @change="addAudio"/>
  <label class="preset-picker">F0 算法 <select v-model="config.f0_method" aria-label="F0 提取算法" title="用于下一次提取；范围使用左侧已应用的 F0 上下限"><option v-for="(label,key) in f0Methods" :key="key" :value="key">{{label}}</option></select></label>
  <button class="primary" :disabled="busy||sourceLoading||!source||!port" @click="run('extract')">提取参数</button><button @click="paramPicker?.click()">导入参数</button>
  <input ref="paramPicker" hidden type="file" accept=".csv,.json" @change="importFile"/><button @click="exportCurrent">导出参数</button>
  <label class="preset-picker">发声类型 <select v-model="selectedPreset" aria-label="发声类型预设"><option v-for="p in Object.keys(presets)" :key="p">{{p}}</option></select></label><button @click="applyPreset">应用预设</button>
  <template #actions><button @click="save">保存参数草稿</button><button @click="help=true">帮助</button><button @click="emit('references')"><AppIcon name="book"/>方法与引用</button></template>
 </ModuleToolbar></template>
 <ModuleWorkbench unified :state-key="stateKey" left-label="合成参数" center-label="参数曲线与音频" right-label="合成与任务">
  <template #left>
   <ModuleSection label="音频文件" class="source-section"><label>音频文件<select class="source-picker" aria-label="音频文件" :value="sourceId" @change="loadSource"><option value="">请选择音频文件</option><option v-for="f in files" :key="f.id" :value="f.id">{{f.name}}</option></select></label><p v-if="sourceLoading" role="status">正在加载音频…</p></ModuleSection>
   <ModuleSection label="合成基础设置" class="base-settings"><div class="controls">
    <label>总时长 (s)<input v-time-precision="'s'" v-model="durationText" aria-label="总时长" @input="editPending"/></label><button @click="applyDuration">应用时长</button>
    <label>淡入 (ms)<input v-time-precision="'ms'" v-model.number="config.fade_in" type="number" min="0" max="1000"/></label><label>淡出 (ms)<input v-time-precision="'ms'" v-model.number="config.fade_out" type="number" min="0" max="1000"/></label>
    <label>平滑点数<input v-model.number="config.smooth" type="number" min="1" max="50"/></label>
    <label>F0 下限 (Hz)<input v-model="low" aria-label="F0 下限" @input="editPending"/></label><label>F0 上限 (Hz)<input v-model="high" aria-label="F0 上限" @input="editPending"/></label><button @click="applyRange">应用 F0 范围</button>
   </div></ModuleSection>
   <ModuleSection label="参数列表" class="parameters-section"><div class="parameter-list"><button v-for="(definition,key) in parameters" :key="key" :aria-pressed="name===key" @click="selectParameter(String(key))">{{key}} <small>{{definition[3]}}</small></button></div></ModuleSection>
  </template>
  <ModuleSection label="参数曲线" class="curve-section">
   <div class="controls curve-controls"><label>曲线覆盖<input v-model="overrideText" aria-label="曲线覆盖" placeholder="输入数值或分段序列" title="单个数值固定全段；逗号连接渐变值，分号分隔等长区段" @input="editPending"/></label><button @click="applyOverride">应用覆盖</button><button @click="overrideText='';applyOverride()">清除覆盖</button><button @click="resetRange">重置范围</button><button @click="clear">清空参数</button></div>
   <CurveEditor :config="config" :name="name" :start="activeRange[0]" :end="activeRange[1]" @range="setRange" @error="error=$event"/>
  </ModuleSection>
  <ModuleSection label="音频预览" class="preview-section">
   <div class="controls preview-controls"><button :aria-pressed="previewTarget==='result'" @click="previewTarget='result';stop()">合成结果</button><button v-if="sourceWave.asset" :aria-pressed="previewTarget==='source'" @click="previewTarget='source';stop()">源音频</button><button :aria-pressed="view==='wave'" @click="view='wave'">波形</button><button :aria-pressed="view==='spectrum'" @click="view='spectrum'">语谱图</button><label v-if="view==='spectrum'">窗长<select v-model.number="windowMs" aria-label="语谱窗长"><option v-for="ms in [5,10,20,40]" :key="ms" :value="ms">{{ms}} ms</option></select></label></div>
   <WaveformViewport v-if="previewWave.asset&&view==='wave'" :state="previewWave" :timeline-duration="timelineDuration" class="synthesis-wave" compact-overview hide-overview-controls auto-amplitude continuous-detail>
    <template #annotations="{x,start,end}"><path v-for="t in config.boundaries.filter(t=>t>=start&&t<=end)" :key="t" class="vowel-boundary" :d="`M${x(t)},0V90`"/></template>
   </WaveformViewport>
   <SynthesisSpectrum v-else-if="spectra?.[String(windowMs)]&&view==='spectrum'" :data="spectra[String(windowMs)]" :start="activeRange[0]" :end="activeRange[1]" :window-ms="windowMs" :boundaries="config.boundaries"/>
   <PlotPlaceholder v-else :message="previewTarget==='source'&&sourceWave.asset?'源音频已加载，可试听波形；提取参数后提供语谱图。':'暂无合成音频。生成元音后，点击合成音频即可试听和导出。'"/>
  </ModuleSection>
  <template #right>
   <ModuleSection label="合成操作"><div class="controls synthesis-actions">
    <label class="ipa-input">IPA 元音序列<input v-model="config.sequence" class="ipa-text" aria-label="IPA 元音序列" placeholder="a-i/- ///u+/e++o/-"/></label>
    <button @click="rules=true">元音规则</button><button class="primary" :disabled="busy||!port" @click="run('generate')">生成元音</button>
    <button class="primary" :disabled="busy||!port" @click="run('synthesize')">合成音频</button><button class="primary" :disabled="!result" @click="exportAudio">导出音频</button><button v-if="busy" @click="cancel">取消任务</button>
   </div></ModuleSection>
   <ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-if="stale" kind="info" message="参数已修改，需重新合成。试听和音频导出仍对应旧结果。"/><ModuleStatus v-if="notice" kind="info" :message="notice"/><ModuleStatus v-if="!port" kind="info" message="当前为参数编辑模式，尚无可用的合成任务服务。"/>
   <details :open="busy" class="task-details"><summary>合成任务状态</summary><span v-if="result">结果 {{result.job.slice(0,8)}} · 随机种子 {{result.metadata.seed}}</span><TaskPanel :tasks="tasks" empty-title="尚未提交合成任务" empty-text="参数编辑不会自动触发音频合成" @cancel="cancel"/></details>
  </template>
 </ModuleWorkbench>
 <div class="m06-transport global-transport"><AudioTransport :state="previewWave" :active="active"/></div>
 <ModalDialog v-if="pending" title="覆盖参数确认" @close="pending=null"><p>{{pendingLabel}}。现有合成音频保留并标记为旧结果。</p><template #footer><button @click="pending=null">取消</button><button @click="commitPending">应用并覆盖</button></template></ModalDialog>
 <ModalDialog v-if="rules" title="元音规则" @close="rules=false"><p class="copy-feedback" role="status">{{copyNotice}}</p><div v-if="copyError" class="copy-fallback"><p role="alert">{{copyError}}</p><input :value="copyValue" readonly aria-label="待复制音标" @focus="($event.target as HTMLInputElement).select()"/></div><table class="vowel-rules"><thead><tr><th>IPA</th><th>F1 Hz</th><th>F2 Hz</th><th>F3 Hz</th></tr></thead><tbody><tr v-for="(f,s) in vowels" :key="s"><td><button class="ipa-text" @click="copySymbol(s)">{{s}}</button></td><td v-for="(v,i) in f" :key="i">{{v}}</td></tr></tbody></table><p>点击音标复制。空格为静音段。+、-、*、/ 依次将相对时长乘以 1.1、0.9、2、0.5，再按总时长归一。生成元音更新共振峰与静音段，保留当前发声预设和 F0 曲线。</p></ModalDialog>
 <ModalDialog v-if="help" title="声学参数合成帮助" @close="help=false"><p>设置并应用时长 → 输入 IPA 并生成元音曲线 → 编辑参数 → 合成音频 → 试听或导出 WAV。</p><p>Shift 拖动绘制，Ctrl 拖动恢复默认值，普通拖动平移，Ctrl 滚轮缩放。曲线覆盖支持单个数值、逗号渐变和分号分段。清除覆盖可恢复手绘。Shimmer 显示百分数，文件保留小数。F1–F5 的虚线是其他共振峰的只读参考。</p><p>应用时长立即更新上下图时间轴。旧音频按真实时间显示，超出音频尾部留空，需重新合成才能得到新时长的音频。</p><p>导出参数保存当前编辑配置。导出音频保存上次合成结果，桌面目录保存同时包含实际合成参数与元数据。完整 CSV 保留采样率、淡入淡出、静音、覆盖下的原曲线。</p><p>加载 WAV 前可在顶栏选择 F0 提取算法：Praat 互相关 CC、自相关 AC或 REAPER。切换后点击提取参数重新计算，范围使用左侧已应用的 F0 上下限。REAPER 不可用时会报错，不自动改用其他算法。旧 m06/2 参数文件默认采用原来的 Praat CC。</p><p>底部时间选区、播放进度和音量跟随当前预览的源音频或合成结果。</p><p>左侧选择 WAV 只加载预览，点击顶栏提取参数后才更新曲线。未加载音频时无法提取，源录音采样率保留。复制合成与五类发声预设效果有限，输入参数不保证等于输出的声学测量值。AV 和 AH 分别控制周期声源与气流噪声，范围 0–80 dB，0 关闭对应声源。60 dB 使用本工具的数字参考标尺，不代表物理声压级。HNR、H1–H2、Slope 为扩展处理，输入值不保证等于输出测量值。假声将基础 F0 曲线整体抬高至平均至少 300 Hz，嘎裂整体降低至约 70 Hz并保留最低 20 Hz。常态浊声、耳语和气声撤去偏移，平移期间的编辑仍保留。重复应用同一预设不会累加偏移。当前任务上限为 10 秒且 480,000 样本。</p></ModalDialog>
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
