<script setup lang="ts">
import AppIcon from '../../components/AppIcon.vue';
import {ref,computed,watch,onUnmounted} from 'vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import ModuleSection from '../../components/ModuleSection.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import ModalDialog from '../../components/ModalDialog.vue';
import SymbolEditor from './SymbolEditor.vue';
import ResultPreview from './ResultPreview.vue';
import type {ResearchContext} from '../../platform/research.ts';
import {projects} from '../../platform/browser.ts';
import {exportFontSnapshot} from '../../state/fonts.ts';
import type {Settings,Preview,Source,Result,Options,Inspection} from './port.ts';
import {clone,swap,defaultOptions} from './state.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active?:boolean}>();
const emit=defineEmits<{references:[];dirty:[boolean]}>();
const importEpoch=ref(0),step=ref(0),error=ref(''),message=ref(''),busy=ref(false),help=ref(false),dirty=ref(false);
const fileInput=ref<HTMLInputElement>(),source=ref<Source>(),preview=ref<Preview>(),settings=ref<Settings>(),result=ref<Result>();
const options=ref<Options>(defaultOptions()),importedOptions=ref<Options>();
const staged=ref<Source>(),inspection=ref<Inspection>(),inspectedSignature=ref('');
const toneDraft=ref<Settings>(),toneDrag=ref<string>(),symbolsDirty=ref(false),symbolEditor=ref<InstanceType<typeof SymbolEditor>>();
const snapshot=ref<{preview:Preview;settings:Settings;signature:string}>();let controller:AbortController|undefined;
const port=computed(()=>props.context.files.m14),key='m14:'+props.stateKey;
const saved=projects.read<any>(key,null);
if(['m14-draft/1','m14-draft/2'].includes(saved?.schema)&&saved.source&&saved.preview&&saved.settings){source.value=saved.source;preview.value=saved.preview;settings.value=saved.settings;importedOptions.value=saved.options;toneDraft.value=clone(saved.settings);options.value={...defaultOptions(),...(saved.preview.computation_revision==='m14/2'?saved.options:{})};step.value=saved.preview.computation_revision==='m14/2'?1:0;message.value=saved.preview.computation_revision==='m14/2'?'已恢复本机草稿。':'已恢复旧版草稿，请重新选择文件导入以使用修正后的声韵解析。';}
const legacy=computed(()=>!!preview.value&&preview.value.computation_revision!=='m14/2');
const toneChanged=computed(()=>!!toneDraft.value&&!!settings.value&&(JSON.stringify(toneDraft.value.tone_order)!==JSON.stringify(settings.value.tone_order)||JSON.stringify(toneDraft.value.tone_map)!==JSON.stringify(settings.value.tone_map)));
const editing=computed(()=>toneChanged.value||symbolsDirty.value);
const signature=computed(()=>JSON.stringify([source.value,importedOptions.value,settings.value]));
const stale=computed(()=>!!snapshot.value&&(snapshot.value.signature!==signature.value||editing.value));
const inspectSignature=()=>JSON.stringify([options.value.table_index,options.value.encoding,options.value.delimiter,options.value.start_row]);
const needsRefresh=computed(()=>!!inspection.value&&inspectedSignature.value!==inspectSignature());
const columns=computed(()=>Array.from({length:inspection.value?.column_count??3},(_,i)=>i+1));
const isText=computed(()=>/\.(csv|txt|tsv)$/i.test(staged.value?.name??''));
const mappingError=computed(()=>{if(!inspection.value)return '';const c=options.value.character_column!,i=options.value.ipa_column!,n=options.value.note_column,s=options.value.start_row!;const chosen=[c,i,...(n==null?[]:[n])];if(chosen.some(v=>!Number.isInteger(v)||v<1||v>inspection.value!.column_count)||new Set(chosen).size!==chosen.length)return '字头、IPA 与备注须选择不同的有效列。';if(!Number.isInteger(s)||s<1||s>10001)return '开始行须为 1–10,001 的整数。';if(s>inspection.value.total_rows&&inspection.value.sample.every(r=>r.row<s))return '开始行超出此表的数据范围。';return '';});
watch(()=>dirty.value||busy.value||editing.value||!!staged.value,value=>emit('dirty',value),{flush:'sync'});
onUnmounted(()=>controller?.abort());
function changed(){dirty.value=true;}
async function operation(run:(signal:AbortSignal)=>Promise<void>){if(busy.value)return;error.value='';message.value='';busy.value=true;controller=new AbortController();try{await run(controller.signal);}catch(e){error.value=e instanceof Error?e.message:'操作失败，请重试。';}finally{busy.value=false;controller=undefined;}}
async function selectFile(event:Event){const input=event.target as HTMLInputElement,file=input.files?.[0];input.value='';if(!file||!port.value)return;const opts={...clone(options.value),table_index:0};await operation(async signal=>{const value=await port.value!.inspect(file,opts,signal);if(signal.aborted)return;staged.value=value.source;inspection.value=value.inspection;options.value=opts;const width=value.inspection.column_count;if(options.value.character_column!>width)options.value.character_column=1;if(options.value.ipa_column!>width)options.value.ipa_column=2;if(options.value.note_column!=null&&options.value.note_column>width)options.value.note_column=null;inspectedSignature.value=inspectSignature();step.value=0;message.value='请选择字头、IPA、备注列与开始行，再确认导入。';});}
async function refreshSample(){if(!staged.value||!port.value)return;const src=clone(staged.value),opts=clone(options.value);await operation(async signal=>{const value=await port.value!.inspectSource(src,opts,signal);if(signal.aborted)return;inspection.value=value;if(options.value.note_column!=null&&options.value.note_column>value.column_count)options.value.note_column=null;inspectedSignature.value=inspectSignature();});}
async function confirmImport(){if(!staged.value||!port.value||mappingError.value||needsRefresh.value)return;const src=clone(staged.value),opts=clone(options.value);opts.skip_first_row=(opts.start_row??2)>1;await operation(async signal=>{const value=await port.value!.analyze(src,opts,signal);if(signal.aborted)return;importEpoch.value++;source.value=src;preview.value=value;settings.value=clone(value.config);toneDraft.value=clone(value.config);importedOptions.value=opts;result.value=undefined;snapshot.value=undefined;staged.value=undefined;symbolsDirty.value=false;step.value=1;changed();message.value=`已读取 ${value.diagnostics.accepted_rows} 条记录，原始文件保持不变。`;});}
function stageCurrent(){if(!source.value)return;staged.value=clone(source.value);options.value={...defaultOptions(),...clone(importedOptions.value??{}),computation_revision:'m14/2'};void refreshSample();step.value=0;}
function toneValue(){if(!toneDraft.value||!settings.value)return;const cfg=clone(settings.value);cfg.tone_order=clone(toneDraft.value.tone_order);cfg.tone_map=Object.fromEntries(cfg.tone_order.map(t=>[t,toneDraft.value!.tone_map[t].trim()||t]));return cfg;}
function saveTones(){const cfg=toneValue();if(!cfg)return;settings.value=cfg;toneDraft.value=clone(cfg);changed();error.value='';message.value='调类设置已保存到当前工作草稿。';step.value=2;}
function saveSymbols(cfg:Settings){if(!settings.value)return;for(const k of ['initial_order','final_order','initial_map','final_map'] as const)Object.assign(settings.value,{[k]:clone(cfg[k])});symbolsDirty.value=false;changed();error.value='';message.value='声韵设置已保存到当前工作草稿。';step.value=3;}
function save(){if(busy.value){error.value='请等待当前任务结束。';return false;}if(staged.value){error.value='请先确认导入或取消此次文件选择。';return false;}if(!source.value||!preview.value||!settings.value){error.value='请先成功导入调查字表。';return false;}if(symbolEditor.value?.pending()){error.value='请先确认或取消当前归并。';return false;}const cfg=toneValue()??clone(settings.value);if(symbolsDirty.value&&symbolEditor.value){const symbols=symbolEditor.value.value();for(const k of ['initial_order','final_order','initial_map','final_map'] as const)Object.assign(cfg,{[k]:clone(symbols[k])});}if(!projects.write(key,{schema:'m14-draft/2',source:source.value,preview:preview.value,settings:cfg,options:importedOptions.value})){error.value='本机草稿保存失败，编辑仍保留。';return false;}settings.value=cfg;toneDraft.value=clone(cfg);symbolsDirty.value=false;dirty.value=false;message.value='草稿已保存到本机。';error.value='';return true;}
defineExpose({save});
function nudgeTone(tone:string,delta:number){if(!toneDraft.value)return;const list=toneDraft.value.tone_order,index=list.indexOf(tone),next=index+delta;if(next>=0&&next<list.length)toneDraft.value.tone_order=swap(list,tone,list[next]);}
function toneCount(tone:string){return preview.value?.analysis.rows.filter(r=>r.tone_value===tone).length??0;}
function toneExamples(tone:string){return preview.value?.analysis.rows.filter(r=>r.tone_value===tone).slice(0,6).map(r=>r.character).join('、')??'';}
async function generate(){if(!port.value||!source.value||!preview.value||!settings.value||!importedOptions.value||editing.value||legacy.value)return;const src=clone(source.value),cfg=clone(settings.value),opts=clone(importedOptions.value),data=clone(preview.value),sig=signature.value;await operation(async signal=>{const value=await port.value!.generate(src,opts,cfg,exportFontSnapshot(),signal);if(signal.aborted)return;result.value=value;snapshot.value={preview:data,settings:cfg,signature:sig};step.value=3;message.value='三份结果已完整生成，预览已绑定本次生成设置。';});}
async function saveResults(){if(result.value)await operation(async()=>{message.value=await port.value!.save(result.value!)?'三份结果已保存。':'已取消保存，结果仍可重新保存。';});}
async function download(id:string){if(result.value)await operation(async()=>{await port.value!.download(result.value!,id);message.value='文件已交给浏览器下载。';});}
const names=['导入','调类与调值','声韵排序归并','结果'],descriptions=['选择列与开始行','编辑调类与顺序','排序、归并与核对','预览、搜索与保存'];
</script>
<template>
<ModuleFrame unified fit class="phonology-page" label="音系归纳工作区" :aria-busy="busy">
 <template #toolbar><ModuleToolbar><button :disabled="busy||!port" @click="fileInput?.click()">导入调查字表</button><input ref="fileInput" type="file" accept=".xlsx,.xls,.csv,.txt,.tsv,.docx" hidden @change="selectFile"/><button v-if="busy" @click="controller?.abort()">取消任务</button><template #actions><button :disabled="busy||!source" @click="save">保存草稿</button><button @click="help=true">帮助</button><button @click="emit('references')"><AppIcon name="book"/>方法与引用</button></template></ModuleToolbar></template>
 <template #status><ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-else-if="busy" kind="loading" message="正在处理调查字表…"/><ModuleStatus v-else-if="message" kind="info" :message="message"/><ModuleStatus v-if="!port" kind="error" message="此宿主的 M14 正式接口尚不可用。"/></template>
 <ModuleWorkbench unified :state-key="stateKey" left-label="步骤与文件" right-label="生成与交付">
  <template #left>
   <nav class="steps" aria-label="音系归纳步骤"><button v-for="(name,i) in names" :key="name" :aria-label="(i+1)+' · '+name" :disabled="busy||(i>0&&!preview)" :aria-current="step===i?'step':undefined" @click="step=i"><span class="step-number">{{i+1}}</span><span><strong>{{name}}</strong><small>{{descriptions[i]}}</small></span></button></nav>
   <ModuleSection v-if="source" title="当前调查表" label="当前调查表"><p class="file-name">{{source.name}}</p><p class="hint">{{preview?.diagnostics.accepted_rows}} 条记录 · {{preview?.analysis.unique_ipa.length}} 种原始 IPA</p><p class="hint">解析 {{preview?.computation_revision??'m14/1'}} · 从第 {{importedOptions?.start_row??(importedOptions?.skip_first_row?2:1)}} 行开始</p><button :disabled="busy" @click="stageCurrent">重新设置导入</button></ModuleSection>
   <ModuleSection v-if="step===0" title="导入设置" label="导入设置">
    <label>单辅音策略<select v-model="options.consonant_only_as_zero_initial" :disabled="busy"><option :value="true">作为韵母，声母记为 Ø</option><option :value="false">作为声母，韵母记为空</option></select></label>
    <p class="hint">支持 XLSX、XLS、CSV、TSV、TXT、DOCX 表格。每条记录对应一个音节，备注可不使用。</p><p class="hint">文件上限 2 MB，最多 10,000 条有效记录。</p>
    <template v-if="!staged"><label>文本编码<select aria-label="文本编码" v-model="options.encoding" :disabled="busy"><option value="auto">自动识别 BOM / UTF-8</option><option value="utf-8-sig">UTF-8</option><option value="gb18030">GB18030 / GBK</option><option value="utf-16">UTF-16</option></select></label><label>分隔符<select aria-label="分隔符" v-model="options.delimiter" :disabled="busy"><option value="auto">自动识别</option><option value="tab">制表符 Tab</option><option value="comma">逗号 ,</option><option value="semicolon">分号 ;</option><option value="chinese_comma">中文逗号 ，</option><option value="space">空格</option></select></label><p class="hint">GBK 文件可先选 GB18030，再选择文件。</p></template>
   </ModuleSection>
   <p v-if="editing" class="edit-notice">有未保存的页内编辑，切换步骤会保留。生成前请保存设置。</p>
  </template>
  <div v-show="step===0">
   <ModuleSection class="import-card" label="调查表导入" title="调查表导入">
    <template #actions><button :disabled="busy||!port" @click="fileInput?.click()">{{staged?'更换文件':'选择文件'}}</button></template>
    <template v-if="inspection&&staged">
     <div class="file-banner"><strong>{{staged.name}}</strong><span>{{inspection.total_rows}} 行 · {{inspection.column_count}} 列</span></div>
     <div class="import-fields">
      <label>工作表 / 表格<select v-model.number="options.table_index" :disabled="busy" @change="refreshSample"><option v-for="(table,i) in inspection.tables" :key="i" :value="i">{{table}}</option></select></label>
      <label>数据从第几行开始<input v-model.number="options.start_row" type="number" min="1" max="10001" aria-label="数据开始行" :disabled="busy" @change="refreshSample"/></label>
      <label>字头列<select v-model.number="options.character_column" :disabled="busy" aria-label="字头列"><option v-for="c in columns" :key="c" :value="c">第 {{c}} 列</option></select></label>
      <label>IPA 列<select v-model.number="options.ipa_column" :disabled="busy" aria-label="IPA 列"><option v-for="c in columns" :key="c" :value="c">第 {{c}} 列</option></select></label>
      <label>备注列<select v-model="options.note_column" :disabled="busy" aria-label="备注列"><option :value="null">不使用</option><option v-for="c in columns" :key="c" :value="c">第 {{c}} 列</option></select></label>
      <label v-if="isText">文本编码<select aria-label="文本编码" v-model="options.encoding" :disabled="busy" @change="refreshSample"><option value="auto">自动识别 BOM / UTF-8</option><option value="utf-8-sig">UTF-8</option><option value="gb18030">GB18030 / GBK</option><option value="utf-16">UTF-16</option></select></label>
      <label v-if="isText">分隔符<select aria-label="分隔符" v-model="options.delimiter" :disabled="busy" @change="refreshSample"><option value="auto">自动识别</option><option value="tab">制表符 Tab</option><option value="comma">逗号 ,</option><option value="semicolon">分号 ;</option><option value="chinese_comma">中文逗号 ，</option><option value="space">空格</option></select></label>
     </div>
     <p class="hint">列号和行号从 1 开始。{{(options.start_row??2)>1?'前 '+((options.start_row??2)-1)+' 行不参与分析。':'从第一行开始分析。'}} 下方仅显示原始表格样例。</p>
     <p v-if="mappingError" role="alert" class="error">{{mappingError}}</p><p v-if="needsRefresh" class="edit-notice">样例设置已变化，请刷新后确认导入。</p>
     <div class="sample-scroll"><table aria-label="原始表格样例"><thead><tr><th>来源行</th><th v-for="c in columns" :key="c" :class="{chosen:[options.character_column,options.ipa_column,options.note_column].includes(c)}">第 {{c}} 列 <small>{{c===options.character_column?'字头':c===options.ipa_column?'IPA':c===options.note_column?'备注':''}}</small></th></tr></thead><tbody><tr v-for="r in inspection.sample" :key="r.row" :class="{skipped:r.row<(options.start_row??2)}"><th>{{r.row}}</th><td v-for="c in columns" :key="c" :class="{'ipa-text':c===options.ipa_column,chosen:[options.character_column,options.ipa_column,options.note_column].includes(c)}">{{r.cells[c-1]??''}}</td></tr></tbody></table></div>
     <div class="editor-footer"><div><button :disabled="busy" @click="staged=undefined">取消此次选择</button><button :disabled="busy" @click="refreshSample">刷新样例</button></div><button class="primary" :disabled="busy||!!mappingError||needsRefresh" @click="confirmImport">确认导入并继续</button></div>
    </template>
    <div v-else class="import-empty"><span class="table-symbol">▦</span><strong>选择调查字表</strong><p>查看原始表格，再选择字头、IPA、备注所在列和开始行。</p><span class="format-list">XLSX · XLS · CSV · TSV · TXT · DOCX</span><button class="primary" :disabled="busy||!port" @click="fileInput?.click()">选择文件</button></div>
   </ModuleSection>
  </div>
  <template v-if="preview&&settings">
   <ModuleSection v-show="step===1" label="调值与调类" title="调值与调类">
    <template #actions><small>{{toneChanged?'有未保存修改':'设置已保存'}} · {{settings.tone_order.length}} 个调值</small></template>
    <p class="hint">同名调类会归组，留空沿用原调值。可拖动交换顺序，或使用上下按钮调整。</p>
    <div class="tone-scroll"><table v-if="toneDraft" aria-label="调类编辑表"><thead><tr><th>顺序</th><th>原始调值</th><th>调类名称</th><th>记录数</th><th>例字</th></tr></thead><tbody><tr v-for="(tone,i) in toneDraft.tone_order" :key="tone" :draggable="!busy" @dragstart="toneDrag=tone" @dragover.prevent @drop.prevent="!busy&&toneDrag!==undefined&&(toneDraft.tone_order=swap(toneDraft.tone_order,toneDrag,tone))"><td><div class="order-buttons"><span>{{i+1}}</span><button :disabled="busy||i===0" :aria-label="'上移调值 '+tone" @click="nudgeTone(tone,-1)">↑</button><button :disabled="busy||i+1===toneDraft.tone_order.length" :aria-label="'下移调值 '+tone" @click="nudgeTone(tone,1)">↓</button></div></td><td>{{tone}}</td><td><input v-model="toneDraft.tone_map[tone]" :disabled="busy" :aria-label="'调类 '+tone" maxlength="100"/></td><td>{{toneCount(tone)}}</td><td>{{toneExamples(tone)}}</td></tr></tbody></table></div>
    <div class="editor-footer"><button :disabled="busy||!toneChanged" @click="toneDraft=clone(settings)">还原本次编辑</button><button class="primary" :disabled="busy" @click="saveTones">保存调类设置并继续</button></div>
   </ModuleSection>
   <SymbolEditor v-show="step===2" ref="symbolEditor" :key="(source?.asset_id??'')+':'+importEpoch" :settings="settings" :preview="preview" :busy="busy" @save="saveSymbols" @dirty="symbolsDirty=$event"/>
   <ResultPreview v-if="step===3" :preview="snapshot?.preview??preview" :settings="snapshot?.settings??settings" :generated="!!snapshot" :stale="stale"/>
  </template>
  <template v-if="step===3" #right>
   <ModuleSection label="导出文件" title="导出文件"><p class="hint">两份分组 DOCX 与一份二维 XLSX，保留全部记录、原字音与备注。</p><button class="primary generate-button" :disabled="busy||!port||!preview||editing||legacy" @click="generate">生成三份结果</button><p v-if="editing" class="edit-notice">请先保存调类或声韵设置。</p><p v-if="legacy" class="edit-notice">旧版草稿需重新导入后生成。</p><p v-if="stale" class="edit-notice">下方是上次生成文件，当前设置已修改。</p><button v-if="result" class="save-results" :disabled="busy" @click="saveResults">{{context.files.kind==='desktop'?'选择目录保存三份结果':'下载完整结果包'}}</button><ul v-if="result" class="result-files"><li v-for="file in result.files" :key="file.id"><strong>{{file.name}}</strong><small>{{(file.size_bytes/1024).toFixed(1)}} KB</small><button :disabled="busy" @click="download(file.id)">下载</button></li></ul></ModuleSection>
  </template>
 </ModuleWorkbench>
 <ModalDialog v-if="help" title="音系归纳操作说明" @close="help=false"><p>选择文件后查看原始表格样例，选择字头、IPA、备注列与数据开始行，再确认导入。支持 XLSX/XLS、CSV/TSV/TXT 和 DOCX 表格，可选工作表与文本编码。</p><p>调类和声韵均在当前页面编辑、保存或还原。Ctrl/⌘、Shift 多选和拖动排序保留。归并只影响分类，原字音与备注保留。顶部保存草稿将当前编辑保存到本机。</p><p>第 4 步提供声母到韵母、韵母到声母、二维声韵表三种预览，搜索覆盖全部记录。预览按窗口排版，DOCX 纸张分页以实际文件为准。生成后预览绑定本次设置，修改设置需重新生成。</p><p>新解析 m14/2 修复连音符与 Unicode 等价字符，支持末尾普通、全角和上标数字调值。单独辅音仍按所选策略处理。多音节或未知符号需要人工核对。文件上限 2 MB、有效记录 10,000 条、矩阵最多 20,000 格。损坏、公式或超预算输入明确拒绝。音标固定 Doulos SIL，外部阅读 DOCX/XLSX 需具备字体。</p></ModalDialog>
</ModuleFrame>
</template>
<style scoped>
.steps{display:flex;flex-direction:column;gap:6px}.steps button{display:flex;justify-content:flex-start;text-align:left;gap:12px;min-height:61px;padding:9px 12px;background:transparent;white-space:normal}.step-number{display:grid;place-items:center;border:1px solid var(--border);border-radius:50%;width:29px;height:29px;flex:none;color:var(--muted)}.steps strong,.steps small{display:block}.steps strong{font-size:1em;font-weight:600}.steps small{margin-top:4px;font-size:.85em}.steps button[aria-current]{background:var(--selected);border-color:var(--accent);color:var(--accent)}.steps button[aria-current] .step-number{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}label{display:flex;flex-direction:column;gap:5px;font-size:.93em}.file-name{overflow-wrap:anywhere;font-weight:600}.file-banner{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:10px 12px;background:var(--app);border:1px solid var(--border);border-radius:6px;margin:10px 0}.file-banner span{color:var(--muted);font-size:.9em}.import-fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:14px 0}.import-fields input,.import-fields select{width:100%;min-width:0}.sample-scroll,.tone-scroll{overflow:auto;margin-top:12px;max-height:58vh;overscroll-behavior:contain}table{width:100%;border-collapse:separate;border-spacing:0}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--border)}thead th{position:sticky;top:0;background:var(--sidebar);z-index:1;white-space:nowrap}th small{display:block;color:var(--accent);font-weight:400}.sample-scroll td{min-width:110px;max-width:320px;overflow-wrap:anywhere;white-space:pre-wrap}.chosen{background:var(--selected)}tr.skipped{opacity:.48}.tone-scroll td>input{width:100%;min-width:125px;max-width:300px}.tone-scroll tr:hover{background:var(--app)}.order-buttons{display:flex;gap:4px;align-items:center;white-space:nowrap}.order-buttons>span{min-width:20px;color:var(--muted)}.order-buttons button{padding:2px 7px;min-width:25px}.editor-footer{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-top:14px;padding-top:12px;border-top:1px solid var(--border);flex-wrap:wrap}.editor-footer>div{display:flex;gap:6px}.import-card{min-height:380px}.import-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;min-height:360px;text-align:center;color:var(--muted);padding:28px}.import-empty strong{font-size:1.1em;color:var(--text)}.table-symbol{font-size:46px;line-height:1;color:var(--accent)}.format-list{font-size:.9em;word-spacing:4px}.error{color:var(--danger)}.edit-notice{font-size:.9em;color:var(--warning);line-height:1.7}.result-files{list-style:none;padding:0;margin:12px 0 0}.result-files li{display:grid;grid-template-columns:1fr auto;gap:6px;padding:12px 0;border-top:1px solid var(--border)}.result-files strong{grid-column:1/-1;font-weight:500;overflow-wrap:anywhere}.result-files small{align-self:center}.generate-button,.save-results{width:100%;margin-top:10px}
@media(max-width:950px){.import-fields{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.import-fields{grid-template-columns:1fr}.editor-footer>.primary{width:100%}.import-empty{padding:14px}.file-banner{align-items:flex-start;flex-direction:column}}
</style>
