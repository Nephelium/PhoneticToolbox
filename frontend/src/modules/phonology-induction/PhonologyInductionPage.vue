<script setup lang="ts">
import {ref,computed,watch,onUnmounted} from 'vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import ModuleSection from '../../components/ModuleSection.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import ModalDialog from '../../components/ModalDialog.vue';
import type {ResearchContext} from '../../platform/research.ts';
import {projects} from '../../platform/browser.ts';
import {exportFontSnapshot} from '../../state/fonts.ts';
import type {Settings,Preview,Source,Result,Options} from './port.ts';
import {clone,select,move,swap,merge,resolve,label} from './state.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active?:boolean}>();
const emit=defineEmits<{references:[];dirty:[boolean]}>();
const step=ref(0),error=ref(''),message=ref(''),busy=ref(false),help=ref(false),dirty=ref(false);
const fileInput=ref<HTMLInputElement>(),source=ref<Source>(),preview=ref<Preview>(),settings=ref<Settings>(),result=ref<Result>();
const options=ref<Options>({skip_first_row:true,consonant_only_as_zero_initial:true});
const importedOptions=ref<Options>();const toneDraft=ref<Settings>(),symbolDraft=ref<Settings>();
const picked=ref<{initial:string[];final:string[]}>({initial:[],final:[]});const anchors:{initial?:string;final?:string}={};
const pending=ref<{kind:'initial'|'final';source:string}>();const toneDrag=ref<string>();let controller:AbortController|undefined;
const port=computed(()=>props.context.files.m14);
const key='m14:'+props.stateKey;
const saved=projects.read<any>(key,null);
if(saved?.schema==='m14-draft/1'&&saved.source&&saved.preview&&saved.settings){source.value=saved.source;preview.value=saved.preview;settings.value=saved.settings;options.value=saved.options;importedOptions.value=clone(saved.options);step.value=1;message.value='已恢复本机草稿；生成时会重新校验输入是否有效。';}
function changed(){dirty.value=true;}
watch(()=>dirty.value||busy.value||!!toneDraft.value||!!symbolDraft.value,value=>emit('dirty',value),{flush:'sync'});
watch(options,changed,{deep:true});
function save(){
 if(busy.value||toneDraft.value||symbolDraft.value){error.value='请先完成或取消当前操作，再保存草稿。';return false;}
 if(!source.value||!preview.value||!settings.value){error.value='请先成功导入调查字表。';return false;}
 if(!projects.write(key,{schema:'m14-draft/1',source:source.value,preview:preview.value,settings:settings.value,options:importedOptions.value})){error.value='本机草稿保存失败，编辑仍保留。';return false;}
 dirty.value=false;emit('dirty',false);message.value='草稿已保存到本机。';return true;
}
defineExpose({save});
onUnmounted(()=>controller?.abort());
async function operation(run:(signal:AbortSignal)=>Promise<void>){
 if(busy.value)return;error.value='';message.value='';busy.value=true;controller=new AbortController();
 try{await run(controller.signal);}catch(e){error.value=e instanceof Error?e.message:'操作失败，请重试。';}finally{busy.value=false;controller=undefined;}
}
async function importFile(event:Event){
 const input=event.target as HTMLInputElement,file=input.files?.[0];input.value='';if(!file)return;
 if(!port.value){error.value='当前宿主未提供音系归纳文件与任务接口。';return;}
 const selected=clone(options.value);
 await operation(async signal=>{const value=await port.value!.import(file,selected,signal);if(signal.aborted)return;source.value=value.source;preview.value=value.preview;settings.value=clone(value.preview.config);importedOptions.value=selected;result.value=undefined;step.value=1;changed();message.value=`已读取 ${value.preview.diagnostics.accepted_rows} 条记录，原始文件保持不变。`;});
}
function editTones(){if(settings.value)toneDraft.value=clone(settings.value);}
function confirmTones(){if(!toneDraft.value)return;for(const t of toneDraft.value.tone_order)toneDraft.value.tone_map[t]=toneDraft.value.tone_map[t].trim()||t;settings.value=toneDraft.value;toneDraft.value=undefined;changed();}
function editSymbols(){if(settings.value){symbolDraft.value=clone(settings.value);picked.value={initial:[],final:[]};pending.value=undefined;}}
function choose(kind:'initial'|'final',value:string,event:MouseEvent){
 if(!symbolDraft.value)return;
 if(pending.value?.kind===kind){try{merge(symbolDraft.value,kind,pending.value.source,value);pending.value=undefined;picked.value[kind]=[];error.value='';}catch(e){error.value=(e as Error).message;}return;}
 const selection=select(symbolDraft.value[`${kind}_order`],picked.value[kind],anchors[kind],value,event.ctrlKey||event.metaKey,event.shiftKey);picked.value[kind]=selection.selected;anchors[kind]=selection.anchor;
}
function prepareMerge(kind:'initial'|'final'){if(picked.value[kind].length!==1){error.value='归并前请单选一个源音标。';return;}pending.value={kind,source:picked.value[kind][0]};error.value='';}
function drag(kind:'initial'|'final',value:string){if(!picked.value[kind].includes(value))picked.value[kind]=[value];}
function drop(kind:'initial'|'final',value:string){if(symbolDraft.value)symbolDraft.value[`${kind}_order`]=move(symbolDraft.value[`${kind}_order`],picked.value[kind],value);}
function confirmSymbols(){if(pending.value){error.value='请先选择归并目标或取消本次归并。';return;}settings.value=symbolDraft.value;symbolDraft.value=undefined;changed();error.value='';}
async function generate(){if(!port.value||!source.value||!settings.value||!importedOptions.value)return;const src=clone(source.value),cfg=clone(settings.value),opts=clone(importedOptions.value);await operation(async signal=>{const value=await port.value!.generate(src,opts,cfg,exportFontSnapshot(),signal);if(signal.aborted)return;result.value=value;step.value=3;message.value='三份结果已完整生成。';});}
async function saveResults(){if(result.value)await operation(async()=>{if(await port.value!.save(result.value!))message.value='三份结果已保存。';else message.value='已取消保存，结果仍可重新保存。';});}
async function download(id:string){if(result.value)await operation(async()=>{await port.value!.download(result.value!,id);message.value='文件已交给浏览器下载。';});}
const mappings=computed(()=>settings.value?(['initial','final'] as const).flatMap(k=>Object.entries(settings.value![`${k}_map`]).map(([a,b])=>`${k==='initial'?'声母':'韵母'} ${label(a)} → ${label(b)} → ${label(resolve(a,settings.value![`${k}_map`]))}`)):[]);
</script>
<template>
<ModuleFrame fit class="phonology-page" label="音系归纳工作区" :aria-busy="busy">
 <template #toolbar><ModuleToolbar><button :disabled="busy||!port" @click="fileInput?.click()">导入调查字表</button><input ref="fileInput" type="file" accept=".xlsx,.xls,.csv,.txt,.tsv" hidden @change="importFile"/><button :disabled="busy||!source" @click="save">保存草稿</button><button v-if="busy" @click="controller?.abort()">取消任务</button><template #actions><button @click="help=true">帮助</button><button @click="emit('references')">方法与来源</button></template></ModuleToolbar></template>
 <template #status><ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-else-if="busy" kind="loading" message="正在处理调查字表…"/><ModuleStatus v-if="!port" kind="error" message="此宿主的 M14 正式接口尚不可用。"/></template>
 <ModuleWorkbench :state-key="stateKey" left-label="导入与步骤" right-label="任务与结果" :left-width="250" :right-width="300">
 <template #left> <nav class="steps" aria-label="音系归纳步骤"><button v-for="(name,i) in ['导入','调类与调值','声韵排序归并','结果']" :key="name" :disabled="busy||(i>0&&!preview)" :aria-current="step===i?'step':undefined" @click="step=i">{{i+1}} · {{name}}</button></nav>
 <ModuleSection label="导入设置" title="导入设置">
  <label><input v-model="options.skip_first_row" type="checkbox" :disabled="busy"/>跳过首行（表头）</label>
  <label>单辅音策略 <select v-model="options.consonant_only_as_zero_initial" :disabled="busy"><option :value="true">按零声母字处理</option><option :value="false">按空韵字处理</option></select></label>
  <p>第一列字头，第二列 IPA，第三列可选备注。支持 UTF-8 文本及 XLSX/XLS。当前选项在下一次导入时生效。</p>
  <p v-if="source">已导入 {{source.name}} · {{importedOptions?.skip_first_row?'跳首行':'不跳首行'}} · {{importedOptions?.consonant_only_as_zero_initial?'单辅音作韵母':'单辅音作声母'}}</p>
 </ModuleSection>
</template>
 <template v-if="preview&&settings">
 <ModuleSection v-show="step===1" label="调值与调类" title="调值与调类"><button :disabled="busy" @click="editTones">编辑调值顺序与调类</button><ol><li v-for="tone in settings.tone_order" :key="tone">{{tone}} → {{settings.tone_map[tone]}}</li></ol></ModuleSection>
 <ModuleSection v-show="step===2" label="声韵排序与归并" title="声韵排序与归并"><button :disabled="busy" @click="editSymbols">编辑声韵顺序与归并</button><div class="symbol-columns"><div><p>声母</p><p class="ipa-text">{{settings.initial_order.map(label).join('　')}}</p></div><div><p>韵母</p><p class="ipa-text">{{settings.final_order.map(label).join('　')}}</p></div></div><ul aria-label="已确认归并映射"><li v-for="item in mappings" :key="item" class="ipa-text">{{item}}</li></ul></ModuleSection>

 <ModuleSection class="phonology-review" label="导入与归并审阅" title="导入与归并审阅"><p>{{source?.name}} · {{preview.diagnostics.accepted_rows}} 条记录 · {{preview.analysis.unique_ipa.length}} 个不同音标 · 重复记录 {{preview.diagnostics.duplicate_rows}} 条（保留）</p><details v-if="preview.diagnostics.skipped.length"><summary>跳过 {{preview.diagnostics.skipped.length}} 行</summary><ul><li v-for="r in preview.diagnostics.skipped" :key="r.row">第 {{r.row}} 行：{{r.reason}}</li></ul></details><p v-if="preview.single_consonants.length">单辅音示例：<span class="ipa-text">{{preview.single_consonants.slice(0,8).map(r=>`${r.character}(${r.ipa})`).join('、')}}</span></p><div class="rows"><table><thead><tr><th>字</th><th>原始 IPA</th><th>备注</th><th>声母 → 归并</th><th>韵母 → 归并</th><th>调值 → 调类</th></tr></thead><tbody><tr v-for="(row,i) in preview.analysis.rows.slice(0,200)" :key="i"><td>{{row.character}}</td><td class="ipa-text">{{row.ipa}}</td><td>{{row.note}}</td><td class="ipa-text">{{label(row.initial)}} → {{label(resolve(row.initial,settings.initial_map))}}</td><td class="ipa-text">{{label(row.final)}} → {{label(resolve(row.final,settings.final_map))}}</td><td>{{row.tone_value}} → {{settings.tone_map[row.tone_value]}}</td></tr></tbody></table></div><p v-if="preview.analysis.rows.length>200">预览前 200 条，导出包含全部 {{preview.analysis.rows.length}} 条。</p></ModuleSection>
 </template><ModuleStatus v-else kind="empty" message="导入调查字表后设置调类、声韵顺序与归并。"/>
 <template #right><ModuleStatus v-if="message" kind="info" :message="message"/> <ModuleSection v-if="preview&&settings" label="生成与交付" title="生成与交付"><button :disabled="busy||!port" @click="generate">生成三份结果</button><button v-if="result" :disabled="busy" @click="saveResults">{{context.files.kind==='desktop'?'选择目录保存三份结果':'下载完整结果包'}}</button><ul v-if="result"><li v-for="file in result.files" :key="file.id">{{file.name}} · {{file.size_bytes}} 字节 <button :disabled="busy" @click="download(file.id)">下载</button></li></ul><p v-else>正序与逆序 DOCX 各一份，同音字矩阵 XLSX 一份。</p></ModuleSection><ModuleStatus v-if="!preview" kind="empty" message="导入并确认归并设置后，可生成和保存结果。"/></template>
 </ModuleWorkbench>
 <ModalDialog v-if="toneDraft" title="调值归并设置" @close="toneDraft=undefined"><p>拖动一行到目标行可互换顺序。调类同名即合并，留空使用原调值。</p><table><thead><tr><th>调值</th><th>调类</th></tr></thead><tbody><tr v-for="tone in toneDraft.tone_order" :key="tone" draggable="true" @dragstart="toneDrag=tone" @dragover.prevent @drop.prevent="toneDrag!==undefined&&(toneDraft.tone_order=swap(toneDraft.tone_order,toneDrag,tone))"><td>{{tone}}</td><td><input v-model="toneDraft.tone_map[tone]" :aria-label="'调类 '+tone" maxlength="100"/></td></tr></tbody></table><template #footer><button @click="toneDraft=undefined">取消</button><button @click="confirmTones">确认调类</button></template></ModalDialog>
 <ModalDialog v-if="symbolDraft" title="声韵排序与归并" wide @close="symbolDraft=undefined;pending=undefined;error=''">
  <p>Ctrl 多选离散项，Shift 选择连续项，拖动可整组排序。归并时单选源项，点击归并按钮，再选目标。</p><ModuleStatus v-if="error" kind="error" :message="error"/><p v-if="pending">已选 <span class="ipa-text">{{label(pending.source)}}</span>，请选择{{pending.kind==='initial'?'声母':'韵母'}}目标。<button @click="pending=undefined">取消本次归并</button></p>
  <div class="symbol-columns"><div v-for="kind in (['initial','final'] as const)" :key="kind"><p>{{kind==='initial'?'声母':'韵母'}}</p><ul class="symbol-list" role="listbox" aria-multiselectable="true" :aria-label="kind==='initial'?'声母列表':'韵母列表'"><li v-for="value in symbolDraft[`${kind}_order`]" :key="value" role="option" :aria-selected="picked[kind].includes(value)" tabindex="0" draggable="true" class="ipa-text" @click="choose(kind,value,$event)" @keydown.enter.prevent="choose(kind,value,$event as unknown as MouseEvent)" @keydown.space.prevent="choose(kind,value,$event as unknown as MouseEvent)" @dragstart="drag(kind,value)" @dragover.prevent @drop.prevent="drop(kind,value)">{{label(value)}}</li></ul><button @click="prepareMerge(kind)">{{kind==='initial'?'归并声母':'归并韵母'}}</button><ul><li v-for="(target,from) in symbolDraft[`${kind}_map`]" :key="from" class="ipa-text">{{label(String(from))}} → {{label(target)}}</li></ul></div></div>
  <template #footer><button @click="symbolDraft=undefined;pending=undefined;error=''">取消</button><button @click="confirmSymbols">确认声韵设置</button></template>
 </ModalDialog>
 <ModalDialog v-if="help" title="音系归纳操作说明" @close="help=false"><p>按四步导入、调类设置、声韵排序归并和生成操作。两次设置均需确认后生效，取消保留之前设置。归并只影响结果分类，原字音与备注保留。</p><p>常见表头会自动过滤，缺字或缺音标行会跳过并列明。Excel/CSV 沿用 V2 默认缺失值识别，文本按字面读取。重复条目保留。文件上限 2 MB，最多 10,000 行、声韵矩阵最多 20,000 格。损坏文件、公式表格、非 UTF-8 文本会拒绝。</p><p>调值为 IPA 末尾数字，缺少数字记为 0。解析沿用 V2 规则，不自动判断音位或纠正未知符号。生成前请审阅映射。草稿保存于当前本机账号/项目作用域。文件到期后需要重新导入。</p><p>DOCX/XLSX 使用生成时字体设置，音标固定 Doulos SIL。跨设备阅读需安装对应字体。重名保存拒绝覆盖，取消或失败可以重试。</p></ModalDialog>
</ModuleFrame>
</template>
<style scoped>
.steps,.symbol-columns{display:flex;gap:var(--module-gap,12px);flex-wrap:wrap}.steps button[aria-current]{background:var(--selected);color:var(--text)}.symbol-columns>div{flex:1;min-width:180px}.symbol-list{list-style:none;padding:0;max-height:280px;overflow:auto;border:1px solid var(--border)}.symbol-list li{padding:8px;cursor:grab;border-bottom:1px solid var(--border)}.symbol-list li[aria-selected=true]{background:var(--selected)}.symbol-list li:focus-visible{outline:2px solid var(--accent)}.rows{overflow:auto;max-height:none}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--border)}label{display:inline-flex;gap:8px;align-items:center;margin:6px 12px 6px 0}p{line-height:1.6}input,select,button{font:inherit}

.phonology-page>:deep(.module-toolbar),.phonology-page>.steps{flex-shrink:0}
.phonology-review{flex:1;min-height:180px;overflow:auto;overscroll-behavior:contain}
</style>
