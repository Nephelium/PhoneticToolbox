<script setup lang="ts">
import {computed,ref} from 'vue';
import ModalDialog from '../../components/ModalDialog.vue';
import {columnLabel,suggestColumns,tableName,tableRows,type ImportColumns,type TableKind,type CustomTable} from './custom-tables.ts';
import {readTableFile} from './table-file.ts';
import ToneMapping from './ToneMapping.vue';
import {defaultToneSettings,tonePreview,transcribeTone,type ToneSettings} from './tones.ts';
const props=defineProps<{saving:boolean;saveError:string;replaceTarget?:CustomTable}>();
const emit=defineEmits<{close:[];save:[table:Pick<CustomTable,'name'|'kind'|'rows'|'tones'>]}>();
const kind=ref<TableKind>(props.replaceTarget?.kind??'hanzi'),name=ref(props.replaceTarget?.name??''),fileName=ref(''),sheet=ref(0),loading=ref(false),error=ref('');
const source=ref<Awaited<ReturnType<typeof readTableFile>>>(),rows=ref<string[][]>([]);
const columns=ref<ImportColumns>({keyColumn:0,ipaColumn:1,startRow:1});
const tones=ref<ToneSettings>(JSON.parse(JSON.stringify(props.replaceTarget?.tones??defaultToneSettings())));
const columnCount=computed(()=>Math.min(256,rows.value.reduce((max,row)=>Math.max(max,row.length),2)));
const rawPreview=computed(()=>{const start=Math.max(0,(Number(columns.value.startRow)||1)-2);return rows.value.slice(start,start+8).map((cells,i)=>({cells,line:start+i+1}));});
const selectedRows=computed(()=>{try{return {rows:tableRows(rows.value,kind.value,columns.value),error:''};}catch(cause){return {rows:[] as [string,string][],error:(cause as Error).message};}});
// Mapping controls must remain available even when a later record has an error.
const toneRows=computed(()=>rows.value.slice(Math.max(0,(Number(columns.value.startRow)||1)-1)).map(row=>[row[columns.value.keyColumn]??'',row[columns.value.ipaColumn]??'']));
const parsed=computed(()=>{if(selectedRows.value.error)return selectedRows.value;try{tonePreview(selectedRows.value.rows,tones.value);return selectedRows.value;}catch(cause){return {rows:[] as [string,string][],error:(cause as Error).message};}});
const preview=computed(()=>selectedRows.value.rows.slice(0,8).map(row=>{try{return [row[0],row[1],transcribeTone(row[1],tones.value)];}catch{return [row[0],row[1],'待设置声调'];}}));
function useRows(values:string[][]){rows.value=values;columns.value=suggestColumns(values);}
let request=0;
async function chooseFile(event:Event){
  const file=(event.target as HTMLInputElement).files?.[0];if(!file)return;
  const current=++request;loading.value=true;error.value='';rows.value=[];source.value=undefined;fileName.value=file.name;
  if(!name.value)name.value=file.name.replace(/\.[^.]+$/,'');
  try{const book=await readTableFile(file);source.value=current===request?book:source.value;const values=await book.rows(0);if(current!==request)return;sheet.value=0;useRows(values);}
  catch(cause){if(current===request)error.value=(cause as Error).message;}
  finally{if(current===request)loading.value=false;(event.target as HTMLInputElement).value='';}
}
async function chooseSheet(){
  if(!source.value)return;const current=++request;loading.value=true;error.value='';rows.value=[];
  try{const values=await source.value.rows(sheet.value);if(current===request)useRows(values);}catch(cause){if(current===request)error.value=(cause as Error).message;}finally{if(current===request)loading.value=false;}
}
function save(){try{if(parsed.value.error||!parsed.value.rows.length)return;emit('save',{name:tableName(name.value),kind:kind.value,rows:parsed.value.rows,tones:JSON.parse(JSON.stringify(tones.value))});}catch(cause){error.value=(cause as Error).message;}}
</script>
<template>
  <ModalDialog :title="replaceTarget?'替换自定义字表':'导入自定义字表'" :close-disabled="saving||loading" @close="emit('close')">
    <div class="m13-import">
      <p v-if="replaceTarget">选择新文件并预览，保存后更新当前字表。可保留原名称或重新命名，多音字需重新核对。</p>
      <label>对照类型<select v-model="kind" aria-label="字表对照类型" :disabled="saving||loading"><option value="hanzi">汉字 → 国际音标</option><option value="syllable">拼音音节 → 国际音标</option></select></label>
      <p>读取文件后选择{{kind==='hanzi'?'汉字':'拼音音节'}}列、音标列及数据起始行，其他列不会导入。</p>
      <p v-if="kind==='syllable'">支持 ma、ma1、mā，ü 可写作 v 或 u:。带调条目优先，不带调条目用于该音节的各调。</p>
      <label>选择字表文件<input type="file" aria-label="选择字表文件" accept=".xlsx,.csv,.tsv,.txt" :disabled="saving||loading" @change="chooseFile"/></label>
      <p class="hint">XLSX、UTF-8 CSV、TSV 或制表符 TXT，最多 8 MB、50,000 行。</p>
      <label v-if="source&&source.names.length>1">工作表<select v-model.number="sheet" aria-label="字表工作表" :disabled="saving||loading" @change="chooseSheet"><option v-for="(title,i) in source.names" :key="i" :value="i">{{title}}</option></select></label>
      <template v-if="rows.length">
        <div class="column-controls">
          <label>{{kind==='hanzi'?'汉字':'拼音音节'}}列<select v-model.number="columns.keyColumn" aria-label="汉字或音节列" :disabled="saving||loading"><option v-for="i in columnCount" :key="i" :value="i-1">{{columnLabel(i-1)}} 列</option></select></label>
          <label>音标列<select v-model.number="columns.ipaColumn" aria-label="音标列" :disabled="saving||loading"><option v-for="i in columnCount" :key="i" :value="i-1">{{columnLabel(i-1)}} 列</option></select></label>
          <label>数据起始行<input v-model.number="columns.startRow" type="number" min="1" :max="rows.length" step="1" aria-label="数据起始行" :disabled="saving||loading"/></label>
        </div>
        <p class="hint">{{fileName}} · 共 {{rows.length.toLocaleString()}} 行。下方显示起始行附近的 {{rawPreview.length}} 行，行号与原表一致。</p>
        <div class="raw-scroll"><table aria-label="原始字表预览"><thead><tr><th>原表行号</th><th v-for="i in columnCount" :key="i" :class="{chosen:i-1===columns.keyColumn||i-1===columns.ipaColumn}">{{columnLabel(i-1)}}{{i-1===columns.keyColumn?' · '+(kind==='hanzi'?'汉字':'音节'):i-1===columns.ipaColumn?' · 音标':''}}</th></tr></thead><tbody><tr v-for="row in rawPreview" :key="row.line" :class="{skipped:row.line<columns.startRow}"><th>{{row.line}}</th><td v-for="i in columnCount" :key="i" :class="{chosen:i-1===columns.keyColumn||i-1===columns.ipaColumn}" :title="row.cells[i-1]">{{(row.cells[i-1]??'').slice(0,160)}}</td></tr></tbody></table></div>
        <ToneMapping v-model="tones" :rows="toneRows" :disabled="saving||loading"/>
      </template>
      <label>字表名称<input v-model="name" aria-label="字表名称" maxlength="60" :disabled="saving" placeholder="例如：方言调查字表"/></label>
      <p v-if="loading" role="status">正在读取字表…</p>
      <p v-else-if="error||saveError||(source&&parsed.error)" role="alert">{{error||saveError||parsed.error}}</p>
      <template v-if="preview.length">
        <p>{{selectedRows.rows.length.toLocaleString()}} 条记录，预览前 {{preview.length}} 条转换结果</p>
        <table aria-label="导入结果预览"><thead><tr><th>{{kind==='hanzi'?'汉字':'拼音音节'}}</th><th>原始音标</th><th>转换结果</th></tr></thead><tbody><tr v-for="(row,i) in preview" :key="i"><td>{{row[0]}}</td><td class="ipa-text">{{row[1]}}</td><td class="ipa-text">{{row[2]}}</td></tr></tbody></table>
      </template>
      <p class="hint">保存后可在转换标准中选择，下次打开自动载入。字表保存在当前设备，建议保留原始文件。</p>
    </div>
    <template #footer><button :disabled="saving||loading" @click="emit('close')">取消</button><button class="primary" :disabled="saving||loading||!parsed.rows.length||!name.trim()" @click="save">{{saving?'正在保存…':replaceTarget?'替换字表并使用':'保存字表并使用'}}</button></template>
  </ModalDialog>
</template>
<style scoped>
.m13-import{display:grid;gap:10px;min-width:0}.m13-import label{display:grid;gap:4px}.m13-import input,.m13-import select{width:100%;min-width:0}.m13-import p{margin:0;line-height:1.6;overflow-wrap:anywhere}.hint{color:var(--muted);font-size:var(--support-size)}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{text-align:left;padding:5px 8px;border:1px solid var(--border);overflow-wrap:anywhere}.ipa-text{font-size:1.2rem}[role=alert]{color:var(--danger)}.column-controls{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.raw-scroll{overflow:auto;max-height:230px}.raw-scroll table{min-width:100%;width:max-content;table-layout:fixed}.raw-scroll th,.raw-scroll td{min-width:90px;max-width:200px}.chosen{background:color-mix(in srgb,var(--accent) 10%,var(--panel));font-weight:600}.skipped{color:var(--muted)}@media(max-width:600px){.column-controls{grid-template-columns:1fr}}
</style>
