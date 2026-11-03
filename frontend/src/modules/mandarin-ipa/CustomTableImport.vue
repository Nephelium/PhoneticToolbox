<script setup lang="ts">
import {tr,message as localizedMessage,messageRef} from '../../i18n/index.ts';
import {computed,ref} from 'vue';
import {uiErrorMessage as displayErrorValue} from '../module-localization.ts';
import ModalDialog from '../../components/ModalDialog.vue';
import {columnLabel,suggestColumns,tableName,tableRows,type ImportColumns,type TableKind,type CustomTable} from './custom-tables.ts';
import {readTableFile} from './table-file.ts';
import ToneMapping from './ToneMapping.vue';
import {defaultToneSettings,tonePreview,transcribeTone,type ToneSettings} from './tones.ts';
const props=defineProps<{saving:boolean;saveError:string;replaceTarget?:CustomTable}>();
const emit=defineEmits<{close:[];save:[table:Pick<CustomTable,'name'|'kind'|'rows'|'tones'>]}>();
const kind=ref<TableKind>(props.replaceTarget?.kind??'hanzi'),name=ref(props.replaceTarget?.name??''),fileName=ref(''),sheet=ref(0),loading=ref(false),error=messageRef();
const source=ref<Awaited<ReturnType<typeof readTableFile>>>(),rows=ref<string[][]>([]);
const columns=ref<ImportColumns>({keyColumn:0,ipaColumn:1,startRow:1});
const tones=ref<ToneSettings>(JSON.parse(JSON.stringify(props.replaceTarget?.tones??defaultToneSettings())));
const columnCount=computed(()=>Math.min(256,rows.value.reduce((max,row)=>Math.max(max,row.length),2)));
const rawPreview=computed(()=>{const start=Math.max(0,(Number(columns.value.startRow)||1)-2);return rows.value.slice(start,start+8).map((cells,i)=>({cells,line:start+i+1}));});
const selectedRows=computed(()=>{try{return {rows:tableRows(rows.value,kind.value,columns.value),error:''};}catch(cause){return {rows:[] as [string,string][],error:(cause as Error).message};}});
// Mapping controls must remain available even when a later record has an error.
const toneRows=computed(()=>rows.value.slice(Math.max(0,(Number(columns.value.startRow)||1)-1)).map(row=>[row[columns.value.keyColumn]??'',row[columns.value.ipaColumn]??'']));
const parsed=computed(()=>{if(selectedRows.value.error)return selectedRows.value;try{tonePreview(selectedRows.value.rows,tones.value);return selectedRows.value;}catch(cause){return {rows:[] as [string,string][],error:(cause as Error).message};}});
const preview=computed(()=>selectedRows.value.rows.slice(0,8).map(row=>{try{return [row[0],row[1],transcribeTone(row[1],tones.value)];}catch{return [row[0],row[1],tr('modules.mandarin-ipa.text.2e900b97a23f')];}}));
function useRows(values:string[][]){rows.value=values;columns.value=suggestColumns(values);}
let request=0;
async function chooseFile(event:Event){
  const file=(event.target as HTMLInputElement).files?.[0];if(!file)return;
  const current=++request;loading.value=true;error.value='';rows.value=[];source.value=undefined;fileName.value=file.name;
  if(!name.value)name.value=file.name.replace(/\.[^.]+$/,'');
  try{const book=await readTableFile(file);source.value=current===request?book:source.value;const values=await book.rows(0);if(current!==request)return;sheet.value=0;useRows(values);}
  catch(cause){if(current===request)error.value=displayErrorValue(cause);}
  finally{if(current===request)loading.value=false;(event.target as HTMLInputElement).value='';}
}
async function chooseSheet(){
  if(!source.value)return;const current=++request;loading.value=true;error.value='';rows.value=[];
  try{const values=await source.value.rows(sheet.value);if(current===request)useRows(values);}catch(cause){if(current===request)error.value=displayErrorValue(cause);}finally{if(current===request)loading.value=false;}
}
function save(){try{if(parsed.value.error||!parsed.value.rows.length)return;emit('save',{name:tableName(name.value),kind:kind.value,rows:parsed.value.rows,tones:JSON.parse(JSON.stringify(tones.value))});}catch(cause){error.value=displayErrorValue(cause);}}
</script>
<template>
  <ModalDialog :title="replaceTarget?tr('modules.mandarin-ipa.text.5f0aa25eb727'):tr('modules.mandarin-ipa.text.4208e69684fe')" :close-disabled="saving||loading" @close="emit('close')">
    <div class="m13-import">
      <p v-if="replaceTarget">{{tr('modules.mandarin-ipa.text.3b3e3235fa1f')}}</p>
      <label>{{tr('modules.mandarin-ipa.text.cdc8fae86e2f')}}<select v-model="kind" :aria-label="tr('modules.mandarin-ipa.text.56cfd74501db')" :disabled="saving||loading"><option value="hanzi">{{tr('modules.mandarin-ipa.text.1a18dd28c536')}}</option><option value="syllable">{{tr('modules.mandarin-ipa.text.cd81026afa22')}}</option></select></label>
      <p>{{tr('modules.mandarin-ipa.text.987039f087ad')}}{{kind==='hanzi'?tr('modules.mandarin-ipa.text.00e5d5601ce3'):tr('modules.mandarin-ipa.text.68da7a474f1f')}}{{tr('modules.mandarin-ipa.text.4faff1c71a3e')}}</p>
      <p v-if="kind==='syllable'">{{tr('modules.mandarin-ipa.text.c36c913e865f')}}</p>
      <label>{{tr('modules.mandarin-ipa.text.48b29fb59e8b')}}<input type="file" :aria-label="tr('modules.mandarin-ipa.text.48b29fb59e8b')" accept=".xlsx,.csv,.tsv,.txt" :disabled="saving||loading" @change="chooseFile"/></label>
      <p class="hint">{{tr('modules.mandarin-ipa.text.57f4558c4e60')}}</p>
      <label v-if="source&&source.names.length>1">{{tr('modules.mandarin-ipa.text.b02781ee06c6')}}<select v-model.number="sheet" :aria-label="tr('modules.mandarin-ipa.text.d3721f15f282')" :disabled="saving||loading" @change="chooseSheet"><option v-for="(title,i) in source.names" :key="i" :value="i">{{title}}</option></select></label>
      <template v-if="rows.length">
        <div class="column-controls">
          <label>{{kind==='hanzi'?tr('modules.mandarin-ipa.text.00e5d5601ce3'):tr('modules.mandarin-ipa.text.68da7a474f1f')}}{{tr('modules.mandarin-ipa.text.c06d21404eee')}}<select v-model.number="columns.keyColumn" :aria-label="tr('modules.mandarin-ipa.text.8e307123157d')" :disabled="saving||loading"><option v-for="i in columnCount" :key="i" :value="i-1">{{columnLabel(i-1)}} {{tr('modules.mandarin-ipa.text.c06d21404eee')}}</option></select></label>
          <label>{{tr('modules.mandarin-ipa.text.166d7e44d167')}}<select v-model.number="columns.ipaColumn" :aria-label="tr('modules.mandarin-ipa.text.166d7e44d167')" :disabled="saving||loading"><option v-for="i in columnCount" :key="i" :value="i-1">{{columnLabel(i-1)}} {{tr('modules.mandarin-ipa.text.c06d21404eee')}}</option></select></label>
          <label>{{tr('modules.mandarin-ipa.text.6075765b6644')}}<input v-model.number="columns.startRow" type="number" min="1" :max="rows.length" step="1" :aria-label="tr('modules.mandarin-ipa.text.6075765b6644')" :disabled="saving||loading"/></label>
        </div>
        <p class="hint">{{fileName}} {{tr('modules.mandarin-ipa.text.c8b281787653')}} {{rows.length.toLocaleString()}} {{tr('modules.mandarin-ipa.text.106e8908d0b3')}} {{rawPreview.length}} {{tr('modules.mandarin-ipa.text.453cc9bd2c20')}}</p>
        <div class="raw-scroll"><table :aria-label="tr('modules.mandarin-ipa.text.1724d6423632')"><thead><tr><th>{{tr('modules.mandarin-ipa.text.c35a735c7a25')}}</th><th v-for="i in columnCount" :key="i" :class="{chosen:i-1===columns.keyColumn||i-1===columns.ipaColumn}">{{columnLabel(i-1)}}{{i-1===columns.keyColumn?' · '+(kind==='hanzi'?tr('modules.mandarin-ipa.text.00e5d5601ce3'):tr('modules.mandarin-ipa.text.4a5f01dedd3e')):i-1===columns.ipaColumn?tr('modules.mandarin-ipa.text.6b8d5c206c97'):''}}</th></tr></thead><tbody><tr v-for="row in rawPreview" :key="row.line" :class="{skipped:row.line<columns.startRow}"><th>{{row.line}}</th><td v-for="i in columnCount" :key="i" :class="{chosen:i-1===columns.keyColumn||i-1===columns.ipaColumn}" :title="row.cells[i-1]">{{(row.cells[i-1]??'').slice(0,160)}}</td></tr></tbody></table></div>
        <ToneMapping v-model="tones" :rows="toneRows" :disabled="saving||loading"/>
      </template>
      <label>{{tr('modules.mandarin-ipa.text.7965c9bd58dc')}}<input v-model="name" :aria-label="tr('modules.mandarin-ipa.text.7965c9bd58dc')" maxlength="60" :disabled="saving" :placeholder="tr('modules.mandarin-ipa.text.0b7cf78ae5a1')"/></label>
      <p v-if="loading" role="status">{{tr('modules.mandarin-ipa.text.3099d8516112')}}</p>
      <p v-else-if="error||saveError||(source&&parsed.error)" role="alert">{{error||saveError||parsed.error}}</p>
      <template v-if="preview.length">
        <p>{{selectedRows.rows.length.toLocaleString()}} {{tr('modules.mandarin-ipa.text.bee51fc6e31f')}} {{preview.length}} {{tr('modules.mandarin-ipa.text.5005798a58af')}}</p>
        <table :aria-label="tr('modules.mandarin-ipa.text.97f991e1664d')"><thead><tr><th>{{kind==='hanzi'?tr('modules.mandarin-ipa.text.00e5d5601ce3'):tr('modules.mandarin-ipa.text.68da7a474f1f')}}</th><th>{{tr('modules.mandarin-ipa.text.8689de4ed54b')}}</th><th>{{tr('modules.mandarin-ipa.text.54a8366124bb')}}</th></tr></thead><tbody><tr v-for="(row,i) in preview" :key="i"><td>{{row[0]}}</td><td class="ipa-text">{{row[1]}}</td><td class="ipa-text">{{row[2]}}</td></tr></tbody></table>
      </template>
      <p class="hint">{{tr('modules.mandarin-ipa.text.d03eee856c29')}}</p>
    </div>
    <template #footer><button :disabled="saving||loading" @click="emit('close')">{{tr('modules.mandarin-ipa.text.2cd0f3be8738')}}</button><button class="primary" :disabled="saving||loading||!parsed.rows.length||!name.trim()" @click="save">{{saving?tr('modules.mandarin-ipa.text.6bdb4435095e'):replaceTarget?tr('modules.mandarin-ipa.text.000f7ee946b0'):tr('modules.mandarin-ipa.text.d1285773cb3d')}}</button></template>
  </ModalDialog>
</template>
<style scoped>
.m13-import{display:grid;gap:10px;min-width:0}.m13-import label{display:grid;gap:4px}.m13-import input,.m13-import select{width:100%;min-width:0}.m13-import p{margin:0;line-height:1.6;overflow-wrap:anywhere}.hint{color:var(--muted);font-size:var(--support-size)}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{text-align:left;padding:5px 8px;border:1px solid var(--border);overflow-wrap:anywhere}.ipa-text{font-size:1.2rem}[role=alert]{color:var(--danger)}.column-controls{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.raw-scroll{overflow:auto;max-height:230px}.raw-scroll table{min-width:100%;width:max-content;table-layout:fixed}.raw-scroll th,.raw-scroll td{min-width:90px;max-width:200px}.chosen{background:color-mix(in srgb,var(--accent) 10%,var(--panel));font-weight:600}.skipped{color:var(--muted)}@media(max-width:600px){.column-controls{grid-template-columns:1fr}}
</style>
