<script setup lang="ts">
import {computed,ref} from 'vue';
import ModalDialog from '../../components/ModalDialog.vue';
import ToneMapping from './ToneMapping.vue';
import type {CustomTable} from './custom-tables.ts';
import {defaultToneSettings,tonePreview,transcribeTone,type ToneSettings} from './tones.ts';
const props=defineProps<{table:CustomTable;saving:boolean;saveError:string}>();
const emit=defineEmits<{close:[];save:[tones:ToneSettings]}>();
const tones=ref<ToneSettings>(JSON.parse(JSON.stringify(props.table.tones??defaultToneSettings())));
const error=computed(()=>{try{tonePreview(props.table.rows,tones.value);return '';}catch(cause){return (cause as Error).message;}});
function result(value:string){try{return transcribeTone(value,tones.value);}catch{return '待设置声调';}}
</script>
<template>
  <ModalDialog title="字表声调映射" :close-disabled="saving" @close="emit('close')">
    <div class="tone-dialog">
      <p>{{table.name}} · {{table.rows.length.toLocaleString()}} 条记录</p>
      <ToneMapping v-model="tones" :rows="table.rows" :disabled="saving"/>
      <table aria-label="声调结果预览"><thead><tr><th>{{table.kind==='hanzi'?'汉字':'拼音音节'}}</th><th>原始音标</th><th>转换结果</th></tr></thead><tbody><tr v-for="(row,i) in table.rows.slice(0,8)" :key="i"><td>{{row[0]}}</td><td class="ipa-text">{{row[1]}}</td><td class="ipa-text">{{result(row[1])}}</td></tr></tbody></table>
      <p v-if="error||saveError" role="alert">{{saveError||error}}</p>
    </div>
    <template #footer><button :disabled="saving" @click="emit('close')">取消</button><button class="primary" :disabled="saving||!!error" @click="emit('save',JSON.parse(JSON.stringify(tones)))">{{saving?'正在保存…':'保存声调映射'}}</button></template>
  </ModalDialog>
</template>
<style scoped>
.tone-dialog{display:grid;gap:12px;min-width:0}.tone-dialog p{margin:0}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{padding:5px 8px;text-align:left;border:1px solid var(--border);overflow-wrap:anywhere}.ipa-text{font-size:1.2rem}[role=alert]{color:var(--danger)}
</style>
