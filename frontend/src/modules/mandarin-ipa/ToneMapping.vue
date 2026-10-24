<script setup lang="ts">
import {computed} from 'vue';
import {toneClasses,toneSymbols,type ToneSettings} from './tones.ts';
const props=defineProps<{modelValue:ToneSettings;rows:readonly (readonly string[])[];disabled?:boolean}>();
const emit=defineEmits<{'update:modelValue':[value:ToneSettings]}>();
const classes=computed(()=>toneClasses(props.rows));
function update(patch:Partial<ToneSettings>){emit('update:modelValue',{...props.modelValue,...patch});}
function map(key:string,event:Event){update({mapping:{...props.modelValue.mapping,[key]:(event.target as HTMLInputElement).value}});}
function symbols(value:string){try{return toneSymbols(value);}catch{return '';}}
</script>
<template>
  <fieldset class="tone-mapping" :disabled="disabled">
    <legend>声调处理</legend>
    <label>音标末尾的声调<select :value="modelValue.mode" aria-label="音标声调表示方式" @change="update({mode:($event.target as HTMLSelectElement).value as ToneSettings['mode']})"><option value="preserve">保留原样</option><option value="values">五度调值 → 声调符号</option><option value="classes">调类</option></select></label>
    <p class="hint">识别音标末尾的普通、上标或全角数字，例如 ma55、ma⁵⁵、ma５５。已有的 ˥˧˩ 等声调符号保留。</p>
    <p v-if="modelValue.mode==='values'" class="hint">1 为最低，5 为最高，例如 55 → ˥˥，35 → ˧˥，214 → ˨˩˦。</p>
    <template v-if="modelValue.mode==='classes'">
      <label>调类显示<select :value="modelValue.classOutput" aria-label="调类显示方式" @change="update({classOutput:($event.target as HTMLSelectElement).value as ToneSettings['classOutput']})"><option value="original">保留原调类</option><option value="values">映射到调值并显示声调符号</option></select></label>
      <div v-if="modelValue.classOutput==='values'&&classes.length" class="mapping-scroll">
        <table aria-label="调类映射表"><thead><tr><th>原调类</th><th>调值</th><th>声调符号</th></tr></thead><tbody><tr v-for="key in classes.slice(0,128)" :key="key"><td>{{key}}</td><td><input :value="modelValue.mapping[key]??''" :aria-label="'调类 '+key+' 的调值'" maxlength="8" placeholder="例如 55" @input="map(key,$event)"/></td><td class="ipa-text">{{symbols(modelValue.mapping[key]??'')}}</td></tr></tbody></table>
      </div>
      <p v-if="!classes.length" class="hint">选定音标列后会列出末尾的调类数字，支持 1a、1b 等分组。</p>
    </template>
  </fieldset>
</template>
<style scoped>
.tone-mapping{display:grid;gap:8px;min-width:0;border:1px solid var(--border);border-radius:8px;padding:10px}.tone-mapping label{display:grid;gap:4px}.tone-mapping input,.tone-mapping select{width:100%;min-width:0}.tone-mapping p{margin:0;line-height:1.6}.hint{color:var(--muted);font-size:var(--support-size)}.mapping-scroll{max-height:220px;overflow:auto}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{text-align:left;padding:5px;border:1px solid var(--border);overflow-wrap:anywhere}.ipa-text{font-family:var(--ipa-font);font-size:1.2rem}
</style>
