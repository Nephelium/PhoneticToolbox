<script setup lang="ts">
import {tr} from '../../i18n/index.ts';
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
    <legend>{{tr('modules.mandarin-ipa.text.fa8d290944e1')}}</legend>
    <label>{{tr('modules.mandarin-ipa.text.bedfd707b4a3')}}<select :value="modelValue.mode" :aria-label="tr('modules.mandarin-ipa.text.dfc4efdcd2dc')" @change="update({mode:($event.target as HTMLSelectElement).value as ToneSettings['mode']})"><option value="preserve">{{tr('modules.mandarin-ipa.text.676c1cba5ebb')}}</option><option value="values">{{tr('modules.mandarin-ipa.text.eed4a351167b')}}</option><option value="classes">{{tr('modules.mandarin-ipa.text.f1eb8ff3fdef')}}</option></select></label>
    <p class="hint">{{tr('modules.mandarin-ipa.text.1373c4864670')}}</p>
    <p v-if="modelValue.mode==='values'" class="hint">{{tr('modules.mandarin-ipa.text.4b257a68efc6')}}</p>
    <template v-if="modelValue.mode==='classes'">
      <label>{{tr('modules.mandarin-ipa.text.5ebb0fd22eba')}}<select :value="modelValue.classOutput" :aria-label="tr('modules.mandarin-ipa.text.39f2b4753556')" @change="update({classOutput:($event.target as HTMLSelectElement).value as ToneSettings['classOutput']})"><option value="original">{{tr('modules.mandarin-ipa.text.9043b1c746a0')}}</option><option value="values">{{tr('modules.mandarin-ipa.text.c854c96b08f3')}}</option></select></label>
      <div v-if="modelValue.classOutput==='values'&&classes.length" class="mapping-scroll">
        <table :aria-label="tr('modules.mandarin-ipa.text.5cb478baf9cb')"><thead><tr><th>{{tr('modules.mandarin-ipa.text.4c43de2b848a')}}</th><th>{{tr('modules.mandarin-ipa.text.020bdef0ce35')}}</th><th>{{tr('modules.mandarin-ipa.text.8f1362a5a7a3')}}</th></tr></thead><tbody><tr v-for="key in classes.slice(0,128)" :key="key"><td>{{key}}</td><td><input :value="modelValue.mapping[key]??''" :aria-label="tr('modules.mandarin-ipa.text.1ab444939c05')+key+tr('modules.mandarin-ipa.text.48b78cc0f72c')" maxlength="8" :placeholder="tr('modules.mandarin-ipa.text.2bb56c965a99')" @input="map(key,$event)"/></td><td class="ipa-text">{{symbols(modelValue.mapping[key]??'')}}</td></tr></tbody></table>
      </div>
      <p v-if="!classes.length" class="hint">{{tr('modules.mandarin-ipa.text.8459e747e06f')}}</p>
    </template>
  </fieldset>
</template>
<style scoped>
.tone-mapping{display:grid;gap:8px;min-width:0;border:1px solid var(--border);border-radius:8px;padding:10px}.tone-mapping label{display:grid;gap:4px}.tone-mapping input,.tone-mapping select{width:100%;min-width:0}.tone-mapping p{margin:0;line-height:1.6}.hint{color:var(--muted);font-size:var(--support-size)}.mapping-scroll{max-height:220px;overflow:auto}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{text-align:left;padding:5px;border:1px solid var(--border);overflow-wrap:anywhere}.ipa-text{font-family:var(--ipa-font);font-size:1.2rem}
</style>
