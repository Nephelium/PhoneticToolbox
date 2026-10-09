<script setup lang="ts">
import {vTimePrecision} from '../design/time-precision.ts';
import {ref,watch} from 'vue';import ModalDialog from './ModalDialog.vue';
import {rules,settingLabels,validateSettings,type Settings,type SettingKey} from '../modules/parameter-estimation/state.ts';
const props=defineProps<{draft:Settings}>();const emit=defineEmits<{close:[];apply:[value:Settings];draft:[value:Settings]}>();
const draft=ref({...props.draft}),tab=ref<'general'|'reaper'>('general'),error=ref('');
const reaper=new Set(['min_f0','max_f0','reaper_hilbert','reaper_no_highpass']);
const keys=Object.keys(rules) as SettingKey[];
watch(draft,v=>emit('draft',{...v}),{deep:true});
function apply(){error.value=validateSettings(draft.value);if(!error.value)emit('apply',{...draft.value});}
</script>
<template><ModalDialog title="分析设置" @close="emit('close')">
<p class="muted">仅影响下一次分析；当前结果与试听不随设置变化。</p>
<div class="parameter-toolbar"><button :aria-pressed="tab==='general'" @click="tab='general'">常用设置 · 10</button><button :aria-pressed="tab==='reaper'" @click="tab='reaper'">REAPER设置 · 4</button></div>
<div class="m01-settings"><label v-for="key in keys.filter(k=>reaper.has(k)===(tab==='reaper'))" :key="key" class="setting-row"><span>{{settingLabels[key][0]}}<small>{{settingLabels[key][1]}}</small></span>
<input v-if="rules[key].type==='boolean'" v-model="draft[key]" type="checkbox" :aria-label="settingLabels[key][0]"/>
<input v-time-precision="settingLabels[key][1]==='ms'?'ms':undefined" v-else v-model.number="draft[key]" type="number" :min="rules[key].minimum??rules[key].exclusiveMinimum" :max="rules[key].maximum" :step="rules[key].type==='integer'?1:'any'" :aria-label="settingLabels[key][0]"/></label></div>
<p class="hint">最小/最大基频同时影响Praat、REAPER与WM链及部分依赖基频的指标；不等于共振峰上限。默认分析窗40 ms，WM jitter/shimmer实际使用max(160 ms, 分析窗)。有声帧由Praat或REAPER检出有限正基频判定；关闭此项时使用能量静音掩码。此掩码不裁掉唇形和文本标签。</p>
<p v-if="error" role="alert" class="error-banner">{{error}}</p><template #footer><button @click="emit('close')">取消</button><button class="primary" @click="apply">应用设置</button></template></ModalDialog></template>
