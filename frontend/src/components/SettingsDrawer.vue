<script setup lang="ts">
import {vTimePrecision} from '../design/time-precision.ts';
import {ref,watch} from 'vue';import ModalDialog from './ModalDialog.vue';
import {rules,settingLabels,validateSettings,type Settings,type SettingKey} from '../modules/parameter-estimation/state.ts';
const props=defineProps<{draft:Settings}>();const emit=defineEmits<{close:[];apply:[value:Settings];draft:[value:Settings]}>();
const draft=ref({...props.draft}),tab=ref<'general'|'reaper'>('general'),error=ref('');
const reaper=new Set<SettingKey>(['reaper_hilbert','reaper_no_highpass']);
const keys=Object.keys(rules) as SettingKey[];
const groups:Record<'general'|'reaper',SettingKey[]>={general:[...keys.filter(k=>!reaper.has(k)&&k!=='min_f0'&&k!=='max_f0'),'min_f0','max_f0'],reaper:keys.filter(k=>reaper.has(k))};
watch(draft,v=>emit('draft',{...v}),{deep:true});
function apply(){error.value=validateSettings(draft.value);if(!error.value)emit('apply',{...draft.value});}
</script>
<template><ModalDialog title="分析设置" @close="emit('close')">
<p class="muted">仅影响下一次分析；当前结果与试听不随设置变化。</p>
<div class="parameter-toolbar"><button :aria-pressed="tab==='general'" @click="tab='general'">常用设置 · {{groups.general.length}}</button><button :aria-pressed="tab==='reaper'" @click="tab='reaper'">REAPER设置 · {{groups.reaper.length}}</button></div>
<div class="m01-settings"><label v-for="key in groups[tab]" :key="key" class="setting-row"><span>{{settingLabels[key][0]}}<small>{{settingLabels[key][1]}}</small></span>
<input v-if="rules[key].type==='boolean'" v-model="draft[key]" type="checkbox" :aria-label="settingLabels[key][0]"/>
<input v-time-precision="settingLabels[key][1]==='ms'?'ms':undefined" v-else v-model.number="draft[key]" type="number" :min="rules[key].minimum??rules[key].exclusiveMinimum" :max="rules[key].maximum" :step="rules[key].type==='integer'?1:'any'" :aria-label="settingLabels[key][0]"/></label></div>
<p class="hint">最小/最大基频用于 Praat、REAPER、WM 及依赖基频的指标，共振峰上限单独设置。默认分析窗为 40 ms，WM jitter/shimmer 使用 max(160 ms, 分析窗)。有声帧按 Praat 或 REAPER 检出的有限正基频筛选；关闭有声帧筛选后按能量排除静音。唇形和文本标签保留完整时间范围。</p>
<p v-if="error" role="alert" class="error-banner">{{error}}</p><template #footer><button @click="emit('close')">取消</button><button class="primary" @click="apply">应用设置</button></template></ModalDialog></template>
