<script setup lang="ts">
import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {vTimePrecision} from '../design/time-precision.ts';
import {ref,watch} from 'vue';import ModalDialog from './ModalDialog.vue';
import {rules,settingLabels,validateSettingsMessage,type Settings,type SettingKey} from '../modules/parameter-estimation/state.ts';
const props=defineProps<{draft:Settings}>();const emit=defineEmits<{close:[];apply:[value:Settings];draft:[value:Settings]}>();
const draft=ref({...props.draft}),tab=ref<'general'|'reaper'>('general'),error=messageRef();
const reaper=new Set<SettingKey>(['reaper_hilbert','reaper_no_highpass']);
const keys=Object.keys(rules) as SettingKey[];
const groups:Record<'general'|'reaper',SettingKey[]>={general:[...keys.filter(k=>!reaper.has(k)&&k!=='min_f0'&&k!=='max_f0'),'min_f0','max_f0'],reaper:keys.filter(k=>reaper.has(k))};
watch(draft,v=>emit('draft',{...v}),{deep:true});
function apply(){error.value=validateSettingsMessage(draft.value);if(!error.value)emit('apply',{...draft.value});}
</script>
<template><ModalDialog :title="tr('components.analysis_settings')" @close="emit('close')">
<p class="muted">{{tr('components.applies_only_to_the_next_analysis_current_results_and')}}</p>
<div class="parameter-toolbar"><button :aria-pressed="tab==='general'" @click="tab='general'">{{tr('components.common_settings')}} {{groups.general.length}}</button><button :aria-pressed="tab==='reaper'" @click="tab='reaper'">{{tr('components.reaper_settings')}} {{groups.reaper.length}}</button></div>
<div class="m01-settings"><label v-for="key in groups[tab]" :key="key" class="setting-row"><span>{{settingLabels[key][0]}}<small>{{settingLabels[key][1]}}</small></span>
<input v-if="rules[key].type==='boolean'" v-model="draft[key]" type="checkbox" :aria-label="settingLabels[key][0]"/>
<input v-time-precision="settingLabels[key][1]==='ms'?'ms':undefined" v-else v-model.number="draft[key]" type="number" :min="rules[key].minimum??rules[key].exclusiveMinimum" :max="rules[key].maximum" :step="rules[key].type==='integer'?1:'any'" :aria-label="settingLabels[key][0]"/></label></div>
<p class="hint">{{tr('components.minimum_and_maximum_f0_apply_to_praat_reaper_wm')}}</p>
<p v-if="error" role="alert" class="error-banner">{{error}}</p><template #footer><button @click="emit('close')">{{tr('components.cancel')}}</button><button class="primary" @click="apply">{{tr('components.apply_settings')}}</button></template></ModalDialog></template>
