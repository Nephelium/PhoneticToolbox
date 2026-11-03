<script setup lang="ts">
import {tr} from '../../i18n/index.ts';
import type {VariantInput,PitchMode} from './state.ts';
defineProps<{values:VariantInput;summary:string}>();
const mode=defineModel<PitchMode>('mode',{required:true});
</script>
<template><div class="variant-settings">
<label>{{tr('modules.pitch-manipulation.tempo_multiplier')}}<input v-model="values.speed" :aria-label="tr('modules.pitch-manipulation.tempo_multiplier')" type="text" placeholder="0.8, 1, 1.2"/></label>
<label>{{tr('modules.pitch-manipulation.pitch_adjustment')}}<select v-model="mode" :aria-label="tr('modules.pitch-manipulation.pitch_adjustment_mode')"><option value="ratio">{{tr('modules.pitch-manipulation.multiplier')}}</option><option value="hz">{{tr('modules.pitch-manipulation.shift_hz')}}</option></select></label>
<label v-if="mode==='ratio'">{{tr('modules.pitch-manipulation.pitch_multiplier')}}<input v-model="values.pitch_ratio" :aria-label="tr('modules.pitch-manipulation.pitch_multiplier')" type="text" placeholder="1, 1.2"/></label>
<label v-else>{{tr('modules.pitch-manipulation.pitch_shift_hz')}}<input v-model="values.pitch_hz" :aria-label="tr('modules.pitch-manipulation.pitch_shift_hz')" type="text" placeholder="-20, 0, 20"/></label>
<small>{{tr('modules.pitch-manipulation.separate_multiple_values_with_commas_or_spaces_tempo_and')}}</small><strong role="status">{{summary}}</strong>
</div></template>
<style scoped>.variant-settings{display:flex;flex-direction:column;gap:8px;min-width:0}.variant-settings label{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px}.variant-settings input{width:160px;max-width:100%;min-width:0}.variant-settings small{color:var(--muted);line-height:1.6}.variant-settings strong{font-size:var(--support-size);font-weight:500}</style>
