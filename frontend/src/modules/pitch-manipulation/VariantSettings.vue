<script setup lang="ts">
import type {VariantInput,PitchMode} from './state.ts';
defineProps<{values:VariantInput;summary:string}>();
const mode=defineModel<PitchMode>('mode',{required:true});
</script>
<template><div class="variant-settings">
<label>语速倍率<input v-model="values.speed" aria-label="语速倍率" type="text" placeholder="0.8, 1, 1.2"/></label>
<label>音高调整<select v-model="mode" aria-label="音高调整方式"><option value="ratio">倍率</option><option value="hz">偏移 Hz</option></select></label>
<label v-if="mode==='ratio'">音高倍率<input v-model="values.pitch_ratio" aria-label="音高倍率" type="text" placeholder="1, 1.2"/></label>
<label v-else>音高偏移 Hz<input v-model="values.pitch_hz" aria-label="音高偏移 Hz" type="text" placeholder="-20, 0, 20"/></label>
<small>多个值用逗号或空格分隔。语速与音高按所有组合生成，音高只使用当前选择的方式。</small><strong role="status">{{summary}}</strong>
</div></template>
<style scoped>.variant-settings{display:flex;flex-direction:column;gap:8px;min-width:0}.variant-settings label{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px}.variant-settings input{width:160px;max-width:100%;min-width:0}.variant-settings small{color:var(--muted);line-height:1.6}.variant-settings strong{font-size:var(--support-size);font-weight:500}</style>
