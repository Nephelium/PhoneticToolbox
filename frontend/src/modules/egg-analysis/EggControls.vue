<script setup lang="ts">
import type {EggTaskConfig} from '../../platform/research.ts';
const config=defineModel<EggTaskConfig>({required:true});
defineProps<{start:number;end:number;duration:number}>();
const emit=defineEmits<{range:[start:number,end:number]}>();
</script>
<template><div class="egg-control-row">
<div class="control-group"><span class="control-group-label">显示范围</span><div class="group-controls egg-field-grid">
<label>时长 s<input aria-label="EGG 选区时长" :value="Number((end-start).toFixed(6))" type="number" min=".001" :max="duration" step=".001" @change="emit('range',start,start+Number(($event.target as HTMLInputElement).value))"/></label>
<label>微观 ms<input v-model.number="config.micro_width_ms" aria-label="EGG 微观窗口" type="number" min="5" max="5000" step="5"/></label>
</div></div>
</div></template>
<style scoped>
.egg-control-row{display:grid;gap:8px;min-width:0}.control-group{display:grid;gap:5px;min-width:0}.control-group-label{color:var(--muted);font-size:var(--control-size,12px)}.group-controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,145px),1fr));gap:5px 10px}.group-controls label{display:grid;grid-template-columns:minmax(0,1fr) minmax(64px,86px);align-items:center;gap:5px;font-size:var(--control-size,12px)}.group-controls input{width:100%;min-width:0;min-height:28px;padding:3px 5px;font-size:inherit;line-height:1.25;font-variant-numeric:tabular-nums}
</style>
