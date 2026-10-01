<script setup lang="ts">
import type {EggTaskConfig} from '../../platform/research.ts';
const config=defineModel<EggTaskConfig>({required:true});
const order=defineModel<number|string|null>('order');
defineProps<{start:number;end:number;duration:number;canUpdate:boolean;canExport:boolean;pending:boolean}>();
const emit=defineEmits<{range:[start:number,end:number];update:[];cancel:[];save:[];inverse:[]}>();
</script>
<template><div class="egg-control-row">
<div class="control-group"><span class="control-group-label">选区</span><div class="group-controls">
<label>起点 s<input aria-label="EGG 选区起点" :value="start" type="number" min="0" :max="duration" step=".001" @change="emit('range',Number(($event.target as HTMLInputElement).value),Number(($event.target as HTMLInputElement).value)+end-start)"/></label>
<label>时长 s<input aria-label="EGG 选区时长" :value="Number((end-start).toFixed(6))" type="number" min=".001" :max="duration" step=".001" @change="emit('range',start,start+Number(($event.target as HTMLInputElement).value))"/></label>
<label>微观 ms<input v-model.number="config.micro_width_ms" aria-label="EGG 微观窗口" type="number" min="5" max="5000" step="5"/></label>
</div></div>
<div class="control-group playback-group"><span class="control-group-label">选区试听</span><slot name="playback"/></div>
<div class="control-group"><span class="control-group-label">导出与逆滤波</span><div class="group-controls">
<button :disabled="!canExport" @click="emit('save')">保存 CSV / 三图</button><span class="control-divider"/><label>LP 阶数<input v-model.number="order" type="number" min="1" max="256" placeholder="自动"/></label><button :disabled="!canExport" @click="emit('inverse')">逆滤波 IF</button>
</div></div></div></template>
<style scoped>
.egg-control-row{display:flex;flex-wrap:wrap;align-items:flex-start;gap:14px 22px}.control-group{display:flex;flex-direction:column;gap:6px;min-width:0;max-width:100%}.control-group-label{color:var(--muted);font-size:12px;font-weight:500}.group-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.group-controls label{display:flex;gap:5px;align-items:center;font-size:12px;white-space:nowrap}.group-controls input{width:65px;min-height:32px;padding:5px 7px;font-variant-numeric:tabular-nums}.group-controls button{min-height:32px;padding:5px 9px;font-size:12px}.control-divider{height:20px;border-left:1px solid var(--border);margin:0 2px}.playback-group{flex:none}.playback-group :deep(.transport-compact){flex:none;min-width:0}.playback-group :deep(.audio-transport){margin:0;gap:8px}.playback-group :deep(.audio-transport button){min-height:32px;padding:5px 9px;font-size:12px}.playback-group :deep(.audio-transport .mono){font-size:11px;white-space:nowrap}
</style>
