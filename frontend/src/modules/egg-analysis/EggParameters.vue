<script setup lang="ts">
import type {EggTaskConfig} from '../../platform/research.ts';
const config=defineModel<EggTaskConfig>({required:true});
</script>
<template><div class="egg-parameter-row" aria-label="EGG 检测参数"><div class="parameter-group"><span class="parameter-group-label">事件检测</span><div class="parameter-fields">
<label>峰显著度<input v-model.number="config.peak_prominence" type="number" min="0" max="10" step=".001" :disabled="config.auto_prominence"/></label><label><input v-model="config.auto_prominence" type="checkbox"/>自动</label><label>谷显著度<input v-model.number="config.valley_prominence" type="number" min="0" max="10" step=".001"/></label>
<label>GCI<select v-model="config.gci_method"><option value="slope">斜率</option><option value="scale">尺度 0.25</option></select></label><label>GOI<select v-model="config.goi_method"><option value="slope">斜率</option><option value="scale">尺度 0.25</option></select></label>
</div></div><div class="parameter-group"><span class="parameter-group-label">语谱图与 F0</span><div class="parameter-fields">
<label>谱窗 ms<input v-model.number="config.spec_window_ms" type="number" min="5" max="50" step="1"/></label><label>dB 下限<input v-model.number="config.spec_vmin" type="number" min="-160" max="20"/></label><label>上限<input v-model.number="config.spec_vmax" type="number" min="-160" max="20"/></label>
<label><input v-model="config.keep_praat_f0" type="checkbox"/>Praat F0</label><label><input v-model="config.keep_gci_f0" type="checkbox"/>GCI F0</label>
</div></div></div></template>
<style scoped>.egg-parameter-row{display:flex;flex-wrap:wrap;gap:14px 22px;align-items:flex-start;border-top:1px solid var(--border);padding-top:10px}.parameter-group{display:flex;flex-direction:column;gap:6px;max-width:100%}.parameter-group-label{font-size:12px;color:var(--muted);font-weight:500}.parameter-fields{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.parameter-fields label{display:flex;align-items:center;gap:5px;font-size:12px;white-space:nowrap}.parameter-fields input[type=number]{width:62px;padding:5px 7px;min-height:32px}.parameter-fields select{width:86px;padding:5px 6px;min-height:32px}</style>
