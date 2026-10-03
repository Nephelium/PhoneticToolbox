<script setup lang="ts">
import type {EggTaskConfig} from '../../platform/research.ts';
const config=defineModel<EggTaskConfig>({required:true});
defineProps<{canAutoDb?:boolean}>();
const emit=defineEmits<{autoDb:[]}>();
</script>
<template><div class="egg-parameter-row" aria-label="EGG 检测参数">
<section class="parameter-group"><div class="parameter-heading"><span>事件检测</span><label class="check" title="200 ms 局部窗口，100 ms 步长；峰阈值=max(0.01,0.6×局部绝对峰值)。只作用于峰，谷阈值保持手动。"><input v-model="config.auto_prominence" type="checkbox"/>峰自动</label></div>
<div class="egg-field-grid">
<label title="相对周围基线的幅度差，非百分比。自动开启时此手动值不参与计算。">峰显著度<input v-model.number="config.peak_prominence" aria-label="EGG 峰显著度" type="number" min="0" max="10" step=".001" :disabled="config.auto_prominence"/></label>
<label title="在负 EGG 上检测谷，相对基线的幅度差。0.01 为起始值，须观察误检与漏检。">谷显著度<input v-model.number="config.valley_prominence" aria-label="EGG 谷显著度" type="number" min="0" max="10" step=".001"/></label>
<label>GCI<select v-model="config.gci_method"><option value="slope">斜率</option><option value="scale">尺度 0.25</option></select></label>
<label>GOI<select v-model="config.goi_method"><option value="slope">斜率</option><option value="scale">尺度 0.25</option></select></label>
</div></section>
<section class="parameter-group"><div class="parameter-heading"><span>语谱图与 F0</span><button type="button" :disabled="!canAutoDb" title="按当前选区 0–5000 Hz 功率谱的最大值设置上限，下限为上限减 50 dB。应用一次后可手动调整。" @click="emit('autoDb')">自动 dB</button></div>
<div class="egg-field-grid">
<label>谱窗 ms<input v-model.number="config.spec_window_ms" type="number" min="5" max="50" step="1"/></label>
<label>dB 下限<input v-model.number="config.spec_vmin" aria-label="EGG dB 下限" type="number" min="-160" max="20"/></label>
<label>dB 上限<input v-model.number="config.spec_vmax" aria-label="EGG dB 上限" type="number" min="-160" max="20"/></label>
</div><div class="egg-checks"><label title="音频 Praat AC，搜索范围 30–800 Hz"><input v-model="config.keep_praat_f0" type="checkbox"/>Praat F0</label><label><input v-model="config.keep_gci_f0" type="checkbox"/>GCI F0</label><label title="音频 REAPER，搜索范围 30–800 Hz"><input v-model="config.keep_reaper_f0" type="checkbox"/>REAPER F0</label></div>
</section></div></template>
<style scoped>
.egg-parameter-row{display:grid;gap:10px;border-top:1px solid var(--border);padding-top:8px}.parameter-group{display:grid;gap:6px;min-width:0}.parameter-heading{display:flex;align-items:center;justify-content:space-between;gap:6px;font-size:var(--control-size,12px);color:var(--muted)}.parameter-heading button{min-height:26px;padding:2px 6px;font-size:inherit}.parameter-heading .check{display:flex;align-items:center;gap:4px;color:var(--text)}.egg-field-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,145px),1fr));gap:5px 10px}.egg-field-grid label{display:grid;grid-template-columns:minmax(0,1fr) minmax(64px,86px);align-items:center;gap:5px;font-size:var(--control-size,12px)}.egg-field-grid input,.egg-field-grid select{width:100%;min-width:0;min-height:28px;padding:3px 5px;font-size:inherit;line-height:1.25}.egg-checks{display:flex;align-items:center;flex-wrap:wrap;gap:6px 14px;font-size:var(--control-size,12px)}.egg-checks label{display:flex;align-items:center;gap:4px}
</style>
