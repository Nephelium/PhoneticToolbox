<script setup lang="ts">
import {ref,computed} from 'vue';import ModalDialog from '../../components/ModalDialog.vue';import type {ResearchFile,EggTaskConfig} from '../../platform/research.ts';
const props=defineProps<{files:ResearchFile[];config:EggTaskConfig;busy:boolean;error?:string}>();const emit=defineEmits<{close:[];cancel:[];submit:[files:ResearchFile[],config:EggTaskConfig]}>();
const settings=ref({...props.config}),selected=ref(props.files.map(f=>f.id));const all=computed(()=>props.files.length>0&&selected.value.length===props.files.length);
</script>
<template>
<ModalDialog title="EGG 批量分析" wide :close-disabled="busy" @close="emit('close')">
  <p class="batch-intro">分析所选文件的完整滤波信号，每个文件独立保存结果。单文件选区不带入批次。</p>
  <p v-if="error" role="alert" class="error-banner">{{error}}</p>
  <fieldset class="batch-settings" :disabled="busy">
    <legend class="visually-hidden">批次参数</legend>
    <section class="batch-section"><h3>滤波与事件</h3><div class="batch-values">
      <label>高通 Hz<input v-model.number="settings.highpass_cutoff" type="number" min="1" max="47999" step="1"/></label>
      <label>低通 Hz<input v-model.number="settings.lowpass_cutoff" type="number" min="1" max="47999" step="1"/></label>
      <label>GCI<select v-model="settings.gci_method"><option value="slope">斜率</option><option value="scale">尺度 0.25</option></select></label>
      <label>GOI<select v-model="settings.goi_method"><option value="scale">尺度 0.25</option><option value="slope">斜率</option></select></label>
    </div></section>
    <section class="batch-section"><h3>输出选项</h3><div class="batch-options">
      <label><input v-model="settings.flip_channels" type="checkbox"/>交换 EGG / 音频声道</label>
      <label><input v-model="settings.generate_images" type="checkbox"/>同时导出三张图片</label>
      <label><input v-model="settings.keep_praat_f0" type="checkbox"/>Praat F0</label>
      <label><input v-model="settings.keep_gci_f0" type="checkbox"/>GCI F0</label>
      <label title="音频 REAPER，搜索范围 30–800 Hz"><input v-model="settings.keep_reaper_f0" type="checkbox"/>REAPER F0</label>
    </div><label class="batch-silence">静音阈值<input v-model.number="settings.silence_threshold" type="number" min="0" max="1" step=".001"/></label>
    <p class="hint">20 ms 平均绝对振幅，仅遮罩 CSV。批次参数独立保存。</p></section>
  </fieldset>
  <div class="batch-file-heading"><h3>待处理文件</h3><label><input type="checkbox" :checked="all" :disabled="busy" @change="selected=all?[]:files.map(f=>f.id)"/>全选（{{selected.length}} / {{files.length}}）</label></div>
  <div class="batch-files"><label v-for="file in files" :key="file.id"><input v-model="selected" type="checkbox" :value="file.id" :disabled="busy"/><span>{{file.name}}</span></label></div>
  <template #footer><button v-if="busy" @click="emit('cancel')">取消提交</button><button :disabled="busy" @click="emit('close')">返回工作台</button><button class="primary" :disabled="busy||!selected.length" @click="emit('submit',files.filter(f=>selected.includes(f.id)),settings)">{{busy?'正在提交…':'提交所选文件'}}</button></template>
</ModalDialog>
</template>
<style scoped>
.batch-intro{margin:0 0 12px;color:var(--muted)}.batch-settings{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 14px;padding:0;border:0;min-width:0}.batch-section{min-width:0;border:1px solid var(--border);border-radius:var(--radius);padding:10px;display:grid;gap:8px;align-content:start}.batch-section h3,.batch-file-heading h3{font-size:var(--control-size,12px);margin:0;color:var(--muted)}.batch-values{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 12px}.batch-values label,.batch-silence{display:grid;grid-template-columns:minmax(0,1fr) minmax(70px,100px);align-items:center;gap:6px;min-width:0}.batch-values input,.batch-values select,.batch-silence input{width:100%;min-width:0;min-height:30px;padding:4px 6px;font-size:inherit}.batch-options{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.batch-options label,.batch-files label,.batch-file-heading label{display:flex;align-items:center;gap:6px;min-width:0}.batch-silence{max-width:240px}.batch-section .hint{margin:0}.batch-file-heading{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}.batch-files{max-height:240px;overflow:auto;margin-top:8px;border:1px solid var(--border);border-radius:var(--radius)}.batch-files label{padding:6px 10px;min-height:32px;border-bottom:1px solid var(--border)}.batch-files label:last-child{border-bottom:0}.batch-files span{overflow-wrap:anywhere}
@media(max-width:1000px){.batch-settings{grid-template-columns:1fr}}@media(max-width:560px){.batch-values,.batch-options{grid-template-columns:1fr}}
</style>
