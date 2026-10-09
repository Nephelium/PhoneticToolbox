<script setup lang="ts">
defineProps<{format:'json'|'xlsx'|'csv';busy:boolean;canConfirm:boolean;confirmVisible:boolean}>();
defineEmits<{'update:format':[value:'json'|'xlsx'|'csv'];export:[];confirm:[]}>();
</script>
<template>
 <div class="m15-export-controls">
  <label>结果格式<select aria-label="结果格式" :value="format" :disabled="busy" @change="$emit('update:format',($event.target as HTMLSelectElement).value as 'json'|'xlsx'|'csv')"><option value="json">JSON（完整溯源）</option><option value="xlsx">XLSX（分析表格）</option><option value="csv">CSV（分析表格）</option></select></label>
  <button class="primary" :disabled="busy" @click="$emit('export')">导出结果</button>
  <button v-if="confirmVisible" :disabled="busy||!canConfirm" @click="$emit('confirm')">确认导出文件已保存</button>
 </div>
</template>
<style scoped>
.m15-export-controls{display:flex;gap:var(--control-gap);align-items:center;flex-wrap:wrap}.m15-export-controls label{display:flex;align-items:center;gap:var(--control-gap)}select{font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--border);border-radius:5px;padding:7px;max-width:100%}
</style>
