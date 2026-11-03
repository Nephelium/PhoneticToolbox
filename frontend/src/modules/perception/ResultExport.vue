<script setup lang="ts">
import {tr} from '../../i18n/index.ts';
defineProps<{format:'json'|'xlsx'|'csv';busy:boolean;canConfirm:boolean;confirmVisible:boolean}>();
defineEmits<{'update:format':[value:'json'|'xlsx'|'csv'];export:[];confirm:[]}>();
</script>
<template>
 <div class="m15-export-controls">
  <label>{{tr('modules.perception.text.0c309e9d7731')}}<select :aria-label="tr('modules.perception.text.0c309e9d7731')" :value="format" :disabled="busy" @change="$emit('update:format',($event.target as HTMLSelectElement).value as 'json'|'xlsx'|'csv')"><option value="json">{{tr('modules.perception.text.615b93735061')}}</option><option value="xlsx">{{tr('modules.perception.text.cef43aab6716')}}</option><option value="csv">{{tr('modules.perception.text.75f0f7b29f81')}}</option></select></label>
  <button class="primary" :disabled="busy" @click="$emit('export')">{{tr('modules.perception.text.1a418b9270f7')}}</button>
  <button v-if="confirmVisible" :disabled="busy||!canConfirm" @click="$emit('confirm')">{{tr('modules.perception.text.eaba263b76aa')}}</button>
 </div>
</template>
<style scoped>
.m15-export-controls{display:flex;gap:var(--control-gap);align-items:center;flex-wrap:wrap}.m15-export-controls label{display:flex;align-items:center;gap:var(--control-gap)}select{font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--border);border-radius:5px;padding:7px;max-width:100%}
</style>
