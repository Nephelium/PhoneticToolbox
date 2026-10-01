<script setup lang="ts">
import {vResizablePanels} from '../layout/resizablePanels.ts';
defineProps<{layoutKey:string}>();
const emit=defineEmits<{'files-keydown':[event:KeyboardEvent]}>();
</script>
<template><div v-resizable-panels="{key:layoutKey,center:'.signal-panel',centerMin:280,panels:[{selector:'.file-panel',side:'left',label:'音频列表',initial:220,min:180,max:520},{selector:'.parameter-summary',side:'right',label:'参数与结果',initial:250,min:200,max:560}]}" class="workbench-grid workbench-columns">
<aside class="file-panel workbench-pane" tabindex="0" aria-label="音频列表滚动区" @keydown="emit('files-keydown',$event)"><div class="pane-content"><slot name="files"/></div></aside>
<div class="signal-panel workbench-pane" tabindex="0" aria-label="音频与标注滚动区"><div class="pane-content"><slot/></div></div>
<aside class="parameter-summary workbench-pane" tabindex="0" aria-label="参数与结果滚动区"><div class="pane-content"><slot name="settings"/></div></aside>
</div></template>

<style scoped>
@container module (max-width:720px){.workbench-columns{display:block;flex:none;overflow:visible}.workbench-columns>.workbench-pane{width:100%;overflow:visible}.workbench-columns :deep(.m01-file-list){max-height:180px;overflow:auto}}
</style>
