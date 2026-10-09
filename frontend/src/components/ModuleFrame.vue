<script setup lang="ts">
import {onUnmounted} from 'vue';
import {provideAudioSelection} from '../state/audio-selection.ts';
// The page owns scientific data and jobs; audio selection is scoped to this frame.
const selection=provideAudioSelection();onUnmounted(selection.dispose);
defineProps<{label:string;fit?:boolean;unified?:boolean}>();
</script>
<template>
  <section class="module-frame" :class="{'module-fit':fit,'module-unified':unified}" :aria-label="label">
    <slot name="toolbar"/>
    <slot name="status"/>
    <slot/>
  </section>
</template>
<style scoped>
.module-frame{display:flex;flex-direction:column;min-width:0;gap:var(--module-gap);padding:4px var(--module-padding) 8px;font-family:var(--font);font-size:var(--control-size);container:module / inline-size}
.module-frame>:deep(*){min-width:0}
.module-fit{height:100%;min-height:0;overflow:auto;overscroll-behavior:contain}
.module-unified{--workbench-side-width:300px;--workbench-pane-padding:8px}
.module-unified :deep(.module-toolbar){min-height:calc(var(--control-height) + var(--control-gap) + 1px)}
.module-unified :deep(.module-toolbar button){min-width:88px}
.module-unified :deep(.module-toolbar-primary),.module-unified :deep(.module-toolbar-actions){row-gap:var(--control-gap)}
.module-unified :deep(.module-toolbar input[type=file]){max-width:260px}
.module-unified :deep(.module-section){padding:10px}
</style>
