<script setup lang="ts">
// Layout only: the page owns data, jobs, dirty state, playback and visibility.
defineProps<{label:string;fit?:boolean}>();
</script>
<template>
  <section class="module-frame" :class="{'module-fit':fit}" :aria-label="label">
    <slot name="toolbar"/>
    <slot name="status"/>
    <slot/>
  </section>
</template>
<style scoped>
.module-frame{display:flex;flex-direction:column;min-width:0;gap:var(--module-gap);padding:4px var(--module-padding) 8px;font-family:var(--font);font-size:var(--control-size);container:module / inline-size}
.module-frame>:deep(*){min-width:0}
.module-fit{height:100%;min-height:0;overflow:auto;overscroll-behavior:contain}
</style>
