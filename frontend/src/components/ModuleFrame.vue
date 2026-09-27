<script setup lang="ts">
// Layout only: the page owns data, jobs, dirty state, playback and visibility.
defineProps<{label:string}>();
</script>
<template>
  <section class="module-frame" :aria-label="label">
    <slot name="toolbar"/>
    <slot name="status"/>
    <slot/>
  </section>
</template>
<style scoped>
.module-frame{display:flex;flex-direction:column;min-width:0;gap:var(--module-gap);padding:var(--module-padding);font-family:var(--font);font-size:var(--control-size);container:module / inline-size}
.module-frame>:deep(*){min-width:0}
</style>
