<script setup lang="ts">
defineProps<{label:string;title?:string}>();
</script>
<template>
  <section class="module-section" :aria-label="label">
    <div v-if="title||$slots.actions" class="module-section-heading">
      <h2 v-if="title">{{title}}</h2><div v-if="$slots.actions" class="module-section-actions"><slot name="actions"/></div>
    </div>
    <slot/>
  </section>
</template>
<style scoped>
.module-section{min-width:0;padding:var(--module-padding);border:1px solid var(--border);border-radius:var(--radius);background:var(--panel)}
.module-section-heading,.module-section-actions{display:flex;align-items:center;flex-wrap:wrap;gap:var(--control-gap)}
.module-section-heading{justify-content:space-between;margin-bottom:var(--control-gap)}
.module-section-heading h2{font-size:var(--control-size);font-weight:600}
</style>
