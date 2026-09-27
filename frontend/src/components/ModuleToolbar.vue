<script setup lang="ts">
withDefaults(defineProps<{label?:string}>(),{label:'模块操作'});
</script>
<template>
  <div class="module-toolbar" role="group" :aria-label="label">
    <div v-if="$slots.default" class="module-toolbar-primary"><slot/></div>
    <div v-if="$slots.actions" class="module-toolbar-actions" role="group" aria-label="当前模块工具"><slot name="actions"/></div>
  </div>
</template>
<style scoped>
/* Native Tab order: do not claim ARIA toolbar keyboard behavior we do not implement. */
.module-toolbar{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:var(--control-gap);flex:none;padding-bottom:var(--control-gap);border-bottom:1px solid var(--border)}
.module-toolbar-primary,.module-toolbar-actions{display:flex;align-items:center;flex-wrap:wrap;gap:var(--control-gap);min-width:0;max-width:100%}
.module-toolbar-actions{margin-left:auto}
.module-toolbar :deep(button){font-size:var(--control-size);min-height:var(--control-height);max-width:100%;white-space:normal;overflow-wrap:anywhere}
</style>
