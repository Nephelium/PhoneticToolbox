<script setup lang="ts">
withDefaults(defineProps<{kind?:'empty'|'loading'|'error'|'info';message:string}>(),{kind:'info'});
</script>
<template>
  <div class="module-status" :class="'module-status-'+kind" :role="kind==='error'?'alert':kind==='empty'?undefined:'status'" :aria-busy="kind==='loading'?true:undefined">
    <p>{{message}}</p><slot/>
  </div>
</template>
<style scoped>
.module-status{min-width:0;padding:var(--control-gap) var(--module-padding);border-radius:var(--radius);background:var(--app);font-size:var(--support-size);overflow-wrap:anywhere}
.module-status-error{color:var(--danger);border:1px solid var(--danger)}
.module-status-empty{padding:24px var(--module-padding);text-align:center;color:var(--muted)}
.module-status-loading{color:var(--muted)}
</style>
