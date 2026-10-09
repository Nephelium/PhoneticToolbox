<script setup lang="ts">
defineProps<{message:string}>();
</script>
<template><div class="plot-placeholder" role="img" :aria-label="message"><p>{{message}}</p></div></template>
<style scoped>
.plot-placeholder{display:grid;place-items:center;width:100%;height:var(--plot-height,260px);min-height:var(--plot-min-height,160px);flex:1;background:var(--app);border-radius:var(--radius);color:var(--muted);padding:16px;text-align:center;font-size:var(--support-size)}.plot-placeholder p{margin:0;max-width:90%}
</style>
