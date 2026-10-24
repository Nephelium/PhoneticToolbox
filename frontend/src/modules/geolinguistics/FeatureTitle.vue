<script setup lang="ts">
import {computed} from 'vue';
const props=defineProps<{text:string}>();
const parts=computed(()=>props.text.split(/(\[[^\]]+\])/u).filter(Boolean));
</script>
<template><span><span v-for="(part,index) in parts" :key="index" :class="{ipa:part.startsWith('[')}">{{part}}</span></span></template>
<style scoped>
.ipa{font-family:var(--font-ipa);font-size:inherit;font-weight:inherit}
</style>
