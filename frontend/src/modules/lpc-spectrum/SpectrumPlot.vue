<script setup lang="ts">
import {computed,ref,watch} from 'vue';import type {LpcResult} from './state.ts';
import ScientificPlot from '../../components/ScientificPlot.vue';
const props=defineProps<{result:LpcResult}>();
const zoom=ref(1),offset=ref(0);
watch(()=>props.result,()=>{zoom.value=1;offset.value=0;});
const maximum=computed(()=>props.result.config.freq_max_hz),length=computed(()=>maximum.value/zoom.value),left=computed(()=>Math.min(offset.value,maximum.value-length.value));
const trace=computed(()=>[{label:'LPC 谱包络',times:props.result.spectrum.frequencies_hz,values:props.result.spectrum.magnitude_db,color:'var(--wave)'}]);
function scale(value:number){const center=left.value+length.value/2;zoom.value=Math.max(1,Math.min(64,value));offset.value=Math.max(0,Math.min(center-length.value/2,maximum.value-length.value));}
function pan(delta:number){offset.value=Math.max(0,Math.min(left.value-delta,maximum.value-length.value));}
function reset(){zoom.value=1;offset.value=0;}
</script>
<template><div class="lpc-spectrum" @dblclick="reset" @keydown.home.prevent="reset">
<p class="lpc-label ipa-text" :title="result.label">{{result.label||'LPC'}}</p>
<ScientificPlot fill interactive title="LPC 频谱" :height="300" :x="[left,left+length]" :y="[result.spectrum.amp_min_db,result.spectrum.amp_max_db]" unit="幅度 dB" x-unit="Hz" :traces="trace" @zoom="scale(zoom/$event)" @pan="pan"/>
<div class="actions"><button aria-label="缩小 LPC 频谱" :disabled="zoom<=1" @click="scale(zoom/2)">−</button><span class="mono">{{Number(zoom.toFixed(2))}}×</span><button aria-label="放大 LPC 频谱" :disabled="zoom>=64" @click="scale(zoom*2)">+</button><button @click="reset">适合频率范围</button><small>Ctrl＋滚轮缩放 · 拖动或方向键平移 · 不改变分析选区</small></div>
</div></template>
<style scoped>
.lpc-spectrum{min-width:0;display:flex;flex-direction:column;height:max(340px,calc(100dvh - 235px))}.lpc-spectrum :deep(.scientific-plot){flex:1;min-height:0}.lpc-label{font-family:var(--font-figure-ipa);font-size:var(--figure-size,12px);text-align:center;margin:4px 0;overflow-wrap:anywhere}.actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:8px}.actions small{color:var(--muted)}
</style>
