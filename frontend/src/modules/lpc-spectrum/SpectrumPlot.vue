<script setup lang="ts">
import {tr} from '../../i18n/index.ts';
import {computed,ref,watch} from 'vue';import type {LpcResult} from './state.ts';
import ScientificPlot from '../../components/ScientificPlot.vue';
import {referenceTrace} from './display.ts';
const props=defineProps<{result:LpcResult;y:[number,number];dynamic:boolean;showSpectrum?:boolean}>();
const emit=defineEmits<{'axis-mode':[dynamic:boolean]}>();
const zoom=ref(1),offset=ref(0);
watch(()=>props.result,()=>{zoom.value=1;offset.value=0;});
const maximum=computed(()=>props.result.config.freq_max_hz),length=computed(()=>maximum.value/zoom.value),left=computed(()=>Math.min(offset.value,maximum.value-length.value));
const trace=computed(()=>{const fft=props.showSpectrum?referenceTrace(props.result):undefined;return [...(fft?[{label:tr('modules.lpc-spectrum.fft_spectrum_peak_aligned'),times:fft.frequencies_hz,values:fft.magnitude_db,color:'var(--muted)',opacity:.65,lineWidth:.8}]:[]),{label:tr('modules.lpc-spectrum.lpc_spectral_envelope'),times:props.result.spectrum.frequencies_hz,values:props.result.spectrum.magnitude_db,color:'var(--wave)',lineWidth:1.5}];});
function scale(value:number){const center=left.value+length.value/2;zoom.value=Math.max(1,Math.min(64,value));offset.value=Math.max(0,Math.min(center-length.value/2,maximum.value-length.value));}
function pan(delta:number){offset.value=Math.max(0,Math.min(left.value-delta,maximum.value-length.value));}
function reset(){zoom.value=1;offset.value=0;}
</script>
<template><div class="lpc-spectrum" @dblclick="reset" @keydown.home.prevent="reset">
<p class="lpc-label ipa-text" :title="result.label">{{result.label||'LPC'}}</p>
<ScientificPlot fill interactive :title="tr('modules.lpc-spectrum.lpc_spectrum')" :height="300" :x="[left,left+length]" :y="y" :unit="tr('modules.lpc-spectrum.amplitude_db')" x-unit="Hz" :traces="trace" @zoom="scale(zoom/$event)" @pan="pan"/>
<div class="actions"><button :aria-label="tr('modules.lpc-spectrum.zoom_out_lpc_spectrum')" :disabled="zoom<=1" @click="scale(zoom/2)">−</button><span class="mono">{{Number(zoom.toFixed(2))}}×</span><button :aria-label="tr('modules.lpc-spectrum.zoom_in_lpc_spectrum')" :disabled="zoom>=64" @click="scale(zoom*2)">+</button><button @click="reset">{{tr('modules.lpc-spectrum.fit_frequency_range')}}</button><button :aria-pressed="dynamic" @click="emit('axis-mode',true)">{{tr('modules.lpc-spectrum.dynamic_vertical_axis')}}</button><button :aria-pressed="!dynamic" @click="emit('axis-mode',false)">{{tr('modules.lpc-spectrum.fixed_vertical_axis')}}</button><small>{{tr('modules.lpc-spectrum.ctrl_wheel_to_zoom_drag_or_arrow_keys_to')}}</small></div>
</div></template>
<style scoped>
.lpc-spectrum{min-width:0;display:flex;flex-direction:column;height:max(340px,calc(58dvh / var(--page-scale,1)))}.lpc-spectrum :deep(.scientific-plot){flex:1;min-height:0}.lpc-label{font-family:var(--font-figure-ipa);font-size:var(--figure-size,12px);text-align:center;margin:4px 0;overflow-wrap:anywhere}.actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:8px}.actions small{color:var(--muted)}
.actions [aria-pressed=true]{color:var(--accent);background:var(--selection)}
</style>
