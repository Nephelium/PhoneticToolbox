<script setup lang="ts">
import {uiErrorMessage} from '../module-localization.ts';
import {tr,messageRef} from '../../i18n/index.ts';
import {vTimePrecision} from '../../design/time-precision.ts';
import {computed,ref,watch} from 'vue';
import ModalDialog from '../../components/ModalDialog.vue';
import ScientificPlot from '../../components/ScientificPlot.vue';
import {checkedOffsetMs,normalizedParameter,parameterKeys,type AlignmentWaveform} from './alignment.ts';
import type {LipRow} from './port.ts';
const props=defineProps<{rows:LipRow[];waveform:AlignmentWaveform|null;offset:number;loading:boolean;error:string}>();
const emit=defineEmits<{close:[];apply:[number];preview:[number]}>();
const value=ref(Math.round(props.offset*1000)),enabled=ref(['open']),validation=messageRef();
const duration=computed(()=>Math.max(.1,props.waveform?.duration??0,props.rows.at(-1)?.time_s??0));
const range=ref<[number,number]>([0,duration.value]);
watch(()=>props.waveform,()=>{range.value=[0,duration.value];});
const colors=['var(--accent)','var(--success)','var(--warning)','var(--danger)'];
const traces=computed(()=>enabled.value.map((key,i)=>({label:key,...normalizedParameter(props.rows,key,(Number.isFinite(value.value)?value.value:0)/1000),color:colors[i%colors.length]})));
const audio=computed(()=>props.waveform?[{label:props.waveform.channels>1?tr('modules.lip-extraction.audio_channel_1'):tr('modules.lip-extraction.audio'),times:props.waveform.times,values:props.waveform.values,color:'var(--waveform-color,var(--wave))'}]:[]);
function change(){try{const s=checkedOffsetMs(value.value);validation.value='';emit('preview',s);}catch(e){validation.value=uiErrorMessage(e);}}
function step(delta:number){value.value=Math.max(-2000,Math.min(2000,(Number.isFinite(value.value)?value.value:0)+delta));change();}
function keys(event:KeyboardEvent){if(event.key!=='ArrowLeft'&&event.key!=='ArrowRight')return;if((event.target as HTMLElement).matches('input:not([type=number]),select'))return;event.preventDefault();event.stopPropagation();step(event.key==='ArrowLeft'?-1:1);}
function apply(){change();if(!validation.value)emit('apply',checkedOffsetMs(value.value));}
function zoom(factor:number){const [a,b]=range.value,c=(a+b)/2,span=Math.min(duration.value,Math.max(.02,(b-a)*factor)),start=Math.max(0,Math.min(duration.value-span,c-span/2));range.value=[start,start+span];}
function pan(delta:number){const [a,b]=range.value,start=Math.max(0,Math.min(duration.value-(b-a),a-delta));range.value=[start,start+b-a];}
</script>
<template><ModalDialog :title="tr('modules.lip-extraction.check_offset')" wide @close="emit('close')">
 <div class="offset-editor" tabindex="0" :aria-label="tr('modules.lip-extraction.offset_editing_area')" @keydown.capture="keys">
  <div class="offset-controls"><label>{{tr('modules.lip-extraction.lip_offset_225')}} <input v-time-precision="'ms'" v-model.number="value" type="number" min="-2000" max="2000" step="1" :aria-label="tr('modules.lip-extraction.lip_offset_ms')" @input="change"/> ms</label><button @click="step(-1)">← 1 ms</button><button @click="step(1)">1 ms →</button><button @click="value=0;change()">{{tr('modules.lip-extraction.zero')}}</button><button @click="zoom(.5)">{{tr('modules.lip-extraction.zoom_in_time_axis')}}</button><button @click="zoom(2)">{{tr('modules.lip-extraction.zoom_out_time_axis')}}</button><button @click="range=[0,duration]">{{tr('modules.lip-extraction.full_interval')}}</button></div>
  <p>{{tr('modules.lip-extraction.positive_values_shift_lips_right_negative_values_shift_left')}}</p>
  <p v-if="loading" role="status">{{tr('modules.lip-extraction.reading_audio_waveform')}}</p><p v-if="error||validation" role="alert">{{error||validation}}</p>
  <ScientificPlot :title="tr('modules.lip-extraction.audio_waveform')" :x="range" :y="[-1,1]" :unit="tr('modules.lip-extraction.amplitude')" :traces="audio" :height="190" :axis-label-width="10" interactive wheel-zoom live-pan @zoom="zoom" @pan="pan"/>
  <div class="offset-parameters"><label v-for="key in parameterKeys" :key="key"><input v-model="enabled" type="checkbox" :value="key"/>{{key}}</label></div>
  <ScientificPlot :title="tr('modules.lip-extraction.lip_parameter_alignment')" :x="range" :y="[0,1]" :unit="tr('modules.lip-extraction.each_track_normalized')" :traces="traces" :height="200" :axis-label-width="10" interactive wheel-zoom live-pan @zoom="zoom" @pan="delta=>step(Math.round(delta*1000))"/>
  <p>{{tr('modules.lip-extraction.only_this_view_is_normalized_by_each_parameter_s')}}</p>
 </div>
 <template #footer><button @click="emit('close')">{{tr('modules.lip-extraction.cancel')}}</button><button class="primary" :disabled="!!validation||loading||!!error" @click="apply">{{tr('modules.lip-extraction.apply_offset')}}</button></template>
</ModalDialog></template>
<style scoped>
.offset-editor{outline:none}.offset-controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.offset-controls label,.offset-parameters label{display:inline-flex;gap:6px;align-items:center}.offset-controls input{width:100px}.offset-editor p{color:var(--muted);font-size:var(--support-size);margin:8px 0}.offset-parameters{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:8px 0}.offset-editor :deep(.scientific-plot){width:100%}
</style>
