<script setup lang="ts">
import {vTimePrecision} from '../../design/time-precision.ts';
import {computed,ref,watch} from 'vue';
import ModalDialog from '../../components/ModalDialog.vue';
import ScientificPlot from '../../components/ScientificPlot.vue';
import {checkedOffsetMs,normalizedParameter,parameterKeys,type AlignmentWaveform} from './alignment.ts';
import type {LipRow} from './port.ts';
const props=defineProps<{rows:LipRow[];waveform:AlignmentWaveform|null;offset:number;loading:boolean;error:string}>();
const emit=defineEmits<{close:[];apply:[number];preview:[number]}>();
const value=ref(Math.round(props.offset*1000)),enabled=ref(['open']),validation=ref('');
const duration=computed(()=>Math.max(.1,props.waveform?.duration??0,props.rows.at(-1)?.time_s??0));
const range=ref<[number,number]>([0,duration.value]);
watch(()=>props.waveform,()=>{range.value=[0,duration.value];});
const colors=['var(--accent)','var(--success)','var(--warning)','var(--danger)'];
const traces=computed(()=>enabled.value.map((key,i)=>({label:key,...normalizedParameter(props.rows,key,(Number.isFinite(value.value)?value.value:0)/1000),color:colors[i%colors.length]})));
const audio=computed(()=>props.waveform?[{label:props.waveform.channels>1?'音频 · 声道 1':'音频',times:props.waveform.times,values:props.waveform.values,color:'var(--waveform-color,var(--wave))'}]:[]);
function change(){try{const s=checkedOffsetMs(value.value);validation.value='';emit('preview',s);}catch(e){validation.value=(e as Error).message;}}
function step(delta:number){value.value=Math.max(-2000,Math.min(2000,(Number.isFinite(value.value)?value.value:0)+delta));change();}
function keys(event:KeyboardEvent){if(event.key!=='ArrowLeft'&&event.key!=='ArrowRight')return;if((event.target as HTMLElement).matches('input:not([type=number]),select'))return;event.preventDefault();event.stopPropagation();step(event.key==='ArrowLeft'?-1:1);}
function apply(){change();if(!validation.value)emit('apply',checkedOffsetMs(value.value));}
function zoom(factor:number){const [a,b]=range.value,c=(a+b)/2,span=Math.min(duration.value,Math.max(.02,(b-a)*factor)),start=Math.max(0,Math.min(duration.value-span,c-span/2));range.value=[start,start+span];}
function pan(delta:number){const [a,b]=range.value,start=Math.max(0,Math.min(duration.value-(b-a),a-delta));range.value=[start,start+b-a];}
</script>
<template><ModalDialog title="检查偏移量" wide @close="emit('close')">
 <div class="offset-editor" tabindex="0" aria-label="偏移量编辑区" @keydown.capture="keys">
  <div class="offset-controls"><label>唇形偏移量 <input v-time-precision="'ms'" v-model.number="value" type="number" min="-2000" max="2000" step="1" aria-label="唇形偏移量 ms" @input="change"/> ms</label><button @click="step(-1)">← 1 ms</button><button @click="step(1)">1 ms →</button><button @click="value=0;change()">归零</button><button @click="zoom(.5)">放大时间轴</button><button @click="zoom(2)">缩小时间轴</button><button @click="range=[0,duration]">全段</button></div>
  <p>正值向右移动唇形，负值向左。左右方向键每次 1 ms，按住连续调整；也可拖动下方参数图。上下两图共用时间轴。</p>
  <p v-if="loading" role="status">正在读取音频波形…</p><p v-if="error||validation" role="alert">{{error||validation}}</p>
  <ScientificPlot title="音频波形" :x="range" :y="[-1,1]" unit="幅度" :traces="audio" :height="190" :axis-label-width="10" interactive wheel-zoom live-pan @zoom="zoom" @pan="pan"/>
  <div class="offset-parameters"><label v-for="key in parameterKeys" :key="key"><input v-model="enabled" type="checkbox" :value="key"/>{{key}}</label></div>
  <ScientificPlot title="唇形参数对齐" :x="range" :y="[0,1]" unit="各轨归一化" :traces="traces" :height="200" :axis-label-width="10" interactive wheel-zoom live-pan @zoom="zoom" @pan="delta=>step(Math.round(delta*1000))"/>
  <p>仅此处按各参数范围归一化显示，原始值保留。起声和唇部动作有生理时差，曲线相似不能单独证明设备同步。</p>
 </div>
 <template #footer><button @click="emit('close')">取消</button><button class="primary" :disabled="!!validation||loading||!!error" @click="apply">应用偏移量</button></template>
</ModalDialog></template>
<style scoped>
.offset-editor{outline:none}.offset-controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.offset-controls label,.offset-parameters label{display:inline-flex;gap:6px;align-items:center}.offset-controls input{width:100px}.offset-editor p{color:var(--muted);font-size:var(--support-size);margin:8px 0}.offset-parameters{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:8px 0}.offset-editor :deep(.scientific-plot){width:100%}
</style>
