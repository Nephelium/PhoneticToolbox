<script setup lang="ts">
import { computed } from 'vue';import type { Workspace } from '../state/workspace.ts';import { envelope,selection } from '../platform/wav.ts';
const props=defineProps<{state:Workspace}>();let anchor:number|null=null;
const duration=computed(()=>props.state.asset?.duration||0);const windowLength=computed(()=>duration.value/props.state.zoom);
const left=computed(()=>Math.min(props.state.offset,Math.max(0,duration.value-windowLength.value)));
const tracks=computed(()=>props.state.asset?.channels.map(samples=>envelope(samples,left.value*props.state.asset!.sampleRate,(left.value+windowLength.value)*props.state.asset!.sampleRate))||[]);
function path(points:[number,number][]) {return points.map(([lo,hi],i)=>`M${i/Math.max(1,points.length-1)*1000},${45-hi*37}V${45-lo*37}`).join(' ');}
function x(t:number){return Math.max(0,Math.min(1000,(t-left.value)/windowLength.value*1000));}
function time(event:PointerEvent){const box=(event.currentTarget as Element).getBoundingClientRect();return left.value+Math.max(0,Math.min(1,(event.clientX-box.left)/box.width))*windowLength.value;}
function down(event:PointerEvent){anchor=time(event);(event.currentTarget as Element).setPointerCapture(event.pointerId);props.state.start=anchor;props.state.end=anchor;}
function move(event:PointerEvent){if(anchor!==null){[props.state.start,props.state.end]=selection(anchor,time(event),duration.value);}}
function zoom(value:number){props.state.zoom=Math.max(1,Math.min(32,value));props.state.offset=Math.min(props.state.offset,duration.value-duration.value/props.state.zoom);}
</script>
<template>
<div class="wave-toolbar">
<span>原始波形 <small>· 振幅 / 秒</small>
</span>
<div>
<button :disabled="state.zoom===1" aria-label="缩小波形" @click="zoom(state.zoom/2)">−</button>
<span class="mono">{{state.zoom}}×</span>
<button :disabled="state.zoom===32" aria-label="放大波形" @click="zoom(state.zoom*2)">+</button>
<button @click="zoom(1);state.offset=0">适合窗口</button>
</div>
</div>
<div v-for="(track,index) in tracks" :key="index" class="wave-track">
<div class="track-label">
<span>声道 {{index+1}}</span>
<small>{{index===state.channel?'试听声道':'原始数据'}}</small>
</div>
<svg viewBox="0 0 1000 90" preserveAspectRatio="none" role="img" :aria-label="'声道 '+(index+1)+' 原始波形；用下方数值控件调整选区'" @pointerdown="down" @pointermove="move" @pointerup="move($event);anchor=null" @pointercancel="anchor=null">
<path d="M0 45H1000" class="wave-baseline"/>
<rect :x="x(state.start)" y="0" :width="Math.max(0,x(state.end)-x(state.start))" height="90" class="wave-selection"/>
<path :d="path(track)" class="wave-line"/>
</svg>
</div>
<div class="time-axis">
<span v-for="i in 5" :key="i">{{(left+(i-1)*windowLength/4).toFixed(3)}}</span>
</div>
<label v-if="state.zoom>1" class="pan-label">时间窗起点 <input v-model.number="state.offset" type="range" min="0" :max="duration-windowLength" :step="1/(state.asset?.sampleRate||1)" aria-label="平移波形时间窗"/>
</label>
<p class="hint">拖动波形选择时间段，也可在下方输入精确起止时间。</p>
</template>
