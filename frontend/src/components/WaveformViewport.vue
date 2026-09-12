<script setup lang="ts">
import { computed,ref,onMounted,onUnmounted } from 'vue';import type { Workspace } from '../state/workspace.ts';import { envelope,selection } from '../platform/wav.ts';
import SpectrogramViewport from './SpectrogramViewport.vue';import type {Spectrogram,SpectrogramView} from '../platform/research.ts';
import {playback} from '../state/audio.ts';
const props=defineProps<{state:Workspace;spectrogramLoader?:(view:SpectrogramView)=>Promise<Spectrogram>}>();let anchor:number|null=null;
const viewport=ref<HTMLElement>(),width=ref(800);let observer:ResizeObserver;
onMounted(()=>{observer=new ResizeObserver(entries=>{width.value=Math.max(100,Math.min(1600,Math.round(entries[0].contentRect.width)));});if(viewport.value)observer.observe(viewport.value);});
onUnmounted(()=>observer?.disconnect());
const duration=computed(()=>props.state.asset?.duration||0);const windowLength=computed(()=>duration.value/props.state.zoom);
const left=computed(()=>Math.min(props.state.offset,Math.max(0,duration.value-windowLength.value)));
const indices=computed(()=>{const a=props.state.asset;if(!a)return [];const selected=Math.min(props.state.channel,a.channels.length-1);return props.state.showBoth&&a.channels.length>1?[0,selected===0?1:selected]:[selected];});
const tracks=computed(()=>indices.value.map(index=>({index,points:envelope(props.state.asset!.channels[index],left.value*props.state.asset!.sampleRate,(left.value+windowLength.value)*props.state.asset!.sampleRate,width.value,props.state.asset!.peaks?.[index])})));
function path(points:[number,number][]) {return points.map(([lo,hi],i)=>`M${i/Math.max(1,points.length-1)*1000},${45-hi*37}V${45-lo*37}`).join(' ');}
function x(t:number){return Math.max(0,Math.min(1000,(t-left.value)/windowLength.value*1000));}
function time(event:PointerEvent){const box=(event.currentTarget as Element).getBoundingClientRect();return left.value+Math.max(0,Math.min(1,(event.clientX-box.left)/box.width))*windowLength.value;}
function down(event:PointerEvent){anchor=time(event);(event.currentTarget as Element).setPointerCapture(event.pointerId);props.state.start=anchor;props.state.end=anchor;}
function move(event:PointerEvent){if(anchor!==null){[props.state.start,props.state.end]=selection(anchor,time(event),duration.value);}}
const maxZoom=computed(()=>Math.max(1,Math.min(65536,duration.value/.01)));
function zoom(value:number,fraction=0){const anchor=left.value+fraction*windowLength.value;props.state.zoom=Math.max(1,Math.min(maxZoom.value,value));props.state.offset=Math.max(0,Math.min(anchor-fraction*windowLength.value,duration.value-windowLength.value));}
function wheel(event:WheelEvent){if(!event.ctrlKey||!duration.value)return;event.preventDefault();const box=(event.currentTarget as Element).getBoundingClientRect();zoom(props.state.zoom*(event.deltaY<0?2:.5),Math.max(0,Math.min(1,(event.clientX-box.left)/box.width)));}
</script>
<template>
<div ref="viewport" class="wave-viewport">
<div class="wave-toolbar">
<span>原始波形 <small>· 振幅 / 秒</small>
</span>
<div>
<button :disabled="state.zoom===1" aria-label="缩小波形" @click="zoom(state.zoom/2)">−</button>
<span class="mono">{{Number(state.zoom.toFixed(1))}}×</span>
<button :disabled="state.zoom>=maxZoom" aria-label="放大波形" @click="zoom(state.zoom*2)">+</button>
<button @click="zoom(1);state.offset=0">适合窗口</button>
</div>
</div>
<div class="display-options"><slot name="controls"/><label v-if="state.asset&&state.asset.channels.length>1"><input v-model="state.showBoth" type="checkbox"/>显示两个声道</label><label v-if="spectrogramLoader"><input v-model="state.showSpectrogram" type="checkbox"/>显示语谱图（Praat）</label><small class="muted">绘图按像素聚合峰值，原音频不变</small></div>
<div v-for="track in tracks" :key="track.index" class="wave-track">
<div class="track-label">
<span>声道 {{track.index+1}}</span>
<small>{{track.index===state.channel?'试听声道':'原始数据'}}</small>
</div>
<svg viewBox="0 0 1000 90" preserveAspectRatio="none" role="img" :aria-label="'声道 '+(track.index+1)+' 原始波形；用下方数值控件调整选区'" @wheel="wheel" @dblclick="zoom(1);state.offset=0" @pointerdown="down" @pointermove="move" @pointerup="move($event);anchor=null" @pointercancel="anchor=null">
<path d="M0 45H1000" class="wave-baseline"/>
<rect :x="x(state.start)" y="0" :width="Math.max(0,x(state.end)-x(state.start))" height="90" class="wave-selection"/>
<path :d="path(track.points)" class="wave-line"/>
<slot name="annotations" :x="x" :start="left" :end="left+windowLength"/>
<path v-if="playback.position>=left&&playback.position<=left+windowLength" :d="`M${x(playback.position)} 0V90`" class="playback-cursor"/>
</svg>
<div class="time-axis" aria-label="波形时间轴（秒）"><span v-for="i in 5" :key="i">{{(left+(i-1)*windowLength/4).toFixed(windowLength<.1?4:3)}}{{i===5?' s':''}}</span></div>
</div>
<SpectrogramViewport v-if="state.showSpectrogram&&spectrogramLoader" :loader="spectrogramLoader" :start="left" :end="left+windowLength" :channel="state.channel" :width="width"/>
<div v-if="state.showSpectrogram&&spectrogramLoader" class="time-axis" aria-label="语谱图时间轴（秒）">
<span v-for="i in 5" :key="i">{{(left+(i-1)*windowLength/4).toFixed(3)}}</span>
</div>
<slot name="timeline" :start="left" :end="left+windowLength"/>
<label v-if="state.zoom>1" class="pan-label">时间窗起点 <input v-model.number="state.offset" type="range" min="0" :max="duration-windowLength" :step="1/(state.asset?.sampleRate||1)" aria-label="平移波形时间窗"/>
</label>
<p class="hint">拖动选区 · Ctrl＋滚轮缩放 · 双击显示全长；下方可输入精确时间。</p>
</div>
</template>
