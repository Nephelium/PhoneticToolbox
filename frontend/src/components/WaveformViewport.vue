<script setup lang="ts">
import { computed,ref,onMounted,onUnmounted } from 'vue';import type { Workspace } from '../state/workspace.ts';import { envelope,selection,positionSelection } from '../platform/wav.ts';
import SpectrogramViewport from './SpectrogramViewport.vue';import type {Spectrogram,SpectrogramView} from '../platform/research.ts';
import {playback,isCurrentAudio} from '../state/audio.ts';import {amplitudeLimit,amplitudeLabel,sampleWavePath} from '../platform/waveform.ts';
const props=withDefaults(defineProps<{state:Workspace;spectrogramLoader?:(view:SpectrogramView)=>Promise<Spectrogram>;spectrogramSelectable?:boolean;maxWindowSeconds?:number;compactOverview?:boolean;overviewTop?:boolean;hideOverviewControls?:boolean;autoAmplitude?:boolean;normalizeDisplay?:boolean;clickMovesSelection?:boolean;selectionRequiresShift?:boolean;doubleClickAction?:'zoom'|'annotation';shiftWheelPan?:boolean;continuousDetail?:boolean;trackHeight?:number;hideTimeAxis?:boolean;timelineDuration?:number;pixelAnnotations?:boolean;timeAxisInsets?:{left:number;right:number}}>(),{autoAmplitude:true,continuousDetail:true});let anchor:number|null=null,anchorX=0,selectionLength=0,dragged=false,boundaryDrag=false;
const emit=defineEmits<{'selection-end':[start:number,end:number];'time-double-click':[time:number,ctrl:boolean];'boundary-start':[event:PointerEvent,time:number,tolerance:number];'boundary-move':[event:PointerEvent,time:number];'boundary-end':[]}>();
const viewport=ref<HTMLElement>(),width=ref(800),annotationSize=ref({width:800,height:130});let observer:ResizeObserver,resizeFrame=0;
const pendingSizes=new Map<Element,DOMRectReadOnly>(),observedTracks=new Set<Element>();
onMounted(()=>{
 observer=new ResizeObserver(entries=>{
  for(const entry of entries)pendingSizes.set(entry.target,entry.contentRect);
  if(resizeFrame)return;
  resizeFrame=requestAnimationFrame(()=>{
   resizeFrame=0;
   for(const [target,size] of pendingSizes){
    if(target===viewport.value){
     width.value=Math.max(100,Math.min(4096,Math.round(size.width)));
     if(props.pixelAnnotations){
      const tracks=new Set<Element>(viewport.value?.querySelectorAll('.wave-track>svg'));
      for(const old of observedTracks)if(!tracks.has(old)){observer.unobserve(old);observedTracks.delete(old);}
      for(const el of tracks)if(!observedTracks.has(el)){observedTracks.add(el);observer.observe(el);}
     }
    }else if(size.width&&size.height)annotationSize.value={width:size.width,height:size.height};
   }
   pendingSizes.clear();
  });
 });
 if(viewport.value)observer.observe(viewport.value);
});
onUnmounted(()=>{observer?.disconnect();cancelAnimationFrame(resizeFrame);});
const waveformBox=computed(()=>props.pixelAnnotations?annotationSize.value:{width:1000,height:90});
const insetStyle=computed(()=>props.timeAxisInsets?{'--wave-axis-width':props.timeAxisInsets.left+'px','--wave-right-width':props.timeAxisInsets.right+'px'}:undefined);
const duration=computed(()=>props.timelineDuration??props.state.asset?.duration??0);const windowLength=computed(()=>duration.value/props.state.zoom);
const minZoom=computed(()=>Math.max(1,duration.value/(props.maxWindowSeconds??Infinity)));
const left=computed(()=>Math.min(props.state.offset,Math.max(0,duration.value-windowLength.value)));
const indices=computed(()=>{const a=props.state.asset;if(!a)return [];const selected=Math.min(props.state.channel,a.channels.length-1);return props.state.showBoth&&a.channels.length>1?[0,selected===0?1:selected]:[selected];});
const tracks=computed(()=>indices.value.map(index=>{const asset=props.state.asset!,start=left.value*asset.sampleRate,end=(left.value+windowLength.value)*asset.sampleRate,points=envelope(asset.channels[index],start,end,width.value,asset.peaks?.[index]),limit=props.normalizeDisplay?Math.max(1e-20,points.reduce((peak,[lo,hi])=>Math.max(peak,Math.abs(lo),Math.abs(hi)),0))*37/40.5:props.autoAmplitude?amplitudeLimit(points):1;return {index,points,limit,span:Math.max(0,Math.min(end,asset.channels[index].length)-start)/(end-start),detail:props.continuousDetail?sampleWavePath(asset.channels[index],start,end,limit,width.value):null};}));
function path(points:[number,number][],limit:number) {return points.map(([lo,hi],i)=>`M${i/Math.max(1,points.length-1)*1000},${45-hi/limit*37}V${45-lo/limit*37}`).join(' ');}
function x(t:number){return Math.max(0,Math.min(1000,(t-left.value)/windowLength.value*1000));}
function time(event:MouseEvent){const box=(event.currentTarget as Element).getBoundingClientRect();return left.value+Math.max(0,Math.min(1,(event.clientX-box.left)/box.width))*windowLength.value;}
function double(event:MouseEvent){if(props.doubleClickAction==='annotation'){emit('time-double-click',time(event),event.ctrlKey);return;}zoom(1);props.state.offset=0;}
function down(event:PointerEvent){
  if(event.button!==0||(props.selectionRequiresShift&&!event.shiftKey))return;
  emit('boundary-start',event,time(event),windowLength.value*6/(event.currentTarget as Element).getBoundingClientRect().width);
  if(event.defaultPrevented){boundaryDrag=true;(event.currentTarget as Element).setPointerCapture(event.pointerId);return;}
  anchor=time(event);anchorX=event.clientX;dragged=false;selectionLength=props.state.end-props.state.start;
  (event.currentTarget as Element).setPointerCapture(event.pointerId);
  if(!props.clickMovesSelection){props.state.start=anchor;props.state.end=anchor;}
}
function move(event:PointerEvent){if(boundaryDrag){emit('boundary-move',event,time(event));return;}if(anchor!==null){
  if(Math.abs(event.clientX-anchorX)>=3)dragged=true;
  if(!props.clickMovesSelection||dragged)[props.state.start,props.state.end]=selection(anchor,time(event),props.state.asset?.duration??0);
}}
function up(event:PointerEvent){
  if(boundaryDrag){emit('boundary-move',event,time(event));boundaryDrag=false;emit('boundary-end');return;}
  if(anchor===null)return;move(event);
  if(props.clickMovesSelection&&!dragged)[props.state.start,props.state.end]=positionSelection(anchor,selectionLength>0?selectionLength:Math.min(.5,duration.value),props.state.asset?.duration??0);
  anchor=null;emit('selection-end',props.state.start,props.state.end);
}
function cancel(){anchor=null;if(boundaryDrag){boundaryDrag=false;emit('boundary-end');}}
function selectSpectrogram(start:number,end:number){props.state.start=start;props.state.end=end;emit('selection-end',start,end);}
const maxZoom=computed(()=>Math.max(1,Math.min(65536,duration.value/.01)));
function zoom(value:number,fraction=0){const anchor=left.value+fraction*windowLength.value;props.state.zoom=Math.max(minZoom.value,Math.min(maxZoom.value,value));props.state.offset=Math.max(0,Math.min(anchor-fraction*windowLength.value,duration.value-windowLength.value));}
function wheel(event:WheelEvent){
 if(!duration.value||(!event.ctrlKey&&!(props.shiftWheelPan&&event.shiftKey)))return;
 event.preventDefault();
 if(event.ctrlKey){const box=(event.currentTarget as Element).getBoundingClientRect();zoom(props.state.zoom*(event.deltaY<0?2:.5),Math.max(0,Math.min(1,(event.clientX-box.left)/box.width)));}
 else props.state.offset=Math.max(0,Math.min(duration.value-windowLength.value,left.value+(event.deltaY||event.deltaX)*windowLength.value*.0015));
}
</script>
<template>
<div ref="viewport" class="wave-viewport" :style="insetStyle" :class="{'compact-overview':compactOverview,'top-overview':overviewTop,'amplitude-scaled':autoAmplitude}">
<div v-if="!compactOverview" class="wave-toolbar">
<span>原始波形 <small>· 振幅 / 秒</small>
</span>
<div>
<button :disabled="state.zoom<=minZoom" aria-label="缩小波形" @click="zoom(state.zoom/2)">−</button>
<span class="mono">{{Number(state.zoom.toFixed(1))}}×</span>
<button :disabled="state.zoom>=maxZoom" aria-label="放大波形" @click="zoom(state.zoom*2)">+</button>
<button @click="zoom(1);state.offset=0">适合窗口</button>
</div>
</div>
<div v-if="!compactOverview" class="display-options"><slot name="controls"/><label v-if="state.asset&&state.asset.channels.length>1"><input v-model="state.showBoth" type="checkbox"/>显示两个声道</label><label v-if="spectrogramLoader"><input v-model="state.showSpectrogram" type="checkbox"/>显示语谱图（Praat）</label><small class="muted">振幅随可见窗调整 · 细节显示原始采样点</small></div>
<div v-for="track in tracks" :key="track.index" class="wave-track">
<div v-if="!compactOverview||state.showBoth" class="track-label">
<span>声道 {{track.index+1}}</span>
<small>{{track.index===state.channel?'试听声道':'原始数据'}}</small>
</div>
<div v-if="autoAmplitude" class="amplitude-axis" aria-label="波形振幅刻度" :data-limit="track.limit"><span v-for="(value,i) in [track.limit,0,-track.limit]" :key="i" :style="{top:(8.888889+i*41.111111)+'%'}">{{amplitudeLabel(value)}}</span></div>
<svg :viewBox="'0 0 '+waveformBox.width+' '+waveformBox.height" preserveAspectRatio="none" role="img" :tabindex="doubleClickAction==='annotation'?0:undefined" :data-start="left" :data-end="left+windowLength" :style="trackHeight?{height:trackHeight+'px'}:undefined" :aria-label="'声道 '+(track.index+1)+' 原始波形；拖动选择时间范围'" @wheel="wheel" @dblclick="double" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="cancel">
<g :transform="pixelAnnotations?'scale('+waveformBox.width/1000+' '+waveformBox.height/90+')':undefined">
<path d="M0 45H1000" class="wave-baseline"/>
<rect :x="x(state.start)" y="0" :width="Math.max(0,x(state.end)-x(state.start))" height="90" class="wave-selection"/>
<path :d="track.detail??path(track.points,track.limit)" :transform="track.detail?undefined:`scale(${track.span},1)`" :data-mode="track.detail?'samples':'envelope'" class="wave-line"/>
<path v-if="state.asset&&isCurrentAudio(state.asset,state.channel)&&playback.position>=left&&playback.position<=left+windowLength" :d="`M${x(playback.position)} 0V90`" class="playback-cursor"/>
</g>
<slot name="annotations" :x="(t:number)=>x(t)*waveformBox.width/1000" :height="waveformBox.height" :start="left" :end="left+windowLength"/>
</svg>
<div v-if="!hideTimeAxis" class="time-axis" aria-label="波形时间轴（秒）"><span v-for="i in 5" :key="i">{{(left+(i-1)*windowLength/4).toFixed(windowLength<.1?4:3)}}{{i===5?' s':''}}</span></div>
</div>
<div v-if="state.showSpectrogram&&spectrogramLoader" class="wave-aligned-layer">
<SpectrogramViewport :loader="spectrogramLoader" :start="left" :end="left+windowLength" :channel="state.channel" :width="width" :selectable="spectrogramSelectable" :selection-start="state.start" :selection-end="state.end" @selection-end="selectSpectrogram" @view-wheel="wheel" @view-double-click="double"/>
<div class="time-axis spectrogram-time-axis" aria-label="语谱图时间轴（秒）">
<span v-for="i in 5" :key="i">{{(left+(i-1)*windowLength/4).toFixed(windowLength<.1?4:3)}}{{i===5?' s':''}}</span>
</div>
</div>
<div v-if="$slots.timeline" class="wave-aligned-layer"><slot name="timeline" :start="left" :end="left+windowLength"/></div>
<div v-if="compactOverview&&!hideOverviewControls" class="wave-toolbar overview-controls">
<strong>音频总览</strong>
<label v-if="state.asset&&state.asset.channels.length>1"><input v-model="state.showBoth" type="checkbox"/>显示两个声道</label>
<div>
<button :disabled="state.zoom<=minZoom" aria-label="缩小波形" @click="zoom(state.zoom/2)">−</button>
<span class="mono">{{Number(state.zoom.toFixed(1))}}×</span>
<button :disabled="state.zoom>=maxZoom" aria-label="放大波形" @click="zoom(state.zoom*2)">+</button>
<button @click="zoom(1);state.offset=0">适合窗口</button>
</div>
<small><template v-if="clickMovesSelection">单击定位 · </template>{{selectionRequiresShift?'Shift＋拖动选区':'拖动选区'}} · Ctrl＋滚轮缩放<span v-if="shiftWheelPan"> · Shift＋滚轮平移</span><span v-if="maxWindowSeconds"> · 最多 {{maxWindowSeconds}} 秒视窗</span></small>
</div>
<label v-if="state.zoom>1" class="pan-label">时间窗起点 <input v-model.number="state.offset" type="range" min="0" :max="duration-windowLength" :step="1/(state.asset?.sampleRate||1)" aria-label="平移波形时间窗"/>
</label>
<p v-if="!compactOverview" class="hint">{{selectionRequiresShift?'Shift＋拖动选区':'拖动选区'}} · Ctrl＋滚轮缩放 · 双击{{maxWindowSeconds?'恢复总览':'显示全长'}}；时间控件可输入精确选区。</p>
</div>
</template>
<style scoped>
.top-overview{display:flex;flex-direction:column}.top-overview .overview-controls{order:-2;margin:0 0 6px}.top-overview .pan-label{order:-1;margin:0 0 8px}
.amplitude-scaled .wave-track{display:grid;grid-template-columns:var(--wave-axis-width,64px) minmax(0,1fr);grid-template-rows:auto auto auto}.amplitude-scaled .track-label{grid-area:1/2}.amplitude-scaled .wave-track svg{grid-area:2/2}.amplitude-scaled .time-axis{grid-area:3/2}.amplitude-axis{grid-area:2/1;position:relative;font-family:var(--font-figure);font-size:var(--figure-size);color:var(--muted);pointer-events:none}.amplitude-axis span{position:absolute;right:8px;transform:translateY(-50%);white-space:nowrap}
.compact-overview .overview-controls{justify-content:flex-start;flex-wrap:wrap;gap:8px 16px;margin:4px 0 0}
.overview-controls label{display:flex;align-items:center;gap:5px}
.compact-overview .wave-track svg{height:110px}
.wave-aligned-layer{min-width:0}
.amplitude-scaled>.wave-aligned-layer{margin-left:var(--wave-axis-width,64px)}
.amplitude-scaled .wave-track,.amplitude-scaled>.wave-aligned-layer{margin-right:var(--wave-right-width,0px)}
.wave-track svg{overflow:hidden}
</style>
