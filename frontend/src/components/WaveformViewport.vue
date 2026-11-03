<script setup lang="ts">
import {tr} from '../i18n/index.ts';
import { computed,ref,onMounted,onUnmounted } from 'vue';import type { Workspace } from '../state/workspace.ts';import { envelope,selection,positionSelection } from '../platform/wav.ts';
import SpectrogramViewport from './SpectrogramViewport.vue';import type {Spectrogram,SpectrogramView} from '../platform/research.ts';
import {useAudioSelection} from '../state/audio-selection.ts';
import {stop} from '../state/audio.ts';
import {playback,isCurrentAudio} from '../state/audio.ts';import {amplitudeLimit,amplitudeLabel,sampleWavePath,endpointClickTime} from '../platform/waveform.ts';
const props=withDefaults(defineProps<{state:Workspace;emptyMessage?:string;spectrogramLoader?:(view:SpectrogramView)=>Promise<Spectrogram>;spectrogramSelectable?:boolean;maxWindowSeconds?:number;compactOverview?:boolean;overviewTop?:boolean;hideOverviewControls?:boolean;autoAmplitude?:boolean;normalizeDisplay?:boolean;clickMovesSelection?:boolean;selectionRequiresShift?:boolean;doubleClickAction?:'zoom'|'annotation';shiftWheelPan?:boolean;continuousDetail?:boolean;trackHeight?:number;hideTimeAxis?:boolean;timelineDuration?:number;pixelAnnotations?:boolean;timeAxisInsets?:{left:number;right:number}}>(),{autoAmplitude:true,continuousDetail:true});const emptyMessage=computed(()=>props.emptyMessage??tr('components.choose_audio_to_display_its_waveform'));let anchor:number|null=null,anchorX=0,selectionLength=0,dragged=false,boundaryDrag=false;
const emit=defineEmits<{'selection-end':[start:number,end:number];'time-double-click':[time:number,ctrl:boolean];'boundary-start':[event:PointerEvent,time:number,tolerance:number];'boundary-move':[event:PointerEvent,time:number];'boundary-end':[]}>();
const selectionGroup=useAudioSelection();const unregister=selectionGroup?.register(()=>props.state);onUnmounted(()=>unregister?.());
function select(){selectionGroup?.select(props.state);stop();}
const viewport=ref<HTMLElement>(),width=ref(800),annotationSize=ref({width:800,height:130});let observer:ResizeObserver,resizeFrame=0;
const pendingSizes=new Map<Element,DOMRectReadOnly>(),observedTracks=new Set<Element>();
onMounted(()=>{
 observer=new ResizeObserver(entries=>{
  for(const entry of entries)pendingSizes.set(entry.target,entry.contentRect);
  if(resizeFrame)return;
  resizeFrame=requestAnimationFrame(()=>{
   resizeFrame=0;
   for(const [target,size] of pendingSizes){
    if(target===viewport.value&&size.width>0){
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
// Preserve the last visible geometry when a tab is hidden, and reuse its SVG
// path for cursor/selection updates rather than rebuilding the sample envelope.
const tracks=computed(()=>indices.value.map(index=>{const asset=props.state.asset!,start=left.value*asset.sampleRate,end=(left.value+windowLength.value)*asset.sampleRate,points=envelope(asset.channels[index],start,end,width.value,asset.peaks?.[index]),limit=props.normalizeDisplay?Math.max(1e-20,points.reduce((peak,[lo,hi])=>Math.max(peak,Math.abs(lo),Math.abs(hi)),0))*37/40.5:props.autoAmplitude?amplitudeLimit(points):1,detail=props.continuousDetail?sampleWavePath(asset.channels[index],start,end,limit,width.value):null;return {index,points,limit,span:Math.max(0,Math.min(end,asset.channels[index].length)-start)/(end-start),detail,svgPath:detail??path(points,limit)};}));
function path(points:[number,number][],limit:number) {return points.map(([lo,hi],i)=>`M${i/Math.max(1,points.length-1)*1000},${45-hi/limit*37}V${45-lo/limit*37}`).join(' ');}
function x(t:number){return Math.max(0,Math.min(1000,(t-left.value)/windowLength.value*1000));}
function time(event:MouseEvent){const box=(event.currentTarget as Element).getBoundingClientRect();return left.value+Math.max(0,Math.min(1,(event.clientX-box.left)/box.width))*windowLength.value;}
function double(event:MouseEvent){if(props.doubleClickAction==='annotation'){emit('time-double-click',endpointClickTime(time(event),left.value,windowLength.value,duration.value,(event.currentTarget as Element).getBoundingClientRect().width),event.ctrlKey);return;}zoom(1);props.state.offset=0;}
function down(event:PointerEvent){
  if(event.button!==0||(props.selectionRequiresShift&&!event.shiftKey))return;
  emit('boundary-start',event,time(event),windowLength.value*6/(event.currentTarget as Element).getBoundingClientRect().width);
  if(event.defaultPrevented){boundaryDrag=true;(event.currentTarget as Element).setPointerCapture(event.pointerId);return;}
  select();(event.currentTarget as SVGElement).focus({preventScroll:true});
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
function selectSpectrogram(start:number,end:number){select();props.state.start=start;props.state.end=end;emit('selection-end',start,end);}
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
<div ref="viewport" class="wave-viewport" :data-selection-active="selectionGroup?.selected(state)" :style="insetStyle" :class="{'compact-overview':compactOverview,'top-overview':overviewTop,'amplitude-scaled':autoAmplitude}">
<div v-if="!compactOverview" class="wave-toolbar">
<span>{{tr('components.original_waveform')}} <small>{{tr('components.amplitude_seconds')}}</small>
</span>
<div>
<button :disabled="state.zoom<=minZoom" :aria-label="tr('components.zoom_out_waveform')" @click="zoom(state.zoom/2)">−</button>
<span class="mono">{{Number(state.zoom.toFixed(1))}}×</span>
<button :disabled="state.zoom>=maxZoom" :aria-label="tr('components.zoom_in_waveform')" @click="zoom(state.zoom*2)">+</button>
<button @click="zoom(1);state.offset=0">{{tr('components.fit_window')}}</button>
</div>
</div>
<div v-if="!compactOverview" class="display-options"><slot name="controls"/><label v-if="state.asset&&state.asset.channels.length>1"><input v-model="state.showBoth" type="checkbox"/>{{tr('components.show_both_channels')}}</label><label v-if="spectrogramLoader"><input v-model="state.showSpectrogram" type="checkbox"/>{{tr('components.show_spectrogram_praat')}}</label><small class="muted">{{tr('components.amplitude_follows_the_visible_window_original_samples_shown_in')}}</small></div>
<div v-if="!state.asset&&emptyMessage" class="wave-track waveform-placeholder" :aria-busy="state.loading||undefined">
<div v-if="!compactOverview" class="track-label"><span>{{tr('components.waveform')}}</span><small>{{tr('components.waiting_for_audio')}}</small></div>
<svg :viewBox="'0 0 '+waveformBox.width+' '+waveformBox.height" :style="trackHeight?{height:trackHeight+'px'}:undefined" role="img" :aria-label="emptyMessage"/>
<p class="wave-empty-message">{{emptyMessage}}</p>
<div v-if="!hideTimeAxis" class="time-axis" :aria-label="tr('components.waveform_time_axis_s')"><span v-for="i in 5" :key="i">—{{i===5?' s':''}}</span></div>
</div>
<div v-for="track in tracks" :key="track.index" class="wave-track">
<div v-if="!compactOverview||state.showBoth" class="track-label">
<span>{{tr('components.channel')}} {{track.index+1}}</span>
<small>{{track.index===state.channel?tr('components.playback_channel'):tr('components.original_data')}}</small>
</div>
<div v-if="autoAmplitude" class="amplitude-axis" :aria-label="tr('components.waveform_amplitude_scale')" :data-limit="track.limit"><span v-for="(value,i) in [track.limit,0,-track.limit]" :key="i" :style="{top:(8.888889+i*41.111111)+'%'}">{{amplitudeLabel(value)}}</span></div>
<svg :viewBox="'0 0 '+waveformBox.width+' '+waveformBox.height" preserveAspectRatio="none" role="img" tabindex="0" :data-start="left" :data-end="left+windowLength" :style="trackHeight?{height:trackHeight+'px'}:undefined" :aria-label="tr('components.channel_529')+(track.index+1)+tr('components.original_waveform_drag_to_select_a_time_range')" @wheel="wheel" @dblclick="double" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="cancel">
<g :transform="pixelAnnotations?'scale('+waveformBox.width/1000+' '+waveformBox.height/90+')':undefined">
<path d="M0 45H1000" class="wave-baseline"/>
<rect :x="x(state.start)" y="0" :width="Math.max(0,x(state.end)-x(state.start))" height="90" class="wave-selection"/>
<path :d="track.svgPath" :transform="track.detail?undefined:`scale(${track.span},1)`" :data-mode="track.detail?'samples':'envelope'" class="wave-line"/>
<path v-if="state.asset&&isCurrentAudio(state.asset,state.channel)&&playback.position>=left&&playback.position<=left+windowLength" :d="`M${x(playback.position)} 0V90`" class="playback-cursor"/>
</g>
<slot name="annotations" :x="(t:number)=>x(t)*waveformBox.width/1000" :height="waveformBox.height" :start="left" :end="left+windowLength"/>
</svg>
<div v-if="!hideTimeAxis" class="time-axis" :aria-label="tr('components.waveform_time_axis_s')"><span v-for="i in 5" :key="i">{{(left+(i-1)*windowLength/4).toFixed(windowLength<.1?4:3)}}{{i===5?' s':''}}</span></div>
</div>
<div v-if="state.showSpectrogram&&spectrogramLoader" class="wave-aligned-layer">
<SpectrogramViewport :loader="spectrogramLoader" :start="left" :end="left+windowLength" :channel="state.channel" :width="width" :selectable="spectrogramSelectable" :selection-start="state.start" :selection-end="state.end" @selection-end="selectSpectrogram" @view-wheel="wheel" @view-double-click="double"/>
<div class="time-axis spectrogram-time-axis" :aria-label="tr('components.spectrogram_time_axis_s')">
<span v-for="i in 5" :key="i">{{(left+(i-1)*windowLength/4).toFixed(windowLength<.1?4:3)}}{{i===5?' s':''}}</span>
</div>
</div>
<div v-if="$slots.timeline" class="wave-aligned-layer"><slot name="timeline" :start="left" :end="left+windowLength"/></div>
<div v-if="compactOverview&&!hideOverviewControls" class="wave-toolbar overview-controls">
<strong>{{tr('components.audio_overview')}}</strong>
<label v-if="state.asset&&state.asset.channels.length>1"><input v-model="state.showBoth" type="checkbox"/>{{tr('components.show_both_channels')}}</label>
<div>
<button :disabled="state.zoom<=minZoom" :aria-label="tr('components.zoom_out_waveform')" @click="zoom(state.zoom/2)">−</button>
<span class="mono">{{Number(state.zoom.toFixed(1))}}×</span>
<button :disabled="state.zoom>=maxZoom" :aria-label="tr('components.zoom_in_waveform')" @click="zoom(state.zoom*2)">+</button>
<button @click="zoom(1);state.offset=0">{{tr('components.fit_window')}}</button>
</div>
<small><template v-if="clickMovesSelection">{{tr('components.click_to_seek')}} </template>{{selectionRequiresShift?tr('components.shift_drag_to_select'):tr('components.drag_to_select')}} {{tr('components.ctrl_wheel_to_zoom')}}<span v-if="shiftWheelPan"> {{tr('components.shift_wheel_to_pan')}}</span><span v-if="maxWindowSeconds"> {{tr('components.up_to')}} {{maxWindowSeconds}} {{tr('components.s_window')}}</span></small>
</div>
<label v-if="state.zoom>1" class="pan-label">{{tr('components.time_window_start')}} <input v-model.number="state.offset" type="range" min="0" :max="duration-windowLength" :step="1/(state.asset?.sampleRate||1)" :aria-label="tr('components.pan_waveform_window')"/>
</label>
<p v-if="!compactOverview" class="hint">{{tr(selectionRequiresShift?(maxWindowSeconds?'components.waveform_hint_shift_reset':'components.waveform_hint_shift_full'):(maxWindowSeconds?'components.waveform_hint_reset':'components.waveform_hint_full'))}}</p>
</div>
</template>
<style scoped>
.wave-empty-message{grid-area:2/2;align-self:center;justify-self:center;max-width:90%;text-align:center;color:var(--muted);font-size:var(--support-size);margin:0;z-index:1;pointer-events:none}.waveform-placeholder svg{background:var(--app);pointer-events:none}

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
