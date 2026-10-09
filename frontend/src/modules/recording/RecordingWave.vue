<script setup lang="ts">
import {computed,ref,watch,nextTick,onMounted,onBeforeUnmount} from 'vue';
import type {Preview,Role} from './types.ts';
import {channelName} from './state.ts';
const props=defineProps<{preview:Preview|null;selection:[number,number];roles:Role[];spectrum:boolean;disabled:boolean;scale:number;channel?:number;allChannels?:boolean}>();
const emit=defineEmits<{selection:[value:[number,number]];focus:[value:boolean]}>();
const host=ref<HTMLElement>(),canvas=ref<HTMLCanvasElement>();let anchor:number|null=null,observer:ResizeObserver|undefined,drawFrame=0;
const wave=computed(()=>props.allChannels?props.preview?.wave??[]:(props.preview?.wave??[]).slice(0,1)),start=computed(()=>props.preview?.start_frame??0),length=computed(()=>props.preview?.window_frames??0),rate=computed(()=>props.preview?.sample_rate??48000);
function points(bins:number[][]){if(!bins.length)return '';return bins.flatMap((pair,i)=>{const x=bins.length===1?500:i/(bins.length-1)*1000;return [`${x},${50-Math.max(-1,Math.min(1,pair[0]*props.scale))*43}`,`${x},${50-Math.max(-1,Math.min(1,pair[1]*props.scale))*43}`];}).join(' ');}
function frame(event:PointerEvent){const box=host.value!.querySelector('.m16-wave-lanes')!.getBoundingClientRect();return Math.round(start.value+Math.max(0,Math.min(1,(event.clientX-box.left)/box.width))*length.value);}
function down(event:PointerEvent){if(props.disabled||!length.value||event.button!==0)return;host.value?.focus();anchor=frame(event);(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);emit('selection',[anchor,anchor]);}
function move(event:PointerEvent){if(anchor===null)return;const at=frame(event);emit('selection',[Math.min(anchor,at),Math.max(anchor,at)]);}
function up(){anchor=null;}
const selectionStyle=computed(()=>{const a=Math.max(0,Math.min(length.value,props.selection[0]-start.value)),b=Math.max(a,Math.min(length.value,props.selection[1]-start.value));return {left:`${a/Math.max(1,length.value)*100}%`,width:`${(b-a)/Math.max(1,length.value)*100}%`};});
const maxFrequency=computed(()=>Math.min(5000,rate.value/2));
function draw(){
 const el=canvas.value;if(!el)return;
 const rect=el.getBoundingClientRect(),ratio=window.devicePixelRatio||1,width=Math.max(1,Math.round(rect.width*ratio)),height=Math.max(1,Math.round(rect.height*ratio));
 if(el.width!==width)el.width=width;if(el.height!==height)el.height=height;
 const ctx=el.getContext('2d');if(!ctx)return;ctx.clearRect(0,0,width,height);
 const spec=props.preview?.spectrum;if(!spec?.rows.length)return;
 const count=spec.frequencies.filter(f=>f<=maxFrequency.value).length;if(!count)return;
 const image=ctx.createImageData(spec.rows.length,count);
 for(let x=0;x<spec.rows.length;x++)for(let y=0;y<count;y++){
  const value=spec.pixels?.[x]?.[y]??255,at=((count-y-1)*spec.rows.length+x)*4;
  image.data[at]=image.data[at+1]=image.data[at+2]=value;image.data[at+3]=255;
 }
 const buffer=document.createElement('canvas');buffer.width=image.width;buffer.height=image.height;buffer.getContext('2d')!.putImageData(image,0,0);ctx.imageSmoothingEnabled=false;
 const first=spec.times[0]??0,last=spec.times.at(-1)??first,step=spec.times.length>1?(last-first)/(spec.times.length-1):256/rate.value;
 const left=Math.max(0,spec.time_edges?.[0]??first-step/2)*rate.value/Math.max(1,length.value),right=Math.min(length.value/rate.value,spec.time_edges?.at(-1)??last+step/2)*rate.value/Math.max(1,length.value);
 // Frequency cells are positioned by actual FFT bin spacing, then cropped to 5 kHz.
 const df=spec.frequency_step??(spec.frequencies[1]-spec.frequencies[0]||rate.value/1024),top=(1-(spec.frequencies[count-1]+df/2)/maxFrequency.value)*height;
 ctx.drawImage(buffer,width*left,top,width*Math.max(0,right-left),count*df/maxFrequency.value*height);
}
watch(()=>[props.preview,props.spectrum],()=>nextTick(draw),{deep:false});onMounted(()=>{observer=new ResizeObserver(()=>{cancelAnimationFrame(drawFrame);drawFrame=requestAnimationFrame(draw);});if(host.value)observer.observe(host.value);draw();});onBeforeUnmount(()=>{observer?.disconnect();cancelAnimationFrame(drawFrame);});
</script>
<template>
 <div ref="host" class="m16-wave" :class="{'has-spectrum':spectrum}" tabindex="0" role="group" aria-label="录音波形选区编辑" @focus="emit('focus',true)" @blur="emit('focus',false)">
  <div class="m16-wave-lanes" title="拖动选择时间范围，各声道选区同步" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="up">
   <div v-if="!wave.length" class="empty-wave">新建工程，选择设备后即可自由录音<br><small>原始采样保留在本地工程，空格只控制试听</small></div>
   <div v-for="(bins,index) in wave" :key="index" class="wave-lane">
    <span class="lane-label">{{channelName(index,roles)}}</span>
    <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true"><line x1="0" x2="1000" y1="50" y2="50" class="zero"/><polyline :points="points(bins)" class="wave-line"/></svg>
   </div>
   <div v-if="selection[1]>selection[0]&&wave.length" class="selection" :style="selectionStyle"/>
  </div>
  <div v-if="spectrum" class="m16-spectrum" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="up">
   <canvas ref="canvas" aria-label="语谱图 0–5000 Hz"/>
   <span class="spectrum-label" title="Praat Gaussian 5 ms · 50 dB 相对灰度 · 6 dB/oct 显示预加重。长窗口按时间抽样显示，可放大选区查看细节。">{{channelName(channel??0,roles)}} · Praat 语谱图</span>
   <span class="frequency-top">{{maxFrequency}} Hz</span><span class="frequency-bottom">0 Hz</span>
   <div v-if="selection[1]>selection[0]&&wave.length" class="selection" :style="selectionStyle"/>
   <small v-if="preview?.spectrum?.time_sampled" class="spectrum-hint">全段预览 · 放大选区查看细节</small>
  </div>
  <div class="m16-time-axis"><span>{{(start/rate).toFixed(3)}} s</span><span>{{((start+length/2)/rate).toFixed(3)}} s</span><span>{{((start+length)/rate).toFixed(3)}} s</span></div>
 </div>
</template>
<style scoped>
.m16-wave{height:100%;min-height:280px;display:flex;flex-direction:column;outline:none;border:1px solid var(--border);border-radius:8px;overflow:hidden;background:var(--panel)}.m16-wave:focus{box-shadow:0 0 0 2px var(--accent)}.m16-wave-lanes{flex:1;min-height:140px;position:relative;display:flex;flex-direction:column;touch-action:none;overflow:hidden}.wave-lane{position:relative;min-height:0;flex:1 1 0;border-bottom:1px solid var(--border)}.wave-lane svg{width:100%;height:100%;position:absolute;inset:0}.lane-label{position:absolute;left:8px;top:6px;font-size:0.857143rem;max-height:100%;overflow:hidden;z-index:1;background:var(--panel);padding:2px 5px;color:var(--muted)}.zero{stroke:var(--border);stroke-width:.5}.wave-line{fill:none;stroke:var(--waveform-color,var(--wave));stroke-width:.7;vector-effect:non-scaling-stroke}.selection{position:absolute;inset-block:0;background:var(--selection);border-inline:1px solid var(--accent);pointer-events:none}.empty-wave{margin:auto;text-align:center;color:var(--muted);line-height:2}.m16-spectrum{position:relative;height:38%;min-height:100px;background:#fff;touch-action:none;overflow:hidden}.m16-spectrum canvas{width:100%;height:100%;display:block}.m16-spectrum span,.m16-spectrum small{position:absolute;padding:2px 4px;background:var(--panel);color:var(--muted);font-size:0.785714rem;pointer-events:none}.spectrum-label{left:8px;top:6px;pointer-events:auto!important}.frequency-top{right:6px;top:6px}.frequency-bottom{right:6px;bottom:4px}.spectrum-hint{left:8px;bottom:4px}.m16-time-axis{display:flex;justify-content:space-between;padding:5px 0;color:var(--muted);font:0.857143rem var(--font-figure,var(--font));background:var(--app)}
</style>
