<script setup lang="ts">
import {computed,ref,watch,nextTick,onMounted,onBeforeUnmount} from 'vue';
import type {Preview,Role} from './types.ts';
const props=defineProps<{preview:Preview|null;selection:[number,number];roles:Role[];spectrum:boolean;disabled:boolean;scale:number}>();
const emit=defineEmits<{selection:[value:[number,number]];focus:[value:boolean]}>();
const host=ref<HTMLElement>(),canvas=ref<HTMLCanvasElement>();let anchor:number|null=null,observer:ResizeObserver|undefined,drawFrame=0;
const wave=computed(()=>props.preview?.wave??[]),start=computed(()=>props.preview?.start_frame??0),length=computed(()=>props.preview?.window_frames??0),rate=computed(()=>props.preview?.sample_rate??48000);
function points(bins:number[][]){if(!bins.length)return '';return bins.flatMap((pair,i)=>{const x=bins.length===1?500:i/(bins.length-1)*1000;return [`${x},${50-Math.max(-1,Math.min(1,pair[0]*props.scale))*43}`,`${x},${50-Math.max(-1,Math.min(1,pair[1]*props.scale))*43}`];}).join(' ');}
function frame(event:PointerEvent){const box=host.value!.querySelector('.m16-wave-lanes')!.getBoundingClientRect();return Math.round(start.value+Math.max(0,Math.min(1,(event.clientX-box.left)/box.width))*length.value);}
function down(event:PointerEvent){if(props.disabled||!length.value||event.button!==0)return;host.value?.focus();anchor=frame(event);(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);emit('selection',[anchor,anchor]);}
function move(event:PointerEvent){if(anchor===null)return;const at=frame(event);emit('selection',[Math.min(anchor,at),Math.max(anchor,at)]);}
function up(){anchor=null;}
const selectionStyle=computed(()=>({left:`${Math.max(0,(props.selection[0]-start.value)/Math.max(1,length.value)*100)}%`,width:`${Math.min(100,(props.selection[1]-props.selection[0])/Math.max(1,length.value)*100)}%`}));
function draw(){const el=canvas.value;if(!el)return;const rect=el.getBoundingClientRect(),ratio=window.devicePixelRatio||1;const width=Math.max(1,Math.round(rect.width*ratio)),height=Math.max(1,Math.round(rect.height*ratio));if(el.width!==width)el.width=width;if(el.height!==height)el.height=height;const ctx=el.getContext('2d');if(!ctx)return;ctx.clearRect(0,0,el.width,el.height);const spec=props.preview?.spectrum;if(!spec?.rows.length)return;const image=ctx.createImageData(spec.rows.length,spec.frequencies.length);for(let x=0;x<spec.rows.length;x++)for(let y=0;y<spec.frequencies.length;y++){const value=Math.max(0,Math.min(1,(spec.rows[x][y]+100)/85)),at=((spec.frequencies.length-y-1)*spec.rows.length+x)*4;image.data[at]=Math.round(15+240*value**1.7);image.data[at+1]=Math.round(35+190*value);image.data[at+2]=Math.round(70+100*Math.sin(value*Math.PI));image.data[at+3]=255;}const buffer=document.createElement('canvas');buffer.width=image.width;buffer.height=image.height;buffer.getContext('2d')!.putImageData(image,0,0);ctx.imageSmoothingEnabled=false;const first=spec.times[0]??0,last=spec.times.at(-1)??first,step=spec.times.length>1?(last-first)/(spec.times.length-1):256/rate.value;const left=Math.max(0,first-step/2)*rate.value/Math.max(1,length.value),right=Math.min(length.value/rate.value,last+step/2)*rate.value/Math.max(1,length.value);ctx.drawImage(buffer,el.width*left,0,el.width*Math.max(0,right-left),el.height);}
watch(()=>[props.preview,props.spectrum],()=>nextTick(draw),{deep:false});onMounted(()=>{observer=new ResizeObserver(()=>{cancelAnimationFrame(drawFrame);drawFrame=requestAnimationFrame(draw);});if(host.value)observer.observe(host.value);draw();});onBeforeUnmount(()=>{observer?.disconnect();cancelAnimationFrame(drawFrame);});
</script>
<template>
 <div ref="host" class="m16-wave" :class="{'has-spectrum':spectrum}" tabindex="0" role="group" aria-label="录音波形选区编辑" @focus="emit('focus',true)" @blur="emit('focus',false)">
  <div class="m16-wave-lanes" title="多通道波形可上下滚动查看，时间选区始终同步" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="up">
   <div v-if="!wave.length" class="empty-wave">新建工程，选择设备后即可自由录音<br><small>原始采样保留在本地工程，空格只控制试听</small></div>
   <div v-for="(bins,index) in wave" :key="index" class="wave-lane">
    <span class="lane-label">输入 {{index+1}} · {{roles[index]==='egg'?'EGG':roles[index]==='microphone'?'音频':'其他'}}</span>
    <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true"><line x1="0" x2="1000" y1="50" y2="50" class="zero"/><polyline :points="points(bins)" class="wave-line"/></svg>
   </div>
   <div v-if="selection[1]>selection[0]&&wave.length" class="selection" :style="selectionStyle"/>
  </div>
  <div v-if="spectrum" class="m16-spectrum"><canvas ref="canvas"/><span>实时显示谱 · Hann 1024 / 步长 256 · 0–{{Math.round(rate/2)}} Hz</span><small v-if="preview?.spectrum_window_frames&&preview.spectrum_window_frames<length">仅显示当前窗口前 {{(preview.spectrum_window_frames/rate).toFixed(1)}} 秒，可缩放选区查看</small></div>
  <div class="m16-time-axis"><span>{{(start/rate).toFixed(3)}} s</span><span>{{((start+length/2)/rate).toFixed(3)}} s</span><span>{{((start+length)/rate).toFixed(3)}} s</span></div>
 </div>
</template>
<style scoped>
.m16-wave{height:100%;min-height:280px;display:flex;flex-direction:column;outline:none;border:1px solid var(--border);border-radius:8px;overflow:hidden;background:var(--panel)}.m16-wave:focus{box-shadow:0 0 0 2px var(--accent)}.m16-wave-lanes{flex:1;min-height:140px;position:relative;display:flex;flex-direction:column;touch-action:pan-y;overflow-y:auto;overflow-x:hidden}.wave-lane{position:relative;min-height:28px;flex:1;border-bottom:1px solid var(--border)}.wave-lane svg{width:100%;height:100%;position:absolute;inset:0}.lane-label{position:absolute;left:10px;top:7px;font-size:12px;z-index:1;background:var(--panel);padding:2px 5px;color:var(--muted)}.zero{stroke:var(--border);stroke-width:.5}.wave-line{fill:none;stroke:var(--wave);stroke-width:.7;vector-effect:non-scaling-stroke}.selection{position:absolute;inset-block:0;background:var(--selection);border-inline:1px solid var(--accent);pointer-events:none}.empty-wave{margin:auto;text-align:center;color:var(--muted);line-height:2}.m16-spectrum{position:relative;height:38%;min-height:100px;background:var(--app)}.m16-spectrum canvas{width:100%;height:100%;display:block}.m16-spectrum span,.m16-spectrum small{position:absolute;left:8px;top:6px;padding:2px 4px;background:var(--panel);color:var(--muted);font-size:11px}.m16-spectrum small{top:auto;bottom:4px}.m16-time-axis{display:flex;justify-content:space-between;padding:5px 10px;color:var(--muted);font:12px var(--font-figure,var(--font));background:var(--app)}
</style>
