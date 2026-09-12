<script setup lang="ts">
import {computed,ref,onMounted,onUnmounted,useId} from 'vue';
import {fontPayload} from '../state/fonts.ts';
export interface PlotTrace {label:string;times:number[];values:(number|null)[];color:string;right?:boolean;points?:boolean}
const props=withDefaults(defineProps<{title:string;x:[number,number];y:[number,number];right?:[number,number];unit?:string;rightUnit?:string;xUnit?:string;traces?:PlotTrace[];markers?:{time:number;label:string;dashed?:boolean;color?:string}[];raster?:{url:string;extent:number[]};height?:number;empty?:boolean;interactive?:boolean}>(),{traces:()=>[],markers:()=>[],height:145,xUnit:'s'});
const emit=defineEmits<{seek:[time:number];zoom:[factor:number];pan:[delta:number]}>();const el=ref<HTMLElement>();const width=ref(500);let observer:ResizeObserver;const clip='plot-'+useId();
onMounted(()=>{observer=new ResizeObserver(e=>width.value=Math.max(180,e[0].contentRect.width));if(el.value)observer.observe(el.value);});onUnmounted(()=>observer?.disconnect());
const fontSize=computed(()=>fontPayload.value?.size??12);const chartHeight=computed(()=>Math.max(props.height,6*fontSize.value+40));
const left=computed(()=>Math.max(54,(fontPayload.value?.size??12)*3.2)),top=computed(()=>(fontPayload.value?.size??12)+4),bottom=computed(()=>(fontPayload.value?.size??12)+22);const plotWidth=computed(()=>Math.max(1,width.value-left.value-(props.right?left.value:16)));const plotHeight=computed(()=>chartHeight.value-top.value-bottom.value);
const X=(v:number)=>left.value+(v-props.x[0])/(props.x[1]-props.x[0])*plotWidth.value;
const Y=(v:number,right=false)=>{const r=right&&props.right?props.right:props.y;return top.value+(r[1]-v)/(r[1]-r[0])*plotHeight.value;};
const ticks=(range:[number,number],count=5)=>Array.from({length:count},(_,i)=>range[0]+i/(count-1)*(range[1]-range[0]));
const xTicks=computed(()=>ticks(props.x,Math.max(2,Math.min(5,Math.floor(plotWidth.value/(fontSize.value*5))))));
const label=(v:number)=>Math.abs(v)>=100?String(Math.round(v)):Number(v.toFixed(3)).toString();
function path(trace:PlotTrace){let gap=true;return trace.times.map((t,i)=>{const v=trace.values[i];if(v==null||!Number.isFinite(v)){gap=true;return '';}const command=gap?'M':'L';gap=false;return `${command}${X(t).toFixed(2)},${Y(v,trace.right).toFixed(2)}`;}).join('');}
let drag:{x:number;span:number;pixels:number;id:number}|undefined;let dragged=false;
function seek(event:MouseEvent){if(dragged){dragged=false;return;}const box=(event.currentTarget as Element).getBoundingClientRect();const x=(event.clientX-box.left)-left.value;if(x>=0&&x<=plotWidth.value)emit('seek',props.x[0]+x/plotWidth.value*(props.x[1]-props.x[0]));}
function wheel(event:WheelEvent){if(!props.interactive||!event.deltaY)return;event.preventDefault();emit('zoom',event.deltaY>0?1.1:.9);}
function down(event:PointerEvent){if(!props.interactive||event.button!==0)return;const node=event.currentTarget as SVGElement;drag={x:event.clientX,span:props.x[1]-props.x[0],pixels:plotWidth.value,id:event.pointerId};dragged=false;node.setPointerCapture(event.pointerId);}
function up(event:PointerEvent){if(!drag||event.pointerId!==drag.id)return;const delta=event.clientX-drag.x;dragged=Math.abs(delta)>2;if(dragged)emit('pan',delta/drag.pixels*drag.span);drag=undefined;}
function keys(event:KeyboardEvent){if(!props.interactive)return;const delta=(props.x[1]-props.x[0])*.1;if(event.key==='ArrowLeft')emit('pan',delta);else if(event.key==='ArrowRight')emit('pan',-delta);else if(event.key==='+'||event.key==='=')emit('zoom',.9);else if(event.key==='-')emit('zoom',1.1);else return;event.preventDefault();}
</script>
<template><div ref="el" class="scientific-plot" :aria-label="title">
<div class="plot-legend"><span v-for="t in traces" :key="t.label" :style="{color:t.color}">{{t.label}}{{t.right?' · 右轴':''}}</span><slot name="legend"/></div>
<div v-if="empty" class="plot-empty" :style="{height:chartHeight+'px'}">选择双声道 WAV 并更新分析后显示</div>
<svg v-else :viewBox="`0 0 ${width} ${chartHeight}`" :style="{height:chartHeight+'px'}" role="img" :aria-label="title" @click="seek" @wheel="wheel" @pointerdown="down" @pointerup="up" @pointercancel="drag=undefined" @keydown="keys" :tabindex="interactive?0:undefined" :class="{interactive}">
<defs><clipPath :id="clip"><rect :x="left" :y="top" :width="plotWidth" :height="plotHeight"/></clipPath></defs>
<g :clip-path="`url(#${clip})`">
<image v-if="raster" :href="raster.url" :x="X(raster.extent[0])" :y="Y(raster.extent[3])" :width="X(raster.extent[1])-X(raster.extent[0])" :height="Y(raster.extent[2])-Y(raster.extent[3])" preserveAspectRatio="none"/>
<line v-for="v in ticks(y)" :key="v" :x1="left" :x2="left+plotWidth" :y1="Y(v)" :y2="Y(v)" stroke="var(--border)" stroke-dasharray="2 4"/>
<g v-for="trace in traces" :key="trace.label" :stroke="trace.color" :fill="trace.color"><template v-if="trace.points"><circle v-for="(t,i) in trace.times" :key="i" v-show="trace.values[i]!=null" :cx="X(t)" :cy="Y(trace.values[i]??0,trace.right)" r="1.6" stroke="none"/></template><path v-else :d="path(trace)" fill="none" stroke-width="1.1"/></g>
<line v-for="(m,i) in markers" :key="i" :x1="X(m.time)" :x2="X(m.time)" :y1="top" :y2="top+plotHeight"  :stroke="m.color??'var(--success)'" :stroke-dasharray="m.dashed?'4 3':undefined" opacity=".75"><title>{{m.label}} {{m.time}}</title></line>
</g>
<rect :x="left" :y="top" :width="plotWidth" :height="plotHeight" fill="none" stroke="var(--border)"/>
<text v-for="v in ticks(y)" :key="'l'+v" :x="left-7" :y="Y(v)+4" text-anchor="end">{{label(v)}}</text>
<text v-for="v in right?ticks(right):[]" :key="'r'+v" :x="left+plotWidth+7" :y="Y(v,true)+4">{{label(v)}}</text>
<text v-for="(v,i) in xTicks" :key="'x'+v" :x="X(v)" :y="chartHeight-14" :text-anchor="i===0?'start':i===xTicks.length-1?'end':'middle'">{{label(v)}}</text>
<text :x="left" :y="fontSize">{{unit}}</text><text v-if="rightUnit" :x="left+plotWidth" :y="fontSize" text-anchor="end">{{rightUnit}}</text><text :x="width-4" :y="chartHeight-1" text-anchor="end">{{xUnit}}</text>
</svg></div></template>
<style scoped>
.scientific-plot{min-width:0;font-family:var(--font-figure,var(--font));font-size:var(--figure-size,12px)}.scientific-plot>svg{display:block;width:100%;flex:none;max-width:100%;cursor:crosshair}.scientific-plot .interactive{touch-action:none}.scientific-plot .interactive:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}.scientific-plot text{font-family:inherit;font-size:inherit;fill:var(--muted)}.plot-legend{display:flex;gap:12px;flex-wrap:wrap;min-height:20px;padding:2px 12px;font:inherit}.plot-empty{display:grid;place-items:center;color:var(--muted);padding:20px;text-align:center}
</style>
