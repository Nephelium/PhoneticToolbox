<script setup lang="ts">
import {computed,ref,onMounted,onUnmounted,useId} from 'vue';
import {fontPayload} from '../state/fonts.ts';
import {plotTickLabel} from '../platform/plotTicks.ts';
export interface PlotTrace {label:string;times:number[];values:(number|null)[];color:string;right?:boolean;points?:boolean;opacity?:number;lineWidth?:number;dash?:string;exportColor?:string}
const props=withDefaults(defineProps<{title:string;x:[number,number];y:[number,number];right?:[number,number];reserveRightAxis?:boolean;unit?:string;rightUnit?:string;xUnit?:string;traces?:PlotTrace[];markers?:{time:number;label:string;dashed?:boolean;color?:string}[];raster?:{url:string;extent:number[]};height?:number;fill?:boolean;empty?:boolean;interactive?:boolean;wheelZoom?:boolean;livePan?:boolean;axisLabelWidth?:number}>(),{traces:()=>[],markers:()=>[],height:145,xUnit:'s'});
const emit=defineEmits<{seek:[time:number];zoom:[factor:number];pan:[delta:number]}>();const el=ref<HTMLElement>();const width=ref(500);let observer:ResizeObserver;const clip='plot-'+useId();
const availableHeight=ref(0),legendHeight=ref(20);
let measureFrame=0;
function measure(){if(!el.value)return;width.value=Math.max(180,el.value.clientWidth);availableHeight.value=el.value.clientHeight;legendHeight.value=el.value.querySelector<HTMLElement>('.plot-legend')?.offsetHeight??20;}
onMounted(()=>{observer=new ResizeObserver(()=>{cancelAnimationFrame(measureFrame);measureFrame=requestAnimationFrame(measure);});if(el.value){measure();observer.observe(el.value);const legend=el.value.querySelector('.plot-legend');if(legend)observer.observe(legend);}});onUnmounted(()=>{observer?.disconnect();cancelAnimationFrame(measureFrame);});
const fontSize=computed(()=>fontPayload.value?.size??12);const chartHeight=computed(()=>Math.max(props.fill&&availableHeight.value?availableHeight.value-legendHeight.value:props.height,6*fontSize.value+40));
const left=computed(()=>Math.max(54,Math.max(props.axisLabelWidth??0,...ticks(props.y).map(v=>label(v).length),...(props.right?ticks(props.right).map(v=>label(v).length):[]))*fontSize.value*.66+14)),top=computed(()=>fontSize.value+4),bottom=computed(()=>fontSize.value*2+22);const plotWidth=computed(()=>Math.max(1,width.value-left.value-(props.right||props.reserveRightAxis?left.value:16)));const plotHeight=computed(()=>chartHeight.value-top.value-bottom.value);
const X=(v:number)=>left.value+(v-props.x[0])/(props.x[1]-props.x[0])*plotWidth.value;
const Y=(v:number,right=false)=>{const r=right&&props.right?props.right:props.y;return top.value+(r[1]-v)/(r[1]-r[0])*plotHeight.value;};
const ticks=(range:[number,number],count=5)=>Array.from({length:count},(_,i)=>range[0]+i/(count-1)*(range[1]-range[0]));
const xTicks=computed(()=>ticks(props.x,Math.max(2,Math.min(5,Math.floor(plotWidth.value/(fontSize.value*5))))));
const label=(v:number,step?:number)=>plotTickLabel(v,step);
function path(trace:PlotTrace){let gap=true;return trace.times.map((t,i)=>{const v=trace.values[i];if(v==null||!Number.isFinite(v)){gap=true;return '';}const command=gap?'M':'L';gap=false;return `${command}${X(t).toFixed(2)},${Y(v,trace.right).toFixed(2)}`;}).join('');}
// Keep every scientific point; batch equal-radius circles to bound DOM nodes.
function dots(trace:PlotTrace){return trace.times.map((t,i)=>{const v=trace.values[i];if(v==null||!Number.isFinite(v))return '';return `M${X(t)-1.6},${Y(v,trace.right)}a1.6,1.6 0 1,0 3.2,0a1.6,1.6 0 1,0 -3.2,0Z`;}).join('');}
let drag:{x:number;last:number;span:number;pixels:number;id:number}|undefined;let dragged=false;
function seek(event:MouseEvent){if(dragged){dragged=false;return;}const box=(event.currentTarget as Element).getBoundingClientRect();const x=(event.clientX-box.left)*width.value/box.width-left.value;if(x>=0&&x<=plotWidth.value)emit('seek',props.x[0]+x/plotWidth.value*(props.x[1]-props.x[0]));}
function wheel(event:WheelEvent){if((!props.wheelZoom&&!event.ctrlKey)||!props.interactive||!event.deltaY)return;event.preventDefault();emit('zoom',event.deltaY>0?1.1:.9);}
function down(event:PointerEvent){if(!props.interactive||event.button!==0)return;const node=event.currentTarget as SVGElement;event.preventDefault();node.focus();drag={x:event.clientX,last:event.clientX,span:props.x[1]-props.x[0],pixels:plotWidth.value*node.getBoundingClientRect().width/width.value,id:event.pointerId};dragged=false;node.setPointerCapture(event.pointerId);}
function move(event:PointerEvent){if(!drag||event.pointerId!==drag.id||!props.livePan)return;if(!dragged&&Math.abs(event.clientX-drag.x)<=2)return;dragged=true;const delta=event.clientX-drag.last;drag.last=event.clientX;if(delta)emit('pan',delta/drag.pixels*drag.span);}
function up(event:PointerEvent){if(!drag||event.pointerId!==drag.id)return;if(props.livePan)move(event);else{const delta=event.clientX-drag.x;dragged=Math.abs(delta)>2;if(dragged)emit('pan',delta/drag.pixels*drag.span);}drag=undefined;}
function cancelDrag(){drag=undefined;dragged=true;}
function keys(event:KeyboardEvent){if(!props.interactive)return;const delta=(props.x[1]-props.x[0])*.1;if(event.key==='ArrowLeft')emit('pan',delta);else if(event.key==='ArrowRight')emit('pan',-delta);else if(event.key==='+'||event.key==='=')emit('zoom',.9);else if(event.key==='-')emit('zoom',1.1);else return;event.preventDefault();}
</script>
<template><div ref="el" class="scientific-plot" :class="{'fill-plot':fill}" :aria-label="title">
<div class="plot-legend"><span v-for="t in traces" :key="t.label" :style="{color:t.color}"><svg v-if="t.lineWidth||t.dash" class="legend-swatch" viewBox="0 0 28 12" aria-hidden="true"><line x1="1" x2="27" y1="6" y2="6" :stroke="t.color" :stroke-width="t.lineWidth??1.1" :stroke-dasharray="t.dash" stroke-linecap="round"/></svg>{{t.label}}{{t.right?' · 右轴':''}}</span><slot name="legend"/></div>
<div v-if="empty" class="plot-empty" :style="{height:chartHeight+'px'}">选择双声道 WAV 后自动显示</div>
<svg v-else :viewBox="`0 0 ${width} ${chartHeight}`" :style="{height:chartHeight+'px'}" role="img" :aria-label="title" @click="seek" @wheel="wheel" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="cancelDrag" @keydown="keys" :tabindex="interactive?0:undefined" :class="{interactive}">
<defs><clipPath :id="clip"><rect :x="left" :y="top" :width="plotWidth" :height="plotHeight"/></clipPath></defs>
<g :clip-path="`url(#${clip})`">
<image v-if="raster" :href="raster.url" :x="X(raster.extent[0])" :y="Y(raster.extent[3])" :width="X(raster.extent[1])-X(raster.extent[0])" :height="Y(raster.extent[2])-Y(raster.extent[3])" preserveAspectRatio="none"/>
<line v-for="v in ticks(y)" :key="v" :x1="left" :x2="left+plotWidth" :y1="Y(v)" :y2="Y(v)" stroke="var(--border)" stroke-dasharray="2 4"/>
<g v-for="trace in traces" :key="trace.label" class="scientific-trace" :data-export-color="trace.exportColor" :stroke="trace.color" :fill="trace.color" :opacity="trace.opacity??1"><path v-if="trace.points&&trace.times.length>3000" :d="dots(trace)" stroke="none" class="scientific-dots"/><template v-else-if="trace.points"><circle v-for="(t,i) in trace.times" :key="i" v-show="trace.values[i]!=null" :cx="X(t)" :cy="Y(trace.values[i]??0,trace.right)" r="1.6" stroke="none"/></template><path v-else :d="path(trace)" fill="none" :stroke-width="trace.lineWidth??1.1" :stroke-dasharray="trace.dash" stroke-linecap="round"/></g>
<line v-for="(m,i) in markers" :key="i" :x1="X(m.time)" :x2="X(m.time)" :y1="top" :y2="top+plotHeight"  :stroke="m.color??'var(--success)'" :stroke-dasharray="m.dashed?'4 3':undefined" opacity=".75"><title>{{m.label}} {{m.time}}</title></line>
</g>
<rect :x="left" :y="top" :width="plotWidth" :height="plotHeight" fill="none" stroke="var(--border)"/>
<text v-for="v in ticks(y)" :key="'l'+v" :x="left-7" :y="Y(v)+4" text-anchor="end">{{label(v,(y[1]-y[0])/4)}}</text>
<text v-for="v in right?ticks(right):[]" :key="'r'+v" :x="left+plotWidth+7" :y="Y(v,true)+4">{{label(v,right?(right[1]-right[0])/4:undefined)}}</text>
<text v-for="(v,i) in xTicks" :key="'x'+v" :x="X(v)" :y="chartHeight-fontSize-10" :text-anchor="i===0?'start':i===xTicks.length-1?'end':'middle'">{{label(v,(x[1]-x[0])/(xTicks.length-1))}}</text>
<text :x="left" :y="fontSize">{{unit}}</text><text v-if="rightUnit" :x="left+plotWidth" :y="fontSize" text-anchor="end">{{rightUnit}}</text><text :x="width-4" :y="chartHeight-1" text-anchor="end">{{xUnit}}</text>
</svg></div></template>
<style scoped>
.scientific-plot{min-width:0;font-family:var(--font-figure,var(--font));font-size:var(--figure-size,12px)}.scientific-plot>svg{display:block;width:100%;flex:none;max-width:100%;cursor:crosshair}.scientific-plot .interactive{touch-action:none;user-select:none}.scientific-plot .interactive:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}.scientific-plot text{font-family:inherit;font-size:inherit;fill:var(--muted)}.plot-legend{display:flex;gap:12px;flex-wrap:wrap;min-height:20px;padding:2px 12px;font:inherit}.plot-empty{display:grid;place-items:center;color:var(--muted);padding:20px;text-align:center}
.scientific-plot.fill-plot{height:100%;min-height:0}
.plot-legend>span{display:inline-flex;align-items:center;gap:5px}.legend-swatch{width:28px;height:12px;flex:none}
</style>
