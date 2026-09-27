<script setup lang="ts">
import {computed,ref} from 'vue';
import type {Config} from './state.ts';import {parameters,draw} from './state.ts';
const props=defineProps<{config:Config;name:string;start:number;end:number}>();
const emit=defineEmits<{change:[];range:[start:number,end:number];error:[message:string]}>();
const svg=ref<SVGSVGElement>();let anchor:number|null=null,panX=0,panStart=0,mode='';
const factor=computed(()=>props.name==='Shimmer'?100:1),limits=computed(()=>props.name==='F0'?props.config.f0_range:parameters[props.name].slice(1,3) as number[]);
const curve=computed(()=>props.config.curves[props.name]);
const points=computed(()=>curve.value.override===null?curve.value.points:[[0,curve.value.override],[props.config.duration,curve.value.override]]);
const x=(t:number)=>60+(t-props.start)/(props.end-props.start)*900;
const y=(v:number)=>240-(v-limits.value[0])/(limits.value[1]-limits.value[0])*210;
const path=computed(()=>points.value.map(([t,v],i)=>(i?'L':'M')+x(t)+','+y(v)).join(' '));
function position(e:PointerEvent){const point=svg.value!.createSVGPoint();point.x=e.clientX;point.y=e.clientY;const p=point.matrixTransform(svg.value!.getScreenCTM()!.inverse());return [props.start+(p.x-60)/900*(props.end-props.start),limits.value[1]-(p.y-30)/210*(limits.value[1]-limits.value[0])];}
function down(e:PointerEvent){if(e.button!==0)return;const [t]=position(e);mode=e.shiftKey?'draw':e.ctrlKey?'reset':'pan';if(mode!=='pan'&&curve.value.override!==null){emit('error','请先清除 Override，再编辑曲线');return;}anchor=t;panX=e.clientX;panStart=props.start;svg.value?.setPointerCapture(e.pointerId);move(e);}
function move(e:PointerEvent){if(anchor===null)return;const [t,v]=position(e);if(mode==='pan'){const delta=(e.clientX-panX)/svg.value!.getBoundingClientRect().width*(props.end-props.start),width=props.end-props.start,a=Math.max(0,Math.min(props.config.duration-width,panStart-delta));emit('range',a,a+width);}else{draw(props.config,props.name,anchor,t,v,mode==='reset');anchor=t;emit('change');}}
function wheel(e:WheelEvent){if(!e.ctrlKey)return;e.preventDefault();const width=Math.min(props.config.duration,Math.max(.01,(props.end-props.start)*(e.deltaY>0?1.2:1/1.2))),a=Math.max(0,Math.min(props.config.duration-width,(props.start+props.end-width)/2));emit('range',a,a+width);}
</script>
<template><div class="curve-editor"><div class="track-label"><span>{{name}} · {{parameters[name][3]||'比例'}}</span><span>{{curve.override===null?'Shift 绘制 · Ctrl 擦除 · 拖动平移 · Ctrl 滚轮缩放':'Override 生效，清除输入后可编辑'}}</span></div><svg ref="svg" viewBox="0 0 1000 280" role="img" :aria-label="name+' 参数曲线'" @pointerdown="down" @pointermove="move" @pointerup="anchor=null" @pointercancel="anchor=null" @wheel="wheel">
<defs><clipPath id="m06-curve-clip"><rect x="60" y="25" width="900" height="220"/></clipPath></defs>
<g v-for="i in 5" :key="i"><path :d="`M60,${30+(i-1)*52.5}H960`" class="grid"/><text x="52" :y="35+(i-1)*52.5" text-anchor="end">{{((limits[1]-(i-1)/4*(limits[1]-limits[0]))*factor).toFixed(1)}}</text></g>
<g clip-path="url(#m06-curve-clip)"><rect v-for="[a,b] in config.silence" :key="a" :x="x(a)" y="25" :width="x(b)-x(a)" height="220" class="silence"/><path v-for="t in config.boundaries" :key="t" :d="`M${x(t)},25V245`" class="boundary"/><path :d="path" class="curve" :class="{override:curve.override!==null}"/></g>
<text x="60" y="270">{{start.toFixed(3)}} s</text><text x="960" y="270" text-anchor="end">{{end.toFixed(3)}} s</text></svg></div></template>
<style scoped>.curve-editor{border:1px solid var(--border);border-radius:8px;overflow:hidden}svg{width:100%;height:260px;touch-action:none;user-select:none;background:var(--panel)}text{fill:var(--text);font-family:var(--font-figure);font-size:var(--figure-size)}.grid{stroke:var(--border);fill:none}.curve{stroke:var(--accent);fill:none;stroke-width:2}.override{stroke:var(--danger)}.boundary{stroke:var(--warning);stroke-dasharray:4 4}.silence{fill:var(--selection)}.track-label{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}</style>
