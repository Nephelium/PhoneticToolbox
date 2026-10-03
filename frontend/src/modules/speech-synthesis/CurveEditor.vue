<script setup lang="ts">
import {computed,ref,onMounted,onUnmounted,useId} from 'vue';
import type {Config} from './state.ts';import {parameters,draw} from './state.ts';
const props=defineProps<{config:Config;name:string;start:number;end:number;readonly?:boolean}>();
const emit=defineEmits<{change:[];range:[start:number,end:number];error:[message:string]}>();
const clipId="m06-curve-"+useId();const svg=ref<SVGSVGElement>();let anchor:number|null=null,panX=0,panStart=0,mode='';
const width=ref(1000),height=ref(280),right=computed(()=>width.value-24),bottom=computed(()=>height.value-40),plotWidth=computed(()=>width.value-96),plotHeight=computed(()=>height.value-70);let observer:ResizeObserver;
onMounted(()=>{observer=new ResizeObserver(e=>{width.value=Math.max(100,e[0].contentRect.width);height.value=Math.max(100,e[0].contentRect.height)});if(svg.value)observer.observe(svg.value)});onUnmounted(()=>observer?.disconnect());
const factor=computed(()=>props.name==='Shimmer'?100:1),limits=computed(()=>props.name==='F0'?props.config.f0_range:parameters[props.name].slice(1,3) as number[]);
const curve=computed(()=>props.config.curves[props.name]);
const points=computed(()=>curve.value.override===null?curve.value.points:[[0,curve.value.override],[props.config.duration,curve.value.override]]);
const x=(t:number)=>72+(t-props.start)/(props.end-props.start)*plotWidth.value;
const y=(v:number)=>bottom.value-(v-limits.value[0])/(limits.value[1]-limits.value[0])*plotHeight.value;
const curvePath=(values:number[][])=>values.map(([t,v],i)=>(i?'L':'M')+x(t)+','+y(v)).join(' ');
const path=computed(()=>curvePath(points.value));
const references=computed(()=>/^F[1-5]$/.test(props.name)?['F1','F2','F3','F4','F5'].filter(n=>n!==props.name).map(n=>{
 const c=props.config.curves[n];return {name:n,path:curvePath(c.override===null?c.points:[[0,c.override],[props.config.duration,c.override]])};
}):[]);
function position(e:PointerEvent){const point=svg.value!.createSVGPoint();point.x=e.clientX;point.y=e.clientY;const p=point.matrixTransform(svg.value!.getScreenCTM()!.inverse());return [props.start+(p.x-72)/plotWidth.value*(props.end-props.start),limits.value[1]-(p.y-30)/plotHeight.value*(limits.value[1]-limits.value[0])];}
function down(e:PointerEvent){if(e.button!==0)return;const [t]=position(e);mode=e.shiftKey?'draw':e.ctrlKey?'reset':'pan';if(mode!=='pan'&&props.readonly){emit('error','选择使用编辑 F0 曲线后可修改音高');return;}if(mode!=='pan'&&curve.value.override!==null){emit('error','请先清除覆盖，再编辑曲线');return;}anchor=t;panX=e.clientX;panStart=props.start;svg.value?.setPointerCapture(e.pointerId);move(e);}
function move(e:PointerEvent){if(anchor===null)return;const [t,v]=position(e);if(mode==='pan'){const delta=(e.clientX-panX)/(plotWidth.value*svg.value!.getBoundingClientRect().width/width.value)*(props.end-props.start),span=props.end-props.start,a=Math.max(0,Math.min(props.config.duration-span,panStart-delta));emit('range',a,a+span);}else{draw(props.config,props.name,anchor,t,v,mode==='reset');anchor=t;emit('change');}}
function wheel(e:WheelEvent){if(!e.ctrlKey)return;e.preventDefault();const width=Math.min(props.config.duration,Math.max(.01,(props.end-props.start)*(e.deltaY>0?1.2:1/1.2))),a=Math.max(0,Math.min(props.config.duration-width,(props.start+props.end-width)/2));emit('range',a,a+width);}
</script>
<template><div class="curve-editor"><div class="track-label"><span>{{name}} · {{parameters[name][3]||'比例'}}<small v-if="references.length"> · 虚线：其他共振峰（只读）</small></span><span>{{readonly?'保留原 F0 · 编辑曲线未启用':curve.override===null?'Shift 绘制 · Ctrl 擦除 · 拖动平移 · Ctrl 滚轮缩放':'覆盖生效，清除覆盖后可编辑'}}</span></div><svg ref="svg" :viewBox="`0 0 ${width} ${height}`" :data-start="start" :data-end="end" role="img" :aria-label="name+' 参数曲线'" @pointerdown="down" @pointermove="move" @pointerup="anchor=null" @pointercancel="anchor=null" @wheel="wheel">
<defs><clipPath  :id="clipId"><rect x="72" y="25" :width="plotWidth" :height="plotHeight+10"/></clipPath></defs>
<g v-for="i in 5" :key="i"><path :d="`M72,${30+(i-1)*plotHeight/4}H${right}`" class="grid"/><text x="64" :y="35+(i-1)*plotHeight/4" text-anchor="end">{{((limits[1]-(i-1)/4*(limits[1]-limits[0]))*factor).toFixed(1)}}</text></g>
<g :clip-path="`url(#${clipId})`"><rect v-for="[a,b] in config.silence" :key="a" :x="x(a)" y="25" :width="x(b)-x(a)" :height="plotHeight+10" class="silence"/><path v-for="t in config.boundaries" :key="t" :d="`M${x(t)},25V${bottom+5}`" class="boundary"/><path v-for="reference in references" :key="reference.name" :data-formant="reference.name" :d="reference.path" class="reference-curve"><title>{{reference.name}} 只读参考</title></path><path :d="path" class="curve" :class="{override:curve.override!==null}"/></g>
<text x="72" :y="height-10">{{start.toFixed(3)}} s</text><text :x="right" :y="height-10" text-anchor="end">{{end.toFixed(3)}} s</text></svg></div></template>
<style scoped>.curve-editor{border:1px solid var(--border);border-radius:8px;overflow:hidden}svg{display:block;width:100%;height:260px;min-height:100px;touch-action:none;user-select:none;background:var(--panel)}text{fill:var(--text);font-family:var(--font-figure);font-size:var(--figure-size)}.grid{stroke:var(--border);fill:none}.reference-curve{stroke:var(--accent);fill:none;stroke-width:1.5;stroke-dasharray:6 5;opacity:.4;pointer-events:none}.curve{stroke:var(--accent);fill:none;stroke-width:2}.override{stroke:var(--danger)}.boundary{stroke:var(--warning);stroke-dasharray:4 4}.silence{fill:var(--selection)}.track-label{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}</style>
