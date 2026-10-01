<script setup lang="ts">
import {plotTickLabel} from '../../platform/plotTicks.ts';
import {computed,ref,getCurrentInstance,onMounted,onUnmounted} from 'vue';
import {editCurve} from './state.ts';
const props=defineProps<{times:number[];original:number[];modified:number[];start:number;end:number;min:number;max:number;references:number[];editable?:boolean;label?:string}>();
const emit=defineEmits<{edit:[number[]];pan:[number];zoom:[number,number]}>();
const svg=ref<SVGSVGElement>();let gesture:'draw'|'restore'|'pan'|null=null,last:{index:number;value:number}|undefined,anchor=0;
const height=ref(250),bottom=computed(()=>height.value-40),plotHeight=computed(()=>height.value-60);
const width=ref(650),right=computed(()=>width.value-20),plotWidth=computed(()=>width.value-80);let observer:ResizeObserver;
onMounted(()=>{observer=new ResizeObserver(e=>{width.value=Math.max(240,e[0].contentRect.width);height.value=Math.max(160,e[0].contentRect.height)});if(svg.value)observer.observe(svg.value);});onUnmounted(()=>observer?.disconnect());
const clip='m08-clip-'+getCurrentInstance()!.uid;
const x=(t:number)=>60+(t-props.start)/(props.end-props.start)*plotWidth.value;
const y=(f:number)=>bottom.value-(f-props.min)/(props.max-props.min)*plotHeight.value;
function path(values:number[]){let pen=false;return props.times.map((t,i)=>{const f=values[i];if(f===0||!Number.isFinite(f)||t<props.start||t>props.end){pen=false;return '';}const cmd=pen?'L':'M';pen=true;return `${cmd}${x(t)},${y(f)}`;}).join(' ');}
const originalPath=computed(()=>path(props.original)),modifiedPath=computed(()=>path(props.modified));
function position(e:PointerEvent|WheelEvent){const matrix=svg.value!.getScreenCTM();if(!matrix)throw Error('图形坐标尚未就绪');const p=new DOMPoint(e.clientX,e.clientY).matrixTransform(matrix.inverse());return {time:props.start+(p.x-60)/plotWidth.value*(props.end-props.start),freq:props.max-(p.y-20)/plotHeight.value*(props.max-props.min)};}
function move(e:PointerEvent){if(!gesture)return;const p=position(e);if(gesture==='pan'){emit('pan',anchor-p.time);return;}if(p.time<props.start||p.time>props.end||p.freq<props.min||p.freq>props.max||!props.times.length)return;let index=0;for(let i=1;i<props.times.length;i++)if(Math.abs(props.times[i]-p.time)<Math.abs(props.times[index]-p.time))index=i;const next=editCurve(props.original,props.modified,index,p.freq,last,gesture==='restore');last=next.last;emit('edit',next.curve);}
function down(e:PointerEvent){if(e.button!==0)return;gesture=props.editable&&e.shiftKey&&!e.ctrlKey?'draw':props.editable&&e.ctrlKey&&!e.shiftKey?'restore':'pan';last=undefined;anchor=position(e).time;svg.value?.setPointerCapture(e.pointerId);move(e);}
function up(){gesture=null;last=undefined;}
function wheel(e:WheelEvent){if(!e.ctrlKey)return;e.preventDefault();emit('zoom',e.deltaY<0?1.25:.8,position(e).time);}
function key(e:KeyboardEvent){if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();emit('pan',(props.end-props.start)*(e.key==='ArrowLeft'?-.1:.1));}}
</script>
<template><svg ref="svg" class="m08-curve" :viewBox="`0 0 ${width} ${height}`" role="img" :aria-label="label??'F0 曲线，Shift 拖动编辑，Ctrl 拖动恢复'" tabindex="0" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="up" @lostpointercapture="up" @wheel="wheel" @keydown="key">
<defs><clipPath :id="clip"><rect x="60" y="20" :width="plotWidth" :height="plotHeight"/></clipPath></defs>
<g class="grid"><path :d="`M60,20V${bottom}H${right}`"/><template v-for="n in 5" :key="n"><path :d="`M60,${20+(n-1)*plotHeight/4}H${right}`"/><text x="53" :y="25+(n-1)*plotHeight/4" text-anchor="end">{{plotTickLabel(max-(n-1)*(max-min)/4)}}</text><text :x="60+(n-1)*plotWidth/4" :y="height-18" text-anchor="middle">{{(start+(n-1)*(end-start)/4).toFixed(3)}}</text></template></g>
<text x="60" y="14">F0 (Hz)</text><text :x="right" :y="height-2" text-anchor="end">Time (s)</text>
<g :clip-path="`url(#${clip})`"><path :d="originalPath" class="original"/><path :d="modifiedPath" class="modified"/><g v-for="(f,i) in references" :key="i"><path :d="`M60,${y(f)}H${right}`" class="reference"/><text x="65" :y="y(f)-3">{{f}} Hz</text></g></g>
</svg></template>
<style scoped>.m08-curve{display:block;width:100%;min-height:190px;touch-action:none;font-family:var(--font-figure);font-size:var(--figure-size);color:var(--text);fill:currentColor}.m08-curve path{fill:none;vector-effect:non-scaling-stroke}.grid path{stroke:var(--border)}.original{stroke:var(--wave);stroke-dasharray:3 4;stroke-width:1.3}.modified{stroke:var(--danger);stroke-width:1.5}.reference{stroke:var(--muted);stroke-dasharray:6 4}.m08-curve:focus-visible{outline:2px solid var(--accent)}</style>
