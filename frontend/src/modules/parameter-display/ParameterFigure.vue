<script setup lang="ts">
import {computed,ref,onMounted,onUnmounted} from 'vue';
import {preferences} from '../../state/fonts.ts';
import type {ParameterTable} from '../../platform/research.ts';
import type {Workspace} from '../../state/workspace.ts';
import {overlayPlot,annotationRuns,type PlotGroup,type OverlayCurve} from './state.ts';
import {styledSvg,editableSvg} from '../../design/svg-fonts.ts';
import {wholeFigurePng,currentFigurePng,downloadImage} from './export.ts';
const props=defineProps<{table:ParameterTable;group:PlotGroup;wave:Workspace;waveform:()=>HTMLElement|undefined}>();
const exporting=ref(false);
const format=ref<'png'|'svg'>('png');
const svg=ref<SVGSVGElement>(),surface=ref<HTMLElement>(),error=ref(''),measuredWidth=ref(600);
const viewportHeight=ref(window.innerHeight);const updateHeight=()=>{viewportHeight.value=window.innerHeight;};
const fontScale=computed(()=>preferences.value.figure.size/12);
const chartWidth=computed(()=>Math.max(Math.ceil(340*fontScale.value),measuredWidth.value));
let resize:ResizeObserver|undefined;
onMounted(()=>{window.addEventListener('resize',updateHeight);resize=new ResizeObserver(entries=>{const width=entries[0]?.contentRect.width;if(width)measuredWidth.value=Math.floor(width);});if(surface.value)resize.observe(surface.value);});
onUnmounted(()=>{resize?.disconnect();window.removeEventListener('resize',updateHeight);});
const duration=computed(()=>props.wave.asset?.duration??1),span=computed(()=>duration.value/props.wave.zoom);
const start=computed(()=>Math.min(props.wave.offset,Math.max(0,duration.value-span.value)));
const chart=computed(()=>overlayPlot(props.table,props.group.parameters,start.value,start.value+span.value,chartWidth.value));
const annotations=computed(()=>props.group.parameters.filter(n=>props.table.kinds[props.table.columns.indexOf(n)]==='text').map(name=>({name,runs:annotationRuns(props.table,name,start.value,start.value+span.value)})));
const plotLeft=computed(()=>76*fontScale.value),plotRight=computed(()=>chartWidth.value-(chart.value.dual?76:22)*fontScale.value),plotWidth=computed(()=>plotRight.value-plotLeft.value);
const plotTop=computed(()=>(28+annotations.value.length*23)*fontScale.value),plotHeight=computed(()=>Math.max(220,viewportHeight.value-780)),plotBottom=computed(()=>plotTop.value+plotHeight.value);
const legendColumns=computed(()=>Math.max(1,Math.floor(chartWidth.value/(210*fontScale.value)))),legendWidth=computed(()=>chartWidth.value/legendColumns.value);
const chartHeight=computed(()=>plotBottom.value+(58+Math.ceil(chart.value.curves.length/legendColumns.value)*24)*fontScale.value);
const colorTokens=['--accent','--warning','--teal','--danger','--violet','--success'];
const styleFor=(name:string)=>{const index=Math.max(0,props.table.columns.indexOf(name)-1);return {color:'var('+colorTokens[index%colorTokens.length]+')',dash:index>=colorTokens.length||name.toLowerCase().includes('rf0')?'7 3':undefined};};
const x=(t:number)=>plotLeft.value+(t-start.value)/span.value*plotWidth.value;
const scale=(curve:OverlayCurve)=>chart.value[curve.axis];
const y=(curve:OverlayCurve,value:number)=>plotBottom.value-(value-scale(curve).min)/(scale(curve).max-scale(curve).min)*plotHeight.value;
function path(curve:OverlayCurve){let pen=false;return curve.points.map(point=>{if(point.value===null){pen=false;return '';}const command=pen?'L':'M';pen=true;return command+x(point.time).toFixed(3)+' '+y(curve,point.value).toFixed(3);}).join(' ');}
function label(name:string){const max=Math.max(12,Math.floor((legendWidth.value-55)/(7*fontScale.value)));return name.length>max?name.slice(0,max-1)+'…':name;}
const printable=(text:string)=>text&&!['sil','<sil>','eps','nan'].includes(text.toLowerCase());
let drag:{x:number;offset:number;time:number;select:boolean}|null=null;
function time(event:PointerEvent){const rect=svg.value!.getBoundingClientRect();return Math.max(0,Math.min(duration.value,start.value+((event.clientX-rect.left)/rect.width*chartWidth.value-plotLeft.value)/plotWidth.value*span.value));}
function down(event:PointerEvent){if(event.button!==0)return;drag={x:event.clientX,offset:start.value,time:time(event),select:event.shiftKey};svg.value!.setPointerCapture(event.pointerId);if(drag.select){props.wave.start=drag.time;props.wave.end=drag.time;}}
function move(event:PointerEvent){if(!drag)return;if(drag.select){props.wave.start=Math.min(drag.time,time(event));props.wave.end=Math.max(drag.time,time(event));}else{const rect=svg.value!.getBoundingClientRect();const dx=(event.clientX-drag.x)/rect.width*chartWidth.value/plotWidth.value*span.value;props.wave.offset=Math.max(0,Math.min(duration.value-span.value,drag.offset-dx));}}
function wheel(event:WheelEvent){if(!event.ctrlKey||!event.deltaY)return;event.preventDefault();const rect=svg.value!.getBoundingClientRect();const fraction=Math.max(0,Math.min(1,((event.clientX-rect.left)/rect.width*chartWidth.value-plotLeft.value)/plotWidth.value));const anchor=start.value+fraction*span.value;props.wave.zoom=Math.max(1,Math.min(Math.max(1,duration.value/.01),props.wave.zoom*(event.deltaY<0?1.25:.8)));props.wave.offset=Math.max(0,Math.min(duration.value-duration.value/props.wave.zoom,anchor-fraction*duration.value/props.wave.zoom));}
async function save(){
 if(exporting.value)return;exporting.value=true;error.value='';
 try{if(format.value==='png'){downloadImage(await currentFigurePng(svg.value!),props.group.title+'.png');return;}
  const clone=styledSvg(svg.value!);clone.style.width=chartWidth.value+'px';clone.style.height=chartHeight.value+'px';clone.setAttribute('width',String(chartWidth.value));clone.setAttribute('height',String(chartHeight.value));
  downloadImage(new Blob([await editableSvg(clone)],{type:'image/svg+xml;charset=utf-8'}),props.group.title+'.svg');
 }catch(e){error.value=e instanceof Error?e.message:'图片导出失败，请重试。';}finally{exporting.value=false;}
}
async function saveWhole(){
 if(exporting.value)return;error.value='';exporting.value=true;
 try{const waveform=props.waveform();if(!svg.value||!waveform)throw Error('波形尚未就绪。');
  const name=(props.wave.asset?.name??'参数').replace(/\.wav$/i,'')+'-'+props.group.title;
  const blob=await wholeFigurePng({chart:svg.value,waveform,title:name,start:start.value,end:start.value+span.value,plotLeft:plotLeft.value,plotRight:plotRight.value});
  downloadImage(blob,name+'.png');
 }catch(e){error.value=e instanceof Error?e.message:'整幅图片导出失败，请重试。';}finally{exporting.value=false;}
}
</script>
<template><section class="parameter-figure">
<header class="section-title"><div><h3>{{group.title}}</h3><small>{{chart.curves.length}} 条曲线叠加 · {{chart.dual?'自动双纵轴':'共用纵轴'}}</small></div><div><slot/><select v-model="format" :aria-label="group.title+'图片格式'" class="image-format"><option value="png">PNG</option><option value="svg">SVG</option></select><button @click="save" :disabled="!group.parameters.length||exporting" :title="'仅保存参数图 '+format.toUpperCase()">保存当前图</button><button @click="saveWhole" :disabled="!group.parameters.length||exporting" title="白底 300 dpi，包含波形、已开启语谱图及此参数图">{{exporting?'正在生成 PNG…':'保存整幅 PNG'}}</button></div></header>
<p v-if="error" role="alert">{{error}}</p><div ref="surface" class="plot-surface">
<div v-if="!group.parameters.length" class="empty-plot"><strong>{{group.title}} · 待分配参数</strong><p>勾选参数并分配到此图窗后，多条曲线将在同一绘图区叠加。</p></div>
<svg v-else ref="svg" class="parameter-chart" :viewBox="'0 0 '+chartWidth+' '+chartHeight" :style="{height:chartHeight+'px',minWidth:Math.ceil(340*fontScale)+'px'}" role="img" :aria-label="group.title+'叠加参数曲线'" @pointerdown="down" @pointermove="move" @pointerup="move($event);drag=null" @pointercancel="drag=null" @wheel="wheel" @dblclick="wave.zoom=1;wave.offset=0">
<rect :width="chartWidth" :height="chartHeight" fill="var(--panel)"/>
<g class="shared-plot-area" :data-plot-top="plotTop" :data-plot-bottom="plotBottom">
<rect :x="Math.max(plotLeft,Math.min(plotRight,x(wave.start)))" :y="plotTop" :width="Math.max(0,Math.min(plotRight,x(wave.end))-Math.max(plotLeft,x(wave.start)))" :height="plotHeight" fill="var(--selection)"/>
<g v-for="i in 5" :key="i">
<path :d="'M'+plotLeft+' '+(plotTop+(i-1)*plotHeight/4)+'H'+plotRight" stroke="var(--border)" stroke-dasharray="2 4"/>
<text class="left-tick" :x="plotLeft-9" :y="plotTop+(i-1)*plotHeight/4+4" text-anchor="end" fill="var(--muted)" font-size="12">{{(chart.left.max-(i-1)*(chart.left.max-chart.left.min)/4).toPrecision(4)}}</text>
<text v-if="chart.dual" class="right-tick" :x="plotRight+9" :y="plotTop+(i-1)*plotHeight/4+4" fill="var(--muted)" font-size="12">{{(chart.right.max-(i-1)*(chart.right.max-chart.right.min)/4).toPrecision(4)}}</text>
<text :x="plotLeft+(i-1)*plotWidth/4" :y="plotBottom+22*fontScale" text-anchor="middle" font-size="12" fill="var(--muted)">{{(start+(i-1)*span/4).toFixed(3)}}</text>
</g>
<path :d="'M'+plotLeft+' '+plotTop+'V'+plotBottom+'H'+plotRight+(chart.dual?'V'+plotTop:'')" stroke="var(--muted)" fill="none"/>
<text :x="plotLeft" :y="plotTop-9" fill="var(--muted)" font-size="12">{{chart.dual?'左轴 · 原数值':'原数值'}}</text>
<text v-if="chart.dual" :x="plotRight" :y="plotTop-9" text-anchor="end" fill="var(--muted)" font-size="12">右轴 · 原数值</text>
<path v-for="curve in chart.curves" :key="curve.name" class="parameter-curve" :data-parameter="curve.name" :data-axis="curve.axis" :data-scale-min="scale(curve).min" :data-scale-max="scale(curve).max" :d="path(curve)" :stroke="styleFor(curve.name).color" :stroke-dasharray="curve.axis==='right'?'7 3':styleFor(curve.name).dash" stroke-width="1.8" fill="none"><title>{{curve.name}} · {{curve.axis==='left'?'左轴':'右轴'}}</title></path>
<g v-for="(tier,index) in annotations" :key="tier.name"><g v-for="run in tier.runs" :key="run.start"><path :d="'M'+x(run.start)+' '+((index*23+5)*fontScale)+'V'+plotBottom" :stroke="styleFor(tier.name).color" stroke-dasharray="2 4" opacity=".65"/><text class="ipa-text" v-if="printable(run.text)" :x="x((run.start+run.end)/2)" :y="(index*23+16)*fontScale" text-anchor="middle" font-size="13" fill="var(--text)"><title>{{tier.name}}</title>{{run.text}}</text></g></g>
</g>
<text :x="plotLeft+plotWidth/2" :y="plotBottom+44*fontScale" text-anchor="middle" fill="var(--text)" font-size="14">时间（秒）</text>
<g v-for="(curve,index) in chart.curves" :key="'legend-'+curve.name" class="curve-legend" :transform="'translate('+((index%legendColumns)*legendWidth+8)+' '+(plotBottom+(64+Math.floor(index/legendColumns)*24)*fontScale)+')'"><path d="M0 -4H25" :stroke="styleFor(curve.name).color" :stroke-dasharray="curve.axis==='right'?'7 3':styleFor(curve.name).dash" stroke-width="2"/><text x="33" y="0" font-size="13" fill="var(--text)"><title>{{curve.name}}</title>{{label(curve.name)}}{{chart.dual?(curve.axis==='right'?' [右]':' [左]'):''}}</text></g>
</svg></div><p v-if="group.parameters.length" class="plot-help">Ctrl＋滚轮缩放时间轴 · 左键拖动平移 · Shift＋拖动选区 · 双击全长<span v-if="chart.dual">。双轴沿用 v2 的量级判定；图例标明所属纵轴，原数值不归一化。</span></p></section></template>
<style scoped>
.image-format{width:auto;max-width:90px}
.parameter-figure{border:1px solid var(--border);border-radius:8px;background:var(--panel);padding:10px;margin:10px 0;min-width:0}.parameter-figure>.section-title{flex-wrap:wrap;margin-bottom:8px}.section-title>div{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.section-title h3{margin:0;font-size:15px}.section-title small{color:var(--muted);font-size:12px}.plot-surface{width:100%;min-width:0;overflow-x:auto}.parameter-chart{display:block;width:100%;min-width:340px;max-width:none;background:var(--panel);touch-action:none;font-family:var(--font-figure);cursor:grab}.parameter-chart:active{cursor:grabbing}.empty-plot{min-height:180px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:24px;color:var(--muted);border:1px dashed var(--border);border-radius:5px;background:var(--app)}.empty-plot strong{font-size:16px;color:var(--text)}.empty-plot p{max-width:340px;line-height:1.7}.plot-help{font-size:12px;color:var(--muted);line-height:1.6;margin:10px 0 0}
</style>
