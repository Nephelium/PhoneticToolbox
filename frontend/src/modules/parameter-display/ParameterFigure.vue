<script setup lang="ts">
import {computed,ref,onMounted,onUnmounted,useId} from 'vue';
import {preferences,fontPayload} from '../../state/fonts.ts';
import type {ParameterTable} from '../../platform/research.ts';
import type {Workspace} from '../../state/workspace.ts';
import {overlayPlot,annotationRuns,formatPlotTick,type PlotGroup,type OverlayCurve} from './state.ts';
import {styledSvg,editableSvg} from '../../design/svg-fonts.ts';
import {wholeFigurePng,currentFigurePng,downloadImage} from './export.ts';
const props=defineProps<{table:ParameterTable;group:PlotGroup;wave:Workspace;waveform:()=>HTMLElement|undefined;timeAxisInsets?:{left:number;right:number};pending?:boolean;active?:boolean;expanded?:boolean}>();
const emit=defineEmits<{activate:[]}>();
const clipId='m02-plot-'+useId().replace(/[^a-zA-Z0-9_-]/g,'');
const exporting=ref(false);
const format=ref<'png'|'svg'>('png');
const svg=ref<SVGSVGElement>(),surface=ref<HTMLElement>(),error=ref(''),measuredWidth=ref(600);
const viewportHeight=ref(window.innerHeight);const updateHeight=()=>{viewportHeight.value=window.innerHeight;};
const fontScale=computed(()=>preferences.value.figure.size/12);
const chartWidth=computed(()=>Math.max(Math.ceil(340*fontScale.value),measuredWidth.value));
let resize:ResizeObserver|undefined,resizeFrame=0;
onMounted(()=>{window.addEventListener('resize',updateHeight);resize=new ResizeObserver(entries=>{const width=entries[0]?.contentRect.width;if(!width)return;cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{measuredWidth.value=Math.floor(width);});});if(surface.value)resize.observe(surface.value);});
onUnmounted(()=>{resize?.disconnect();cancelAnimationFrame(resizeFrame);window.removeEventListener('resize',updateHeight);});
const duration=computed(()=>props.wave.asset?.duration??1),span=computed(()=>duration.value/props.wave.zoom);
const start=computed(()=>Math.min(props.wave.offset,Math.max(0,duration.value-span.value)));
const chart=computed(()=>overlayPlot(props.table,props.group.parameters,start.value,start.value+span.value,chartWidth.value));
const annotations=computed(()=>props.group.parameters.filter(n=>props.table.kinds[props.table.columns.indexOf(n)]==='text').map(name=>({name,runs:annotationRuns(props.table,name,start.value,start.value+span.value)})));
const plotLeft=computed(()=>props.timeAxisInsets?.left??76*fontScale.value),plotRight=computed(()=>chartWidth.value-(props.timeAxisInsets?.right??(chart.value.dual?76:22)*fontScale.value)),plotWidth=computed(()=>Math.max(1,plotRight.value-plotLeft.value));
const plotHeight=computed(()=>Math.max(props.expanded?360:220,viewportHeight.value-(props.expanded?490:780)));
const measure=document.createElement('canvas').getContext('2d')!;
const legend=computed(()=>{
 const s=fontScale.value,pad=8*s,lineHeight=17*s,gap=5*s;
 measure.font=`${13*s}px ${fontPayload.value?.figure??'Times New Roman, SimSun, serif'}`;
 const maxText=Math.max(25,Math.min(300*s,plotWidth.value-66*s));
 const entries=chart.value.curves.map(curve=>{
  const text=curve.name+(chart.value.dual?(curve.axis==='right'?' [右]':' [左]'):'');
  const lines:string[]=[];let line='';
  for(const character of text){if(line&&measure.measureText(line+character).width>maxText){lines.push(line);line='';}line+=character;}if(line)lines.push(line);
  return {curve,lines};
 });
 let next=pad;
 const rows=entries.map(entry=>{const y=next;next+=entry.lines.length*lineHeight+gap;return {...entry,y};});
 const width=Math.min(plotWidth.value,Math.max(0,...entries.flatMap(e=>e.lines.map(t=>measure.measureText(t).width)))+48*s),height=next-gap+pad;
 const candidates=[{x:plotWidth.value-width-10*s,y:10*s},{x:10*s,y:10*s},{x:plotWidth.value-width-10*s,y:plotHeight.value-height-10*s},{x:10*s,y:plotHeight.value-height-10*s}];
 // Test complete line segments, including gaps, against each candidate's bounds.
 const free=(box:{x:number;y:number})=>chart.value.curves.every(curve=>{
  const axis=chart.value[curve.axis];let previous:{x:number;y:number}|undefined;
  for(const point of curve.points){if(point.value===null){previous=undefined;continue;}
   const p={x:(point.time-start.value)/span.value*plotWidth.value,y:plotHeight.value-(point.value-axis.min)/(axis.max-axis.min)*plotHeight.value};
   const a=previous??p;previous=p;
   if(Math.max(a.x,p.x)>=box.x-5*s&&Math.min(a.x,p.x)<=box.x+width+5*s&&Math.max(a.y,p.y)>=box.y-5*s&&Math.min(a.y,p.y)<=box.y+height+5*s)return false;
  }return true;
 });
 const inside=entries.length&&height<plotHeight.value*.5?candidates.find(c=>c.x>=0&&c.y>=0&&free(c)):undefined;
 return {rows,width,height:entries.length?height:0,lineHeight,inside};
});
const plotTop=computed(()=>(30+annotations.value.length*23)*fontScale.value+(!legend.value.inside&&legend.value.rows.length?legend.value.height+16*fontScale.value:0));
const plotBottom=computed(()=>plotTop.value+plotHeight.value);
const chartHeight=computed(()=>plotBottom.value+58*fontScale.value);
const legendPosition=computed(()=>({x:plotLeft.value+(legend.value.inside?.x??(plotWidth.value-legend.value.width)/2),y:legend.value.inside?plotTop.value+legend.value.inside.y:annotations.value.length*23*fontScale.value}));
const colorTokens=Array.from({length:8},(_,i)=>'--m02-curve-'+(i+1));
const styleFor=(name:string)=>{const index=Math.max(0,props.table.columns.indexOf(name)-1);return {color:'var('+colorTokens[index%colorTokens.length]+')',dash:index>=colorTokens.length?'3 3':undefined};};
const x=(t:number)=>plotLeft.value+(t-start.value)/span.value*plotWidth.value;
const scale=(curve:OverlayCurve)=>chart.value[curve.axis];
const y=(curve:OverlayCurve,value:number)=>plotBottom.value-(value-scale(curve).min)/(scale(curve).max-scale(curve).min)*plotHeight.value;
function path(curve:OverlayCurve){let pen=false;return curve.points.map(point=>{if(point.value===null){pen=false;return '';}const command=pen?'L':'M';pen=true;return command+x(point.time).toFixed(3)+' '+y(curve,point.value).toFixed(3);}).join(' ');}
const curvePaths=computed(()=>new Map(chart.value.curves.map(curve=>[curve.name,path(curve)])));
const printable=(text:string)=>text&&!['sil','<sil>','eps','nan'].includes(text.toLowerCase());
let drag:{x:number;offset:number;time:number;select:boolean;id:number}|null=null;
function time(event:PointerEvent){const rect=svg.value!.getBoundingClientRect();return Math.max(0,Math.min(duration.value,start.value+((event.clientX-rect.left)/rect.width*chartWidth.value-plotLeft.value)/plotWidth.value*span.value));}
function down(event:PointerEvent){if(event.button!==0)return;const rect=svg.value!.getBoundingClientRect(),px=(event.clientX-rect.left)/rect.width*chartWidth.value;if(px<plotLeft.value||px>plotRight.value)return;event.preventDefault();svg.value!.focus({preventScroll:true});drag={x:event.clientX,offset:start.value,time:time(event),select:!event.shiftKey,id:event.pointerId};svg.value!.setPointerCapture(event.pointerId);if(drag.select){props.wave.start=drag.time;props.wave.end=drag.time;}}
function move(event:PointerEvent){if(!drag||drag.id!==event.pointerId)return;if(drag.select){props.wave.start=Math.min(drag.time,time(event));props.wave.end=Math.max(drag.time,time(event));}else{const rect=svg.value!.getBoundingClientRect();const dx=(event.clientX-drag.x)/rect.width*chartWidth.value/plotWidth.value*span.value;props.wave.offset=Math.max(0,Math.min(duration.value-span.value,drag.offset-dx));}}
function wheel(event:WheelEvent){if(!event.ctrlKey||!event.deltaY)return;event.preventDefault();const rect=svg.value!.getBoundingClientRect();const fraction=Math.max(0,Math.min(1,((event.clientX-rect.left)/rect.width*chartWidth.value-plotLeft.value)/plotWidth.value));const anchor=start.value+fraction*span.value;props.wave.zoom=Math.max(1,Math.min(Math.max(1,duration.value/.01),props.wave.zoom*(event.deltaY<0?1.25:.8)));props.wave.offset=Math.max(0,Math.min(duration.value-duration.value/props.wave.zoom,anchor-fraction*duration.value/props.wave.zoom));}
async function save(){
 if(exporting.value||props.pending)return;exporting.value=true;error.value='';
 try{if(format.value==='png'){downloadImage(await currentFigurePng(svg.value!,props.group.title),props.group.title+'.png');return;}
  const clone=styledSvg(svg.value!);clone.style.width=chartWidth.value+'px';clone.style.height=chartHeight.value+'px';clone.setAttribute('width',String(chartWidth.value));clone.setAttribute('height',String(chartHeight.value));
  downloadImage(new Blob([await editableSvg(clone)],{type:'image/svg+xml;charset=utf-8'}),props.group.title+'.svg');
 }catch(e){error.value=e instanceof Error?e.message:'图片导出失败，请重试。';}finally{exporting.value=false;}
}
async function saveWhole(){
 if(exporting.value||props.pending)return;error.value='';exporting.value=true;
 try{const waveform=props.waveform();if(!svg.value||!waveform)throw Error('波形尚未就绪。');
  const name=(props.wave.asset?.name??'参数').replace(/\.wav$/i,'')+'-'+props.group.title;
  const blob=await wholeFigurePng({chart:svg.value,waveform,title:name,start:start.value,end:start.value+span.value,plotLeft:plotLeft.value,plotRight:plotRight.value});
  downloadImage(blob,name+'.png');
 }catch(e){error.value=e instanceof Error?e.message:'整幅图片导出失败，请重试。';}finally{exporting.value=false;}
}
</script>
<template><section class="parameter-figure" :class="{'is-active':active}" :data-group-id="group.id" :aria-label="group.title" :aria-busy="!!pending" @pointerdown.capture="emit('activate')" @focusin="emit('activate')">
<header class="section-title"><div class="figure-heading"><h3><button class="figure-title" :aria-label="'选择'+group.title" :aria-pressed="!!active" @click="emit('activate')">{{group.title}}<span v-if="active" class="active-indicator">当前</span></button></h3><small>{{pending?'正在读取当前时间窗…':chart.curves.length+' 条曲线'+(annotations.length?' · '+annotations.length+' 层标注':'')}}<template v-if="chart.curves.length"> · {{chart.dual?'双纵轴':'共用纵轴'}}</template></small></div><div class="figure-actions"><slot/><select v-model="format" :aria-label="group.title+'图片格式'" class="image-format"><option value="png">PNG</option><option value="svg">SVG</option></select><button @click="save" :disabled="!group.parameters.length||exporting||pending" :title="'仅保存参数图 '+format.toUpperCase()">保存当前图</button><button @click="saveWhole" :disabled="!group.parameters.length||exporting||pending" title="白底 300 dpi，包含波形、已开启语谱图及此参数图">{{exporting?'正在生成 PNG…':'保存整幅 PNG'}}</button></div></header>
<p v-if="error" role="alert">{{error}}</p><div ref="surface" class="plot-surface">
<div v-if="!group.parameters.length" class="empty-plot"><strong>在右侧勾选参数</strong><p>{{active?'勾选后，曲线立即显示在这里。':'点击此图窗，再在右侧勾选要显示的参数。'}}</p></div>
<svg v-else ref="svg" class="parameter-chart" tabindex="0" :viewBox="'0 0 '+chartWidth+' '+chartHeight" :data-plot-left="plotLeft" :data-plot-right="plotRight" :style="{height:chartHeight+'px',minWidth:Math.ceil(340*fontScale)+'px'}" role="img" :aria-label="group.title+'叠加参数曲线'" @pointerdown="down" @pointermove="move" @pointerup="move($event);drag=null" @pointercancel="drag=null" @wheel="wheel" @dblclick="wave.zoom=1;wave.offset=0">
<rect :width="chartWidth" :height="chartHeight" fill="var(--panel)"/>
<defs><clipPath :id="clipId"><rect :x="plotLeft" :y="plotTop" :width="plotWidth" :height="plotHeight"/></clipPath></defs>
<g class="shared-plot-area" :data-plot-top="plotTop" :data-plot-bottom="plotBottom">
<rect :x="Math.max(plotLeft,Math.min(plotRight,x(wave.start)))" :y="plotTop" :width="Math.max(0,Math.min(plotRight,x(wave.end))-Math.max(plotLeft,x(wave.start)))" :height="plotHeight" fill="var(--selection)" opacity=".5"/>
<g v-for="i in 5" :key="i">
<path :d="'M'+plotLeft+' '+(plotTop+(i-1)*plotHeight/4)+'H'+plotRight" stroke="var(--border)" opacity=".45" vector-effect="non-scaling-stroke"/>
<text class="left-tick" :x="plotLeft-9*fontScale" :y="plotTop+(i-1)*plotHeight/4+4*fontScale" text-anchor="end" fill="var(--muted)" font-size="12">{{formatPlotTick(chart.left.max-(i-1)*(chart.left.max-chart.left.min)/4,(chart.left.max-chart.left.min)/4)}}</text>
<text v-if="chart.dual" class="right-tick" :x="plotRight+9*fontScale" :y="plotTop+(i-1)*plotHeight/4+4*fontScale" fill="var(--muted)" font-size="12">{{formatPlotTick(chart.right.max-(i-1)*(chart.right.max-chart.right.min)/4,(chart.right.max-chart.right.min)/4)}}</text>
<text :x="plotLeft+(i-1)*plotWidth/4" :y="plotBottom+22*fontScale" text-anchor="middle" font-size="12" fill="var(--muted)">{{(start+(i-1)*span/4).toFixed(3)}}</text>
</g>
<path :d="'M'+plotLeft+' '+plotTop+'V'+plotBottom+'H'+plotRight+(chart.dual?'V'+plotTop:'')" stroke="var(--muted)" stroke-width=".8" fill="none" vector-effect="non-scaling-stroke"/>
<text :x="plotLeft" :y="plotTop-11*fontScale" fill="var(--muted)" font-size="12">{{chart.dual?'左轴 · 原数值':'原数值'}}</text>
<text v-if="chart.dual" :x="plotRight" :y="plotTop-11*fontScale" text-anchor="end" fill="var(--muted)" font-size="12">右轴 · 原数值</text>
<g :clip-path="'url(#'+clipId+')'"><path v-for="curve in chart.curves" :key="curve.name" class="parameter-curve" :data-parameter="curve.name" :data-axis="curve.axis" :data-scale-min="scale(curve).min" :data-scale-max="scale(curve).max" :d="curvePaths.get(curve.name)" :stroke="styleFor(curve.name).color" :stroke-dasharray="curve.axis==='right'?'7 3':styleFor(curve.name).dash" stroke-width="1.65" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke" fill="none"><title>{{curve.name}} · {{curve.axis==='left'?'左轴':'右轴'}}</title></path></g>
<g v-for="(tier,index) in annotations" :key="tier.name"><g v-for="run in tier.runs" :key="run.start"><path :d="'M'+x(run.start)+' '+((index*23+5)*fontScale)+'V'+plotBottom" :stroke="styleFor(tier.name).color" stroke-dasharray="2 4" opacity=".65"/><text class="ipa-text" v-if="printable(run.text)" :x="x((run.start+run.end)/2)" :y="(index*23+16)*fontScale" text-anchor="middle" font-size="13" fill="var(--text)"><title>{{tier.name}}</title>{{run.text}}</text></g></g>
</g>
<text :x="plotLeft+plotWidth/2" :y="plotBottom+44*fontScale" text-anchor="middle" fill="var(--text)" font-size="14">时间（秒）</text>
<g v-if="legend.rows.length" class="plot-legend" :data-placement="legend.inside?'inside':'above'" :transform="'translate('+legendPosition.x+' '+legendPosition.y+')'" :data-width="legend.width" :data-height="legend.height">
<rect :width="legend.width" :height="legend.height" fill="var(--panel)" opacity=".94"/>
<g v-for="row in legend.rows" :key="'legend-'+row.curve.name" class="curve-legend" :data-parameter="row.curve.name" :transform="'translate('+(8*fontScale)+' '+(row.y+13*fontScale)+')'"><title>{{row.curve.name}}</title><path :d="'M0 '+(-4*fontScale)+'H'+(24*fontScale)" :stroke="styleFor(row.curve.name).color" :stroke-dasharray="row.curve.axis==='right'?'7 3':styleFor(row.curve.name).dash" stroke-width="1.65" vector-effect="non-scaling-stroke"/><text v-for="(line,index) in row.lines" :key="index" :x="32*fontScale" :y="index*legend.lineHeight" font-size="13" fill="var(--text)">{{line}}</text></g>
</g>
</svg></div><p v-if="group.parameters.length" class="plot-help">拖动选区 · Shift＋拖动平移 · Ctrl＋滚轮缩放时间轴 · 双击全长<span v-if="chart.dual">。双轴沿用 v2 的量级判定；图例标明所属纵轴，原数值不归一化。</span></p></section></template>
<style scoped>
.image-format{width:auto;max-width:90px}
.parameter-figure{--m02-curve-1:#3567a8;--m02-curve-2:#b86b32;--m02-curve-3:#26877f;--m02-curve-4:#8965aa;--m02-curve-5:#b65576;--m02-curve-6:#73833e;--m02-curve-7:#64748b;--m02-curve-8:#357e96;transition:border-color .15s}
:global([data-theme=dark] .parameter-figure){--m02-curve-1:#86afe4;--m02-curve-2:#e5ac7b;--m02-curve-3:#78c5b8;--m02-curve-4:#bda2d5;--m02-curve-5:#e59cb6;--m02-curve-6:#b9c77f;--m02-curve-7:#adb9cb;--m02-curve-8:#83c5dc}
.parameter-figure.is-active{border-color:var(--accent);box-shadow:inset 3px 0 0 var(--accent)}
.figure-title{display:flex;align-items:center;gap:8px;border:0;background:transparent;color:var(--text);padding:2px 0;font:inherit;font-weight:600;cursor:pointer}.figure-title:hover{color:var(--accent)}.active-indicator{font-size:11px;font-weight:400;color:var(--accent);padding:2px 6px;border-radius:4px;background:var(--selection)}
.section-title>.figure-heading{display:grid;gap:3px}.figure-actions button,.figure-actions select{font-size:.857143rem}.figure-actions{justify-content:flex-end}
@media(prefers-reduced-motion:reduce){.parameter-figure{transition:none}}
.parameter-figure{border:1px solid var(--border);border-radius:8px;background:var(--panel);padding:10px;margin:10px 0;min-width:0}.parameter-figure>.section-title{flex-wrap:wrap;margin-bottom:8px}.section-title>div{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.section-title h3{margin:0;font-size:1.071429rem}.section-title small{color:var(--muted);font-size:0.857143rem}.plot-surface{width:100%;min-width:0;overflow-x:auto}.parameter-chart{display:block;width:100%;min-width:340px;max-width:none;background:var(--panel);touch-action:none;user-select:none;font-family:var(--font-figure);cursor:crosshair}.empty-plot{min-height:180px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:24px;color:var(--muted);border:1px dashed var(--border);border-radius:5px;background:var(--app)}.empty-plot strong{font-size:1.142857rem;color:var(--text)}.empty-plot p{max-width:340px;line-height:1.7}.plot-help{font-size:0.857143rem;color:var(--muted);line-height:1.6;margin:10px 0 0}
</style>
