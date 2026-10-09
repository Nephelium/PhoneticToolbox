<script setup lang="ts">
import {computed,ref,watch,onUnmounted,useId} from 'vue';
import {styledSvg,paintSvg} from '../../design/svg-fonts.ts';
import {withFigureTitle} from '../../design/figure-title.ts';
import {exportSize,png300dpi} from '../../design/png.ts';
import {plotTickLabel} from '../../platform/plotTicks.ts';
import type {Points} from './state.ts';
import type {F0Track} from './f0.ts';
const props=defineProps<{points:Points;alignment:'normalize'|'onset';tracks?:F0Track[];note?:string;minimum?:number;maximum?:number|null;activeStep?:string}>();
const svg=ref<SVGSVGElement>(),surface=ref<HTMLDivElement>(),width=ref(600),height=ref(260);let observer:ResizeObserver|undefined;
const clipId='m07-f0-'+useId();
watch(surface,el=>{observer?.disconnect();if(el){observer=new ResizeObserver(entries=>{width.value=Math.max(180,entries[0].contentRect.width);height.value=Math.max(140,entries[0].contentRect.height);});observer.observe(el);}});
onUnmounted(()=>observer?.disconnect());
const curves=computed(()=>[
 {name:'源 F0',axis:props.points.axis,values:props.points.source,color:'var(--accent)',dash:'',role:'source'},
 {name:'目标 F0',axis:props.points.axis,values:props.points.target,color:'var(--text)',dash:'7 4',role:'target'},
 ...(props.tracks??[]).map(track=>({...track,dash:track.dash??'',role:'synthesis'}))
]);
const minimum=computed(()=>props.minimum??0);
const maximum=computed(()=>props.maximum??Math.max(minimum.value+50,Math.ceil(Math.max(0,...curves.value.flatMap(c=>c.values.filter(Number.isFinite)))/50)*50));
const end=computed(()=>Math.max(1,...curves.value.flatMap(c=>c.axis)));
const hasData=computed(()=>curves.value.some(curve=>curve.values.some(value=>Number.isFinite(value)&&value>0)));
const orderedCurves=computed(()=>[...curves.value].sort((a,b)=>Number(a.name===props.activeStep)-Number(b.name===props.activeStep)));
function path(curve:{axis:number[];values:number[]}){
 let pen=false;return curve.values.map((value,i)=>{if(!Number.isFinite(value)||value<=0||!Number.isFinite(curve.axis[i])){pen=false;return '';}const command=pen?'L':'M';pen=true;return `${command}${64+curve.axis[i]/end.value*(width.value-86)},${height.value-44-(value-minimum.value)/(maximum.value-minimum.value)*(height.value-76)}`;}).join(' ');
}
async function png(){
 if(!svg.value||!hasData.value)throw Error('请先提取 F0 或选择合成组');
 // Capture the live range and paths before any asynchronous font/raster work.
 const clone=styledSvg(svg.value),font=getComputedStyle(svg.value),fontSize=parseFloat(font.fontSize)||12;
 clone.style.fontFamily=font.fontFamily;
 clone.querySelectorAll<SVGElement>('text').forEach(text=>text.style.fill='#182332');
 clone.querySelectorAll<SVGElement>('.plot-frame').forEach(frame=>{frame.style.fill='white';frame.style.stroke='#b8c1cc';});
 clone.querySelectorAll<SVGElement>('.axis').forEach(axis=>axis.style.stroke='#b8c1cc');
 clone.querySelectorAll<SVGElement>('[data-curve]').forEach(line=>{line.style.opacity='1';line.style.strokeWidth='2';if(line.dataset.curve==='target')line.style.stroke='#354052';});
 const ns='http://www.w3.org/2000/svg',measure=document.createElement('canvas').getContext('2d');
 if(!measure)throw Error('图像绘制环境不可用');
 measure.font=`${fontSize}px ${font.fontFamily}`;
 let x=14,y=height.value+fontSize+6;
 for(const curve of curves.value){
  const span=36+measure.measureText(curve.name).width+16;
  if(x+span>width.value-10&&x>14){x=14;y+=fontSize+12;}
  const line=document.createElementNS(ns,'path');line.setAttribute('d',`M${x},${y-fontSize/3} h22`);line.style.cssText=`fill:none;stroke-width:2;stroke-dasharray:${curve.dash||'none'}`;
  const original=svg.value.querySelector<SVGElement>(`[data-name="${curve.name}"]`);
  line.style.stroke=curve.role==='target'?'#354052':original?getComputedStyle(original).stroke:curve.color;clone.append(line);
  const text=document.createElementNS(ns,'text');text.textContent=curve.name;text.setAttribute('x',String(x+28));text.setAttribute('y',String(y));text.style.cssText=`fill:#182332;font-size:${fontSize}px;font-family:${font.fontFamily}`;clone.append(text);x+=span;
 }
 const exportHeight=y+14;clone.setAttribute('viewBox',`0 0 ${width.value} ${exportHeight}`);
 const titled=withFigureTitle(clone,'F0 曲线对比',fontSize),canvas=document.createElement('canvas'),size=exportSize(titled.width,titled.height);
 canvas.width=size.width;canvas.height=size.height;await paintSvg(canvas,new XMLSerializer().serializeToString(titled.root),titled.width,titled.height);
 const blob=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(value=>value?resolve(value):reject(Error('图片编码失败')),'image/png'));
 return new Blob([png300dpi(new Uint8Array(await blob.arrayBuffer()))],{type:'image/png'});
}
defineExpose({png});
</script>
<template>
 <div class="f0-comparison">
  <div ref="surface" class="f0-surface"><svg ref="svg" :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="源、目标与试听音频 F0 对照" class="f0-plot" :data-y-min="minimum" :data-y-max="maximum" :data-active-step="activeStep||''">
   <defs><clipPath :id="clipId"><rect x="64" y="32" :width="width-86" :height="height-76"/></clipPath></defs>
   <rect x="64" :y="hasData?32:0" :width="width-(hasData?86:64)" :height="height-(hasData?76:26)" rx="4" class="plot-frame"/>
   <template v-if="hasData">
   <path :d="`M64,32 V${height-44} H${width-22}`" class="axis"/>
   <text x="12" y="18">F0 (Hz)</text>
   <text :x="width/2" :y="height-3" text-anchor="middle">{{alignment==='normalize'?'归一化有声时间 (%)':'有声起点后时间 (ms)'}}</text>
   <g v-for="tick in 5" :key="tick">
    <text x="57" :y="height-39-(tick-1)*(height-76)/4" text-anchor="end">{{plotTickLabel(minimum+(tick-1)*(maximum-minimum)/4)}}</text>
    <text :x="64+(tick-1)*(width-86)/4" :y="height-24" text-anchor="middle">{{plotTickLabel((tick-1)*end/4)}}</text>
   </g>
   <g :clip-path="`url(#${clipId})`"><path v-for="curve in orderedCurves" :key="curve.name" :d="path(curve)" :data-curve="curve.role" :data-name="curve.name" :data-axis-end="curve.axis.at(-1)" :data-active="curve.name===activeStep" :style="{stroke:curve.color,strokeDasharray:curve.dash,strokeWidth:curve.name===activeStep?4:2,opacity:activeStep&&curve.role==='synthesis'&&curve.name!==activeStep ? 0.55 : 1}" class="f0-line"><title>{{curve.name}}</title></path></g>
   </template>
   <template v-else><text v-for="i in 5" :key="i" :x="64+(i-1)*(width-64)/4" :y="height-5" text-anchor="middle" class="empty-label">—</text></template>
  </svg><div v-if="!hasData" class="f0-empty"><p>提取 F0 后显示源与目标基频曲线</p></div></div>
  <div class="f0-legend" aria-label="F0 图例"><span v-for="curve in curves" :key="curve.name" :data-name="curve.name" :data-active="curve.name===activeStep"><svg width="22" height="12" aria-hidden="true"><path d="M0,6 H22" :style="{stroke:curve.color,strokeDasharray:curve.dash}" class="f0-line"/></svg>{{curve.name}}</span></div>
  <small v-if="note" class="f0-note">{{note}}</small>
 </div>
</template>
<style scoped>
.f0-comparison{display:flex;flex-direction:column;gap:6px;flex:1;min-height:0}.f0-surface{position:relative;flex:1;min-height:140px}.f0-plot{position:absolute;inset:0;display:block;width:100%;height:100%;font:var(--figure-size) var(--font-figure)}.f0-plot text{fill:var(--text);font:inherit}.plot-frame{fill:var(--app);stroke:var(--border);stroke-width:1}.f0-empty{position:absolute;inset:0 0 26px 64px;display:grid;place-items:center;padding:0 10px;text-align:center;color:var(--muted);font-size:var(--support-size);pointer-events:none}.f0-empty p{line-height:1.5}.f0-plot .empty-label{fill:var(--muted);font-size:var(--support-size)}.axis{stroke:var(--border);fill:none}.f0-line{fill:none;stroke-width:2;vector-effect:non-scaling-stroke}.f0-legend{display:flex;flex-wrap:wrap;gap:2px 10px;font-size:var(--support-size);max-height:54px;overflow:auto;flex-shrink:0}.f0-legend span{display:inline-flex;align-items:center;gap:5px;padding:1px 3px;border-radius:3px}.f0-legend span[data-active=true]{font-weight:700;background:var(--selected);outline:1px solid var(--accent)}.f0-legend svg{flex-shrink:0}.f0-note{color:var(--muted);font-size:var(--support-size)}
</style>
