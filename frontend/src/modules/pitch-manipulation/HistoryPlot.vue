<script setup lang="ts">
import {computed,ref,onMounted,onUnmounted} from 'vue';import type {Result} from './port.ts';
import {styledSvg,paintSvg} from '../../design/svg-fonts.ts';
const props=defineProps<{results:Result[];min:number;max:number}>();const svg=ref<SVGSVGElement>();
const width=ref(650),right=computed(()=>width.value-20),plotWidth=computed(()=>width.value-80);let observer:ResizeObserver;
onMounted(()=>{observer=new ResizeObserver(e=>width.value=Math.max(240,e[0].contentRect.width));if(svg.value)observer.observe(svg.value);});onUnmounted(()=>observer?.disconnect());
const duration=computed(()=>Math.max(.01,...props.results.flatMap(r=>r.times)));
const palette=['var(--accent)','var(--danger)','var(--teal)','var(--violet)'];
function path(r:Result){let pen=false;return r.times.map((t,i)=>{const f=r.original_f0[i];if(!(f>0)){pen=false;return '';}const c=pen?'L':'M';pen=true;return `${c}${60+t/duration.value*plotWidth.value},${200-(f-props.min)/(props.max-props.min)*175}`;}).join(' ');}
async function png(){
 if(!svg.value||!props.results.length)throw Error('尚无已保存历史结果');
 const clone=styledSvg(svg.value),ns='http://www.w3.org/2000/svg',style=getComputedStyle(svg.value);
 clone.setAttribute('xmlns',ns);clone.style.background=style.backgroundColor;
 const outputWidth=width.value,fontSize=parseFloat(style.fontSize)||12,chars=Math.max(12,Math.floor((outputWidth-90)/(fontSize*.65)));let y=280;
 props.results.forEach((r,i)=>{const name=`${i+1} · ${r.name}`;for(let a=0;a<name.length;a+=chars){const text=document.createElementNS(ns,'text');text.setAttribute('x','60');text.setAttribute('y',String(y));text.style.fill=style.getPropertyValue(['--accent','--danger','--teal','--violet'][i%4]);text.style.fontFamily=style.fontFamily;text.style.fontSize=style.fontSize;text.textContent=name.slice(a,a+chars);clone.append(text);y+=fontSize*1.5;}});
 const height=y+12;clone.setAttribute('viewBox',`0 0 ${outputWidth} ${height}`);clone.setAttribute('width',String(outputWidth));clone.setAttribute('height',String(height));
 const canvas=document.createElement('canvas');canvas.width=Math.round(outputWidth*1.5);canvas.height=Math.round(height*1.5);
 await paintSvg(canvas,new XMLSerializer().serializeToString(clone),outputWidth,height);
 return await new Promise<Blob>((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(Error('PNG 导出失败')),'image/png'));
}
defineExpose({png});
</script>
<template><svg ref="svg" class="history-plot" :viewBox="`0 0 ${width} 260`" role="img" aria-label="已保存音频的实际 F0 历史对比，零起点对齐"><path :d="`M60 25V200H${right}`" class="axis"/><text x="60" y="16">F0 (Hz)</text><text :x="right" y="235" text-anchor="end">Relative time (s)</text><g v-for="n in 5" :key="n"><text x="54" :y="205-(n-1)*43.75" text-anchor="end">{{Math.round(min+(n-1)*(max-min)/4)}}</text><text :x="60+(n-1)*plotWidth/4" y="220" text-anchor="middle">{{((n-1)*duration/4).toFixed(2)}}</text></g><svg x="60" y="25" :width="plotWidth" height="175" :viewBox="`60 25 ${plotWidth} 175`" overflow="hidden"><path v-for="(r,i) in results" :key="r.id" :d="path(r)" fill="none" :stroke="palette[i%4]" :stroke-dasharray="i<4?'none':i<8?'6 3':'2 3'" stroke-width="1.5"/></svg><text x="60" y="254">{{results.length}} 条已保存音频</text></svg></template>
<style scoped>.history-plot{display:block;width:100%;height:auto;min-height:180px;background:var(--panel);fill:var(--text);font-family:var(--font-figure);font-size:var(--figure-size)}.axis{fill:none;stroke:var(--border)}</style>
