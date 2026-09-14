<script setup lang="ts">
import {ref,watch,onMounted,onUnmounted,nextTick} from 'vue';
import type {Editor} from './editor.mjs';import type {Workspace} from '../../state/workspace.ts';import type {LipTrack} from '../../platform/annotation.ts';
import {playback} from '../../state/audio.ts';import {spectrum} from './spectrum.ts';
import {lipCurve} from './display.ts';
const props=defineProps<{editor:Editor;revision:number;wave:Workspace;lip:LipTrack|null;offset:number;showOpen:boolean;showWidth:boolean;active:boolean}>();
const emit=defineEmits<{changed:[]}>();
const grid=ref<HTMLCanvasElement>(),spec=ref<HTMLCanvasElement>(),container=ref<HTMLElement>();let observer:ResizeObserver,themeObserver:MutationObserver,frame=0,cacheKey='',image:HTMLCanvasElement|undefined,oldAsset:unknown;
const css=(name:string)=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function resize(canvas:HTMLCanvasElement){const rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;const w=Math.max(1,Math.round(rect.width*dpr)),h=Math.max(1,Math.round(rect.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}return dpr;}
function draw(){
 if(!grid.value||!spec.value||!props.active)return;
 const canvas=grid.value,dpr=resize(canvas),ctx=canvas.getContext('2d')!,e=props.editor,s=e.state;
 e.setCanvas(canvas);s.visibleStart=props.wave.offset;s.visibleDuration=(props.wave.asset?.duration??1)/props.wave.zoom;
 const x=(time:number)=>(time-s.visibleStart)/s.visibleDuration*canvas.width;
 ctx.fillStyle=css('--panel');ctx.fillRect(0,0,canvas.width,canvas.height);
 const fontSize=parseFloat(css('--figure-size'))||14;ctx.font=`${fontSize*dpr}px ${css('--font-figure-ipa')||'"PTB-Doulos",serif'}`;ctx.textAlign='center';ctx.textBaseline='middle';
 for(const [row,name] of [s.wordTierName,s.phoneTierName].entries()){
  const top=row===0?0:canvas.height*.48,bottom=row===0?canvas.height*.48:canvas.height,tier=e.tierByName(name);
  ctx.strokeStyle=css('--border');ctx.strokeRect(0,top,canvas.width,bottom-top);
  if(!tier){ctx.fillStyle=css('--muted');ctx.fillText('需要覆盖音频全时域的区间层：'+name,canvas.width/2,(top+bottom)/2);continue;}
  for(const [index,item] of tier.intervals.entries()){
   if(item.xmax<s.visibleStart||item.xmin>s.visibleStart+s.visibleDuration)continue;
   const a=Math.max(0,x(item.xmin)),b=Math.min(canvas.width,x(item.xmax));
   const selected=(s.selected?.tier===name&&s.selected.index===index)||(row===0&&s.selectedIndices.includes(index));
   if(selected||row===0&&s.labWords.has(item.text.toLowerCase())){ctx.fillStyle=css(selected?'--selected':'--teal-bg');ctx.fillRect(a,top,b-a,bottom-top);}
   ctx.strokeStyle=css('--teal');ctx.lineWidth=dpr;ctx.beginPath();ctx.moveTo(a,top);ctx.lineTo(a,bottom);ctx.moveTo(b,top);ctx.lineTo(b,bottom);ctx.stroke();
   ctx.save();ctx.beginPath();ctx.rect(a+2*dpr,top,Math.max(0,b-a-4*dpr),bottom-top);ctx.clip();ctx.fillStyle=css('--text');ctx.fillText(item.text,(a+b)/2,(top+bottom)/2);ctx.restore();
  }
  if(s.selectedBoundary?.tier===name){ctx.strokeStyle=css('--danger');ctx.setLineDash([4*dpr,3*dpr]);ctx.beginPath();ctx.moveTo(x(s.selectedBoundary.time),top);ctx.lineTo(x(s.selectedBoundary.time),bottom);ctx.stroke();ctx.setLineDash([]);}
 }
 if(s.drag?.mode==='rangeSelect'){const a=x(Math.min(s.drag.startTime,s.drag.endTime)),b=x(Math.max(s.drag.startTime,s.drag.endTime));ctx.fillStyle=css('--selection');ctx.fillRect(a,0,b-a,canvas.height*.48);}
 const sc=spec.value;resize(sc);const sx=sc.getContext('2d')!,asset=props.wave.asset;
 if(!asset){sx.clearRect(0,0,sc.width,sc.height);return;}
 const dark=document.documentElement.dataset.theme==='dark',cols=Math.min(700,sc.width),height=Math.min(360,sc.height);
 const key=[s.visibleStart,s.visibleDuration,cols,height,dark].join(':');
 if(key!==cacheKey||oldAsset!==asset||!image){
  const data=spectrum(asset.channels[0],asset.sampleRate,s.visibleStart,s.visibleDuration,cols,height,dark,Math.min(900,sc.width));
  image=document.createElement('canvas');image.width=cols;image.height=height;const c=image.getContext('2d')!;c.putImageData(new ImageData(data.rgba,cols,height),0,0);
  c.strokeStyle=css('--success');c.lineWidth=1.6;c.beginPath();data.intensity.forEach((v,i)=>{const xx=i/Math.max(1,data.intensity.length-1)*cols,yy=(1-v)*height;i===0?c.moveTo(xx,yy):c.lineTo(xx,yy);});c.stroke();cacheKey=key;oldAsset=asset;
 }
 sx.drawImage(image,0,0,sc.width,sc.height);
 const plotLip=(values:(number|null)[],color:string)=>{
  if(!props.lip)return;sx.strokeStyle=color;sx.lineWidth=1.8*dpr;sx.beginPath();
  for(const p of lipCurve(props.lip.times,values,props.offset,s.visibleStart,s.visibleDuration,sc.width,sc.height))p.move?sx.moveTo(p.x,p.y):sx.lineTo(p.x,p.y);sx.stroke();
 };
 if(props.showOpen)plotLip(props.lip?.open??[],css('--danger'));if(props.showWidth)plotLip(props.lip?.width??[],css('--wave'));
 for(const tier of [e.wordTier(),e.phoneTier()])for(const item of tier?.intervals??[]){const px=(item.xmin-s.visibleStart)/s.visibleDuration*sc.width;if(px>0&&px<sc.width){sx.strokeStyle=css('--teal');sx.globalAlpha=.35;sx.beginPath();sx.moveTo(px,0);sx.lineTo(px,sc.height);sx.stroke();sx.globalAlpha=1;}}
}
function schedule(){cancelAnimationFrame(frame);frame=requestAnimationFrame(draw);}
function down(event:PointerEvent){if(event.button!==0)return;props.editor.onGridMouseDown(event);grid.value?.setPointerCapture(event.pointerId);grid.value?.focus();emit('changed');}
function move(event:PointerEvent){props.editor.onGridMouseMove(event);if(props.editor.state.drag)emit('changed');}
function up(){if(props.editor.state.drag){props.editor.onGridMouseUp();emit('changed');}}
function double(event:MouseEvent){const hit=props.editor.hitTest(event);if(!hit)return;if(hit.tier===props.editor.state.phoneTierName){props.editor.saveUndoState();props.editor.splitPhoneAt(hit.time);}else{props.editor.state.selected={tier:hit.tier,index:hit.index};props.editor.autoPhonesForSelection();}emit('changed');}
function wheel(event:WheelEvent){if(!event.ctrlKey&&!event.shiftKey)return;event.preventDefault();const wave=props.wave,duration=wave.asset?.duration??0;if(!duration)return;const length=duration/wave.zoom,rect=(event.currentTarget as HTMLElement).getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width));if(event.ctrlKey){const next=Math.max(.08,Math.min(duration,length*(event.deltaY>0?1.18:.85))),anchor=wave.offset+fraction*length;wave.zoom=duration/next;wave.offset=Math.max(0,Math.min(duration-next,anchor-fraction*next));}else wave.offset=Math.max(0,Math.min(duration-length,wave.offset+(event.deltaY||event.deltaX)*length*.0015));schedule();}
watch(()=>[props.revision,props.wave.offset,props.wave.zoom,props.offset,props.showOpen,props.showWidth,props.active],()=>void nextTick(schedule));
onMounted(()=>{observer=new ResizeObserver(schedule);if(container.value)observer.observe(container.value);themeObserver=new MutationObserver(()=>{cacheKey='';schedule();});themeObserver.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme','style']});window.addEventListener('ptb-fonts-changed',schedule);void document.fonts.ready.then(schedule);schedule();});
onUnmounted(()=>{observer?.disconnect();themeObserver?.disconnect();cancelAnimationFrame(frame);window.removeEventListener('ptb-fonts-changed',schedule);});
</script>
<template>
<div ref="container" class="annotation-tracks" @wheel="wheel">
 <div class="track-caption"><strong>语谱图</strong><span>0–{{Math.min(5000,(wave.asset?.sampleRate??10000)/2)}} Hz · Hann 1024 · 相对幅度</span><span class="intensity-key">强度</span><span v-if="lip&&showOpen" class="open-key">唇开</span><span v-if="lip&&showWidth" class="width-key">唇宽</span></div>
 <div class="spectrum-wrap"><canvas ref="spec" aria-label="标注语谱图与唇形曲线"/><div v-if="playback.playing" class="annotation-cursor" :style="{left:((playback.position-wave.offset)/((wave.asset?.duration??1)/wave.zoom)*100)+'%'}"/></div>
 <div class="track-caption"><strong>标注层</strong><span>{{editor.state.wordTierName}} / {{editor.state.phoneTierName}}</span><small>单击编辑 · 拖动边界/词 · Ctrl 多选 · 双击填充/插点</small></div>
 <canvas ref="grid" class="annotation-grid" tabindex="0" aria-label="词层与音素层编辑器" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="up" @dblclick="double"/>
</div>
</template>
<style scoped>
.annotation-tracks{min-width:0}.track-caption{display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--muted);padding:6px 0}.track-caption strong{color:var(--text)}canvas{display:block;width:100%;height:175px}.annotation-grid{height:150px;touch-action:none;cursor:crosshair;border:1px solid var(--border);border-radius:4px}.spectrum-wrap{position:relative;overflow:hidden;border:1px solid var(--border);border-radius:4px}.annotation-cursor{position:absolute;top:0;bottom:0;width:1px;background:var(--danger);pointer-events:none}.intensity-key{color:var(--success)}.open-key{color:var(--danger)}.width-key{color:var(--wave)}
</style>
