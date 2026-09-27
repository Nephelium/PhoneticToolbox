<script setup lang="ts">
import {ref,watch,nextTick,onUnmounted,computed} from 'vue';
import type {Spectrogram,SpectrogramView} from '../platform/research.ts';
const props=defineProps<{loader:(view:SpectrogramView)=>Promise<Spectrogram>;start:number;end:number;channel:number;width:number;selectable?:boolean;selectionStart?:number;selectionEnd?:number}>();
const emit=defineEmits<{'selection-end':[start:number,end:number]}>();
const canvas=ref<HTMLCanvasElement>(),data=ref<Spectrogram|null>(null),error=ref(''),loading=ref(false);
const dragging=ref(false),dragStart=ref(0),dragEnd=ref(0);let anchorX=0;
function pointerTime(event:PointerEvent){const box=(event.currentTarget as HTMLCanvasElement).getBoundingClientRect();return props.start+Math.max(0,Math.min(1,(event.clientX-box.left)/box.width))*(props.end-props.start);}
function down(event:PointerEvent){if(!props.selectable||event.button!==0)return;dragging.value=true;anchorX=event.clientX;dragStart.value=dragEnd.value=pointerTime(event);(event.currentTarget as HTMLCanvasElement).setPointerCapture(event.pointerId);}
function move(event:PointerEvent){if(dragging.value)dragEnd.value=pointerTime(event);}
function up(event:PointerEvent){if(!dragging.value)return;dragEnd.value=pointerTime(event);dragging.value=false;if(Math.abs(event.clientX-anchorX)<3)return;emit('selection-end',Math.min(dragStart.value,dragEnd.value),Math.max(dragStart.value,dragEnd.value));}
function cancel(){dragging.value=false;}
const selected=computed(()=>{const a=dragging.value?dragStart.value:props.selectionStart,b=dragging.value?dragEnd.value:props.selectionEnd;if(a===undefined||b===undefined||b===a||props.end<=props.start)return null;const left=Math.max(props.start,Math.min(a,b)),right=Math.min(props.end,Math.max(a,b));return right>left?{left:(left-props.start)/(props.end-props.start)*100,width:(right-left)/(props.end-props.start)*100}:null;});
let version=0,running=false,again=false,disposed=false,attempt=0,timer:ReturnType<typeof setTimeout>;
const messages:Record<string,string>={preview_busy:'另一个语谱图正在计算，请稍后重试。',preview_runtime_unavailable:'当前环境没有Praat科学运行库，请使用完整桌面开发入口。',preview_timeout:'语谱图计算超时，请缩小时间窗。',preview_memory_exceeded:'语谱图超过预览内存预算，请选择较短音频。',invalid_spectrogram_input:'当前音频或时间窗无法生成语谱图（至少需要10 ms）。',preview_failed:'语谱图读取失败，请检查音频或重试。'};
function paint(value:Spectrogram){
 const target=canvas.value;if(!target)return;const ctx=target.getContext('2d');if(!ctx)return;
 target.width=Math.max(100,props.width);target.height=220;ctx.fillStyle='#fff';ctx.fillRect(0,0,target.width,target.height);
 const raw=atob(value.pixels_base64);if(raw.length!==value.width*value.height)throw Error('语谱图像素长度不一致。');
 const source=document.createElement('canvas');source.width=value.width;source.height=value.height;const sc=source.getContext('2d')!;
 const pixels=sc.createImageData(value.width,value.height);
 for(let y=0;y<value.height;y++)for(let x=0;x<value.width;x++){const p=((value.height-1-y)*value.width+x)*4,v=raw.charCodeAt(y*value.width+x);pixels.data[p]=pixels.data[p+1]=pixels.data[p+2]=v;pixels.data[p+3]=255;}
 sc.putImageData(pixels,0,0);ctx.imageSmoothingEnabled=false;
 const span=props.end-props.start;
 ctx.drawImage(source,(value.x1-value.dx/2-props.start)/span*target.width,(value.frequency_max-value.y1-value.dy*(value.height-.5))/value.frequency_max*target.height,value.dx*value.width/span*target.width,value.dy*value.height/value.frequency_max*target.height);
}
async function run(){
 if(disposed)return;if(running){again=true;return;}running=true;again=false;const current=version;let retrying=false;loading.value=true;error.value='';
 try{const result=await props.loader({channel:props.channel,start:props.start,end:props.end,width:Math.max(100,Math.min(1000,Math.round(props.width)))});
  if(disposed||version!==current)return;data.value=result;await nextTick();paint(result);
 }catch(e){if(!disposed&&current===version){const text=e instanceof Error?e.message:'preview_failed';if(text==='preview_busy'&&attempt<8){retrying=true;timer=setTimeout(run,500+attempt++*250);}else error.value=messages[text]??text;}}
 finally{running=false;if(!disposed){if(current===version)loading.value=retrying;if(again)void run();}}
}
function schedule(){version++;attempt=0;data.value=null;error.value='';clearTimeout(timer);loading.value=true;timer=setTimeout(run,250);}
watch(()=>[props.loader,props.start,props.end,props.channel,props.width],schedule,{immediate:true});
onUnmounted(()=>{disposed=true;version++;clearTimeout(timer);});
</script>
<template><section class="spectrogram-view" aria-label="Praat语谱图">
<div class="track-label"><span>语谱图 · 声道 {{channel+1}}</span><small>Praat · Gaussian 5 ms · 50 dB</small></div>
<p v-if="loading" role="status" class="spectrogram-empty">正在计算当前时间窗的语谱图…</p>
<p v-else-if="error" role="alert" class="error-banner">{{error}} <button @click="schedule">重试</button></p>
<div v-if="data" class="spectrogram-canvas"><canvas ref="canvas" role="img" :class="{'selectable-spectrogram':selectable}" :aria-label="'Praat语谱图，'+start.toFixed(3)+'至'+end.toFixed(3)+'秒，0至'+data.frequency_max+'Hz'+(selectable?'；拖动选择LPC分析时间范围':'')" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="cancel"/><div v-if="selectable&&selected" class="spectrogram-selection" :style="{left:selected.left+'%',width:selected.width+'%'}"/><div class="frequency-axis"><span>{{data.frequency_max}} Hz</span><span>{{data.frequency_max/2}}</span><span>0</span></div></div>
<p v-if="data" class="hint">Praat {{data.praat_version}} · 当前时间窗相对灰度 · 6 dB/oct显示预加重。仅为预览，不是已校准声压级。</p>
</section></template>
<style scoped>
.selectable-spectrogram{touch-action:none;cursor:crosshair}
.spectrogram-selection{position:absolute;top:0;bottom:0;background:var(--selection);border-left:1px solid var(--accent);border-right:1px solid var(--accent);pointer-events:none}
</style>
