<script setup lang="ts">
import {messageRef,type MessageDescriptor} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {ref,watch,nextTick,onUnmounted,computed} from 'vue';
import type {Spectrogram,SpectrogramView} from '../platform/research.ts';
const props=defineProps<{loader:(view:SpectrogramView)=>Promise<Spectrogram>;start:number;end:number;channel:number;width:number;selectable?:boolean;selectionStart?:number;selectionEnd?:number}>();
const emit=defineEmits<{'selection-end':[start:number,end:number];'view-wheel':[event:WheelEvent];'view-double-click':[event:MouseEvent]}>();
const canvas=ref<HTMLCanvasElement>(),data=ref<Spectrogram|null>(null),error=messageRef(),loading=ref(false);
const dragging=ref(false),dragStart=ref(0),dragEnd=ref(0);let anchorX=0;
function pointerTime(event:PointerEvent){const box=(event.currentTarget as HTMLCanvasElement).getBoundingClientRect();return props.start+Math.max(0,Math.min(1,(event.clientX-box.left)/box.width))*(props.end-props.start);}
function down(event:PointerEvent){if(!props.selectable||event.button!==0)return;dragging.value=true;anchorX=event.clientX;dragStart.value=dragEnd.value=pointerTime(event);(event.currentTarget as HTMLCanvasElement).setPointerCapture(event.pointerId);}
function move(event:PointerEvent){if(dragging.value)dragEnd.value=pointerTime(event);}
function up(event:PointerEvent){if(!dragging.value)return;dragEnd.value=pointerTime(event);dragging.value=false;if(Math.abs(event.clientX-anchorX)<3)return;emit('selection-end',Math.min(dragStart.value,dragEnd.value),Math.max(dragStart.value,dragEnd.value));}
function cancel(){dragging.value=false;}
const selected=computed(()=>{const a=dragging.value?dragStart.value:props.selectionStart,b=dragging.value?dragEnd.value:props.selectionEnd;if(a===undefined||b===undefined||b===a||props.end<=props.start)return null;const left=Math.max(props.start,Math.min(a,b)),right=Math.min(props.end,Math.max(a,b));return right>left?{left:(left-props.start)/(props.end-props.start)*100,width:(right-left)/(props.end-props.start)*100}:null;});
const requestWidth=computed(()=>Math.max(100,Math.min(1000,Math.round(props.width))));
let version=0,running=false,again=false,disposed=false,attempt=0,timer:ReturnType<typeof setTimeout>;
const messages:Record<string,MessageDescriptor>={preview_busy:uiMessage('components.another_spectrogram_is_being_computed_try_again_shortly'),preview_runtime_unavailable:uiMessage('components.the_praat_scientific_runtime_is_unavailable_in_this_environment'),preview_timeout:uiMessage('components.spectrogram_computation_timed_out_choose_a_shorter_time_window'),preview_memory_exceeded:uiMessage('components.the_spectrogram_exceeds_the_preview_memory_budget_choose_shorter'),invalid_spectrogram_input:uiMessage('components.a_spectrogram_cannot_be_generated_for_this_audio_or'),preview_failed:uiMessage('components.the_spectrogram_could_not_be_read_check_the_audio')};
function paint(value:Spectrogram){
 const target=canvas.value;if(!target)return;const ctx=target.getContext('2d');if(!ctx)return;
 target.width=Math.max(100,props.width);target.height=220;ctx.fillStyle='#fff';ctx.fillRect(0,0,target.width,target.height);
 const raw=atob(value.pixels_base64);if(raw.length!==value.width*value.height)throw uiError(uiMessage('components.the_spectrogram_pixel_count_is_inconsistent'));
 const source=document.createElement('canvas');source.width=value.width;source.height=value.height;const sc=source.getContext('2d')!;
 const pixels=sc.createImageData(value.width,value.height);
 for(let y=0;y<value.height;y++)for(let x=0;x<value.width;x++){const p=((value.height-1-y)*value.width+x)*4,v=raw.charCodeAt(y*value.width+x);pixels.data[p]=pixels.data[p+1]=pixels.data[p+2]=v;pixels.data[p+3]=255;}
 sc.putImageData(pixels,0,0);ctx.imageSmoothingEnabled=false;
 const span=props.end-props.start;
 ctx.drawImage(source,(value.x1-value.dx/2-props.start)/span*target.width,(value.frequency_max-value.y1-value.dy*(value.height-.5))/value.frequency_max*target.height,value.dx*value.width/span*target.width,value.dy*value.height/value.frequency_max*target.height);
}
async function run(){
 if(disposed)return;if(running){again=true;return;}clearTimeout(timer);running=true;again=false;const current=version;let retrying=false;loading.value=true;error.value='';
 const loader=props.loader,view={channel:props.channel,start:props.start,end:props.end,width:requestWidth.value};
 try{const result=await loader(view);
  if(disposed||version!==current)return;data.value=result;await nextTick();if(!disposed&&version===current)paint(result);
 }catch(e){if(!disposed&&current===version){const text=e instanceof Error?e.message:'preview_failed';if(text==='preview_busy'&&attempt<8){retrying=true;timer=setTimeout(run,500+attempt++*250);}else error.value=messages[text]??uiErrorMessage(e);}}
 finally{running=false;if(!disposed){if(current===version)loading.value=retrying;if(again){clearTimeout(timer);void run();}}}
}
function schedule(){version++;attempt=0;data.value=null;error.value='';clearTimeout(timer);loading.value=true;timer=setTimeout(run,60);}
watch(()=>[props.loader,props.start,props.end,props.channel,requestWidth.value],schedule,{immediate:true});
watch(()=>props.width,()=>{if(data.value&&!loading.value)paint(data.value);});
onUnmounted(()=>{disposed=true;version++;clearTimeout(timer);});
</script>
<template><section class="spectrogram-view" :aria-label="tr('components.praat_spectrogram')">
<div class="track-label"><span>{{tr('components.spectrogram_channel')}} {{channel+1}}</span><small>Praat · Gaussian 5 ms · 50 dB</small></div>
<div class="spectrogram-canvas" :aria-busy="loading"><canvas :style="{visibility:data&&!loading&&!error?'visible':'hidden'}" ref="canvas" role="img" :class="{'selectable-spectrogram':selectable}" :aria-label="tr('components.praat_spectrogram_370')+start.toFixed(3)+tr('components.to')+end.toFixed(3)+tr('components.s_0_to')+(data?.frequency_max??0)+'Hz'+(selectable?tr('components.drag_to_select_a_time_range'):'')" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="cancel" @wheel="emit('view-wheel',$event)" @dblclick="emit('view-double-click',$event)"/><p v-if="loading" role="status" class="spectrogram-empty">{{tr('components.computing_the_spectrogram_for_the_current_time_window')}}</p><p v-else-if="error" role="alert" class="error-banner">{{error}} <button @click="schedule">{{tr('components.retry')}}</button></p><template v-if="data&&!loading&&!error"><div v-if="selectable&&selected" class="spectrogram-selection" :style="{left:selected.left+'%',width:selected.width+'%'}"/><div class="frequency-axis"><span>{{data.frequency_max}} Hz</span><span>{{data.frequency_max/2}}</span><span>0</span></div></template></div>
<p class="hint" :style="{visibility:data&&!loading&&!error?'visible':'hidden'}">Praat {{data?.praat_version??'—'}} {{tr('components.relative_grayscale_within_the_current_window_display_pre_emphasis')}}</p>
</section></template>
<style scoped>
.selectable-spectrogram{touch-action:none;cursor:crosshair}
.spectrogram-canvas{background:var(--panel)}
.spectrogram-empty,.error-banner{position:absolute;inset:0;min-height:0;margin:0;display:flex;align-items:center;justify-content:center;flex-wrap:wrap;padding:12px}
.spectrogram-selection{position:absolute;top:0;bottom:0;background:var(--selection);border-left:1px solid var(--accent);border-right:1px solid var(--accent);pointer-events:none}
</style>
