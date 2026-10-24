<script setup lang="ts">
import {ref,watch,onMounted,onBeforeUnmount} from 'vue';
import IpaKeyboardChart,{type KeyboardSymbol} from './IpaKeyboardChart.vue';
import type {ChartSystem} from '../ipa-plus/types.ts';
defineProps<{active:boolean;disabled:boolean}>();
const emit=defineEmits<{insert:[entry:KeyboardSymbol];close:[]}>();
const panel=ref<HTMLElement>(),system=ref<ChartSystem>('ipa'),zoom=ref(100);
let x=16,y=16,width=760,baseWidth=760,height=440,frame=0;
let zoomObserver:MutationObserver|undefined,lastScale=1;
let gesture:{id:number;target:HTMLElement;mode:'move'|'resize';x:number;y:number;left:number;top:number;width:number;height:number}|undefined;
const scale=()=>parseFloat(document.documentElement.style.zoom)||1;
const viewport=()=>({width:window.innerWidth/scale(),height:window.innerHeight/scale()});
function layout(){const view=viewport();width=Math.min(view.width-16,Math.max(Math.min(320,view.width-16),baseWidth*zoom.value/100));height=Math.min(view.height-16,Math.max(Math.min(220,view.height-16),height));x=Math.max(8,Math.min(view.width-width-8,x));y=Math.max(8,Math.min(view.height-height-8,y));const el=panel.value;if(el){el.style.width=width+'px';el.style.height=height+'px';el.style.transform=`translate(${x}px,${y}px)`;}}
function schedule(){if(!frame)frame=requestAnimationFrame(()=>{frame=0;layout();});}
function begin(event:PointerEvent,mode:'move'|'resize'){
 if(event.button!==0||gesture)return;event.preventDefault();event.stopPropagation();
 const target=event.currentTarget as HTMLElement;target.setPointerCapture(event.pointerId);gesture={id:event.pointerId,target,mode,x:event.clientX,y:event.clientY,left:x,top:y,width,height};
}
function move(event:PointerEvent){const g=gesture;if(!g||g.id!==event.pointerId)return;const dx=(event.clientX-g.x)/scale(),dy=(event.clientY-g.y)/scale();if(g.mode==='move'){x=g.left+dx;y=g.top+dy;}else{baseWidth=Math.max(320,g.width+dx)*100/zoom.value;height=g.height+dy;}schedule();}
function end(event?:PointerEvent){if(!gesture||event&&gesture.id!==event.pointerId)return;const g=gesture;gesture=undefined;if(g.target.hasPointerCapture(g.id))g.target.releasePointerCapture(g.id);cancelAnimationFrame(frame);frame=0;layout();}
function keyboard(event:KeyboardEvent,mode:'move'|'resize'){const offsets:Record<string,[number,number]>={ArrowLeft:[-10,0],ArrowRight:[10,0],ArrowUp:[0,-10],ArrowDown:[0,10]};const d=offsets[event.key];if(!d)return;event.preventDefault();event.stopPropagation();if(mode==='move'){x+=d[0];y+=d[1];}else{baseWidth=Math.max(320,width+d[0])*100/zoom.value;height+=d[1];}layout();}
watch(zoom,layout);
onMounted(()=>{const view=viewport();x=view.width-width-24;y=view.height-height-24;layout();window.addEventListener('resize',layout);lastScale=scale();zoomObserver=new MutationObserver(()=>{const value=scale();if(value!==lastScale){lastScale=value;schedule();}});zoomObserver.observe(document.documentElement,{attributes:true,attributeFilter:['style']});});
onBeforeUnmount(()=>{end();cancelAnimationFrame(frame);zoomObserver?.disconnect();window.removeEventListener('resize',layout);});
</script>
<template>
 <Teleport to="body"><section v-show="active" ref="panel" class="ipa-keyboard" role="dialog" aria-label="国际音标小键盘" :style="{'--ipa-key-size':19*zoom/100+'px'}" @keydown.esc.stop="emit('close')">
  <header><button type="button" class="keyboard-drag" aria-label="拖动音标键盘" @pointerdown="begin($event,'move')" @pointermove="move" @pointerup="end" @pointercancel="end" @lostpointercapture="end" @keydown="keyboard($event,'move')">⠿ 音标键盘</button><nav aria-label="音标键盘表格"><button v-for="(name,id) in {ipa:'IPA',extipa:'extIPA',voqs:'VoQS'}" :key="id" type="button" :aria-pressed="system===id" @click="system=id">{{name}}</button></nav><div class="keyboard-scale"><button aria-label="缩小音标键盘字号" :disabled="zoom<=60" @click="zoom=Math.max(60,zoom-10)">−</button><output>{{zoom}}%</output><button aria-label="放大音标键盘字号" :disabled="zoom>=180" @click="zoom=Math.min(180,zoom+10)">＋</button></div><button type="button" aria-label="关闭音标键盘" @click="emit('close')">×</button></header>
  <fieldset :disabled="disabled" class="keyboard-content"><IpaKeyboardChart :system="system" @insert="emit('insert',$event)"/></fieldset>
  <button type="button" class="keyboard-resize" aria-label="调整音标键盘大小" @pointerdown="begin($event,'resize')" @pointermove="move" @pointerup="end" @pointercancel="end" @lostpointercapture="end" @keydown="keyboard($event,'resize')">◢</button>
 </section></Teleport>
</template>
<style scoped>
.ipa-keyboard{position:fixed;left:0;top:0;z-index:80;display:flex;flex-direction:column;box-sizing:border-box;border:1px solid var(--border);border-radius:9px;background:var(--panel)!important;color:var(--text);box-shadow:0 8px 28px #0003;contain:layout style;backdrop-filter:none!important;filter:none!important;overflow:hidden}.ipa-keyboard header{display:flex;align-items:center;gap:5px;flex-shrink:0;padding:4px 6px;background:var(--app);border-bottom:1px solid var(--border);flex-wrap:wrap}.ipa-keyboard button{min-height:26px;padding:2px 6px;font-size:12px;line-height:1.2;box-shadow:none!important;backdrop-filter:none!important;filter:none!important;transition:none!important;animation:none!important}.keyboard-drag{flex:1;text-align:left;cursor:move;touch-action:none;white-space:nowrap}.ipa-keyboard nav,.keyboard-scale{display:flex;gap:2px;align-items:center}.ipa-keyboard nav button[aria-pressed=true]{background:var(--accent)!important;color:var(--on-accent)!important}.keyboard-scale output{font-size:11px;font-family:var(--font-mono);min-width:35px;text-align:center}.keyboard-content{margin:0;padding:5px 5px 18px;min-width:0;min-height:0;overflow:auto;flex:1;border:0}.keyboard-resize{position:absolute;bottom:0;right:0;min-height:18px!important;width:23px;padding:0!important;cursor:nwse-resize;touch-action:none;color:var(--muted);background:var(--panel)!important;border:0!important}
</style>
<style>
/* Hundreds of keys share a solid surface even when global buttons use wallpaper. */
html:root[data-button-style][data-button-material] .ipa-keyboard.ipa-keyboard button:is(:enabled,:disabled){background-image:none!important;background-attachment:initial!important;backdrop-filter:none!important;filter:none!important;box-shadow:none!important;transition:none!important;animation:none!important}
html:root[data-button-style][data-button-material] .ipa-keyboard.ipa-keyboard .ipa-key.ipa-key:is(:enabled,:disabled){background:transparent!important;color:var(--text)!important;border-color:transparent!important}
html:root[data-button-style][data-button-material] .ipa-keyboard.ipa-keyboard .ipa-key.ipa-key:enabled:hover{background:var(--selected)!important;border-color:var(--accent)!important}
</style>
