<script setup lang="ts">
import {ref,computed,nextTick,watch,onUnmounted} from 'vue';
import {palettes,normalizePalette,paletteTokens,type ThemeMode,type ColorMode} from '../design/themes.ts';
import {useFloatingPicker} from './floating-picker.ts';
const props=defineProps<{modelValue:string;mode:ThemeMode}>();
const emit=defineEmits<{'update:modelValue':[value:string]}>();
const anchor=ref<HTMLElement>(),panel=ref<HTMLElement>();
const {open,position}=useFloatingPicker(anchor,panel);
const current=computed(()=>palettes.find(p=>p.id===normalizePalette(props.modelValue))!);
const system=window.matchMedia('(prefers-color-scheme: dark)'),systemDark=ref(system.matches);
const systemChanged=(event:MediaQueryListEvent)=>systemDark.value=event.matches;
system.addEventListener('change',systemChanged);onUnmounted(()=>system.removeEventListener('change',systemChanged));
const mode=computed<ColorMode>(()=>props.mode==='system'?(systemDark.value?'dark':'light'):props.mode);
const hover=ref(''),previewStyle=ref<Record<string,string>>({}),active=ref(0);
const preview=computed(()=>palettes.find(p=>p.id===hover.value));
const tokens=(id:string)=>paletteTokens(id,mode.value);
function badge(id:string){const t=tokens(id);return {background:t['--app'],color:t['--accent'],borderColor:t['--border']};}
function previewAt(id:string,element:HTMLElement){
 hover.value=id;const scale=parseFloat(document.documentElement.style.zoom)||1,r=element.getBoundingClientRect(),p=panel.value!.getBoundingClientRect(),w=Math.min(264,window.innerWidth/scale-16),h=220;
 const right=p.right/scale+8,left=p.left/scale-w-8;
 previewStyle.value={...tokens(id),width:w+'px',left:(right+w<=window.innerWidth/scale-8?right:left>=8?left:Math.max(8,Math.min(p.left/scale,window.innerWidth/scale-w-8)))+'px',top:Math.max(8,Math.min(r.top/scale,window.innerHeight/scale-h-8))+'px'};
}
function movePreview(){
 const e=panel.value?.querySelector<HTMLElement>('[data-palette-option="'+hover.value+'"]');
 if(!e||!panel.value)return;const r=e.getBoundingClientRect(),p=panel.value.getBoundingClientRect();
 if(r.bottom<=p.top||r.top>=p.bottom)hover.value='';else previewAt(hover.value,e);
}
async function show(){open.value=true;active.value=Math.max(0,palettes.findIndex(p=>p.id===current.value.id));await nextTick();panel.value?.querySelectorAll<HTMLElement>('[role=option]')[active.value]?.focus();}
function choose(id:string){emit('update:modelValue',id);open.value=false;anchor.value?.querySelector('button')?.focus();}
function keyboard(event:KeyboardEvent){
 if(['ArrowDown','ArrowUp','Home','End'].includes(event.key)){
  event.preventDefault();if(!open.value){void show();return;}
  active.value=event.key==='Home'?0:event.key==='End'?palettes.length-1:(active.value+(event.key==='ArrowUp'?-1:1)+palettes.length)%palettes.length;
  panel.value?.querySelectorAll<HTMLElement>('[role=option]')[active.value]?.focus();
 }else if(event.key==='Tab')open.value=false;
}
watch(open,value=>{if(!value)hover.value='';});
watch(mode,()=>{if(hover.value){const e=panel.value?.querySelector<HTMLElement>('[data-palette-option="'+hover.value+'"]');if(e)previewAt(hover.value,e);}});
</script>
<template><div ref="anchor" class="palette-picker" @keydown="keyboard"><button id="palette-choice" type="button" role="combobox" aria-label="配色方案" aria-haspopup="listbox" :aria-expanded="open" aria-controls="palette-list" @click="open?open=false:show()"><i class="palette-aa" :style="badge(current.id)">Aa</i><span>{{current.name}}</span><span class="palette-arrow" aria-hidden="true">⌄</span></button></div>
 <Teleport to="body"><div v-if="open" ref="panel" id="palette-list" class="palette-list" role="listbox" aria-label="配色方案列表" :style="position" @keydown="keyboard" @pointerleave="hover=''" @scroll="movePreview">
 <button v-for="(p,index) in palettes" :key="p.id" type="button" role="option" :aria-selected="p.id===modelValue" :data-palette-option="p.id" @focus="active=index;previewAt(p.id,$event.target as HTMLElement)" @pointerenter="active=index;previewAt(p.id,$event.currentTarget as HTMLElement)" @click="choose(p.id)"><i class="palette-aa" :style="badge(p.id)">Aa</i><span>{{p.name}}</span><span class="palette-check" aria-hidden="true">{{p.id===modelValue?'✓':''}}</span></button>
 </div><aside v-if="open&&preview" class="palette-hover-preview" :style="previewStyle" aria-label="悬浮配色预览" :data-preview-palette="preview.id" :data-preview-mode="mode"><header><span class="preview-dot"/><strong>{{preview.name}}</strong><small>{{mode==='dark'?'深色':'浅色'}}</small></header><div class="preview-body"><p>声学工作台 <span>Aa</span></p><small>用声音描绘细微的变化。</small><svg viewBox="0 0 220 48" aria-label="示例波形"><path d="M0 24H10L16 20L22 28L28 12L34 36L40 6L46 42L52 14L58 34L64 20L70 28L76 24H102L108 16L114 32L120 8L126 40L132 3L138 45L144 12L150 36L156 20L162 28L168 24H186L192 18L198 30L204 22L210 26L216 24H220"/></svg><div class="preview-actions"><span class="preview-primary">开始分析</span><span>保存结果</span><span class="preview-selected">F0 ✓</span></div><p class="preview-status"><i/>已读取音频 <code>220 Hz</code></p></div></aside></Teleport>
</template>
<style scoped>
.palette-picker>button{width:100%;justify-content:flex-start;min-height:40px;font:inherit}.palette-aa{display:grid;place-items:center;width:30px;height:30px;flex:none;border:1px solid;border-radius:8px;font:600 1.142857rem/1 var(--font);letter-spacing:-.6px}.palette-arrow{margin-left:auto;color:var(--muted)}
.palette-list{position:fixed;z-index:180;overflow:auto;overscroll-behavior:contain;padding:5px;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:9px;box-shadow:var(--shadow);scrollbar-width:thin}.palette-list button{width:100%;justify-content:flex-start;min-height:40px;padding:4px 8px;border:0;border-radius:5px;box-shadow:none}.palette-list button[aria-selected=true]{background:var(--selected)}.palette-check{margin-left:auto;width:16px;color:var(--accent)}
.palette-hover-preview{position:fixed;z-index:181;pointer-events:none;border:1px solid var(--border);border-radius:10px;overflow:hidden;box-shadow:var(--shadow);background:var(--app);color:var(--text);font:0.928571rem var(--font);isolation:isolate}.palette-hover-preview header{display:flex;align-items:center;gap:7px;padding:10px 12px;background:var(--sidebar);border-bottom:1px solid var(--border)}.palette-hover-preview header small{margin-left:auto;color:var(--muted);font-size:0.785714rem}.preview-dot{width:7px;height:7px;background:var(--accent);border-radius:50%}.preview-body{padding:12px;background:var(--panel)}.preview-body p{display:flex;align-items:center;justify-content:space-between;font-size:1rem;line-height:1.4}.preview-body>small{display:block;color:var(--muted);margin-top:5px;font-size:0.785714rem}.preview-body svg{display:block;width:100%;height:48px;margin:8px 0;border:1px solid var(--border);border-radius:5px;background:var(--app)}.preview-body path{fill:none;stroke:var(--accent);stroke-width:1.3}.preview-actions{display:flex;gap:6px;font-size:0.785714rem}.preview-actions span{padding:5px 7px;border:1px solid var(--border);border-radius:4px;white-space:nowrap}.preview-actions .preview-primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}.preview-actions .preview-selected{background:var(--selected);border-color:var(--accent);color:var(--accent)}.preview-body .preview-status{justify-content:flex-start;gap:5px;margin-top:10px;color:var(--muted);font-size:0.785714rem}.preview-status i{width:5px;height:5px;border-radius:50%;background:var(--success)}.preview-status code{margin-left:auto;font-size:0.785714rem}
</style>
