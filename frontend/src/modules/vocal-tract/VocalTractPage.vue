<script setup lang="ts">
import AppIcon from '../../components/AppIcon.vue';
import {ref,onMounted,onUnmounted,watch} from 'vue';
import {vocalRequest} from '../../platform/desktop.ts';
import {fontPayload,fontRevision} from '../../state/fonts.ts';
import {resizePanels} from '../../layout/resizablePanels.ts';
import buttonStyles from '../../design/buttons.css?inline';
import backgroundStyles from '../../design/background.css?inline';
import {syncBackgroundFrame} from '../../state/background.ts';
import {presetChart} from './ipa-chart.ts';
const props=defineProps<{active:boolean;stateKey?:string}>();
const emit=defineEmits<{references:[];help:[];close:[];dirty:[value:boolean]}>();
const frame=ref<HTMLIFrameElement>();const ready=!!vocalRequest;
let columns:ReturnType<typeof resizePanels>|undefined;
function chart(){frame.value?.contentWindow?.postMessage({type:'m10-ipa-chart',chart:presetChart},'*');}
function loaded(){theme();chart();columns?.destroy();const root=frame.value?.contentDocument?.querySelector<HTMLElement>('.workspace');if(root)columns=resizePanels(root,{key:props.stateKey??'M10',center:'.studio',centerMin:320,panels:[{selector:'.analysis',side:'right',variable:'--panel-analysis',label:'声学分析',initial:260,min:220,max:520},{selector:'.inspector',side:'right',label:'操作面板',initial:330,min:260,max:560}]});}
const theme=()=>{const style=getComputedStyle(document.documentElement);const colors=Object.fromEntries(['--app','--panel','--text','--muted','--border','--teal','--accent','--selected','--on-accent','--warning','--danger','--sidebar','--rail','--waveform-color','--button-glass-opacity','--button-glass-blur'].map(key=>[key,style.getPropertyValue(key).trim()]));frame.value?.contentWindow?.postMessage({type:'m10-theme',theme:document.documentElement.dataset.theme,colors,buttons:{mode:document.documentElement.dataset.buttonStyle,effects:document.documentElement.dataset.buttonEffects,css:buttonStyles}},'*');
 const doc=frame.value?.contentDocument;if(doc){let sheet=doc.getElementById('ptb-background-style');if(!sheet){sheet=doc.createElement('style');sheet.id='ptb-background-style';sheet.textContent=backgroundStyles;doc.head.append(sheet);}doc.documentElement.dataset.backgroundImage=document.documentElement.dataset.backgroundImage;doc.documentElement.dataset.backgroundFrame='on';doc.documentElement.dataset.buttonMaterial=document.documentElement.dataset.buttonMaterial;for(const key of ['--button-glass-opacity','--button-glass-blur'])doc.documentElement.style.setProperty(key,style.getPropertyValue(key).trim());if(frame.value)syncBackgroundFrame(frame.value);}
 if(fontPayload.value)frame.value?.contentWindow?.postMessage({type:'m10-fonts',fonts:JSON.parse(JSON.stringify(fontPayload.value))},'*');};
watch(fontRevision,theme);
async function message(event:MessageEvent){
  if(event.source!==frame.value?.contentWindow||!event.data||typeof event.data!=='object')return;
  const {type,id,op,body}=event.data;
  if(type==='m10-ready'){theme();chart();return;}
  if(type==='m10-references'){emit('references');return;}
  if(type==='m10-help'){emit('help');return;}
  if(type==='m10-close'){emit('close');return;}
  if(type==='m10-dirty'){emit('dirty',!!event.data.dirty);return;}
  if(type!=='m10-request'||typeof id!=='string'||typeof op!=='string')return;
  try{const value=await vocalRequest!(op,body);event.source?.postMessage({type:'m10-response',id,ok:true,value},{targetOrigin:'*'});}
  catch(error){event.source?.postMessage({type:'m10-response',id,ok:false,error:error instanceof Error?error.message:'声道请求失败'},{targetOrigin:'*'});}
}
let observer:MutationObserver,frameSize:ResizeObserver;
onMounted(()=>{window.addEventListener('message',message);observer=new MutationObserver(theme);observer.observe(document.documentElement,{attributes:true,attributeFilter:['style','data-theme','data-palette','data-custom-palette','data-button-style','data-button-effects','data-button-material','data-background-image','data-waveform-color']});frameSize=new ResizeObserver(()=>{if(frame.value)syncBackgroundFrame(frame.value);});if(frame.value)frameSize.observe(frame.value);});
watch(()=>props.active,active=>{if(!active)void vocalRequest?.('deactivate').catch(()=>{});else theme();});
onUnmounted(()=>{columns?.destroy();observer?.disconnect();frameSize?.disconnect();window.removeEventListener('message',message);void vocalRequest?.('shutdown').catch(()=>{});});
</script>
<template>
<section class="vocal-page" aria-label="声道工作区">
<iframe v-if="ready" ref="frame" src="./vocal-tract/index.html" title="生理参数合成" @load="loaded"/>
<div v-else class="vocal-unavailable"><p>生理参数合成请使用 Windows 桌面版。</p><div class="vocal-help-actions"><button type="button" @click="emit('help')">帮助</button><button @click="emit('references')"><AppIcon name="book"/>方法与引用</button></div></div>
</section>
</template>
<style scoped>
.vocal-page{height:100%;min-height:0;width:100%;display:flex;flex:1}.vocal-page iframe{border:0;width:100%;height:100%;min-height:0;background:var(--app)}.vocal-unavailable{padding:30px;line-height:2}
.vocal-help-actions{display:flex;align-items:center;gap:var(--control-gap);margin-top:var(--control-gap)}.vocal-help-actions button{min-height:var(--control-height);font-size:var(--control-size)}
</style>
