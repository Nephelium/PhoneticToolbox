<script setup lang="ts">
import {ref,onMounted,onUnmounted,watch} from 'vue';
import {vocalRequest} from '../../platform/desktop.ts';
import {fontPayload,fontRevision} from '../../state/fonts.ts';
const props=defineProps<{active:boolean}>();
const emit=defineEmits<{references:[];close:[];dirty:[value:boolean]}>();
const frame=ref<HTMLIFrameElement>();const ready=!!vocalRequest;
const theme=()=>{frame.value?.contentWindow?.postMessage({type:'m10-theme',theme:document.documentElement.dataset.theme},'*');if(fontPayload.value)frame.value?.contentWindow?.postMessage({type:'m10-fonts',fonts:JSON.parse(JSON.stringify(fontPayload.value))},'*');};
watch(fontRevision,theme);
async function message(event:MessageEvent){
  if(event.source!==frame.value?.contentWindow||!event.data||typeof event.data!=='object')return;
  const {type,id,op,body}=event.data;
  if(type==='m10-ready'){theme();return;}
  if(type==='m10-references'){emit('references');return;}
  if(type==='m10-close'){emit('close');return;}
  if(type==='m10-dirty'){emit('dirty',!!event.data.dirty);return;}
  if(type!=='m10-request'||typeof id!=='string'||typeof op!=='string')return;
  try{const value=await vocalRequest!(op,body);event.source?.postMessage({type:'m10-response',id,ok:true,value},{targetOrigin:'*'});}
  catch(error){event.source?.postMessage({type:'m10-response',id,ok:false,error:error instanceof Error?error.message:'声道请求失败'},{targetOrigin:'*'});}
}
let observer:MutationObserver;
onMounted(()=>{window.addEventListener('message',message);observer=new MutationObserver(theme);observer.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});});
watch(()=>props.active,active=>{if(!active)void vocalRequest?.('deactivate').catch(()=>{});else theme();});
onUnmounted(()=>{observer?.disconnect();window.removeEventListener('message',message);void vocalRequest?.('shutdown').catch(()=>{});});
</script>
<template>
<section class="vocal-page">
<iframe v-if="ready" ref="frame" src="./vocal-tract/index.html" title="声道工作台" @load="theme"/>
<div v-else class="vocal-unavailable"><h2>声道工作台</h2><p>本轮已接入 Windows 本机应用。请从 v3 桌面应用打开声道工作台。</p><button @click="emit('references')">方法与来源</button></div>
</section>
</template>
<style scoped>
.vocal-page{height:100%;min-height:640px;width:100%;display:flex;flex:1}.vocal-page iframe{border:0;width:100%;height:100%;min-height:0;background:var(--app)}.vocal-unavailable{padding:30px;line-height:2}
</style>
