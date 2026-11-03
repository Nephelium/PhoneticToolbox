<script setup lang="ts">
import {tr,locale} from '../../i18n/index.ts';
import {webSettingsKey,readWebSettings,writeWebSettings} from '../../state/web-settings.ts';
import {preferenceChanges} from '../../platform/account-preferences.ts';
import WebModuleSeal from '../../components/WebModuleSeal.vue';
import ModalDialog from '../../components/ModalDialog.vue';
import {ref,onMounted,onUnmounted,watch} from 'vue';
import {vocalRequest} from '../../platform/desktop.ts';
import type {ResearchContext} from '../../platform/research.ts';
import {fontPayload,fontRevision} from '../../state/fonts.ts';
import {resizePanels} from '../../layout/resizablePanels.ts';
import buttonStyles from '../../design/buttons.css?inline';
import backgroundStyles from '../../design/background.css?inline';
import {syncBackgroundFrame} from '../../state/background.ts';
import {localizedPresetChart} from './ipa-chart.ts';
import {automaticAudioGain} from '../../state/audition.ts';
import {uiError,uiErrorMessage,messageSnapshot} from '../module-localization.ts';
const props=defineProps<{active:boolean;stateKey?:string;context?:ResearchContext}>();
const emit=defineEmits<{references:[];help:[];close:[];dirty:[value:boolean]}>();
const webPort=props.context?.files.vocal?.(props.stateKey??props.context.key);
const invoke=vocalRequest??webPort?.request;
const frame=ref<HTMLIFrameElement>();const ready=!!invoke;
const settingsKey=webSettingsKey(props.stateKey);
let frameSettings:Record<string,unknown>={};
const webNotice=ref(false);let serverFallback=false;
watch(()=>props.active,active=>{if(active&&serverFallback&&readWebSettings(props.stateKey).webNoticeSeen!==true)webNotice.value=true;});
function acknowledgeWebNotice(){writeWebSettings(props.stateKey,{webNoticeSeen:true});webNotice.value=false;}
const frameUrl='./vocal-tract/index.html?'+new URLSearchParams({...(!ready?{presentation:'1'}:{}),...(settingsKey?{settings:settingsKey}:{})});
let columns:ReturnType<typeof resizePanels>|undefined;
function chart(){frame.value?.contentWindow?.postMessage({type:'m10-ipa-chart',chart:localizedPresetChart()},location.origin);}
function frameLocale(){frame.value?.contentWindow?.postMessage({type:'m10-locale',locale:locale.value},location.origin);}
watch(locale,()=>{frameLocale();chart();});
function loaded(){theme();chart();frameLocale();columns?.destroy();const root=frame.value?.contentDocument?.querySelector<HTMLElement>('.workspace');if(root)columns=resizePanels(root,{key:props.stateKey??'M10',center:'.studio',centerMin:320,panels:[{selector:'.analysis',side:'right',variable:'--panel-analysis',label:()=>tr('modules.vocal-tract.acoustic_analysis'),initial:260,min:220,max:520},{selector:'.inspector',side:'right',label:()=>tr('modules.vocal-tract.controls'),initial:330,min:260,max:560}]});}
const theme=()=>{const style=getComputedStyle(document.documentElement);const colors=Object.fromEntries(['--app','--panel','--text','--muted','--border','--teal','--accent','--selected','--on-accent','--warning','--danger','--sidebar','--rail','--waveform-color','--button-glass-opacity','--button-glass-blur'].map(key=>[key,style.getPropertyValue(key).trim()]));frame.value?.contentWindow?.postMessage({type:'m10-theme',theme:document.documentElement.dataset.theme,colors,buttons:{mode:document.documentElement.dataset.buttonStyle,effects:document.documentElement.dataset.buttonEffects,css:buttonStyles}},'*');
 const doc=frame.value?.contentDocument;if(doc?.head){let sheet=doc.getElementById('ptb-background-style');if(!sheet){sheet=doc.createElement('style');sheet.id='ptb-background-style';sheet.textContent=backgroundStyles;doc.head.append(sheet);}doc.documentElement.dataset.backgroundImage=document.documentElement.dataset.backgroundImage;doc.documentElement.dataset.backgroundFrame='on';doc.documentElement.dataset.buttonMaterial=document.documentElement.dataset.buttonMaterial;for(const key of ['--button-glass-opacity','--button-glass-blur'])doc.documentElement.style.setProperty(key,style.getPropertyValue(key).trim());if(frame.value)syncBackgroundFrame(frame.value);}
 if(fontPayload.value)frame.value?.contentWindow?.postMessage({type:'m10-fonts',fonts:JSON.parse(JSON.stringify(fontPayload.value))},'*');};
watch(fontRevision,theme);
async function message(event:MessageEvent){
  if(event.source!==frame.value?.contentWindow||!event.data||typeof event.data!=='object')return;
  const {type,id,op,body}=event.data;
  if(event.origin===location.origin&&type==='m10-capabilities'){
    serverFallback=!vocalRequest&&props.context?.files.kind==='server'&&event.data.execution!=='browser-wasm';
    webNotice.value=!!(props.active&&serverFallback&&props.context?.files.kind==='server'&&readWebSettings(props.stateKey).webNoticeSeen!==true);return;
  }
  if(settingsKey&&event.origin===location.origin&&type==='m10-settings-read'){frameSettings=JSON.parse(JSON.stringify(readWebSettings(props.stateKey).frame??{}));event.source?.postMessage({type:'m10-settings-value',key:settingsKey,value:frameSettings},{targetOrigin:event.origin});return;}
  if(settingsKey&&event.origin===location.origin&&type==='m10-settings-write'&&body&&typeof body==='object'){const changes=preferenceChanges(frameSettings,body);if(writeWebSettings(props.stateKey,{frame:changes}))frameSettings=JSON.parse(JSON.stringify(body));return;}
  if(type==='m10-ready'){theme();chart();frameLocale();return;}
  if(type==='m10-references'){emit('references');return;}
  if(type==='m10-help'){emit('help');return;}
  if(type==='m10-close'){emit('close');return;}
  if(type==='m10-dirty'){emit('dirty',!!event.data.dirty);return;}
  if(!ready||type!=='m10-request'||typeof id!=='string'||typeof op!=='string')return;
  try{const payload=['live','preview','animation/play','audio/test','deactivate'].includes(op)?{...body,audition_auto_gain:automaticAudioGain.value}:body;let value=await invoke!(op,payload);if(value&&typeof value==='object'&&typeof value.audio_error==='string'&&value.audio_error&&!value.audio_error_display)value={...value,audio_error_display:messageSnapshot(uiErrorMessage(uiError(value.audio_error)))};event.source?.postMessage({type:'m10-response',id,ok:true,value},{targetOrigin:'*'});}
  catch(error){const display=uiErrorMessage(error);event.source?.postMessage({type:'m10-response',id,ok:false,error:error instanceof Error?error.message:tr('modules.vocal-tract.vocal_tract_request_failed'),...(typeof display==='object'?{display:messageSnapshot(display)}:{}),recoveryAvailable:!!(error&&typeof error==='object'&&'recoveryAvailable'in error&&error.recoveryAvailable===true)},{targetOrigin:location.origin});}
}
let observer:MutationObserver,frameSize:ResizeObserver;
onMounted(()=>{window.addEventListener('message',message);observer=new MutationObserver(theme);observer.observe(document.documentElement,{attributes:true,attributeFilter:['style','data-theme','data-palette','data-custom-palette','data-button-style','data-button-effects','data-button-material','data-background-image','data-waveform-color']});frameSize=new ResizeObserver(()=>{if(frame.value)syncBackgroundFrame(frame.value);});if(frame.value)frameSize.observe(frame.value);});
watch(()=>props.active,active=>{if(!active)void invoke?.('deactivate').catch(()=>{});else theme();});
watch(automaticAudioGain,()=>{void invoke?.('deactivate',{audition_auto_gain:automaticAudioGain.value}).catch(()=>{});});
onUnmounted(()=>{columns?.destroy();observer?.disconnect();frameSize?.disconnect();window.removeEventListener('message',message);void invoke?.('shutdown').catch(()=>{});});
</script>
<template>
<div class="vocal-host">
<WebModuleSeal :sealed="!ready" :title="tr('modules.vocal-tract.physiological_parameter_synthesis')" :reason="tr('modules.vocal-tract.the_server_vtl_engine_and_session_interface_are_still')" @help="emit('help')">
<section class="vocal-page" :aria-label="tr('modules.vocal-tract.vocal_tract_workspace')">
<iframe ref="frame" :src="frameUrl" :title="tr('modules.vocal-tract.physiological_parameter_synthesis')" @load="loaded"/>
</section>
</WebModuleSeal>
<ModalDialog v-if="webNotice&&active" :title="tr('modules.vocal-tract.physiological_parameter_synthesis_web')" variant="confirm" @close="acknowledgeWebNotice">
<p>{{tr('modules.vocal-tract.due_to_server_performance_limits_adjust_and_fix_the')}}</p>
<p>{{tr('modules.vocal-tract.the_web_version_does_not_yet_support_real_time')}}</p>
<template #footer><button class="primary" @click="acknowledgeWebNotice">{{tr('modules.vocal-tract.got_it')}}</button></template>
</ModalDialog>
</div>
</template>
<style scoped>
.vocal-host{height:100%;min-height:0;min-width:0;display:flex;flex:1}
.vocal-page{height:100%;min-height:0;width:100%;display:flex;flex:1}.vocal-page iframe{border:0;width:100%;height:100%;min-height:0;background:var(--app)}.vocal-unavailable{padding:30px;line-height:2}
.vocal-help-actions{display:flex;align-items:center;gap:var(--control-gap);margin-top:var(--control-gap)}.vocal-help-actions button{min-height:var(--control-height);font-size:var(--control-size)}
</style>
