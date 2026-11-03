<script setup lang="ts">
import {ipaTextFilename} from './filename.ts';
import {uiErrorMessage as displayErrorValue} from '../module-localization.ts';

import {tr,message as localizedMessage,messageRef} from '../../i18n/index.ts';
import {useWebSettings,settingsFields} from '../../state/web-settings.ts';
import {syncNumberInput} from '../../design/numeric-input.ts';
import AppIcon from '../../components/AppIcon.vue';
import {computed,nextTick,onBeforeUnmount,onMounted,reactive,ref,watch} from 'vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import ChartView from './ChartView.vue';
import SymbolDetails from './SymbolDetails.vue';
import SymbolPlayback from './SymbolPlayback.vue';
import {placeHover,type HoverTarget} from './hover-position.ts';
import {catalog,systems} from './catalog.ts';
import {EditorHistory,insertSymbol} from './editor.ts';
import {codePoints,suspiciousCharacters} from './unicode.ts';
import {createDraft,loadDraft,persistDraft,restoreDraft} from './storage.ts';
import type {Draft,EditorSnapshot,SymbolEntry} from './types.ts';
import './fonts.css';
import {copyText} from '../../platform/clipboard.ts';
import {isHandheldBrowser} from '../../state/desktop-viewport.ts';
import fontLicense from '../../assets/ipa-plus/OFL-PTBIPAPlus.txt?raw';
import notoLicense from '../../assets/ipa-plus/OFL-Noto.txt?raw';

const props=withDefaults(defineProps<{stateKey?:string;active?:boolean}>(),{stateKey:'M17',active:true});
const emit=defineEmits<{dirty:[value:boolean];references:[]}>();
const writer=globalThis.crypto?.randomUUID?.()??'m17-'+Date.now().toString(36)+Math.random().toString(36).slice(2);
const draft=reactive(createDraft(writer)),history=new EditorHistory();
const editor=ref<HTMLTextAreaElement>(),page=ref<HTMLElement>(),chart=ref<HTMLElement>();
const loaded=ref(false),fontReady=ref(false),fontError=messageRef(),dirty=ref(false),error=messageRef(),notice=messageRef();
const query=ref(''),selected=ref<SymbolEntry>(),hovered=ref<SymbolEntry>(),detailsOpen=ref(false),helpOpen=ref(false);
const historyTick=ref(0),blocked=ref(false);let composing=false,disposed=false,saveTimer:ReturnType<typeof setTimeout>|undefined,hoverTimer:ReturnType<typeof setTimeout>|undefined;
const playOnClick=ref(false),playRequest=ref(0),playing=ref(false),hoverStyle=ref<Record<string,string>>({visibility:'hidden'});
const hoverPanel=ref<HTMLElement>();
let hoverTarget:HoverTarget|undefined,hoverOpening=false;
let key='ipa-plus.v1.'+props.stateKey,revision=0,saving:Promise<boolean>|undefined,resizeStart:{y:number;height:number;scale:number}|undefined;
const canUndo=computed(()=>{historyTick.value;return history.canUndo;}),canRedo=computed(()=>{historyTick.value;return history.canRedo;});
const currentSystem=computed(()=>systems.find(s=>s.id===draft.system)!);
const suspicious=computed(()=>suspiciousCharacters(draft.text));
const selectedCodes=computed(()=>{
  const value=draft.text.slice(draft.start,draft.end),points:string[]=[];
  for(const char of value){points.push(...codePoints(char));if(points.length>8)return '';}
  return points.join(' ');
});
watch(dirty,v=>emit('dirty',v),{immediate:true});
function changed(){if(!loaded.value)return;dirty.value=true;clearTimeout(saveTimer);if(!blocked.value)saveTimer=setTimeout(()=>void save(),450);}
watch(()=>[draft.system,draft.introductions,draft.textSize,draft.editorHeight],changed);
watch(()=>draft.introductions,clearHover);
watch(()=>props.active,value=>{if(!value){clearHover();playing.value=false;}});
watch(()=>draft.system,()=>{clearHover();playing.value=false;});
watch(playOnClick,()=>{playing.value=false;clearHover();});
watch(query,clearHover);
const textSizeInput=computed({get:()=>draft.textSize,set:(input:number|string)=>{const value=Number(input);draft.textSize=Number.isFinite(value)&&String(input).trim()?Math.max(18,Math.min(54,Math.round(value))):26;}});
function captureSelection(){if(!editor.value||composing)return;draft.start=editor.value.selectionStart;draft.end=editor.value.selectionEnd;history.select(draft.start,draft.end);}
function snapshot():EditorSnapshot{return {text:draft.text,start:draft.start,end:draft.end};}
async function showSelection(focus=true){await nextTick();if(!editor.value)return;
  // Updating a selection does not need focus. On handhelds, only a direct tap
  // in the editor should request the software keyboard; desktop typing resumes.
  if(focus&&!isHandheldBrowser())editor.value.focus({preventScroll:true});
  editor.value.setSelectionRange(draft.start,draft.end);
}
function commit(value:EditorSnapshot){history.commit(value);Object.assign(draft,value);historyTick.value++;changed();void showSelection();}
function input(){const el=editor.value!;draft.text=el.value;draft.start=el.selectionStart;draft.end=el.selectionEnd;if(!composing){history.commit(snapshot());historyTick.value++;changed();}}
function compositionStart(){captureSelection();composing=true;}
function compositionEnd(){composing=false;input();}
function undo(){if(composing)return;const value=history.undo();if(value){Object.assign(draft,value);historyTick.value++;changed();void showSelection();}}
function redo(){if(composing)return;const value=history.redo();if(value){Object.assign(draft,value);historyTick.value++;changed();void showSelection();}}
function keydown(event:KeyboardEvent){
  if(event.isComposing||composing)return;
  if((event.ctrlKey||event.metaKey)&&!event.altKey){const key=event.key.toLowerCase();if(key==='z'||key==='y'){event.preventDefault();event.stopPropagation();if(key==='y'||event.shiftKey)redo();else undo();}}
}
function beforeInput(event:InputEvent){if(event.inputType==='historyUndo'||event.inputType==='historyRedo'){event.preventDefault();event.stopPropagation();event.inputType==='historyUndo'?undo():redo();}}
function activate(entry:SymbolEntry){
  if(playOnClick.value){
   const preserved=snapshot();selected.value=entry;detailsOpen.value=true;playing.value=true;playRequest.value++;clearHover();
   // Native WebEngine can reset textarea selection when the media panel mounts.
   void nextTick(()=>{draft.start=preserved.start;draft.end=preserved.end;editor.value?.setSelectionRange(preserved.start,preserved.end);});return;
  }
  insert(entry);
}
function insert(entry:SymbolEntry){
  if(!loaded.value||composing){notice.value=composing?localizedMessage('modules.ipa-plus.text.c32771e9a62f'):localizedMessage('modules.ipa-plus.text.d92f8b4450d9');return;}
  selected.value=entry;notice.value='';
  try{commit(insertSymbol(snapshot(),entry));if(entry.insertionMode==='combining'&&draft.start===entry.insertText.length)notice.value=localizedMessage('modules.ipa-plus.text.2fdf1c744876');if(entry.representation)notice.value=localizedMessage('modules.ipa-plus.text.ff01391e56fe');}catch(e){notice.value=displayErrorValue(e);}
}
function positionHover(){
 if(!hoverTarget||!hovered.value||!page.value)return;
 const body=page.value.getBoundingClientRect(),scale=body.width/page.value.offsetWidth||1;
 const bounds={left:Math.max(0,body.left),top:Math.max(0,body.top),right:Math.min(innerWidth,body.right),bottom:Math.min(innerHeight,body.bottom)};
 const p=placeHover(hoverTarget.target.getBoundingClientRect(),hoverTarget.pointer,bounds,scale,hoverPanel.value?(hoverPanel.value.scrollHeight+2)*scale:undefined);
 hoverStyle.value={left:(p.left-body.left)/scale+'px',top:(p.top-body.top)/scale+'px',width:p.width/scale+'px',maxHeight:p.maxHeight/scale+'px',visibility:'visible'};
}
function hover(value:HoverTarget|null){
 if(!value){hoverOpening=false;clearTimeout(hoverTimer);hoverTimer=setTimeout(clearHover,350);return;}
 if(hoverTarget?.target===value.target&&(hovered.value||hoverOpening)){hoverTarget=value;if(hovered.value){clearTimeout(hoverTimer);positionHover();}return;}
 clearTimeout(hoverTimer);hoverTarget=value;hovered.value=undefined;hoverStyle.value={visibility:'hidden'};
 if(draft.introductions&&props.active&&!detailsOpen.value){hoverOpening=true;hoverTimer=setTimeout(()=>{hoverOpening=false;if(hoverTarget?.target===value.target){hovered.value=value.entry;positionHover();void nextTick(positionHover);}},420);}
}
function clearHover(){clearTimeout(hoverTimer);hoverOpening=false;hovered.value=undefined;hoverTarget=undefined;}
function keepHover(){clearTimeout(hoverTimer);}
function inspect(entry:SymbolEntry){selected.value=entry;detailsOpen.value=true;playing.value=false;clearHover();}
async function save():Promise<boolean>{
  clearTimeout(saveTimer);
  if(!loaded.value||blocked.value||composing)return false;
  if(saving){await saving;return dirty.value&&!blocked.value?save():!dirty.value;}
  if(!dirty.value)return true;
  const value:Draft={...draft,revision,writer,catalogVersion:catalog.version};
  saving=(async()=>{try{
    const next=await persistDraft(key,value,revision);revision=next;draft.revision=next;
    const same=['text','system','introductions','textSize','editorHeight'].every(field=>(draft as unknown as Record<string,unknown>)[field]===(value as unknown as Record<string,unknown>)[field]);
    if(same)dirty.value=false;error.value='';return true;
  }catch(e){error.value=displayErrorValue(e);blocked.value=true;return false;}finally{saving=undefined;}})();
  return saving;
}
async function forkDraft(){if(saving)await saving;key='ipa-plus.v1.'+props.stateKey+'.'+writer;revision=0;blocked.value=false;dirty.value=true;if(await save())notice.value=localizedMessage('modules.ipa-plus.text.4420cb65222b');}
async function copyAll(){
  if(!draft.text){notice.value=localizedMessage('modules.ipa-plus.text.057e0443cad7');return;}
  try{await copyText(draft.text);notice.value=localizedMessage('modules.ipa-plus.text.d688831e0216');}
  catch{notice.value=localizedMessage('modules.ipa-plus.text.6c7b085ae454');draft.start=0;draft.end=draft.text.length;void showSelection();}
}
function exportText(){const blob=new Blob([draft.text],{type:'text/plain;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=ipaTextFilename();a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);notice.value=localizedMessage('modules.ipa-plus.text.821407425965');}
function selectAll(){draft.start=0;draft.end=draft.text.length;history.select(draft.start,draft.end);void showSelection();}
function clear(){if(composing)return;commit({text:'',start:0,end:0});notice.value=localizedMessage('modules.ipa-plus.text.b8fa39012185');}
function resizeDown(event:PointerEvent){const target=event.currentTarget as HTMLElement;target.setPointerCapture(event.pointerId);resizeStart={y:event.clientY,height:draft.editorHeight,scale:page.value?(page.value.getBoundingClientRect().width/page.value.offsetWidth||1):1};event.preventDefault();}
function resizeMove(event:PointerEvent){if(resizeStart)draft.editorHeight=Math.max(96,Math.min(360,resizeStart.height+(resizeStart.y-event.clientY)/resizeStart.scale));}
function resizeKey(event:KeyboardEvent){if(['ArrowUp','ArrowDown','Home','End'].includes(event.key)){event.preventDefault();draft.editorHeight=event.key==='Home'?96:event.key==='End'?360:Math.max(96,Math.min(360,draft.editorHeight+(event.key==='ArrowUp'?16:-16)));}}
function beforeUnload(event:BeforeUnloadEvent){if(dirty.value){event.preventDefault();event.returnValue='';}}
onMounted(async()=>{
  window.addEventListener('beforeunload',beforeUnload);
  window.addEventListener('resize',clearHover);
  try{const result=restoreDraft(await loadDraft(key),writer);if(disposed)return;Object.assign(draft,result.draft);revision=draft.revision;blocked.value=result.blocked;error.value=result.messageDisplay??result.message;history.current=snapshot();}
  catch(e){error.value=displayErrorValue(e);blocked.value=true;}
  // Flush settings watchers while hydration is still read-only. Opening a second
  // window must not revise the saved record before that window edits anything.
  await nextTick();if(disposed)return;loaded.value=true;void showSelection(false);
  try{await document.fonts.load('24px "PTB-IPA-Plus"','z̥᫃ C⃝ 𝼆 V𐞀');if(disposed)return;if(![...document.fonts].some(f=>f.family==='PTB-IPA-Plus'&&f.status==='loaded'))throw Error('font missing');fontReady.value=true;}
  catch{fontError.value=localizedMessage('modules.ipa-plus.text.fa4188602f82');}
});
onBeforeUnmount(()=>{disposed=true;clearTimeout(saveTimer);clearTimeout(hoverTimer);window.removeEventListener('beforeunload',beforeUnload);window.removeEventListener('resize',clearHover);});
defineExpose({save});
useWebSettings(props.stateKey,{...settingsFields(draft,['system','introductions','textSize','textSizeVersion','editorHeight']),playOnClick},()=>loaded.value);
</script>
<template>
 <ModuleFrame :label="tr('modules.ipa-plus.text.79bb75c4a4e7')" class="ipa-plus-page" :style="{'--m17-font-size':draft.textSize+'px'}" fit>
  <template #toolbar><ModuleToolbar :label="tr('modules.ipa-plus.text.4a89e0135494')"><div class="m17-tabs" role="group" :aria-label="tr('modules.ipa-plus.text.3eb3ed6e05cd')"><button v-for="system in systems" :key="system.id" type="button" :aria-pressed="draft.system===system.id" :class="{primary:draft.system===system.id}" @click="draft.system=system.id;hovered=undefined">{{system.label}}</button></div><input v-model="query" class="m17-search" :aria-label="tr('modules.ipa-plus.text.1508528b129b')" :placeholder="tr('modules.ipa-plus.text.e031d718828f')"><label class="m17-size-control">{{tr('modules.ipa-plus.text.f71561b5af0a')}} <input v-model.lazy.number="textSizeInput" @change="syncNumberInput($event,textSizeInput)" type="number" min="18" max="54" step="1" :aria-label="tr('modules.ipa-plus.text.4dbb035fe149')"></label><label class="m17-toggle"><input v-model="draft.introductions" type="checkbox">{{tr('modules.ipa-plus.text.c7939737e6ae')}}</label><label class="m17-toggle m17-toggle-unavailable" :title="tr('modules.ipa-plus.text.fd30e8f09aa8')"><input v-model="playOnClick" type="checkbox" disabled>{{tr('modules.ipa-plus.text.864ead58a317')}}</label><template #actions><button type="button" @click="helpOpen=!helpOpen">{{tr('modules.ipa-plus.text.a57cfcb8428d')}}</button><button type="button" @click="emit('references')"><AppIcon name="book"/>{{tr('modules.ipa-plus.text.7eaa43419522')}}</button></template></ModuleToolbar></template>
  <div ref="page" class="m17-body" :data-font-ready="fontReady" :data-loaded="loaded">
   <ModuleStatus v-if="fontError" kind="error" :message="fontError"/>
   <ModuleStatus v-if="error" kind="error" :message="error"><button class="primary" type="button" @click="forkDraft">{{tr('modules.ipa-plus.text.8d0eb8782a94')}}</button></ModuleStatus>
   <div ref="chart" class="m17-chart-viewport" @scroll="clearHover">
    <div class="m17-chart-meta"><span>{{currentSystem.version}}</span><span>{{playOnClick?tr('modules.ipa-plus.text.9d743e275b4d'):tr('modules.ipa-plus.text.6dc3042751a1')}}</span></div>
    <ChartView :system="draft.system" :query="query" :selected="selected?.id" :play-on-click="playOnClick" @insert="activate" @hover="hover" @inspect="inspect"/>
   </div>
   <div class="m17-divider" role="separator" tabindex="0" :aria-label="tr('modules.ipa-plus.text.88c964ce8261')" aria-orientation="horizontal" :aria-valuenow="draft.editorHeight" :aria-valuemin="96" :aria-valuemax="360" @pointerdown="resizeDown" @pointermove="resizeMove" @pointerup="resizeStart=undefined" @pointercancel="resizeStart=undefined" @keydown="resizeKey"><span></span></div>
   <div class="m17-editor-area" :style="{height:draft.editorHeight+'px'}"><textarea ref="editor" :value="draft.text" class="m17-editor m17-ipa" :style="{fontSize:draft.textSize+'px'}" :aria-label="tr('modules.ipa-plus.text.bf6f767d83cc')" spellcheck="false" autocomplete="off" autocapitalize="off" :disabled="!loaded" :placeholder="tr('modules.ipa-plus.text.6cd087f49d34')" @input="input" @select="captureSelection" @keyup="captureSelection" @pointerup="captureSelection" @blur="captureSelection" @keydown="keydown" @beforeinput="beforeInput" @compositionstart="compositionStart" @compositionend="compositionEnd"/></div>
   <div class="m17-editor-toolbar"><button type="button" class="primary" @click="copyAll">{{tr('modules.ipa-plus.text.75c393923f87')}}</button><button type="button" @click="selectAll">{{tr('modules.ipa-plus.text.3a5040b68abf')}}</button><button type="button" :disabled="!canUndo" @click="undo">{{tr('modules.ipa-plus.text.926a50b98ece')}}</button><button type="button" :disabled="!canRedo" @click="redo">{{tr('modules.ipa-plus.text.03717b6f1070')}}</button><button class="primary" type="button" @click="exportText">{{tr('modules.ipa-plus.text.5dd6704c1e9a')}}</button><button type="button" :disabled="!draft.text" @click="clear">{{tr('modules.ipa-plus.text.1ef3de06b32e')}}</button><span class="m17-save-state" role="status">{{!loaded?tr('modules.ipa-plus.text.97ddb9bc838a'):dirty?tr('modules.ipa-plus.text.ce2e52454c5f'):tr('modules.ipa-plus.text.dbd49f29603d')}}</span></div>
   <div class="m17-status" role="status"><span>{{notice||tr('modules.ipa-plus.text.7f471c4d1c3f',{v0:Array.from(draft.text).length,v1:fontReady?tr('modules.ipa-plus.text.d142bcb1226c'):tr('modules.ipa-plus.text.498023064041')})}}</span><code v-if="draft.end>draft.start&&selectedCodes">{{selectedCodes}}</code><span v-if="suspicious.length">{{tr('modules.ipa-plus.text.0259eebf57c8')}} {{suspicious.length}} {{tr('modules.ipa-plus.text.e3651f462d02')}}{{suspicious.slice(0,4).map(c=>tr('modules.ipa-plus.text.aa94f28ad11b',{v0:c.code,v1:c.position})).join('、')}}</span></div>
   <aside v-if="detailsOpen&&selected" class="m17-detail-panel" :aria-label="tr('modules.ipa-plus.text.920cc94666ac')"><div class="m17-panel-actions"><button type="button" @click="activate(selected)">{{playOnClick?tr('modules.ipa-plus.text.b325ea00e74b'):tr('modules.ipa-plus.text.075b42b82b51')}}</button><button type="button" @click="detailsOpen=false;playing=false">{{tr('modules.ipa-plus.text.afd4b783536b')}}</button></div><SymbolPlayback v-if="playing&&active" :key="playRequest" :entry="selected" :request="playRequest"/><SymbolDetails :entry="selected"/></aside>
   <aside ref="hoverPanel" v-if="hovered&&draft.introductions&&!detailsOpen" class="m17-hover-panel" :style="hoverStyle" :aria-label="tr('modules.ipa-plus.text.b89a88edf218')" role="tooltip" @mouseenter="keepHover" @mouseleave="hover(null)"><SymbolDetails :entry="hovered" compact/></aside>
   <aside v-if="helpOpen" class="m17-help-panel" :aria-label="tr('modules.ipa-plus.text.6397772a6b0c')"><button type="button" class="m17-help-close" @click="helpOpen=false">{{tr('modules.ipa-plus.text.b222169bce16')}}</button><h3>{{tr('modules.ipa-plus.text.b89d27a1c92f')}}</h3><p>{{tr('modules.ipa-plus.text.e29e145a3a1e')}}</p><p>{{tr('modules.ipa-plus.text.730f371bdc16')}}</p><p>{{tr('modules.ipa-plus.text.f15d5bc7f5cd')}}</p><p>{{tr('modules.ipa-plus.text.f50455d33d7c')}}</p><p>{{tr('modules.ipa-plus.text.ac3b6cca7894')}}</p><p>{{tr('modules.ipa-plus.text.c468f01851c1')}}</p><p>{{tr('modules.ipa-plus.text.f6a4989f3e63')}}</p><p>{{tr('modules.ipa-plus.text.ce2f539f3077')}}</p><details class="m17-font-license"><summary>{{tr('modules.ipa-plus.text.ab225400e587')}}</summary><pre>{{fontLicense}}

{{notoLicense}}</pre></details><small>{{catalog.version}} {{tr('modules.ipa-plus.text.0ecf00d75b7f')}}</small></aside>
  </div>
 </ModuleFrame>
</template>
<style scoped>
.ipa-plus-page{gap:4px;padding-top:4px;overflow:hidden}.ipa-plus-page :deep(.module-toolbar){padding-bottom:2px}.m17-tabs{display:flex;gap:3px}.m17-tabs button{min-width:61px;padding:4px 9px}.m17-search{width:190px;flex:1;max-width:350px}.m17-size-control{display:flex;align-items:center;gap:4px;font-size:0.857143rem;white-space:nowrap}.m17-size-control input{width:58px;min-height:29px;padding:3px 5px}.m17-toggle{display:flex;align-items:center;gap:5px;font-size:0.857143rem;white-space:nowrap}
.m17-toggle-unavailable{color:var(--muted);cursor:not-allowed}.m17-toggle-unavailable input{pointer-events:none}
.m17-body{position:relative;display:flex;flex-direction:column;flex:1;min-height:0;gap:0}.m17-chart-meta{display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:24px;font-size:0.785714rem;color:var(--muted)}
.m17-chart-viewport{flex:1;min-height:120px;overflow:auto;overscroll-behavior:contain;scrollbar-width:auto;scrollbar-color:var(--muted) var(--app);scrollbar-gutter:stable;padding-bottom:1px}
.m17-divider{height:8px;flex:none;display:flex;align-items:center;justify-content:center;cursor:row-resize;touch-action:none}.m17-divider span{height:3px;width:52px;border-radius:3px;background:var(--border)}.m17-divider:hover span{background:var(--accent)}
.m17-editor-area{flex:none;min-height:96px;max-height:44%}.m17-editor{display:block;width:100%;height:100%;min-height:0;resize:none;border:1px solid var(--border);border-radius:6px;color:var(--text);background:var(--panel);padding:8px 12px;line-height:1.65;tab-size:4}.m17-editor::placeholder{font-family:var(--font);font-size:1rem;color:var(--muted)}
.m17-editor-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:5px;padding:5px 0 2px;flex:none}.m17-editor-toolbar button{min-height:29px;padding:3px 9px;font-size:0.857143rem}.m17-editor-toolbar label{display:flex;align-items:center;gap:4px;font-size:0.857143rem}.m17-editor-toolbar input{width:59px;min-height:29px;padding:3px 5px}.m17-save-state{margin-left:auto;font-size:0.785714rem;color:var(--muted)}.m17-status{font-size:0.785714rem;color:var(--muted);display:flex;justify-content:space-between;gap:8px;min-height:19px;flex-wrap:wrap;overflow-wrap:anywhere}.m17-status code{max-width:50%;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.m17-detail-panel,.m17-hover-panel,.m17-help-panel{position:absolute;z-index:20;right:8px;top:28px;background:var(--panel);border:1px solid var(--border);border-radius:8px;box-shadow:var(--shadow);padding:14px;width:min(410px,calc(100% - 16px));max-height:calc(100% - 70px);overflow:auto;overscroll-behavior:contain}.m17-hover-panel{right:auto;bottom:auto;pointer-events:auto;box-sizing:border-box}.m17-panel-actions{display:flex;justify-content:space-between;gap:8px;margin-bottom:10px}.m17-panel-actions button,.m17-help-close{font-size:0.857143rem;min-height:28px;padding:3px 8px}.m17-help-panel{width:min(620px,calc(100% - 16px));display:flex;flex-direction:column;gap:10px}.m17-help-close{align-self:flex-end}.m17-help-panel p{font-size:0.928571rem}.m17-font-license summary{cursor:pointer;font-size:0.928571rem}.m17-font-license pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:230px;overflow:auto;font-size:0.785714rem;line-height:1.45;padding:8px;border:1px solid var(--border);margin-top:6px}
@container module (max-width:800px){.m17-chart-meta>span:nth-child(2){display:none}.m17-save-state{margin-left:0}.m17-search{width:140px}.m17-chart-viewport{min-height:110px}.m17-editor-area{max-height:34%}}
</style>
