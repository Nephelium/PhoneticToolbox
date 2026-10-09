<script setup lang="ts">
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
import fontLicense from '../../assets/ipa-plus/OFL-PTBIPAPlus.txt?raw';
import notoLicense from '../../assets/ipa-plus/OFL-Noto.txt?raw';

const props=withDefaults(defineProps<{stateKey?:string;active?:boolean}>(),{stateKey:'M17',active:true});
const emit=defineEmits<{dirty:[value:boolean];references:[]}>();
const writer=globalThis.crypto?.randomUUID?.()??'m17-'+Date.now().toString(36)+Math.random().toString(36).slice(2);
const draft=reactive(createDraft(writer)),history=new EditorHistory();
const editor=ref<HTMLTextAreaElement>(),page=ref<HTMLElement>(),chart=ref<HTMLElement>();
const loaded=ref(false),fontReady=ref(false),fontError=ref(''),dirty=ref(false),error=ref(''),notice=ref('');
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
function changeTextSize(event:Event){const input=event.target as HTMLInputElement;const value=Number(input.value);draft.textSize=Number.isFinite(value)&&input.value.trim()?Math.max(18,Math.min(54,Math.round(value))):26;input.value=String(draft.textSize);}
function captureSelection(){if(!editor.value||composing)return;draft.start=editor.value.selectionStart;draft.end=editor.value.selectionEnd;history.select(draft.start,draft.end);}
function snapshot():EditorSnapshot{return {text:draft.text,start:draft.start,end:draft.end};}
async function showSelection(focus=true){await nextTick();if(!editor.value)return;if(focus)editor.value.focus({preventScroll:true});editor.value.setSelectionRange(draft.start,draft.end);}
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
  if(!loaded.value||composing){notice.value=composing?'请先结束当前输入法组合，再点选符号。':'正在读取草稿。';return;}
  selected.value=entry;notice.value='';
  try{commit(insertSymbol(snapshot(),entry));if(entry.insertionMode==='combining'&&draft.start===entry.insertText.length)notice.value='已输入附加记号；通常放在基底字母之后，未补入虚线圆。';if(entry.representation)notice.value='已输入圈围的文本替代表示，详见该项介绍。';}catch(e){notice.value=(e as Error).message;}
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
  }catch(e){error.value=(e as Error).message;blocked.value=true;return false;}finally{saving=undefined;}})();
  return saving;
}
async function forkDraft(){if(saving)await saving;key='ipa-plus.v1.'+props.stateKey+'.'+writer;revision=0;blocked.value=false;dirty.value=true;if(await save())notice.value='已另存独立草稿。请同时导出文本，原冲突记录保持不变。';}
async function copyAll(){
  if(!draft.text){notice.value='文本为空。';return;}
  try{await copyText(draft.text);notice.value='已复制全部文字。';}
  catch{notice.value='剪贴板不可用，已全选文字。请按 Ctrl+C（macOS：⌘C）手工复制。';draft.start=0;draft.end=draft.text.length;void showSelection();}
}
function exportText(){const blob=new Blob([draft.text],{type:'text/plain;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='国际音标-'+new Date().toISOString().slice(0,10)+'.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);notice.value='已发起 UTF-8 文本下载，请确认目标文件已保存。';}
function selectAll(){draft.start=0;draft.end=draft.text.length;history.select(draft.start,draft.end);void showSelection();}
function clear(){if(composing)return;commit({text:'',start:0,end:0});notice.value='文字已清空，可撤销恢复。';}
function resizeDown(event:PointerEvent){const target=event.currentTarget as HTMLElement;target.setPointerCapture(event.pointerId);resizeStart={y:event.clientY,height:draft.editorHeight,scale:page.value?(page.value.getBoundingClientRect().width/page.value.offsetWidth||1):1};event.preventDefault();}
function resizeMove(event:PointerEvent){if(resizeStart)draft.editorHeight=Math.max(96,Math.min(360,resizeStart.height+(resizeStart.y-event.clientY)/resizeStart.scale));}
function resizeKey(event:KeyboardEvent){if(['ArrowUp','ArrowDown','Home','End'].includes(event.key)){event.preventDefault();draft.editorHeight=event.key==='Home'?96:event.key==='End'?360:Math.max(96,Math.min(360,draft.editorHeight+(event.key==='ArrowUp'?16:-16)));}}
function beforeUnload(event:BeforeUnloadEvent){if(dirty.value){event.preventDefault();event.returnValue='';}}
onMounted(async()=>{
  window.addEventListener('beforeunload',beforeUnload);
  window.addEventListener('resize',clearHover);
  try{const result=restoreDraft(await loadDraft(key),writer);if(disposed)return;Object.assign(draft,result.draft);revision=draft.revision;blocked.value=result.blocked;error.value=result.message;history.current=snapshot();}
  catch(e){error.value=(e as Error).message;blocked.value=true;}
  // Flush settings watchers while hydration is still read-only. Opening a second
  // window must not revise the saved record before that window edits anything.
  await nextTick();if(disposed)return;loaded.value=true;void showSelection(false);
  try{await document.fonts.load('24px "PTB-IPA-Plus"','z̥᫃ C⃝ 𝼆 V𐞀');if(disposed)return;if(![...document.fonts].some(f=>f.family==='PTB-IPA-Plus'&&f.status==='loaded'))throw Error('font missing');fontReady.value=true;}
  catch{fontError.value='固定音标字体载入失败。文字保留，请恢复模块资源后重开；当前不保证字形正确。';}
});
onBeforeUnmount(()=>{disposed=true;clearTimeout(saveTimer);clearTimeout(hoverTimer);window.removeEventListener('beforeunload',beforeUnload);window.removeEventListener('resize',clearHover);});
defineExpose({save});
</script>
<template>
 <ModuleFrame label="国际音标表Plus 工作区" class="ipa-plus-page" :style="{'--m17-font-size':draft.textSize+'px'}" fit>
  <template #toolbar><ModuleToolbar label="国际音标表与检索"><div class="m17-tabs" role="group" aria-label="选择音标体系"><button v-for="system in systems" :key="system.id" type="button" :aria-pressed="draft.system===system.id" :class="{primary:draft.system===system.id}" @click="draft.system=system.id;hovered=undefined">{{system.label}}</button></div><input v-model="query" class="m17-search" aria-label="查找符号、名称、码位或 CIN" placeholder="符号 / 名称 / U+ / CIN"><label class="m17-size-control">字号 <input :value="draft.textSize" type="number" min="18" max="54" step="1" aria-label="音标字号" @change="changeTextSize"></label><label class="m17-toggle"><input v-model="draft.introductions" type="checkbox">悬停介绍</label><label class="m17-toggle"><input v-model="playOnClick" type="checkbox">点击播放</label><template #actions><button type="button" @click="helpOpen=!helpOpen">帮助</button><button type="button" @click="emit('references')"><AppIcon name="book"/>方法与引用</button></template></ModuleToolbar></template>
  <div ref="page" class="m17-body" :data-font-ready="fontReady" :data-loaded="loaded">
   <div class="m17-chart-meta"><span>{{currentSystem.version}}</span><span>{{playOnClick?'点击符号播放演示':'深色符号＝独立输入 · 浅色＝完整示例'}}</span></div>
   <ModuleStatus v-if="fontError" kind="error" :message="fontError"/>
   <ModuleStatus v-if="error" kind="error" :message="error"><button class="primary" type="button" @click="forkDraft">另存独立草稿</button></ModuleStatus>
   <div ref="chart" class="m17-chart-viewport" @scroll="clearHover"><ChartView :system="draft.system" :query="query" :selected="selected?.id" :play-on-click="playOnClick" @insert="activate" @hover="hover" @inspect="inspect"/></div>
   <div class="m17-divider" role="separator" tabindex="0" aria-label="拖动调整文本区高度" aria-orientation="horizontal" :aria-valuenow="draft.editorHeight" :aria-valuemin="96" :aria-valuemax="360" @pointerdown="resizeDown" @pointermove="resizeMove" @pointerup="resizeStart=undefined" @pointercancel="resizeStart=undefined" @keydown="resizeKey"><span></span></div>
   <div class="m17-editor-area" :style="{height:draft.editorHeight+'px'}"><textarea ref="editor" :value="draft.text" class="m17-editor m17-ipa" :style="{fontSize:draft.textSize+'px'}" aria-label="国际音标文本" spellcheck="false" autocomplete="off" autocapitalize="off" :disabled="!loaded" placeholder="在此手动输入或粘贴，点击上方符号插入到光标处……" @input="input" @select="captureSelection" @keyup="captureSelection" @pointerup="captureSelection" @blur="captureSelection" @keydown="keydown" @beforeinput="beforeInput" @compositionstart="compositionStart" @compositionend="compositionEnd"/></div>
   <div class="m17-editor-toolbar"><button type="button" class="primary" @click="copyAll">复制全部</button><button type="button" @click="selectAll">全选</button><button type="button" :disabled="!canUndo" @click="undo">撤销</button><button type="button" :disabled="!canRedo" @click="redo">重做</button><button class="primary" type="button" @click="exportText">保存文本</button><button type="button" :disabled="!draft.text" @click="clear">清空</button><span class="m17-save-state" role="status">{{!loaded?'读取草稿…':dirty?'本机草稿未保存':'本机草稿已保存'}}</span></div>
   <div class="m17-status" role="status"><span>{{notice||`${Array.from(draft.text).length} 字符 · ${fontReady?'固定字体就绪':'字体加载中'} · 仅本机编辑`}}</span><code v-if="draft.end>draft.start&&selectedCodes">{{selectedCodes}}</code><span v-if="suspicious.length">检测到 {{suspicious.length}} 个控制／私用区字符，原文保留：{{suspicious.slice(0,4).map(c=>`${c.code}（第${c.position}码点）`).join('、')}}</span></div>
   <aside v-if="detailsOpen&&selected" class="m17-detail-panel" aria-label="符号详细介绍"><div class="m17-panel-actions"><button type="button" @click="activate(selected)">{{playOnClick?'播放此项':'输入此项'}}</button><button type="button" @click="detailsOpen=false;playing=false">收起</button></div><SymbolPlayback v-if="playing&&active" :key="playRequest" :entry="selected" :request="playRequest"/><SymbolDetails :entry="selected"/></aside>
   <aside ref="hoverPanel" v-if="hovered&&draft.introductions&&!detailsOpen" class="m17-hover-panel" :style="hoverStyle" aria-label="悬停符号介绍" role="tooltip" @mouseenter="keepHover" @mouseleave="hover(null)"><SymbolDetails :entry="hovered" compact/></aside>
   <aside v-if="helpOpen" class="m17-help-panel" aria-label="国际音标表Plus 帮助"><button type="button" class="m17-help-close" @click="helpOpen=false">收起帮助</button><h3>输入与保存</h3><p>点击符号替换选区，或插入到最后的光标处。浅色按钮写入完整例示。虚线圆只用于展示附加记号的位置。切表和搜索保留正文与撤销历史。</p><p>附加符号区按窗口宽度分成多列，平常显示名称与符号。名称上悬停可读简释，符号上悬停可读完整介绍。表区可以上下滚动，编辑框始终保留。顶部字号同时调整三套表内的音标与编辑框，范围为18–54；名称和解释保持原有字号。</p><p>范围工具包住所选内容。连音线与滑动箭头可连结选中的两个字素。Ctrl/⌘＋Z 撤销，Ctrl/⌘＋Shift＋Z 重做。中文输入法组合期间请先完成选词。</p><p>右键符号或 Alt＋Enter 可打开详细介绍。关闭悬停介绍后不再自动弹出，名称的简释提示仍可用。介绍窗在符号附近避开指针，移入后可阅读与滚动。</p><p>开启点击播放后，点击符号只播放演示，正文与选区保持。尚未添加素材时会显示提示。关闭后恢复点击输入。</p><p>草稿在本机自动保存。跨窗口冲突停止写入，保留当前文字和原记录。保存文本会发起 UTF-8 下载，请确认文件。复制只携带字符，外部软件需要相应字体。</p><p>PTB IPA Plus 1.000 保留 Doulos SIL 7.000 原字形，补充组合括号和圈围定位，按 OFL 1.1 随包提供。多字符圈围采用显式 ⟅…⟆ 文本替代，可在该项介绍中核对差异。</p><p>网页加载资源后可以断网编辑；浏览器关闭后的离线重开未提供 PWA 缓存保证。输入不会提交给服务器，来源链接只在主动打开时访问。</p><details class="m17-font-license"><summary>字体许可（OFL 1.1）</summary><pre>{{fontLicense}}

{{notoLicense}}</pre></details><small>{{catalog.version}} · IPA 衍生表 CC BY-SA 4.0 · extIPA 中文重排资料 CC BY-SA 3.0 · VoQS 自有分类说明，未打包论文图版</small></aside>
  </div>
 </ModuleFrame>
</template>
<style scoped>
.ipa-plus-page{gap:4px;padding-top:4px;overflow:hidden}.ipa-plus-page :deep(.module-toolbar){padding-bottom:2px}.m17-tabs{display:flex;gap:3px}.m17-tabs button{min-width:61px;padding:4px 9px}.m17-search{width:190px;flex:1;max-width:350px}.m17-size-control{display:flex;align-items:center;gap:4px;font-size:0.857143rem;white-space:nowrap}.m17-size-control input{width:58px;min-height:29px;padding:3px 5px}.m17-toggle{display:flex;align-items:center;gap:5px;font-size:0.857143rem;white-space:nowrap}
.m17-body{position:relative;display:flex;flex-direction:column;flex:1;min-height:0;gap:0}.m17-chart-meta{display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:24px;font-size:0.785714rem;color:var(--muted)}
.m17-chart-viewport{flex:1;min-height:120px;overflow:auto;overscroll-behavior:contain;scrollbar-width:auto;scrollbar-color:var(--muted) var(--app);scrollbar-gutter:stable;padding-bottom:1px}
.m17-divider{height:8px;flex:none;display:flex;align-items:center;justify-content:center;cursor:row-resize;touch-action:none}.m17-divider span{height:3px;width:52px;border-radius:3px;background:var(--border)}.m17-divider:hover span{background:var(--accent)}
.m17-editor-area{flex:none;min-height:96px;max-height:44%}.m17-editor{display:block;width:100%;height:100%;min-height:0;resize:none;border:1px solid var(--border);border-radius:6px;color:var(--text);background:var(--panel);padding:8px 12px;line-height:1.65;tab-size:4}.m17-editor::placeholder{font-family:var(--font);font-size:1rem;color:var(--muted)}
.m17-editor-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:5px;padding:5px 0 2px;flex:none}.m17-editor-toolbar button{min-height:29px;padding:3px 9px;font-size:0.857143rem}.m17-editor-toolbar label{display:flex;align-items:center;gap:4px;font-size:0.857143rem}.m17-editor-toolbar input{width:59px;min-height:29px;padding:3px 5px}.m17-save-state{margin-left:auto;font-size:0.785714rem;color:var(--muted)}.m17-status{font-size:0.785714rem;color:var(--muted);display:flex;justify-content:space-between;gap:8px;min-height:19px;flex-wrap:wrap;overflow-wrap:anywhere}.m17-status code{max-width:50%;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.m17-detail-panel,.m17-hover-panel,.m17-help-panel{position:absolute;z-index:20;right:8px;top:28px;background:var(--panel);border:1px solid var(--border);border-radius:8px;box-shadow:var(--shadow);padding:14px;width:min(410px,calc(100% - 16px));max-height:calc(100% - 70px);overflow:auto;overscroll-behavior:contain}.m17-hover-panel{right:auto;bottom:auto;pointer-events:auto;box-sizing:border-box}.m17-panel-actions{display:flex;justify-content:space-between;gap:8px;margin-bottom:10px}.m17-panel-actions button,.m17-help-close{font-size:0.857143rem;min-height:28px;padding:3px 8px}.m17-help-panel{width:min(620px,calc(100% - 16px));display:flex;flex-direction:column;gap:10px}.m17-help-close{align-self:flex-end}.m17-help-panel p{font-size:0.928571rem}.m17-font-license summary{cursor:pointer;font-size:0.928571rem}.m17-font-license pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:230px;overflow:auto;font-size:0.785714rem;line-height:1.45;padding:8px;border:1px solid var(--border);margin-top:6px}
@container module (max-width:800px){.m17-chart-meta>span:nth-child(2){display:none}.m17-save-state{margin-left:0}.m17-search{width:140px}.m17-chart-viewport{min-height:110px}.m17-editor-area{max-height:34%}}
</style>
