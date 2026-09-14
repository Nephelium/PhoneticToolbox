<script setup lang="ts">
import {ref,shallowRef,computed,watch,reactive,onMounted,onUnmounted,markRaw,nextTick} from 'vue';
import type {ResearchContext,ResearchFile} from '../../platform/research.ts';import {audioPreview} from '../../platform/research.ts';
import {portableAnnotation,lipTrack,type LipTrack,type AnnotationTarget} from '../../platform/annotation.ts';
import {host,workspace} from '../../state/workspace.ts';import {stop,play,pause,playback} from '../../state/audio.ts';
import WaveformViewport from '../../components/WaveformViewport.vue';import AudioTransport from '../../components/AudioTransport.vue';import ModalDialog from '../../components/ModalDialog.vue';
import AnnotationTracks from './AnnotationTracks.vue';import {createEditor,type Grid} from './editor.mjs';import {parseGrid,serializeGrid,decodeText,validateEditingGrid,validateEditingAudio,preferredGrid,selectedInterval} from './format.ts';import dictionary from './default.dict?raw';
const props=defineProps<{context:ResearchContext;stateKey:string;active:boolean}>();const emit=defineEmits<{references:[];close:[]}>();
const wave=workspace(props.stateKey),revision=ref(0),error=ref(''),notice=ref(''),busy=ref(false),saving=ref(false),files=ref<ResearchFile[]>([]),directory=ref(''),directoryLabel=ref('');
const source=shallowRef<ResearchFile|null>(null),gridFile=shallowRef<ResearchFile|null>(null),gridSha=ref(''),lipFile=shallowRef<ResearchFile|null>(null),lipSha=ref(''),lip=shallowRef<LipTrack|null>(null),lipOffset=ref(0),savedLipOffset=ref(0),savedGrid=ref('');
const filter=ref(''),suffix=ref('_webedit'),showOpen=ref(true),showWidth=ref(true),sidebar=ref(true),referenceName=ref(''),dictName=ref('内置词典'),labName=ref(''),label=ref(''),replace=ref(''),query=ref(''),durationInput=ref('3.2');
const dialog=ref<'replace'|'overwrite'|''>(''),composing=ref(false),picker=ref<HTMLInputElement>(),dictPicker=ref<HTMLInputElement>(),labPicker=ref<HTMLInputElement>(),refPicker=ref<HTMLInputElement>();
const port=props.context.files.annotation??portableAnnotation(props.context.files);
let ticket=0,alive=true,timer:ReturnType<typeof setInterval>,repeat:ReturnType<typeof setTimeout>|undefined;
let reading:AbortController|undefined,saveFlight:Promise<boolean>|undefined;
const targets=new Map<string,AnnotationTarget>();
const editor=createEditor({changed:changed,message:m=>{notice.value=m;}}),s=editor.state;
const controls=reactive(editor.controls);
editor.state.phoneDict=editor.parseDictText(dictionary);
const prefs=host.projects.read<{word?:string;phone?:string;suffix?:string}>('annotation.'+props.stateKey,{});
const wordName=ref(prefs.word??'words'),phoneName=ref(prefs.phone??'phones');suffix.value=prefs.suffix??'_webedit';s.wordTierName=wordName.value;s.phoneTierName=phoneName.value;
const lipDirty=computed(()=>!!lip.value&&lipOffset.value!==savedLipOffset.value);
const pairs=computed(()=>files.value.filter(f=>f.kind==='audio').map(audio=>{try{return {audio,grid:preferredGrid(audio,files.value)};}catch{return {audio,grid:undefined};}}).filter(p=>p.grid));
const visibleFiles=computed(()=>pairs.value.filter(p=>p.audio.name.toLowerCase().includes(filter.value.toLowerCase())));
const gridChoices=computed(()=>{const name=source.value?.name??'',parent=name.includes('/')?name.slice(0,name.lastIndexOf('/')+1):'';return files.value.filter(f=>f.kind==='textgrid'&&(!parent||f.name.startsWith(parent)));});
const lipChoices=computed(()=>{const name=source.value?.name??'',parent=name.includes('/')?name.slice(0,name.lastIndexOf('/')+1):'';return files.value.filter(f=>['lip','lip_pickle'].includes(f.kind)&&!f.name.toLowerCase().endsWith('_timestamps.pkl')&&(!parent||f.name.startsWith(parent)));});
const targetName=computed(()=>source.value?source.value.name.replace(/\.wav$/i,'')+suffix.value+'.TextGrid':'尚未选择文件');
const selected=computed(()=>{revision.value;return selectedInterval(s.textgrid,s.selected);});
// The migrated document is deliberately raw. Text changes retain object identity.
const labelDirty=computed(()=>{revision.value;return !!selected.value&&label.value!==selected.value.text;});
const matchCount=computed(()=>{revision.value;return s.searchResults.length;});
function changed(){
 if(!alive)return;revision.value++;wave.dirty=s.dirty||lipDirty.value;
 label.value=selectedInterval(s.textgrid,s.selected)?.text??'';
 if(s.selected){const item=selectedInterval(s.textgrid,s.selected);if(item){wave.start=item.xmin;wave.end=item.xmax;}}
 if(s.audioBuffer){wave.offset=s.visibleStart;wave.zoom=s.audioBuffer.duration/s.visibleDuration;}
}
function syncView(){s.visibleStart=wave.offset;s.visibleDuration=(wave.asset?.duration??3.2)/wave.zoom;durationInput.value=s.visibleDuration.toFixed(3);}
watch(()=>[wave.offset,wave.zoom],syncView);
watch([lipDirty,labelDirty,composing],()=>{wave.dirty=s.dirty||lipDirty.value||labelDirty.value||composing.value;});
function finishInput(){if(composing.value){error.value='请先完成输入法组字，再保存或切换。';return false;}if(labelDirty.value){editor.editText(label.value);changed();}return true;}
watch(suffix,()=>{persist();});
function persist(){return host.projects.write('annotation.'+props.stateKey,{word:wordName.value,phone:phoneName.value,suffix:suffix.value});}
function names(){
 const w=wordName.value.trim()||'words',p=phoneName.value.trim()||'phones';
 if(w===p){error.value='词层与音素层必须使用不同的层名。';return;}
 if([w,p].some(n=>s.textgrid?.tiers.find(t=>t.name===n)?.points)){error.value='编辑层必须是 IntervalTier，点层会原样保留。';return;}
 wordName.value=w;phoneName.value=p;s.wordTierName=w;s.phoneTierName=p;s.selected=null;s.selectedBoundary=null;s.selectedIndices=[];
 if(!persist())error.value='层名偏好保存失败，当前编辑仍保留。';else error.value='';changed();
}
async function prepare(role:'textgrid'|'lip',file:ResearchFile,suff='_webedit'){
 const key=role+':'+file.id+':'+suff;let target=targets.get(key);if(!target){target=await port.target(file,role,suff);targets.set(key,target);}return target;
}
function updateSaved(file:ResearchFile,original:ResearchFile){
 const parent=original.name.includes('/')?original.name.slice(0,original.name.lastIndexOf('/')+1):'';
 const named={...file,name:file.name.includes('/')?file.name:parent+file.name};
 files.value=files.value.filter(f=>f.id!==named.id&&f.name.toLowerCase()!==named.name.toLowerCase()).concat(named);return named;
}
async function saveGrid(auto=false,confirmed=false):Promise<boolean>{
 if(!finishInput())return false;
 if(!s.textgrid||!source.value||!gridFile.value)return true;
 if(!auto&&suffix.value===''&&!confirmed&&props.context.files.kind==='desktop'){dialog.value='overwrite';return false;}
 if(saving.value)return false;
 const generation=ticket,document=s.textgrid,original=gridFile.value,audio=source.value,sourceHash=gridSha.value,suff=auto?'_webedit':suffix.value;
 saving.value=true;error.value='';
 try{
  const text=serializeGrid(document),target=await prepare('textgrid',audio,suff);
  const result=await port.save({target:target.id,source:{id:original.id,sha256:sourceHash},text});
  if(alive&&generation===ticket&&s.textgrid===document){
   gridFile.value=updateSaved(result.file,original);gridSha.value=result.sha256;savedGrid.value=text;s.dirty=serializeGrid(document)!==text;wave.dirty=s.dirty||lipDirty.value;notice.value=(props.context.files.kind==='server'?'已保存项目版本：':'已保存：')+result.name;revision.value++;
  }return true;
 }catch(e){if(alive&&generation===ticket)error.value=(e as Error).message;return false;}finally{if(alive)saving.value=false;}
}
async function saveLip():Promise<boolean>{
 if(!lip.value||!lipFile.value||!lipDirty.value)return true;
 if(saving.value)return false;
 const generation=ticket,original=lipFile.value,hash=lipSha.value,offset=lipOffset.value;saving.value=true;error.value='';
 try{
  const target=await prepare('lip',original,''),result=await port.save({target:target.id,source:{id:original.id,sha256:hash},offset});
  if(alive&&generation===ticket){lipFile.value=updateSaved(result.file,original);lipSha.value=result.sha256;savedLipOffset.value=offset;wave.dirty=s.dirty||lipDirty.value;notice.value='唇偏已独立保存：'+result.name;}return true;
 }catch(e){if(alive&&generation===ticket)error.value=(e as Error).message;return false;}finally{if(alive)saving.value=false;}
}
function savePending():Promise<boolean>{
 if(saveFlight)return saveFlight;
 saveFlight=(async()=>{if(!finishInput())return false;if(s.dirty&&!await saveGrid(true))return false;if(lipDirty.value&&!await saveLip())return false;if(s.dirty||lipDirty.value||labelDirty.value||composing.value){error.value='保存期间有新的编辑，请再次保存后切换。';return false;}return true;})().finally(()=>{saveFlight=undefined;});return saveFlight;
}
function download(role:'textgrid'|'lip'){
 if(!finishInput())return;
 try{
  let text:string,name:string;
  if(role==='textgrid'){if(!s.textgrid||!source.value)return;text=serializeGrid(s.textgrid);name=targetName.value.split('/').at(-1)!;}
  else{if(!lip.value||!lipFile.value)return;const wire=structuredClone(lip.value.wire);wire.data.metadata??={};wire.data.metadata.lip_manual_offset=lipOffset.value;text=JSON.stringify(wire);name=lipFile.value.name.split('/').at(-1)!.replace(/\.pkl$/i,'.lip.json');}
  const url=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notice.value='已发起下载：'+name+'。当前编辑状态保留。';
 }catch(e){error.value=(e as Error).message;}
}
async function scan(choose=false){
 if(busy.value||saving.value)return;error.value='';
 if(!await savePending())return;busy.value=true;
 try{
  let next=directory.value;
  if(choose&&props.context.files.choose){const grant=await props.context.files.choose('input');if(!grant)return;next=grant.id;directoryLabel.value=grant.label;}
  if(props.context.files.kind==='desktop'&&!next)return;
  const values=await port.scan(next);if(!alive)return;files.value=values;directory.value=next;
  notice.value=`已扫描 ${values.filter(f=>f.kind==='audio').length} 个 WAV，${pairs.value.length} 组可编辑标注`;
 }catch(e){error.value=(e as Error).message;}finally{busy.value=false;}
}
async function addFiles(event:Event){const element=event.target as HTMLInputElement;props.context.files.add?.(Array.from(element.files??[]));element.value='';await scan();}
async function open(audio:ResearchFile,explicitGrid?:ResearchFile){
 if(busy.value||saving.value)return;if(!await savePending())return;
 const own=++ticket;reading?.abort();reading=new AbortController();busy.value=true;error.value='';notice.value='正在读取音频与标注…';stop();
 try{
  const grid=explicitGrid??preferredGrid(audio,files.value);if(!grid)throw Error('没有匹配的 TextGrid。');
  const [decoded,read]=await Promise.all([audioPreview(props.context.files,audio,reading.signal),props.context.files.read(grid,reading.signal)]);
  validateEditingAudio(decoded.asset);const document=validateEditingGrid(parseGrid(decodeText(read.buffer)),decoded.asset.duration);
  if(!alive||own!==ticket)return;
  source.value={...audio,sha256:decoded.sha256};gridFile.value=grid;gridSha.value=read.sha256;wave.asset=markRaw(decoded.asset);wave.channel=0;wave.offset=0;wave.zoom=Math.max(1,decoded.asset.duration/3.2);wave.start=0;wave.end=decoded.asset.duration;
  s.textgrid=document;s.audioBuffer={duration:decoded.asset.duration,sampleRate:decoded.asset.sampleRate,getChannelData:index=>decoded.asset.channels[index]};s.visibleStart=0;s.visibleDuration=decoded.asset.duration/wave.zoom;
  s.selected=null;s.selectedBoundary=null;s.selectedIndices=[];s.undoStack=[];s.drag=null;s.dirty=false;s.copiedWord='';s.copiedLabIndex=null;s.searchResults=[];s.searchIndex=-1;query.value='';
  savedGrid.value=serializeGrid(document);lip.value=null;lipFile.value=null;lipOffset.value=0;savedLipOffset.value=0;s.labSequence=[];s.labWords=new Set();labName.value='';
  controls.fitStart.value='0';controls.fitEnd.value=decoded.asset.duration.toFixed(3);controls.spliceStart.value='0';controls.spliceEnd.value=decoded.asset.duration.toFixed(3);targets.clear();
  await prepare('textgrid',source.value,'_webedit');
  const lab=files.value.find(f=>f.kind==='lab'&&f.name.toLowerCase()===audio.name.replace(/\.wav$/i,'.lab').toLowerCase());
  if(lab){const bytes=await props.context.files.read(lab);if(own!==ticket||!alive)return;setLab(decodeText(bytes.buffer),lab.name);}
  const stem=audio.name.replace(/\.wav$/i,''),parent=audio.name.includes('/')?audio.name.slice(0,audio.name.lastIndexOf('/')+1):'';
  const preferred=[stem+'.lip.json',stem+'.pkl',parent+'audio_recording.lip.json',parent+'audio_recording.pkl'];
  const candidates=preferred.map(name=>files.value.find(f=>f.name.toLowerCase()===name.toLowerCase())).filter(Boolean) as ResearchFile[];
  // Fallback only in a single-recording directory, so unrelated WAVs never share a PKL silently.
  const candidate=candidates.find(f=>f.name.toLowerCase().startsWith(stem.toLowerCase()+'.'))??(files.value.filter(f=>f.kind==='audio'&&(f.name.includes('/')?f.name.slice(0,f.name.lastIndexOf('/')+1):'')===parent).length===1?candidates[0]:undefined);
  if(candidate)try{await loadLip(candidate,own);}catch(e){error.value='标注已加载；唇形读取失败：'+(e as Error).message;}
  if(own===ticket&&alive){notice.value='已加载：'+grid.name;syncView();changed();}
 }catch(e){if(alive&&own===ticket)error.value=(e as Error).message;}finally{if(alive&&own===ticket)busy.value=false;}
}
async function loadLip(file:ResearchFile,own=ticket){
 const read=await port.lip(file),track=lipTrack(read.wire);await prepare('lip',file,'');if(own!==ticket||!alive)return;
 lip.value=markRaw(track);lipFile.value=file;lipSha.value=read.sha256;lipOffset.value=track.offset;savedLipOffset.value=track.offset;
}
async function chooseLip(id:string){if(busy.value||saving.value)return;if(!await saveLip())return;if(lipDirty.value){error.value='保存期间有新的唇偏编辑，请再次保存后切换。';return;}const file=files.value.find(f=>f.id===id);if(!file){lip.value=null;lipFile.value=null;return;}busy.value=true;try{await loadLip(file);error.value='';}catch(e){error.value=(e as Error).message;}finally{busy.value=false;}}
function setLab(text:string,name:string){if(text.length>2_000_000)throw Error('词表超过 2 MB。');const words=text.split(/\s+/).filter(Boolean);if(!words.length)throw Error('词表为空。');s.labSequence=words;s.labWords=new Set(words.map(w=>w.toLowerCase()));s.copiedLabIndex=null;labName.value=name;revision.value++;}
async function resource(event:Event,kind:'dict'|'lab'|'reference'){
 const input=event.target as HTMLInputElement,file=input.files?.[0],own=ticket;if(!file)return;input.value='';
 try{if(file.size>2_000_000)throw Error('文本资源超过 2 MB。');const text=decodeText(await file.arrayBuffer());if(!alive||ticket!==own)return;
  if(kind==='dict'){const map=editor.parseDictText(text);if(!map.size||text.split(/\r?\n/).some(l=>l.trim()&&l.trim().split(/\s+/).length<2))throw Error('词典每行需要：词 音素1 音素2 …');s.phoneDict=map;dictName.value=file.name+' · '+map.size+' 条';}
  else if(kind==='lab')setLab(text,file.name);else{s.referenceTextGrid=parseGrid(text);referenceName.value=file.name;}
  error.value='';notice.value='已加载：'+file.name;changed();
 }catch(e){if(ticket===own&&alive)error.value=(e as Error).message;}
}
function view(){const value=Number(durationInput.value),d=wave.asset?.duration??0;if(!Number.isFinite(value)||value<.08||value>d){error.value='视窗需要在 0.08 秒与文件时长之间。';return;}wave.zoom=d/value;wave.offset=Math.min(wave.offset,d-value);syncView();}
function nudge(ms:number){if(!lip.value)return;const value=lipOffset.value+ms/1000;if(Math.abs(value)<=3600)lipOffset.value=Number(value.toFixed(6));}
function stopRepeat(){if(repeat)clearTimeout(repeat);repeat=undefined;}
function startRepeat(ms:number){stopRepeat();nudge(ms);repeat=setTimeout(function tick(){nudge(ms);repeat=setTimeout(tick,40);},150);}
function offsetInput(event:Event){const value=(event.target as HTMLInputElement).value;if(!value.trim()||!Number.isFinite(Number(value))||Math.abs(Number(value))>3_600_000){error.value='唇偏毫秒数无效。';return;}lipOffset.value=Number(value)/1000;error.value='';}
function search(){controls.searchInput.value=query.value;editor.doSearch(query.value);changed();}
async function action(name:'fit'|'splice'|'phones'|'delete'){
 error.value='';try{
  if(!s.textgrid)throw Error('请先选择音频和标注。');
  if(name==='splice'){if(!s.referenceTextGrid)throw Error('请先选择参考 TextGrid。');await editor.applyReferenceSplice();}
  else{if(!editor.wordTier())throw Error('找不到词层，请检查层名。');if(name==='fit')editor.fitIntensityRange();else if(name==='phones')editor.autoPhonesForSelection();else{editor.saveUndoState();editor.deleteSelectedBoundary();}}
  changed();
 }catch(e){error.value=(e as Error).message;}
}
function key(event:KeyboardEvent){
 if(!props.active||!s.textgrid||dialog.value||busy.value||saving.value||event.isComposing)return;
 const element=event.target as HTMLElement;if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(element.tagName)||element.isContentEditable)return;
 const k=event.key.toLowerCase();
 if(event.altKey&&['ArrowLeft','ArrowRight'].includes(event.key)){nudge((event.key==='ArrowLeft'?-1:1)*(event.shiftKey?10:1));}
 else if(event.altKey&&event.key==='Backspace'){void action('delete');}
 else if(event.ctrlKey&&k==='z'){editor.undo();}
 else if(event.ctrlKey&&k==='c'){editor.copy();}
 else if(event.ctrlKey&&k==='v'){editor.pasteCopiedWord();}
 else if(event.ctrlKey&&k==='s'){void saveGrid();}
 else if(event.key==='Backspace'){editor.clearText();}
 else if(k==='p'||event.key===' '){if(playback.playing)pause();else if(wave.asset)void play(wave.asset,wave.offset,wave.asset.duration,0);}
 else if(event.key.length===1&&event.key!==' '&&!event.ctrlKey&&!event.altKey&&!event.metaKey){editor.editText((selectedInterval(s.textgrid,s.selected)?.text??'')+event.key);}
 else return;event.preventDefault();changed();
}
watch(()=>props.active,active=>{if(!active)stopRepeat();});
onMounted(()=>{window.addEventListener('keydown',key);window.addEventListener('blur',stopRepeat);timer=setInterval(()=>{if(props.context.files.kind!=='preview'&&!busy.value&&!saving.value&&!dialog.value&&!composing.value&&(s.dirty||lipDirty.value||labelDirty.value))void savePending();},60000);if(props.context.files.kind==='server')void scan();});
onUnmounted(()=>{alive=false;++ticket;reading?.abort();clearInterval(timer);stopRepeat();window.removeEventListener('keydown',key);window.removeEventListener('blur',stopRepeat);});
defineExpose({save:savePending});
</script>
<template>
<section class="annotation-page" :data-revision="revision" :aria-busy="busy||saving">
 <header class="annotation-header"><div><p class="eyebrow">M12 · {{context.files.kind==='server'?'网页项目':'本机语料'}}</p><h1>语音标注对齐</h1><p>校订词与音素边界，对照声音与唇形。</p></div><div class="actions"><button @click="emit('references')">方法与引用</button><button @click="emit('close')">关闭模块</button></div></header>
 <p v-if="error" role="alert" class="error-text">{{error}}</p><p v-if="notice" class="notice" role="status">{{notice}}</p>
 <div class="annotation-layout" :class="{collapsed:!sidebar}">
 <aside v-show="sidebar" class="annotation-files annotation-card">
  <h2>语料文件</h2><div class="actions"><button v-if="context.files.choose" :disabled="busy||saving" @click="scan(true)">选择语料文件夹</button><button v-else-if="context.files.add" :disabled="busy||saving" @click="picker?.click()">打开语料文件</button><button :disabled="busy||saving" @click="scan()">扫描</button></div>
  <input ref="picker" hidden type="file" multiple accept=".wav,.TextGrid,.textgrid,.lab,.json" @change="addFiles"/><small>{{directoryLabel||context.label}} · {{pairs.length}} 组 WAV / TextGrid</small>
  <input v-model="filter" aria-label="筛选标注文件" placeholder="筛选文件…"/>
  <div class="annotation-file-list"><button v-for="pair in visibleFiles" :key="pair.audio.id" :class="{selected:source?.id===pair.audio.id}" :disabled="busy||saving" :title="pair.audio.name" @click="open(pair.audio,pair.grid)"><span>{{pair.audio.name}}</span><small>{{pair.grid?.name}}</small></button><p v-if="!visibleFiles.length" class="hint">选择包含 WAV 和同名 TextGrid 的目录。支持子目录及 _webedit、_post、_auto 标注。</p></div>
 </aside>
 <main class="annotation-editor">
  <div class="annotation-card edit-toolbar"><div class="actions"><button :aria-pressed="sidebar" @click="sidebar=!sidebar">文件列表</button><button :disabled="!source||busy||saving" class="primary" @click="saveGrid()">{{saving?'正在保存…':'保存 TextGrid'}}{{s.dirty?' *':''}}</button><button :disabled="!selected||busy" @click="action('phones')">音素自动填充</button><button :disabled="!s.undoStack.length||busy" @click="editor.undo();changed()">撤销</button><button :disabled="!selected" @click="editor.copy()">复制词</button><button :disabled="!selected||!s.copiedWord" @click="editor.pasteCopiedWord();changed()">连续粘贴</button></div>
   <div class="actions"><button @click="dictPicker?.click()">上传词典</button><small>{{dictName}}</small><button @click="labPicker?.click()">上传词表</button><button v-if="labName" @click="s.labSequence=[];s.labWords=new Set();s.copiedLabIndex=null;labName='';changed()">清除词表</button><small>{{labName}}</small></div>
   <input ref="dictPicker" hidden type="file" accept=".dict,.txt" @change="resource($event,'dict')"/><input ref="labPicker" hidden type="file" accept=".lab,.txt" @change="resource($event,'lab')"/>
  </div>
  <div class="annotation-card search-tools"><div class="actions"><input v-model="query" aria-label="搜索词层文本" placeholder="搜索词层文本" @input="search" @keydown.enter.prevent="$event.shiftKey?editor.findPrev():editor.findNext();changed()"/><span class="mono">{{matchCount?s.searchIndex+1:0}} / {{matchCount}}</span><button :disabled="!matchCount" @click="editor.findPrev();changed()">上一个</button><button :disabled="!matchCount" @click="editor.findNext();changed()">下一个</button><input v-model="replace" aria-label="替换文本" placeholder="替换为…"/><button :disabled="!matchCount" @click="editor.replaceCurrent(replace);changed()">替换</button><button :disabled="!matchCount" @click="dialog='replace'">全部替换</button></div>
   <div class="actions"><strong>强度贴合</strong><label>起点 <input v-model="controls.fitStart.value" aria-label="强度起点" type="number" step=".001"/> s</label><label>终点 <input v-model="controls.fitEnd.value" aria-label="强度终点" type="number" step=".001"/> s</label><label>内收/外扩 <input v-model="controls.fitTrimMs.value" aria-label="强度内收毫秒" type="number" min="-50" max="80"/> ms</label><button :disabled="!source||busy" @click="action('fit')">强度贴合</button><small>正数内收，负数外扩 · 第一声道</small></div>
  </div>
  <div v-if="wave.asset" class="annotation-card annotation-plots">
   <div class="actions view-controls"><strong>{{source?.name}}</strong><label>视窗 <input v-model="durationInput" aria-label="标注可视时长" type="number" step=".1" @change="view"/> s</label><button @click="wave.offset=Math.max(0,wave.offset-s.visibleDuration*.8)">前一窗</button><button @click="wave.offset=Math.min(wave.asset!.duration-s.visibleDuration,wave.offset+s.visibleDuration*.8)">后一窗</button></div>
   <WaveformViewport :state="wave" :compact-overview="true" @selection-end="(a,b)=>{controls.fitStart.value=a.toFixed(3);controls.fitEnd.value=b.toFixed(3)}"/>
   <AnnotationTracks :editor="editor" :revision="revision" :wave="wave" :lip="lip" :offset="lipOffset" :show-open="showOpen" :show-width="showWidth" :active="active" @changed="changed"/>
   <div class="label-editor"><label>选中区间文本 <input v-model="label" aria-label="编辑选中标注文本" :disabled="!selected" @compositionstart="composing=true" @compositionend="composing=false;finishInput()" @change="finishInput()" @keydown.enter="!$event.isComposing&&finishInput()"/></label><span v-if="selected" class="mono">{{selected.xmin.toFixed(6)}}–{{selected.xmax.toFixed(6)}} s</span><button :disabled="!selected" @click="editor.clearText();changed()">清空文本</button><button :disabled="!s.selectedBoundary" @click="action('delete')">删除边界</button></div>
   <AudioTransport :state="wave" :active="false"/>
   <p class="hint">Ctrl＋滚轮缩放 · Shift＋滚轮平移标注视图 · Ctrl＋Z 撤销 · Alt＋Backspace 合并边界。输入框支持中文输入法。</p>
  </div>
  <div v-else class="annotation-card annotation-empty"><h2>打开一组录音与标注</h2><p>选择语料目录后，从左侧列表开始编辑。音频、TextGrid 和唇形共用秒时间轴。</p></div>
 </main>
 <aside class="annotation-settings annotation-card">
  <h2>层级与保存</h2><label>词层名<input v-model="wordName" aria-label="词层名" @change="names"/></label><label>音素层名<input v-model="phoneName" aria-label="音素层名" @change="names"/></label>
  <label>当前 TextGrid<select :value="gridFile?.id??''" aria-label="当前 TextGrid" :disabled="!source||busy||saving" @change="open(source!,files.find(f=>f.id===($event.target as HTMLSelectElement).value))"><option v-for="f in gridChoices" :key="f.id" :value="f.id">{{f.name}}</option></select></label>
  <label>保存后缀<input v-model="suffix" aria-label="TextGrid 保存后缀"/></label><p class="target-preview">目标：{{targetName}}</p><p v-if="suffix===''" class="overwrite-note">{{context.files.kind==='desktop'?'留空将覆盖 WAV 同名原始 TextGrid，保存时确认。':'网页将保存同名的新版本，原资源保留。'}}</p>
  <small>切换文件前及每分钟自动保存到 _webedit。TextGrid 与唇偏分别保存。</small><button :disabled="!source" @click="download('textgrid')">下载当前 TextGrid</button><button v-if="lip" @click="download('lip')">下载安全唇形 JSON</button>
  <hr/><h2>参考标注复用</h2><div class="actions"><button @click="refPicker?.click()">选择参考 TextGrid</button><button v-if="referenceName" @click="s.referenceTextGrid=null;referenceName=''">清除参考</button></div><input ref="refPicker" hidden type="file" accept=".TextGrid,.textgrid" @change="resource($event,'reference')"/><small>{{referenceName||'未选择参考文件'}}</small>
  <label>复用模式<select v-model="controls.spliceMode.value" aria-label="参考复用模式"><option value="outside">区间之外</option><option value="inside">区间之内</option><option value="before">起点之前</option><option value="after">起点之后</option></select></label>
  <label>起点（秒）<input v-model="controls.spliceStart.value" aria-label="参考起点" type="number" step=".001"/></label><label>终点（秒）<input v-model="controls.spliceEnd.value" aria-label="参考终点" type="number" step=".001" :disabled="['before','after'].includes(controls.spliceMode.value)"/></label>
  <button :disabled="!source||busy" @click="action('splice')">复用参考标注</button><small>对所有同名层应用所选范围，可撤销。起点之后从起点延伸到文件末尾。</small>
  <hr/><h2>唇形对齐</h2><label>唇形记录<select :value="lipFile?.id??''" aria-label="唇形记录" :disabled="!source||busy||saving" @change="chooseLip(($event.target as HTMLSelectElement).value)"><option value="">不关联</option><option v-for="f in lipChoices" :key="f.id" :value="f.id">{{f.name}}</option></select></label>
  <template v-if="lip"><div class="actions"><label><input v-model="showOpen" type="checkbox"/>唇开</label><label><input v-model="showWidth" type="checkbox" :disabled="!lip.width.length"/>唇宽</label></div><label>共同时间偏移（ms）<input :value="Number((lipOffset*1000).toFixed(3))" aria-label="唇形共同偏移毫秒" type="number" step="1" @change="offsetInput"/></label><div class="actions"><button aria-label="唇形左移" @pointerdown="startRepeat($event.shiftKey?-10:-1)" @pointerup="stopRepeat" @pointerleave="stopRepeat" @pointercancel="stopRepeat" @keydown.enter.prevent="nudge(-1)">← 1 ms</button><button aria-label="唇形右移" @pointerdown="startRepeat($event.shiftKey?10:1)" @pointerup="stopRepeat" @pointerleave="stopRepeat" @pointercancel="stopRepeat" @keydown.enter.prevent="nudge(1)">1 ms →</button></div><button :disabled="!lipDirty||saving||busy" @click="saveLip">保存唇偏{{lipDirty?' *':''}}</button><small>正值向更晚平移。按住连续微调，Shift 为 10 ms；Alt＋方向键同效。两条曲线只有一个偏移。</small></template>
  <p v-else class="hint">找到同名唇形记录时自动关联，也可明确选择。网页使用 .lip.json。</p>
 </aside>
 </div>
 <ModalDialog v-if="dialog" :title="dialog==='replace'?'确认全部替换':'确认覆盖原始标注'" :close-disabled="saving" @close="dialog=''">
  <p v-if="dialog==='replace'">将当前文件词层的 {{matchCount}} 个匹配区间替换为 {{replace||'空文本'}}，同时按词典重新填充音素。可用撤销恢复。</p><p v-else>保存目标：{{targetName}}。已有同名文件会在版本核对后被覆盖，唇偏保持独立保存。</p><p v-if="error" role="alert" class="error-text">{{error}}</p>
  <template #footer><button :disabled="saving" @click="dialog=''">取消</button><button class="primary" :disabled="saving" @click="dialog==='replace'?(editor.replaceAll(replace),changed(),dialog=''):saveGrid(false,true).then(ok=>{if(ok)dialog=''})">{{dialog==='replace'?'替换全部':'确认保存'}}</button></template>
 </ModalDialog>
</section>
</template>
<style scoped>
.annotation-page{min-height:100%;padding:20px;display:flex;flex-direction:column;gap:12px}.annotation-header{display:flex;justify-content:space-between;gap:16px;align-items:start}.annotation-header h1{font-size:23px}.annotation-header p:not(.eyebrow){color:var(--muted);font-size:13px;margin-top:5px}.actions{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.annotation-layout{display:grid;grid-template-columns:210px minmax(350px,1fr) 230px;gap:12px;align-items:start}.annotation-layout.collapsed{grid-template-columns:minmax(350px,1fr) 230px}.annotation-card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:12px;min-width:0}.annotation-files{display:flex;flex-direction:column;gap:10px}.annotation-file-list{max-height:630px;overflow:auto;display:flex;flex-direction:column;gap:4px}.annotation-file-list button{display:flex;flex-direction:column;align-items:start;text-align:left;min-width:0;white-space:normal;padding:8px}.annotation-file-list span,.annotation-file-list small{overflow-wrap:anywhere}.annotation-file-list .selected{border-color:var(--accent);background:var(--selected)}.annotation-editor{display:flex;flex-direction:column;gap:10px;min-width:0}.edit-toolbar,.search-tools{display:flex;flex-direction:column;gap:10px}.search-tools>.actions+.actions{padding-top:10px;border-top:1px solid var(--border)}.search-tools input:not([type=number]){width:150px}.search-tools label,.view-controls label{display:flex;align-items:center;gap:4px;font-size:12px}.search-tools input[type=number],.view-controls input{width:78px}.search-tools strong{font-size:12px}.annotation-settings{display:flex;flex-direction:column;gap:9px}.annotation-settings>label{display:flex;flex-direction:column;gap:4px;font-size:12px}.annotation-settings hr{margin:8px 0}.annotation-settings small{line-height:1.65}.target-preview{font-size:12px;overflow-wrap:anywhere;background:var(--app);padding:8px;border-radius:4px}.overwrite-note{color:var(--warning);font-size:12px}.annotation-plots :deep(.wave-track svg){height:110px}.view-controls strong{font-size:12px;max-width:260px;overflow-wrap:anywhere}.label-editor{display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding:12px 0}.label-editor label{font-size:12px;display:flex;gap:8px;align-items:center}.label-editor input{font-family:var(--font-figure-ipa);font-size:var(--figure-size,14px);width:200px}.annotation-empty{padding:60px 24px;text-align:center}.annotation-empty p{margin-top:10px;color:var(--muted)}.notice{font-size:12px;color:var(--teal);overflow-wrap:anywhere}.error-text{overflow-wrap:anywhere}.annotation-settings input,.annotation-settings select{width:100%}.annotation-settings input[type=checkbox]{width:16px}
@media(max-width:1250px){.annotation-layout{grid-template-columns:180px minmax(350px,1fr)}.annotation-layout.collapsed{grid-template-columns:1fr}.annotation-settings{grid-column:1/-1;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px 16px}.annotation-settings h2,.annotation-settings hr{grid-column:1/-1}.annotation-settings hr{width:100%}.annotation-file-list{max-height:600px}}
@media(max-width:780px){.annotation-page{padding:12px}.annotation-header{flex-direction:column}.annotation-layout,.annotation-layout.collapsed{display:flex;flex-direction:column}.annotation-files,.annotation-settings,.annotation-editor{width:100%}.annotation-file-list{max-height:160px}.annotation-settings{grid-template-columns:repeat(2,minmax(0,1fr))}.annotation-plots{padding:8px}.search-tools input:not([type=number]){width:125px}}
@media(max-width:480px){.annotation-settings{display:flex}.label-editor label{width:100%}.label-editor input{width:130px}.annotation-header .actions{flex-wrap:wrap}}
</style>
