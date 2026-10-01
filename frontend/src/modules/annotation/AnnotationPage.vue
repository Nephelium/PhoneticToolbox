<script setup lang="ts">
import {vResizablePanels} from '../../layout/resizablePanels.ts';
import {annotationShortcut} from './shortcuts.ts';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import {ref,shallowRef,computed,watch,reactive,onMounted,onUnmounted,markRaw,nextTick} from 'vue';
import type {ResearchContext,ResearchFile} from '../../platform/research.ts';import {annotationAudio} from '../../platform/annotation.ts';
import {portableAnnotation,lipTrack,type LipTrack,type AnnotationTarget} from '../../platform/annotation.ts';
import {host,workspace} from '../../state/workspace.ts';import {stop,play,pause,playback,volume,isCurrentAudio} from '../../state/audio.ts';
import WaveformViewport from '../../components/WaveformViewport.vue';import ModalDialog from '../../components/ModalDialog.vue';
import AnnotationTracks from './AnnotationTracks.vue';import {createEditor,type Grid} from './editor.mjs';import {parseGrid,serializeGrid,decodeText,loadEditingGrid,validateEditingAudio,preferredGrid,selectedInterval} from './format.ts';import dictionary from './default.dict?raw';
import {sequenceDoubleClick,manualDoubleClick,resetSequence,sequenceParts,hasUnsplitPhones,splitInitialPhones,type PhoneSplitMode} from './sequence.ts';
import {editableTier,editingNames,newIntervalTiers,editorSelection} from './layers.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active:boolean}>();const emit=defineEmits<{references:[];close:[]}>();
const wave=workspace(props.stateKey),revision=ref(0),error=ref(''),notice=ref(''),busy=ref(false),saving=ref(false),files=ref<ResearchFile[]>([]),directory=ref(''),directoryLabel=ref('');
const source=shallowRef<ResearchFile|null>(null),gridFile=shallowRef<ResearchFile|null>(null),gridSha=ref(''),lipFile=shallowRef<ResearchFile|null>(null),lipSha=ref(''),lip=shallowRef<LipTrack|null>(null),lipOffset=ref(0),savedLipOffset=ref(0),savedGrid=ref('');
const filter=ref(''),suffix=ref('_自动保存'),showOpen=ref(true),showWidth=ref(true),sidebar=ref(true),referenceName=ref(''),dictName=ref('内置词典'),labName=ref(''),label=ref(''),replace=ref(''),query=ref(''),durationInput=ref('3.2');
const settingsOpen=ref(host.projects.read('m12-settings-open:'+props.stateKey,true));
watch(settingsOpen,value=>host.projects.write('m12-settings-open:'+props.stateKey,value));
const dialog=ref<'replace'|'overwrite'|'words'|'layers'|'association'|''>(''),composing=ref(false),picker=ref<HTMLInputElement>(),dictPicker=ref<HTMLInputElement>(),labPicker=ref<HTMLInputElement>(),refPicker=ref<HTMLInputElement>();
const newWord=ref(''),newPhone=ref(''),layerRole=ref<'both'|'word'|'phone'>('both'),associationAudio=shallowRef<ResearchFile|null>(null),associationId=ref('');
const sequenceEnabled=ref(false),sequenceText=ref(''),nudgeStep=ref('1');
const port=props.context.files.annotation??portableAnnotation(props.context.files);
let ticket=0,alive=true,timer:ReturnType<typeof setInterval>,repeat:ReturnType<typeof setTimeout>|undefined;
let reading:AbortController|undefined,saveFlight:Promise<boolean>|undefined;
const previewNote=ref('');
const targets=new Map<string,AnnotationTarget>();
const editor=createEditor({changed:changed,message:m=>{notice.value=m;}}),s=editor.state;
const controls=reactive(editor.controls);
editor.state.phoneDict=editor.parseDictText(dictionary);
const prefs=host.projects.read<{word?:string;phone?:string;suffix?:string;phoneSplit?:PhoneSplitMode}>('annotation.'+props.stateKey,{});
const phoneSplit=ref<PhoneSplitMode>(prefs.phoneSplit==='equal'?'equal':'cursor');
const wordName=ref(''),phoneName=ref('');suffix.value=!prefs.suffix||prefs.suffix==='_webedit'?(prefs.suffix===''?'':'_自动保存'):prefs.suffix;s.wordTierName='';s.phoneTierName='';
const tierChoices=computed(()=>{revision.value;return s.textgrid?.tiers.map(t=>({name:t.name,editable:editableTier(s.textgrid!,t),kind:t.points?'点层':'非全域区间层'}))??[];});
const hasLayers=computed(()=>{revision.value;return !!s.textgrid?.tiers.length;});
const editingReady=computed(()=>{revision.value;return !!editor.wordTier()&&!!editor.phoneTier();});
const boundaries=computed(()=>{revision.value;const times=new Map<number,string>();for(const [tier,kind] of [[editor.phoneTier(),'phone'],[editor.wordTier(),'word']] as const)for(const item of tier?.intervals??[])for(const t of [item.xmin,item.xmax])times.set(t,kind);return [...times].map(([time,kind])=>({time,kind}));});
const lipDirty=computed(()=>!!lip.value&&lipOffset.value!==savedLipOffset.value);
const pairs=computed(()=>files.value.filter(f=>f.kind==='audio').map(audio=>{try{return {audio,grid:preferredGrid(audio,files.value)};}catch{return {audio,grid:undefined};}}));
function associatedGrids(audio:ResearchFile){const stem=audio.name.replace(/\.wav$/i,'').toLowerCase();return files.value.filter(f=>f.kind==='textgrid'&&(f.name.toLowerCase()===stem+'.textgrid'||f.name.toLowerCase().startsWith(stem+'_')));}
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
 const range=editorSelection(s);if(range)[wave.start,wave.end]=range;
 if(s.audioBuffer){wave.offset=s.visibleStart;wave.zoom=s.audioBuffer.duration/s.visibleDuration;}
}
function waveformSelection(a:number,b:number){s.selected=null;s.selectedIndices=[];s.selectedBoundary=null;s.lastMouseTime=a;label.value='';wave.start=a;wave.end=b;controls.fitStart.value=String(a);controls.fitEnd.value=String(b);revision.value++;}
watch(()=>[wave.start,wave.end],([a,b])=>{controls.fitStart.value=String(a);controls.fitEnd.value=String(b);},{flush:'sync'});
function fitRangeInput(){
 const a=Number(controls.fitStart.value),b=Number(controls.fitEnd.value);
 if(!String(controls.fitStart.value).trim()||!String(controls.fitEnd.value).trim()||!Number.isFinite(a)||!Number.isFinite(b)||a<0||b<a||b>(wave.asset?.duration??0)){error.value='强度范围需满足 0 ≤ 起点 ≤ 终点 ≤ 录音时长。';return false;}
 waveformSelection(a,b);error.value='';return true;
}
function boundaryStart(event:PointerEvent,time:number,tolerance:number){
 if(busy.value||saving.value){event.preventDefault();return;}
 if(event.shiftKey)return;
 syncView();const hit=editor.boundaryAt(time,tolerance);if(!hit)return;
 event.preventDefault();if(!finishInput())return;
 (event.currentTarget as SVGElement).focus({preventScroll:true});editor.beginBoundaryDrag(hit,event);changed();
}
function boundaryMove(event:PointerEvent,time:number){editor.dragBoundaryTo(time,event.clientX);}
function boundaryEnd(){if(s.drag)editor.onGridMouseUp();}
let pausedSelection='';
const selectionKey=()=>[source.value?.id,wave.channel,wave.start,wave.end].join(':');
function toggleSelection(){
 if(!wave.asset)return;
 if(isCurrentAudio(wave.asset,wave.channel)&&playback.playing){pausedSelection=selectionKey();pause();return;}
 if(wave.end<=wave.start){notice.value='请先拖动波形选择时间范围，或单击一个音节或音素，再按空格播放。';return;}
 const start=pausedSelection===selectionKey()&&isCurrentAudio(wave.asset,wave.channel)&&playback.position>=wave.start&&playback.position<wave.end?playback.position:wave.start;
 pausedSelection='';void play(wave.asset,start,wave.end,wave.channel);
}
watch(()=>[wave.start,wave.end,wave.channel],()=>{pausedSelection='';if(wave.asset&&isCurrentAudio(wave.asset,wave.channel)&&playback.playing)pause();});
function syncView(){s.visibleStart=wave.offset;s.visibleDuration=(wave.asset?.duration??3.2)/wave.zoom;durationInput.value=s.visibleDuration.toFixed(3);}
watch(()=>[wave.offset,wave.zoom],syncView);
watch([lipDirty,labelDirty,composing],()=>{wave.dirty=s.dirty||lipDirty.value||labelDirty.value||composing.value;});
function editLabel(text:string){editor.editText(text,sequenceEnabled.value?sequenceParts(editor,text):undefined);}
function finishInput(){if(composing.value){error.value='请先完成输入法组字，再保存或切换。';return false;}if(labelDirty.value){editLabel(label.value);changed();}return true;}
watch([suffix,phoneSplit],()=>{persist();});
function persist(){return host.projects.write('annotation.'+props.stateKey,{word:wordName.value,phone:phoneName.value,suffix:suffix.value,phoneSplit:phoneSplit.value});}
function names(){
 if(!finishInput()){wordName.value=s.wordTierName;phoneName.value=s.phoneTierName;return;}
 const w=wordName.value,p=phoneName.value;
 if(w&&w===p){wordName.value=s.wordTierName;phoneName.value=s.phoneTierName;error.value='音节与音素请选择不同的层。';return;}
 wordName.value=w;phoneName.value=p;s.wordTierName=w;s.phoneTierName=p;s.selected=null;s.selectedBoundary=null;s.selectedIndices=[];
 s.searchResults=[];s.searchIndex=-1;query.value='';s.sequenceStart=null;resetSequence(editor);
 if(!persist())error.value='层名偏好保存失败，当前编辑仍保留。';else error.value='';changed();
}
function layerDialog(role:'both'|'word'|'phone'){if(!finishInput())return;layerRole.value=role;newWord.value='';newPhone.value='';error.value='';dialog.value='layers';}
function createLayers(){
 if(!s.textgrid)return;
 try{const names=layerRole.value==='both'?[newWord.value,newPhone.value]:[layerRole.value==='word'?newWord.value:newPhone.value],tiers=newIntervalTiers(s.textgrid,names);
  editor.saveUndoState();s.textgrid.tiers.push(...tiers);s.dirty=true;
  if(layerRole.value!=='phone')wordName.value=s.wordTierName=tiers[0].name;
  if(layerRole.value!=='word')phoneName.value=s.phoneTierName=tiers.at(-1)!.name;
  s.selected=null;s.selectedIndices=[];s.selectedBoundary=null;resetSequence(editor);persist();dialog.value='';error.value='';notice.value='已创建：'+tiers.map(t=>t.name).join('、')+'。可开始标注，保存后写入 TextGrid。';changed();
 }catch(e){error.value=(e as Error).message;}
}
function chooseAudio(audio:ResearchFile){try{void open(audio,preferredGrid(audio,files.value));}catch{associationAudio.value=audio;associationId.value='';dialog.value='association';}}
function applyAssociation(){const audio=associationAudio.value,file=files.value.find(f=>f.id===associationId.value);if(audio&&file){dialog.value='';void open(audio,file);}}
async function prepare(role:'textgrid'|'lip',file:ResearchFile,suff='_自动保存'){
 const key=role+':'+file.id+':'+suff;let target=targets.get(key);if(!target){target=await port.target(file,role,suff);targets.set(key,target);}return target;
}
function updateSaved(file:ResearchFile,original:ResearchFile){
 const parent=original.name.includes('/')?original.name.slice(0,original.name.lastIndexOf('/')+1):'';
 const named={...file,name:file.name.includes('/')?file.name:parent+file.name};
 files.value=files.value.filter(f=>f.id!==named.id&&f.name.toLowerCase()!==named.name.toLowerCase()).concat(named);return named;
}
async function saveGrid(auto=false,confirmed=false):Promise<boolean>{
 if(!finishInput())return false;
 if(!s.textgrid||!source.value)return true;
 if(!s.textgrid.tiers.length){if(auto)return true;error.value='请先创建标注层。';return false;}
 if(!auto&&suffix.value===''&&!confirmed&&props.context.files.kind==='desktop'){dialog.value='overwrite';return false;}
 if(saving.value)return false;
 const generation=ticket,document=s.textgrid,original=gridFile.value??source.value,audio=source.value,sourceHash=gridFile.value?gridSha.value:source.value.sha256!,suff=auto?'_自动保存':suffix.value;
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
  const grid=explicitGrid??preferredGrid(audio,files.value);
  const [decoded,read]=await Promise.all([annotationAudio(props.context.files,audio,reading.signal),grid?props.context.files.read(grid,reading.signal):Promise.resolve(undefined)]);
  validateEditingAudio(decoded.asset);const {grid:document,created}=loadEditingGrid(read?decodeText(read.buffer):undefined,decoded.asset.duration);
  if(!alive||own!==ticket)return;
  previewNote.value=decoded.previewNote;
  source.value={...audio,sha256:decoded.sha256};gridFile.value=grid??null;gridSha.value=read?.sha256??'';wave.asset=markRaw(decoded.asset);wave.channel=0;wave.offset=0;wave.zoom=Math.max(1,decoded.asset.duration/3.2);wave.start=0;wave.end=0;pausedSelection='';
  s.textgrid=document;s.audioBuffer={duration:decoded.asset.duration,sampleRate:decoded.asset.sampleRate,getChannelData:index=>decoded.asset.channels[index]};s.visibleStart=0;s.visibleDuration=decoded.asset.duration/wave.zoom;
  const roles=editingNames(document,{word:wordName.value||prefs.word,phone:phoneName.value||prefs.phone});wordName.value=s.wordTierName=roles.word;phoneName.value=s.phoneTierName=roles.phone;
  s.selected=null;s.selectedBoundary=null;s.selectedIndices=[];s.undoStack=[];s.drag=null;s.dirty=false;s.copiedWord='';s.copiedLabIndex=null;s.searchResults=[];s.searchIndex=-1;query.value='';s.sequenceIndex=0;s.sequenceStart=null;
  savedGrid.value=serializeGrid(document);lip.value=null;lipFile.value=null;lipOffset.value=0;savedLipOffset.value=0;s.labSequence=[];s.labWords=new Set();labName.value='';
  controls.fitStart.value='0';controls.fitEnd.value='0';controls.spliceStart.value='0';controls.spliceEnd.value=decoded.asset.duration.toFixed(3);targets.clear();
  await prepare('textgrid',source.value,'_自动保存');
  const lab=files.value.find(f=>f.kind==='lab'&&f.name.toLowerCase()===audio.name.replace(/\.wav$/i,'.lab').toLowerCase());
  if(lab){const bytes=await props.context.files.read(lab);if(own!==ticket||!alive)return;setLab(decodeText(bytes.buffer),lab.name);}
  const stem=audio.name.replace(/\.wav$/i,''),parent=audio.name.includes('/')?audio.name.slice(0,audio.name.lastIndexOf('/')+1):'';
  const preferred=[stem+'.lip.json',stem+'.pkl',parent+'audio_recording.lip.json',parent+'audio_recording.pkl'];
  const candidates=preferred.map(name=>files.value.find(f=>f.name.toLowerCase()===name.toLowerCase())).filter(Boolean) as ResearchFile[];
  // Fallback only in a single-recording directory, so unrelated WAVs never share a PKL silently.
  const candidate=candidates.find(f=>f.name.toLowerCase().startsWith(stem.toLowerCase()+'.'))??(files.value.filter(f=>f.kind==='audio'&&(f.name.includes('/')?f.name.slice(0,f.name.lastIndexOf('/')+1):'')===parent).length===1?candidates[0]:undefined);
  if(candidate)try{await loadLip(candidate,own);}catch(e){error.value='标注已加载；唇形读取失败：'+(e as Error).message;}
  if(own===ticket&&alive){notice.value=created?'音频已加载。请点击“创建标注层”，输入音节与音素层名后开始标注。':'已加载：'+grid!.name;syncView();changed();}
 }catch(e){if(alive&&own===ticket){error.value=(e as Error).message;notice.value='读取未完成，请检查文件；已有编辑保留。';}}finally{if(alive&&own===ticket)busy.value=false;}
}
async function loadLip(file:ResearchFile,own=ticket){
 const read=await port.lip(file),track=lipTrack(read.wire);await prepare('lip',file,'');if(own!==ticket||!alive)return;
 lip.value=markRaw(track);lipFile.value=file;lipSha.value=read.sha256;lipOffset.value=track.offset;savedLipOffset.value=track.offset;
}
async function chooseLip(id:string){if(busy.value||saving.value)return;if(!await saveLip())return;if(lipDirty.value){error.value='保存期间有新的唇偏编辑，请再次保存后切换。';return;}const file=files.value.find(f=>f.id===id);if(!file){lip.value=null;lipFile.value=null;return;}busy.value=true;try{await loadLip(file);error.value='';}catch(e){error.value=(e as Error).message;}finally{busy.value=false;}}
function setLab(text:string,name:string){if(text.length>2_000_000)throw Error('词表超过 2 MB。');const words=text.split(/\s+/).filter(Boolean);if(!words.length)throw Error('词表为空。');if(words.length>10000||words.some(w=>w.length>200))throw Error('词表最多 10000 个条目，每项最多 200 字符。');s.labSequence=words;s.labWords=new Set(words.map(w=>w.toLowerCase()));s.copiedLabIndex=null;labName.value=name;resetSequence(editor);revision.value++;}
function clearLab(){s.labSequence=[];s.labWords=new Set();s.copiedLabIndex=null;labName.value='';resetSequence(editor);changed();}
function applySequenceText(){try{setLab(sequenceText.value,'粘贴词表');dialog.value='';error.value='';notice.value='已应用词表，可开启自动生成音节和音素。';changed();}catch(e){error.value=(e as Error).message;}}
function doubleTime(time:number,ctrl=false){
 if(busy.value||saving.value||!finishInput())return;
 try{notice.value=sequenceEnabled.value?sequenceDoubleClick(editor,time,ctrl,phoneSplit.value):manualDoubleClick(editor,time);error.value='';changed();if(s.sequenceStart!==null){wave.start=s.sequenceStart;wave.end=s.sequenceStart;}}
 catch(e){error.value=(e as Error).message;}
}
function doublePhone(time:number){
 if(busy.value||saving.value||!finishInput())return;
 try{
  const word=editor.wordTier()?.intervals.find(w=>w.text.trim()&&time>w.xmin&&time<w.xmax);
  const labels=word?(sequenceEnabled.value?sequenceParts(editor,word.text):editor.pinyinToPhones(word.text)):[];
  if(word&&labels.length>1&&hasUnsplitPhones(editor,word)){
   notice.value=splitInitialPhones(editor,word,time,phoneSplit.value,labels);
   s.selected={tier:s.phoneTierName,index:editor.phoneTier()!.intervals.findIndex(i=>i.xmin===word.xmin)};
  }else{editor.saveUndoState();editor.splitPhoneAt(time);notice.value='已在双击位置插入音素边界，已有边界保留。';}
  error.value='';changed();
 }catch(e){error.value=(e as Error).message;}
}
function cancelStart(){s.sequenceStart=null;notice.value='已取消待定起点。';changed();}
function undo(){if(s.sequenceStart!==null){cancelStart();return;}editor.undo();if(s.textgrid){const roles=editingNames(s.textgrid,{word:wordName.value,phone:phoneName.value});wordName.value=s.wordTierName=roles.word;phoneName.value=s.phoneTierName=roles.phone;s.dirty=serializeGrid(s.textgrid)!==savedGrid.value;}changed();}
function moveAnnotation(direction:number){
 if(busy.value||saving.value||!finishInput())return;
 try{const ms=Number(nudgeStep.value);if(!String(nudgeStep.value).trim()||!Number.isFinite(ms)||ms<.001||ms>1000)throw Error('微调步长需在 0.001–1000 ms 之间。');editor.moveSelected(direction*ms/1000);error.value='';notice.value=`选中音节及对应音素已${direction<0?'左':'右'}移 ${ms} ms。`;changed();}
 catch(e){error.value=(e as Error).message;}
}
function nextSequence(event:Event){s.sequenceIndex=Number((event.target as HTMLSelectElement).value);s.sequenceStart=null;changed();}
watch(sequenceEnabled,()=>{s.sequenceStart=null;changed();});
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
  else{if(!editor.wordTier())throw Error('找不到词层，请检查层名。');if(name==='fit'){if(!fitRangeInput())return;editor.fitIntensityRange();controls.fitStart.value=String(wave.start);controls.fitEnd.value=String(wave.end);}else if(name==='phones')editor.autoPhonesForSelection();else{editor.saveUndoState();editor.deleteSelectedBoundary();}}
  changed();
 }catch(e){error.value=(e as Error).message;}
}
function key(event:KeyboardEvent){
 if(!props.active||!s.textgrid||dialog.value||busy.value||saving.value||event.isComposing)return;
 const command=annotationShortcut(event);
 // Save commits the label field too; editing shortcuts retain native input behavior.
 if(command==='save'){event.preventDefault();void saveGrid();return;}
 const element=event.target as HTMLElement;if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(element.tagName)||element.isContentEditable)return;
 const k=event.key.toLowerCase();
 const clipboardAction=command&&['copy','cut','paste'].includes(command),remove=event.key==='Backspace'&&!s.selectedBoundary;
 if(clipboardAction||remove){
  event.preventDefault();if(!finishInput())return;
  try{if(remove)editor.deleteAnnotation();else if(k==='v')editor.pasteAnnotation();else editor.copyAnnotation(k==='x');error.value='';changed();}
  catch(e){error.value=(e as Error).message;}return;
 }
 if(event.altKey&&['ArrowLeft','ArrowRight'].includes(event.key)){nudge((event.key==='ArrowLeft'?-1:1)*(event.shiftKey?10:1));}
 else if(event.key==='Backspace'&&s.selectedBoundary){void action('delete');}
 else if(event.key==='Escape'&&s.sequenceStart!==null){cancelStart();}
 else if(['ArrowLeft','ArrowRight'].includes(event.key)&&!event.ctrlKey&&!event.metaKey&&!event.altKey){moveAnnotation(event.key==='ArrowLeft'?-1:1);}
 else if(command==='undo'){undo();}
 else if((k==='p'||event.key===' ')&&!event.ctrlKey&&!event.metaKey&&!event.altKey){event.preventDefault();if(!event.repeat)toggleSelection();return;}
 else if(event.key.length===1&&event.key!==' '&&!event.ctrlKey&&!event.altKey&&!event.metaKey){editLabel((selectedInterval(s.textgrid,s.selected)?.text??'')+event.key);}
 else return;event.preventDefault();changed();
}
watch(()=>props.active,active=>{if(!active)stopRepeat();});
onMounted(()=>{window.addEventListener('keydown',key);window.addEventListener('blur',stopRepeat);timer=setInterval(()=>{if(props.context.files.kind!=='preview'&&!busy.value&&!saving.value&&!dialog.value&&!composing.value&&(s.dirty||lipDirty.value||labelDirty.value))void savePending();},60000);if(props.context.files.kind==='server')void scan();});
onUnmounted(()=>{alive=false;++ticket;reading?.abort();clearInterval(timer);stopRepeat();window.removeEventListener('keydown',key);window.removeEventListener('blur',stopRepeat);});
defineExpose({save:savePending});
</script>
<template>
<ModuleFrame fit label="语音标注对齐 工作区" class="annotation-page" :data-revision="revision" :aria-busy="busy||saving">
 <ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-if="notice&&busy" kind="loading" :message="notice"/>
 <div v-resizable-panels="{key:stateKey,center:'.annotation-editor',centerMin:350,panels:[{selector:'.annotation-files',side:'left',label:'标注文件',initial:220,min:180,max:520},{selector:'.annotation-settings',side:'right',label:'标注设置',initial:250,min:210,max:560}]}" class="annotation-layout" :class="{collapsed:!sidebar,'settings-collapsed':!settingsOpen}">
 <aside v-show="sidebar" class="annotation-files annotation-card">
  <h2>语料文件</h2><div class="actions"><button v-if="context.files.choose" :disabled="busy||saving" @click="scan(true)">选择语料文件夹</button><button v-else-if="context.files.add" :disabled="busy||saving" @click="picker?.click()">打开语料文件</button><button :disabled="busy||saving" @click="scan()">扫描</button></div>
  <input ref="picker" hidden type="file" multiple accept=".wav,.TextGrid,.textgrid,.lab,.json" @change="addFiles"/><small>{{directoryLabel||context.label}} · {{pairs.length}} 份录音</small>
  <input v-model="filter" aria-label="筛选标注文件" placeholder="筛选文件…"/>
  <small class="file-legend"><span class="has-grid">绿色：有 TextGrid</span><span>普通色：无 TextGrid</span></small>
  <div class="annotation-file-list"><button v-for="pair in visibleFiles" :key="pair.audio.id" :class="{selected:source?.id===pair.audio.id,'has-grid':!!pair.grid||!!associatedGrids(pair.audio).length}" :disabled="busy||saving" :title="pair.audio.name" @click="chooseAudio(pair.audio)"><span>{{pair.audio.name}}</span></button><p v-if="!visibleFiles.length" class="hint">选择包含 WAV 的目录后，从这里打开录音。</p></div>
 </aside>
 <div class="annotation-editor">
  <div class="annotation-card annotation-plots"><div class="actions plot-actions"><button :aria-pressed="settingsOpen" @click="settingsOpen=!settingsOpen">标注设置</button><button :aria-pressed="sidebar" @click="sidebar=!sidebar">文件列表</button><button :disabled="!hasLayers||busy||saving" class="primary" @click="saveGrid()">{{saving?'正在保存…':'保存 TextGrid'}}{{s.dirty?' *':''}}</button></div><template v-if="wave.asset"><p v-if="playback.error" class="error-text" role="alert">{{playback.error}}</p>
   <p v-if="previewNote" class="hint">{{previewNote}}波形、试听、语谱及强度贴合使用此预览，原音频与标注时间不变。</p>
   <div class="actions view-controls"><strong :title="source?.name">{{source?.name}}</strong><label>视窗 <input v-model="durationInput" aria-label="标注可视时长" type="number" step=".1" @change="view"/> s</label><button @click="wave.offset=Math.max(0,wave.offset-s.visibleDuration*.8)">前一窗</button><button @click="wave.offset=Math.min(wave.asset!.duration-s.visibleDuration,wave.offset+s.visibleDuration*.8)">后一窗</button><label class="annotation-volume">音量 <input :value="playback.volume" type="range" min="0" max="1" step=".01" aria-label="音量" @input="volume(Number(($event.target as HTMLInputElement).value))"/><small>{{Math.round(playback.volume*100)}}%</small></label></div>
   <div v-if="!editingReady" class="layer-prompt"><span>{{hasLayers?'请选择音节与音素的编辑层，或创建所需层。':'当前还没有标注层。输入自己的层名后开始标注。'}}</span><button :disabled="busy||saving" @click="layerDialog('both')">创建标注层</button></div>
   <WaveformViewport :state="wave" :compact-overview="true" :hide-overview-controls="true" :overview-top="true" :auto-amplitude="true" :shift-wheel-pan="true" :continuous-detail="true" :hide-time-axis="true" double-click-action="annotation" @time-double-click="doubleTime" @selection-end="waveformSelection" @boundary-start="boundaryStart" @boundary-move="boundaryMove" @boundary-end="boundaryEnd"><template #annotations="{x,start,end}"><template v-for="boundary in boundaries" :key="boundary.time"><path v-if="boundary.time>start&&boundary.time<end" :d="`M${x(boundary.time)} 0V90`" class="annotation-boundary" :class="[boundary.kind,{chosen:s.selectedBoundary?.time===boundary.time}]"/></template><path v-if="s.sequenceStart!==null&&s.sequenceStart>=start&&s.sequenceStart<=end" :d="`M${x(s.sequenceStart)} 0V90`" class="sequence-pending"/></template></WaveformViewport>
   <AnnotationTracks :disabled="busy||saving||composing" :editor="editor" :revision="revision" :wave="wave" :lip="lip" :offset="lipOffset" :show-open="showOpen" :show-width="showWidth" :active="active" :sequence-editing="sequenceEnabled" @time-double-click="doubleTime" @phone-double-click="doublePhone" @selection-end="waveformSelection" @changed="changed"/>
   <div class="label-editor"><label>选中区间文本 <input v-model="label" aria-label="编辑选中标注文本" :disabled="!selected" @compositionstart="composing=true" @compositionend="composing=false;finishInput()" @change="finishInput()" @keydown.enter="!$event.isComposing&&finishInput()"/></label><small>Backspace 删除标注或边界 · Ctrl＋X 剪切 · Ctrl＋V 粘贴</small></div>
   <p class="hint">Ctrl＋滚轮缩放 · Shift＋滚轮平移标注视图 · Ctrl＋Z 撤销 · Ctrl＋拖动拉开边界 · Backspace 合并选中边界。输入框支持中文输入法。</p>
  </template>
  <ModuleStatus v-else kind="empty" message="打开一组录音与标注"><p>选择语料目录后，从左侧列表开始编辑。音频、TextGrid 和唇形共用秒时间轴。</p></ModuleStatus>
  </div>
  <div class="annotation-card search-tools intensity-tools">
   <div class="actions"><strong>强度贴合</strong><label>起点 <input v-model="controls.fitStart.value" aria-label="强度起点" type="number" step=".000001" @change="fitRangeInput"/> s</label><label>终点 <input v-model="controls.fitEnd.value" aria-label="强度终点" type="number" step=".000001" @change="fitRangeInput"/> s</label><label>内收/外扩 <input v-model="controls.fitTrimMs.value" aria-label="强度内收毫秒" type="number" min="-50" max="80"/> ms</label><button :disabled="!source||busy" @click="action('fit')">强度贴合</button><small>正数内收，负数外扩 · 第一声道</small></div>
  </div>
  <div class="annotation-card edit-toolbar">
   <div class="actions resource-controls"><button title="词典：标签到音素的对应，每行一个标签及其音素" @click="dictPicker?.click()">上传词典</button><small :title="dictName">{{dictName}}</small><button title="词表：按顺序等待标注的拼音条目，以空格或换行分隔" @click="labPicker?.click()">上传词表</button><button @click="sequenceText=s.labSequence.join(' ');dialog='words'">粘贴词表</button><button v-if="labName" @click="clearLab">清除词表</button><small v-if="labName" :title="labName">{{labName}}</small><label class="sequence-toggle"><input v-model="sequenceEnabled" type="checkbox" :disabled="!editingReady||busy||saving"/>自动生成音节和音素</label><label class="sequence-next">首次音素切分<select v-model="phoneSplit" aria-label="音素首个切分点"><option value="cursor">首个边界用双击位置</option><option value="equal">按音素数等分</option></select></label><label v-if="sequenceEnabled&&s.labSequence.length" class="sequence-next">下一音节<select aria-label="下一音节" :value="s.sequenceIndex" :disabled="busy||saving" @change="nextSequence"><option v-for="(token,index) in s.labSequence" :key="index" :value="index">{{index+1}} / {{s.labSequence.length}} · {{token}}</option><option :value="s.labSequence.length">词表已标完</option></select></label></div>
   <small class="resource-help">词典：标签 → 音素对应。词表：等待标注的拼音顺序。</small>
   <p v-if="sequenceEnabled||s.sequenceStart!==null" class="hint sequence-hint"><span v-if="sequenceEnabled">普通双击：起点 → 终点。Ctrl＋双击：沿用上一音节终点。音节内部双击：分声母、完整韵母。</span><span v-else>普通双击：空白标注起点 → 终点，然后输入文字。</span><strong v-if="s.sequenceStart!==null">待定起点 {{s.sequenceStart.toFixed(6)}} s，等待终点。</strong><button v-if="s.sequenceStart!==null" @click="cancelStart">取消起点（Esc）</button></p>
   <input ref="dictPicker" hidden type="file" accept=".dict,.txt" @change="resource($event,'dict')"/><input ref="labPicker" hidden type="file" accept=".lab,.txt" @change="resource($event,'lab')"/>
  </div>
  <div class="annotation-card search-tools"><div class="actions"><input v-model="query" aria-label="搜索词层文本" placeholder="搜索词层文本" @input="search" @keydown.enter.prevent="$event.shiftKey?editor.findPrev():editor.findNext();changed()"/><span class="mono">{{matchCount?s.searchIndex+1:0}} / {{matchCount}}</span><button :disabled="!matchCount" @click="editor.findPrev();changed()">上一个</button><button :disabled="!matchCount" @click="editor.findNext();changed()">下一个</button><input v-model="replace" aria-label="替换文本" placeholder="替换为…"/><button :disabled="!matchCount" @click="editor.replaceCurrent(replace);changed()">替换</button><button :disabled="!matchCount" @click="dialog='replace'">全部替换</button></div>
  </div>
 </div>
 <aside v-show="settingsOpen" class="annotation-settings annotation-card">
 <button @click="emit('references')">方法与引用</button><ModuleStatus v-if="notice&&!busy" kind="info" :message="notice"/>
  <h2>层级与保存</h2><label>标注微调步长（ms）<input v-model="nudgeStep" aria-label="标注微调步长毫秒" type="number" min=".001" max="1000" step=".1"/></label><small>左右键微调音节 · Shift＋拖动框选</small><label>音节 / 词层<select v-model="wordName" aria-label="词层名" :disabled="!source||busy||saving" @change="names"><option value="">请选择音节或词层</option><option v-for="t in tierChoices" :key="t.name" :value="t.name" :disabled="!t.editable||t.name===phoneName">{{t.name}}{{t.editable?'':'（'+t.kind+'，保留）'}}</option></select></label><button :disabled="!source||busy||saving" @click="layerDialog('word')">新建音节 / 词层</button><label>音素层<select v-model="phoneName" aria-label="音素层名" :disabled="!source||busy||saving" @change="names"><option value="">请选择音素层</option><option v-for="t in tierChoices" :key="t.name" :value="t.name" :disabled="!t.editable||t.name===wordName">{{t.name}}{{t.editable?'':'（'+t.kind+'，保留）'}}</option></select></label><button :disabled="!source||busy||saving" @click="layerDialog('phone')">新建音素层</button>
  <label>当前 TextGrid<select :value="gridFile?.id??''" aria-label="当前 TextGrid" :disabled="!source||busy||saving" @change="open(source!,files.find(f=>f.id===($event.target as HTMLSelectElement).value))"><option v-if="source&&!gridFile" value="" disabled>新建空白标注（尚未保存）</option><option v-for="f in gridChoices" :key="f.id" :value="f.id">{{f.name}}</option></select></label>
  <label>保存后缀<input v-model="suffix" aria-label="TextGrid 保存后缀"/></label><p class="target-preview">目标：{{targetName}}</p><p v-if="suffix===''" class="overwrite-note">{{context.files.kind==='desktop'?'留空将覆盖 WAV 同名原始 TextGrid，保存时确认。':'网页将保存同名的新版本，原资源保留。'}}</p>
  <small>编辑后，切换文件前及每分钟自动保存到 _自动保存。TextGrid 与唇偏分别保存。</small><button :disabled="!hasLayers" @click="download('textgrid')">下载当前 TextGrid</button><button v-if="lip" @click="download('lip')">下载安全唇形 JSON</button>
  <hr/><h2>参考标注复用</h2><div class="actions"><button @click="refPicker?.click()">选择参考 TextGrid</button><button v-if="referenceName" @click="s.referenceTextGrid=null;referenceName=''">清除参考</button></div><input ref="refPicker" hidden type="file" accept=".TextGrid,.textgrid" @change="resource($event,'reference')"/><small>{{referenceName||'未选择参考文件'}}</small>
  <label>复用模式<select v-model="controls.spliceMode.value" aria-label="参考复用模式"><option value="outside">区间之外</option><option value="inside">区间之内</option><option value="before">起点之前</option><option value="after">起点之后</option></select></label>
  <label>起点（秒）<input v-model="controls.spliceStart.value" aria-label="参考起点" type="number" step=".001"/></label><label>终点（秒）<input v-model="controls.spliceEnd.value" aria-label="参考终点" type="number" step=".001" :disabled="['before','after'].includes(controls.spliceMode.value)"/></label>
  <button :disabled="!source||busy" @click="action('splice')">复用参考标注</button><small>对所有同名层应用所选范围，可撤销。起点之后从起点延伸到文件末尾。</small>
  <hr/><h2>唇形对齐</h2><label>唇形记录<select :value="lipFile?.id??''" aria-label="唇形记录" :disabled="!source||busy||saving" @change="chooseLip(($event.target as HTMLSelectElement).value)"><option value="">不关联</option><option v-for="f in lipChoices" :key="f.id" :value="f.id">{{f.name}}</option></select></label>
  <template v-if="lip"><div class="actions"><label><input v-model="showOpen" type="checkbox"/>唇开</label><label><input v-model="showWidth" type="checkbox" :disabled="!lip.width.length"/>唇宽</label></div><label>共同时间偏移（ms）<input :value="Number((lipOffset*1000).toFixed(3))" aria-label="唇形共同偏移毫秒" type="number" step="1" @change="offsetInput"/></label><div class="actions"><button aria-label="唇形左移" @pointerdown="startRepeat($event.shiftKey?-10:-1)" @pointerup="stopRepeat" @pointerleave="stopRepeat" @pointercancel="stopRepeat" @keydown.enter.prevent="nudge(-1)">← 1 ms</button><button aria-label="唇形右移" @pointerdown="startRepeat($event.shiftKey?10:1)" @pointerup="stopRepeat" @pointerleave="stopRepeat" @pointercancel="stopRepeat" @keydown.enter.prevent="nudge(1)">1 ms →</button></div><button :disabled="!lipDirty||saving||busy" @click="saveLip">保存唇偏{{lipDirty?' *':''}}</button><small>正值向更晚平移。按住连续微调，Shift 为 10 ms；Alt＋方向键同效。两条曲线只有一个偏移。</small></template>
  <p v-else class="hint">找到同名唇形记录时自动关联，也可明确选择。网页使用 .lip.json。</p>
 </aside>
 </div>
 <ModalDialog v-if="dialog" :title="dialog==='layers'?'创建标注层':dialog==='association'?'选择关联 TextGrid':dialog==='words'?'粘贴拼音词表':dialog==='replace'?'确认全部替换':'确认覆盖原始标注'" :close-disabled="saving" @close="dialog=''">
  <template v-if="dialog==='layers'"><p>输入新的区间层名称。每层初始覆盖整段音频，已有层保留。</p><label v-if="layerRole!=='phone'" class="new-layer-field">音节 / 词层名<input v-model="newWord" aria-label="新建音节层名" maxlength="200" placeholder="例如：音节"/></label><label v-if="layerRole!=='word'" class="new-layer-field">音素层名<input v-model="newPhone" aria-label="新建音素层名" maxlength="200" placeholder="例如：音素"/></label></template>
  <template v-else-if="dialog==='association'"><p>{{associationAudio?.name}} 有多个可能的标注文件，请选择本次使用的文件。</p><select v-model="associationId" aria-label="选择关联标注文件"><option value="" disabled>请选择 TextGrid</option><option v-for="file in associationAudio?associatedGrids(associationAudio):[]" :key="file.id" :value="file.id">{{file.name}}</option></select></template>
  <template v-else-if="dialog==='words'"><p>按标注顺序粘贴拼音，用空格或换行分隔。每完成一个音节的起终点，就使用下一个条目。</p><textarea v-model="sequenceText" aria-label="拼音词表" rows="6" maxlength="2000000" placeholder="zhe4 shi4 shang4 yi1 ding4 …"/></template>
  <p v-else-if="dialog==='replace'">将当前文件词层的 {{matchCount}} 个匹配区间替换为 {{replace||'空文本'}}，同时按词典重新填充音素。可用撤销恢复。</p><p v-else>保存目标：{{targetName}}。已有同名文件会在版本核对后被覆盖，唇偏保持独立保存。</p><p v-if="error" role="alert" class="error-text">{{error}}</p>
  <template #footer><button :disabled="saving" @click="dialog=''">取消</button><button class="primary" :disabled="saving||(dialog==='association'&&!associationId)" @click="dialog==='layers'?createLayers():dialog==='association'?applyAssociation():dialog==='words'?applySequenceText():dialog==='replace'?(editor.replaceAll(replace),changed(),dialog=''):saveGrid(false,true).then(ok=>{if(ok)dialog=''})">{{dialog==='layers'?'创建层':dialog==='association'?'打开标注':dialog==='words'?'应用词表':dialog==='replace'?'替换全部':'确认保存'}}</button></template>
 </ModalDialog>
</ModuleFrame>
</template>
<style scoped>
.plot-actions{margin-bottom:10px}
.annotation-page{--wave-axis-width:max(64px,calc(var(--figure-size,14px) * 4.8))}
.annotation-file-list button span{max-width:100%}
.file-legend{display:flex;flex-wrap:wrap;gap:3px 10px;font-size:11px}.has-grid{color:var(--success)}.annotation-file-list button{flex:0 0 auto;min-height:36px;display:block;line-height:1.45}.annotation-file-list button span{display:block;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}.annotation-file-list button.has-grid{color:var(--success);border-left:3px solid var(--success)}.annotation-file-list button.selected{box-shadow:inset 0 0 0 1px var(--accent)}
.resource-controls label{display:flex;align-items:center;gap:5px;font-size:12px}.resource-controls .sequence-toggle{white-space:nowrap}.resource-controls input[type=checkbox]{width:16px;height:16px}.resource-controls small{max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.sequence-next select{max-width:160px}.resource-help{font-size:11px;color:var(--muted)}
.new-layer-field{display:flex;flex-direction:column;gap:5px;margin:12px 0}.layer-prompt{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:8px 0;background:var(--teal-bg);padding:8px;border-radius:4px;font-size:12px}.selection-readout{display:flex;align-items:center;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:4px 0 8px}.selection-readout>span:first-child{font-family:var(--font-mono)}
.annotation-boundary{stroke:var(--teal);stroke-width:1;opacity:.45;vector-effect:non-scaling-stroke;pointer-events:none}.annotation-boundary.word{opacity:.9;stroke-width:1.2}.annotation-boundary.chosen{stroke:var(--danger);opacity:1;stroke-width:2}.annotation-volume{margin-left:auto}.annotation-volume input[type=range]{width:90px}.view-controls strong{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.movement-controls{font-size:12px;padding:6px 0}.movement-controls label{display:flex;align-items:center;gap:4px}.movement-controls input{width:80px}.movement-controls button[aria-pressed=true]{color:var(--accent);border-color:var(--accent);background:var(--selected)}.annotation-file-list select{width:100%;min-width:0;font-size:11px}
.sequence-controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.sequence-controls label{display:flex;align-items:center;gap:6px;font-size:12px}.sequence-controls select{max-width:220px}.sequence-controls input[type=checkbox]{width:16px;height:16px}.sequence-hint{line-height:1.8;margin:0}.sequence-hint strong{display:block;color:var(--accent)}.sequence-pending{stroke:var(--danger);stroke-width:2;stroke-dasharray:5 3;vector-effect:non-scaling-stroke;pointer-events:none}textarea[aria-label="拼音词表"]{width:100%;margin-top:12px;resize:vertical;font-family:var(--font);line-height:1.8}
.annotation-page{min-height:100%}.annotation-header{display:flex;justify-content:space-between;gap:16px;align-items:start}.annotation-header h1{font-size:23px}.annotation-header p:not(.eyebrow){color:var(--muted);font-size:13px;margin-top:5px}.actions{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.annotation-layout{display:grid;grid-template-columns:var(--panel-left,220px) minmax(350px,1fr) var(--panel-right,250px);gap:var(--module-gap);align-items:start}.annotation-layout.collapsed{grid-template-columns:minmax(350px,1fr) var(--panel-right,250px)}.annotation-card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:12px;min-width:0}.annotation-files{display:flex;flex-direction:column;gap:10px}.annotation-file-list{max-height:630px;overflow:auto;display:flex;flex-direction:column;gap:4px}.annotation-file-list button{display:flex;flex-direction:column;align-items:start;text-align:left;min-width:0;white-space:normal;padding:8px}.annotation-file-list span,.annotation-file-list small{overflow-wrap:anywhere}.annotation-file-list .selected{border-color:var(--accent);background:var(--selected)}.annotation-editor{display:flex;flex-direction:column;gap:10px;min-width:0}.edit-toolbar,.search-tools{display:flex;flex-direction:column;gap:10px}.search-tools>.actions+.actions{padding-top:10px;border-top:1px solid var(--border)}.search-tools input:not([type=number]){width:150px}.search-tools label,.view-controls label{display:flex;align-items:center;gap:4px;font-size:12px}.search-tools input[type=number],.view-controls input{width:78px}.search-tools strong{font-size:12px}.annotation-settings{display:flex;flex-direction:column;gap:9px}.annotation-settings>label{display:flex;flex-direction:column;gap:4px;font-size:12px}.annotation-settings hr{margin:8px 0}.annotation-settings small{line-height:1.65}.target-preview{font-size:12px;overflow-wrap:anywhere;background:var(--app);padding:8px;border-radius:4px}.overwrite-note{color:var(--warning);font-size:12px}.annotation-plots :deep(.wave-track svg){height:var(--annotation-plot-height,175px)}.view-controls strong{font-size:12px;max-width:260px;overflow-wrap:anywhere}.label-editor{display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding:12px 0}.label-editor label{font-size:12px;display:flex;gap:8px;align-items:center}.label-editor input{font-family:var(--font-figure-ipa);font-size:var(--figure-size,14px);width:200px}.annotation-empty{padding:60px 24px;text-align:center}.annotation-empty p{margin-top:10px;color:var(--muted)}.notice{font-size:12px;color:var(--teal);overflow-wrap:anywhere}.error-text{overflow-wrap:anywhere}.annotation-settings input,.annotation-settings select{width:100%}.annotation-settings input[type=checkbox]{width:16px}
@media(max-width:1250px){.annotation-layout{grid-template-columns:var(--panel-left,220px) minmax(350px,1fr)}.annotation-layout.collapsed{grid-template-columns:1fr}.annotation-settings{grid-column:1/-1;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px 16px}.annotation-settings h2,.annotation-settings hr{grid-column:1/-1}.annotation-settings hr{width:100%}.annotation-file-list{max-height:600px}}
@media(max-width:780px){.annotation-page{padding:12px}.annotation-header{flex-direction:column}.annotation-layout,.annotation-layout.collapsed{display:flex;flex-direction:column}.annotation-files,.annotation-settings,.annotation-editor{width:100%}.annotation-file-list{max-height:160px}.annotation-settings{grid-template-columns:repeat(2,minmax(0,1fr))}.annotation-plots{padding:8px}.search-tools input:not([type=number]){width:125px}}
@media(max-width:480px){.annotation-settings{display:flex}.label-editor label{width:100%}.label-editor input{width:130px}.annotation-header .actions{flex-wrap:wrap}}

/* P17: bound the three panes to the available workbench height. */
.annotation-page{height:100%;min-height:0;overflow:auto;--annotation-plot-height:max(110px,calc(15dvh / var(--page-scale,1)));--annotation-grid-height:max(96px,calc(12dvh / var(--page-scale,1)))}
.annotation-layout{flex:1;min-height:0;align-items:stretch;gap:var(--module-gap)}
.annotation-files,.annotation-settings,.annotation-editor{min-height:0;overflow:auto;overscroll-behavior:contain}
.annotation-files,.annotation-settings{height:100%}
.annotation-file-list{flex:1;min-height:80px;max-height:none}
.annotation-card{padding:8px}.annotation-editor{gap:var(--module-gap)}
.annotation-editor>.annotation-card{flex-shrink:0}
@container module (max-width:1000px){.annotation-layout{flex:none;min-height:min-content}.annotation-files,.annotation-settings,.annotation-editor{height:auto;overflow:visible}.annotation-file-list{max-height:400px}}
.annotation-layout.settings-collapsed{grid-template-columns:var(--panel-left,220px) minmax(350px,1fr)}
.annotation-layout.collapsed.settings-collapsed{grid-template-columns:minmax(350px,1fr)}
</style>
<style scoped>
/* Use the actual available pane width, including the user's page scale. */
@container module (max-width:1000px){.annotation-layout{grid-template-columns:var(--panel-left,220px) minmax(350px,1fr)}.annotation-layout.collapsed{grid-template-columns:1fr}.annotation-settings{grid-column:1/-1;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px 16px}.annotation-settings h2,.annotation-settings hr{grid-column:1/-1}.annotation-settings hr{width:100%}}
@container module (max-width:720px){.annotation-layout,.annotation-layout.collapsed{display:flex;flex-direction:column}.annotation-files,.annotation-settings,.annotation-editor{width:100%}.annotation-file-list{max-height:160px}.annotation-settings{grid-template-columns:repeat(2,minmax(0,1fr))}}
@container module (max-width:440px){.annotation-settings{display:flex}.label-editor label{width:100%}.label-editor input{width:130px}}
.annotation-layout.settings-collapsed{grid-template-columns:var(--panel-left,220px) minmax(350px,1fr)}
.annotation-layout.collapsed.settings-collapsed{grid-template-columns:minmax(350px,1fr)}
</style>
