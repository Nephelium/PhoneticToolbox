<script setup lang="ts">
import {joinMessages as joinStatusMessages} from '../module-localization.ts';
import {uiMessage as localizedMessage} from '../module-localization.ts';
import {uiErrorMessage as displayErrorValue,uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';

import {localized} from '../module-localization.ts';
import {tr,messageRef} from '../../i18n/index.ts';
import {useWebSettings,settingsFields} from '../../state/web-settings.ts';
import ResearchUpload from '../../components/ResearchUpload.vue';
import {syncNumberInput} from '../../design/numeric-input.ts';
import AppIcon from '../../components/AppIcon.vue';
import {vTimePrecision} from '../../design/time-precision.ts';
import {vResizablePanels} from '../../layout/resizablePanels.ts';
import {annotationShortcut} from './shortcuts.ts';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import {ref,shallowRef,computed,watch,reactive,onMounted,onUnmounted,markRaw,nextTick,defineAsyncComponent} from 'vue';
import type {ResearchContext,ResearchFile} from '../../platform/research.ts';import {annotationAudio} from '../../platform/annotation.ts';
import {portableAnnotation,lipTrack,type LipTrack,type AnnotationTarget} from '../../platform/annotation.ts';
import {host,workspace} from '../../state/workspace.ts';import {stop,play,pause,playback,volume,isCurrentAudio} from '../../state/audio.ts';
import WaveformViewport from '../../components/WaveformViewport.vue';import ModalDialog from '../../components/ModalDialog.vue';
import AnnotationTracks from './AnnotationTracks.vue';import {createEditor,type Grid} from './editor.mjs';import {parseGrid,serializeGrid,decodeText,loadEditingGrid,validateEditingAudio,preferredGrid,selectedInterval} from './format.ts';import dictionary from './default.dict?raw';
import type {KeyboardSymbol} from './IpaKeyboardChart.vue';
import {insertSymbol} from '../ipa-plus/editor.ts';
const IpaKeyboard=defineAsyncComponent(()=>import('./IpaKeyboard.vue'));
import {sequenceDoubleClick,manualDoubleClick,resetSequence,sequenceParts,hasUnsplitPhones,splitInitialPhones,type PhoneSplitMode} from './sequence.ts';
import {editableTier,editingNames,newIntervalTiers,editorSelection} from './layers.ts';
import {AnnotationHistory,type AnnotationHistorySession,type AnnotationView,type HistoryKind} from './history.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active:boolean}>();const emit=defineEmits<{references:[];help:[];close:[]}>();
const wave=workspace(props.stateKey),revision=ref(0),error=messageRef(),notice=messageRef(),busy=ref(false),saving=ref(false),files=ref<ResearchFile[]>([]),directory=ref(''),directoryLabel=ref('');
const source=shallowRef<ResearchFile|null>(null),gridFile=shallowRef<ResearchFile|null>(null),gridSha=ref(''),lipFile=shallowRef<ResearchFile|null>(null),lipSha=ref(''),lip=shallowRef<LipTrack|null>(null),lipOffset=ref(0),savedLipOffset=ref(0),savedGrid=ref('');
const filter=ref(''),suffix=ref('_自动保存'),showOpen=ref(true),showWidth=ref(true),sidebar=ref(true),referenceName=ref(''),dictName=messageRef(localizedMessage('modules.annotation.text.390b2d289c67')),labName=ref(''),label=ref(''),replace=ref(''),query=ref(''),durationInput=ref('3.2');
const settingsOpen=ref(host.projects.read('m12-settings-open:'+props.stateKey,true));
watch(settingsOpen,value=>host.projects.write('m12-settings-open:'+props.stateKey,value));
const dialog=ref<'replace'|'overwrite'|'words'|'layers'|'association'|'links'|'intensity'|'search'|''>(''),composing=ref(false),picker=ref<HTMLInputElement>(),dictPicker=ref<HTMLInputElement>(),labPicker=ref<HTMLInputElement>(),refPicker=ref<HTMLInputElement>();
const newWord=ref(''),newPhone=ref(''),newExtra=ref(''),layerRole=ref<'both'|'word'|'phone'|'extra'>('both'),associationAudio=shallowRef<ResearchFile|null>(null),associationId=ref('');
const fitChannel=ref('');
const keyboardOpen=ref(false),labelInput=ref<HTMLInputElement>();
const sequenceEnabled=ref(false),sequenceText=ref(''),nudgeStep=ref('1');
const port=props.context.files.annotation??portableAnnotation(props.context.files);
let ticket=0,alive=true,timer:ReturnType<typeof setInterval>,repeat:ReturnType<typeof setTimeout>|undefined;
let reading:AbortController|undefined,saveFlight:Promise<boolean>|undefined;
const previewNote=ref('');
const historyStore=new AnnotationHistory(props.context.ownerId??'local',props.context.files.annotationScope??props.context.key);
const historySession=shallowRef<AnnotationHistorySession|null>(null),historyRevision=ref(0),historyChoice=ref('');
const historyVersions=computed(()=>{historyRevision.value;return [...(historySession.value?.value.versions??[])].reverse();});
const historyKinds:Record<HistoryKind,string>=localized(()=>({original:tr('modules.annotation.text.35f2e3dbba30'),recovered:tr('modules.annotation.text.6462ee2fe999'),auto:tr('modules.annotation.text.f2db3712a685'),manual:tr('modules.annotation.text.1bbb28053119'),restored:tr('modules.annotation.text.3e98228515b9')}));
function historyView():AnnotationView{return {offset:wave.offset,zoom:wave.zoom,channel:wave.channel,word:wordName.value,phone:phoneName.value,extra:extraName.value,lab:[...s.labSequence],labName:labName.value,sequence:sequenceEnabled.value,index:s.sequenceIndex};}
async function checkpoint(kind:HistoryKind='auto'){
 if(!historySession.value||!s.textgrid)return;
 await historyStore.save(historySession.value,serializeGrid(s.textgrid),kind,historyView(),gridFile.value?.name);historyRevision.value++;historyChoice.value=historySession.value.value.head;
}
const targets=new Map<string,AnnotationTarget>();
const editor=createEditor({changed:changed,message:m=>{notice.value=m;},localizedMessage:m=>{notice.value=m;}}),s=editor.state;
const controls=reactive(editor.controls);
editor.state.phoneDict=editor.parseDictText(dictionary);
const prefs=host.projects.read<{word?:string;phone?:string;extra?:string;suffix?:string;phoneSplit?:PhoneSplitMode;snapEnabled?:boolean;linkWordPhone?:boolean;linkWordExtra?:boolean;linkPhoneExtra?:boolean}>('annotation.'+props.stateKey,{});
const snapEnabled=ref(prefs.snapEnabled===true),linkWordPhone=ref(prefs.linkWordPhone!==false),linkWordExtra=ref(prefs.linkWordExtra===true),linkPhoneExtra=ref(prefs.linkPhoneExtra===true);
function syncLinks(){Object.assign(s,{snapEnabled:snapEnabled.value,linkWordPhone:linkWordPhone.value,linkWordExtra:linkWordExtra.value,linkPhoneExtra:linkPhoneExtra.value});}
syncLinks();watch([snapEnabled,linkWordPhone,linkWordExtra,linkPhoneExtra],()=>{syncLinks();persist();revision.value++;},{flush:'sync'});
const phoneSplit=ref<PhoneSplitMode>(prefs.phoneSplit==='equal'?'equal':'cursor');
const wordName=ref(''),phoneName=ref(''),extraName=ref('');suffix.value=!prefs.suffix||prefs.suffix==='_webedit'?(prefs.suffix===''?'':'_自动保存'):prefs.suffix;s.wordTierName='';s.phoneTierName='';
const tierChoices=computed(()=>{revision.value;return s.textgrid?.tiers.map(t=>({name:t.name,editable:editableTier(s.textgrid!,t),kind:t.points?tr('modules.annotation.text.e0e85b8f6393'):tr('modules.annotation.text.ad2923c0230e')}))??[];});
const hasLayers=computed(()=>{revision.value;return !!s.textgrid?.tiers.length;});
const editingReady=computed(()=>{revision.value;return !!editor.wordTier()&&!!editor.phoneTier();});
const boundaries=computed(()=>{revision.value;const times=new Map<number,string>();for(const [tier,kind] of [[editor.tierByName(s.extraTierName),'extra'],[editor.phoneTier(),'phone'],[editor.wordTier(),'word']] as const)for(const item of tier?.intervals??[])for(const t of [item.xmin,item.xmax])times.set(t,kind);return [...times].map(([time,kind])=>({time,kind}));});
const lipDirty=computed(()=>!!lip.value&&lipOffset.value!==savedLipOffset.value);
const pairs=computed(()=>files.value.filter(f=>f.kind==='audio').map(audio=>{try{return {audio,grid:preferredGrid(audio,files.value)};}catch{return {audio,grid:undefined};}}));
function associatedGrids(audio:ResearchFile){const stem=audio.name.replace(/\.wav$/i,'').toLowerCase();return files.value.filter(f=>f.kind==='textgrid'&&(f.name.toLowerCase()===stem+'.textgrid'||f.name.toLowerCase().startsWith(stem+'_')));}
const visibleFiles=computed(()=>pairs.value.filter(p=>p.audio.name.toLowerCase().includes(filter.value.toLowerCase())));
const gridChoices=computed(()=>{const name=source.value?.name??'',parent=name.includes('/')?name.slice(0,name.lastIndexOf('/')+1):'';return files.value.filter(f=>f.kind==='textgrid'&&(!parent||f.name.startsWith(parent)));});
const lipChoices=computed(()=>{const name=source.value?.name??'',parent=name.includes('/')?name.slice(0,name.lastIndexOf('/')+1):'';return files.value.filter(f=>['lip','lip_pickle'].includes(f.kind)&&!f.name.toLowerCase().endsWith('_timestamps.pkl')&&(!parent||f.name.startsWith(parent)));});
const targetName=computed(()=>source.value?source.value.name.replace(/\.wav$/i,'')+suffix.value+'.TextGrid':tr('modules.annotation.text.a96f374e86bc'));
const selected=computed(()=>{revision.value;return selectedInterval(s.textgrid,s.selected);});
// The migrated document is deliberately raw. Text changes retain object identity.
const labelDirty=computed(()=>{revision.value;return !!selected.value&&label.value!==selected.value.text;});
const matchCount=computed(()=>{revision.value;return s.searchResults.length;});
const matchPosition=computed(()=>{revision.value;return s.searchResults.length?s.searchIndex+1:0;});
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
 if(!String(controls.fitStart.value).trim()||!String(controls.fitEnd.value).trim()||!Number.isFinite(a)||!Number.isFinite(b)||a<0||b<a||b>(wave.asset?.duration??0)){error.value=localizedMessage('modules.annotation.text.4479587327c3');return false;}
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
 if(wave.end<=wave.start){notice.value=localizedMessage('modules.annotation.text.c23507eb4cfe');return;}
 const start=pausedSelection===selectionKey()&&isCurrentAudio(wave.asset,wave.channel)&&playback.position>=wave.start&&playback.position<wave.end?playback.position:wave.start;
 pausedSelection='';void play(wave.asset,start,wave.end,wave.channel);
}
watch(()=>[wave.start,wave.end,wave.channel],()=>{pausedSelection='';if(wave.asset&&isCurrentAudio(wave.asset,wave.channel)&&playback.playing)pause();});
function syncView(){s.visibleStart=wave.offset;s.visibleDuration=(wave.asset?.duration??3.2)/wave.zoom;durationInput.value=s.visibleDuration.toFixed(3);}
watch(()=>[wave.offset,wave.zoom],syncView);
watch([lipDirty,labelDirty,composing],()=>{wave.dirty=s.dirty||lipDirty.value||labelDirty.value||composing.value;});
function editLabel(text:string){editor.editText(text,sequenceEnabled.value?sequenceParts(editor,text):undefined);}
// Qt can collapse an input selection when a distant floating key scrolls into
// view, without moving focus. Keep the last deliberate edit/caret interaction.
let labelSelection:{text:string;start:number;end:number;tier:string;index:number;item:object}|undefined;
// Weak identities avoid retaining a second set of large TextGrid snapshots.
const symbolUndoIds=new WeakMap<object,number>();let symbolRevision=0;
const symbolEdits:{undo:number;selection:{tier:string;index:number};start:number;end:number}[]=[];
function rememberLabelSelection(allowCollapse=true){
 const input=labelInput.value,selection=s.selected;if(!input||!selection)return;
 const start=input.selectionStart??input.value.length,end=input.selectionEnd??start;
 if(!allowCollapse&&start===end&&labelSelection?.start!==labelSelection?.end)return;
 labelSelection={text:input.value,start,end,...selection,item:selected.value!};
}
async function insertIpa(entry:KeyboardSymbol){
 const input=labelInput.value;if(!input||!selected.value||busy.value||saving.value||composing.value)return;
 const saved=labelSelection&&labelSelection.item===selected.value&&labelSelection.text===label.value&&labelSelection.tier===s.selected?.tier&&labelSelection.index===s.selected?.index?labelSelection:undefined;
 const start=saved?.start??input.selectionStart??label.value.length,end=saved?.end??input.selectionEnd??start;
 try{const before=s.undoStack.at(-1),selection={...s.selected!},next=insertSymbol({text:label.value,start,end},entry);label.value=next.text;editLabel(next.text);changed();
 const snapshot=s.undoStack.at(-1);if(snapshot&&snapshot!==before){symbolUndoIds.set(snapshot,++symbolRevision);symbolEdits.push({undo:symbolRevision,selection,start,end});if(symbolEdits.length>50)symbolEdits.shift();}
 await nextTick();input.focus({preventScroll:true});input.setSelectionRange(next.start,next.end);rememberLabelSelection();error.value='';}catch(e){error.value=displayErrorValue(e);}
}
function finishInput(){if(composing.value){error.value=localizedMessage('modules.annotation.text.fa3c51ae2a27');return false;}if(labelDirty.value){editLabel(label.value);changed();}return true;}
watch([suffix,phoneSplit],()=>{persist();});
function persist(){return host.projects.write('annotation.'+props.stateKey,{word:wordName.value,phone:phoneName.value,extra:extraName.value,suffix:suffix.value,phoneSplit:phoneSplit.value,snapEnabled:snapEnabled.value,linkWordPhone:linkWordPhone.value,linkWordExtra:linkWordExtra.value,linkPhoneExtra:linkPhoneExtra.value});}
function applyRoles(grid:Grid,preferred:{word?:string;phone?:string;extra?:string}){const roles=editingNames(grid,preferred);wordName.value=s.wordTierName=roles.word;phoneName.value=s.phoneTierName=roles.phone;extraName.value=s.extraTierName=roles.extra;}
function names(){
 const restore=()=>{wordName.value=s.wordTierName;phoneName.value=s.phoneTierName;extraName.value=s.extraTierName;};
 if(!finishInput()){restore();return;}
 const w=wordName.value,p=phoneName.value,e=extraName.value,names=[w,p,e].filter(Boolean);
 if(new Set(names).size!==names.length){restore();error.value=localizedMessage('modules.annotation.text.990665ae9af9');return;}
 s.wordTierName=w;s.phoneTierName=p;s.extraTierName=e;s.selected=null;s.selectedBoundary=null;s.selectedIndices=[];
 s.searchResults=[];s.searchIndex=-1;query.value='';s.sequenceStart=null;resetSequence(editor);
 if(!persist())error.value=localizedMessage('modules.annotation.text.9b95cbaf2215');else error.value='';changed();
}
function layerDialog(role:'both'|'word'|'phone'|'extra'){if(!finishInput())return;layerRole.value=role;newWord.value='';newPhone.value='';newExtra.value='';error.value='';dialog.value='layers';}
function createLayers(){
 if(!s.textgrid)return;
 try{const names=layerRole.value==='both'?[newWord.value,newPhone.value,...(newExtra.value.trim()?[newExtra.value]:[])]:[layerRole.value==='word'?newWord.value:layerRole.value==='phone'?newPhone.value:newExtra.value],tiers=newIntervalTiers(s.textgrid,names);
  editor.saveUndoState();s.textgrid.tiers.push(...tiers);s.dirty=true;
  if(['both','word'].includes(layerRole.value))wordName.value=s.wordTierName=tiers[0].name;
  if(['both','phone'].includes(layerRole.value))phoneName.value=s.phoneTierName=tiers[layerRole.value==='both'?1:0].name;
  if(layerRole.value==='extra'||layerRole.value==='both'&&tiers.length===3)extraName.value=s.extraTierName=tiers.at(-1)!.name;
  s.selected=null;s.selectedIndices=[];s.selectedBoundary=null;resetSequence(editor);persist();dialog.value='';error.value='';notice.value=localizedMessage('modules.annotation.notice.tiersCreated',{tiers:tiers.map(t=>t.name).join('、')});changed();
 }catch(e){error.value=displayErrorValue(e);}
}
function workingGrid(audio:ResearchFile){const auto=files.value.filter(f=>f.kind==='textgrid'&&f.name.toLowerCase()===audio.name.replace(/\.wav$/i,'_自动保存.TextGrid').toLowerCase());if(auto.length>1)throw displayError(displayMessage('modules.annotation.text.afdac9d5df3b'));return auto[0]??preferredGrid(audio,files.value);}
function chooseAudio(audio:ResearchFile){void open(audio);}
function applyAssociation(){const audio=associationAudio.value,file=files.value.find(f=>f.id===associationId.value);if(audio&&file){dialog.value='';void open(audio,file);}}
async function prepare(role:'textgrid'|'lip',file:ResearchFile,suff='_自动保存'){
 const key=role+':'+file.id+':'+suff;let target=targets.get(key);if(!target){target=await port.target(file,role,suff);targets.set(key,target);}return target;
}
function updateSaved(file:ResearchFile,original:ResearchFile){
 const parent=original.name.includes('/')?original.name.slice(0,original.name.lastIndexOf('/')+1):'';
 const named={...file,name:file.name.includes('/')?file.name:parent+file.name};
 files.value=files.value.filter(f=>f.id!==named.id&&f.name.toLowerCase()!==named.name.toLowerCase()).concat(named);return named;
}
async function saveGrid(auto=false,confirmed=false,kind:HistoryKind=auto?'auto':'manual'):Promise<boolean>{
 if(!finishInput())return false;
 if(!s.textgrid||!source.value)return true;
 if(!s.textgrid.tiers.length){if(auto)return true;error.value=localizedMessage('modules.annotation.text.ea55edc1c7e1');return false;}
 if(!auto&&suffix.value===''&&!confirmed&&props.context.files.kind==='desktop'){dialog.value='overwrite';return false;}
 if(saving.value)return false;
 const generation=ticket,document=s.textgrid,original=gridFile.value??source.value,audio=source.value,sourceHash=gridFile.value?gridSha.value:source.value.sha256!,suff=auto?'_自动保存':suffix.value;
 saving.value=true;error.value='';
 try{
  const text=serializeGrid(document);await checkpoint(kind);const target=await prepare('textgrid',audio,suff);
  const result=await port.save({target:target.id,source:{id:original.id,sha256:sourceHash},text});
  if(alive&&generation===ticket&&s.textgrid===document){
   gridFile.value=updateSaved(result.file,original);gridSha.value=result.sha256;savedGrid.value=text;s.dirty=serializeGrid(document)!==text;wave.dirty=s.dirty||lipDirty.value;notice.value=joinStatusMessages((props.context.files.kind==='server'?localizedMessage('modules.annotation.text.449a5fafde76'):localizedMessage('modules.annotation.text.cfe6f505c0b2')),result.name);revision.value++;
  }return true;
 }catch(e){if(alive&&generation===ticket)error.value=displayErrorValue(e);return false;}finally{if(alive)saving.value=false;}
}
async function saveLip():Promise<boolean>{
 if(!lip.value||!lipFile.value||!lipDirty.value)return true;
 if(saving.value)return false;
 const generation=ticket,original=lipFile.value,hash=lipSha.value,offset=lipOffset.value;saving.value=true;error.value='';
 try{
  const target=await prepare('lip',original,''),result=await port.save({target:target.id,source:{id:original.id,sha256:hash},offset});
  if(alive&&generation===ticket){lipFile.value=updateSaved(result.file,original);lipSha.value=result.sha256;savedLipOffset.value=offset;wave.dirty=s.dirty||lipDirty.value;notice.value=localizedMessage('modules.annotation.notice.lipSaved',{name:result.name});}return true;
 }catch(e){if(alive&&generation===ticket)error.value=displayErrorValue(e);return false;}finally{if(alive)saving.value=false;}
}
function savePending():Promise<boolean>{
 if(saveFlight)return saveFlight;
 saveFlight=(async()=>{try{if(!finishInput())return false;if(s.dirty&&!await saveGrid(true))return false;if(lipDirty.value&&!await saveLip())return false;if(s.dirty||lipDirty.value||labelDirty.value||composing.value){error.value=localizedMessage('modules.annotation.text.adc483eee763');return false;}await checkpoint();return true;}catch(e){error.value=displayErrorValue(e);return false;}})().finally(()=>{saveFlight=undefined;});return saveFlight;
}
async function restoreVersion(){
 const version=historySession.value?.value.versions.find(v=>v.id===historyChoice.value);if(!version||busy.value||saving.value||!finishInput())return;
 if(!await savePending())return;
 editor.saveUndoState();s.textgrid=parseGrid(version.text,true);applyRoles(s.textgrid,{word:wordName.value,phone:phoneName.value,extra:extraName.value});
 s.selected=null;s.selectedIndices=[];s.selectedBoundary=null;resetSequence(editor);s.dirty=true;changed();
 if(await saveGrid(true,false,'restored'))notice.value=localizedMessage('modules.annotation.text.d2a3d65da7a0');
}
function download(role:'textgrid'|'lip'){
 if(!finishInput())return;
 try{
  let text:string,name:string;
  if(role==='textgrid'){if(!s.textgrid||!source.value)return;text=serializeGrid(s.textgrid);name=targetName.value.split('/').at(-1)!;}
  else{if(!lip.value||!lipFile.value)return;const wire=structuredClone(lip.value.wire);wire.data.metadata??={};wire.data.metadata.lip_manual_offset=lipOffset.value;text=JSON.stringify(wire);name=lipFile.value.name.split('/').at(-1)!.replace(/\.pkl$/i,'.lip.json');}
  const url=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notice.value=localizedMessage('modules.annotation.notice.download',{name});
 }catch(e){error.value=displayErrorValue(e);}
}
async function scan(choose=false){
 if(busy.value||saving.value)return;error.value='';
 if(!await savePending())return;busy.value=true;
 try{
  let next=directory.value;
  if(choose&&props.context.files.choose){const grant=await props.context.files.choose('input');if(!grant)return;next=grant.id;directoryLabel.value=grant.label;}
  if(props.context.files.kind==='desktop'&&!next)return;
  const values=await port.scan(next);if(!alive)return;files.value=values;directory.value=next;
  notice.value=localizedMessage('modules.annotation.text.023aa903e47c',{v0:values.filter(f=>f.kind==='audio').length,v1:pairs.value.length});
 }catch(e){error.value=displayErrorValue(e);}finally{busy.value=false;}
}
async function addFiles(event:Event){const element=event.target as HTMLInputElement;props.context.files.add?.(Array.from(element.files??[]));element.value='';await scan();}
async function open(audio:ResearchFile,explicitGrid?:ResearchFile){
 if(busy.value||saving.value)return;busy.value=true;let own=ticket;
 try{
  if(!await savePending())return;
  own=++ticket;reading?.abort();reading=new AbortController();error.value='';notice.value=localizedMessage('modules.annotation.text.79af5b36603b');stop();
  let grid:ResearchFile|undefined;try{grid=explicitGrid??workingGrid(audio);}catch{associationAudio.value=audio;associationId.value='';dialog.value='association';return;}
  const [decoded,read]=await Promise.all([annotationAudio(props.context.files,audio,reading.signal),grid?props.context.files.read(grid,reading.signal):Promise.resolve(undefined)]);
  validateEditingAudio(decoded.asset);const loaded=loadEditingGrid(read?decodeText(read.buffer):undefined,decoded.asset.duration),diskText=serializeGrid(loaded.grid);
  const original=files.value.find(f=>f.kind==='textgrid'&&f.name.toLowerCase()===audio.name.replace(/\.wav$/i,'.TextGrid').toLowerCase());
  const baseline=original&&original.id!==grid?.id?serializeGrid(loadEditingGrid(decodeText((await props.context.files.read(original,reading.signal)).buffer),decoded.asset.duration).grid):diskText;
  const session=await historyStore.open(audio.annotationKey??audio.id,decoded.sha256,baseline,diskText);
  const current=historyStore.head(session),document=!explicitGrid?loadEditingGrid(current.text,decoded.asset.duration).grid:loaded.grid,created=!document.tiers.length;
  if(!alive||own!==ticket)return;
  previewNote.value=decoded.previewNote;
  source.value={...audio,sha256:decoded.sha256};gridFile.value=grid??null;gridSha.value=read?.sha256??'';wave.asset=markRaw(decoded.asset);wave.channel=0;wave.offset=0;wave.zoom=Math.max(1,decoded.asset.duration/3.2);wave.start=0;wave.end=0;pausedSelection='';
  s.textgrid=document;s.audioBuffer={duration:decoded.asset.duration,sampleRate:decoded.asset.sampleRate,numberOfChannels:decoded.asset.channels.length,getChannelData:index=>decoded.asset.channels[index]};s.visibleStart=0;s.visibleDuration=decoded.asset.duration/wave.zoom;
  applyRoles(document,{word:wordName.value||prefs.word,phone:phoneName.value||prefs.phone,extra:extraName.value||prefs.extra});
  s.selected=null;s.selectedBoundary=null;s.selectedIndices=[];s.undoStack=[];s.drag=null;s.dirty=serializeGrid(document)!==diskText;s.copiedWord='';s.copiedLabIndex=null;s.searchResults=[];s.searchIndex=-1;query.value='';s.sequenceIndex=0;s.sequenceStart=null;
  savedGrid.value=diskText;historySession.value=session;historyRevision.value++;historyChoice.value=explicitGrid?'':session.value.head;lip.value=null;lipFile.value=null;lipOffset.value=0;savedLipOffset.value=0;s.labSequence=[];s.labWords=new Set();labName.value='';
  fitChannel.value='';controls.fitStart.value='0';controls.fitEnd.value='0';controls.spliceStart.value='0';controls.spliceEnd.value=decoded.asset.duration.toFixed(3);targets.clear();
  await prepare('textgrid',source.value,'_自动保存');
  const lab=files.value.find(f=>f.kind==='lab'&&f.name.toLowerCase()===audio.name.replace(/\.wav$/i,'.lab').toLowerCase());
  if(lab){const bytes=await props.context.files.read(lab);if(own!==ticket||!alive)return;setLab(decodeText(bytes.buffer),lab.name);}
  const view=!explicitGrid?session.value.view:undefined;
  if(view){wave.zoom=Math.max(1,view.zoom);wave.offset=Math.max(0,Math.min(decoded.asset.duration-decoded.asset.duration/wave.zoom,view.offset));wave.channel=Math.max(0,Math.min(decoded.asset.channels.length-1,view.channel));
   applyRoles(document,{word:view.word,phone:view.phone,extra:view.extra});
   s.labSequence=[...view.lab];s.labWords=new Set(view.lab.map(w=>w.toLowerCase()));labName.value=view.labName;sequenceEnabled.value=view.sequence;s.sequenceIndex=Math.max(0,Math.min(view.lab.length,view.index));}
  const stem=audio.name.replace(/\.wav$/i,''),parent=audio.name.includes('/')?audio.name.slice(0,audio.name.lastIndexOf('/')+1):'';
  const preferred=[stem+'.lip.json',stem+'.pkl',parent+'audio_recording.lip.json',parent+'audio_recording.pkl'];
  const candidates=preferred.map(name=>files.value.find(f=>f.name.toLowerCase()===name.toLowerCase())).filter(Boolean) as ResearchFile[];
  // Fallback only in a single-recording directory, so unrelated WAVs never share a PKL silently.
  const candidate=candidates.find(f=>f.name.toLowerCase().startsWith(stem.toLowerCase()+'.'))??(files.value.filter(f=>f.kind==='audio'&&(f.name.includes('/')?f.name.slice(0,f.name.lastIndexOf('/')+1):'')===parent).length===1?candidates[0]:undefined);
  if(candidate)try{await loadLip(candidate,own);}catch(e){error.value=localizedMessage('modules.annotation.error.lipRead',{detail:displayErrorValue(e)});}
  if(own===ticket&&alive){notice.value=created?localizedMessage('modules.annotation.text.5d615906fc4d'):!explicitGrid&&session.raw?joinStatusMessages(localizedMessage('modules.annotation.text.1cd91dc88d6a'),audio.name):joinStatusMessages(localizedMessage('modules.annotation.text.83948700a8b7'),(grid?.name??audio.name));syncView();changed();}
 }catch(e){if(alive&&own===ticket){error.value=displayErrorValue(e);notice.value=localizedMessage('modules.annotation.text.7214910c5403');}}finally{if(alive&&own===ticket)busy.value=false;}
}
async function loadLip(file:ResearchFile,own=ticket){
 const read=await port.lip(file),track=lipTrack(read.wire);await prepare('lip',file,'');if(own!==ticket||!alive)return;
 lip.value=markRaw(track);lipFile.value=file;lipSha.value=read.sha256;lipOffset.value=track.offset;savedLipOffset.value=track.offset;
}
async function chooseLip(id:string){if(busy.value||saving.value)return;if(!await saveLip())return;if(lipDirty.value){error.value=localizedMessage('modules.annotation.text.6701d56420c8');return;}const file=files.value.find(f=>f.id===id);if(!file){lip.value=null;lipFile.value=null;return;}busy.value=true;try{await loadLip(file);error.value='';}catch(e){error.value=displayErrorValue(e);}finally{busy.value=false;}}
function setLab(text:string,name:string){if(text.length>2_000_000)throw displayError(displayMessage('modules.annotation.text.5fdc8ed8dc05'));const words=text.split(/\s+/).filter(Boolean);if(!words.length)throw displayError(displayMessage('modules.annotation.text.62ad55d5cb72'));if(words.length>10000||words.some(w=>w.length>200))throw displayError(displayMessage('modules.annotation.text.8c2f542e09e5'));s.labSequence=words;s.labWords=new Set(words.map(w=>w.toLowerCase()));s.copiedLabIndex=null;labName.value=name;resetSequence(editor);revision.value++;}
function clearLab(){s.labSequence=[];s.labWords=new Set();s.copiedLabIndex=null;labName.value='';resetSequence(editor);changed();}
function applySequenceText(){try{setLab(sequenceText.value,tr('modules.annotation.text.b417fd5278b9'));dialog.value='';error.value='';notice.value=localizedMessage('modules.annotation.text.fdf321321329');changed();}catch(e){error.value=displayErrorValue(e);}}
function doubleTime(time:number,ctrl=false){
 if(busy.value||saving.value||!finishInput())return;
 try{notice.value=sequenceEnabled.value?sequenceDoubleClick(editor,time,ctrl,phoneSplit.value):manualDoubleClick(editor,time);error.value='';changed();if(s.sequenceStart!==null){wave.start=s.sequenceStart;wave.end=s.sequenceStart;}}
 catch(e){error.value=displayErrorValue(e);}
}
function doublePhone(time:number){
 if(busy.value||saving.value||!finishInput())return;
 try{
  const word=editor.tiersLinked(s.wordTierName,s.phoneTierName)?editor.wordTier()?.intervals.find(w=>w.text.trim()&&time>w.xmin&&time<w.xmax):undefined;
  const labels=word?(sequenceEnabled.value?sequenceParts(editor,word.text):editor.pinyinToPhones(word.text)):[];
  if(word&&labels.length>1&&hasUnsplitPhones(editor,word)){
   notice.value=splitInitialPhones(editor,word,time,phoneSplit.value,labels);
   s.selected={tier:s.phoneTierName,index:editor.phoneTier()!.intervals.findIndex(i=>i.xmin===word.xmin)};
  }else{editor.saveUndoState();editor.splitPhoneAt(time);notice.value=localizedMessage('modules.annotation.text.6e047262ad69');}
  error.value='';changed();
 }catch(e){error.value=displayErrorValue(e);}
}
function cancelStart(){s.sequenceStart=null;notice.value=localizedMessage('modules.annotation.text.18818b391123');changed();}
function undo(){if(s.sequenceStart!==null){cancelStart();return;}editor.undo();if(s.textgrid){applyRoles(s.textgrid,{word:wordName.value,phone:phoneName.value,extra:extraName.value});s.dirty=serializeGrid(s.textgrid)!==savedGrid.value;}changed();}
function moveAnnotation(direction:number){
 if(busy.value||saving.value||!finishInput())return;
 try{const ms=Number(nudgeStep.value);if(!String(nudgeStep.value).trim()||!Number.isFinite(ms)||ms<.01||ms>1000)throw displayError(displayMessage('modules.annotation.text.fa616f8f8d6a'));editor.moveSelected(direction*ms/1000);error.value='';notice.value=localizedMessage('modules.annotation.text.991fa03f3598',{v0:direction<0?localizedMessage('modules.annotation.text.e2e0c454e5d5'):localizedMessage('modules.annotation.text.14ab1f2bce0a'),v1:ms});changed();}
 catch(e){error.value=displayErrorValue(e);}
}
function nextSequence(event:Event){s.sequenceIndex=Number((event.target as HTMLSelectElement).value);s.sequenceStart=null;changed();}
watch(sequenceEnabled,()=>{s.sequenceStart=null;changed();});
async function resource(event:Event,kind:'dict'|'lab'|'reference'){
 const input=event.target as HTMLInputElement,file=input.files?.[0],own=ticket;if(!file)return;input.value='';
 try{if(file.size>2_000_000)throw displayError(displayMessage('modules.annotation.text.3462d5700cd1'));const text=decodeText(await file.arrayBuffer());if(!alive||ticket!==own)return;
  if(kind==='dict'){const map=editor.parseDictText(text);if(!map.size||text.split(/\r?\n/).some(l=>l.trim()&&l.trim().split(/\s+/).length<2))throw displayError(displayMessage('modules.annotation.text.d22446fafffa'));s.phoneDict=map;dictName.value=localizedMessage('modules.annotation.dictionary.loaded',{name:file.name,count:map.size});}
  else if(kind==='lab')setLab(text,file.name);else{s.referenceTextGrid=parseGrid(text);referenceName.value=file.name;}
  error.value='';notice.value=localizedMessage('modules.annotation.notice.loaded',{name:file.name});changed();
 }catch(e){if(ticket===own&&alive)error.value=displayErrorValue(e);}
}
function view(){const value=Number(durationInput.value),d=wave.asset?.duration??0;if(!Number.isFinite(value)||value<.08||value>d){error.value=localizedMessage('modules.annotation.text.46914c13726f');return;}wave.zoom=d/value;wave.offset=Math.min(wave.offset,d-value);syncView();}
function nudge(ms:number){if(!lip.value)return;const value=lipOffset.value+ms/1000;if(Math.abs(value)<=3600)lipOffset.value=Number(value.toFixed(6));}
function stopRepeat(){if(repeat)clearTimeout(repeat);repeat=undefined;}
function startRepeat(ms:number){stopRepeat();nudge(ms);repeat=setTimeout(function tick(){nudge(ms);repeat=setTimeout(tick,40);},150);}
const lipOffsetInput=computed({get:()=>lipOffset.value*1000,set:(value:number|string)=>{if(!String(value).trim()||!Number.isFinite(Number(value))||Math.abs(Number(value))>3_600_000){error.value=localizedMessage('modules.annotation.text.5df02683c381');return;}lipOffset.value=Number(value)/1000;error.value='';}});
function search(){controls.searchInput.value=query.value;editor.doSearch(query.value);changed();}
function deleteCurrent(){if(busy.value||saving.value||!finishInput())return;try{if(s.selectedBoundary)editor.deleteSelectedBoundary();else editor.deleteAnnotation();error.value='';changed();}catch(e){error.value=displayErrorValue(e);}}
function selectFitChannel(){if(wave.asset&&isCurrentAudio(wave.asset,wave.channel)&&playback.playing)pause();wave.channel=Number(fitChannel.value);}
function openTool(name:'links'|'intensity'|'search'){if(busy.value||saving.value||!finishInput())return;error.value='';notice.value='';dialog.value=name;if(name==='search')search();}
async function action(name:'fit'|'splice'|'phones'|'delete'){
 error.value='';try{
  if(!s.textgrid)throw displayError(displayMessage('modules.annotation.text.1e0894c51150'));
  if(name==='splice'){if(!s.referenceTextGrid)throw displayError(displayMessage('modules.annotation.text.a79ceb29567d'));await editor.applyReferenceSplice();}
  else if(name==='delete')editor.deleteSelectedBoundary();
  else{if(!editor.wordTier())throw displayError(displayMessage('modules.annotation.text.dbd1a2b7714f'));if(name==='fit'){if(fitChannel.value==='')throw displayError(displayMessage('modules.annotation.text.d47dd32e32a5'));if(!fitRangeInput())return;editor.fitIntensityRange(Number(fitChannel.value));controls.fitStart.value=String(wave.start);controls.fitEnd.value=String(wave.end);}else if(name==='phones')editor.autoPhonesForSelection();}
  changed();
 }catch(e){error.value=displayErrorValue(e);}
}
function key(event:KeyboardEvent){
 if(!props.active||!s.textgrid||dialog.value||busy.value||saving.value||event.isComposing)return;
 if(event.key==='Escape'&&keyboardOpen.value){event.preventDefault();keyboardOpen.value=false;return;}
 const command=annotationShortcut(event);
 // Save commits the label field too; editing shortcuts retain native input behavior.
 if(command==='save'){event.preventDefault();void saveGrid();return;}
 const element=event.target as HTMLElement,symbol=symbolEdits.at(-1),snapshot=s.undoStack.at(-1);
 if(command==='undo'&&element===labelInput.value&&symbol&&!labelDirty.value&&snapshot&&symbolUndoIds.get(snapshot)===symbol.undo&&s.selected?.tier===symbol.selection.tier&&s.selected.index===symbol.selection.index){
  event.preventDefault();editor.undo();s.selected={...symbol.selection};symbolEdits.pop();s.dirty=serializeGrid(s.textgrid)!==savedGrid.value;changed();
  void nextTick(()=>{labelInput.value?.focus({preventScroll:true});labelInput.value?.setSelectionRange(symbol.start,symbol.end);rememberLabelSelection();});return;
 }
 if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(element.tagName)||element.isContentEditable)return;
 const k=event.key.toLowerCase();
 const deletion=['Backspace','Delete'].includes(event.key),clipboardAction=command&&['copy','cut','paste'].includes(command),remove=deletion&&!s.selectedBoundary;
 if(clipboardAction||remove){
  event.preventDefault();if(!finishInput())return;
  try{if(remove)editor.deleteAnnotation();else if(k==='v')editor.pasteAnnotation();else editor.copyAnnotation(k==='x');error.value='';changed();}
  catch(e){error.value=displayErrorValue(e);}return;
 }
 if(event.altKey&&['ArrowLeft','ArrowRight'].includes(event.key)){nudge((event.key==='ArrowLeft'?-1:1)*(event.shiftKey?10:1));}
 else if(deletion&&s.selectedBoundary){void action('delete');}
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
useWebSettings(props.stateKey,{suffix,showOpen,showWidth,sidebar,settingsOpen,phoneSplit,durationInput,nudgeStep,snapEnabled,linkWordPhone,linkWordExtra,linkPhoneExtra,...settingsFields(wave,['showBoth','showSpectrogram'])});
</script>
<template>
<ModuleFrame unified fit :label="tr('modules.annotation.text.e181fd112a63')" class="annotation-page" :data-revision="revision" :aria-busy="busy||saving">
<template #toolbar><ModuleToolbar><ResearchUpload :files="context.files" accept=".wav,.mp3,.flac,.TextGrid" @uploaded="scan()"/><button class="primary" v-if="context.files.choose" :disabled="busy||saving" @click="scan(true)"><AppIcon name="folder"/>{{tr('modules.annotation.text.05f4f621d83d')}}</button><button class="primary" v-else-if="context.files.add" :disabled="busy||saving" @click="picker?.click()">{{tr('modules.annotation.text.5cfe692834d8')}}</button><button :disabled="busy||saving" @click="scan()">{{tr('modules.annotation.text.c2a291b64f6b')}}</button><button aria-haspopup="dialog" :disabled="busy||saving" @click="openTool('links')">{{tr('modules.annotation.text.d801b0746892')}}</button><button aria-haspopup="dialog" :disabled="!hasLayers||busy||saving" @click="openTool('intensity')">{{tr('modules.annotation.text.9906153dfb81')}}</button><button aria-haspopup="dialog" :disabled="!hasLayers||busy||saving" @click="openTool('search')">{{tr('modules.annotation.text.6bbc1cb4cfc7')}}</button><button :aria-pressed="keyboardOpen" :disabled="busy||saving" @click="keyboardOpen=!keyboardOpen">{{tr('modules.annotation.text.495962b5b403')}}</button><template #actions><button :aria-pressed="settingsOpen" @click="settingsOpen=!settingsOpen">{{tr('modules.annotation.text.972008fc4872')}}</button><button :aria-pressed="sidebar" @click="sidebar=!sidebar">{{tr('modules.annotation.text.d846847cc6c4')}}</button><button :disabled="!hasLayers||busy||saving" class="primary" @click="saveGrid()">{{saving?tr('modules.annotation.text.6bdb4435095e'):tr('modules.annotation.text.6814391911b5')}}{{s.dirty?' *':''}}</button><button type="button" data-action="module-help" @click="emit('help')">{{tr('modules.annotation.text.a57cfcb8428d')}}</button><button @click="emit('references')"><AppIcon name="book"/>{{tr('modules.annotation.text.7eaa43419522')}}</button></template></ModuleToolbar></template>
 <ModuleStatus v-if="error" kind="error" :message="error"/><ModuleStatus v-if="notice&&busy" kind="loading" :message="notice"/>
 <div v-resizable-panels="{key:stateKey,center:'.annotation-editor',centerMin:350,panels:[{selector:'.annotation-files',side:'left',label:tr('modules.annotation.text.e00a01c34038'),initial:300,min:180,max:520},{selector:'.annotation-settings',side:'right',label:tr('modules.annotation.text.972008fc4872'),initial:300,min:210,max:560}]}" class="annotation-layout" :class="{collapsed:!sidebar,'settings-collapsed':!settingsOpen}">
 <aside v-show="sidebar" class="annotation-files annotation-card">
  <h2>{{tr('modules.annotation.text.401d2473112a')}}</h2>
  <input ref="picker" hidden type="file" multiple accept=".wav,.TextGrid,.textgrid,.lab,.json" @change="addFiles"/><small>{{directoryLabel||context.label}} · {{pairs.length}} {{tr('modules.annotation.text.5c2fb21bb390')}}</small>
  <input v-model="filter" :aria-label="tr('modules.annotation.text.8fb9af5d453e')" :placeholder="tr('modules.annotation.text.22a3c599b3c0')"/>
  <small class="file-legend"><span class="has-grid">{{tr('modules.annotation.text.ec7bb9c2333f')}}</span><span>{{tr('modules.annotation.text.ee34b14bec2e')}}</span></small>
  <div class="annotation-file-list"><button v-for="pair in visibleFiles" :key="pair.audio.id" :class="{selected:source?.id===pair.audio.id,'has-grid':!!pair.grid||!!associatedGrids(pair.audio).length}" :disabled="busy||saving" :title="pair.audio.name" @click="chooseAudio(pair.audio)"><span>{{pair.audio.name}}</span></button><p v-if="!visibleFiles.length" class="hint">{{tr('modules.annotation.text.818989e4e875')}}</p></div>
 </aside>
 <div class="annotation-editor">
  <div class="annotation-card annotation-plots"><template v-if="wave.asset"><p v-if="playback.error" class="error-text" role="alert">{{playback.error}}</p>
   <p v-if="previewNote" class="hint">{{previewNote}}{{tr('modules.annotation.text.a40205879607')}}</p>
   <div class="actions view-controls"><strong :title="source?.name">{{source?.name}}</strong><label>{{tr('modules.annotation.text.a2e1ba236886')}} <input v-time-precision="'s'" v-model="durationInput" :aria-label="tr('modules.annotation.text.154991ab9bb2')" type="number" step=".1" @change="view"/> s</label><button @click="wave.offset=Math.max(0,wave.offset-s.visibleDuration*.8)">{{tr('modules.annotation.text.fdbe51b40829')}}</button><button @click="wave.offset=Math.min(wave.asset!.duration-s.visibleDuration,wave.offset+s.visibleDuration*.8)">{{tr('modules.annotation.text.456f3494aee2')}}</button><label class="annotation-volume">{{tr('modules.annotation.text.8bf8b9780d34')}} <input :value="playback.volume" type="range" min="0" max="1" step=".01" :aria-label="tr('modules.annotation.text.8bf8b9780d34')" @input="volume(Number(($event.target as HTMLInputElement).value))"/><small>{{Math.round(playback.volume*100)}}%</small></label></div>
   <div v-if="!editingReady" class="layer-prompt"><span>{{hasLayers?tr('modules.annotation.text.0a0125689f11'):tr('modules.annotation.text.072e8f7cf946')}}</span><button class="primary" :disabled="busy||saving" @click="layerDialog('both')">{{tr('modules.annotation.text.564dfee2a9ba')}}</button></div>
   <WaveformViewport :state="wave" :compact-overview="true" :hide-overview-controls="true" :overview-top="true" :auto-amplitude="true" :shift-wheel-pan="true" :continuous-detail="true" :hide-time-axis="true" double-click-action="annotation" @time-double-click="doubleTime" @selection-end="waveformSelection" @boundary-start="boundaryStart" @boundary-move="boundaryMove" @boundary-end="boundaryEnd"><template #annotations="{x,start,end}"><template v-for="boundary in boundaries" :key="boundary.time"><path v-if="boundary.time>start&&boundary.time<end" :d="`M${x(boundary.time)} 0V90`" class="annotation-boundary" :class="[boundary.kind,{chosen:s.selectedBoundary?.time===boundary.time}]"/></template><path v-if="s.sequenceStart!==null&&s.sequenceStart>=start&&s.sequenceStart<=end" :d="`M${x(s.sequenceStart)} 0V90`" class="sequence-pending"/></template></WaveformViewport>
   <AnnotationTracks :disabled="busy||saving||composing" :editor="editor" :revision="revision" :wave="wave" :lip="lip" :offset="lipOffset" :show-open="showOpen" :show-width="showWidth" :active="active" :sequence-editing="sequenceEnabled" @time-double-click="doubleTime" @phone-double-click="doublePhone" @selection-end="waveformSelection" @changed="changed"/>
   <div class="label-editor"><label>{{tr('modules.annotation.text.178695ee5b80')}} <input ref="labelInput" v-model="label" :aria-label="tr('modules.annotation.text.9bbd8617af0d')" :disabled="!selected||busy||saving" @input="rememberLabelSelection()" @keyup="rememberLabelSelection()" @pointerup="rememberLabelSelection()" @select="rememberLabelSelection(false)" @compositionstart="composing=true" @compositionend="composing=false;finishInput();rememberLabelSelection()" @change="finishInput()" @keydown.enter="!$event.isComposing&&finishInput()"/></label><button :disabled="busy||saving||(!selected&&!s.selectedBoundary)" @click="deleteCurrent">{{s.selectedBoundary?tr('modules.annotation.text.4044fb026f88'):tr('modules.annotation.text.4d3df8d27c75')}}</button><small>{{tr('modules.annotation.text.b73a79c0f2bd')}}</small></div>
   <p class="hint">{{tr('modules.annotation.text.4815f2dc4a86')}}</p>
  </template>
  <ModuleStatus v-else kind="empty" :message="tr('modules.annotation.text.38939c956907')"><p>{{tr('modules.annotation.text.9610c0c15951')}}</p></ModuleStatus>
  </div>
  <div class="annotation-card edit-toolbar">
   <div class="actions resource-controls"><button :title="tr('modules.annotation.text.134e5014ecab')" @click="dictPicker?.click()">{{tr('modules.annotation.text.fd007a66d0a5')}}</button><small :title="dictName">{{dictName}}</small><button :title="tr('modules.annotation.text.e7665e0a1802')" @click="labPicker?.click()">{{tr('modules.annotation.text.1754f337d51c')}}</button><button @click="sequenceText=s.labSequence.join(' ');dialog='words'">{{tr('modules.annotation.text.b417fd5278b9')}}</button><button v-if="labName" @click="clearLab">{{tr('modules.annotation.text.da68c680d3bd')}}</button><small v-if="labName" :title="labName">{{labName}}</small><label class="sequence-toggle"><input v-model="sequenceEnabled" type="checkbox" :disabled="!editingReady||busy||saving"/>{{tr('modules.annotation.text.f69765fe054f')}}</label><label class="sequence-next">{{tr('modules.annotation.text.d66669199700')}}<select v-model="phoneSplit" :aria-label="tr('modules.annotation.text.b2774cdd3c1e')"><option value="cursor">{{tr('modules.annotation.text.7d55418459dd')}}</option><option value="equal">{{tr('modules.annotation.text.b29c5fb4b279')}}</option></select></label><label v-if="sequenceEnabled&&s.labSequence.length" class="sequence-next">{{tr('modules.annotation.text.f821f04ad1a7')}}<select :aria-label="tr('modules.annotation.text.f821f04ad1a7')" :value="s.sequenceIndex" :disabled="busy||saving" @change="nextSequence"><option v-for="(token,index) in s.labSequence" :key="index" :value="index">{{index+1}} / {{s.labSequence.length}} · {{token}}</option><option :value="s.labSequence.length">{{tr('modules.annotation.text.cb9cba1113ce')}}</option></select></label></div>
   <small class="resource-help">{{tr('modules.annotation.text.8a440b86925f')}}</small>
   <p v-if="sequenceEnabled||s.sequenceStart!==null" class="hint sequence-hint"><span v-if="sequenceEnabled">{{tr('modules.annotation.text.e22c519dffbd')}}</span><span v-else>{{tr('modules.annotation.text.488468a975a4')}}</span><strong v-if="s.sequenceStart!==null">{{tr('modules.annotation.text.db9def14a1a9')}} {{s.sequenceStart.toFixed(6)}} {{tr('modules.annotation.text.900ebe778e35')}}</strong><button v-if="s.sequenceStart!==null" @click="cancelStart">{{tr('modules.annotation.text.4b4e4f5a4959')}}</button></p>
   <input ref="dictPicker" hidden type="file" accept=".dict,.txt" @change="resource($event,'dict')"/><input ref="labPicker" hidden type="file" accept=".lab,.txt" @change="resource($event,'lab')"/>
  </div>
 </div>
 <aside v-show="settingsOpen" class="annotation-settings annotation-card">
 <ModuleStatus v-if="notice&&!busy" kind="info" :message="notice"/>
  <h2>{{tr('modules.annotation.text.9d458203f6cc')}}</h2><label>{{tr('modules.annotation.text.9bbaf9b21198')}}<input v-time-precision="'ms'" v-model="nudgeStep" :aria-label="tr('modules.annotation.text.27698ec9123f')" type="number" min=".01" max="1000" step=".1"/></label><small>{{tr('modules.annotation.text.12c67f88b409')}}</small><label>{{tr('modules.annotation.text.9c2deb645a16')}}<select v-model="wordName" :aria-label="tr('modules.annotation.text.4166b62ab240')" :disabled="!source||busy||saving" @change="names"><option value="">{{tr('modules.annotation.text.fec1af2dbfca')}}</option><option v-for="t in tierChoices" :key="t.name" :value="t.name" :disabled="!t.editable||t.name===phoneName||t.name===extraName">{{t.name}}{{t.editable?'':'（'+t.kind+tr('modules.annotation.text.4a6c8c91cb90')}}</option></select></label><button :disabled="!source||busy||saving" @click="layerDialog('word')">{{tr('modules.annotation.text.c2bb7e8f029e')}}</button><label>{{tr('modules.annotation.text.742732f3000e')}}<select v-model="phoneName" :aria-label="tr('modules.annotation.text.8e2ade385253')" :disabled="!source||busy||saving" @change="names"><option value="">{{tr('modules.annotation.text.91526a1ee34f')}}</option><option v-for="t in tierChoices" :key="t.name" :value="t.name" :disabled="!t.editable||t.name===wordName||t.name===extraName">{{t.name}}{{t.editable?'':'（'+t.kind+tr('modules.annotation.text.4a6c8c91cb90')}}</option></select></label><button :disabled="!source||busy||saving" @click="layerDialog('phone')">{{tr('modules.annotation.text.12c5aa7c60ed')}}</button>
  <label>{{tr('modules.annotation.text.b8a18447e41d')}}<select v-model="extraName" :aria-label="tr('modules.annotation.text.e60456403ad1')" :disabled="!source||busy||saving" @change="names"><option value="">{{tr('modules.annotation.text.a41a2a3ad140')}}</option><option v-for="t in tierChoices" :key="t.name" :value="t.name" :disabled="!t.editable||t.name===wordName||t.name===phoneName">{{t.name}}{{t.editable?'':'（'+t.kind+tr('modules.annotation.text.4a6c8c91cb90')}}</option></select></label><button :disabled="!source||busy||saving" @click="layerDialog('extra')">{{tr('modules.annotation.text.afa7fb4663d3')}}</button>
  <label>{{tr('modules.annotation.text.4fa317ce4f92')}}<select :value="gridFile?.id??''" :aria-label="tr('modules.annotation.text.4fa317ce4f92')" :disabled="!source||busy||saving" @change="open(source!,files.find(f=>f.id===($event.target as HTMLSelectElement).value))"><option v-if="source&&!gridFile" value="" disabled>{{tr('modules.annotation.text.c8377d882a74')}}</option><option v-for="f in gridChoices" :key="f.id" :value="f.id">{{f.name}}</option></select></label>
  <label>{{tr('modules.annotation.text.a2727b0df009')}}<input v-model="suffix" :aria-label="tr('modules.annotation.text.fccfea25e04f')"/></label><p class="target-preview">{{tr('modules.annotation.text.4d2d25a2888d')}}{{targetName}}</p><p v-if="suffix===''" class="overwrite-note">{{context.files.kind==='desktop'?tr('modules.annotation.text.6e73a4491054'):tr('modules.annotation.text.86e8ae92fdad')}}</p>
  <small>{{tr('modules.annotation.text.da63d79108ed')}}</small><button :disabled="!hasLayers" @click="download('textgrid')">{{tr('modules.annotation.text.1b12eda60520')}}</button><button v-if="lip" @click="download('lip')">{{tr('modules.annotation.text.2eb946711698')}}</button>
  <label>{{tr('modules.annotation.text.6fd29579ab66')}}<select v-model="historyChoice" :aria-label="tr('modules.annotation.text.56c3bd0567da')" :disabled="busy||saving||!historyVersions.length"><option v-for="version in historyVersions" :key="version.id" :value="version.id">{{historyKinds[version.kind]}} · {{new Date(version.time).toLocaleString()}}</option></select></label><button :disabled="busy||saving||!historyChoice" @click="restoreVersion">{{tr('modules.annotation.text.655f0bbd7cd2')}}</button><small>{{tr('modules.annotation.text.29b275456e38')}}</small>
  <hr/><h2>{{tr('modules.annotation.text.5bea837fab9b')}}</h2><div class="actions"><button @click="refPicker?.click()">{{tr('modules.annotation.text.1004b1442452')}}</button><button v-if="referenceName" @click="s.referenceTextGrid=null;referenceName=''">{{tr('modules.annotation.text.00946be17329')}}</button></div><input ref="refPicker" hidden type="file" accept=".TextGrid,.textgrid" @change="resource($event,'reference')"/><small>{{referenceName||tr('modules.annotation.text.73de1b778a48')}}</small>
  <label>{{tr('modules.annotation.text.70268b1fd067')}}<select v-model="controls.spliceMode.value" :aria-label="tr('modules.annotation.text.e6d8932b8984')"><option value="outside">{{tr('modules.annotation.text.4adeea6840fa')}}</option><option value="inside">{{tr('modules.annotation.text.caab6681d045')}}</option><option value="before">{{tr('modules.annotation.text.c11fbbc3920d')}}</option><option value="after">{{tr('modules.annotation.text.0ed161a79bb2')}}</option></select></label>
  <label>{{tr('modules.annotation.text.b90c4bc0103b')}}<input v-time-precision="'s'" v-model="controls.spliceStart.value" :aria-label="tr('modules.annotation.text.751e08e15d70')" type="number" step=".001"/></label><label>{{tr('modules.annotation.text.ae59579f2e31')}}<input v-time-precision="'s'" v-model="controls.spliceEnd.value" :aria-label="tr('modules.annotation.text.db0ce00d48b5')" type="number" step=".001" :disabled="['before','after'].includes(controls.spliceMode.value)"/></label>
  <button :disabled="!source||busy" @click="action('splice')">{{tr('modules.annotation.text.11c9bdad5c92')}}</button><small>{{tr('modules.annotation.text.3ab819b64c02')}}</small>
  <hr/><h2>{{tr('modules.annotation.text.596d0c5243cc')}}</h2><label>{{tr('modules.annotation.text.143a0109a963')}}<select :value="lipFile?.id??''" :aria-label="tr('modules.annotation.text.143a0109a963')" :disabled="!source||busy||saving" @change="chooseLip(($event.target as HTMLSelectElement).value)"><option value="">{{tr('modules.annotation.text.f2d5f8ca66e6')}}</option><option v-for="f in lipChoices" :key="f.id" :value="f.id">{{f.name}}</option></select></label>
  <template v-if="lip"><div class="actions"><label><input v-model="showOpen" type="checkbox"/>{{tr('modules.annotation.text.23fb336c9fd7')}}</label><label><input v-model="showWidth" type="checkbox" :disabled="!lip.width.length"/>{{tr('modules.annotation.text.0bf235a89bc7')}}</label></div><label>{{tr('modules.annotation.text.7ad58a507be5')}}<input v-time-precision="'ms'" v-model.lazy.number="lipOffsetInput" @change="syncNumberInput($event,lipOffsetInput)" :aria-label="tr('modules.annotation.text.9731a48b3db4')" type="number" step="1"/></label><div class="actions"><button :aria-label="tr('modules.annotation.text.08d6c5d04a33')" @pointerdown="startRepeat($event.shiftKey?-10:-1)" @pointerup="stopRepeat" @pointerleave="stopRepeat" @pointercancel="stopRepeat" @keydown.enter.prevent="nudge(-1)">← 1 ms</button><button :aria-label="tr('modules.annotation.text.e07f2cf9fc64')" @pointerdown="startRepeat($event.shiftKey?10:1)" @pointerup="stopRepeat" @pointerleave="stopRepeat" @pointercancel="stopRepeat" @keydown.enter.prevent="nudge(1)">1 ms →</button></div><button class="primary" :disabled="!lipDirty||saving||busy" @click="saveLip">{{tr('modules.annotation.text.9138b6fedfd9')}}{{lipDirty?' *':''}}</button><small>{{tr('modules.annotation.text.1c1e43ee7336')}}</small></template>
  <p v-else class="hint">{{tr('modules.annotation.text.625a747df0fb')}}</p>
 </aside>
 </div>
 <ModalDialog v-if="dialog==='search'" :title="tr('modules.annotation.text.6bbc1cb4cfc7')" @close="dialog=''">
  <div class="annotation-search-fields">
   <label>{{tr('modules.annotation.text.dfad9bbb853c')}}<input v-model="query" autofocus :aria-label="tr('modules.annotation.text.dfad9bbb853c')" :placeholder="tr('modules.annotation.text.ba7cf2d92022')" @input="search" @keydown.enter.prevent="$event.shiftKey?editor.findPrev():editor.findNext();changed()"/></label>
   <div class="actions"><span class="mono" role="status" :aria-label="tr('modules.annotation.text.c00bcfdede17')">{{matchPosition}} / {{matchCount}}</span><button :disabled="!matchCount" @click="editor.findPrev();changed()">{{tr('modules.annotation.text.bd4b9718f1f0')}}</button><button :disabled="!matchCount" @click="editor.findNext();changed()">{{tr('modules.annotation.text.a898d7c3d78c')}}</button></div>
   <label>{{tr('modules.annotation.text.7d37f059645b')}}<input v-model="replace" :aria-label="tr('modules.annotation.text.353d819bd956')" :placeholder="tr('modules.annotation.text.f71d1e59aeaf')"/></label>
   <div class="actions"><button :disabled="!matchCount" @click="editor.replaceCurrent(replace);changed()">{{tr('modules.annotation.text.131b13aa2659')}}</button><button :disabled="!matchCount" @click="dialog='replace'">{{tr('modules.annotation.text.2357445f30da')}}</button></div>
   <ModuleStatus v-if="notice" kind="info" :message="notice"/>
  </div>
  <template #footer><button @click="dialog=''">{{tr('modules.annotation.text.3fd47edce45b')}}</button></template>
 </ModalDialog>
 <ModalDialog v-if="dialog==='links'||dialog==='intensity'" :title="dialog==='links'?tr('modules.annotation.text.d801b0746892'):tr('modules.annotation.text.9906153dfb81')" :close-disabled="saving" @close="dialog=''">
  <template v-if="dialog==='links'">
   <div class="tier-link-controls" :aria-label="tr('modules.annotation.text.c45384b474d4')">
    <label><input v-model="snapEnabled" type="checkbox" :disabled="busy||saving"/>{{tr('modules.annotation.text.fd2d8cf53580')}}</label>
    <label><input v-model="linkWordPhone" type="checkbox" :disabled="!wordName||!phoneName||busy||saving"/>{{tr('modules.annotation.text.89aeff39f532')}}</label>
    <label v-if="extraName"><input v-model="linkWordExtra" type="checkbox" :disabled="!wordName||busy||saving"/>{{tr('modules.annotation.text.c485af210e15')}}</label>
    <label v-if="extraName"><input v-model="linkPhoneExtra" type="checkbox" :disabled="!phoneName||busy||saving"/>{{tr('modules.annotation.text.2ec3aac76698')}}</label>
    <small>{{tr('modules.annotation.text.8329941fcbbd')}}</small>
   </div>
  </template>
  <template v-else>
   <label class="fit-channel-field">{{tr('modules.annotation.text.3a79beab1fbe')}}<select v-model="fitChannel" :aria-label="tr('modules.annotation.text.2f206a5493ac')" @change="selectFitChannel"><option value="" disabled>{{tr('modules.annotation.text.26f0d65ed4e7')}}</option><option v-for="(_,index) in wave.asset?.channels??[]" :key="index" :value="String(index)">{{tr('modules.annotation.channel_number',{number:index+1})}}</option></select></label>
   <div class="intensity-fields"><label>{{tr('modules.annotation.text.48385426822f')}}<input v-time-precision="'s'" v-model="controls.fitStart.value" :aria-label="tr('modules.annotation.text.194fffcd7be0')" type="number" step=".00001" @change="fitRangeInput"/></label><label>{{tr('modules.annotation.text.71501280764a')}}<input v-time-precision="'s'" v-model="controls.fitEnd.value" :aria-label="tr('modules.annotation.text.c8af37800700')" type="number" step=".00001" @change="fitRangeInput"/></label><label>{{tr('modules.annotation.text.456675ba06e9')}}<input v-time-precision="'ms'" v-model="controls.fitTrimMs.value" :aria-label="tr('modules.annotation.text.f0a7f52dc805')" type="number" min="-50" max="80"/></label></div>
   <p class="hint">{{tr('modules.annotation.text.a7a3b926b33e')}}</p>
   <ModuleStatus v-if="notice" kind="info" :message="notice"/>
  </template>
  <p v-if="error" role="alert" class="error-text">{{error}}</p>
  <template #footer><button @click="dialog=''">{{dialog==='links'?tr('modules.annotation.text.c0b3fbff51cc'):tr('modules.annotation.text.3fd47edce45b')}}</button><button v-if="dialog==='intensity'" class="primary" :disabled="!source||busy||saving||fitChannel===''" @click="action('fit')">{{tr('modules.annotation.text.1877f59ab581')}}</button></template>
 </ModalDialog>
 <ModalDialog v-if="dialog&&dialog!=='links'&&dialog!=='intensity'&&dialog!=='search'" :title="dialog==='layers'?tr('modules.annotation.text.564dfee2a9ba'):dialog==='association'?tr('modules.annotation.text.916c42314928'):dialog==='words'?tr('modules.annotation.text.c3d885a1781f'):dialog==='replace'?tr('modules.annotation.text.1861f2d2db44'):tr('modules.annotation.text.1cdce61e36ba')" :close-disabled="saving" @close="dialog=dialog==='replace'?'search':''">
  <template v-if="dialog==='layers'"><p>{{tr('modules.annotation.text.820f91fac751')}}</p><label v-if="['both','word'].includes(layerRole)" class="new-layer-field">{{tr('modules.annotation.text.1e56440a8ac5')}}<input v-model="newWord" :aria-label="tr('modules.annotation.text.83b1b1fae524')" maxlength="200" :placeholder="tr('modules.annotation.text.6d7903621d9f')"/></label><label v-if="['both','phone'].includes(layerRole)" class="new-layer-field">{{tr('modules.annotation.text.8e2ade385253')}}<input v-model="newPhone" :aria-label="tr('modules.annotation.text.b78e22eba334')" maxlength="200" :placeholder="tr('modules.annotation.text.5168b1beffd9')"/></label><label v-if="['both','extra'].includes(layerRole)" class="new-layer-field">{{tr('modules.annotation.text.e60456403ad1')}}{{layerRole==='both'?tr('modules.annotation.text.a1b2c8118f07'):''}}<input v-model="newExtra" :aria-label="tr('modules.annotation.text.56482a7d35aa')" maxlength="200" :placeholder="tr('modules.annotation.text.390e8e0f9d21')"/></label></template>
  <template v-else-if="dialog==='association'"><p>{{associationAudio?.name}} {{tr('modules.annotation.text.4c2ae6e830b2')}}</p><select v-model="associationId" :aria-label="tr('modules.annotation.text.d73444f6c7ad')"><option value="" disabled>{{tr('modules.annotation.text.9596e0ef888a')}}</option><option v-for="file in associationAudio?associatedGrids(associationAudio):[]" :key="file.id" :value="file.id">{{file.name}}</option></select></template>
  <template v-else-if="dialog==='words'"><p>{{tr('modules.annotation.text.0501801d25f2')}}</p><textarea v-model="sequenceText" :aria-label="tr('modules.annotation.text.e665ccaeb7ec')" rows="6" maxlength="2000000" placeholder="zhe4 shi4 shang4 yi1 ding4 …"/></template>
  <p v-else-if="dialog==='replace'">{{tr('modules.annotation.text.c121c1f7130c')}} {{matchCount}} {{tr('modules.annotation.text.0c3d31d7c725')}} {{replace||tr('modules.annotation.text.98f963cb664a')}}{{tr('modules.annotation.text.e8bd99e6f5a1')}}</p><p v-else>{{tr('modules.annotation.text.beb51ed23d6e')}}{{targetName}}{{tr('modules.annotation.text.5798592f0e13')}}</p><p v-if="error" role="alert" class="error-text">{{error}}</p>
  <template #footer><button :disabled="saving" @click="dialog=dialog==='replace'?'search':''">{{tr('modules.annotation.text.2cd0f3be8738')}}</button><button class="primary" :disabled="saving||(dialog==='association'&&!associationId)" @click="dialog==='layers'?createLayers():dialog==='association'?applyAssociation():dialog==='words'?applySequenceText():dialog==='replace'?(editor.replaceAll(replace),changed(),dialog='search'):saveGrid(false,true).then(ok=>{if(ok)dialog=''})">{{dialog==='layers'?tr('modules.annotation.text.648c56cab32e'):dialog==='association'?tr('modules.annotation.text.a3cb5833dbfe'):dialog==='words'?tr('modules.annotation.text.b4f3b9976af3'):dialog==='replace'?tr('modules.annotation.text.2918d1b3a5c8'):tr('modules.annotation.text.4a1edb18155e')}}</button></template>
 </ModalDialog>
 <IpaKeyboard v-if="keyboardOpen" :active="active" :disabled="!selected||busy||saving||composing" @insert="insertIpa" @close="keyboardOpen=false"/>
</ModuleFrame>
</template>
<style scoped>
.plot-actions{margin-bottom:10px}
.annotation-search-fields{display:flex;flex-direction:column;gap:12px}.annotation-search-fields>label{display:flex;flex-direction:column;gap:6px}.annotation-search-fields input{width:100%;min-width:0;box-sizing:border-box}
.tier-link-controls{display:flex;flex-direction:column;gap:12px}.fit-channel-field{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}.intensity-fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.intensity-fields label{display:flex;flex-direction:column;gap:6px}.intensity-fields input{width:100%;min-width:0;box-sizing:border-box}@media(max-width:550px){.intensity-fields{grid-template-columns:1fr}}.tier-link-controls label{display:flex;align-items:center;gap:5px;font-size:.857143rem}.tier-link-controls small{color:var(--muted)}
.annotation-page{--wave-axis-width:max(64px,calc(var(--figure-size,14px) * 4.8))}
.annotation-file-list button span{max-width:100%}
.file-legend{display:flex;flex-wrap:wrap;gap:3px 10px;font-size:0.785714rem}.file-legend .has-grid{color:var(--accent)}.annotation-file-list button{flex:0 0 auto;min-height:36px;display:block;line-height:1.45;position:relative;--annotation-file-mark:var(--accent)}.annotation-file-list button span{display:block;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
/* File status must survive the shared button material, selection and border rules. */
.annotation-page .annotation-file-list button.has-grid:not(.selected)>span{color:var(--annotation-file-mark)!important}
.annotation-file-list button.has-grid::after{content:'';position:absolute;left:0;top:7px;bottom:7px;width:3px;border-radius:2px;background:var(--annotation-file-mark);box-shadow:0 0 0 1px var(--panel);pointer-events:none}
:global(html[data-button-style=all][data-button-material=solid] .annotation-file-list button.has-grid:not(.selected)),
:global(html[data-button-style=all][data-button-material=solid][data-background-image=on] .annotation-file-list button.has-grid){--annotation-file-mark:var(--on-accent)}
.annotation-file-list button.selected{box-shadow:inset 0 0 0 1px var(--accent)}
.resource-controls label{display:flex;align-items:center;gap:5px;font-size:0.857143rem}.resource-controls .sequence-toggle{white-space:nowrap}.resource-controls input[type=checkbox]{width:16px;height:16px}.resource-controls small{max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.sequence-next select{max-width:160px}.resource-help{font-size:0.785714rem;color:var(--muted)}
.new-layer-field{display:flex;flex-direction:column;gap:5px;margin:12px 0}.layer-prompt{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:8px 0;background:var(--teal-bg);padding:8px;border-radius:4px;font-size:0.857143rem}.selection-readout{display:flex;align-items:center;gap:14px;flex-wrap:wrap;font-size:0.857143rem;color:var(--muted);margin:4px 0 8px}.selection-readout>span:first-child{font-family:var(--font-mono)}
.annotation-boundary{stroke:var(--teal);stroke-width:1;opacity:.45;vector-effect:non-scaling-stroke;pointer-events:none}.annotation-boundary.word{opacity:.9;stroke-width:1.2}.annotation-boundary.chosen{stroke:var(--danger);opacity:1;stroke-width:2}.annotation-volume{margin-left:auto}.annotation-volume input[type=range]{width:90px}.view-controls strong{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.movement-controls{font-size:0.857143rem;padding:6px 0}.movement-controls label{display:flex;align-items:center;gap:4px}.movement-controls input{width:80px}.movement-controls button[aria-pressed=true]{color:var(--accent);border-color:var(--accent);background:var(--selected)}.annotation-file-list select{width:100%;min-width:0;font-size:0.785714rem}
.sequence-controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.sequence-controls label{display:flex;align-items:center;gap:6px;font-size:0.857143rem}.sequence-controls select{max-width:220px}.sequence-controls input[type=checkbox]{width:16px;height:16px}.sequence-hint{line-height:1.8;margin:0}.sequence-hint strong{display:block;color:var(--accent)}.sequence-pending{stroke:var(--danger);stroke-width:2;stroke-dasharray:5 3;vector-effect:non-scaling-stroke;pointer-events:none}textarea[aria-label="拼音词表"]{width:100%;margin-top:12px;resize:vertical;font-family:var(--font);line-height:1.8}
.annotation-page{min-height:100%}.annotation-header{display:flex;justify-content:space-between;gap:16px;align-items:start}.annotation-header h1{font-size:1.642857rem}.annotation-header p:not(.eyebrow){color:var(--muted);font-size:0.928571rem;margin-top:5px}.actions{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.annotation-layout{display:grid;grid-template-columns:var(--panel-left,300px) minmax(350px,1fr) var(--panel-right,300px);gap:var(--module-gap);align-items:start}.annotation-layout.collapsed{grid-template-columns:minmax(350px,1fr) var(--panel-right,300px)}.annotation-card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:12px;min-width:0}.annotation-files{display:flex;flex-direction:column;gap:10px}.annotation-file-list{max-height:630px;overflow:auto;display:flex;flex-direction:column;gap:4px}.annotation-file-list button{display:flex;flex-direction:column;align-items:start;text-align:left;min-width:0;white-space:normal;padding:8px}.annotation-file-list span,.annotation-file-list small{overflow-wrap:anywhere}.annotation-file-list .selected{border-color:var(--accent);background:var(--selected)}.annotation-editor{display:flex;flex-direction:column;gap:10px;min-width:0}.edit-toolbar,.search-tools{display:flex;flex-direction:column;gap:10px}.search-tools>.actions+.actions{padding-top:10px;border-top:1px solid var(--border)}.search-tools input:not([type=number]){width:150px}.search-tools label,.view-controls label{display:flex;align-items:center;gap:4px;font-size:0.857143rem}.search-tools input[type=number],.view-controls input{width:78px}.search-tools strong{font-size:0.857143rem}.annotation-settings{display:flex;flex-direction:column;gap:9px}.annotation-settings>label{display:flex;flex-direction:column;gap:4px;font-size:0.857143rem}.annotation-settings hr{margin:8px 0}.annotation-settings small{line-height:1.65}.target-preview{font-size:0.857143rem;overflow-wrap:anywhere;background:var(--app);padding:8px;border-radius:4px}.overwrite-note{color:var(--warning);font-size:0.857143rem}.annotation-plots :deep(.wave-track svg){height:var(--annotation-plot-height,175px)}.view-controls strong{font-size:0.857143rem;max-width:260px;overflow-wrap:anywhere}.label-editor{display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding:12px 0}.label-editor label{font-size:0.857143rem;display:flex;gap:8px;align-items:center}.label-editor input{font-family:var(--font-figure-ipa);font-size:var(--figure-size,14px);width:200px}.annotation-empty{padding:60px 24px;text-align:center}.annotation-empty p{margin-top:10px;color:var(--muted)}.notice{font-size:0.857143rem;color:var(--teal);overflow-wrap:anywhere}.error-text{overflow-wrap:anywhere}.annotation-settings input,.annotation-settings select{width:100%}.annotation-settings input[type=checkbox]{width:16px}
@media(max-width:1250px){.annotation-layout{grid-template-columns:var(--panel-left,300px) minmax(350px,1fr)}.annotation-layout.collapsed{grid-template-columns:1fr}.annotation-settings{grid-column:1/-1;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px 16px}.annotation-settings h2,.annotation-settings hr{grid-column:1/-1}.annotation-settings hr{width:100%}.annotation-file-list{max-height:600px}}
@media(max-width:780px){.annotation-page{padding:12px}.annotation-header{flex-direction:column}.annotation-layout,.annotation-layout.collapsed{display:flex;flex-direction:column}.annotation-files,.annotation-settings,.annotation-editor{width:100%}.annotation-file-list{max-height:160px}.annotation-settings{grid-template-columns:repeat(2,minmax(0,1fr))}.annotation-plots{padding:8px}.search-tools input:not([type=number]){width:125px}}
@media(max-width:480px){.annotation-settings{display:flex}.label-editor label{width:100%}.label-editor input{width:130px}.annotation-header .actions{flex-wrap:wrap}}

/* P17: bound the three panes to the available workbench height. */
.annotation-page{height:100%;min-height:0;overflow:auto;--annotation-plot-height:max(110px,calc(15dvh / var(--page-scale,1)));--annotation-grid-height:max(96px,calc(12dvh / var(--page-scale,1)))}
.annotation-layout{flex:1;min-height:0;align-items:stretch;gap:var(--module-gap)}
.annotation-files,.annotation-settings,.annotation-editor{min-height:0;overflow:auto;overscroll-behavior:contain}
.annotation-files,.annotation-settings{height:100%;padding:8px;background:var(--rail,var(--app));scrollbar-gutter:stable}
.annotation-file-list{flex:1;min-height:80px;max-height:none}
.annotation-card{padding:8px}.annotation-editor{gap:var(--module-gap)}
.annotation-editor>.annotation-card{flex-shrink:0}
@container module (max-width:1000px){.annotation-layout{flex:none;min-height:min-content}.annotation-files,.annotation-settings,.annotation-editor{height:auto;overflow:visible}.annotation-file-list{max-height:400px}}
.annotation-layout.settings-collapsed{grid-template-columns:var(--panel-left,300px) minmax(350px,1fr)}
.annotation-layout.collapsed.settings-collapsed{grid-template-columns:minmax(350px,1fr)}
</style>
<style scoped>
/* Use the actual available pane width, including the user's page scale. */
@container module (max-width:1000px){.annotation-layout{grid-template-columns:var(--panel-left,300px) minmax(350px,1fr)}.annotation-layout.collapsed{grid-template-columns:1fr}.annotation-settings{grid-column:1/-1;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px 16px}.annotation-settings h2,.annotation-settings hr{grid-column:1/-1}.annotation-settings hr{width:100%}}
@container module (max-width:720px){.annotation-layout,.annotation-layout.collapsed{display:flex;flex-direction:column}.annotation-files,.annotation-settings,.annotation-editor{width:100%}.annotation-file-list{max-height:160px}.annotation-settings{grid-template-columns:repeat(2,minmax(0,1fr))}}
@container module (max-width:440px){.annotation-settings{display:flex}.label-editor label{width:100%}.label-editor input{width:130px}}
.annotation-layout.settings-collapsed{grid-template-columns:var(--panel-left,300px) minmax(350px,1fr)}
.annotation-layout.collapsed.settings-collapsed{grid-template-columns:minmax(350px,1fr)}
</style>
