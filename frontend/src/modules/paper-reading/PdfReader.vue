<script setup lang="ts">
import {useWebSettings} from '../../state/web-settings.ts';
import {syncNumberInput} from '../../design/numeric-input.ts';
import {computed,nextTick,onBeforeUnmount,onMounted,ref,shallowRef,useId,watch} from 'vue';
import ModalDialog from '../../components/ModalDialog.vue';
import AnnotationNoteDialog from './AnnotationNoteDialog.vue';
import {AnnotationHistory,copyAnnotation,type AnnotationChange,type AnnotationDirection} from './annotation-history.ts';
import {papers as desktopPapers,type PapersClient,type Paper,type PaperLanguage,type PaperPage,type PaperLine,type PaperAnnotation,type PaperAnnotations} from '../../platform/papers.ts';
import {copyText} from '../../platform/clipboard.ts';
import {pageScale} from '../../state/pageZoom.ts';
import {rasterWidth,wheelZoom,MIN_PDF_ZOOM,MAX_PDF_ZOOM} from './pdf-view.ts';
const props=defineProps<{paper:Paper;language:PaperLanguage;initialPage:number;active?:boolean;port?:PapersClient;settingsScope?:string}>();
const papers:PapersClient=props.port??desktopPapers;
const emit=defineEmits<{language:[PaperLanguage];position:[number];fullscreen:[boolean]}>();
const reader=ref<HTMLElement>(),viewport=ref<HTMLElement>();
const notesId=useId(),notesPanel=ref<HTMLElement>();
type CachedPage=PaperPage&{renderWidth:number};
const page=ref(props.initialPage),ratios=ref<number[]>([]),cache=shallowRef<Record<number,CachedPage>>({}),loading=ref(0),failure=ref(''),notice=ref('');
const zoom=ref(100),fit=ref(true),width=ref(800),full=ref(false),showText=ref(false),sidebar=ref(false),color=ref<PaperAnnotation['color']>('yellow');
const annotations=ref<PaperAnnotations>({revision:0,items:[]}),saving=ref(false),annotationError=ref(''),annotationsReady=ref(false);
const selection=ref<{page:number;quote:string;rects:number[][]}[]>([]),editing=ref<PaperAnnotation>(),draft=ref('');
const history=ref(new AnnotationHistory()),selectedHighlightId=ref('');
const selectedHighlight=computed(()=>annotations.value.items.find(item=>item.id===selectedHighlightId.value&&item.kind==='highlight'));
const exporting=ref(false),exportOpen=ref(false),exportLanguage=ref<PaperLanguage>(props.language),exportAnnotated=ref(false);
const pageWidth=computed(()=>fit.value?Math.max(280,Math.min(1100,width.value-32)):816*zoom.value/100);
const deviceRatio=ref(window.devicePixelRatio||1);
const renderWidth=computed(()=>rasterWidth(pageWidth.value,deviceRatio.value,pageScale.value));
const zoomOptions=computed(()=>[...new Set([25,50,75,100,125,150,175,200,250,300,zoom.value])].sort((a,b)=>a-b));
const activeText=computed(()=>cache.value[page.value]?.text??'当前页正在加载。');
const languageTitle=computed(()=>props.paper.notice?'栏目说明':props.paper.local?'本地 PDF':{original:'原文',translation:'中文译文',analysis:'论文解读'}[props.language]);
const offsets=computed(()=>{let y=16;return ratios.value.map(r=>{const top=y;y+=pageWidth.value*r+12;return top;});});
const lineStyles=new Map<string,Record<string,string>>();
let disposed=false,controller:AbortController|undefined,annotationController:AbortController|undefined,resize:ResizeObserver|undefined,pumping=false,epoch=0,rasterEpoch=0;
let frame=0,selectionFrame=0,renderTimer:ReturnType<typeof setTimeout>|undefined;
type ZoomAnchor={n:number;x:number;y:number;vx:number;vy:number};
let zoomAnchor:ZoomAnchor|undefined;
const wanted=new Set<number>();
const canvas=document.createElement('canvas'),measure=canvas.getContext('2d')!;
function textStyle(line:PaperLine,ratio:number){const key=`${pageWidth.value}:${ratio}:${line.x}:${line.y}:${line.w}:${line.h}:${line.text}`;const previous=lineStyles.get(key);if(previous)return previous;const height=line.h*pageWidth.value*ratio;measure.font=`${height*1.12}px "Times New Roman", "SimSun", serif`;const natural=measure.measureText(line.text).width||1;const style={left:`${line.x*100}%`,top:`${line.y*100}%`,fontSize:`${height*1.12}px`,height:`${height}px`,transform:`scaleX(${line.w*pageWidth.value/natural})`};if(lineStyles.size>10000)lineStyles.clear();lineStyles.set(key,style);return style;}
function sheet(n:number){return viewport.value?.querySelector<HTMLElement>(`[data-page="${n}"]`);}
const readerActive=()=>props.active!==false;
async function pump(){
 if(pumping||disposed||!readerActive())return;pumping=true;
 try{while(wanted.size&&!disposed&&readerActive()){const n=[...wanted].sort((a,b)=>Math.abs(a-page.value)-Math.abs(b-page.value))[0];wanted.delete(n);const pixels=renderWidth.value;if(cache.value[n]?.renderWidth===pixels)continue;
  controller=new AbortController();loading.value=n;const token=rasterEpoch;
  try{const result=await papers.render(props.paper.id,props.language,n,pixels,controller.signal);if(disposed)return;
   if(token!==rasterEpoch)continue;const next={...cache.value,[n]:{...result,renderWidth:pixels}};for(const key of Object.keys(next).map(Number))if(Math.abs(key-page.value)>4)delete next[key];
   const decoded=(p:CachedPage)=>(p.imageWidth??p.renderWidth)*(p.imageHeight??p.renderWidth*p.ratio)*4;
   const farthest=Object.keys(next).map(Number).sort((a,b)=>Math.abs(b-page.value)-Math.abs(a-page.value));let bytes=Object.values(next).reduce((v,p)=>v+p.image.length,0),decodedBytes=Object.values(next).reduce((v,p)=>v+decoded(p),0);for(const key of farthest){if(bytes<=48_000_000&&decodedBytes<=160_000_000)break;if(key!==n&&key!==page.value){bytes-=next[key].image.length;decodedBytes-=decoded(next[key]);delete next[key];}}
   cache.value=next;if(!ratios.value.length){ratios.value=result.ratios;await nextTick();jump(Math.max(1,Math.min(props.initialPage,result.pages)));}
  }catch(e){if(token!==rasterEpoch||controller.signal.aborted)continue;if(!disposed)failure.value=(e as Error).message;break;}
 }}finally{pumping=false;loading.value=0;}
}
function schedule(){if(frame)return;frame=requestAnimationFrame(()=>{frame=0;scan();});}
function scan(){
 if(props.active===false||!viewport.value||!ratios.value.length)return;
 const y=viewport.value.scrollTop+32,top=offsets.value;let low=0,high=top.length-1;
 while(low<high){const middle=Math.ceil((low+high)/2);if(top[middle]<=y)low=middle;else high=middle-1;}
 const closest=low+1;if(page.value!==closest){page.value=closest;emit('position',closest);}wanted.clear();
 if(loading.value&&Math.abs(loading.value-closest)>4)controller?.abort();
 if(renderTimer)return;
 const nearby=Array.from({length:Math.min(ratios.value.length,closest+2)-Math.max(1,closest-1)+1},(_,i)=>Math.max(1,closest-1)+i).sort((a,b)=>Math.abs(a-closest)-Math.abs(b-closest)||b-a);let pixels=0;
 for(const n of nearby){const size=Math.min(16_777_216,renderWidth.value**2*ratios.value[n-1])*4;if(n!==closest&&pixels+size>160_000_000)continue;pixels+=size;if(cache.value[n]?.renderWidth!==renderWidth.value)wanted.add(n);}
 void pump();
}
function jump(n:number){if(!Number.isInteger(n)||n<1||n>ratios.value.length)return;clearSelection(true);page.value=n;emit('position',n);const el=sheet(n);if(el&&viewport.value)viewport.value.scrollTop=el.offsetTop-16;schedule();}
const pageInput=computed({get:()=>page.value,set:(value:number|string)=>jump(Number(value))});
function anchor(clientX?:number,clientY?:number):ZoomAnchor|undefined{
 const v=viewport.value;if(!v||!ratios.value.length)return;const b=v.getBoundingClientRect(),scale=pageScale.value/100;
 const x=clientX??b.left+b.width/2,y=clientY??b.top+Math.min(b.height/2,200*scale);
 const n=[...v.querySelectorAll<HTMLElement>('[data-page]')].find(el=>{const r=el.getBoundingClientRect();return r.top<=y&&r.bottom>y;})??sheet(page.value);if(!n)return;
 const r=n.getBoundingClientRect();return {n:Number(n.dataset.page),x:Math.max(0,Math.min(1,(x-r.left)/r.width)),y:Math.max(0,Math.min(1,(y-r.top)/r.height)),vx:(x-b.left)/scale,vy:(y-b.top)/scale};
}
function restoreAnchor(a:ZoomAnchor|undefined){const v=viewport.value,e=a&&sheet(a.n);if(v&&e&&a){v.scrollTop=e.offsetTop+a.y*e.offsetHeight-a.vy;v.scrollLeft=e.offsetLeft+a.x*e.offsetWidth-a.vx;}}
function changeZoom(value:string){const fitting=value==='fit',percent=fitting?zoom.value:Math.max(MIN_PDF_ZOOM,Math.min(MAX_PDF_ZOOM,Number(value)));if(fitting===fit.value&&(fitting||percent===zoom.value))return;zoomAnchor=anchor();fit.value=fitting;if(!fitting)zoom.value=percent;}
function wheel(event:WheelEvent){
 if(!event.ctrlKey||props.active===false||showText.value||document.querySelector('dialog[open]'))return;
 event.preventDefault();if(!event.deltaY)return;
 const next=wheelZoom(pageWidth.value/816*100,event.deltaY,event.deltaMode,viewport.value?.clientHeight??800);if(!fit.value&&next===zoom.value)return;zoomAnchor=anchor(event.clientX,event.clientY);fit.value=false;zoom.value=next;
}
function refreshRaster(){rasterEpoch++;controller?.abort();wanted.clear();clearTimeout(renderTimer);renderTimer=undefined;if(props.active===false)return;renderTimer=setTimeout(()=>{renderTimer=undefined;if(!ratios.value.length){wanted.add(1);void pump();}else schedule();},140);}
function densityChanged(){deviceRatio.value=window.devicePixelRatio||1;}
function key(event:KeyboardEvent){if(props.active===false||event.defaultPrevented||document.querySelector('dialog[open]')||(event.target as HTMLElement)?.closest('input,textarea,[contenteditable=true]'))return;
 if((event.ctrlKey||event.metaKey)&&!event.altKey&&event.key.toLowerCase()==='z'){event.preventDefault();void travel(event.shiftKey?'redo':'undo');return;}
 if((event.target as HTMLElement)?.closest('select'))return;
 if(event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
 if((event.key==='Delete'||event.key==='Backspace')&&selectedHighlight.value){event.preventDefault();void save(selectedHighlight.value,true);return;}
 if(event.key==='Escape'&&selectedHighlight.value){event.preventDefault();clearSelection(true);return;}
 if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();jump(page.value+(event.key==='ArrowLeft'?-1:1));}
 if(event.key==='Escape'&&full.value){event.preventDefault();void document.exitFullscreen();}
}
async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else await reader.value?.requestFullscreen();}catch(e){notice.value='无法进入全屏：'+(e as Error).message;}}
function fullscreenChanged(){full.value=document.fullscreenElement===reader.value;emit('fullscreen',full.value);nextTick(schedule);}
function selectText(){
 if(props.active===false)return;
 // Keep the last PDF selection when toolbar focus collapses the native range.
 // A new pointer selection, navigation or document switch explicitly clears it.
 const s=window.getSelection();if(!s||s.isCollapsed||!s.rangeCount)return;const range=s.getRangeAt(0);
 if(!viewport.value?.contains(range.commonAncestorContainer))return;
 const groups:typeof selection.value=[];
 for(const el of viewport.value.querySelectorAll<HTMLElement>('[data-page]')){
  const r=el.getBoundingClientRect(),rects:number[][]=[],texts:string[]=[];
  for(const span of el.querySelectorAll<HTMLElement>('.pdf-text-line')){if(!range.intersectsNode(span)||!span.firstChild)continue;
   const part=document.createRange(),node=span.firstChild;part.selectNodeContents(span);
   if(range.startContainer===node)part.setStart(node,range.startOffset);if(range.endContainer===node)part.setEnd(node,range.endOffset);
   if(!part.toString())continue;texts.push(part.toString());
   for(const b of part.getClientRects()){const x=Math.max(0,(b.left-r.left)/r.width),y=Math.max(0,(b.top-r.top)/r.height);if(b.width>0&&b.height>0)rects.push([x,y,Math.min(1-x,b.width/r.width),Math.min(1-y,b.height/r.height)]);}
  }
  if(rects.length)groups.push({page:Number(el.dataset.page),quote:texts.join('\n'),rects});
 }
 selection.value=groups;
}
function clearSelection(native=false){cancelAnimationFrame(selectionFrame);selectionFrame=0;selection.value=[];selectedHighlightId.value='';if(native){const s=window.getSelection();if(s?.rangeCount&&viewport.value?.contains(s.getRangeAt(0).commonAncestorContainer))s.removeAllRanges();}}
function selectionPointerDown(event:PointerEvent){if(props.active===false||event.button!==0)return;const target=event.target as Element;if(viewport.value?.contains(target))clearSelection(!target.closest('.pdf-text-line'));else if(!reader.value?.contains(target))clearSelection(true);}
function selectionChanged(){if(!selectionFrame)selectionFrame=requestAnimationFrame(()=>{selectionFrame=0;selectText();});}
async function copySelection(){selectText();try{await copyText(selection.value.map(x=>x.quote).join('\n\n'));notice.value='已复制所选文字。';}catch(e){notice.value=(e as Error).message;}}
async function reloadAnnotations(){if(saving.value)return;annotationController?.abort();const request=new AbortController();annotationController=request;const token=epoch;try{const value=await papers.annotations(props.paper.id,props.language,request.signal);if(token!==epoch||disposed||request.signal.aborted)return;annotations.value=value;history.value.clear();selectedHighlightId.value='';annotationError.value='';annotationsReady.value=true;}catch(e){if(token!==epoch||disposed||request.signal.aborted)return;annotationError.value=(e as Error).message;annotationsReady.value=false;}}
async function persist(item:PaperAnnotation,remove:boolean,token:number,id:string,language:PaperLanguage){
 try{const result=await papers.saveAnnotation(id,language,annotations.value.revision,copyAnnotation(item),remove);if(token!==epoch||disposed)return false;annotations.value=result;return true;}
 catch(e){if(token===epoch&&!disposed)annotationError.value=(e as Error).message;return false;}
}
async function commit(label:string,changes:AnnotationChange[],onApplied?:(count:number)=>void){
 if(saving.value||!annotationsReady.value)return false;
 saving.value=true;annotationError.value='';notice.value='';
 const token=epoch,id=props.paper.id,language=props.language,applied:AnnotationChange[]=[];
 try{
  for(const change of changes){
   const item=change.after??change.before!;
   if(!await persist(item,change.after===null,token,id,language))break;
   applied.push(change);onApplied?.(applied.length);
  }
  if(token!==epoch||disposed)return false;
  history.value.record(label,applied);
  if(applied.length!==changes.length)return false;
  notice.value=changes.every(c=>c.after===null)?'标注已删除，可撤销。':(papers.web?'批注已保存。':'批注已保存到本机。');
  if(selectedHighlightId.value&&!selectedHighlight.value)selectedHighlightId.value='';
  return true;
 }finally{saving.value=false;}
}
async function save(item:PaperAnnotation,remove=false){const before=annotations.value.items.find(a=>a.id===item.id)??null;return commit(remove?'删除标注':before?'编辑批注':'添加批注',[{before,after:remove?null:item}]);}
async function travel(direction:AnnotationDirection){
 if(saving.value||!annotationsReady.value)return;
 const label=direction==='undo'?history.value.undoLabel:history.value.redoLabel;if(!label)return;
 saving.value=true;annotationError.value='';notice.value='';const token=epoch,id=props.paper.id,language=props.language;
 try{
  const result=await history.value.travel(direction,(item,remove)=>persist(item,remove,token,id,language));
  if(token!==epoch||disposed)return;
  if(result.changed){clearSelection(true);notice.value=result.complete?`已${direction==='undo'?'撤销':'重做'}${label}。`:`部分标注已${direction==='undo'?'撤销':'重做'}，处理错误提示后可继续。`;}
 }finally{saving.value=false;}
}
function chooseHighlight(event:MouseEvent,n:number){
 if(saving.value||showText.value||(event.target as Element).closest('.pdf-note-pin'))return;
 const native=window.getSelection();if(native&&!native.isCollapsed)return;
 const bounds=sheet(n)?.getBoundingClientRect();if(!bounds)return;
 const x=(event.clientX-bounds.left)/bounds.width,y=(event.clientY-bounds.top)/bounds.height;
 const hit=[...annotations.value.items].reverse().find(item=>item.kind==='highlight'&&item.page===n&&item.rects.some(r=>x>=r[0]&&x<=r[0]+r[2]&&y>=r[1]&&y<=r[1]+r[3]));
 if(hit){clearSelection(true);selectedHighlightId.value=hit.id;}
}
function locateAnnotation(item:PaperAnnotation){jump(item.page);if(item.kind==='highlight')selectedHighlightId.value=item.id;}
function make(group:typeof selection.value[number],kind:PaperAnnotation['kind']):PaperAnnotation{return {id:crypto.randomUUID(),page:group.page,kind,color:color.value,text:'',quote:group.quote.slice(0,5000),rects:group.rects};}
async function highlight(){if(saving.value||!annotationsReady.value)return;selectText();const groups=[...selection.value];if(!groups.length)return;await commit('添加荧光笔',groups.map(group=>({before:null,after:make(group,'highlight')})),count=>{clearSelection(true);selection.value=groups.slice(count);});}
function note(){if(saving.value||!annotationsReady.value)return;selectText();if(!selection.value.length)return;if(selection.value.length>1){notice.value='文字批注请在一页内选择，跨页内容可分别标注。';return;}editing.value=make(selection.value[0],'note');draft.value='';}
function pageNote(){if(!annotationsReady.value||!ratios.value.length)return;editing.value={id:crypto.randomUUID(),page:page.value,kind:'note',color:color.value,text:'',quote:'',rects:[[.94,.025,.025,.025]]};draft.value='';}
function toggleNotes(){sidebar.value=!sidebar.value;if(sidebar.value)nextTick(()=>notesPanel.value?.focus());}
function edit(item:PaperAnnotation){if(saving.value)return;editing.value=copyAnnotation(item);draft.value=item.text;}
async function saveNote(){if(!editing.value)return;const ok=await save({...editing.value,text:draft.value});if(ok){editing.value=undefined;clearSelection(true);}}
async function exportPdf(){exporting.value=true;try{const result=await papers.export(props.paper.id,exportLanguage.value,exportAnnotated.value);notice.value=result.cancelled?'已取消导出。':`已导出 ${result.name}`;if(!result.cancelled)exportOpen.value=false;}catch(e){notice.value=(e as Error).message;}finally{exporting.value=false;}}
watch(pageWidth,()=>{const a=zoomAnchor??anchor();zoomAnchor=undefined;lineStyles.clear();nextTick(()=>{restoreAnchor(a);schedule();});});
watch(renderWidth,refreshRaster);
watch(showText,()=>clearSelection(true));
watch(()=>props.language,()=>{epoch++;clearSelection(true);cache.value={};ratios.value=[];page.value=props.initialPage;annotations.value={revision:0,items:[]};annotationsReady.value=false;failure.value='';notice.value='';refreshRaster();void reloadAnnotations();exportLanguage.value=props.language;});
watch(()=>props.active,active=>{if(active===false){clearSelection(true);if(full.value)void document.exitFullscreen();clearTimeout(renderTimer);renderTimer=undefined;wanted.clear();cancelAnimationFrame(frame);frame=0;}else if(!ratios.value.length){if(!pumping)refreshRaster();}else schedule();});
onMounted(()=>{document.addEventListener('keydown',key);document.addEventListener('fullscreenchange',fullscreenChanged);document.addEventListener('selectionchange',selectionChanged);document.addEventListener('pointerdown',selectionPointerDown,true);window.addEventListener('resize',densityChanged);resize=new ResizeObserver(entries=>{const next=entries[0].contentRect.width;if(next<=0)return;width.value=next;densityChanged();schedule();});if(viewport.value){resize.observe(viewport.value);viewport.value.addEventListener('wheel',wheel,{passive:false});}refreshRaster();void reloadAnnotations();});
onBeforeUnmount(()=>{disposed=true;clearSelection(true);controller?.abort();annotationController?.abort();resize?.disconnect();clearTimeout(renderTimer);cancelAnimationFrame(frame);document.removeEventListener('keydown',key);document.removeEventListener('fullscreenchange',fullscreenChanged);document.removeEventListener('selectionchange',selectionChanged);document.removeEventListener('pointerdown',selectionPointerDown,true);window.removeEventListener('resize',densityChanged);viewport.value?.removeEventListener('wheel',wheel);if(full.value)void document.exitFullscreen();});
useWebSettings(props.settingsScope,{readerZoom:zoom,readerFit:fit,readerText:showText,readerSidebar:sidebar,annotationColor:color,exportLanguage,exportAnnotated});
</script>
<template>
<section ref="reader" class="pdf-reader" aria-label="论文 PDF 阅读器">
 <div class="pdf-toolbar">
  <div class="pdf-controls"><button :disabled="saving" :class="{primary:language==='original'}" @click="emit('language','original')">{{paper.notice?'栏目说明':paper.local?'本地 PDF':'原文'}}</button><button v-if="!paper.local&&!paper.notice" :disabled="saving" :class="{primary:language==='translation'}" @click="emit('language','translation')">中文译文</button><button v-if="paper.analysisPdf&&paper.analysisDownloaded!==false" :class="{primary:language==='analysis'}" :disabled="saving" @click="emit('language','analysis')">论文解读</button></div>
  <div class="pdf-controls"><button aria-label="上一页" :disabled="page<=1" @click="jump(page-1)">‹</button><label>第 <input aria-label="页码" type="number" v-model.lazy.number="pageInput" @change="syncNumberInput($event,pageInput)" min="1" :max="ratios.length"> / {{ratios.length||'—'}} 页</label><button aria-label="下一页" :disabled="page>=ratios.length" @click="jump(page+1)">›</button></div>
  <div class="pdf-controls"><select aria-label="阅读缩放" title="也可按住 Ctrl 在页面上滚动鼠标滚轮" :value="fit?'fit':zoom" @change="changeZoom(($event.target as HTMLSelectElement).value)"><option value="fit">适宽</option><option v-for="z in zoomOptions" :key="z" :value="z">{{z}}%</option></select><button @click="fullscreen">{{full?'退出全屏':'全屏阅读'}}</button></div>
  <div class="pdf-controls"><button :aria-expanded="sidebar" :aria-controls="notesId" title="打开批注列表，可添加本页笔记" @click="toggleNotes">批注 {{annotations.items.length}}</button><button @click="exportOpen=true">导出 PDF</button><button :aria-pressed="showText" @click="showText=!showText">本页文字</button></div>
 </div>
 <div class="pdf-mark-tools" @pointerdown.capture="selectText">
  <span>{{selection.length?`已选择 ${selection.length} 页的文字`:'拖选页面文字可复制或标记'}}</span><button :disabled="!selection.length" @mousedown.prevent @click="copySelection">复制选文</button><select aria-label="荧光笔颜色" v-model="color"><option value="yellow">黄色</option><option value="green">绿色</option><option value="pink">粉色</option></select><button :disabled="!selection.length||saving||!annotationsReady" @mousedown.prevent @click="highlight">荧光笔</button><button :disabled="!selection.length||saving||!annotationsReady" @mousedown.prevent @click="note">添加批注</button>
  <div class="pdf-history-tools"><button aria-label="撤销标注" :title="history.canUndo?`撤销${history.undoLabel}（Ctrl+Z / ⌘Z）`:'撤销标注（Ctrl+Z / ⌘Z）'" :disabled="saving||!annotationsReady||!history.canUndo" @mousedown.prevent @click="travel('undo')">撤销</button><button aria-label="重做标注" :title="history.canRedo?`重做${history.redoLabel}（Ctrl+Shift+Z / ⌘⇧Z）`:'重做标注（Ctrl+Shift+Z / ⌘⇧Z）'" :disabled="saving||!annotationsReady||!history.canRedo" @mousedown.prevent @click="travel('redo')">重做</button></div>
  <template v-if="selectedHighlight"><span class="pdf-selected-mark" role="status">已选中荧光笔</span><button :disabled="saving" title="Delete 或 Backspace，删除后可撤销" @mousedown.prevent @click="save(selectedHighlight,true)">删除荧光笔</button></template>
  <small v-if="saving">正在保存…</small><small v-else-if="notice" role="status">{{notice}}</small>
 </div>
 <div v-if="annotationError" class="pdf-warning" role="alert">{{annotationError}} <button @click="reloadAnnotations">重新载入批注</button></div>
 <div class="pdf-layout" :class="{annotating:sidebar}">
  <div ref="viewport" class="paper-viewport" tabindex="0" aria-label="连续 PDF 页面" @scroll.passive="schedule" @mouseup="selectText" @keyup="selectText">
   <p v-if="failure" class="pdf-warning" role="alert">{{failure}} <button @click="failure='';wanted.add(page);pump()">重试页面</button></p>
   <p v-if="!ratios.length">正在打开论文…</p>
   <pre v-if="showText" class="paper-text">{{activeText}}</pre>
   <div v-for="(ratio,index) in ratios" v-show="!showText" :key="index" class="pdf-page" :data-page="index+1" :style="{width:pageWidth+'px',height:pageWidth*ratio+'px'}" @click="chooseHighlight($event,index+1)">
    <template v-if="cache[index+1]&&Math.abs(index+1-page)<=2"><img :src="cache[index+1].image" class="paper-sheet" :alt="`${languageTitle} 第 ${index+1} 页`" draggable="false" decoding="async">
     <div class="pdf-text-layer"><span v-for="(line,i) in cache[index+1].lines" :key="i" class="pdf-text-line" :style="textStyle(line,ratio)">{{line.text+'\n'}}</span></div>
    </template><span v-else class="pdf-placeholder">第 {{index+1}} 页{{loading===index+1?' · 正在加载…':''}}</span>
    <div class="pdf-selection-layer" aria-hidden="true"><template v-for="group in selection.filter(s=>s.page===index+1)" :key="group.page"><span v-for="(r,i) in group.rects" :key="i" :style="{left:r[0]*100+'%',top:r[1]*100+'%',width:r[2]*100+'%',height:r[3]*100+'%'}"/></template></div>
    <div class="pdf-marks" aria-hidden="true"><template v-for="item in annotations.items.filter(a=>a.page===index+1)" :key="item.id"><span v-for="(r,i) in item.rects" :key="i" :data-annotation-id="item.id" :class="['pdf-mark',item.kind,item.color,{'is-selected':selectedHighlightId===item.id}]" :style="{left:r[0]*100+'%',top:r[1]*100+'%',width:r[2]*100+'%',height:r[3]*100+'%'}"/></template></div>
    <button v-for="item in annotations.items.filter(a=>a.page===index+1&&a.kind==='note')" :key="item.id" class="pdf-note-pin" :style="{left:Math.min(.96,item.rects[0][0]+item.rects[0][2])*100+'%',top:item.rects[0][1]*100+'%'}" :title="item.text" aria-label="编辑此处批注" @click.stop="edit(item)">✎</button>
   </div>
  </div>
  <aside v-if="sidebar" :id="notesId" ref="notesPanel" class="pdf-notes" :aria-label="papers.web?'个人批注':'本机批注'" tabindex="-1"><strong>{{papers.web?'个人批注':'本机批注'}} · {{languageTitle}}</strong><button :disabled="saving||!annotationsReady||!ratios.length" @click="pageNote">添加本页笔记</button><p class="hint">可为整页写笔记，也可拖选文字后添加批注。</p><p v-if="!annotations.items.length" class="hint">还没有批注，先记下一个想法吧。</p>
   <div v-for="item in annotations.items" :key="item.id" :data-annotation-id="item.id" :class="['pdf-note-card','note-'+item.color,{'is-selected':selectedHighlightId===item.id}]">
    <button class="pdf-note-location" :disabled="saving" @click="locateAnnotation(item)">第 {{item.page}} 页 · {{item.kind==='highlight'?'荧光笔':item.quote?'便签':'本页笔记'}}</button><p v-if="item.text">{{item.text}}</p>
    <div class="pdf-note-actions"><button :disabled="saving" @click="edit(item)">编辑</button><button :disabled="saving" @click="save(item,true)">删除</button></div>
   </div>
  </aside>
 </div>
 <AnnotationNoteDialog v-if="editing" v-model="draft" :page="editing.page" :color="editing.color" :saving="saving" :error="annotationError" @close="editing=undefined" @save="saveNote" @reload="reloadAnnotations"/>
 <ModalDialog v-if="exportOpen" title="导出论文 PDF" :close-disabled="exporting" @close="exportOpen=false"><label class="note-label">文档<select v-model="exportLanguage"><option value="original">{{paper.notice?'栏目说明':paper.local?'本地 PDF':'原文'}}</option><option v-if="!paper.local&&!paper.notice" value="translation">中文译文</option><option v-if="paper.analysisPdf&&paper.analysisDownloaded!==false" value="analysis">论文解读</option></select></label><label class="export-check"><input type="checkbox" v-model="exportAnnotated">包含此文档的{{papers.web?'个人批注':'本机批注'}}与荧光笔</label><p class="hint">未勾选时导出原文件，勾选时生成含标准 PDF 批注的副本。注释显示方式取决于外部阅读器。</p><p role="status">{{notice}}</p><template #footer><button :disabled="exporting" @click="exportOpen=false">取消</button><button class="primary" :disabled="exporting||saving" @click="exportPdf">{{exporting?'正在导出…':'选择位置并导出'}}</button></template></ModalDialog>
</section>
</template>
<style scoped>
.pdf-reader{display:flex;flex-direction:column;flex:1;min-height:0;min-width:0;background:var(--app);color:var(--text)}.pdf-reader:fullscreen{width:100vw;height:100vh}.pdf-toolbar,.pdf-mark-tools{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:6px 10px;background:var(--panel);border-bottom:1px solid var(--border)}.pdf-toolbar{justify-content:space-between}.pdf-controls{display:flex;align-items:center;gap:5px}.pdf-controls label{display:flex;align-items:center;gap:4px;white-space:nowrap}.pdf-controls input{width:4.1em}.pdf-toolbar button,.pdf-toolbar select,.pdf-toolbar input,.pdf-mark-tools button,.pdf-mark-tools select{min-height:28px;padding:2px 7px}.pdf-mark-tools{font-size:.85rem;color:var(--muted)}.pdf-mark-tools small{margin-left:auto}.pdf-layout{display:flex;flex:1;min-height:0}.paper-viewport{position:relative;flex:1;min-width:0;overflow:auto;padding:16px;scrollbar-gutter:stable;background:var(--app);outline-offset:-3px;overflow-anchor:none}.pdf-page{position:relative;margin:0 auto 12px;background:white;box-shadow:0 1px 8px var(--border);flex:none;isolation:isolate}.paper-sheet{position:absolute;inset:0;display:block;width:100%;height:100%;pointer-events:none;user-select:none}.pdf-text-layer{position:absolute;inset:0;overflow:hidden;z-index:2;user-select:text;opacity:0}.pdf-selection-layer{position:absolute;inset:0;pointer-events:none;z-index:2}.pdf-selection-layer span{position:absolute;background:#4388e866}.pdf-text-line{position:absolute;display:block;white-space:pre;color:transparent;line-height:1;transform-origin:left top;font-family:'Times New Roman','SimSun',serif;cursor:text}.pdf-text-line::selection{background:#4388e866;color:transparent}.pdf-placeholder{position:absolute;top:40%;width:100%;text-align:center;color:#767676}.pdf-marks{position:absolute;inset:0;pointer-events:none;z-index:3;mix-blend-mode:multiply}.pdf-mark{position:absolute}.pdf-mark.yellow{background:#ffe06680}.pdf-mark.green{background:#8de4a780}.pdf-mark.pink{background:#ffa6ce80}.pdf-mark.note{background:transparent;border-bottom:2px dotted #cf7300}.pdf-note-pin{position:absolute;z-index:4;background:#ffe8a4;color:#734500;border:1px solid #dca643;border-radius:3px;min-height:20px;min-width:20px;padding:0 3px;transform:translateY(-70%)}.pdf-notes{width:260px;min-width:210px;overflow:auto;background:var(--panel);border-left:1px solid var(--border);padding:12px}.pdf-note-card{padding:12px 0;border-bottom:1px solid var(--border);overflow-wrap:anywhere}.pdf-note-card p{white-space:pre-wrap}.pdf-note-card blockquote{font-size:.86rem;max-height:100px;overflow:auto;color:var(--muted);margin:8px 0}.pdf-note-card button{font-size:.86rem;min-height:28px}.paper-text{white-space:pre-wrap;font:inherit;line-height:1.8;background:var(--panel);padding:20px}.pdf-warning{padding:8px 12px;background:var(--selected)}.note-label{display:grid;gap:8px;margin-block:12px}.note-label textarea{width:100%;resize:vertical}.export-check{display:flex;align-items:center;gap:8px}.export-check input{width:16px;height:16px}blockquote{white-space:pre-wrap;max-height:140px;overflow:auto;color:var(--muted)}@media(max-width:950px){.pdf-notes{width:210px;min-width:180px}.pdf-toolbar{justify-content:flex-start}}
</style>
<style scoped>
.pdf-layout{min-width:0}.pdf-notes{flex:0 0 260px;box-sizing:border-box;outline-offset:-3px}
.pdf-notes>button{width:100%;margin-top:10px;font:inherit}.note-label textarea{font:inherit;line-height:1.7}
@media(max-width:950px){.pdf-notes{flex-basis:210px}}
.pdf-history-tools{display:flex;gap:4px;border-left:1px solid var(--border);padding-left:8px;margin-left:2px}.pdf-selected-mark{color:var(--text);font-size:.82rem}.pdf-mark.is-selected{outline:2px solid #985e19;outline-offset:1px}
.pdf-note-card{--note-paper:#fff4c5;--note-rule:#d7bf6d;--note-ink:#514021;margin:12px 0;padding:10px 12px;border:1px solid var(--note-rule);border-radius:3px 12px 3px 3px;background:var(--note-paper);color:var(--note-ink);box-shadow:1px 3px 7px #362a1010}
.pdf-note-card.note-green{--note-paper:#e2f3d9;--note-rule:#a7c79a;--note-ink:#29442b}.pdf-note-card.note-pink{--note-paper:#fbe3e9;--note-rule:#d7b0bd;--note-ink:#643e4d}
.pdf-note-card.is-selected{outline:2px solid var(--accent);outline-offset:2px}.pdf-note-card .pdf-note-location{background:transparent;color:inherit;border:0;padding:0;min-height:24px;font-size:.78rem;opacity:.8;box-shadow:none;text-align:left}.pdf-note-card p{font-size:.94rem;line-height:1.85;margin:8px 0 12px}
.pdf-note-actions{display:flex;justify-content:flex-end;gap:6px;border-top:1px solid var(--note-rule);padding-top:7px;margin-top:7px}.pdf-note-actions button{background:transparent;color:inherit;border-color:transparent;box-shadow:none;padding:1px 5px;font-size:.76rem;min-height:24px}
[data-theme=dark] .pdf-note-card{--note-paper:#393425;--note-rule:#796a3b;--note-ink:#f4e6b8}[data-theme=dark] .pdf-note-card.note-green{--note-paper:#29372b;--note-rule:#5c7857;--note-ink:#dfefd5}[data-theme=dark] .pdf-note-card.note-pink{--note-paper:#3e2e36;--note-rule:#805e70;--note-ink:#f6dce6}
</style>
