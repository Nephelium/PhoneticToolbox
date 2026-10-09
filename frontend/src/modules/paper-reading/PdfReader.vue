<script setup lang="ts">
import {computed,nextTick,onBeforeUnmount,onMounted,ref,watch} from 'vue';
import ModalDialog from '../../components/ModalDialog.vue';
import {papers,type Paper,type PaperLanguage,type PaperPage,type PaperLine,type PaperAnnotation,type PaperAnnotations} from '../../platform/papers.ts';
import {copyText} from '../../platform/clipboard.ts';
const props=defineProps<{paper:Paper;language:PaperLanguage;initialPage:number;active?:boolean}>();
const emit=defineEmits<{language:[PaperLanguage];position:[number];fullscreen:[boolean]}>();
const reader=ref<HTMLElement>(),viewport=ref<HTMLElement>();
const page=ref(props.initialPage),ratios=ref<number[]>([]),cache=ref<Record<number,PaperPage>>({}),loading=ref(0),failure=ref(''),notice=ref('');
const zoom=ref(100),fit=ref(true),width=ref(800),full=ref(false),showText=ref(false),sidebar=ref(false),color=ref<PaperAnnotation['color']>('yellow');
const annotations=ref<PaperAnnotations>({revision:0,items:[]}),saving=ref(false),annotationError=ref(''),annotationsReady=ref(false);
const selection=ref<{page:number;quote:string;rects:number[][]}[]>([]),editing=ref<PaperAnnotation>(),draft=ref(''),confirmDelete=ref('');
const exporting=ref(false),exportOpen=ref(false),exportLanguage=ref<PaperLanguage>(props.language),exportAnnotated=ref(false);
const pageWidth=computed(()=>fit.value?Math.max(280,Math.min(1100,width.value-32)):816*zoom.value/100);
const activeText=computed(()=>cache.value[page.value]?.text??'当前页正在加载。');
let disposed=false,controller:AbortController|undefined,resize:ResizeObserver|undefined,pumping=false,epoch=0;
let frame=0;const wanted=new Set<number>();
const canvas=document.createElement('canvas'),measure=canvas.getContext('2d')!;
function textStyle(line:PaperLine,ratio:number){const height=line.h*pageWidth.value*ratio;measure.font=`${height*1.12}px "Times New Roman", "SimSun", serif`;const natural=measure.measureText(line.text).width||1;return {left:`${line.x*100}%`,top:`${line.y*100}%`,fontSize:`${height*1.12}px`,height:`${height}px`,transform:`scaleX(${line.w*pageWidth.value/natural})`};}
function sheet(n:number){return viewport.value?.querySelector<HTMLElement>(`[data-page="${n}"]`);}
async function pump(){
 if(pumping||disposed)return;pumping=true;
 try{while(wanted.size&&!disposed){const n=[...wanted].sort((a,b)=>Math.abs(a-page.value)-Math.abs(b-page.value))[0];wanted.delete(n);if(cache.value[n])continue;
  controller=new AbortController();loading.value=n;const token=epoch;
  try{const result=await papers.render(props.paper.id,props.language,n,Math.min(2200,Math.max(1200,Math.round(pageWidth.value*1.5))),controller.signal);if(disposed)return;
   if(token!==epoch)continue;cache.value[n]=result;if(!ratios.value.length){ratios.value=result.ratios;await nextTick();jump(props.initialPage);}
   for(const key of Object.keys(cache.value).map(Number))if(Math.abs(key-page.value)>3)delete cache.value[key];
  }catch(e){if(token!==epoch)continue;if(!disposed)failure.value=(e as Error).message;break;}
 }}finally{pumping=false;loading.value=0;}
}
function schedule(){if(frame)return;frame=requestAnimationFrame(()=>{frame=0;scan();});}
function scan(){
 if(!viewport.value||!ratios.value.length)return;
 const view=viewport.value.getBoundingClientRect();let closest=page.value,best=Infinity;
 for(const el of viewport.value.querySelectorAll<HTMLElement>('[data-page]')){const r=el.getBoundingClientRect(),n=Number(el.dataset.page);const distance=Math.abs(r.top-view.top-8);
  if(r.bottom>view.top+16&&r.top<view.bottom&&distance<best){best=distance;closest=n;}
 }
 page.value=closest;emit('position',closest);wanted.clear();
 for(let n=Math.max(1,closest-1);n<=Math.min(ratios.value.length,closest+2);n++)if(!cache.value[n])wanted.add(n);
 void pump();
}
function jump(n:number){if(!Number.isInteger(n)||n<1||n>ratios.value.length)return;page.value=n;emit('position',n);const el=sheet(n);if(el&&viewport.value)viewport.value.scrollTop=el.offsetTop-16;schedule();}
function key(event:KeyboardEvent){if(props.active===false||event.defaultPrevented||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||document.querySelector('dialog[open]')||(event.target as HTMLElement)?.closest('input,textarea,select,[contenteditable=true]'))return;
 if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();jump(page.value+(event.key==='ArrowLeft'?-1:1));}
 if(event.key==='Escape'&&full.value){event.preventDefault();void document.exitFullscreen();}
}
async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else await reader.value?.requestFullscreen();}catch(e){notice.value='无法进入全屏：'+(e as Error).message;}}
function fullscreenChanged(){full.value=document.fullscreenElement===reader.value;emit('fullscreen',full.value);nextTick(schedule);}
function selectText(){
 const s=window.getSelection();if(!s||s.isCollapsed||!s.rangeCount){selection.value=[];return;}const range=s.getRangeAt(0);
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
 if(groups.length)selection.value=groups;
}
async function copySelection(){try{await copyText(selection.value.map(x=>x.quote).join('\n\n'));notice.value='已复制所选文字。';}catch(e){notice.value=(e as Error).message;}}
async function reloadAnnotations(){const token=epoch;try{const value=await papers.annotations(props.paper.id,props.language);if(token!==epoch||disposed)return;annotations.value=value;annotationError.value='';annotationsReady.value=true;}catch(e){if(token!==epoch||disposed)return;annotationError.value=(e as Error).message;annotationsReady.value=false;}}
async function save(item:PaperAnnotation,remove=false){saving.value=true;annotationError.value='';try{annotations.value=await papers.saveAnnotation(props.paper.id,props.language,annotations.value.revision,item,remove);notice.value=remove?'批注已删除。':'批注已保存到本机。';return true;}catch(e){annotationError.value=(e as Error).message;return false;}finally{saving.value=false;}}
function make(group:typeof selection.value[number],kind:PaperAnnotation['kind']):PaperAnnotation{return {id:crypto.randomUUID(),page:group.page,kind,color:color.value,text:'',quote:group.quote.slice(0,5000),rects:group.rects};}
async function highlight(){if(!selection.value.length)return;for(const group of selection.value)if(!await save(make(group,'highlight')))return;selection.value=[];window.getSelection()?.removeAllRanges();}
function note(){if(!selection.value.length)return;if(selection.value.length>1){notice.value='文字批注请在一页内选择，跨页内容可分别标注。';return;}editing.value=make(selection.value[0],'note');draft.value='';}
function edit(item:PaperAnnotation){editing.value={...item};draft.value=item.text;}
async function saveNote(){if(!editing.value)return;const ok=await save({...editing.value,text:draft.value});if(ok){editing.value=undefined;selection.value=[];window.getSelection()?.removeAllRanges();}}
async function exportPdf(){exporting.value=true;try{const result=await papers.export(props.paper.id,exportLanguage.value,exportAnnotated.value);notice.value=result.cancelled?'已取消导出。':`已导出 ${result.name}`;if(!result.cancelled)exportOpen.value=false;}catch(e){notice.value=(e as Error).message;}finally{exporting.value=false;}}
watch(pageWidth,()=>nextTick(()=>{jump(page.value);schedule();}));
watch(()=>props.language,()=>{epoch++;controller?.abort();cache.value={};ratios.value=[];page.value=props.initialPage;selection.value=[];annotations.value={revision:0,items:[]};annotationsReady.value=false;failure.value='';notice.value='';wanted.clear();wanted.add(1);void pump();void reloadAnnotations();exportLanguage.value=props.language;});
watch(()=>props.active,active=>{if(!active&&full.value)void document.exitFullscreen();});
onMounted(()=>{document.addEventListener('keydown',key);document.addEventListener('fullscreenchange',fullscreenChanged);resize=new ResizeObserver(entries=>{width.value=entries[0].contentRect.width;schedule();});if(viewport.value)resize.observe(viewport.value);wanted.add(1);void pump();void reloadAnnotations();});
onBeforeUnmount(()=>{disposed=true;controller?.abort();resize?.disconnect();cancelAnimationFrame(frame);document.removeEventListener('keydown',key);document.removeEventListener('fullscreenchange',fullscreenChanged);if(full.value)void document.exitFullscreen();});
</script>
<template>
<section ref="reader" class="pdf-reader" aria-label="论文 PDF 阅读器">
 <div class="pdf-toolbar">
  <div class="pdf-controls"><button :disabled="saving" :class="{primary:language==='original'}" @click="emit('language','original')">原文</button><button :disabled="saving" :class="{primary:language==='translation'}" @click="emit('language','translation')">中文译文</button></div>
  <div class="pdf-controls"><button aria-label="上一页" :disabled="page<=1" @click="jump(page-1)">‹</button><label>第 <input aria-label="页码" type="number" :value="page" min="1" :max="ratios.length" @change="jump(Number(($event.target as HTMLInputElement).value))"> / {{ratios.length||'—'}} 页</label><button aria-label="下一页" :disabled="page>=ratios.length" @click="jump(page+1)">›</button></div>
  <div class="pdf-controls"><button :aria-pressed="fit" @click="fit=true">适宽</button><select aria-label="阅读缩放" :value="fit?'fit':zoom" @change="fit=false;zoom=Number(($event.target as HTMLSelectElement).value)"><option disabled value="fit">适宽</option><option v-for="z in [75,100,125,150,200]" :key="z" :value="z">{{z}}%</option></select><button @click="fullscreen">{{full?'退出全屏':'全屏阅读'}}</button></div>
  <div class="pdf-controls"><button :aria-pressed="sidebar" @click="sidebar=!sidebar">批注 {{annotations.items.length}}</button><button @click="exportOpen=true">导出 PDF</button><button :aria-pressed="showText" @click="showText=!showText">本页文字</button></div>
 </div>
 <div class="pdf-mark-tools"><span>{{selection.length?`已选择 ${selection.length} 页的文字`:'拖选页面文字可复制或标记'}}</span><button :disabled="!selection.length" @mousedown.prevent @click="copySelection">复制选文</button><select aria-label="荧光笔颜色" v-model="color"><option value="yellow">黄色</option><option value="green">绿色</option><option value="pink">粉色</option></select><button :disabled="!selection.length||saving||!annotationsReady" @mousedown.prevent @click="highlight">荧光笔</button><button :disabled="!selection.length||saving||!annotationsReady" @mousedown.prevent @click="note">添加批注</button><small v-if="saving">正在保存…</small><small v-else-if="notice" role="status">{{notice}}</small></div>
 <div v-if="annotationError" class="pdf-warning" role="alert">{{annotationError}} <button @click="reloadAnnotations">重新载入批注</button></div>
 <div class="pdf-layout" :class="{annotating:sidebar}">
  <div ref="viewport" class="paper-viewport" tabindex="0" aria-label="连续 PDF 页面" @scroll.passive="schedule" @mouseup="selectText" @keyup="selectText">
   <p v-if="failure" class="pdf-warning" role="alert">{{failure}} <button @click="failure='';wanted.add(page);pump()">重试页面</button></p>
   <p v-if="!ratios.length">正在打开论文…</p>
   <pre v-if="showText" class="paper-text">{{activeText}}</pre>
   <div v-for="(ratio,index) in ratios" v-show="!showText" :key="index" class="pdf-page" :data-page="index+1" :style="{width:pageWidth+'px',height:pageWidth*ratio+'px'}">
    <template v-if="cache[index+1]"><img :src="cache[index+1].image" class="paper-sheet" :alt="`${language==='original'?'原文':'中文译文'} 第 ${index+1} 页`" draggable="false">
     <div class="pdf-text-layer"><span v-for="(line,i) in cache[index+1].lines" :key="i" class="pdf-text-line" :style="textStyle(line,ratio)">{{line.text+'\n'}}</span></div>
    </template><span v-else class="pdf-placeholder">第 {{index+1}} 页{{loading===index+1?' · 正在加载…':''}}</span>
    <div class="pdf-marks" aria-hidden="true"><template v-for="item in annotations.items.filter(a=>a.page===index+1)" :key="item.id"><span v-for="(r,i) in item.rects" :key="i" :class="['pdf-mark',item.kind,item.color]" :style="{left:r[0]*100+'%',top:r[1]*100+'%',width:r[2]*100+'%',height:r[3]*100+'%'}"/></template></div>
    <button v-for="item in annotations.items.filter(a=>a.page===index+1&&a.kind==='note')" :key="item.id" class="pdf-note-pin" :style="{left:Math.min(.96,item.rects[0][0]+item.rects[0][2])*100+'%',top:item.rects[0][1]*100+'%'}" :title="item.text" aria-label="编辑此处批注" @click="edit(item)">✎</button>
   </div>
  </div>
  <aside v-if="sidebar" class="pdf-notes" aria-label="本机批注"><strong>本机批注 · {{language==='original'?'原文':'中文译文'}}</strong><p class="hint">选中文字后添加。不同版本与译文分别保存。</p><p v-if="!annotations.items.length" class="hint">还没有批注。</p><div v-for="item in annotations.items" :key="item.id" class="pdf-note-card"><button @click="jump(item.page)">第 {{item.page}} 页 · {{item.kind==='highlight'?'荧光笔':'批注'}}</button><blockquote>{{item.quote}}</blockquote><p>{{item.text}}</p><div class="pdf-controls"><button :disabled="saving" @click="edit(item)">编辑</button><button :disabled="saving" @click="confirmDelete===item.id?(save(item,true),confirmDelete=''):confirmDelete=item.id">{{confirmDelete===item.id?'确认删除':'删除'}}</button></div></div></aside>
 </div>
 <ModalDialog v-if="editing" title="编辑批注" :close-disabled="saving" @close="editing=undefined"><blockquote>{{editing.quote}}</blockquote><label class="note-label">批注内容<textarea v-model="draft" aria-label="批注内容" rows="6" maxlength="5000" placeholder="记下疑问、解释或想法…"/></label><p v-if="annotationError" role="alert">{{annotationError}}<button @click="reloadAnnotations">重新载入批注</button></p><template #footer><button :disabled="saving" @click="editing=undefined">取消</button><button class="primary" :disabled="saving||!draft.trim()" @click="saveNote">保存批注</button></template></ModalDialog>
 <ModalDialog v-if="exportOpen" title="导出论文 PDF" :close-disabled="exporting" @close="exportOpen=false"><label class="note-label">文档<select v-model="exportLanguage"><option value="original">原文</option><option value="translation">中文译文</option></select></label><label class="export-check"><input type="checkbox" v-model="exportAnnotated">包含此文档的本机批注与荧光笔</label><p class="hint">未勾选时导出下载的原文件，勾选时生成含标准 PDF 批注的副本。注释显示方式取决于外部阅读器。</p><p role="status">{{notice}}</p><template #footer><button :disabled="exporting" @click="exportOpen=false">取消</button><button class="primary" :disabled="exporting||saving" @click="exportPdf">{{exporting?'正在导出…':'选择位置并导出'}}</button></template></ModalDialog>
</section>
</template>
<style scoped>
.pdf-reader{display:flex;flex-direction:column;flex:1;min-height:0;min-width:0;background:var(--app);color:var(--text)}.pdf-reader:fullscreen{width:100vw;height:100vh}.pdf-toolbar,.pdf-mark-tools{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:6px 10px;background:var(--panel);border-bottom:1px solid var(--border)}.pdf-toolbar{justify-content:space-between}.pdf-controls{display:flex;align-items:center;gap:5px}.pdf-controls label{display:flex;align-items:center;gap:4px;white-space:nowrap}.pdf-controls input{width:4.1em}.pdf-toolbar button,.pdf-toolbar select,.pdf-toolbar input,.pdf-mark-tools button,.pdf-mark-tools select{min-height:28px;padding:2px 7px}.pdf-mark-tools{font-size:.85rem;color:var(--muted)}.pdf-mark-tools small{margin-left:auto}.pdf-layout{display:flex;flex:1;min-height:0}.paper-viewport{position:relative;flex:1;min-width:0;overflow:auto;padding:16px;scrollbar-gutter:stable;background:var(--app);outline-offset:-3px;overflow-anchor:none}.pdf-page{position:relative;margin:0 auto 12px;background:white;box-shadow:0 1px 8px var(--border);flex:none;isolation:isolate}.paper-sheet{position:absolute;inset:0;display:block;width:100%;height:100%;pointer-events:none;user-select:none}.pdf-text-layer{position:absolute;inset:0;overflow:hidden;z-index:2;user-select:text}.pdf-text-line{position:absolute;display:block;white-space:pre;color:transparent;line-height:1;transform-origin:left top;font-family:'Times New Roman','SimSun',serif;cursor:text}.pdf-text-line::selection{background:#4388e866;color:transparent}.pdf-placeholder{position:absolute;top:40%;width:100%;text-align:center;color:#767676}.pdf-marks{position:absolute;inset:0;pointer-events:none;z-index:3;mix-blend-mode:multiply}.pdf-mark{position:absolute}.pdf-mark.yellow{background:#ffe06680}.pdf-mark.green{background:#8de4a780}.pdf-mark.pink{background:#ffa6ce80}.pdf-mark.note{background:transparent;border-bottom:2px dotted #cf7300}.pdf-note-pin{position:absolute;z-index:4;background:#ffe8a4;color:#734500;border:1px solid #dca643;border-radius:3px;min-height:20px;min-width:20px;padding:0 3px;transform:translateY(-70%)}.pdf-notes{width:260px;min-width:210px;overflow:auto;background:var(--panel);border-left:1px solid var(--border);padding:12px}.pdf-note-card{padding:12px 0;border-bottom:1px solid var(--border);overflow-wrap:anywhere}.pdf-note-card p{white-space:pre-wrap}.pdf-note-card blockquote{font-size:.86rem;max-height:100px;overflow:auto;color:var(--muted);margin:8px 0}.pdf-note-card button{font-size:.86rem;min-height:28px}.paper-text{white-space:pre-wrap;font:inherit;line-height:1.8;background:var(--panel);padding:20px}.pdf-warning{padding:8px 12px;background:var(--selected)}.note-label{display:grid;gap:8px;margin-block:12px}.note-label textarea{width:100%;resize:vertical}.export-check{display:flex;align-items:center;gap:8px}.export-check input{width:16px;height:16px}blockquote{white-space:pre-wrap;max-height:140px;overflow:auto;color:var(--muted)}@media(max-width:950px){.pdf-notes{width:210px;min-width:180px}.pdf-toolbar{justify-content:flex-start}}
</style>
