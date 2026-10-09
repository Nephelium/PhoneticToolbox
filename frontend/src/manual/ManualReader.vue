<script setup lang="ts">
import {computed,nextTick,onBeforeUnmount,onDeactivated,onMounted,ref,shallowRef,watch} from 'vue';
import type {AssetResolver,ChapterLoader,ManualChapter,ManualLocation,ManualProject,ManualTarget} from './types.ts';
import {chapterSearchEntries,chapterSections,fetchChapter,manualAnchor,searchManual} from './content.ts';
import {headingAtReadingLine,manualReadingPath,manualTocSections} from './navigation.ts';
import {ManualChapterRepository} from './loader.ts';
import {manualPlayback} from './playback.ts';
import ManualDocument from './ManualDocument.vue';
const props=withDefaults(defineProps<{project:ManualProject;target?:ManualTarget;requestKey?:number;chapterLoader?:ChapterLoader;assetBaseUrl?:string;assetResolver?:AssetResolver;playbackAllowed?:boolean;active?:boolean;returnLabel?:string;initialLocation?:ManualLocation}>(),{assetBaseUrl:'./manual/',playbackAllowed:true,active:true});
const emit=defineEmits<{navigate:[target:ManualTarget];location:[location:ManualLocation];returnTool:[]}>();
const root=ref<HTMLElement>(),navigation=ref<HTMLElement>(),scroll=ref<HTMLElement>(),query=ref(''),drawer=ref(false),loading=ref(false),error=ref(''),notice=ref(''),currentId=ref(''),targetId=ref<string>();
const chapter=shallowRef<ManualChapter>(),cacheRevision=ref(0);
let request=0,abort:AbortController|undefined,observer:ResizeObserver|undefined,frame:number|undefined,jumpTop:number|undefined;
let repository=new ManualChapterRepository((descriptor,signal)=>props.chapterLoader?props.chapterLoader(descriptor,signal):fetchChapter(props.assetBaseUrl,descriptor,signal));
const positions=new Map<string,number>();
const currentIndex=computed(()=>props.project.chapters.findIndex(item=>item.id===currentId.value));
const currentDescriptor=computed(()=>props.project.chapters[currentIndex.value]);
const sections=computed(()=>chapter.value?chapterSections(chapter.value):[]);
const readingPath=computed(()=>chapter.value?manualReadingPath(chapter.value,targetId.value):{});
const numberedToc=computed(()=>props.project.chapters.map((item,index)=>{
  const available=item.id===currentId.value&&chapter.value?sections.value:item.sections??[];
  return {item,number:index+1,sections:manualTocSections(available,index+1)};
}));
const index=computed(()=>{
  void cacheRevision.value;
  if(props.project.searchIndex?.length)return props.project.searchIndex;
  return props.project.chapters.flatMap(item=>[
    {chapterId:item.id,text:item.title,title:item.title},
    ...(item.sections??[]).map(section=>({chapterId:item.id,targetId:section.id,text:section.title,title:item.title})),
    ...(repository.get(item.id)?chapterSearchEntries(repository.get(item.id)!):[])
  ]);
});
const hits=computed(()=>searchManual(index.value,query.value));
const searching=computed(()=>!!query.value.trim());
function remember(){if(scroll.value&&currentId.value){positions.set(currentId.value,scroll.value.scrollTop);emit('location',{chapterId:currentId.value,targetId:targetId.value,scrollTop:scroll.value.scrollTop});}}
function targetNode(id:string){return scroll.value?.querySelector<HTMLElement>('[id="'+manualAnchor(currentId.value,id)+'"]');}
function headingInView(){
  const container=scroll.value;
  if(!container||!chapter.value||loading.value||!props.active)return;
  // A clamped jump near the chapter end must still select the explicitly clicked entry.
  if(jumpTop!==undefined&&Math.abs(container.scrollTop-jumpTop)<1)return;
  jumpTop=undefined;
  const top=container.getBoundingClientRect().top;
  const headings=sections.value.flatMap(section=>{const node=targetNode(section.id);return node?[{id:section.id,top:node.getBoundingClientRect().top-top}]:[];});
  targetId.value=headingAtReadingLine(headings,{viewportHeight:container.clientHeight,scrollTop:container.scrollTop,scrollHeight:container.scrollHeight});
  remember();
}
function onScroll(){if(frame!==undefined)return;const generation=request;frame=requestAnimationFrame(()=>{frame=undefined;if(generation===request)headingInView();});}
function stopTracking(){observer?.disconnect();if(frame!==undefined)cancelAnimationFrame(frame);frame=undefined;}
function observe(){
  observer?.disconnect();if(typeof ResizeObserver==='undefined'||!scroll.value)return;
  observer=new ResizeObserver(onScroll);observer.observe(scroll.value);
  const document=scroll.value.querySelector('.manual-document');if(document)observer.observe(document);
}
async function revealCurrent(){
  await nextTick();const nav=navigation.value;if(!nav||searching.value)return;
  const node=nav.querySelector<HTMLElement>('[aria-current="location"]')??nav.querySelector<HTMLElement>('[aria-current="page"]');
  if(!node)return;
  const bounds=nav.getBoundingClientRect(),item=node.getBoundingClientRect();
  if(item.top<bounds.top+8)nav.scrollTop+=item.top-bounds.top-8;
  else if(item.bottom>bounds.top+nav.clientHeight-8)nav.scrollTop+=item.bottom-bounds.top-nav.clientHeight+8;
}
async function locate(target:ManualTarget,restore=false,generation=request){
  await nextTick();if(generation!==request||!scroll.value||!chapter.value||chapter.value.id!==target.chapterId)return;
  const container=scroll.value;
  const node=target.targetId?targetNode(target.targetId):undefined;
  jumpTop=undefined;
  if(restore){container.scrollTop=positions.get(target.chapterId)??0;headingInView();}
  else if(target.targetId&&!node){notice.value='目标小节已调整，当前显示所在章节。';container.scrollTop=0;targetId.value=undefined;}
  else if(node){container.scrollTop+=node.getBoundingClientRect().top-container.getBoundingClientRect().top-20;targetId.value=target.targetId;node.classList.add('manual-target-highlight');setTimeout(()=>node.classList.remove('manual-target-highlight'),1600);}
  else {container.scrollTop=0;targetId.value=undefined;}
  if(!restore)jumpTop=container.scrollTop;
  observe();remember();
}
async function navigate(target:ManualTarget,notify=true,restore=false){
  const descriptor=props.project.chapters.find(item=>item.id===target.chapterId);
  if(!descriptor){notice.value='此章节尚未收录，请从目录选择已有章节。';return;}
  remember();manualPlayback.pause();abort?.abort();abort=new AbortController();const generation=++request;
  loading.value=true;error.value='';notice.value='';drawer.value=false;stopTracking();jumpTop=undefined;
  try{
    const loaded=await repository.load(descriptor,abort.signal);
    if(generation!==request)return;
    chapter.value=loaded;currentId.value=descriptor.id;targetId.value=target.targetId;loading.value=false;cacheRevision.value++;
    if(notify)emit('navigate',target);
    await locate(target,restore,generation);
  }catch(reason){if(generation!==request)return;loading.value=false;if(reason instanceof DOMException&&reason.name==='AbortError')return;chapter.value=undefined;currentId.value=descriptor.id;error.value=reason instanceof Error&&/^(章节|正文)/.test(reason.message)?reason.message:'章节加载失败，请检查说明书资源后重试。';}
}
function previous(){const item=props.project.chapters[currentIndex.value-1];if(item)void navigate({chapterId:item.id});}
function next(){const item=props.project.chapters[currentIndex.value+1];if(item)void navigate({chapterId:item.id});}
watch(()=>[props.target?.chapterId,props.target?.targetId,props.requestKey] as const,()=>{if(props.target)void navigate(props.target,false);});
watch(()=>[currentId.value,readingPath.value.sectionId,readingPath.value.subsectionId,searching.value,drawer.value],()=>{void revealCurrent();});
watch(()=>props.active,active=>{if(!active){manualPlayback.pause();remember();stopTracking();}else if(chapter.value)void locate({chapterId:currentId.value},true);});
watch(()=>[props.project.id,props.project.version,props.project.chapters.map(item=>item.id+':'+item.path).join('|')] as const,()=>{repository.clear();const target=props.target??{chapterId:props.project.chapters[0]?.id};if(target.chapterId)void navigate(target,false);});
onMounted(()=>{const location=props.initialLocation;if(location)positions.set(location.chapterId,location.scrollTop);const target=props.target??location??{chapterId:props.project.chapters[0]?.id};if(target.chapterId)void navigate(target,false,!!location&&!props.target);});
onBeforeUnmount(()=>{remember();request++;abort?.abort();stopTracking();manualPlayback.pause();});
onDeactivated(()=>{remember();stopTracking();manualPlayback.pause();});
defineExpose({navigate,pause:()=>manualPlayback.pause()});
</script>
<template>
  <section ref="root" class="manual-reader" aria-label="使用说明" :class="{'manual-drawer-open':drawer}">
    <nav ref="navigation" class="manual-navigation" aria-label="说明书目录">
      <header><strong>{{project.title}}</strong><button class="manual-drawer-close" type="button" @click="drawer=false">收起目录</button></header>
      <label class="manual-search-label">搜索说明书<input v-model="query" type="search" placeholder="输入控件、参数或关键词" aria-label="搜索说明书"/></label>
      <div v-if="searching" class="manual-search-results" role="region" aria-label="搜索结果"><p class="manual-search-count">{{hits.length}} 条结果<span v-if="hits.length>=80">，请缩小搜索范围</span></p><p v-if="!project.searchIndex?.length" class="manual-index-warning">当前搜索范围包含目录及已打开的章节。</p><button v-for="(hit,index) in hits" :key="index" type="button" @click="navigate(hit)"><strong>{{hit.title||project.chapters.find(item=>item.id===hit.chapterId)?.title}}</strong><span>{{hit.excerpt}}</span></button><p v-if="!hits.length" class="manual-empty">未找到匹配内容。</p></div>
      <ol v-else class="manual-toc">
        <li v-for="entry in numberedToc" :key="entry.item.id">
          <button type="button" :data-manual-chapter="entry.item.id" :class="{active:entry.item.id===currentId}" :aria-current="entry.item.id===currentId?'page':undefined" @click="navigate({chapterId:entry.item.id})"><span class="manual-toc-number">{{entry.number}}</span>{{entry.item.title}}</button>
          <ol v-if="entry.sections.length">
            <li v-for="section in entry.sections" :key="section.id">
              <button type="button" :data-manual-section="section.id" :class="{active:entry.item.id===currentId&&section.id===readingPath.sectionId}" :aria-current="entry.item.id===currentId&&section.id===readingPath.sectionId&&!readingPath.subsectionId?'location':undefined" :aria-expanded="section.children.length?entry.item.id===currentId&&section.id===readingPath.sectionId:undefined" :aria-controls="section.children.length?manualAnchor(entry.item.id,section.id)+'-toc':undefined" @click="navigate({chapterId:entry.item.id,targetId:section.id})"><span class="manual-toc-number">{{section.number}}</span>{{section.title}}</button>
              <ol v-if="section.children.length&&entry.item.id===currentId&&section.id===readingPath.sectionId" :id="manualAnchor(entry.item.id,section.id)+'-toc'" class="manual-toc-subsections">
                <li v-for="subsection in section.children" :key="subsection.id"><button type="button" :data-manual-subsection="subsection.id" :class="{active:subsection.id===readingPath.subsectionId}" :aria-current="subsection.id===readingPath.subsectionId?'location':undefined" @click="navigate({chapterId:entry.item.id,targetId:subsection.id})"><span class="manual-toc-number">{{subsection.number}}</span>{{subsection.title}}</button></li>
              </ol>
            </li>
          </ol>
        </li>
      </ol>
      <p v-if="project.version" class="manual-version">手册 {{project.version}}<span v-if="project.softwareVersion"> · 适用软件 {{project.softwareVersion}}</span></p>
    </nav>
    <button v-if="drawer" class="manual-drawer-shade" aria-label="收起目录" type="button" @click="drawer=false"/>
    <div class="manual-reading-pane">
      <header class="manual-reading-toolbar"><button type="button" class="manual-drawer-toggle" @click="drawer=!drawer">目录与搜索</button><span class="manual-breadcrumb">{{currentDescriptor?.title||'使用说明'}}</span><button v-if="returnLabel" type="button" @click="manualPlayback.pause();emit('returnTool')">{{returnLabel}}</button></header>
      <div ref="scroll" class="manual-reading-scroll" tabindex="0" aria-label="说明书正文" @scroll.passive="onScroll">
        <p v-if="notice" class="manual-notice" role="status">{{notice}}</p>
        <p v-if="loading" class="manual-loading" role="status">正在加载章节…</p>
        <div v-else-if="error" class="manual-load-error" role="alert"><p>{{error}}</p><button type="button" @click="navigate({chapterId:currentId})">重新加载</button></div>
        <ManualDocument v-else-if="chapter" :chapter="chapter" :chapter-number="currentIndex+1" :assets="project.assets" :references="project.references" :asset-base-url="assetBaseUrl" :asset-resolver="assetResolver" :playback-allowed="playbackAllowed&&active" @navigate="navigate($event)"/>
        <p v-else class="manual-empty">说明书尚未收录章节。</p>
        <footer v-if="chapter&&!loading" class="manual-chapter-pagination"><button type="button" :disabled="currentIndex<=0" @click="previous">上一章</button><span>{{currentIndex+1}} / {{project.chapters.length}}</span><button type="button" :disabled="currentIndex>=project.chapters.length-1" @click="next">下一章</button></footer>
      </div>
    </div>
  </section>
</template>
