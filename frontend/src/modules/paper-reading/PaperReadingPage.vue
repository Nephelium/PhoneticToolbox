<script setup lang="ts">
import {computed,onMounted,onBeforeUnmount,ref,watch} from 'vue';
import PdfReader from './PdfReader.vue';
import {citation,citationStyles,type CitationStyle} from './citations.ts';
import {copyText} from '../../platform/clipboard.ts';
import ModalDialog from '../../components/ModalDialog.vue';
import {papers,papersAvailable,eligible,missing,bytesLabel,type PaperStatus} from '../../platform/papers.ts';
const props=defineProps<{active?:boolean}>();
const emit=defineEmits<{help:[]}>();
const details=ref(false),citeOpen=ref(false),citeStyle=ref<CitationStyle>('GB/T 7714—2015'),copyNotice=ref('');
const state=ref<PaperStatus>({firstLaunch:'',warning:'',papers:[]});const selected=ref('');
const language=ref<'original'|'translation'>('original'),page=ref(1);
const busy=ref(false),error=ref(''),progress=ref('');
const historical=ref(false),since=ref('');
const today=ref(new Date().toLocaleDateString('sv-SE'));
const current=computed(()=>state.value.papers.find(p=>p.id===selected.value));
const visible=computed(()=>state.value.papers.filter(p=>p.publishedAt<=today.value&&(p.downloaded||p.publishedAt>=state.value.firstLaunch)).sort((a,b)=>b.publishedAt.localeCompare(a.publishedAt)||a.title.localeCompare(b.title)));
const dates=computed(()=>[...new Set(visible.value.map(p=>p.publishedAt))]);
const historicalPapers=computed(()=>since.value?eligible(state.value.papers,since.value,today.value):[]);
const toDownload=computed(()=>since.value?missing(state.value.papers,since.value,today.value):[]);
const downloadBytes=computed(()=>toDownload.value.reduce((n,p)=>n+p.original.size+p.translation.size,0));
let downloadController:AbortController|undefined,disposed=false,lastCheck=0;
const readingPositions=new Map<string,number>();
let refreshTimer:ReturnType<typeof setInterval>|undefined;
function apply(value:PaperStatus){state.value=value;if(!selected.value||!visible.value.some(p=>p.id===selected.value))selected.value=visible.value[0]?.id??'';}
const citationText=computed(()=>current.value?citation(current.value,citeStyle.value,today.value):'');
async function copyCitation(){try{await copyText(citationText.value);copyNotice.value='已复制引用。';}catch(e){copyNotice.value=(e as Error).message;}}
watch(selected,()=>{page.value=readingPositions.get(selected.value+':'+language.value)??1;});
function changeLanguage(value:'original'|'translation'){if(language.value===value)return;language.value=value;page.value=readingPositions.get(selected.value+':'+value)??1;}
function remember(value:number){page.value=value;readingPositions.set(selected.value+':'+language.value,value);}
async function download(start:string){
  busy.value=true;error.value='';downloadController=new AbortController();
  try{apply(await papers.download(start,downloadController.signal,p=>{progress.value=`正在获取 ${bytesLabel(p.received)} / ${bytesLabel(p.total)}`;}));}
  catch(e){error.value=(e as Error).message;try{apply(await papers.status());}catch{/* Keep last known directory. */}}
  finally{busy.value=false;progress.value='';}
}
async function refresh(){
  if(busy.value||!papersAvailable())return;today.value=new Date().toLocaleDateString('sv-SE');busy.value=true;error.value='';downloadController=new AbortController();lastCheck=Date.now();
  try{apply(await papers.refresh(downloadController.signal));}
  catch(e){error.value=(e as Error).message;return;}
  finally{busy.value=false;}
  if(!disposed&&missing(state.value.papers,state.value.firstLaunch,today.value).length)await download(state.value.firstLaunch);
}
function history(){since.value=state.value.papers.map(p=>p.publishedAt).sort()[0]??today.value;historical.value=true;}
watch(()=>props.active,active=>{if(active&&Date.now()-lastCheck>15*60*1000)void refresh();});
onMounted(async()=>{if(!papersAvailable())return;refreshTimer=setInterval(()=>{if(props.active!==false)void refresh();},15*60*1000);try{apply(await papers.status());await refresh();}catch(e){error.value=(e as Error).message;}});
onBeforeUnmount(()=>{disposed=true;clearInterval(refreshTimer);downloadController?.abort();});
</script>
<template>
<section class="paper-reading" aria-label="语音学论文精读">
  <header class="paper-heading"><div><h2>语音学论文精读</h2><small>从声音的研究，走近研究的细节</small></div><div class="paper-actions"><button @click="emit('help')">帮助</button><button :disabled="busy||!papersAvailable()" @click="history">获取往期</button><button :disabled="busy||!papersAvailable()" @click="refresh">检查新论文</button></div></header>
  <p v-if="!papersAvailable()" class="paper-notice">请在桌面版打开此模块。论文由服务器单独分发，下载后可离线阅读。</p>
  <div v-if="busy||error||state.warning" class="paper-notice" role="status"><span>{{error||state.warning||progress||'正在检查论文目录…'}}</span><button v-if="busy" @click="downloadController?.abort()">取消</button></div>
  <div class="paper-body">
    <aside class="paper-library" aria-label="按日期浏览论文"><div class="paper-library-summary"><strong>论文目录</strong><small>{{visible.length}} 篇 · 已下载 {{state.papers.filter(p=>p.downloaded).length}} 篇</small></div>
      <div class="paper-date-list"><section v-for="date in dates" :key="date"><h3>{{date}} <small>{{visible.filter(p=>p.publishedAt===date).length}} 篇</small></h3><button v-for="p in visible.filter(p=>p.publishedAt===date)" :key="p.id" class="paper-card" :class="{selected:p.id===selected}" :aria-pressed="p.id===selected" @click="selected=p.id"><span>{{p.titleZh}}</span><small>{{p.downloaded?'已下载':'待获取'}} · {{p.license.id}}</small></button></section><p v-if="!visible.length" class="hint">暂时没有新论文。可以检查更新，或选择获取往期。</p></div>
      <p class="hint paper-subscription" v-if="state.firstLaunch">默认获取 {{state.firstLaunch}} 起发布的论文。<br>日期为本栏目发布日期。</p>
    </aside>
    <article class="paper-reader" v-if="current">
      <div class="paper-summary"><button class="paper-fold" :aria-expanded="details" aria-controls="paper-details" @click="details=!details"><span>{{details?'▾':'▸'}}</span><strong :title="current.titleZh">{{current.titleZh}}</strong><small>{{details?'收起信息':'论文信息'}}</small></button><a :href="current.license.url" target="_blank" rel="noopener noreferrer">{{current.license.id}}</a><button @click="citeOpen=true;copyNotice=''">引用格式</button></div>
      <div v-if="details" id="paper-details" class="paper-details"><div class="paper-metadata"><h2>{{current.titleZh}}</h2><p class="paper-original-title">{{current.title}}</p><small>{{current.authors}} · {{current.version}} · 原稿 {{current.submittedAt}}</small><p class="paper-license"><a :href="current.license.url" target="_blank" rel="noopener noreferrer">许可：{{current.license.id}}</a><a :href="current.sourceUrl" target="_blank" rel="noopener noreferrer">原文与版本</a></p><p class="hint">{{current.translationNote}}</p></div><div class="paper-guide"><strong>阅读导引</strong><p>{{current.guide}}</p></div></div>
      <PdfReader v-if="current.downloaded" :key="current.id" :paper="current" :language="language" :initial-page="page" :active="active" @language="changeLanguage" @position="remember"/>
      <div v-else class="paper-empty"><p>这篇论文还未下载。</p><button class="primary" :disabled="busy" @click="download(current.publishedAt)">获取此日期起的论文</button></div>
    </article>
    <div v-else class="paper-empty"><h2>选一篇论文，慢慢读</h2><p>新论文会在这里出现，往期内容可按需获取。</p></div>
  </div>
  <ModalDialog v-if="citeOpen&&current" title="复制论文引用" @close="citeOpen=false"><label class="paper-citation-label">引用格式<select v-model="citeStyle"><option v-for="style in citationStyles" :key="style" :value="style">{{style}}</option></select></label><textarea class="paper-citation" readonly :value="citationText" rows="7" aria-label="引用文本"/><p class="hint">引用指向原文的固定 arXiv 版本。这里提供纯文本参考文献，粘贴后请按学校或期刊要求设置斜体等排版。GB/T 明确采用 2015 版，ASA 为美国社会学会格式。</p><p role="status">{{copyNotice}}</p><template #footer><button @click="citeOpen=false">关闭</button><button class="primary" @click="copyCitation">复制引用</button></template></ModalDialog>
  <ModalDialog v-if="historical" title="获取往期论文" :close-disabled="busy" @close="historical=false"><p>选择起始日期，获取该日期当日及之后的论文。已下载的完整文件会跳过。</p><label class="paper-history-date">起始日期 <input type="date" v-model="since" :max="today" :disabled="busy"></label><p>共 {{historicalPapers.length}} 篇，待下载 {{toDownload.length}} 篇，约 {{bytesLabel(downloadBytes)}}。</p><p class="hint">这次选择不会改变默认订阅起点 {{state.firstLaunch}}。</p><p v-if="progress||error" role="status">{{error||progress}}</p><template #footer><button v-if="busy" @click="downloadController?.abort()">取消下载</button><button v-else @click="historical=false">关闭</button><button class="primary" :disabled="busy||!since||!toDownload.length" @click="download(since)">获取 {{toDownload.length}} 篇</button></template></ModalDialog>
</section>
</template>
<style scoped>
.paper-reading{height:100%;min-height:0;display:flex;flex-direction:column;background:var(--app)}
.paper-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 16px;border-bottom:1px solid var(--border);background:var(--panel);flex-wrap:wrap}.paper-heading h2{font-size:1.15rem}.paper-actions{display:flex;align-items:center;gap:6px;flex-wrap:wrap}.paper-body{display:grid;grid-template-columns:248px minmax(0,1fr);min-height:0;flex:1}.paper-library{display:flex;flex-direction:column;min-height:0;border-right:1px solid var(--border);background:var(--panel)}.paper-library-summary{padding:14px 12px;display:grid;gap:5px;border-bottom:1px solid var(--border)}.paper-date-list{overflow:auto;flex:1;padding:8px}.paper-date-list h3{display:flex;justify-content:space-between;font-size:.93rem;padding:12px 6px 8px;color:var(--muted)}.paper-card{display:flex;flex-direction:column;align-items:flex-start;text-align:left;width:100%;padding:10px;white-space:normal;background:transparent;border-color:transparent;line-height:1.65;gap:6px}.paper-card.selected{background:var(--selected);border-left:3px solid var(--accent)}.paper-subscription{padding:12px;border-top:1px solid var(--border)}.paper-reader{display:flex;flex-direction:column;min-width:0;min-height:0}.paper-metadata{padding:12px 18px 10px;background:var(--panel)}.paper-original-title{font-size:.93rem;color:var(--muted)}.paper-license{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:6px;font-size:.9rem}.paper-license button{margin-left:auto;min-height:28px;padding:2px 8px}.paper-guide{padding:10px 18px;background:var(--selected);border-top:1px solid var(--border);overflow:auto;font-size:.93rem;flex-shrink:0}.paper-guide p{white-space:pre-line;line-height:1.65}.paper-toolbar{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:8px 12px;flex-wrap:wrap;border-block:1px solid var(--border);background:var(--panel)}.paper-toolbar button,.paper-toolbar select{min-height:30px;padding:3px 8px}.paper-page-label{display:flex;align-items:center;gap:5px;white-space:nowrap}.paper-page-label input{width:4.5em;padding:3px 5px;min-height:30px}.paper-viewport{overflow:auto;min-height:0;flex:1;padding:18px;background:var(--app);text-align:center}.paper-sheet{display:block;margin:0 auto;background:white;box-shadow:0 2px 12px var(--border);max-width:none;height:auto}.paper-sheet.fit{width:100%;max-width:1100px}.paper-text{margin:0;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.8;background:var(--panel);color:var(--text);padding:20px;font:inherit}.paper-empty{display:grid;place-content:center;gap:14px;text-align:center;padding:32px;color:var(--muted)}.paper-notice{display:flex;justify-content:space-between;align-items:center;padding:8px 16px;background:var(--selected);gap:12px}.paper-history-date{display:flex;align-items:center;gap:10px;margin:20px 0}@media(max-width:1000px){.paper-body{grid-template-columns:196px minmax(0,1fr)}.paper-toolbar{justify-content:flex-start}.paper-metadata{padding:10px}.paper-guide{padding:8px 10px}}@media(max-width:680px){.paper-body{grid-template-columns:155px minmax(0,1fr)}.paper-heading small{display:none}.paper-card{padding:6px}.paper-viewport{padding:8px}}
.paper-summary{display:flex;align-items:center;gap:10px;padding:6px 10px;background:var(--panel);border-bottom:1px solid var(--border);flex-shrink:0}.paper-summary>a{font-size:.85rem;white-space:nowrap}.paper-summary>button{min-height:30px}.paper-fold{display:flex;align-items:center;gap:8px;flex:1;min-width:0;text-align:left;background:transparent;border-color:transparent}.paper-fold strong{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;font-size:.95rem}.paper-fold small{white-space:nowrap}.paper-details{max-height:40%;overflow:auto;flex-shrink:0}.paper-citation-label{display:flex;gap:12px;align-items:center;margin-bottom:12px}.paper-citation{width:100%;resize:vertical;line-height:1.6}
</style>
