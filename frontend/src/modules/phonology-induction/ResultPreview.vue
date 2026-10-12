<script setup lang="ts">
import {computed,ref,watch,nextTick,onUnmounted} from 'vue';
import {quoteFamily} from '../../design/fonts.ts';
import type {Preview,Settings} from './port.ts';
import {entries,groupEntries,orderedEntries,searchEntries,pairKey,label,summaryEntries} from './state.ts';
const props=defineProps<{preview:Preview;settings:Settings;generated:boolean;stale:boolean;font?:{zh:string;latin:string;size_px:number}}>();
const root=ref<HTMLElement>(),mode=ref<'initial'|'final'|'matrix'>('initial');
const query=ref(''),scope=ref<'all'|'character'|'ipa'>('all'),hit=ref(0);
const wordPage=ref(0),rowPage=ref(0),colPage=ref(0),review=ref(false),reviewPage=ref(0);
const cellPages=ref<Record<string,number>>({});
const paperStyle=computed(()=>props.font?{'--word-font':quoteFamily(props.font.latin)+','+quoteFamily(props.font.zh)+',serif','--word-size':props.font.size_px+'px'}:{});
const all=computed(()=>entries(props.preview.analysis.rows,props.settings));
const buckets=computed(()=>groupEntries(all.value,props.settings));
const statistics=computed(()=>summaryEntries(all.value,props.settings));
const matches=computed(()=>searchEntries(all.value,query.value,scope.value));
const matchSet=computed(()=>new Set(matches.value));
const focused=computed(()=>matches.value[hit.value]);
const ordered=computed(()=>orderedEntries(buckets.value,props.settings,mode.value==='final'?'final':'initial'));
const pageCount=computed(()=>Math.max(1,Math.ceil(ordered.value.length/100)));
const wordGroups=computed(()=>{
 const result:{initial:string;final:string;items:typeof all.value}[]=[];
 for(const entry of ordered.value.slice(wordPage.value*100,(wordPage.value+1)*100)){
  const last=result.at(-1);if(last&&last.initial===entry.mapped_initial&&last.final===entry.mapped_final)last.items.push(entry);
  else result.push({initial:entry.mapped_initial,final:entry.mapped_final,items:[entry]});
 }return result;
});
const columns=computed(()=>props.settings.initial_order.slice(colPage.value*10,(colPage.value+1)*10));
const finals=computed(()=>props.settings.final_order.slice(rowPage.value*12,(rowPage.value+1)*12));
const rowCount=computed(()=>Math.max(1,Math.ceil(props.settings.final_order.length/12)));
const colCount=computed(()=>Math.max(1,Math.ceil(props.settings.initial_order.length/10)));
const reviewed=computed(()=>query.value.trim()?all.value.filter(r=>matchSet.value.has(r.index)):all.value);
const reviewCount=computed(()=>Math.max(1,Math.ceil(reviewed.value.length/100)));
function cell(initial:string,final:string){return buckets.value.get(pairKey(initial,final))??[];}
// Prepare each visible cell once instead of slicing again for every tone label.
const visibleRows=ref(0);let frame=0;
watch([mode,rowPage,colPage,buckets],()=>{
 cancelAnimationFrame(frame);visibleRows.value=mode.value==='matrix'?2:0;
 const append=()=>{visibleRows.value=Math.min(visibleRows.value+2,finals.value.length);if(visibleRows.value<finals.value.length)frame=requestAnimationFrame(append);};
 if(mode.value==='matrix')frame=requestAnimationFrame(append);
},{immediate:true,flush:'sync'});
onUnmounted(()=>cancelAnimationFrame(frame));
const matrixRows=computed(()=>mode.value==='matrix'?finals.value.slice(0,visibleRows.value).map(final=>({final,cells:columns.value.map(initial=>{
 const key=pairKey(initial,final),list=cell(initial,final),page=cellPages.value[key]??0;
 return {key,initial,page,count:list.length,pages:Math.ceil(list.length/60),items:list.slice(page*60,(page+1)*60)};
})})):[]);
function cellPage(initial:string,final:string,delta:number){const key=pairKey(initial,final);cellPages.value[key]=(cellPages.value[key]??0)+delta;}
function navigate(delta:number){if(matches.value.length)hit.value=(hit.value+delta+matches.value.length)%matches.value.length;}
async function locate(){
 const index=focused.value;if(index===undefined)return;
 wordPage.value=Math.floor(ordered.value.findIndex(r=>r.index===index)/100);
 const entry=all.value[index];rowPage.value=Math.floor(props.settings.final_order.indexOf(entry.mapped_final)/12);colPage.value=Math.floor(props.settings.initial_order.indexOf(entry.mapped_initial)/10);
 const key=pairKey(entry.mapped_initial,entry.mapped_final);cellPages.value[key]=Math.floor(cell(entry.mapped_initial,entry.mapped_final).findIndex(r=>r.index===index)/60);
 reviewPage.value=Math.floor(reviewed.value.findIndex(r=>r.index===index)/100);
 if(mode.value==='matrix'){cancelAnimationFrame(frame);visibleRows.value=finals.value.length;}
 await nextTick();root.value?.querySelector<HTMLElement>(`.result-sheet [data-index="${index}"]`)?.scrollIntoView({block:'nearest',inline:'nearest'});
}
watch([query,scope],()=>{hit.value=0;wordPage.value=0;reviewPage.value=0;void locate();});
watch([focused,mode],locate,{flush:'post'});
watch(()=>[props.preview,props.settings],()=>{hit.value=0;wordPage.value=0;rowPage.value=0;colPage.value=0;reviewPage.value=0;cellPages.value={};void locate();});
</script>
<template>
<section ref="root" class="results-preview" aria-label="三种结果预览">
 <div class="preview-tabs" role="group" aria-label="结果预览类型">
  <button v-for="item in ([['initial','声母 → 韵母'],['final','韵母 → 声母'],['matrix','二维声韵表']] as const)" :key="item[0]" :aria-pressed="mode===item[0]" @click="mode=item[0]">{{item[1]}}<small>{{item[0]==='matrix'?'XLSX':'DOCX'}}</small></button>
 </div>
 <div class="search-bar">
  <input v-model="query" aria-label="搜索字头或 IPA" placeholder="搜索字头或 IPA，定位到全部记录" @keydown.enter.prevent="navigate(1)"/>
  <select v-model="scope" aria-label="搜索范围"><option value="all">字头与 IPA</option><option value="character">字头</option><option value="ipa">原始 / 归并 IPA</option></select>
  <button :disabled="!matches.length" aria-label="上一个命中" @click="navigate(-1)">↑</button><button :disabled="!matches.length" aria-label="下一个命中" @click="navigate(1)">↓</button>
  <button v-if="query" @click="query=''">清空</button>
 </div>
 <div class="preview-caption" role="status"><span>{{generated?'生成结果快照':'当前设置预览'}}<span v-if="stale" class="warning"> · 设置已修改，需重新生成</span></span><span v-if="query">{{matches.length?'第 '+(hit+1)+' / '+matches.length+' 个命中':'没有匹配记录'}}</span><span v-else>{{all.length}} 条记录 · {{settings.initial_order.length}} 声母 · {{settings.final_order.length}} 韵母</span></div>
 <div v-if="mode!=='matrix'" class="result-sheet word-sheet" :style="paperStyle"><div class="word-paper">
  <header class="sheet-title"><strong>同音字表</strong><span>（{{mode==='initial'?'声母 → 韵母':'韵母 → 声母'}}）</span></header>
  <details class="summary" open><summary>声母、韵母与声调统计</summary>
   <section v-for="stat in statistics" :key="stat.title" class="word-statistic"><h3>{{stat.title}}</h3>
    <table :aria-label="stat.title"><tbody><tr v-for="(line,i) in stat.rows" :key="i"><td v-for="(column,j) in line" :key="j"><template v-if="column"><strong class="ipa-text">{{column.label}}</strong> <span v-for="entry in column.items" :key="entry.index" class="stat-entry">{{entry.character}}<sub v-if="entry.note">{{entry.note}}</sub> </span></template></td></tr></tbody></table>
   </section>
  </details>
  <article v-for="(group,groupIndex) in wordGroups" :key="pairKey(group.initial,group.final)" class="word-group">
   <h3 v-if="groupIndex===0||(mode==='initial'?wordGroups[groupIndex-1].initial!==group.initial:wordGroups[groupIndex-1].final!==group.final)" class="ipa-text">{{label(mode==='initial'?group.initial:group.final)}}</h3>
   <div class="group-line"><strong class="ipa-text inner-symbol">{{label(mode==='initial'?group.final:group.initial)}}</strong><div><span v-for="(entry,i) in group.items" :key="entry.index" class="word-item"><span v-if="i===0||group.items[i-1].tone_class!==entry.tone_class" class="tone-label">[{{entry.tone_class}}]</span><span :data-index="entry.index" class="preview-entry" :class="{match:matchSet.has(entry.index),focused:focused===entry.index}" :title="entry.ipa+' · '+label(entry.mapped_initial)+' / '+label(entry.mapped_final)">{{entry.character}}<sub v-if="entry.note">{{entry.note}}</sub></span></span></div></div>
  </article>
 </div></div>
 <div v-else class="matrix-sheet result-sheet">
  <div class="matrix-controls"><span>行：韵母 · 列：声母</span><div><button :disabled="rowPage===0" @click="rowPage--">上一组韵母</button><span>{{rowPage+1}} / {{rowCount}}</span><button :disabled="rowPage+1>=rowCount" @click="rowPage++">下一组韵母</button></div><div><button :disabled="colPage===0" @click="colPage--">上一组声母</button><span>{{colPage+1}} / {{colCount}}</span><button :disabled="colPage+1>=colCount" @click="colPage++">下一组声母</button></div></div>
  <div class="matrix-scroll"><table aria-label="二维声韵表"><thead><tr><th>韵母 / 声母</th><th v-for="initial in columns" :key="initial" class="ipa-text">{{label(initial)}}</th></tr></thead><tbody><tr v-for="row in matrixRows" :key="row.final"><th class="ipa-text">{{label(row.final)}}</th><td v-for="item in row.cells" :key="item.key" v-memo="[buckets,item.page,matchSet,item.items.some(entry=>entry.index===focused)?focused:-1]"><div class="matrix-cell"><span v-for="(entry,i) in item.items" :key="entry.index" class="word-item"><span v-if="i===0||item.items[i-1].tone_class!==entry.tone_class" class="tone-label">[{{entry.tone_class}}]</span><span :data-index="entry.index" class="preview-entry" :class="{match:matchSet.has(entry.index),focused:focused===entry.index}" :title="entry.ipa">{{entry.character}}<sub v-if="entry.note">{{entry.note}}</sub></span></span></div><div v-if="item.count>60" class="cell-pages"><button :disabled="!item.page" @click="cellPage(item.initial,row.final,-1)">‹</button><small>{{item.page+1}} / {{item.pages}}</small><button :disabled="item.page+1>=item.pages" @click="cellPage(item.initial,row.final,1)">›</button></div></td></tr></tbody></table></div>
 </div>
 <div v-if="mode!=='matrix'" class="pagination"><button :disabled="wordPage===0" @click="wordPage--">上一页</button><span>第 {{wordPage+1}} / {{pageCount}} 页 · 每页 100 条</span><button :disabled="wordPage+1>=pageCount" @click="wordPage++">下一页</button></div>
 <details class="review" :open="review" @toggle="review=($event.target as HTMLDetailsElement).open">
  <summary>导入与归并审阅 <small>{{preview.diagnostics.accepted_rows}} 条 · 重复 {{preview.diagnostics.duplicate_rows}} 条，均保留</small></summary>
  <details v-if="preview.diagnostics.skipped.length"><summary>跳过 {{preview.diagnostics.skipped.length}} 行</summary><p v-for="r in preview.diagnostics.skipped" :key="r.row">第 {{r.row}} 行：{{r.reason}}</p></details>
  <p v-for="r in preview.diagnostics.warnings??[]" :key="r.row" class="warning">第 {{r.row}} 行：{{r.reason}}</p>
  <div v-if="review" class="review-scroll"><table aria-label="导入与归并审阅表"><thead><tr><th>来源行</th><th>字头</th><th>原始 IPA</th><th>备注</th><th>声母 → 归并</th><th>韵母 → 归并</th><th>调值 → 调类</th></tr></thead><tbody><tr v-for="entry in reviewed.slice(reviewPage*100,(reviewPage+1)*100)" :key="entry.index" :class="{focused:focused===entry.index}"><td>{{preview.diagnostics.source_rows?.[entry.index]??entry.index+1}}</td><td>{{entry.character}}</td><td class="ipa-text">{{entry.ipa}}</td><td>{{entry.note}}</td><td class="ipa-text">{{label(entry.initial)}} → {{label(entry.mapped_initial)}}</td><td class="ipa-text">{{label(entry.final)}} → {{label(entry.mapped_final)}}</td><td>{{entry.tone_value}} → {{entry.tone_class}}</td></tr></tbody></table></div>
  <div class="pagination"><button :disabled="reviewPage===0" @click="reviewPage--">上一页记录</button><span>{{reviewPage+1}} / {{reviewCount}}</span><button :disabled="reviewPage+1>=reviewCount" @click="reviewPage++">下一页记录</button></div>
 </details>
</section>
</template>
<style scoped>
.results-preview{display:flex;flex-direction:column;gap:var(--module-gap);min-width:0}.preview-tabs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:var(--control-gap)}.preview-tabs button{min-height:46px;white-space:normal;flex-wrap:wrap;justify-content:space-between;padding:8px 12px}.preview-tabs button[aria-pressed=true]{background:var(--selected);border-color:var(--accent);color:var(--accent)}.preview-tabs small{font-size:.8em;opacity:.8}.search-bar{display:flex;flex-wrap:wrap;gap:var(--control-gap)}.search-bar>input{flex:1;min-width:160px}.preview-caption{display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;font-size:.9em;color:var(--muted)}.warning{color:var(--warning)}.result-sheet{border:1px solid var(--border);border-radius:var(--radius);background:var(--panel);overflow:hidden}.word-sheet{padding:18px 22px}.sheet-title{display:flex;justify-content:center;align-items:center;gap:12px;padding-bottom:14px;border-bottom:1px solid var(--border)}.sheet-title strong{font-size:1.15em}.sheet-title span{color:var(--muted)}.summary{margin:12px 0;color:var(--muted)}summary{cursor:pointer;padding:6px 0}.word-group{padding:12px 0;border-bottom:1px solid var(--border)}.word-group h3{font-size:1.1em;color:var(--accent);margin-bottom:8px}.group-line{display:flex;gap:18px;align-items:baseline}.inner-symbol{flex:none;min-width:45px}.group-line>div{line-height:2.2;min-width:0;overflow-wrap:anywhere}.tone-label{color:var(--muted);font-size:.9em;margin-right:3px}.word-item{display:inline;margin-right:5px}.preview-entry{border-radius:3px;padding:2px;scroll-margin:80px}.preview-entry.match{background:var(--selected)}.preview-entry.focused{outline:2px solid var(--accent);color:var(--accent)}sub{font-size:.8em;vertical-align:sub;margin-left:2px}.pagination{display:flex;justify-content:center;align-items:center;gap:12px;margin-top:14px;font-size:.9em;color:var(--muted)}.matrix-controls{display:flex;gap:8px;flex-wrap:wrap;padding:10px;border-bottom:1px solid var(--border);font-size:.85em;align-items:center}.matrix-controls>div{display:flex;align-items:center;gap:6px}.matrix-scroll,.review-scroll{overflow:auto;max-height:65vh;overscroll-behavior:contain}table{border-collapse:separate;border-spacing:0;width:100%}th,td{text-align:left;padding:10px;border-right:1px solid var(--border);border-bottom:1px solid var(--border);line-height:1.8}thead th{position:sticky;top:0;z-index:2;background:var(--sidebar)}.matrix-scroll td{min-width:170px;max-width:360px;overflow-wrap:anywhere}.matrix-scroll tbody th{position:sticky;left:0;background:var(--sidebar);z-index:1;min-width:100px}.matrix-scroll thead th:first-child{left:0;z-index:3;min-width:130px}.cell-pages{display:flex;align-items:center;gap:6px;margin-top:6px}.cell-pages button{min-width:25px;padding:2px}.review{border:1px solid var(--border);border-radius:var(--radius);padding:10px;background:var(--panel)}.review>summary{font-weight:600;display:flex;align-items:center;gap:10px;flex-wrap:wrap}.review small{font-weight:400;color:var(--muted)}.review-scroll{margin-top:8px}.review-scroll td{min-width:85px}.review-scroll tr.focused{background:var(--selected)}
@media(max-width:700px){.preview-tabs{grid-template-columns:1fr}.preview-tabs button{min-height:34px}.word-sheet{padding:12px}.group-line{gap:10px}.preview-caption{font-size:.85em}}
/* Paper structure follows the exported DOCX; screen batches retain all records. */
.results-preview{flex:1;min-height:0;overflow:hidden}.results-preview>:not(.result-sheet){flex:none}
.result-sheet{flex:1;min-height:0}.word-sheet{overflow:auto;padding:12px;background:var(--app);overscroll-behavior:contain}
.word-paper{max-width:794px;min-height:100%;margin:auto;padding:24px clamp(12px,4%,48px);background:var(--panel);box-shadow:0 1px 5px #0002;font-family:var(--word-font,var(--font-figure));font-size:var(--word-size,var(--figure-size));line-height:1.4}
.sheet-title{flex-direction:column;gap:4px;border:0;padding-bottom:8px}.sheet-title strong{color:var(--word-heading,#2f5496);font-size:1.4em}.sheet-title span{color:var(--text);font-weight:600;font-size:.92em}
.summary{margin:0 0 18px;color:var(--text)}.summary>summary{font-size:.8em;color:var(--muted)}
.word-statistic{margin:8px 0 12px}.word-statistic h3,.word-group h3{color:var(--word-heading,#2f5496);text-align:center;font-weight:600;font-size:1em;margin:10px 0 4px}
.word-statistic table{table-layout:fixed;border-collapse:collapse}.word-statistic td{border:1px solid var(--muted);width:50%;padding:1px 5px;line-height:1.25;vertical-align:top;overflow-wrap:anywhere}
.word-statistic strong{font-weight:600}.stat-entry{white-space:normal}.word-paper sub{font-size:.7em}
.word-group{padding:0;border:0;margin:0 0 8px}.word-group h3{margin:18px 0 8px}
.group-line,.group-line>div{display:inline;line-height:1.5;overflow-wrap:anywhere}.inner-symbol{min-width:0;margin-right:4px}
.word-paper .tone-label{font-size:1em;color:var(--text);margin:0}.word-paper .word-item{margin-right:3px}.word-paper .preview-entry{padding:0}
.pagination{margin-top:0;gap:8px;flex-wrap:wrap}.matrix-sheet{display:flex;flex-direction:column}.matrix-controls{flex:none;padding:6px}
.matrix-scroll{flex:1;min-height:0;max-height:none}.matrix-scroll table{table-layout:fixed;width:max-content;min-width:100%}.matrix-scroll th,.matrix-scroll td{width:190px;max-width:190px;padding:6px;vertical-align:top}
.matrix-scroll th:first-child{width:110px;min-width:110px}.matrix-cell{max-height:190px;overflow:auto;overscroll-behavior:contain;scrollbar-width:thin;contain:content}
.review{padding:6px 10px;max-height:45%;overflow:auto}.review-scroll{max-height:25vh}.review:not([open]){overflow:hidden}
:global([data-theme=dark]) .word-paper{--word-heading:#8cbaff}
@container module (max-width:1060px){.preview-tabs button{min-height:34px;padding:4px 6px}.preview-tabs{grid-template-columns:repeat(3,minmax(0,1fr))}.word-sheet{padding:6px}.word-paper{padding:16px 12px}.search-bar>input{min-width:100px}.search-bar select{max-width:125px}}
</style>
