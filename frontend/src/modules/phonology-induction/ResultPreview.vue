<script setup lang="ts">
import {computed,ref,watch,nextTick} from 'vue';
import type {Preview,Settings} from './port.ts';
import {entries,groupEntries,orderedEntries,searchEntries,pairKey,label} from './state.ts';
const props=defineProps<{preview:Preview;settings:Settings;generated:boolean;stale:boolean}>();
const root=ref<HTMLElement>(),mode=ref<'initial'|'final'|'matrix'>('initial');
const query=ref(''),scope=ref<'all'|'character'|'ipa'>('all'),hit=ref(0);
const wordPage=ref(0),rowPage=ref(0),colPage=ref(0),review=ref(false),reviewPage=ref(0);
const cellPages=ref<Record<string,number>>({});
const all=computed(()=>entries(props.preview.analysis.rows,props.settings));
const buckets=computed(()=>groupEntries(all.value,props.settings));
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
function cellItems(initial:string,final:string){const key=pairKey(initial,final),page=cellPages.value[key]??0;return cell(initial,final).slice(page*60,(page+1)*60);}
function cellPage(initial:string,final:string,delta:number){const key=pairKey(initial,final);cellPages.value[key]=(cellPages.value[key]??0)+delta;}
function navigate(delta:number){if(matches.value.length)hit.value=(hit.value+delta+matches.value.length)%matches.value.length;}
async function locate(){
 const index=focused.value;if(index===undefined)return;
 wordPage.value=Math.floor(ordered.value.findIndex(r=>r.index===index)/100);
 const entry=all.value[index];rowPage.value=Math.floor(props.settings.final_order.indexOf(entry.mapped_final)/12);colPage.value=Math.floor(props.settings.initial_order.indexOf(entry.mapped_initial)/10);
 const key=pairKey(entry.mapped_initial,entry.mapped_final);cellPages.value[key]=Math.floor(cell(entry.mapped_initial,entry.mapped_final).findIndex(r=>r.index===index)/60);
 reviewPage.value=Math.floor(reviewed.value.findIndex(r=>r.index===index)/100);
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
 <div v-if="mode!=='matrix'" class="result-sheet word-sheet">
  <header class="sheet-title"><strong>同音字表</strong><span>{{mode==='initial'?'声母 → 韵母':'韵母 → 声母'}}</span></header>
  <details class="summary"><summary>声母、韵母与声调统计</summary><p>声母：<span class="ipa-text">{{settings.initial_order.map(label).join('　')}}</span></p><p>韵母：<span class="ipa-text">{{settings.final_order.map(label).join('　')}}</span></p><p>调类：{{settings.tone_order.map(t=>t+' → '+settings.tone_map[t]).join('　')}}</p></details>
  <article v-for="group in wordGroups" :key="pairKey(group.initial,group.final)" class="word-group">
   <h3 class="ipa-text">{{label(mode==='initial'?group.initial:group.final)}}</h3>
   <div class="group-line"><strong class="ipa-text inner-symbol">{{label(mode==='initial'?group.final:group.initial)}}</strong><div><span v-for="(entry,i) in group.items" :key="entry.index" class="word-item"><span v-if="i===0||group.items[i-1].tone_class!==entry.tone_class" class="tone-label">[{{entry.tone_class}}]</span><span :data-index="entry.index" class="preview-entry" :class="{match:matchSet.has(entry.index),focused:focused===entry.index}" :title="entry.ipa+' · '+label(entry.mapped_initial)+' / '+label(entry.mapped_final)">{{entry.character}}<sub v-if="entry.note">{{entry.note}}</sub></span></span></div></div>
  </article>
  <div class="pagination"><button :disabled="wordPage===0" @click="wordPage--">上一页</button><span>第 {{wordPage+1}} / {{pageCount}} 页 · 每页 100 条</span><button :disabled="wordPage+1>=pageCount" @click="wordPage++">下一页</button></div>
 </div>
 <div v-else class="matrix-sheet result-sheet">
  <div class="matrix-controls"><span>行：韵母 · 列：声母</span><div><button :disabled="rowPage===0" @click="rowPage--">上一组韵母</button><span>{{rowPage+1}} / {{rowCount}}</span><button :disabled="rowPage+1>=rowCount" @click="rowPage++">下一组韵母</button></div><div><button :disabled="colPage===0" @click="colPage--">上一组声母</button><span>{{colPage+1}} / {{colCount}}</span><button :disabled="colPage+1>=colCount" @click="colPage++">下一组声母</button></div></div>
  <div class="matrix-scroll"><table aria-label="二维声韵表"><thead><tr><th>韵母 / 声母</th><th v-for="initial in columns" :key="initial" class="ipa-text">{{label(initial)}}</th></tr></thead><tbody><tr v-for="final in finals" :key="final"><th class="ipa-text">{{label(final)}}</th><td v-for="initial in columns" :key="initial"><span v-for="(entry,i) in cellItems(initial,final)" :key="entry.index" class="word-item"><span v-if="i===0||cellItems(initial,final)[i-1].tone_class!==entry.tone_class" class="tone-label">[{{entry.tone_class}}]</span><span :data-index="entry.index" class="preview-entry" :class="{match:matchSet.has(entry.index),focused:focused===entry.index}" :title="entry.ipa">{{entry.character}}<sub v-if="entry.note">{{entry.note}}</sub></span></span><div v-if="cell(initial,final).length>60" class="cell-pages"><button :disabled="!(cellPages[pairKey(initial,final)]??0)" @click="cellPage(initial,final,-1)">‹</button><small>{{(cellPages[pairKey(initial,final)]??0)+1}} / {{Math.ceil(cell(initial,final).length/60)}}</small><button :disabled="((cellPages[pairKey(initial,final)]??0)+1)*60>=cell(initial,final).length" @click="cellPage(initial,final,1)">›</button></div></td></tr></tbody></table></div>
 </div>
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
</style>
