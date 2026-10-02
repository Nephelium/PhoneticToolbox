<script setup lang="ts">
import {computed} from 'vue';
import {catalog,entries,searchEntries} from './catalog.ts';
import SymbolButton from './SymbolButton.vue';
import type {ChartSystem,SymbolEntry} from './types.ts';
const props=defineProps<{system:ChartSystem;query:string;selected?:string}>();
const emit=defineEmits<{insert:[entry:SymbolEntry];inspect:[entry:SymbolEntry];hover:[entry:SymbolEntry|null]}>();
const sections=computed(()=>catalog.charts[props.system]);const results=computed(()=>searchEntries(props.system,props.query));
const entry=(id:string)=>entries.get(id)!;
const groupIds={ipa:[['pulmonic','vowels'],['diacritics'],['tones'],['nonpulmonic'],['other','suprasegmentals']],extipa:[['consonants'],['diacritics','voicing'],['rhythm'],['uncertainty'],['other']],voqs:[['phonation'],['lingual'],['labial','velum','jaw'],['airstream','larynx','scope']]};
const groups=computed(()=>groupIds[props.system].map(ids=>ids.map(id=>sections.value.find(s=>s.id===id)!)));
</script>
<template>
 <div v-if="query.trim()" class="m17-search-results"><p class="hint">找到 {{results.length}} 项 · 当前 {{system.toUpperCase()}} 表</p><div class="m17-search-grid"><SymbolButton v-for="item in results" :key="item.id" :entry="item" :selected="selected===item.id" named @insert="emit('insert',$event)" @inspect="emit('inspect',$event)" @hover="emit('hover',$event)"/></div><p v-if="!results.length">没有匹配项目。可按符号、中英文名称、U+ 码位或 CIN 编码检索。</p></div>
 <div v-else class="m17-chart-grid" :class="'m17-chart-'+system" :data-chart="system">
  <div v-for="(group,gi) in groups" :key="gi" class="m17-chart-column" :class="'m17-group-'+gi"><section v-for="section in group" :key="section.id" class="m17-chart-section" :class="'m17-section-'+section.id" :aria-label="section.title">
   <h3>{{section.title}}</h3>
   <table v-if="section.kind==='matrix'" class="m17-matrix"><thead><tr><th scope="col"></th><th v-for="column in section.columns" :key="column" scope="col">{{column}}</th></tr></thead><tbody><tr v-for="(row,ri) in section.rows" :key="ri"><th scope="row">{{row.label}}</th><td v-for="(cell,ci) in row.cells" :key="ci" :colspan="cell.span??1" :class="{'m17-impossible':cell.shaded,'m17-half-impossible':cell.rightHalfShaded}" :aria-label="cell.shaded?'原表阴影：判定不可能的构音':cell.rightHalfShaded?'原表右半阴影：浊音判定不可能':undefined"><SymbolButton v-for="id in cell.ids" :key="id" :entry="entry(id)" :selected="selected===id" @insert="emit('insert',$event)" @inspect="emit('inspect',$event)" @hover="emit('hover',$event)"/></td></tr></tbody></table>
   <div v-else-if="section.kind==='vowels'" class="m17-vowels"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d="M20 9 L83 9 L83 91 L42 91 Z M27 37 L83 37 M34 65 L83 65 M50 9 L66 91"/></svg><span class="m17-vowel-label m17-vowel-front">前</span><span class="m17-vowel-label m17-vowel-central">央</span><span class="m17-vowel-label m17-vowel-back">后</span><div v-for="(point,i) in section.points" :key="i" class="m17-vowel-point" :style="{left:point.x+'%',top:point.y+'%'}"><SymbolButton v-for="id in point.ids" :key="id" :entry="entry(id)" :selected="selected===id" @insert="emit('insert',$event)" @inspect="emit('inspect',$event)" @hover="emit('hover',$event)"/></div></div>
   <div v-else class="m17-symbol-flow"><SymbolButton v-for="id in section.ids" :key="id" :entry="entry(id)" :named="system==='voqs'&&section.id!=='scope'" :selected="selected===id" @insert="emit('insert',$event)" @inspect="emit('inspect',$event)" @hover="emit('hover',$event)"/></div>
  </section></div>
 </div>
</template>
<style scoped>
.m17-chart-grid{display:grid;gap:5px;align-content:start;min-width:900px}
.m17-chart-section{border:1px solid var(--border);border-radius:5px;min-width:0;overflow:visible;background:var(--panel)}
h3{font-size:12px;line-height:1.2;color:var(--muted);background:var(--app);padding:3px 6px;border-bottom:1px solid var(--border);font-weight:600}
.m17-matrix{width:100%;border-collapse:collapse;table-layout:fixed}.m17-matrix th{font-size:12px;font-weight:400;line-height:1.15;padding:1px;border:1px solid var(--border)}
.m17-matrix th:first-child{width:67px}.m17-matrix td{padding:0;text-align:center;border:1px solid var(--border);height:30px;white-space:nowrap}.m17-matrix .m17-impossible{background:var(--border)}
.m17-matrix .m17-half-impossible{background:linear-gradient(to right,transparent 50%,var(--border) 50%)}.m17-half-impossible :deep(.m17-symbol){margin-right:50%}
.m17-symbol-flow{display:flex;flex-wrap:wrap;align-items:center;align-content:start;padding:2px 3px;gap:0}
.m17-chart-column{display:flex;flex-direction:column;gap:4px;min-width:0}.m17-chart-ipa{grid-template-columns:2.4fr 1.1fr .9fr 1fr}
.m17-chart-ipa .m17-group-0{grid-column:1/-1;display:grid;grid-template-columns:3.6fr 1fr;gap:5px}
.m17-section-nonpulmonic .m17-matrix th:first-child{width:14px}.m17-section-nonpulmonic .m17-matrix th{font-size:10px}.m17-section-nonpulmonic .m17-matrix td{height:24px}
.m17-vowels{position:relative;height:195px;margin:10px 1px 2px}.m17-vowels svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}.m17-vowels path{fill:none;stroke:var(--border);stroke-width:.7}.m17-vowel-point{position:absolute;display:flex;transform:translate(-50%,-50%);background:var(--panel);border-radius:3px}
.m17-vowel-point :deep(.m17-symbol){min-width:23px;padding:0 2px}.m17-vowel-label{position:absolute;top:-10px;font-size:12px;color:var(--muted)}.m17-vowel-front{left:18%}.m17-vowel-central{left:47%}.m17-vowel-back{left:81%}
.m17-chart-extipa{grid-template-columns:repeat(16,minmax(0,1fr))}.m17-chart-extipa .m17-group-0{grid-column:1/12}.m17-chart-extipa .m17-group-1{grid-column:12/-1}.m17-chart-extipa .m17-group-2{grid-column:1/10}.m17-chart-extipa .m17-group-3{grid-column:10/14}.m17-chart-extipa .m17-group-4{grid-column:14/-1}
.m17-chart-extipa .m17-matrix th:first-child{width:55px}.m17-chart-extipa .m17-matrix th{font-size:11px}.m17-chart-extipa .m17-matrix :deep(.m17-symbol){min-width:21px;padding:0 1px}.m17-chart-extipa .m17-matrix :deep(.m17-symbol>.m17-ipa){font-size:22px}.m17-chart-extipa .m17-matrix td{height:30px}
.m17-chart-voqs{grid-template-columns:2.1fr 1.05fr 1.1fr 1.1fr}
.m17-chart-voqs .m17-section-phonation .m17-symbol-flow{display:grid;grid-template-columns:1fr 1fr}.m17-chart-voqs .m17-section-phonation :deep(.m17-symbol-name){font-size:11px}.m17-chart-voqs .m17-section-phonation :deep(.m17-named>.m17-ipa){min-width:34px}
.m17-section-scope :deep(.m17-named){width:auto}.m17-section-scope :deep(.m17-example){width:100%;flex-wrap:wrap}.m17-section-scope :deep(.m17-example>.m17-ipa){font-size:18px}
.m17-section-scope :deep(.m17-example>.m17-ipa){white-space:normal;overflow-wrap:anywhere;line-height:1.2}.m17-section-scope :deep(.m17-symbol){max-width:100%}
.m17-search-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:5px}.m17-search-results>p{padding:5px}
.m17-matrix td{height:24px}.m17-chart-extipa .m17-matrix td{height:26px}.m17-chart-extipa .m17-matrix :deep(.m17-symbol>.m17-ipa){font-size:22px}
@container module (min-width:1500px){.m17-chart-grid{gap:8px}.m17-matrix td{height:30px}.m17-chart-extipa .m17-matrix td{height:30px}.m17-vowels{height:235px}.m17-symbol-flow{gap:2px;padding:4px}}
</style>
