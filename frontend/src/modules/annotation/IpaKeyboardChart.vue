<script setup lang="ts">
import raw from '../ipa-plus/data/keyboard.json';
import type {ChartSection,ChartSystem,SymbolEntry} from '../ipa-plus/types.ts';
import '../ipa-plus/fonts.css';
export type KeyboardSymbol=Pick<SymbolEntry,'display'|'insertText'|'insertionMode'|'prefix'|'suffix'>;
const props=defineProps<{system:ChartSystem}>();
const emit=defineEmits<{insert:[entry:KeyboardSymbol]}>();
const charts=raw.charts as Record<ChartSystem,ChartSection[]>,entries=raw.entries as Record<string,KeyboardSymbol>;
function insert(event:MouseEvent){const id=(event.target as HTMLElement).closest<HTMLButtonElement>('button[data-key]')?.dataset.key;if(id&&entries[id])emit('insert',entries[id]);}
</script>
<template>
 <div class="ipa-keyboard-chart" @click="insert" @pointerdown="($event.target as HTMLElement).closest('button')&&$event.preventDefault()">
  <div v-memo="[system]" class="compact-sections" :data-system="system">
   <section v-for="section in charts[system]" :key="section.id" :class="['compact-section','section-'+section.kind]" :aria-label="section.title">
    <h3>{{section.title}}</h3>
    <table v-if="section.kind==='matrix'"><thead><tr><th/><th v-for="column in section.columns" :key="column" scope="col">{{column}}</th></tr></thead><tbody><tr v-for="(row,ri) in section.rows" :key="ri"><th scope="row">{{row.label}}</th><td v-for="(cell,ci) in row.cells" :key="ci" :colspan="cell.span??1" :class="{shaded:cell.shaded,half:cell.rightHalfShaded}"><button v-for="id in cell.ids" :key="id" :data-key="id" type="button" class="ipa-key" :aria-label="entries[id].display">{{entries[id].display}}</button></td></tr></tbody></table>
    <div v-else-if="section.kind==='vowels'" class="compact-vowels"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d="M20 9 L83 9 L83 91 L42 91 Z M27 37 L83 37 M34 65 L83 65 M50 9 L66 91"/></svg><span class="vowel-heading">前　　　　央　　　　后</span><div v-for="(point,i) in section.points" :key="i" class="vowel-point" :style="{left:point.x+'%',top:point.y+'%'}"><button v-for="id in point.ids" :key="id" :data-key="id" type="button" class="ipa-key" :aria-label="entries[id].display">{{entries[id].display}}</button></div></div>
    <div v-else class="compact-symbols"><button v-for="id in section.ids" :key="id" :data-key="id" type="button" class="ipa-key" :aria-label="entries[id].display">{{entries[id].display}}</button></div>
   </section>
  </div>
 </div>
</template>
<style scoped>
.compact-sections{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.21em;min-width:38.95em;font-size:var(--ipa-key-size,19px)}.compact-section{border:1px solid var(--border);min-width:0;background:var(--panel)}.section-matrix{grid-column:1/-1}h3{margin:0;padding:.16em .26em;background:var(--app);color:var(--muted);font-size:1em;line-height:1.25;font-weight:500}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{border:1px solid var(--border);padding:0;text-align:center;line-height:1.1}th{font-size:1em;font-weight:400;overflow-wrap:anywhere}th:first-child{width:4.5em}td{height:1.6em}.shaded{background:var(--border)}.half{background:linear-gradient(to right,transparent 50%,var(--border) 50%)}.half .ipa-key{margin-right:50%}.compact-symbols{display:flex;flex-wrap:wrap;align-content:start;padding:.1em;gap:0}.compact-vowels{position:relative;width:13.68em;height:9.47em;margin:1.25em auto .16em}.compact-vowels svg{position:absolute;inset:0;width:100%;height:100%}.compact-vowels path{stroke:var(--muted);stroke-width:.6;fill:none}.vowel-point{position:absolute;transform:translate(-50%,-50%);display:flex;background:var(--panel);white-space:nowrap}.vowel-heading{position:absolute;top:-1.2em;left:1.68em;font-size:1em;color:var(--muted)}
.ipa-keyboard-chart .ipa-key.ipa-key{display:inline-flex;align-items:center;justify-content:center;min-width:1.1em;min-height:1.6em;margin:0;padding:.05em .1em;border:1px solid transparent!important;border-radius:3px;line-height:1.1;font-family:PTB-IPA-Plus,'PTB-Doulos',serif!important;font-size:var(--ipa-key-size,19px);font-weight:400;font-synthesis:none;font-feature-settings:'ccmp' 1,'mark' 1,'mkmk' 1;color:var(--text)!important;background:transparent!important;box-shadow:none!important;backdrop-filter:none!important;filter:none!important;transition:none!important;animation:none!important;transform:none!important}.ipa-keyboard-chart .ipa-key:hover{background:var(--selected)!important;border-color:var(--accent)!important}.ipa-keyboard-chart .ipa-key:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
</style>
