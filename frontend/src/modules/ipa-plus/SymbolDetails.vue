<script setup lang="ts">
import type {SymbolEntry} from './types.ts';
import {sourceLinks} from './catalog.ts';
import {computed} from 'vue';
import {contentFor} from './content.ts';
const props=defineProps<{entry:SymbolEntry;compact?:boolean}>();
const content=computed(()=>contentFor(props.entry));
// Keep the short chart citation here. Complete bibliography remains in Sources.
const visibleSources=computed(()=>props.entry.sourceRefs.filter(s=>!['ipa-chart-zh-2007','ipa-handbook-jiang-2008'].includes(s.sourceId)));
</script>
<template>
  <div class="m17-details-content">
    <div class="m17-detail-heading"><strong class="m17-ipa">{{entry.display}}</strong><div><b>{{content.nameZh}}</b><small>{{content.nameEn}}</small></div></div>
    <small v-if="entry.notationStatus==='combination'" class="m17-notation-status">组合输入示例 · 沿用既有字母与附加符号</small><small v-else-if="entry.notationStatus==='historical'" class="m17-notation-status">旧版文献记号 · 未列入当前2025基本表</small>
    <p v-if="content.descriptionZh">{{content.descriptionZh}}</p><p v-if="content.usageZh">{{content.usageZh}}</p>
    <p v-if="content.contrastZh" class="hint">{{content.contrastZh}}</p>
    <p v-if="content.notesZh" class="m17-custom-notes">{{content.notesZh}}</p>
    <p v-if="entry.representation" class="m17-representation">{{entry.representation}}</p>
    <p v-if="!compact&&entry.display!==entry.insertText" class="hint">实际输入：<span class="m17-ipa m17-detail-example">{{entry.insertText}}</span></p>
    <p v-if="!compact&&entry.examples.length" class="hint">例示：<span v-for="(example,i) in entry.examples" :key="i" class="m17-ipa m17-detail-example">{{example.text}}</span></p>
    <code v-if="!entry.isExample&&entry.codePoints.length<=8">{{entry.codePoints.join(' ')}}</code>
    <p v-if="!compact&&entry.aliases.length" class="hint">检索别名（含 CIN）：{{entry.aliases.join(' · ')}}</p>
    <template v-for="source in visibleSources" :key="source.sourceId+source.locator"><small>{{source.locator}}</small><a :href="sourceLinks[source.sourceId]?.url" target="_blank" rel="noopener noreferrer">{{sourceLinks[source.sourceId]?.title}}</a></template>
  </div>
</template>
<style scoped>
.m17-details-content{display:flex;flex-direction:column;gap:8px;line-height:1.5;font-size:0.928571rem}
.m17-detail-heading{display:flex;align-items:center;gap:12px}.m17-detail-heading>strong{font-size:2.571429rem;flex:none;line-height:1.4}.m17-detail-heading small{display:block;margin-top:3px}
.m17-details-content p{line-height:1.65}.m17-details-content code{white-space:normal;overflow-wrap:anywhere;font-size:0.857143rem;color:var(--muted)}
.m17-detail-example{font-size:1.571429rem;margin-right:12px}.m17-representation{color:var(--warning)}.m17-details-content a{font-size:0.857143rem}
.m17-custom-notes{white-space:pre-wrap}.m17-details-content{overflow-wrap:anywhere}
</style>
