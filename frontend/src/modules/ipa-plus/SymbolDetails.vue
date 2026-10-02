<script setup lang="ts">
import type {SymbolEntry} from './types.ts';
import {sourceLinks} from './catalog.ts';
defineProps<{entry:SymbolEntry;compact?:boolean}>();
</script>
<template>
  <div class="m17-details-content">
    <div class="m17-detail-heading"><strong class="m17-ipa">{{entry.display}}</strong><div><b>{{entry.nameZh}}</b><small>{{entry.nameEn}}</small></div></div>
    <p>{{entry.descriptionZh}}</p><p v-if="!compact">{{entry.usageZh}}</p>
    <p v-if="!compact" class="hint">{{entry.contrastZh}}</p>
    <p v-if="entry.representation" class="m17-representation">{{entry.representation}}</p>
    <p v-if="!compact&&entry.display!==entry.insertText" class="hint">实际输入：<span class="m17-ipa m17-detail-example">{{entry.insertText}}</span></p>
    <p v-if="!compact&&entry.examples.length" class="hint">例示：<span v-for="(example,i) in entry.examples" :key="i" class="m17-ipa m17-detail-example">{{example.text}}</span></p>
    <code v-if="!entry.isExample&&entry.codePoints.length<=8">{{entry.codePoints.join(' ')}}</code>
    <p v-if="!compact&&entry.aliases.length" class="hint">检索别名（含 CIN）：{{entry.aliases.join(' · ')}}</p>
    <template v-for="source in entry.sourceRefs" :key="source.sourceId"><small>{{source.locator}}</small><a :href="sourceLinks[source.sourceId]?.url" target="_blank" rel="noopener noreferrer">{{sourceLinks[source.sourceId]?.title}}</a></template>
  </div>
</template>
<style scoped>
.m17-details-content{display:flex;flex-direction:column;gap:8px;line-height:1.5;font-size:13px}
.m17-detail-heading{display:flex;align-items:center;gap:12px}.m17-detail-heading>strong{font-size:38px;flex:none;line-height:1.4}.m17-detail-heading small{display:block;margin-top:3px}
.m17-details-content p{line-height:1.65}.m17-details-content code{white-space:normal;overflow-wrap:anywhere;font-size:12px;color:var(--muted)}
.m17-detail-example{font-size:24px;margin-right:12px}.m17-representation{color:var(--warning)}.m17-details-content a{font-size:12px}
</style>
