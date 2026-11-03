<script setup lang="ts">
import {tr} from '../../i18n/index.ts';
import type {SymbolEntry} from './types.ts';
import {sourceLinks} from './catalog.ts';
import {computed} from 'vue';
import {contentFor} from './content.ts';
import {localizedSourceLocator} from './source-localization.ts';
const props=defineProps<{entry:SymbolEntry;compact?:boolean}>();
const content=computed(()=>contentFor(props.entry));
const longExample=computed(()=>props.entry.isExample&&Array.from(props.entry.display).length>12);
// Keep the short chart citation here. Complete bibliography remains in Sources.
const visibleSources=computed(()=>props.entry.sourceRefs.filter(s=>!['ipa-chart-zh-2007','ipa-handbook-jiang-2008'].includes(s.sourceId)).map(source=>({...source,locator:localizedSourceLocator(props.entry.id,source)})));
</script>
<template>
  <div class="m17-details-content">
    <div class="m17-detail-heading" :class="{'m17-detail-heading-example':longExample}"><strong class="m17-ipa">{{entry.display}}</strong><div><b>{{content.nameZh}}</b><small>{{content.nameEn}}</small></div></div>
    <small v-if="entry.notationStatus==='combination'" class="m17-notation-status">{{tr('modules.ipa-plus.text.ae65e272bf72')}}</small><small v-else-if="entry.notationStatus==='historical'" class="m17-notation-status">{{tr('modules.ipa-plus.text.20f222347b87')}}</small>
    <p v-if="content.descriptionZh">{{content.descriptionZh}}</p><p v-if="content.usageZh">{{content.usageZh}}</p>
    <p v-if="content.contrastZh" class="hint">{{content.contrastZh}}</p>
    <p v-if="content.notesZh" class="m17-custom-notes">{{content.notesZh}}</p>
    <p v-if="content.representation" class="m17-representation">{{content.representation}}</p>
    <p v-if="!compact&&entry.display!==entry.insertText" class="hint">{{tr('modules.ipa-plus.text.ecb93019776a')}}<span class="m17-ipa m17-detail-example">{{entry.insertText}}</span></p>
    <p v-if="!compact&&entry.examples.length" class="hint">{{tr('modules.ipa-plus.text.40aa3392ef1a')}}<span v-for="(example,i) in entry.examples" :key="i" class="m17-ipa m17-detail-example">{{example.text}}</span></p>
    <code v-if="!entry.isExample&&entry.codePoints.length<=8">{{entry.codePoints.join(' ')}}</code>
    <p v-if="!compact&&entry.aliases.length" class="hint">{{tr('modules.ipa-plus.text.0056d14fbd30')}}{{entry.aliases.join(' · ')}}</p>
    <template v-for="source in visibleSources" :key="source.sourceId+source.locator"><small>{{source.locator}}</small><a :href="sourceLinks[source.sourceId]?.url" target="_blank" rel="noopener noreferrer">{{sourceLinks[source.sourceId]?.title}}</a></template>
  </div>
</template>
<style scoped>
.m17-details-content{display:flex;flex-direction:column;gap:8px;line-height:1.5;font-size:0.928571rem}
.m17-detail-heading{display:flex;align-items:center;gap:12px}.m17-detail-heading>strong{font-size:2.571429rem;flex:none;line-height:1.4}.m17-detail-heading small{display:block;margin-top:3px}
.m17-detail-heading>div{min-width:0}.m17-detail-heading-example{flex-direction:column;align-items:flex-start;gap:6px}.m17-detail-heading-example>strong{font-size:1.571429rem;max-width:100%;white-space:normal;overflow-wrap:anywhere}
.m17-details-content p{line-height:1.65}.m17-details-content code{white-space:normal;overflow-wrap:anywhere;font-size:0.857143rem;color:var(--muted)}
.m17-detail-example{font-size:1.571429rem;margin-right:12px}.m17-representation{color:var(--warning)}.m17-details-content a{font-size:0.857143rem}
.m17-custom-notes{white-space:pre-wrap}.m17-details-content{overflow-wrap:anywhere}
</style>
