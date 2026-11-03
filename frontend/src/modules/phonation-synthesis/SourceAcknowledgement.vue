<script setup lang="ts">
import {uiMessage,joinMessages} from '../module-localization.ts';
import {tr,messageRef} from '../../i18n/index.ts';
import AppIcon from '../../components/AppIcon.vue';
import {nextTick,ref} from 'vue';
import ModalDialog from '../../components/ModalDialog.vue';
import sources from '../../generated/sources.json';
import {sourceText} from '../../components/source-display.ts';
import {copyText} from '../../platform/clipboard.ts';
const emit=defineEmits<{references:[]}>();
const upstream=sources.find(source=>source.id==='SRC-ZAIWA')!;
const paper=sources.find(source=>source.id==='REF-ZAIWA')!;
const message=messageRef('');
const details=ref(false);
async function references(){details.value=false;await nextTick();emit('references');}
async function copy(){
 try{await copyText(paper.citation!);message.value=uiMessage('modules.phonation-synthesis.citation_copied');}
 catch{message.value=uiMessage('modules.phonation-synthesis.clipboard_unavailable_select_and_copy_the_paper_citation');}
}
</script>
<template>
 <footer class="m07-attribution" :aria-label="tr('modules.phonation-synthesis.phonation_type_synthesis_sources_and_acknowledgements')">
  <p class="attribution-line">{{tr('modules.phonation-synthesis.attributionWithPermission',{attribution:sourceText(upstream,'attribution'),date:upstream.permission_date??''})}}</p>
  <div class="citation-line">
   <p class="paper-citation">{{paper.citation}}</p>
   <div class="attribution-actions">
    <a :href="paper.urls.doi" target="_blank" rel="noopener noreferrer">{{tr('modules.phonation-synthesis.paper')}}</a>
    <a :href="upstream.urls.repository" target="_blank" rel="noopener noreferrer">{{tr('modules.phonation-synthesis.original_repository')}}</a>
    <button type="button" @click="copy">{{tr('modules.phonation-synthesis.copy_citation')}}</button>
    <button type="button" @click="details=true">{{tr('modules.phonation-synthesis.adaptation_notes')}}</button>
    <span v-if="message" role="status">{{message}}</span>
   </div>
  </div>
 </footer>
 <ModalDialog v-if="details" :title="tr('modules.phonation-synthesis.phonation_type_synthesis_adaptation_notes')" @close="details=false">
  <p>{{sourceText(upstream,'adaptation_note')}}</p>
  <p>{{sourceText(upstream,'license')}}</p>
  <p>{{sourceText(upstream,'adaptation_caution')}} <a :href="upstream.urls.repository" target="_blank" rel="noopener noreferrer">{{tr('modules.phonation-synthesis.try_the_authors_official_open_source_code')}}</a></p>
  <p>{{tr('modules.phonation-synthesis.this_module_creates_experimental_stimuli_by_manipulating_lpc_residuals')}}</p>
  <p class="paper-citation">{{paper.citation}}</p>
  <button type="button" @click="references"><AppIcon name="book"/>{{tr('modules.phonation-synthesis.methods_and_references')}}</button>
 </ModalDialog>
</template>
<style scoped>
.m07-attribution{flex:none;min-width:0;padding:6px 10px;border:1px solid var(--border);border-radius:var(--radius);background:var(--panel);font-size:var(--support-size);line-height:1.45;color:var(--text)}
.m07-attribution p{margin:0;overflow-wrap:anywhere}.permission-date{color:var(--muted);white-space:nowrap}.citation-line{display:flex;align-items:center;flex-wrap:wrap;gap:2px 14px;margin-top:3px}.paper-citation{flex:1 1 700px}.attribution-actions{display:flex;align-items:center;flex-wrap:wrap;gap:4px 10px}.attribution-actions a{color:var(--accent);text-decoration:underline;text-underline-offset:2px}.attribution-actions button{min-height:24px;padding:2px 7px;font-size:inherit}.attribution-actions [role=status]{color:var(--muted)}
</style>
