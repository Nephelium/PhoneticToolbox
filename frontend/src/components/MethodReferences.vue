<script setup lang="ts">
import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr,locale} from '../i18n/index.ts';
import {sourceText} from './source-display.ts';
import { computed, nextTick, ref, useId } from 'vue';
import sources from '../generated/sources.json';
import { modules } from '../app/registry.ts';
import { copyText } from '../platform/clipboard.ts';
const props = defineProps<{ moduleId?: string }>();
const query = ref(''), message = messageRef();
const selectedModule = ref('');
const active = ref('academic');
const id = useId();
const groups = localized(()=>([
  { id: 'academic', title: tr('components.phonetics_and_linguistics_references'), description: tr('components.papers_research_methods_language_rules_and_research_data') },
  { id: 'software', title: tr('components.software_and_code_sources'), description: tr('components.software_tools_implementations_libraries_and_project_assets') },
]));
const matches = computed(() => sources.filter(s =>
  (!(props.moduleId || selectedModule.value) || s.modules.includes(props.moduleId || selectedModule.value)) &&
  [s.title, s.authors, s.citation || '', ...modules.filter(m => s.modules.includes(m.id)).map(m => m.title)].join(' ').toLowerCase().includes(query.value.trim().toLowerCase())));
const visible = computed(() => matches.value.filter(s => s.acknowledgement_group === active.value));
const moduleNames = (ids: string[]) => modules.filter(module => ids.includes(module.id)).map(module => module.title).join(locale.value==='en'?', ':'、') || tr('components.shared_components');
function groupKey(event: KeyboardEvent) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  active.value = event.key === 'Home' ? 'academic' : event.key === 'End' ? 'software' : active.value === 'academic' ? 'software' : 'academic';
  void nextTick(() => document.getElementById(id + '-' + active.value)?.focus());
}
async function copy(text: string) {
  try { await copyText(text); message.value = uiMessage('components.citation_copied'); }
  catch { message.value = uiMessage('components.clipboard_access_failed_select_the_citation_text_and_copy'); }
}
const citation = (source: { authors: string; title: string; citation?: string }) => source.citation || [source.authors, source.title].filter(Boolean).join(' · ');
</script>
<template>
  <p class="muted">{{tr('components.academic_references_retain_their_citations_and_sources_third_party')}}</p>
  <div v-if="!moduleId" class="reference-scope">
    <label>{{tr('components.scope')}} <select v-model="selectedModule" :aria-label="tr('components.view_acknowledgements_by_module')">
      <option value="">{{tr('components.all_modules_and_shared_components')}}</option>
      <option v-for="module in modules" :key="module.id" :value="module.id">{{ module.title }}</option>
    </select></label>
    <p class="hint">{{tr('components.academic_code_model_asset_and_font_sources_are_collected')}}</p>
  </div>
  <div class="reference-groups" role="tablist" :aria-label="tr('components.acknowledgement_groups')" @keydown="groupKey">
    <button v-for="group in groups" :id="id + '-' + group.id" :key="group.id" role="tab"
      :aria-selected="active === group.id" :aria-controls="id + '-panel'" :tabindex="active === group.id ? 0 : -1"
      @click="active = group.id">
      {{ group.title }} <span class="mono">{{ matches.filter(s => s.acknowledgement_group === group.id).length }}</span>
    </button>
  </div>
  <input v-model="query" :aria-label="tr('components.search_sources')" :placeholder="tr('components.search_project_method_author_or_module')"/>
  <p role="status">{{ message }}</p>
  <section :id="id + '-panel'" role="tabpanel" :aria-labelledby="id + '-' + active" tabindex="0">
    <p class="hint">{{ groups.find(group => group.id === active)?.description }}</p>
    <article v-for="s in visible" :key="s.id" class="reference-row">
      <h3>{{ s.title }}</h3>
      <small class="reference-modules">{{tr('components.sourceUsedBy',{modules:moduleNames(s.modules)})}}</small>
      <p class="selectable reference-citation">{{ citation(s) }}</p>
      <p v-if="s.reference_note" class="hint selectable">{{sourceText(s,'reference_note')}}</p>
      <small v-if="s.version">{{tr('components.sourceVersion',{version:s.version})}}</small>
      <small v-if="s.license">{{sourceText(s,'license')}}</small>
      <p v-if="s.attribution" class="selectable">{{sourceText(s,'attribution')}}<span v-if="s.permission_date">{{tr('components.sourcePermissionDate',{date:s.permission_date})}}</span></p>
      <details v-if="s.adaptation_note" class="reference-adaptation"><summary>{{tr('components.adaptation_notes')}}</summary>
        <p>{{sourceText(s,'adaptation_note')}}</p><p v-if="s.adaptation_caution">{{sourceText(s,'adaptation_caution')}}</p>
      </details>
      <div class="reference-links">
        <a v-for="(url,label) in s.urls" :key="label" :href="url" target="_blank" rel="noopener noreferrer">{{ label }}</a>
        <button @click="copy(citation(s))">{{tr('components.copy_citation')}}</button>
      </div>
    </article>
    <p v-if="!visible.length" class="empty-small">{{tr('components.no_matching_sources_in_this_group_try_another_group')}}</p>
  </section>
</template>
<style scoped>
.reference-scope label{display:flex;align-items:center;flex-wrap:wrap;gap:8px}
.reference-scope select{max-width:100%}.reference-modules{color:var(--muted)}
.reference-adaptation{margin:6px 0}.reference-adaptation p{overflow-wrap:anywhere}
.reference-citation{white-space:pre-line;overflow-wrap:anywhere}
</style>
