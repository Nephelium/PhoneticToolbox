<script setup lang="ts">
import {tr} from '../i18n/index.ts';
import {parameterLabel} from '../i18n/parameters.ts';
import { ref,computed,watch } from 'vue';import parameters from '../generated/parameters.json';import ModalDialog from './ModalDialog.vue';
const props=defineProps<{selected:string[];draft?:string[];requireSelection?:boolean;columns?:2|4}>();const emit=defineEmits<{close:[];apply:[keys:string[]];draft:[keys:string[]]}>();
const query=ref(''),draft=ref([...(props.draft??props.selected)]);
const displayParameters=computed(()=>parameters.map(p=>({...p,label:parameterLabel(p.key,p.label)})));
const filtered=computed(()=>displayParameters.value.filter(p=>(p.label+' '+p.key).toLowerCase().includes(query.value.toLowerCase())));
watch(draft,v=>emit('draft',[...v]),{deep:true});
</script>
<template>
<ModalDialog :title="tr('components.output_parameters')" wide @close="emit('close')">
<p class="muted">{{tr('components.edit_the_parameter_draft_for_the_next_analysis_existing')}}</p>
<div class="parameter-toolbar">
<input v-model="query" :aria-label="tr('components.search_parameters')" :placeholder="tr('components.search_parameter_name_or_key')"/>
<button @click="draft=parameters.map(p=>p.key)">{{tr('components.select_all')}}</button>
<button @click="draft=[]">{{tr('components.deselect_all')}}</button>
<span class="mono">{{draft.length}} {{tr('components.80_parameters')}}</span>
</div>
<div class="parameter-grid" :class="{'four-columns':columns===4}">
<label v-for="p in filtered" :key="p.key">
<input v-model="draft" type="checkbox" :value="p.key"/>
<span>{{p.label}}<small>{{p.key}}</small>
</span>
</label>
</div>
<p v-if="!filtered.length" class="empty-small">{{tr('components.no_matching_parameters')}}</p>
<template #footer>
<button @click="emit('close')">{{tr('components.cancel')}}</button>
<button class="primary" :disabled="requireSelection&&!draft.length" @click="emit('apply',draft)">{{tr('components.apply_to_draft')}}</button>
</template>
</ModalDialog>
</template>
<style scoped>
.parameter-grid.four-columns{grid-template-columns:repeat(4,minmax(0,1fr));gap:0 12px}
.parameter-grid.four-columns label{min-width:0;gap:6px}
.parameter-grid.four-columns span{min-width:0;overflow-wrap:anywhere}
@media(max-width:800px){.parameter-grid.four-columns{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:500px){.parameter-grid.four-columns{grid-template-columns:1fr}}
</style>
