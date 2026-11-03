<script setup lang="ts">
import {tr} from '../../i18n/index.ts';
import {computed,ref} from 'vue';
import {summarizeSession,type AnswerSource} from './analysis.ts';
import {analysisBlob,download} from './formats.ts';
import type {Project,Session} from './model.ts';
const props=defineProps<{project:Project;sessions:Session[];busy:boolean}>();
defineEmits<{refresh:[]}>();
const source=ref<AnswerSource>('session'),format=ref<'csv'|'xlsx'>('xlsx');
const included=computed(()=>props.sessions.filter(s=>s.project.id===props.project.id));
const summaries=computed(()=>included.value.map(s=>summarizeSession(s,source.value==='current'?props.project:undefined)));
const number=(v:number|null)=>v===null?'—':v.toFixed(2);
function exportSummary(){
 download(analysisBlob(included.value,source.value==='current'?props.project:undefined,format.value),`m15-analysis-${props.project.id}.${format.value}`);
}
</script>
<template>
 <div class="analysis-controls">
  <button :disabled="busy" @click="$emit('refresh')">{{tr('modules.perception.text.18c3aff41145')}}</button>
  <label>{{tr('modules.perception.text.ef70c33216aa')}} <select v-model="source" :aria-label="tr('modules.perception.text.ef70c33216aa')"><option value="session">{{tr('modules.perception.text.1c83d025b32c')}}</option><option value="current">{{tr('modules.perception.text.d0c7a7bad7c2')}}</option></select></label>
  <label>{{tr('modules.perception.text.d47387afcb51')}} <select v-model="format" :aria-label="tr('modules.perception.text.d47387afcb51')"><option value="xlsx">XLSX</option><option value="csv">CSV</option></select></label>
  <button class="primary" :disabled="busy||!included.length" @click="exportSummary">{{tr('modules.perception.text.8e5d24ca017c')}}</button>
 </div>
 <p>{{tr('modules.perception.text.717d9232eb60')}}{{project.config.title}} · {{included.length}} {{tr('modules.perception.text.be09b990fe58')}}</p>
 <p v-if="source==='current'">{{tr('modules.perception.text.039ece134557')}}</p>
 <div v-if="summaries.length" class="analysis-table" tabindex="0" :aria-label="tr('modules.perception.text.066c75ce2d05')">
  <table><thead><tr><th>{{tr('modules.perception.text.34b6b3a5f5a2')}}</th><th>{{tr('modules.perception.text.ec9a650502ed')}}</th><th>{{tr('modules.perception.text.9e66d4b62dec')}}</th><th>{{tr('modules.perception.text.c749665dfccd')}}</th><th>{{tr('modules.perception.text.a87b09e50e71')}}</th><th>{{tr('modules.perception.text.482234010388')}}</th><th>{{tr('modules.perception.text.63564f436b6a')}}</th><th>{{tr('modules.perception.text.3e3c17e0c1f2')}}</th><th>Q1（ms）</th><th>Q3（ms）</th><th>{{tr('modules.perception.text.6051a884361e')}}</th><th>{{tr('modules.perception.text.57c0b78b4027')}}</th><th>{{tr('modules.perception.text.84ba4d696e18')}}</th><th>{{tr('modules.perception.text.e7bbb2ba3d08')}}</th><th>{{tr('modules.perception.text.6707de42c29d')}}</th><th>{{tr('modules.perception.text.6fb6da6620cc')}}</th><th>{{tr('modules.perception.text.93609029c06d')}}</th></tr></thead>
   <tbody><tr v-for="s in summaries" :key="s.sessionId"><th scope="row"><span>{{s.participant}}</span><small :title="s.participantId">{{s.participantId.slice(0,8)}}</small></th><td><time>{{new Date(s.createdAt).toLocaleString()}}</time><small :title="s.sessionId">{{s.sessionId.slice(0,8)}} · {{({ready:tr('modules.perception.text.5b715411b1f9'),running:tr('modules.perception.text.dc9591e56d50'),paused:tr('modules.perception.text.eb0c326b60ae'),completed:tr('modules.perception.text.f28461bb49c8'),ended:tr('modules.perception.text.277c52c08fe6')})[s.status]}}</small></td><td>{{s.completed}} / {{s.total}}</td><td>{{s.rtN}}</td><td>{{number(s.mean)}}</td><td>{{number(s.median)}}</td><td>{{s.accuracy===null?tr('modules.perception.text.e055d7347a20'):(s.accuracy*100).toFixed(2)+'%'}}</td><td>{{s.correct}} / {{s.scored}}</td><td>{{number(s.q1)}}</td><td>{{number(s.q3)}}</td><td>{{number(s.iqr)}}</td><td>{{number(s.sd)}}</td><td>{{number(s.min)}} / {{number(s.max)}}</td><td>{{s.unscored}}</td><td>{{s.unanswered}}</td><td>{{s.interrupted}} / {{s.invalid}}</td><td>{{s.missingRT}}</td></tr></tbody>
  </table>
 </div>
 <p v-else>{{tr('modules.perception.text.e6dbc7845f72')}}</p>
 <p class="muted">{{tr('modules.perception.text.e3e609b6f857')}}</p>
 <p class="muted">{{tr('modules.perception.text.cb7a074d1270')}}</p>
</template>
<style scoped>
.analysis-controls{display:flex;gap:var(--control-gap);align-items:center;flex-wrap:wrap;margin:8px 0}.analysis-controls label{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.analysis-table{overflow:auto;max-width:100%;margin:16px 0}.analysis-table table{border-collapse:collapse;min-width:1500px;width:100%;font-variant-numeric:tabular-nums}.analysis-table th,.analysis-table td{padding:10px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}.analysis-table th{font-weight:600}.analysis-table small{display:block;font-size:.8rem;color:var(--muted);margin-top:4px}.analysis-table tbody th span{display:block;max-width:220px;white-space:normal;overflow-wrap:anywhere}.muted{color:var(--muted)}
</style>
