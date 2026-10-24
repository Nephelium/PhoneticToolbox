<script setup lang="ts">
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
  <button :disabled="busy" @click="$emit('refresh')">刷新分析</button>
  <label>标准答案来源 <select v-model="source" aria-label="标准答案来源"><option value="session">各会话保存的标准答案</option><option value="current">当前设计的标准答案（重新评分）</option></select></label>
  <label>分析格式 <select v-model="format" aria-label="分析格式"><option value="xlsx">XLSX</option><option value="csv">CSV</option></select></label>
  <button class="primary" :disabled="busy||!included.length" @click="exportSummary">导出被试分析</button>
 </div>
 <p>当前项目：{{project.config.title}} · {{included.length}} 个被试会话。每次会话单独列出，同名被试保留独立记录。</p>
 <p v-if="source==='current'">重新评分使用当前序列中设置的标准答案，只匹配相同试次与刺激，保留原会话。可先在序列页补设标准答案。</p>
 <div v-if="summaries.length" class="analysis-table" tabindex="0" aria-label="被试成绩分析，可横向滚动">
  <table><thead><tr><th>被试</th><th>会话 / 状态</th><th>完成 / 总试次</th><th>RT 样本数</th><th>平均 RT（ms）</th><th>中位数（ms）</th><th>准确率</th><th>正确 / 计分</th><th>Q1（ms）</th><th>Q3（ms）</th><th>四分位距（ms）</th><th>样本 SD（ms）</th><th>最小 / 最大（ms）</th><th>未设答案 / 无法计分</th><th>未完成</th><th>中断 / 无效</th><th>缺失 RT</th></tr></thead>
   <tbody><tr v-for="s in summaries" :key="s.sessionId"><th scope="row"><span>{{s.participant}}</span><small :title="s.participantId">{{s.participantId.slice(0,8)}}</small></th><td><time>{{new Date(s.createdAt).toLocaleString()}}</time><small :title="s.sessionId">{{s.sessionId.slice(0,8)}} · {{({ready:'待开始',running:'进行中',paused:'已暂停',completed:'已完成',ended:'提前结束'})[s.status]}}</small></td><td>{{s.completed}} / {{s.total}}</td><td>{{s.rtN}}</td><td>{{number(s.mean)}}</td><td>{{number(s.median)}}</td><td>{{s.accuracy===null?'未计分':(s.accuracy*100).toFixed(2)+'%'}}</td><td>{{s.correct}} / {{s.scored}}</td><td>{{number(s.q1)}}</td><td>{{number(s.q3)}}</td><td>{{number(s.iqr)}}</td><td>{{number(s.sd)}}</td><td>{{number(s.min)}} / {{number(s.max)}}</td><td>{{s.unscored}}</td><td>{{s.unanswered}}</td><td>{{s.interrupted}} / {{s.invalid}}</td><td>{{s.missingRT}}</td></tr></tbody>
  </table>
 </div>
 <p v-else>当前项目还没有被试会话。完成实验或从本机记录载入原项目后，在此查看。</p>
 <p class="muted">RT 统计采用每试次最后一次完成的作答，只纳入有限且非负的反应时，保留 0 ms，不自动剔除离群值。四分位数使用线性插值（R type 7），四分位距 = Q3 − Q1，样本标准差使用 n−1。</p>
 <p class="muted">准确率 = 正确数 ÷ 有标准答案且已完成作答的试次数。未设答案、无法匹配和未完成试次分别列出；无可计分试次时显示未计分。RT 仍从作答窗口开放时起算。</p>
</template>
<style scoped>
.analysis-controls{display:flex;gap:var(--control-gap);align-items:center;flex-wrap:wrap;margin:8px 0}.analysis-controls label{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.analysis-table{overflow:auto;max-width:100%;margin:16px 0}.analysis-table table{border-collapse:collapse;min-width:1500px;width:100%;font-variant-numeric:tabular-nums}.analysis-table th,.analysis-table td{padding:10px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}.analysis-table th{font-weight:600}.analysis-table small{display:block;font-size:.8rem;color:var(--muted);margin-top:4px}.analysis-table tbody th span{display:block;max-width:220px;white-space:normal;overflow-wrap:anywhere}.muted{color:var(--muted)}
</style>
