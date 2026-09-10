<script setup lang="ts">
import type {BatchView,JobView} from '../../platform/research.ts';
defineProps<{batches:BatchView[];active:BatchView|null;jobs:JobView[];busy:boolean;desktop:boolean}>();
defineEmits<{select:[id:string];cancel:[];save:[];retry:[id:string];download:[id:string,name:string]}>();
const labels:Record<string,string>={not_started:'未开始',queued:'排队',running:'计算中',cancel_requested:'正在取消',cancelled:'已取消',succeeded:'已完成',failed:'失败',interrupted:'已中断'};
const errors:Record<string,string>={input_unavailable:'输入已变化、删除或到期',worker_interrupted:'计算进程已中断，可重试',execution_failed:'处理失败，请检查音频与关联文件',deadline_exceeded:'处理时间超过上限',output_budget_exceeded:'结果超过当前大小上限',disk_space_low:'可用磁盘空间不足',quota_exceeded:'项目空间不足',cancelled:'已取消'};
</script>
<template>
<section class="m01-results" aria-label="批次与结果">
  <label v-if="batches.length">处理记录<select :value="active?.id" :disabled="busy" @change="$emit('select',($event.target as HTMLSelectElement).value)"><option v-for="batch in batches" :key="batch.id" :value="batch.id">{{batch.operation==='acoustic_analysis'?'参数分析':'TextGrid切分'}} · {{batch.summary.total}}个 · {{new Date(batch.created_at*1000).toLocaleTimeString()}}</option></select></label>
  <template v-if="active"><p aria-live="polite"><strong>{{active.summary.counts.succeeded}} / {{active.summary.total}} 已完成</strong><br/><span class="muted">失败 {{active.summary.counts.failed}} · 中断 {{active.summary.counts.interrupted}} · 未开始 {{active.summary.counts.not_started}}</span></p>
    <progress aria-label="批次处理进度" :max="active.summary.total" :value="active.summary.counts.succeeded+active.summary.counts.failed+active.summary.counts.cancelled+active.summary.counts.interrupted"/>
    <div class="toolbar"><button v-if="!active.summary.closed" :disabled="busy" @click="$emit('cancel')">取消后续处理</button><button v-if="desktop&&active.summary.counts.succeeded" :disabled="busy" @click="$emit('save')">保存已完成结果</button></div>
    <p v-if="active.summary.closed&&!active.summary.complete" class="hint">本批次未全部成功。已完成文件仍可保存；失败或中断项可以单独重试。</p>
    <ol class="m01-result-items"><li v-for="item in active.summary.items" :key="item.index"><div><strong>{{active.audio_names[item.index]}}</strong><span :class="['failed','interrupted'].includes(item.state)?'danger-text':'muted'">{{labels[item.state]}}</span></div><small v-if="item.error_code" class="danger-text" :title="item.error_code">{{errors[item.error_code]??'处理未完成，请检查输入后重试'}}</small><button v-if="['failed','interrupted','cancelled'].includes(item.state)&&item.job_id&&active.summary.closed" :disabled="busy" @click="$emit('retry',item.job_id)">重试此文件</button></li></ol>
    <template v-if="!desktop"><div v-for="job in jobs.filter(j=>j.state==='succeeded')" :key="job.id" class="m01-result-downloads"><template v-if="job.result_manifest?.kind==='managed_acoustic_files'"><button v-for="file in job.result_manifest.files" :key="file.id" :disabled="busy" @click="$emit('download',file.id,file.name)">{{file.name}}</button></template></div></template>
  </template><p v-else class="empty-small">尚无处理记录。</p>
</section>
</template>
<style scoped>
.m01-results label{display:flex;flex-direction:column;gap:6px}.m01-results select{max-width:100%;min-width:0}.m01-result-items{list-style:none;padding:0;margin:12px 0;max-height:300px;overflow:auto}.m01-result-items li{padding:8px 0;border-bottom:1px solid var(--border)}.m01-result-items li>div{display:flex;gap:8px;justify-content:space-between}.m01-result-items strong{font-size:12px;overflow-wrap:anywhere}.m01-result-items span,.m01-result-items small{font-size:12px}.m01-result-items button{margin-top:5px}.m01-result-downloads{display:flex;flex-direction:column;gap:5px;margin-top:8px}.m01-result-downloads button{overflow-wrap:anywhere;text-align:left;font-size:12px}
</style>
