<script setup lang="ts">
import type {BatchView,JobView} from '../../platform/research.ts';
defineProps<{batches:BatchView[];active:BatchView|null;jobs:JobView[];busy:boolean;desktop:boolean}>();
defineEmits<{select:[id:string];cancel:[];save:[];retry:[id:string];download:[id:string,name:string]}>();
const labels:Record<string,string>={not_started:'未开始',queued:'排队',running:'计算中',cancel_requested:'正在取消',cancelled:'已取消',succeeded:'已完成',failed:'失败',interrupted:'已中断'};
const errors:Record<string,string>={input_unavailable:'输入已变化、删除或到期',worker_interrupted:'计算进程已中断，可重试',execution_failed:'处理失败，请检查音频与关联文件',deadline_exceeded:'处理时间超过上限',output_budget_exceeded:'结果超过当前大小上限',disk_space_low:'可用磁盘空间不足',quota_exceeded:'项目空间不足',cancelled:'已取消'};
Object.assign(errors,{
 invalid_audio:'WAV 内容或编码不受支持，请检查文件。',
 invalid_textgrid:'TextGrid 无法解析，请检查格式与层内容。',
 invalid_lip:'唇形关联文件无效，请使用受限转换得到的 .lip.json。',
 analysis_sample_limit:'音频超过本次计算的 200 万采样值上限（所有声道合计）。请先按 TextGrid 切分，再分析片段。',
 analysis_output_limit:'参数结果超过当前表格或大小预算。请减少输出参数，或先切分音频。',
 analysis_resource_limit:'本次计算超过资源上限。请先切分音频，再重试。',
 no_parameter_frames:'音频太短，没有可导出的参数帧。请检查片段长度和分析窗设置。',
 deadline_exceeded:'本次处理超过时间上限。请缩短音频或减少区间后重试。',
 invalid_segment_input:'无法切分，请检查 WAV、TextGrid 和区间范围。',
 segment_budget_exceeded:'切分超出资源预算，请缩短音频或减少区间。',
 missing_or_invalid_tier:'未找到所选层，或层内区间越界、重叠。请检查本文件的 TextGrid。',
 no_labelled_segments:'所选层没有可保存的非静音标注区间。',
 parent_result_source_mismatch:'参数结果与当前 WAV 的来源不一致，请重新分析该音频。',
 legacy_parameter_invalid:'历史参数表无效。请检查单工作表、Time_s、数值列；公式、重复表头与非普通 params 表不受支持。',
 legacy_parameter_budget:'历史参数表超过读取预算（16 MB、20 万单元格或 XML 32 MB）。请缩小表格。',
 legacy_parameter_time_mismatch:'历史参数时间超出当前音频，或表内已有 Source_Time_s。请选择与完整音频对应的原参数表。',
});
function audioName(batch:BatchView,job:JobView){const index=batch.summary.items.find(item=>item.job_id===job.id)?.index;return index===undefined?'音频':batch.audio_names[index]??'音频';}
function downloadName(batch:BatchView,job:JobView,name:string){return job.operation==='acoustic_analysis'?audioName(batch,job).replace(/\.wav$/i,'').replace(/[\x00-\x1f/\\:<>"|?*]/g,'_').slice(0,160)+name.slice('result'.length):name;}
</script>
<template>
<section class="m01-results" aria-label="批次与结果">
  <label v-if="batches.length">处理记录<select :value="active?.id" :disabled="busy" @change="$emit('select',($event.target as HTMLSelectElement).value)"><option v-for="batch in batches" :key="batch.id" :value="batch.id">{{batch.operation==='acoustic_analysis'?'参数分析':'TextGrid切分'}} · {{batch.summary.total}}个 · {{new Date(batch.created_at*1000).toLocaleTimeString()}}</option></select></label>
  <template v-if="active"><p aria-live="polite"><strong>{{active.summary.counts.succeeded}} / {{active.summary.total}} 已完成</strong><br/><span class="muted">失败 {{active.summary.counts.failed}} · 中断 {{active.summary.counts.interrupted}} · 未开始 {{active.summary.counts.not_started}}</span></p>
    <progress aria-label="批次处理进度" :max="active.summary.total" :value="active.summary.counts.succeeded+active.summary.counts.failed+active.summary.counts.cancelled+active.summary.counts.interrupted"/>
    <div class="toolbar"><button v-if="!active.summary.closed" :disabled="busy" @click="$emit('cancel')">取消后续处理</button><button v-if="desktop&&active.summary.counts.succeeded" :disabled="busy" @click="$emit('save')">保存已完成结果</button></div>
    <p v-if="active.summary.closed&&!active.summary.complete" class="hint">本批次未全部成功。已完成文件仍可保存；失败或中断项可以单独重试。</p>
    <ol class="m01-result-items"><li v-for="item in active.summary.items" :key="item.index"><div><strong>{{active.audio_names[item.index]}}</strong><span :class="['failed','interrupted'].includes(item.state)?'danger-text':'muted'">{{labels[item.state]}}</span></div><small v-if="item.error_code" class="danger-text" :title="item.error_code">{{errors[item.error_code]??'处理未完成，请检查输入后重试'}}</small><button v-if="['failed','interrupted','cancelled'].includes(item.state)&&item.job_id&&active.summary.closed" :disabled="busy" @click="$emit('retry',item.job_id)">重试此文件</button></li></ol>
    <template v-if="!desktop"><div v-for="job in jobs.filter(j=>j.state==='succeeded')" :key="job.id" class="m01-result-downloads"><template v-if="job.result_manifest?.kind==='managed_acoustic_files'"><strong>{{audioName(active,job)}}</strong><button v-for="file in job.result_manifest.files" :key="file.id" :disabled="busy" @click="$emit('download',file.id,downloadName(active,job,file.name))">{{downloadName(active,job,file.name)}}</button></template></div></template>
  </template><p v-else class="empty-small">尚无处理记录。</p>
</section>
</template>
<style scoped>
.m01-results{min-width:0}.m01-results progress{display:block;width:100%;max-width:100%}.m01-results button{white-space:normal}.m01-result-downloads strong,.m01-results .danger-text{overflow-wrap:anywhere}
.m01-results label{display:flex;flex-direction:column;gap:6px}.m01-results select{max-width:100%;min-width:0}.m01-result-items{list-style:none;padding:0;margin:12px 0;max-height:300px;overflow:auto}.m01-result-items li{padding:8px 0;border-bottom:1px solid var(--border)}.m01-result-items li>div{display:flex;gap:8px;justify-content:space-between}.m01-result-items strong{font-size:12px;overflow-wrap:anywhere}.m01-result-items span,.m01-result-items small{font-size:12px}.m01-result-items button{margin-top:5px}.m01-result-downloads{display:flex;flex-direction:column;gap:5px;margin-top:8px}.m01-result-downloads button{overflow-wrap:anywhere;text-align:left;font-size:12px}
</style>
