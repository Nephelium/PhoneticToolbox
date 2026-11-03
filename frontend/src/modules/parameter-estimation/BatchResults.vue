<script setup lang="ts">
import {localized} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import type {BatchView,JobView} from '../../platform/research.ts';
import {computed} from 'vue';
const props=defineProps<{batches:BatchView[];active:BatchView|null;jobs:JobView[];busy:boolean;desktop:boolean}>();
const current=computed(()=>props.active?.summary.items.find(i=>['running','cancel_requested'].includes(i.state))??props.active?.summary.items.filter(i=>!['not_started','queued'].includes(i.state)).at(-1));
const fileProgress=computed(()=>current.value?.state==='succeeded'?1:current.value?.progress??0);
const phase=computed(()=>!current.value?tr('modules.parameter-estimation.waiting_to_start'):current.value.state!=='running'?labels[current.value.state]:fileProgress.value<.1?tr('modules.parameter-estimation.reading_and_computing_full_interval_statistics'):fileProgress.value<.25?tr('modules.parameter-estimation.detecting_egg_cycles'):fileProgress.value<.75?tr('modules.parameter-estimation.computing_parameters_in_segments'):fileProgress.value<.94?tr('modules.parameter-estimation.writing_parameter_table'):tr('modules.parameter-estimation.preparing_results'));
defineEmits<{select:[id:string];cancel:[];save:[];retry:[id:string];download:[id:string,name:string]}>();
const labels:Record<string,string>=localized(()=>({not_started:tr('modules.parameter-estimation.not_started'),queued:tr('modules.parameter-estimation.queued'),running:tr('modules.parameter-estimation.computing'),cancel_requested:tr('modules.parameter-estimation.cancelling'),cancelled:tr('modules.parameter-estimation.cancelled'),succeeded:tr('modules.parameter-estimation.completed'),failed:tr('modules.parameter-estimation.failed'),interrupted:tr('modules.parameter-estimation.interrupted')}));
const errors:Record<string,string>=localized(()=>Object.assign({input_unavailable:tr('modules.parameter-estimation.input_has_changed_been_removed_or_expired'),worker_interrupted:tr('modules.parameter-estimation.the_computation_process_was_interrupted_you_can_retry'),execution_failed:tr('modules.parameter-estimation.processing_failed_check_the_audio_and_associated_files'),deadline_exceeded:tr('modules.parameter-estimation.processing_exceeded_the_time_limit'),output_budget_exceeded:tr('modules.parameter-estimation.results_exceed_the_current_size_limit'),disk_space_low:tr('modules.parameter-estimation.insufficient_disk_space'),quota_exceeded:tr('modules.parameter-estimation.insufficient_project_space'),cancelled:tr('modules.parameter-estimation.cancelled')},{
 m01_egg_filter_range:tr('modules.parameter-estimation.the_egg_low_pass_cutoff_must_be_below_half'),
 m01_duration_limit:tr('modules.parameter-estimation.the_maximum_supported_duration_is_30_minutes_split_the'),m01_invalid_source:tr('modules.parameter-estimation.the_audio_format_or_sample_rate_is_unsupported'),m01_channel_invalid:tr('modules.parameter-estimation.the_analysis_channel_does_not_exist_or_audio_and'),m01_frame_budget:tr('modules.parameter-estimation.the_frame_step_is_too_small_and_exceeds_the'),m01_output_budget:tr('modules.parameter-estimation.output_exceeds_the_current_disk_budget_reduce_parameters_or'),m01_runtime_not_admitted:tr('modules.parameter-estimation.this_segmentation_feature_is_currently_available_only_in_the'),m01_process_failed:tr('modules.parameter-estimation.the_computation_subprocess_did_not_finish_check_local_task'),
 analysis_energy_failed:tr('modules.parameter-estimation.intensity_calculation_failed_no_result_was_generated_for_this'),
 analysis_formants_failed:tr('modules.parameter-estimation.formant_calculation_failed_no_result_was_generated_for_this'),
 analysis_praat_pitch_failed:tr('modules.parameter-estimation.praat_fundamental_frequency_calculation_failed_no_result_was_generated'),
 analysis_reaper_failed:tr('modules.parameter-estimation.the_selected_reaper_backend_failed_no_result_was_generated'),
 analysis_spectrum_failed:tr('modules.parameter-estimation.harmonic_spectrum_calculation_failed_no_result_was_generated_for'),
 analysis_tilt_failed:tr('modules.parameter-estimation.spectral_tilt_calculation_failed_no_result_was_generated_for'),
 analysis_correction_failed:tr('modules.parameter-estimation.harmonic_correction_failed_no_result_was_generated_for_this'),
 analysis_cpp_failed:tr('modules.parameter-estimation.cpp_calculation_failed_no_result_was_generated_for_this'),
 analysis_hnr_failed:tr('modules.parameter-estimation.hnr_calculation_failed_no_result_was_generated_for_this'),
 analysis_shr_failed:tr('modules.parameter-estimation.shr_calculation_failed_no_result_was_generated_for_this'),
 analysis_slope_failed:tr('modules.parameter-estimation.spectral_slope_calculation_failed_no_result_was_generated_for'),
 analysis_soe_failed:tr('modules.parameter-estimation.excitation_strength_calculation_failed_no_result_was_generated_for'),
 analysis_jitter_shimmer_failed:tr('modules.parameter-estimation.perturbation_parameter_calculation_failed_no_result_was_generated_for'),
 invalid_audio:tr('modules.parameter-estimation.wav_content_or_encoding_is_unsupported_check_the_file'),
 invalid_textgrid:tr('modules.parameter-estimation.textgrid_could_not_be_parsed_check_its_format_and'),
 invalid_lip:tr('modules.parameter-estimation.the_associated_lip_file_is_invalid_use_a_lip'),
 analysis_sample_limit:tr('modules.parameter-estimation.audio_exceeds_the_two_million_sample_limit_across_all'),
 analysis_output_limit:tr('modules.parameter-estimation.parameter_results_exceed_the_table_or_size_budget_reduce'),
 analysis_resource_limit:tr('modules.parameter-estimation.computation_exceeded_resource_limits_split_the_audio_before_retrying'),
 no_parameter_frames:tr('modules.parameter-estimation.audio_is_too_short_to_export_parameter_frames_check'),
 deadline_exceeded:tr('modules.parameter-estimation.processing_exceeded_the_time_limit_shorten_the_audio_or'),
 invalid_segment_input:tr('modules.parameter-estimation.segmentation_failed_check_the_wav_textgrid_and_interval_boundaries'),
 segment_budget_exceeded:tr('modules.parameter-estimation.segmentation_exceeds_resource_limits_shorten_the_audio_or_reduce'),
 missing_or_invalid_tier:tr('modules.parameter-estimation.the_selected_tier_is_missing_or_contains_out_of'),
 no_labelled_segments:tr('modules.parameter-estimation.the_selected_tier_has_no_non_silent_labelled_intervals'),
 parent_result_source_mismatch:tr('modules.parameter-estimation.parameter_results_do_not_match_the_source_wav_analyse'),
 legacy_parameter_invalid:tr('modules.parameter-estimation.the_historical_parameter_table_is_invalid_check_the_single'),
 legacy_parameter_budget:tr('modules.parameter-estimation.the_historical_table_exceeds_reading_limits_16_mb_200'),
 legacy_parameter_time_mismatch:tr('modules.parameter-estimation.historical_parameter_times_exceed_the_audio_duration_or_source'),
}));
function audioName(batch:BatchView,job:JobView){const index=batch.summary.items.find(item=>item.job_id===job.id)?.index;return index===undefined?tr('modules.parameter-estimation.audio'):batch.audio_names[index]??tr('modules.parameter-estimation.audio');}
function downloadName(batch:BatchView,job:JobView,name:string){return job.operation==='acoustic_analysis'?audioName(batch,job).replace(/\.wav$/i,'').replace(/[\x00-\x1f/\\:<>"|?*]/g,'_').slice(0,160)+name.slice('result'.length):name;}
</script>
<template>
<section class="m01-results" :aria-label="tr('modules.parameter-estimation.batches_and_results')">
  <div class="m01-file-progress" :aria-label="tr('modules.parameter-estimation.current_audio_progress')"><div><strong>{{tr('modules.parameter-estimation.single_audio_file')}}</strong><span>{{Math.round(fileProgress*100)}}%</span></div><p :title="current&&active?active.audio_names[current.index]:''">{{current&&active?active.audio_names[current.index]:tr('modules.parameter-estimation.waiting_to_start')}}</p><progress :aria-label="tr('modules.parameter-estimation.single_file_processing_progress')" :max="1" :value="fileProgress"/><small aria-live="polite">{{phase}}</small></div>
  <label v-if="batches.length">{{tr('modules.parameter-estimation.processing_records')}}<select :value="active?.id" :disabled="busy" @change="$emit('select',($event.target as HTMLSelectElement).value)"><option v-for="batch in batches" :key="batch.id" :value="batch.id">{{tr('modules.parameter-estimation.batchLabel',{operation:batch.operation==='acoustic_analysis'?tr('modules.parameter-estimation.parameter_analysis'):tr('modules.parameter-estimation.textgrid_segmentation'),count:batch.summary.total,time:new Date(batch.created_at*1000).toLocaleTimeString()})}}</option></select></label>
  <template v-if="active"><p aria-live="polite"><strong>{{active.summary.counts.succeeded}} / {{active.summary.total}} {{tr('modules.parameter-estimation.completed')}}</strong><br/><span class="muted">{{tr('modules.parameter-estimation.failed')}} {{active.summary.counts.failed}} {{tr('modules.parameter-estimation.interrupted_68')}} {{active.summary.counts.interrupted}} {{tr('modules.parameter-estimation.not_started_69')}} {{active.summary.counts.not_started}}</span></p>
    <progress :aria-label="tr('modules.parameter-estimation.batch_processing_progress')" :max="active.summary.total" :value="active.summary.counts.succeeded+active.summary.counts.failed+active.summary.counts.cancelled+active.summary.counts.interrupted"/>
    <div class="toolbar m01-result-actions"><button :disabled="busy||active.summary.closed" @click="$emit('cancel')">{{tr('modules.parameter-estimation.cancel_remaining_processing')}}</button><button class="primary" v-if="desktop" :disabled="busy||!active.summary.counts.succeeded" @click="$emit('save')">{{tr('modules.parameter-estimation.save_completed_results')}}</button></div>
    <p class="hint result-outcome" :class="{hidden:!active.summary.closed||active.summary.complete}">{{tr('modules.parameter-estimation.some_files_in_this_batch_did_not_succeed_completed')}}</p>
    <ol class="m01-result-items"><li v-for="item in active.summary.items" :key="item.index"><div><strong>{{active.audio_names[item.index]}}</strong><span :class="['failed','interrupted'].includes(item.state)?'danger-text':'muted'">{{labels[item.state]}}</span></div><small v-if="item.error_code" class="danger-text" :title="item.error_code">{{errors[item.error_code]??tr('modules.parameter-estimation.processing_is_incomplete_check_input_and_retry')}}</small><button v-if="['failed','interrupted','cancelled'].includes(item.state)&&item.job_id&&active.summary.closed" :disabled="busy" @click="$emit('retry',item.job_id)">{{tr('modules.parameter-estimation.retry_this_file')}}</button></li></ol>
    <template v-if="!desktop"><div v-for="job in jobs.filter(j=>j.state==='succeeded')" :key="job.id" class="m01-result-downloads"><template v-if="job.result_manifest?.kind==='managed_acoustic_files'"><strong>{{audioName(active,job)}}</strong><button v-for="file in job.result_manifest.files.filter(f=>!f.name.toLowerCase().endsWith('.json'))" :key="file.id" :disabled="busy" @click="$emit('download',file.id,downloadName(active,job,file.name))">{{downloadName(active,job,file.name)}}</button></template></div></template>
  </template><p v-else class="empty-small">{{tr('modules.parameter-estimation.no_processing_records_yet')}}</p>
</section>
</template>
<style scoped>
.m01-results{min-width:0}.m01-results progress{display:block;width:100%;max-width:100%}.m01-results button{white-space:normal}.m01-result-downloads strong,.m01-results .danger-text{overflow-wrap:anywhere}
.m01-file-progress{padding:10px;border:1px solid var(--border);border-radius:var(--radius);margin:8px 0 12px}.m01-file-progress>div{display:flex;justify-content:space-between;gap:8px}.m01-file-progress p{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:6px 0;min-height:1.5em}.m01-file-progress small{display:block;min-height:1.5em;margin-top:5px}.m01-result-actions{display:grid;grid-template-columns:1fr;gap:6px}.result-outcome{min-height:3em}.result-outcome.hidden{visibility:hidden}
.m01-results label{display:flex;flex-direction:column;gap:6px}.m01-results select{max-width:100%;min-width:0}.m01-result-items{list-style:none;padding:0;margin:12px 0;max-height:300px;overflow:auto}.m01-result-items li{padding:8px 0;border-bottom:1px solid var(--border)}.m01-result-items li>div{display:flex;gap:8px;justify-content:space-between}.m01-result-items strong{font-size:0.857143rem;overflow-wrap:anywhere}.m01-result-items span,.m01-result-items small{font-size:0.857143rem}.m01-result-items button{margin-top:5px}.m01-result-downloads{display:flex;flex-direction:column;gap:5px;margin-top:8px}.m01-result-downloads button{overflow-wrap:anywhere;text-align:left;font-size:0.857143rem}
</style>
