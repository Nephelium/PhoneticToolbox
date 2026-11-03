<script setup lang="ts">
import {tr} from '../../i18n/index.ts';
import {vTimePrecision} from '../../design/time-precision.ts';
import {eggDefaults,type Extended} from './state.ts';
const props=defineProps<{value:Extended;mono:boolean;selected:boolean;override:number|undefined}>();
const emit=defineEmits<{override:[value:number|undefined]}>();
function enable(event:Event){props.value.egg=(event.target as HTMLInputElement).checked?eggDefaults():null;if(props.value.egg)props.value.audio_channel=1;}
function channel(){if(props.value.egg)props.value.egg.egg_channel=1-(props.value.audio_channel??1);}
</script>
<template><section class="m01-joint" :aria-label="tr('modules.parameter-estimation.joint_audio_and_egg_analysis')">
<h2>{{tr('modules.parameter-estimation.analysis_channel')}}</h2>
<label>{{tr('modules.parameter-estimation.default_audio_channel')}}<select v-model="value.audio_channel" :aria-label="tr('modules.parameter-estimation.default_analysis_channel')" @change="channel"><option v-if="!value.egg" :value="null">{{tr('modules.parameter-estimation.mix_all_channels')}}</option><option :value="0">{{tr('modules.parameter-estimation.channel_1_left')}}</option><option :value="1">{{tr('modules.parameter-estimation.channel_2_right')}}</option></select></label>
<label class="check"><input type="checkbox" :checked="!!value.egg" @change="enable"/>{{tr('modules.parameter-estimation.compute_egg_jointly')}}</label>
<p class="hint">{{tr('modules.parameter-estimation.up_to_30_minutes_processed_in_segments_automatically_analysis')}}</p>
<template v-if="value.egg">
<p class="hint">{{tr('modules.parameter-estimation.default_egg_channel')}} {{(value.egg.egg_channel??0)+1}}{{tr('modules.parameter-estimation.mono_files_produce_acoustic_parameters_only')}}</p>
<label v-if="selected&&!mono">{{tr('modules.parameter-estimation.current_file_channels')}}<select :value="override??''" :aria-label="tr('modules.parameter-estimation.current_file_analysis_channel')" @change="emit('override',($event.target as HTMLSelectElement).value===''?undefined:Number(($event.target as HTMLSelectElement).value))"><option value="">{{tr('modules.parameter-estimation.use_default')}}</option><option value="0">{{tr('modules.parameter-estimation.audio_1_egg_2')}}</option><option value="1">{{tr('modules.parameter-estimation.audio_2_egg_1')}}</option></select></label>
<p v-if="mono" class="hint">{{tr('modules.parameter-estimation.the_current_file_is_mono_egg_will_be_skipped')}}</p>
<label>{{tr('modules.parameter-estimation.egg_parameter_table')}}<select v-model="value.egg.storage" :aria-label="tr('modules.parameter-estimation.egg_parameter_storage')"><option value="aligned">{{tr('modules.parameter-estimation.interpolate_into_main_table_and_retain_raw_cycle_table')}}</option><option value="cycles">{{tr('modules.parameter-estimation.separate_raw_cycle_table_same_file')}}</option></select></label>
<label v-if="value.egg.storage==='aligned'">{{tr('modules.parameter-estimation.smoothing_after_interpolation_ms_0_disables')}}<input v-time-precision="'ms'" v-model.number="value.egg.smooth_ms" :aria-label="tr('modules.parameter-estimation.egg_smoothing_in_milliseconds')" type="number" min="0" max="1000" step="5"/></label>
<details><summary>{{tr('modules.parameter-estimation.egg_calculation_settings')}}</summary>
<label>{{tr('modules.parameter-estimation.high_pass_hz')}}<input v-model.number="value.egg.highpass_cutoff" type="number" min="1" max="10000"/></label>
<label>{{tr('modules.parameter-estimation.low_pass_hz')}}<input v-model.number="value.egg.lowpass_cutoff" type="number" min="1" max="48000"/></label>
<label>{{tr('modules.parameter-estimation.gci_method')}}<select v-model="value.egg.gci_method"><option value="slope">{{tr('modules.parameter-estimation.maximum_slope')}}</option><option value="scale">{{tr('modules.parameter-estimation.relative_threshold')}}</option></select></label>
<label>{{tr('modules.parameter-estimation.goi_method')}}<select v-model="value.egg.goi_method"><option value="slope">{{tr('modules.parameter-estimation.maximum_slope')}}</option><option value="scale">{{tr('modules.parameter-estimation.relative_threshold')}}</option></select></label>
<label class="check"><input v-model="value.egg.auto_prominence" type="checkbox"/>{{tr('modules.parameter-estimation.automatic_peak_and_valley_prominence')}}</label>
<template v-if="!value.egg.auto_prominence"><label>{{tr('modules.parameter-estimation.peak_prominence')}}<input v-model.number="value.egg.peak_prominence" type="number" min="0" max="1" step=".001"/></label><label>{{tr('modules.parameter-estimation.valley_prominence')}}<input v-model.number="value.egg.valley_prominence" type="number" min="0" max="1" step=".001"/></label></template>
<label>{{tr('modules.parameter-estimation.audio_silence_threshold')}}<input v-model.number="value.egg.silence_threshold" type="number" min="0" max="1" step=".001"/></label>
<label>{{tr('modules.parameter-estimation.maximum_interpolation_gap_ms')}}<input v-time-precision="'ms'" v-model.number="value.egg.max_gap_ms" type="number" min="1" max="1000"/></label>
<label class="check"><input v-model="value.egg.derived" type="checkbox"/>{{tr('modules.parameter-estimation.compute_acoustic_parameters_derived_from_gci_f0')}}</label>
<p class="hint">{{tr('modules.parameter-estimation.adds_gf0_harmonics_cpp_hnr_and_other_columns_pf0')}}</p>
</details></template>
</section></template>
<style scoped>
.m01-joint{border-top:1px solid var(--border);margin-top:12px;padding-top:10px;min-width:0}.m01-joint h2{font-size:1rem;margin:0 0 8px}.m01-joint label{display:grid;gap:4px;margin:8px 0}.m01-joint .check{display:flex;align-items:center;gap:6px}.m01-joint select,.m01-joint input:not([type=checkbox]){width:100%;min-width:0}.m01-joint summary{cursor:pointer;margin:8px 0}.m01-joint .hint{font-size:0.857143rem;line-height:1.5}
</style>
