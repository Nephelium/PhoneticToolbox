<script setup lang="ts">
import {tr} from '../../i18n/index.ts';
import {vTimePrecision} from '../../design/time-precision.ts';
import type {EggTaskConfig} from '../../platform/research.ts';
const config=defineModel<EggTaskConfig>({required:true});
defineProps<{autoDbEnabled?:boolean;autoDbEmpty?:boolean}>();
const emit=defineEmits<{autoDb:[]}>();
</script>
<template><div class="egg-parameter-row" :aria-label="tr('modules.egg-analysis.egg_detection_parameters')">
<section class="parameter-group"><div class="parameter-heading"><span>{{tr('modules.egg-analysis.event_detection')}}</span><label class="check" :title="tr('modules.egg-analysis.200_ms_local_window_100_ms_step_peak_threshold')"><input v-model="config.auto_prominence" type="checkbox"/>{{tr('modules.egg-analysis.automatic_peaks')}}</label></div>
<div class="egg-field-grid">
<label :title="tr('modules.egg-analysis.amplitude_difference_from_the_surrounding_baseline_not_a_percentage')">{{tr('modules.egg-analysis.peak_prominence')}}<input v-model.number="config.peak_prominence" :aria-label="tr('modules.egg-analysis.egg_peak_prominence')" type="number" min="0" max="10" step=".001" :disabled="config.auto_prominence"/></label>
<label :title="tr('modules.egg-analysis.detects_valleys_on_negative_egg_using_amplitude_difference_from')">{{tr('modules.egg-analysis.valley_prominence')}}<input v-model.number="config.valley_prominence" :aria-label="tr('modules.egg-analysis.egg_valley_prominence')" type="number" min="0" max="10" step=".001"/></label>
<label>GCI<select v-model="config.gci_method"><option value="slope">{{tr('modules.egg-analysis.slope')}}</option><option value="scale">{{tr('modules.egg-analysis.scale_0_25')}}</option></select></label>
<label>GOI<select v-model="config.goi_method"><option value="slope">{{tr('modules.egg-analysis.slope')}}</option><option value="scale">{{tr('modules.egg-analysis.scale_0_25')}}</option></select></label>
</div></section>
<section class="parameter-group"><div class="parameter-heading"><span>{{tr('modules.egg-analysis.spectrogram_and_f0')}}</span></div>
<div class="egg-field-grid egg-spectral-grid">
<label>{{tr('modules.egg-analysis.spectral_window_ms')}}<input v-time-precision="'ms'" v-model.number="config.spec_window_ms" type="number" min="5" max="50" step="1"/></label>
<button class="auto-db" type="button" :aria-pressed="!!autoDbEnabled" :title="autoDbEnabled?tr('modules.egg-analysis.enabled_after_panning_or_zooming_updates_a_50_db'):tr('modules.egg-analysis.disabled_retains_the_manual_db_range_click_to_enable')" @click="emit('autoDb')">{{tr('modules.egg-analysis.automatic_db')}}</button>
<label>{{tr('modules.egg-analysis.minimum_db')}}<input v-model.number="config.spec_vmin" :aria-label="tr('modules.egg-analysis.egg_minimum_db')" type="number" min="-160" max="20" :disabled="autoDbEnabled"/></label>
<label>{{tr('modules.egg-analysis.maximum_db')}}<input v-model.number="config.spec_vmax" :aria-label="tr('modules.egg-analysis.egg_maximum_db')" type="number" min="-160" max="20" :disabled="autoDbEnabled"/></label>
</div><small class="db-mode-note">{{autoDbEnabled?(autoDbEmpty?tr('modules.egg-analysis.automatic_enabled_silent_intervals_retain_current_scale'):tr('modules.egg-analysis.automatic_enabled_updates_with_visible_interval')):tr('modules.egg-analysis.automatic_disabled_manual_scale')}}</small>
<div class="egg-field-grid egg-f0-grid" :title="tr('modules.egg-analysis.praat_ac_and_reaper_share_the_search_range_gci')">
<label>{{tr('modules.egg-analysis.minimum_f0_hz')}}<input v-model.number="config.f0_min_hz" :aria-label="tr('modules.egg-analysis.egg_minimum_f0_hz')" type="number" min="10" max="2000" step="1"/></label>
<label>{{tr('modules.egg-analysis.maximum_f0_hz')}}<input v-model.number="config.f0_max_hz" :aria-label="tr('modules.egg-analysis.egg_maximum_f0_hz')" type="number" min="10" max="2000" step="1"/></label>
</div><div class="egg-checks"><label :title="tr('modules.egg-analysis.audio_praat_ac_using_f0_bounds_above')"><input v-model="config.keep_praat_f0" type="checkbox"/>Praat F0</label><label><input v-model="config.keep_gci_f0" type="checkbox"/>GCI F0</label><label :title="tr('modules.egg-analysis.audio_reaper_using_f0_bounds_above')"><input v-model="config.keep_reaper_f0" type="checkbox"/>REAPER F0</label></div>
</section></div></template>
<style scoped>
.auto-db[aria-pressed=true]{background:var(--selected);color:var(--accent);border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent);font-weight:600}.auto-db[aria-pressed=true]::after{content:'✓';font-size:0.785714rem}.auto-db{justify-self:end;min-height:28px;padding:3px 6px;font-size:var(--control-size,12px)}.db-mode-note{color:var(--muted);font-size:0.785714rem}
.egg-parameter-row{display:grid;gap:10px;border-top:1px solid var(--border);padding-top:8px}.parameter-group{display:grid;gap:6px;min-width:0}.parameter-heading{display:flex;align-items:center;justify-content:space-between;gap:6px;font-size:var(--control-size,12px);color:var(--muted)}.parameter-heading button{min-height:26px;padding:2px 6px;font-size:inherit}.parameter-heading .check{display:flex;align-items:center;gap:4px;color:var(--text)}.egg-field-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,145px),1fr));gap:5px 10px}.egg-field-grid label{display:grid;grid-template-columns:minmax(0,1fr) minmax(64px,86px);align-items:center;gap:5px;font-size:var(--control-size,12px)}.egg-field-grid input,.egg-field-grid select{width:100%;min-width:0;min-height:28px;padding:3px 5px;font-size:inherit;line-height:1.25}.egg-checks{display:flex;align-items:center;flex-wrap:wrap;gap:6px 14px;font-size:var(--control-size,12px)}.egg-checks label{display:flex;align-items:center;gap:4px}
.egg-spectral-grid,.egg-f0-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
.egg-spectral-grid label{grid-template-columns:minmax(0,1fr) minmax(50px,64px)}.egg-f0-grid label{grid-template-columns:1fr;gap:2px}
</style>
