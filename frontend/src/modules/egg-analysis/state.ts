import {uiError,uiErrorMessage,uiMessage,joinMessages} from '../module-localization.ts';
import {localized} from '../module-localization.ts';
import {tr,messageText,type MessageDescriptor} from '../../i18n/index.ts';
import type {EggTaskConfig,EggPreviewData} from '../../platform/research.ts';
export const defaults=():EggTaskConfig=>({mode:'preview',signal_mode:'filtered',flip_channels:false,roi_start:0,roi_end:.5,micro_center:.25,micro_width_ms:50,gci_method:'slope',goi_method:'scale',peak_prominence:.01,valley_prominence:.01,auto_prominence:true,highpass_cutoff:25,lowpass_cutoff:2000,spec_window_ms:20,spec_vmin:-70,spec_vmax:-10,keep_praat_f0:false,keep_gci_f0:false,keep_reaper_f0:false,f0_policy:'audio-f0/2',f0_min_hz:30,f0_max_hz:800,glottal_movement:false,silence_threshold:.01,generate_images:false,lp_order:null,export_policy:'sample-aligned/1'});
export function taskConfig(config:EggTaskConfig,mode:EggTaskConfig['mode'],order:number|null=null):EggTaskConfig {
  const result={...config,mode,font:undefined,lp_order:mode==='inverse'?order:null};
  if(mode!=='preview'){result.micro_center=null;result.micro_width_ms=50;}
  if(mode==='batch')Object.assign(result,{roi_start:0,roi_end:null,signal_mode:'filtered'});
  return result;
}
export const batchDefaults=():EggTaskConfig=>({...taskConfig(defaults(),'batch'),keep_praat_f0:true,keep_gci_f0:true});
export function signature(config:EggTaskConfig){const c={...defaults(),...config};c.f0_min_hz=config.f0_min_hz??(c.f0_policy==='audio-f0/2'?30:75);c.f0_max_hz=config.f0_max_hz??(c.f0_policy==='audio-f0/2'?800:600);delete c.font;return JSON.stringify(Object.keys(c).sort().map(k=>[k,c[k as keyof EggTaskConfig]]));}
// Parameter-only checks mirror M03/1. Per-file rate/ROI checks remain separate.
export function validateParameters(config:EggTaskConfig){
  const bounded=(value:unknown,min:number,max:number)=>typeof value==='number'&&Number.isFinite(value)&&value>=min&&value<=max;
  if(!bounded(config.highpass_cutoff,0,48000)||!bounded(config.lowpass_cutoff,0,48000)||config.highpass_cutoff!<=0||config.highpass_cutoff!>=config.lowpass_cutoff!||config.lowpass_cutoff!>=48000)throw uiError(uiMessage("modules.egg-analysis.filter_frequencies_must_satisfy_0_high_pass_low_pass"));
  if(!bounded(config.silence_threshold,0,1))throw uiError(uiMessage("modules.egg-analysis.silence_threshold_must_be_within_0_1_and_cannot"));
  if(!bounded(config.peak_prominence,0,10)||!bounded(config.valley_prominence,0,10))throw uiError(uiMessage("modules.egg-analysis.peak_and_valley_prominence_must_be_within_0_10"));
  if(!bounded(config.spec_window_ms,5,50))throw uiError(uiMessage("modules.egg-analysis.spectral_window_must_be_within_5_50_ms"));
  if(!bounded(config.spec_vmin,-160,20)||!bounded(config.spec_vmax,-160,20)||config.spec_vmin!>=config.spec_vmax!)throw uiError(uiMessage("modules.egg-analysis.spectrogram_db_bounds_must_be_within_160_20_with"));
  const low=config.f0_min_hz??(config.f0_policy==='audio-f0/2'?30:75),high=config.f0_max_hz??(config.f0_policy==='audio-f0/2'?800:600);
  if(!bounded(low,10,2000)||!bounded(high,10,2000)||low>=high)throw uiError(uiMessage("modules.egg-analysis.f0_bounds_must_be_within_10_2000_hz_with"));
}
export function validate(config:EggTaskConfig,duration:number,rate:number){
  validateParameters(config);
  if(config.mode==='inverse'&&config.lp_order!=null&&(!Number.isInteger(config.lp_order)||config.lp_order<1||config.lp_order>Math.min(256,Math.floor(.003*rate)-1)))throw uiError(uiMessage("modules.egg-analysis.lp_order_must_be_an_integer_from_1_to",{v0:(Math.min(256,Math.floor(.003*rate)-1))}));
  if(config.mode==='preview'&&(!Number.isFinite(config.micro_width_ms)||config.micro_width_ms!<5||config.micro_width_ms!>5000))throw uiError(uiMessage("modules.egg-analysis.microscopic_window_must_be_within_5_5000_ms"));
  const end=config.roi_end??duration,start=config.roi_start??0;
  if(config.mode==='inverse'&&(end-start>10||(end-start)*rate>960000))throw uiError(uiMessage("modules.egg-analysis.inverse_filtering_selection_is_limited_to_10_seconds_and"));
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>duration)throw uiError(uiMessage("modules.egg-analysis.analysis_selection_must_be_within_the_audio_with_end"));
  if(!Number.isFinite(config.highpass_cutoff)||!Number.isFinite(config.lowpass_cutoff)||config.highpass_cutoff!<=0||config.highpass_cutoff!>=config.lowpass_cutoff!||config.lowpass_cutoff!>=rate/2)throw uiError(uiMessage("modules.egg-analysis.filter_frequencies_must_satisfy_0_high_pass_low_pass_217"));
  if(!Number.isFinite(config.spec_vmin)||!Number.isFinite(config.spec_vmax)||config.spec_vmin!>=config.spec_vmax!)throw uiError(uiMessage("modules.egg-analysis.spectrogram_minimum_db_must_be_below_maximum_db"));
  if(duration>1800||rate<8000||rate>96000)throw uiError(uiMessage("modules.egg-analysis.egg_supports_stereo_wav_up_to_30_minutes_at"));
}
export interface PreviewRecord {config:EggTaskConfig;input_sha256:string;sample_rate_hz:number;sample_count:number;preview:EggPreviewData;selection:{start_s:number;end_s:number}}
const errorKeys:Record<string,string>={"egg_reaper_unavailable":"modules.egg-analysis.the_native_reaper_engine_is_not_ready_use_a","egg_reaper_failed":"modules.egg-analysis.reaper_fundamental_frequency_extraction_failed_uncheck_it_to_continue","egg_preview_failed":"modules.egg-analysis.live_preview_did_not_complete_retry_preview","egg_invalid_preview":"modules.egg-analysis.invalid_preview_parameters_check_current_settings","preview_timeout":"modules.egg-analysis.preview_update_timed_out_and_its_computation_process_was","preview_platform_unverified":"modules.egg-analysis.a_validated_egg_live_preview_environment_is_not_configured","egg_runtime_mismatch":"modules.egg-analysis.the_egg_scientific_environment_differs_from_the_project_s","quota_exceeded":"modules.egg-analysis.file_storage_is_full_remove_unneeded_files_and_retry","asset_expired":"modules.egg-analysis.results_have_expired_recompute","egg_stereo_required":"modules.egg-analysis.egg_requires_stereo_wav_with_egg_on_the_left","egg_viewport_budget":"modules.egg-analysis.live_views_of_long_recordings_are_limited_to_60","egg_input_budget":"modules.egg-analysis.egg_input_is_limited_to_30_minutes_2_gb","egg_filter_failed":"modules.egg-analysis.filtering_failed_check_cutoff_frequencies_and_segment_duration","egg_invalid_roi":"modules.egg-analysis.selection_or_microscopic_centre_is_outside_the_audio","egg_inverse_unavailable":"modules.egg-analysis.the_segment_lacks_sufficient_valid_cycles_for_simplified_cp","egg_inverse_budget":"modules.egg-analysis.inverse_filtering_selection_is_limited_to_10_seconds_and","egg_runtime_unavailable":"modules.egg-analysis.the_independent_egg_scientific_runtime_is_not_ready","analysis_resource_limit":"modules.egg-analysis.computation_exceeded_resource_limits_shorten_the_audio_or_selection","deadline_exceeded":"modules.egg-analysis.computation_timed_out_shorten_the_audio_and_retry","input_unavailable":"modules.egg-analysis.source_file_is_no_longer_available_select_it_again","font_unavailable":"modules.egg-analysis.export_fonts_are_unavailable_check_shared_font_settings"};
export function eggErrorMessage(error:unknown):string|MessageDescriptor {const value=uiErrorMessage(error);return typeof value==='string'?(errorKeys[value]?uiMessage(errorKeys[value]):uiErrorMessage(uiError(value))):value;}
export const errors:Record<string,string>=localized(()=>Object.fromEntries(Object.keys(errorKeys).map(code=>[code,messageText(eggErrorMessage(code))])));

export function inverseAudioFiles<T extends {name:string}>(files:T[]){return ['egg_ORIG.wav','egg_IF.wav'].flatMap(name=>files.filter(file=>file.name===name));}
