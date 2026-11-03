import {uiError,uiErrorMessage,uiMessage,joinMessages} from '../module-localization.ts';
import {tr,formatErrorCode,messageText,type MessageDescriptor} from '../../i18n/index.ts';
import type {LpcTaskConfig,LpcSpectrumData,FigureFontSnapshot,Tier} from '../../platform/research.ts';

export interface LpcDraft {order:string;freq_max_hz:string;amp_min_db:string;amp_max_db:string;dynamic_y:boolean;analysis_channel?:string}
export const defaults=():LpcDraft=>({order:'50',freq_max_hz:'8000',amp_min_db:'-5',amp_max_db:'35',dynamic_y:false,analysis_channel:'mean'});
function number(value:unknown,label:string|MessageDescriptor,min:number,max:number,integer=false){
  if((typeof value!=='number'&&typeof value!=='string')||String(value).trim()==='')throw uiError(uiMessage("modules.lpc-spectrum.value_cannot_be_empty",{v0:(label)}));
  const n=Number(value);if(!Number.isFinite(n)||n<min||n>max||(integer&&!Number.isInteger(n)))throw uiError(uiMessage("modules.lpc-spectrum.value_must_be_a_value_within_value_value",{v0:(label),v1:(min),v2:(max),v3:(integer?uiMessage('modules.lpc-spectrum.integer'):uiMessage('modules.lpc-spectrum.number'))}));return n;
}
export function parameters(draft:LpcDraft){
  const order=number(draft.order,uiMessage("modules.lpc-spectrum.lpc_order"),1,200,true),freq_max_hz=number(draft.freq_max_hz,uiMessage("modules.lpc-spectrum.maximum_frequency"),100,48000),amp_min_db=number(draft.amp_min_db,uiMessage("modules.lpc-spectrum.minimum_amplitude"),-200,100),amp_max_db=number(draft.amp_max_db,uiMessage("modules.lpc-spectrum.maximum_amplitude"),-200,100);
  if(amp_max_db<=amp_min_db)throw uiError(uiMessage("modules.lpc-spectrum.maximum_amplitude_must_exceed_minimum_amplitude"));
  if(typeof draft.dynamic_y!=='boolean')throw uiError(uiMessage("modules.lpc-spectrum.invalid_dynamic_vertical_axis_setting"));
  const analysis_channel=draft.analysis_channel==null||draft.analysis_channel==='mean'?undefined:number(draft.analysis_channel,uiMessage("modules.lpc-spectrum.analysis_channel"),0,7,true);
  return {order,freq_max_hz,amp_min_db,amp_max_db,dynamic_y:draft.dynamic_y,...(analysis_channel===undefined?{}:{analysis_channel})};
}
export function restoreDraft(raw:unknown):LpcDraft {
  if(!raw||typeof raw!=='object')return defaults();
  const values=raw as LpcDraft;try{parameters(values);return {order:String(values.order),freq_max_hz:String(values.freq_max_hz),amp_min_db:String(values.amp_min_db),amp_max_db:String(values.amp_max_db),dynamic_y:values.dynamic_y,analysis_channel:values.analysis_channel==null?'mean':String(values.analysis_channel)};}catch{return defaults();}
}
export function visibleRange(duration:number,zoom:number,offset:number):[number,number]{const length=duration/Math.max(1,zoom),start=Math.max(0,Math.min(offset,duration-length));return [start,start+length];}
export function audioSelection(a:number,b:number,duration:number):[number,number]|null {
  if(![a,b,duration].every(Number.isFinite)||duration<=0)return null;
  const start=Math.max(0,Math.min(a,b)),end=Math.min(duration,Math.max(a,b));
  return end>start?[start,end]:null;
}
export function gridRangeStatus(tiers:Tier[],duration:number,sampleRate:number):'valid'|'blank-overhang'|'mismatch' {
  let blank=false;
  for(const tier of tiers)for(const interval of tier.intervals){
    if(interval.xmin<0||interval.xmax>duration+1/sampleRate){
      if(interval.text.trim())return 'mismatch';
      blank=true;
    }
  }
  return blank?'blank-overhang':'valid';
}
export function configuration(draft:LpcDraft,start:unknown,end:unknown,sampleRate:number,frames:number,tier:string|null,font:FigureFontSnapshot):LpcTaskConfig {
  const p=parameters(draft),duration=frames/sampleRate,roi_start=number(start,uiMessage("modules.lpc-spectrum.selection_start"),0,duration),roi_end=number(end,uiMessage("modules.lpc-spectrum.selection_end"),0,duration);
  const n=Math.trunc(roi_end*sampleRate)-Math.trunc(roi_start*sampleRate);
  if(roi_end<=roi_start)throw uiError(uiMessage("modules.lpc-spectrum.selection_end_must_be_after_start"));
  if(n<p.order+2)throw uiError(uiMessage("modules.lpc-spectrum.the_current_order_needs_at_least_value_samples_enlarge",{v0:(p.order+2)}));
  if(n>48000)throw uiError(uiMessage("modules.lpc-spectrum.the_selection_has_value_samples_exceeding_the_48_000",{v0:(n.toLocaleString())}));
  return {...p,roi_start,roi_end,tier_name:tier,font:{...font}};
}
export interface LpcResult {
  schema_version:'m04/1';config:LpcTaskConfig;input_name:string;input_sha256:string;textgrid_sha256:string|null;
  sample_rate_hz:number;sample_count:number;channels:number;label:string;tier_name:string|null;
  selection:{start_sample:number;end_sample:number;start_s:number;end_s:number;interval:'half-open'};
  spectrum:LpcSpectrumData;reference_spectrum?:{frequencies_hz:number[];magnitude_db:number[]};export_names:Record<string,string>;
}
export function parseResult(raw:ArrayBuffer):LpcResult {
  const r=JSON.parse(new TextDecoder().decode(raw)) as LpcResult,s=r.spectrum;
  if(r.schema_version!=='m04/1'||!s||s.frequencies_hz.length!==1024||s.magnitude_db.length!==1024||!s.frequencies_hz.every((f,i,a)=>Number.isFinite(f)&&(i===0?f===0:f>a[i-1]))||!s.magnitude_db.every(Number.isFinite)||!Number.isFinite(s.amp_min_db)||!Number.isFinite(s.amp_max_db)||s.amp_max_db<=s.amp_min_db)throw uiError(uiMessage("modules.lpc-spectrum.invalid_lpc_result_format_or_spectral_values"));
  const fft=r.reference_spectrum;
  if(fft&&(!Array.isArray(fft.frequencies_hz)||!Array.isArray(fft.magnitude_db)||fft.frequencies_hz.length<2||fft.frequencies_hz.length>32769||fft.frequencies_hz.length!==fft.magnitude_db.length||!fft.frequencies_hz.every((f,i,a)=>Number.isFinite(f)&&(i===0?f===0:f>a[i-1]))||!fft.magnitude_db.every(Number.isFinite)))throw uiError(uiMessage("modules.lpc-spectrum.invalid_fft_reference_spectrum_format"));
  if(!r.selection||!Number.isInteger(r.selection.start_sample)||!Number.isInteger(r.selection.end_sample)||r.selection.end_sample<=r.selection.start_sample||r.selection.interval!=='half-open'||!r.config||typeof r.input_name!=='string'||typeof r.input_sha256!=='string')throw uiError(uiMessage("modules.lpc-spectrum.lpc_result_lacks_valid_time_and_source_information"));return r;
}
export function sameAnalysis(result:LpcResult,config:LpcTaskConfig,sha:string,gridSha:string|null){
  return (result.config.analysis_channel??null)===(config.analysis_channel??null)&&result.input_sha256===sha&&result.textgrid_sha256===gridSha&&(['order','freq_max_hz','roi_start','roi_end','tier_name'] as const).every(k=>result.config[k]===config[k]);
}
const jobErrorKeys:Record<string,string>={"lpc_invalid_channel":"modules.lpc-spectrum.the_selected_analysis_channel_does_not_exist_select_again","lpc_roi_budget":"modules.lpc-spectrum.selection_exceeds_48_000_samples_narrow_the_range","lpc_segment_too_short":"modules.lpc-spectrum.selection_is_too_short_enlarge_the_range_or_lower","lpc_solver_failed":"modules.lpc-spectrum.cannot_solve_lpc_for_this_selection_check_for_silence","lpc_invalid_roi":"modules.lpc-spectrum.invalid_time_selection_select_again","lpc_textgrid_range":"modules.lpc-spectrum.textgrid_has_nonempty_labels_outside_the_audio_link_matching","missing_or_invalid_tier":"modules.lpc-spectrum.invalid_textgrid_tier_link_it_again","lpc_runtime_unavailable":"modules.lpc-spectrum.lpc_computation_environment_is_unavailable_restart_the_workbench","lpc_runtime_mismatch":"modules.lpc-spectrum.lpc_computation_environment_version_mismatch","lpc_input_budget":"modules.lpc-spectrum.audio_exceeds_limits_of_64_mb_eight_million_frames","lpc_sample_rate":"modules.lpc-spectrum.sample_rate_must_be_within_8_96_khz","font_unavailable":"modules.lpc-spectrum.export_fonts_are_unavailable_check_plot_fonts_in_workbench","deadline_exceeded":"modules.lpc-spectrum.computation_timed_out_narrow_the_selection_and_retry","input_unavailable":"modules.lpc-spectrum.source_file_is_no_longer_available_upload_or_select","quota_exceeded":"modules.lpc-spectrum.file_storage_is_full_clean_up_project_files_and","invalid_audio":"modules.lpc-spectrum.audio_is_invalid_or_contains_unsupported_samples"};
export function jobErrorMessage(error:unknown):string|MessageDescriptor {const value=uiErrorMessage(error);return typeof value==='string'?(jobErrorKeys[value]?uiMessage(jobErrorKeys[value]):uiErrorMessage(uiError(value))):value;}
export const jobError=(code:string):string=>messageText(jobErrorMessage(code));
