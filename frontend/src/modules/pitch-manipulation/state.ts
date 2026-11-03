import {uiError,uiErrorMessage,uiMessage,joinMessages} from '../module-localization.ts';
import {localized} from '../module-localization.ts';
import {tr,messageText} from '../../i18n/index.ts';
import type {Config,Mode,Point} from './port.ts';
export const modes:{value:Mode;label:string}[]=localized(()=>([{value:'full',label:tr("modules.pitch-manipulation.all_combinations")},{value:'order',label:tr("modules.pitch-manipulation.ascending")},{value:'reverse',label:tr("modules.pitch-manipulation.descending")},{value:'constant',label:tr("modules.pitch-manipulation.constant")}]));
export const defaults=():Config=>({action:'synthesize',start:0,end:null,speed:1,pitch_ratio:1,pitch_hz:0,modified_f0:null,points:[],offset:false});
export type PitchMode='ratio'|'hz';
export interface VariantInput {speed:string;pitch_ratio:string;pitch_hz:string}
export function variantValues(text:string,label:string,positive=false){
 const parts=text.trim().split(/[,，;；\s]+/).filter(Boolean);
 if(!parts.length)throw uiError(joinMessages(label,uiMessage("modules.pitch-manipulation.list_cannot_be_empty")));
 const values=[...new Set(parts.map(x=>finite(x,label)))];
 if(positive&&values.some(x=>x<=0))throw uiError(joinMessages(label,uiMessage("modules.pitch-manipulation.must_be_greater_than_0")));
 return values;
}
export function variants(input:VariantInput,mode:PitchMode){
 const speeds=variantValues(input.speed,tr("modules.pitch-manipulation.tempo_multiplier"),true);
 const pitches=variantValues(mode==='ratio'?input.pitch_ratio:input.pitch_hz,mode==='ratio'?tr("modules.pitch-manipulation.pitch_multiplier"):tr("modules.pitch-manipulation.pitch_shift"),mode==='ratio');
 if(speeds.length*pitches.length>256)throw uiError(uiMessage("modules.pitch-manipulation.up_to_256_versions_per_audio_file_reduce_parameter"));
 return speeds.flatMap(speed=>pitches.map(pitch=>({speed,pitch_ratio:mode==='ratio'?pitch:1,pitch_hz:mode==='hz'?pitch:0})));
}
export function shiftedF0(curve:number[],ratio:number,hz:number,range?:{times:number[];start:number;end:number}){
 return curve.map((f,i)=>{if(f<=0||range&&(range.times[i]<range.start||range.times[i]>range.end))return f;const value=f*ratio+hz;if(!Number.isFinite(value)||value<=0)throw uiError(uiMessage("modules.pitch-manipulation.voiced_frames_must_remain_above_0_hz_after_pitch"));return value;});
}
export function variantLabel(config:Pick<Config,'speed'|'pitch_ratio'|'pitch_hz'>){return tr("modules.pitch-manipulation.tempo_value_value",{v0:(config.speed),v1:(config.pitch_hz!==0?tr('modules.pitch-manipulation.pitch_value_value_hz',{v0:config.pitch_hz>0?'+':'',v1:config.pitch_hz}):tr('modules.pitch-manipulation.pitch_value',{v0:config.pitch_ratio}))});}
export function finite(value:unknown,label:string){if(value===null||value===undefined||!String(value).trim()||!Number.isFinite(Number(value)))throw uiError(joinMessages(label,uiMessage("modules.pitch-manipulation.must_be_a_finite_number")));return Number(value);}
export function frequencies(text:string){const values=text.trim().split(/[,，;；\s]+/).filter(Boolean).map(x=>finite(x,tr("modules.pitch-manipulation.frequency_202")));if(!values.length)throw uiError(uiMessage("modules.pitch-manipulation.frequency_list_cannot_be_empty"));return values;}
export function validate(points:Point[],start:number,end:number){
 if(points.length<2)throw uiError(uiMessage("modules.pitch-manipulation.at_least_start_and_end_points_are_required"));let diagonal:number|undefined,count=1;
 for(let i=0;i<points.length;i++){const p=points[i];finite(p.time,tr("modules.pitch-manipulation.time"));if(p.time<start||p.time>end||(i&&p.time<=points[i-1].time))throw uiError(uiMessage("modules.pitch-manipulation.turning_point_times_must_increase_and_remain_within_the"));if(!p.freqs.length||p.freqs.some(x=>!Number.isFinite(x)))throw uiError(uiMessage("modules.pitch-manipulation.frequency_list_cannot_be_empty_or_contain_nonfinite_values"));if(!modes.some(m=>m.value===p.mode))throw uiError(uiMessage("modules.pitch-manipulation.unknown_connection_type"));if(p.mode==='full')count*=p.freqs.length;if(p.mode==='order'||p.mode==='reverse'){if(diagonal!==undefined&&diagonal!==p.freqs.length)throw uiError(uiMessage("modules.pitch-manipulation.ascending_descending_lists_must_have_equal_lengths"));diagonal=p.freqs.length;}}
 count*=diagonal??1;if(count>256)throw uiError(uiMessage("modules.pitch-manipulation.this_run_exceeds_256_results_split_the_frequency_lists"));return count;
}
export function importF0(times:number[],modified:number[],start:number,end:number,text:string){
 const values=text.replace(/[,，;；]/g,' ').split(/\r?\n/).filter(x=>x.trim()).map(line=>{const parts=line.trim().split(/\s+/);return finite(parts.length>1?parts[1]:parts[0],tr("modules.pitch-manipulation.import_fundamental_frequency"));});
 if(!values.length)throw uiError(uiMessage("modules.pitch-manipulation.fundamental_frequency_sequence_cannot_be_empty"));const view=times.map((t,i)=>t>=start&&t<=end?i:-1).filter(i=>i>=0),voiced=view.filter(i=>modified[i]>0);
 if(!voiced.length)throw uiError(uiMessage("modules.pitch-manipulation.no_valid_fundamental_frequency_in_the_current_view"));if(voiced.some((n,i)=>i>0&&n!==voiced[i-1]+1))throw uiError(uiMessage("modules.pitch-manipulation.fundamental_frequency_is_discontinuous_in_this_view_zoom_to"));
 const result=[...modified];for(let i=0;i<voiced.length;i++){const x=voiced.length===1?0:i/(voiced.length-1)*(values.length-1),a=Math.floor(x),b=Math.min(values.length-1,a+1);result[voiced[i]]=values[a]+(values[b]-values[a])*(x-a);}return result;
}
export function editCurve(original:number[],modified:number[],index:number,value:number,last?:{index:number;value:number},restore=false){
 const result=[...modified];if(!restore&&original[index]<=0)return {curve:result,last};const lo=Math.min(index,last?.index??index),hi=Math.max(index,last?.index??index);
 for(let i=lo;i<=hi;i++){if(restore)result[i]=original[i];else if(original[i]>0)result[i]=last&&hi>lo?last.value+(value-last.value)*(i-last.index)/(index-last.index):value;}
 return {curve:result,last:{index,value}};
}
export function renamePrefix(files:{name:string}[]){let common=files[0]?.name??'';for(const file of files)while(!file.name.startsWith(common))common=common.slice(0,-1);return common.replace(/\.wav$/i,'');}
export function renamePlan(files:{id:string;name:string}[],prefix:string,existing:string[]){
 if(!files.length)throw uiError(uiMessage("modules.pitch-manipulation.select_results_from_this_batch_first"));const common=renamePrefix(files);
 if(!common)throw uiError(uiMessage("modules.pitch-manipulation.selected_files_have_no_shared_prefix"));if(!prefix.trim()||/[\x00-\x1f/\\:<>"|?*]/.test(prefix))throw uiError(uiMessage("modules.pitch-manipulation.prefix_contains_invalid_filename_characters"));
 const seen=new Set<string>();return files.map(f=>{const base=prefix+f.name.slice(common.length);let name=base,index=2;while(seen.has(name.toLowerCase()))name=base.replace(/\.wav$/i,` (${index++}).wav`);const key=name.toLowerCase();if(!/\.wav$/i.test(name)||name.length>180)throw uiError(uiMessage("modules.pitch-manipulation.name_must_retain_wav_and_be_no_longer_than"));if(existing.some(n=>n.toLowerCase()===key))throw uiError(uiMessage("modules.pitch-manipulation.duplicate_names_renaming_was_not_performed"));seen.add(key);return {id:f.id,name};});
}
export function visible(duration:number,zoom:number,offset:number):[number,number]{const span=duration/Math.max(1,zoom),left=Math.max(0,Math.min(offset,duration-span));return [left,left+span];}
const errorKeys:Record<string,string>={"m08_audio_decode_failed": "modules.pitch-manipulation.audio_cannot_be_decoded_check_file_format_and_contents", "m08_praat_error": "modules.pitch-manipulation.praat_cannot_process_this_audio_or_frequency_configuration_check", "m08_input_budget": "modules.pitch-manipulation.audio_exceeds_this_input_budget_select_shorter_material_explicitly", "m08_output_budget": "modules.pitch-manipulation.expected_results_exceed_this_output_budget_split_the_task", "m08_cancelled": "modules.pitch-manipulation.task_cancelled", "m08_diagonal_length": "modules.pitch-manipulation.ascending_descending_list_lengths_must_match", "m08_invalid_range": "modules.pitch-manipulation.time_range_exceeds_audio_bounds", "m08_input_changed": "modules.pitch-manipulation.original_file_changed_read_it_again", "m08_curve_length": "modules.pitch-manipulation.f0_frame_count_differs_from_source_audio_read_it"};
export function errorMessage(error:unknown){const value=uiErrorMessage(error);return typeof value==='string'&&errorKeys[value]?uiMessage(errorKeys[value]):value;}
export function errorText(error:unknown){return messageText(errorMessage(error));}
