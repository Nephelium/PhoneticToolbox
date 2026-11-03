import {joinMessages} from '../modules/module-localization.ts';
import {uiMessage,uiError,localized,uiErrorMessage} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {waitForTaskPoll} from './task-poll.ts';
import {m07Archive} from './m07Archive.ts';
import {taskDescription} from '../modules/phonation-synthesis/history.ts';
import type {components} from '../../../contracts/generated/api';
import type {ResearchFile,JobView} from './research.ts';
import {sha256} from './research.ts';
import type {Request,Analysis,Generation,Points} from '../modules/phonation-synthesis/state.ts';
export interface Metadata {action:'analyze'|'apply'|'generate';analysis:Analysis;generation:Generation;alignment:'normalize'|'onset';point_count:number;controls:Points;source_f0:number[];target_f0:number[];source_samples:number;target_samples:number;continuum_type:1|2|3;reverse_direction:boolean;batch_id:string|null;batch_group_index:number;batch_group_count:1|6;analysis_job_id:string|null;inputs:Record<'source'|'target',{asset_id:string;sha256:string}>;runtime:Record<string,string>;algorithm:string}
export interface M07Result {job:string;files:components['schemas']['M07Manifest']['files'];metadata:Metadata}
export interface M07F0Display {kind:'synthesis-control-f0';alignment:'normalize'|'onset';curves:{name:string;axis:number[];values:number[]}[]}
export interface M07Transport {project:string;source(file:ResearchFile):Promise<{asset_id:string;sha256:string}>;create(body:Request):Promise<JobView>;job(id:string):Promise<JobView>;list():Promise<JobView[]>;cancel(id:string):Promise<unknown>;retry(id:string,key:string):Promise<JobView>;read(job:string,id:string,sha:string):Promise<ArrayBuffer>;f0?(job:string):Promise<M07F0Display>;save?(job:string,directory:string,name:string):Promise<unknown>}
export type Settings=Omit<Request,'project_id'|'idempotency_key'|'source'|'target'>;
export interface M07Port {run(source:ResearchFile,target:ResearchFile,settings:Settings,signal:AbortSignal,progress:(job:JobView)=>void):Promise<M07Result>;result(job:JobView):Promise<M07Result>;list():Promise<JobView[]>;cancel(id:string):Promise<unknown>;retry(id:string):Promise<JobView>;read(result:M07Result,name:string):Promise<ArrayBuffer>;f0?(result:M07Result):Promise<M07F0Display>;save(result:M07Result,directory:string):Promise<void>}
const messageKeys:Record<string,string>={"m07_analysis_stale": "platform.analysis_is_no_longer_valid_extract_f0_again", "m07_reaper_unavailable": "platform.reaper_is_unavailable_the_backend_was_not_changed", "m07_platform_unverified": "platform.phonation_synthesis_is_unavailable_on_this_platform_use_the", "m07_no_voiced_region": "platform.no_valid_voiced_region_check_the_recording_and_f0", "m07_insufficient_pulses": "platform.not_enough_valid_pulses_check_the_recording_or_pulse", "m07_f0_analysis_failed": "platform.f0_analysis_failed_the_input_may_be_too_short", "m07_input_budget": "platform.input_exceeds_the_10_second_480_000_frame_or", "m07_invalid_audio": "platform.audio_is_empty_damaged_or_contains_nonfinite_values", "m07_timeout": "platform.computation_timed_out_the_worker_process_was_reclaimed", "m07_execution_failed": "platform.scientific_computation_failed_the_current_group_was_not_published", "cancelled": "platform.task_cancelled_completed_groups_are_retained", "quota_exceeded": "platform.not_enough_free_space_completed_groups_are_retained", "input_unavailable": "platform.the_input_is_unavailable_or_nearly_expired_select_it", "m07_invalid_f0": "platform.invalid_f0_control_points_check_their_values"};
export function m07Message(error:unknown){const value=uiErrorMessage(error);return typeof value==='string'&&messageKeys[value]?uiMessage(messageKeys[value]):value;}
export const messages:Record<string,string>=localized(()=>Object.fromEntries(Object.entries(messageKeys).map(([code,key])=>[code,tr(key)])));
export function downloadM07(raw:ArrayBuffer,name:string){const url=URL.createObjectURL(new Blob([raw]));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
export function m07Port(t:M07Transport):M07Port{
 async function read(result:Pick<M07Result,'job'|'files'>,name:string){const f=result.files.find(f=>f.name===name);if(!f)throw uiError(joinMessages(uiMessage('platform.result_file_missing'),name));const raw=await t.read(result.job,f.id,f.sha256);if(raw.byteLength!==f.size_bytes||await sha256(raw)!==f.sha256)throw uiError(uiMessage('platform.result_file_verification_failed'));return raw;}
 async function result(job:JobView):Promise<M07Result>{if(job.state!=='succeeded'||job.result_manifest?.kind!=='managed_m07_files')throw uiError(job.error_code?m07Message(job.error_code):uiMessage('platform.the_current_group_has_not_completed_successfully'));const base={job:job.id,files:job.result_manifest.files};return {...base,metadata:JSON.parse(new TextDecoder().decode(await read(base,'m07.ptb.json')))};}
 return {read,result,f0:t.f0?r=>t.f0!(r.job):undefined,list:t.list,cancel:t.cancel,retry:id=>t.retry(id,crypto.randomUUID()),async save(r,directory){const name=taskDescription(r.metadata);if(t.save){await t.save(r.job,directory,name);return;}const files=[];for(const f of r.files)files.push({name:f.name,raw:await read(r,f.name)});downloadM07(m07Archive(files),name+'.zip');},async run(source,target,settings,signal,progress){
  const refs=await Promise.all([t.source(source),t.source(target)]);if(signal.aborted)throw uiError(uiMessage('platform.operation_cancelled'));
  let job=await t.create({...settings,project_id:t.project,idempotency_key:crypto.randomUUID(),source:refs[0],target:refs[1]});progress(job);
  const deadline=Date.now()+300000;
  while(['queued','running','cancel_requested'].includes(job.state)){
   if(signal.aborted||Date.now()>deadline){await t.cancel(job.id);throw uiError(uiMessage('platform.operation_cancelled_completed_groups_are_retained'));}
   await waitForTaskPoll(job.state,signal);if(signal.aborted)continue;job=await t.job(job.id);progress(job);
  }
  if(signal.aborted)throw uiError(uiMessage('platform.operation_cancelled_completed_groups_are_retained'));return result(job);
 }};
}
