import {joinMessages} from '../modules/module-localization.ts';
import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {waitForTaskPoll} from './task-poll.ts';
import {exportParams} from '../modules/speech-synthesis/state.ts';
import type {JobView,ResearchFile} from './research.ts';
import {sha256} from './research.ts';
import type {M06Port,Result} from '../modules/speech-synthesis/port.ts';
export interface M06Transport {project:string;parameters(text:string,signal:AbortSignal):Promise<{asset_id:string;sha256:string}>;source(file:ResearchFile):Promise<{asset_id:string;sha256:string}>;create(body:unknown):Promise<JobView>;job(id:string):Promise<JobView>;cancel(id:string):Promise<unknown>;read(job:string,id:string,sha:string):Promise<ArrayBuffer>;save?(job:string):Promise<boolean>}
export function downloadBytes(bytes:BlobPart,name:string){const url=URL.createObjectURL(new Blob([bytes]));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
export function m06Port(t:M06Transport):M06Port {
 async function read(r:Pick<Result,'job'|'files'>,name:string){const f=r.files.find(f=>f.name===name);if(!f)throw uiError(joinMessages(uiMessage('platform.synthesis_results_are_missing'),name));const raw=await t.read(r.job,f.id,f.sha256);if(raw.byteLength!==f.size_bytes||await sha256(raw)!==f.sha256)throw uiError(uiMessage('platform.synthesis_artifact_verification_failed'));return raw;}
 return {cancel:t.cancel,
 async run(action,config,file,signal,progress){
  const audio=action==='extract'&&file?await t.source(file):undefined;if(signal.aborted)throw uiError(uiMessage('platform.operation_cancelled'));
  const text=exportParams(config);if(new TextEncoder().encode(text).length>2_000_000)throw uiError(uiMessage('platform.the_full_parameters_exceed_2_mb_257'));
  const parameters=await t.parameters(text,signal);if(signal.aborted)throw uiError(uiMessage('platform.operation_cancelled'));
  let job=await t.create({project_id:t.project,idempotency_key:crypto.randomUUID(),action,parameters,audio});progress(job);
  const deadline=Date.now()+180000;
  while(['queued','running','cancel_requested'].includes(job.state)){
   if(signal.aborted||Date.now()>deadline){await t.cancel(job.id);throw uiError(signal.aborted?uiMessage('platform.operation_cancelled'):uiMessage('platform.the_wait_timed_out_cancellation_was_requested'));}
   await waitForTaskPoll(job.state,signal);if(signal.aborted)continue;job=await t.job(job.id);progress(job);
  }
  if(signal.aborted)throw uiError(uiMessage('platform.operation_cancelled'));
  if(job.state!=='succeeded')throw uiError(m06Error(job.error_code));
  if(!job.result_manifest||!('files' in job.result_manifest))throw uiError(uiMessage('platform.the_result_manifest_is_incomplete_259'));
  const base={job:job.id,files:job.result_manifest.files};
  const result:Result={...base,metadata:JSON.parse(new TextDecoder().decode(await read(base,'m06.ptb.json')))};
  if(action==='synthesize')result.wav=await read(result,'synthesis.wav');return result;
 },async download(result,name){if(t.save&&name==='synthesis.wav'){if(!await t.save(result.job))throw uiError(uiMessage('platform.not_saved_synthesis_artifacts_are_still_retained'));return;}downloadBytes(await read(result,name),name);}};
}
function m06Error(code?:string|null){const messages={
 m06_reaper_unavailable:uiMessage('platform.no_verified_reaper_is_available_in_this_environment_choose'),
 m06_reaper_failed:uiMessage('platform.reaper_extraction_failed_original_parameters_are_preserved_adjust_the'),
 m06_removed_synthesis_method:uiMessage('platform.this_file_uses_a_removed_resynthesis_method_use_a'),
 m06_removed_f0_method:uiMessage('platform.harvest_has_been_removed_choose_praat_or_reaper'),
 } as Record<string,ReturnType<typeof uiMessage>>;return code&&messages[code]||code||uiMessage('platform.the_synthesis_task_failed');}
