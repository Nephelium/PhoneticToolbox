import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import type {AssetFetch} from './local-asset-transport.ts';
import type {M11Port,CorpusItem,Catalog,Grant} from '../modules/mfa/port.ts';
import type {JobView} from './research.ts';
import {sha256} from './research.ts';
import {pairFiles} from '../modules/mfa/state.ts';

const messages:Record<string,string>=localized(()=>({
 m11_download_not_uploaded:tr('platform.the_server_does_not_yet_provide_this_component_version'),
 m11_download_failed:tr('platform.component_download_failed_check_your_connection_and_retry_or'),
 m11_download_incomplete:tr('platform.the_component_download_is_incomplete_retry_to_resume_it'),
 m11_download_resume_rejected:tr('platform.the_server_cannot_resume_this_download_download_the_complete'),
 m11_delivery_folder_required:tr('platform.select_the_root_folder_extracted_from_the_component_archive'),
 m11_folder_files_changed:tr('platform.the_component_folder_has_missing_or_extra_files_extract'),
 m11_file_hash_mismatch:tr('platform.component_files_are_damaged_or_modified_download_the_original'),
 m11_component_busy:tr('platform.another_mfa_component_is_being_installed_wait_for_it'),
 m11_component_disk_full:tr('platform.not_enough_installation_space_free_disk_space_and_retry'),
 m11_runtime_missing:tr('platform.the_runtime_no_longer_exists_select_and_check_it'),m11_model_missing:tr('platform.a_registered_model_or_dictionary_no_longer_exists_check'),m11_version_mismatch:tr('platform.mfa_3_3_8_is_currently_supported_select_this'),m11_bootstrap_missing:tr('platform.mfa_alignment_cannot_start_right_now_you_can_report'),
 m11_install_path_too_long:tr('platform.the_component_path_is_too_long_for_reliable_loading'),
 m11_transcript_tiers:tr('platform.the_textgrid_has_no_usable_transcription_tier_or_has'),
 m11_no_alignments:tr('platform.the_current_model_and_default_search_parameters_did_not'),
 m11_oov_words:tr('platform.the_transcription_contains_words_missing_from_the_dictionary_check'),
 m11_model_mismatch:tr('platform.the_acoustic_model_and_dictionary_use_incompatible_phone_sets'),m11_dictionary_mismatch:tr('platform.the_dictionary_format_or_content_is_unsuitable_for_the'),
 m11_waiting_verified_node:tr('platform.waiting_for_an_available_computation_service'),
 m11_web_disabled:tr('platform.online_mfa_alignment_is_not_available_in_the_web'),
 m11_download_not_published:tr('platform.the_component_cannot_currently_be_downloaded_online_import_an'),
 m11_timeout:tr('platform.alignment_reached_the_180_second_limit_this_task_s'),
 m11_component_not_registered:tr('platform.the_selected_component_or_model_is_not_registered_check'),
 m11_temp_budget:tr('platform.temporary_files_reached_the_current_512_mb_limit_and'),
 m11_audio_budget:tr('platform.the_current_single_file_limit_is_120_seconds_and'),
 m11_self_test_required:tr('platform.this_component_requires_a_real_small_task_check'),m11_feature_failed:tr('platform.mfa_feature_extraction_failed_check_the_runtime_and_audio'),
 m11_execution_failed:tr('platform.mfa_execution_failed_inputs_and_configuration_are_retained_check'),
 m11_probe_word_unavailable:tr('platform.this_dictionary_has_no_a_or_a1_entry_needed'),
 m11_process_crashed:tr('platform.the_mfa_subprocess_exited_unexpectedly_this_task_s_process'),m11_runtime_changed:tr('platform.the_runtime_has_changed_check_it_again'),
 m11_model_changed:tr('platform.the_model_or_dictionary_has_changed_check_it_again'),m11_missing_transcript:tr('platform.the_audio_has_no_existing_transcription'),
 m11_manifest_hash_mismatch:tr('platform.the_offline_manifest_checksum_differs_from_the_independently_trusted'),
 m11_archive_hash_mismatch:tr('platform.the_component_archive_checksum_does_not_match_the_current'),m11_component_failed:tr('platform.component_checking_or_installation_failed_the_previous_working_version'),
 progress:tr('platform.task_stage_update'),failed:tr('platform.task_failed'),
 cancelled:tr('platform.task_cancelled_partial_local_outputs_are_retained_no_complete'),
 m11_preparing:tr('platform.preparing_audio_and_transcription'),m11_runtime_checked:tr('platform.external_runtime_check_passed'),m11_aligning:tr('platform.mfa_alignment_in_progress'),m11_exporting:tr('platform.generating_textgrid'),m11_complete:tr('platform.textgrid_generated'),queued:tr('platform.queued'),running:tr('platform.task_running'),succeeded:tr('platform.complete_results_published'),cancel_requested:tr('platform.stopping_task'),
}));
export const m11Message=(code:string)=>messages[code]??code;
interface Transport {
 log?:M11Port['log'];
 local:boolean;project:string;catalog():Promise<Catalog>;history():Promise<JobView[]>;
 upload(file:File,role:string,signal:AbortSignal):Promise<{asset_id:string;sha256:string}>;
 create(body:unknown):Promise<JobView>;job(id:string):Promise<JobView>;cancel(id:string):Promise<unknown>;
 events(id:string):ReturnType<M11Port['events']>;read(job:string,id:string,sha:string):Promise<ArrayBuffer>;
 save?:M11Port['save'];chooseOutput?:M11Port['chooseOutput'];pick?:M11Port['pick'];corpus?:M11Port['corpus'];component?:M11Port['component'];dictionaryFromGrant?:M11Port['dictionaryFromGrant'];
}
export function m11Port(t:Transport):M11Port {
 return {local:t.local,log:t.log,catalog:t.catalog,history:t.history,job:t.job,cancel:t.cancel,events:t.events,
  async import(files,signal,source){const pairs=pairFiles(files,source),items:CorpusItem[]=[];for(const p of pairs){if(signal.aborted)throw Error('cancelled');const audio=await t.upload(p.audio,'audio',signal),transcript=await t.upload(p.transcript,'transcript',signal);items.push({name:p.name,audio,transcript,transcript_format:/\.textgrid$/i.test(p.transcript.name)?'.TextGrid':/\.txt$/i.test(p.transcript.name)?'.txt':'.lab'});}return items;},
  dictionary:(file,signal)=>t.upload(file,'dictionary',signal),dictionaryFromGrant:t.dictionaryFromGrant,
  create:body=>t.create({...body,project_id:t.project,idempotency_key:crypto.randomUUID(),schema_version:'m11/1'}),
  async download(job,id){const m=job.result_manifest;if(!m||!('files'in m))throw uiError(uiMessage('platform.results_are_unavailable'));const f=m.files.find(f=>f.id===id);if(!f)throw uiError(uiMessage('platform.results_are_unavailable'));const raw=await t.read(job.id,id,f.sha256);if(raw.byteLength!==f.size_bytes||await sha256(raw)!==f.sha256)throw uiError(uiMessage('platform.result_integrity_verification_failed'));const url=URL.createObjectURL(new Blob([raw]));const a=document.createElement('a');a.href=url;a.download=f.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);},
  save:t.save,chooseOutput:t.chooseOutput,pick:t.pick,corpus:t.corpus,component:t.component};
}
export function serverM11(owner:string,project:string,csrf:()=>string,onInvalid:()=>void,signal:AbortSignal,assetFetch:AssetFetch=(path,init)=>fetch('/api/v1/'+path,init)):M11Port {
 async function request(path:string,method='GET',body?:unknown,raw?:ArrayBuffer){const r=await assetFetch(path,{method,signal,credentials:'same-origin',cache:'no-store',headers:{'X-PTB-Account':owner,...(method!=='GET'?{'X-CSRF-Token':csrf()}:{}),...(raw?{'Content-Type':'application/octet-stream'}:body?{'Content-Type':'application/json'}:{})},body:raw??(body?JSON.stringify(body):undefined)});if(!r.ok){if(r.status===401)onInvalid();const e=await r.json().catch(()=>({}));throw uiError(m11Message(typeof e.detail==='string'?e.detail:uiMessage('platform.request_failed')));}return r;}
 return m11Port({local:false,project,catalog:async()=> (await request('jobs/m11/catalog')).json(),history:async()=> (await (await request('jobs?project_id='+project)).json()).jobs.filter((j:JobView)=>j.operation==='mfa_alignment'),
  async upload(file,_role,abort){const bytes=await file.arrayBuffer(),hash=await sha256(bytes),a=await(await request('uploads','POST',{project_id:project,name:file.name,expected_bytes:bytes.byteLength,idempotency_key:'m11-'+crypto.randomUUID()})).json();for(let offset=0;offset<bytes.byteLength;offset+=262144){if(abort.aborted)throw Error('cancelled');await request(`uploads/${a.id}/blocks?offset=${offset}`,'PUT',undefined,bytes.slice(offset,offset+262144));}const ready=await(await request('uploads/'+a.id+'/finalize','POST',{sha256:hash})).json();return {asset_id:ready.id,sha256:ready.sha256};},
  create:async body=>(await request('jobs/m11/create','POST',body)).json(),job:async id=>(await request('jobs/'+id)).json(),cancel:async id=>(await request('jobs/'+id+'/cancel','POST')).json(),events:async id=>(await request('jobs/'+id+'/events')).json(),read:async(_job,id)=>(await request('assets/'+id+'/content')).arrayBuffer()});
}
