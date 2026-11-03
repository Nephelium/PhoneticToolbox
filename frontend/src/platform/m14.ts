import {uiMessage,uiError} from '../modules/module-localization.ts';
import {messageText,type MessageDescriptor} from '../i18n/index.ts';
import type {AssetFetch} from './local-asset-transport.ts';
import {waitForTaskPoll} from './task-poll.ts';
import {zipStore} from './zip-store.ts';
import type {M14Port,Source,Preview,Result,Options,Inspection} from '../modules/phonology-induction/port.ts';
import type {JobView} from './research.ts';
import {sha256} from './research.ts';
interface Transport {project:string;upload(file:File,signal:AbortSignal):Promise<Source>;create(body:unknown):Promise<JobView>;job(id:string):Promise<JobView>;cancel(id:string):Promise<unknown>;read(job:string,id:string,sha:string):Promise<ArrayBuffer>;save?(result:Result):Promise<boolean>}
const errors:Record<string,string>={"m14_timeout": "platform.export_exceeded_the_60_second_limit_and_stopped_split", "m14_unsupported_format": "platform.xlsx_xls_csv_tsv_txt_and_docx_tables_are", "m14_cell_output_budget": "platform.a_cell_in_the_homophone_matrix_exceeds_excel_s", "m14_invalid_cell": "platform.a_cell_is_too_long_or_contains_invalid_control", "m14_expanded_budget": "platform.expanded_excel_content_exceeds_the_reading_budget", "m14_column_budget": "platform.the_survey_table_exceeds_64_columns_keep_only_the", "m14_output_budget": "platform.the_three_results_exceed_the_16_mb_output_budget", "m14_execution_failed": "platform.document_generation_failed_edits_and_results_are_preserved_check", "m14_decode_failed": "platform.the_file_could_not_be_parsed_check_its_format_383", "m14_no_valid_rows": "platform.no_valid_records_check_the_column_numbers_start_row", "m14_empty_input": "platform.the_file_is_empty", "m14_missing_columns": "platform.the_survey_table_requires_at_least_headword_and_ipa", "m14_formula_input": "platform.the_survey_table_contains_formulas_convert_them_to_explicit", "m14_runtime_unavailable": "platform.survey_tables_cannot_be_processed_on_this_platform_use", "m14_input_budget": "platform.the_file_exceeds_the_2_mb_reading_budget", "m14_row_budget": "platform.the_survey_table_exceeds_10_000_rows", "m14_output_shape_budget": "platform.the_onset_rime_matrix_exceeds_20_000_cells_split", "input_unavailable": "platform.the_source_file_is_unavailable_changed_or_nearly_expired", "cancelled": "platform.operation_cancelled_edits_and_results_are_preserved", "execution_failed": "platform.the_task_failed_check_the_input_and_retry", "quota_exceeded": "platform.the_project_has_insufficient_space_clean_up_files_and", "m14_column_selection": "platform.choose_distinct_valid_columns_for_headword_ipa_and_notes", "m14_start_row": "platform.starting_row_must_be_an_integer_from_1_to", "m14_table_index": "platform.the_worksheet_or_table_does_not_exist_select_it", "m14_encoding": "platform.choose_a_supported_text_encoding", "m14_delimiter": "platform.choose_a_supported_delimiter", "m14_start_inside_record": "platform.the_start_row_is_inside_a_quoted_multiline_record"};
export const m14ErrorMessage=(code:string|MessageDescriptor)=>typeof code==='string'?(errors[code]?uiMessage(errors[code]):code):code;
export const m14Error=(code:string)=>messageText(m14ErrorMessage(code));
function download(raw:ArrayBuffer,name:string){const url=URL.createObjectURL(new Blob([raw]));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
export function m14Port(t:Transport):M14Port {
 async function wait(job:JobView,signal:AbortSignal){
  const deadline=Date.now()+180000;
  while(['queued','running','cancel_requested'].includes(job.state)){
   if(signal.aborted){await t.cancel(job.id);throw uiError(m14ErrorMessage('cancelled'));}
   if(Date.now()>deadline){await t.cancel(job.id);throw uiError(uiMessage('platform.waiting_for_the_task_timed_out_cancellation_was_requested'));}
   await waitForTaskPoll(job.state,signal);if(signal.aborted)continue;job=await t.job(job.id);
  }
  if(signal.aborted)throw uiError(m14ErrorMessage('cancelled'));
  if(job.state!=='succeeded')throw uiError(m14ErrorMessage(job.error_code??'execution_failed'));
  if(!job.result_manifest||!('files' in job.result_manifest))throw uiError(uiMessage('platform.the_result_manifest_is_incomplete_386'));return {job:job.id,files:job.result_manifest.files} as Result;
 }
 async function read(r:Result,id:string){const f=r.files.find(f=>f.id===id);if(!f)throw uiError(uiMessage('platform.results_are_no_longer_available'));const raw=await t.read(r.job,id,f.sha256);if(raw.byteLength!==f.size_bytes||await sha256(raw)!==f.sha256)throw uiError(uiMessage('platform.file_integrity_verification_failed'));return raw;}
 async function upload(file:File,signal:AbortSignal){if(!file.size)throw uiError(m14ErrorMessage('m14_empty_input'));if(file.size>2_000_000)throw uiError(m14ErrorMessage('m14_input_budget'));const source=await t.upload(file,signal);if(signal.aborted)throw uiError(m14ErrorMessage('cancelled'));return source;}
 async function json(source:Source,options:Options,action:'inspect'|'preview',signal:AbortSignal){const r=await wait(await t.create({project_id:t.project,idempotency_key:crypto.randomUUID(),table:{asset_id:source.asset_id,sha256:source.sha256},config:{action,...options}}),signal);return JSON.parse(new TextDecoder().decode(await read(r,r.files[0].id)));}
 async function analyze(source:Source,options:Options,signal:AbortSignal){const preview=await json(source,options,'preview',signal) as Preview;if(preview.input_sha256!==source.sha256)throw uiError(uiMessage('platform.input_checksums_do_not_match'));return preview;}
 return {
  async inspect(file,options,signal){const source=await upload(file,signal);return {source,inspection:await json(source,options,'inspect',signal) as Inspection};},
  async inspectSource(source,options,signal){return await json(source,options,'inspect',signal) as Inspection;},
  analyze,
  async import(file,options,signal){if(file.size===0)throw uiError(m14ErrorMessage('m14_empty_input'));if(file.size>2_000_000)throw uiError(m14ErrorMessage('m14_input_budget'));const source=await t.upload(file,signal);if(signal.aborted)throw uiError(m14ErrorMessage('cancelled'));const r=await wait(await t.create({project_id:t.project,idempotency_key:crypto.randomUUID(),table:{asset_id:source.asset_id,sha256:source.sha256},config:{action:'preview',...options}}),signal);const preview=JSON.parse(new TextDecoder().decode(await read(r,r.files[0].id))) as Preview;if(preview.input_sha256!==source.sha256)throw uiError(uiMessage('platform.input_checksums_do_not_match'));return {source,preview};},
  async generate(source,options,settings,font,signal){return wait(await t.create({project_id:t.project,idempotency_key:crypto.randomUUID(),table:{asset_id:source.asset_id,sha256:source.sha256},config:{action:'export',...options,settings,font}}),signal);},
  async save(result){if(t.save)return t.save(result);const files=[];for(const f of result.files)files.push({name:f.name,raw:await read(result,f.id)});download(zipStore(files),'同音字表_完整结果.zip');return true;},
  async download(result,id){const f=result.files.find(f=>f.id===id)!;download(await read(result,id),f.name);}
 };
}
export function serverM14(owner:string,project:string,csrf:()=>string,onInvalid:()=>void,signal:AbortSignal,assetFetch:AssetFetch=(path,init)=>fetch('/api/v1/'+path,init)):M14Port {
 async function request(path:string,method='GET',body?:unknown,raw?:ArrayBuffer){const r=await assetFetch(path,{method,signal,credentials:'same-origin',cache:'no-store',headers:{'X-PTB-Account':owner,...(method!=='GET'?{'X-CSRF-Token':csrf()}:{}),...(raw?{'Content-Type':'application/octet-stream'}:body?{'Content-Type':'application/json'}:{})},body:raw??(body?JSON.stringify(body):undefined)});if(!r.ok){if(r.status===401)onInvalid();const e=await r.json().catch(()=>({}));throw uiError(m14ErrorMessage(typeof e.detail==='string'?e.detail:uiMessage('platform.request_failed_check_the_input_and_project_state')));}return r;}
 return m14Port({project,
  async upload(file,abort){const bytes=await file.arrayBuffer(),hash=await sha256(bytes),a=await(await request('uploads','POST',{project_id:project,name:file.name,expected_bytes:bytes.byteLength,idempotency_key:'m14-'+crypto.randomUUID()})).json();for(let offset=0;offset<bytes.byteLength;offset+=262144){if(abort.aborted)throw uiError(m14ErrorMessage('cancelled'));await request(`uploads/${a.id}/blocks?offset=${offset}`,'PUT',undefined,bytes.slice(offset,offset+262144));}const ready=await(await request('uploads/'+a.id+'/finalize','POST',{sha256:hash})).json();return {asset_id:ready.id,sha256:ready.sha256,name:file.name};},
  async create(body){return(await request('jobs/m14/create','POST',body)).json();},async job(id){return(await request('jobs/'+id)).json();},async cancel(id){return(await request('jobs/'+id+'/cancel','POST')).json();},async read(_job,id){return(await request('assets/'+id+'/content')).arrayBuffer();}
 });
}
