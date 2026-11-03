<script setup lang="ts">
import {messageRef,message as uiMessage,type MessageDescriptor} from '../i18n/index.ts';
import {localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import { computed, onMounted, onUnmounted, ref } from 'vue';
import type { components } from '../../../contracts/generated/api';
import TaskPanel from '../components/TaskPanel.vue';
type Job = components['schemas']['JobView'];
type Event = components['schemas']['JobEvent'];
const props=defineProps<{projectId:string;ownerId:string;csrfToken:string}>();
const emit=defineEmits<{sessionInvalid:[]}>();
const jobs=ref<Job[]>([]), events=ref<Event[]>([]), message=messageRef(), enabled=ref(false), busy=ref(false);
const abort=new AbortController(); let timer:ReturnType<typeof setTimeout>|undefined, disposed=false;
const pending=new Map<string,string>();
const titles:Record<string,string>=localized(()=>({pipeline_check:tr('account.workflow_check'),storage_check:tr('account.storage_workflow_check'),archive_zip:tr('account.create_zip_archive'),extract_zip:tr('account.extract_zip_archive'),lpc_analysis:tr('account.lpc_analysis'),egg_analysis:tr('account.egg_analysis'),acoustic_analysis:tr('account.parameter_estimation'),textgrid_segment:tr('account.split_by_annotations')}));
const jobErrors:Record<string,string>=localized(()=>({worker_interrupted:tr('account.execution_was_interrupted_you_can_run_it_again'),quota_exceeded:tr('account.there_is_not_enough_account_storage_incomplete_outputs_will'),output_budget_exceeded:tr('account.the_output_limit_for_this_batch_was_exceeded_adjust'),archive_rejected:tr('account.the_zip_is_unsupported_or_failed_validation_check_its'),input_unavailable:tr('account.the_input_has_expired_or_is_unavailable_upload_it'),resource_removed:tr('account.related_files_were_deleted_and_a_request_to_stop'),cancelled:tr('account.the_task_was_cancelled')}));
const tasks=computed(()=>jobs.value.map(j=>({id:j.id,title:titles[j.operation]||tr('account.processing_task'),status:j.state,progress:j.progress,
  error:j.error_code ? (jobErrors[j.error_code] || tr('account.the_task_did_not_finish_inspect_its_record_before')):undefined,
  canCancel:['queued','running'].includes(j.state),canRetry:['failed','interrupted','cancelled'].includes(j.state)})));
async function request<T>(path:string,method='GET',body?:unknown):Promise<T> {
  const response=await fetch('/api/v1/'+path,{method,credentials:'same-origin',signal:abort.signal,
    headers:{'Content-Type':'application/json','X-PTB-Account':props.ownerId,...(method==='POST'?{'X-CSRF-Token':props.csrfToken}:{})},
    body:body===undefined?undefined:JSON.stringify(body)});
  const value=await response.json();
  if(!response.ok) {
    if(!disposed&&!abort.signal.aborted&&(response.status===401 || value.detail==='account_changed')) { jobs.value=[];events.value=[];emit('sessionInvalid'); }
    throw new Error(value.detail || 'unavailable');
  }
  return value;
}
async function refresh() {
  try {
    const result=await request<components['schemas']['JobList']>('jobs?project_id='+props.projectId);
    if(!disposed){jobs.value=result.jobs;message.value='';}
  } catch {if(!disposed)message.value=uiMessage('account.the_task_service_is_temporarily_unreachable_reconnecting_refreshing_will');}
  finally {if(!disposed)timer=setTimeout(refresh,1500);}
}
async function act(action:'create'|'cancel'|'retry',id='') {
  if(busy.value)return;busy.value=true;message.value='';
  const slot=action+id;
  if(!pending.has(slot))pending.set(slot,crypto.randomUUID());
  try {
    const path=action==='create'?'jobs':`jobs/${id}/${action}`;
    const body=action==='create'?{project_id:props.projectId,idempotency_key:pending.get(slot),operation:'pipeline_check',config:{sample_count:4096,seed:0}}
      : action==='retry'?{idempotency_key:pending.get(slot)}:undefined;
    const job=await request<Job>(path,'POST',body);
    if(!disposed){jobs.value=[job,...jobs.value.filter(j=>j.id!==job.id)];pending.delete(slot);}
  } catch {if(!disposed)message.value=uiMessage('account.the_operation_has_not_been_confirmed_retry_shortly');}
  finally {busy.value=false;}
}
async function showEvents(id:string) {
  try { const result=await request<components['schemas']['JobEvents']>(`jobs/${id}/events`);if(!disposed)events.value=result.events; }
  catch {if(!disposed)message.value=uiMessage('account.task_records_are_temporarily_unavailable');}
}
const states:Record<string,string>=localized(()=>({queued:tr('account.queued'),running:tr('account.running'),cancel_requested:tr('account.cancellation_requested'),cancelled:tr('account.cancelled'),failed:tr('account.failed'),interrupted:tr('account.interrupted'),succeeded:tr('account.completed')}));
onMounted(async()=>{try {const caps=await request<components['schemas']['Capabilities']>('capabilities');if(disposed)return;enabled.value=caps.task_operations.length>0;void refresh();}catch {if(!disposed)message.value=uiMessage('account.the_task_service_is_not_ready_yet');}});
onUnmounted(()=>{disposed=true;abort.abort();clearTimeout(timer);pending.clear();});
</script>
<template>
  <section class="project-jobs" :aria-label="tr('account.project_tasks')">
    <h3>{{tr('account.task_history')}}</h3><p class="muted">{{tr('account.view_task_progress_and_processing_records_for_this_project')}}</p>
    <p v-if="!enabled" class="hint">{{tr('account.the_task_service_has_not_been_enabled')}}</p><p v-if="message" role="status">{{message}}</p>
    <TaskPanel :tasks="tasks" :empty-title="tr('account.no_tasks_yet')" :empty-text="tr('account.this_project_has_no_tasks_yet')" @cancel="act('cancel',$event)" @retry="act('retry',$event)"/>
    <div class="job-record-buttons"><button v-for="job in jobs" :key="job.id" @click="showEvents(job.id)">{{tr('account.view')}} {{job.id.slice(0,8)}} {{tr('account.records')}}</button></div>
    <ol v-if="events.length" :aria-label="tr('account.task_execution_record')"><li v-for="event in events" :key="event.sequence">{{states[event.state]}} · {{Math.round(event.progress*100)}}%</li></ol>
  </section>
</template>
<style scoped>
.project-jobs {margin-top:1.5rem;border-top:1px solid var(--border);padding-top:1rem;min-width:0}
.project-jobs :deep(.task-panel) {margin-top:1rem;display:flex;flex-direction:column;align-items:flex-start;gap:.5rem;font-size:0.928571rem;line-height:1.6}
.project-jobs :deep(.status-dot) {display:none}
.project-jobs :deep(.task-row) {display:flex;flex-wrap:wrap;gap:.6rem;align-items:center;padding:.7rem 0}
.job-record-buttons {display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.75rem}
</style>
