<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import type { components } from '../../../contracts/generated/api';
import TaskPanel from '../components/TaskPanel.vue';
type Job = components['schemas']['JobView'];
type Event = components['schemas']['JobEvent'];
const props=defineProps<{projectId:string;ownerId:string;csrfToken:string}>();
const emit=defineEmits<{sessionInvalid:[]}>();
const jobs=ref<Job[]>([]), events=ref<Event[]>([]), message=ref(''), enabled=ref(false), busy=ref(false);
const abort=new AbortController(); let timer:ReturnType<typeof setTimeout>|undefined, disposed=false;
const pending=new Map<string,string>();
const tasks=computed(()=>jobs.value.map(j=>({id:j.id,title:'流程检查 · '+j.id.slice(0,8),status:j.state,progress:j.progress,
  error:j.error_code ? (j.error_code==='worker_interrupted'?'执行中断，可重新运行。':'任务未完成，可查看记录后重试。'):undefined,
  canCancel:['queued','running'].includes(j.state),canRetry:['failed','interrupted','cancelled'].includes(j.state)})));
async function request<T>(path:string,method='GET',body?:unknown):Promise<T> {
  const response=await fetch('/api/v1/'+path,{method,credentials:'same-origin',signal:abort.signal,
    headers:{'Content-Type':'application/json','X-PTB-Account':props.ownerId,...(method==='POST'?{'X-CSRF-Token':props.csrfToken}:{})},
    body:body===undefined?undefined:JSON.stringify(body)});
  const value=await response.json();
  if(!response.ok) {
    if(response.status===401 || value.detail==='account_changed') { jobs.value=[];events.value=[];emit('sessionInvalid'); }
    throw new Error(value.detail || 'unavailable');
  }
  return value;
}
async function refresh() {
  try {
    const result=await request<components['schemas']['JobList']>('jobs?project_id='+props.projectId);
    if(!disposed){jobs.value=result.jobs;message.value='';}
  } catch {if(!disposed)message.value='任务服务暂时无法连接，正在重新连接。已有任务不会因刷新而重复提交。';}
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
  } catch {if(!disposed)message.value='操作未确认，请稍后重试。同一次提交会保留请求标识，避免重复创建任务。';}
  finally {busy.value=false;}
}
async function showEvents(id:string) {
  try { const result=await request<components['schemas']['JobEvents']>(`jobs/${id}/events`);if(!disposed)events.value=result.events; }
  catch {if(!disposed)message.value='暂时无法读取任务记录。';}
}
const states:Record<string,string>={queued:'排队',running:'运行',cancel_requested:'请求取消',cancelled:'取消',failed:'失败',interrupted:'中断',succeeded:'完成'};
onMounted(async()=>{try {const caps=await request<components['schemas']['Capabilities']>('capabilities');if(disposed)return;enabled.value=caps.task_operations.includes('pipeline_check');if(enabled.value)void refresh();}catch {if(!disposed)message.value='任务服务尚未就绪。';}});
onUnmounted(()=>{disposed=true;abort.abort();clearTimeout(timer);pending.clear();});
</script>
<template>
  <section class="project-jobs" aria-label="项目任务">
    <h3>任务记录</h3><p class="muted">当前可检查任务流程：排队、执行、取消和恢复。语音分析将在模块接入后开放。</p>
    <button :disabled="busy || !enabled" @click="act('create')">运行流程检查</button>
    <p v-if="!enabled" class="hint">任务服务尚未启用。</p><p v-if="message" role="status">{{message}}</p>
    <TaskPanel :tasks="tasks" empty-title="暂无任务" empty-text="此项目还没有任务。流程检查不会上传或分析音频。" @cancel="act('cancel',$event)" @retry="act('retry',$event)"/>
    <div class="job-record-buttons"><button v-for="job in jobs" :key="job.id" @click="showEvents(job.id)">查看 {{job.id.slice(0,8)}} 的记录</button></div>
    <ol v-if="events.length" aria-label="任务运行记录"><li v-for="event in events" :key="event.sequence">{{states[event.state]}} · {{Math.round(event.progress*100)}}%</li></ol>
  </section>
</template>
<style scoped>
.project-jobs {margin-top:1.5rem;border-top:1px solid var(--border);padding-top:1rem;min-width:0}
.project-jobs :deep(.task-panel) {margin-top:1rem;display:flex;flex-direction:column;align-items:flex-start;gap:.5rem;font-size:13px;line-height:1.6}
.project-jobs :deep(.status-dot) {display:none}
.project-jobs :deep(.task-row) {display:flex;flex-wrap:wrap;gap:.6rem;align-items:center;padding:.7rem 0}
.job-record-buttons {display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.75rem}
</style>
