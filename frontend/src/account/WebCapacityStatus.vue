<script setup lang="ts">
import {messageRef,message as uiMessage,type MessageDescriptor} from '../i18n/index.ts';
import {localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {onMounted,onUnmounted,ref,watch} from 'vue';
import type {components} from '../../../contracts/generated/api';
import ModalDialog from '../components/ModalDialog.vue';
const props=defineProps<{ownerId:string;csrfToken:string}>();
const usage=ref<components['schemas']['StorageUsage']|null>(null);
const presence=ref<components['schemas']['PresenceView']|null>(null);
const server=ref<components['schemas']['ServerLoadView']|null>(null);
const application=ref<components['schemas']['CapacityRequestView']|null>(null);
const expanded=ref(false),loadExpanded=ref(false),amount=ref(1000),reason=ref(''),notice=messageRef(),busy=ref(false);
const duration=ref<'today'|'7d'|'30d'>('today');
const durations=localized(()=>({today:tr('account.today'), '7d':tr('account.7_days'), '30d':tr('account.30_days')}));
let timer:ReturnType<typeof setInterval>|undefined,loadTimer:ReturnType<typeof setInterval>|undefined,loading=false,loadLoading=false,disposed=false;
const abort=new AbortController();
const mb=(n:number)=>(n/1_000_000).toLocaleString('zh-CN',{maximumFractionDigits:1})+' MB';
async function request<T>(path:string,body?:unknown):Promise<T>{
 const r=await fetch('/api/v1/'+path,{method:body?'POST':'GET',credentials:'same-origin',cache:'no-store',signal:abort.signal,
  headers:{'X-PTB-Account':props.ownerId,...(body?{'Content-Type':'application/json','X-CSRF-Token':props.csrfToken}:{})},body:body?JSON.stringify(body):undefined});
 const value=await r.json();if(!r.ok)throw Error(value.detail);return value;
}
async function refresh(){
 if(loading||disposed||document.hidden)return;loading=true;
 try{const [u,p,a]=await Promise.all([request<components['schemas']['StorageUsage']>('storage/usage'),request<components['schemas']['PresenceView']>('workspaces/presence?include_load=false'),request<components['schemas']['CapacityRequestView']|null>('workspaces/capacity-request')]);
  if(!disposed){usage.value=u;presence.value=p;application.value=a;if(a && a.state!=='pending')notice.value='';}
 }catch{if(!disposed){presence.value=null;usage.value=null;}}finally{loading=false;}
}
async function refreshLoad(){
 if(loadLoading||disposed||document.hidden)return;loadLoading=true;const details=loadExpanded.value;
 try{const value=await request<components['schemas']['ServerLoadView']>('workspaces/server-load?details='+details);if(!disposed&&details===loadExpanded.value)server.value=value;}
 catch{if(!disposed&&details===loadExpanded.value)server.value=null;}
 finally{loadLoading=false;if(!disposed&&details!==loadExpanded.value)void refreshLoad();}
}
watch(loadExpanded,()=>void refreshLoad());
async function submit(){
 if(busy.value)return;busy.value=true;notice.value='';
 try{application.value=await request('workspaces/capacity-request',{requested_bytes:Math.round(amount.value*1_000_000),reason:reason.value.trim(),duration:duration.value});notice.value=uiMessage('account.your_request_was_submitted_and_is_awaiting_administrator_review');}
 catch(e){notice.value=({capacity_not_increase:uiMessage('account.the_requested_quota_must_exceed_your_current_quota'),capacity_request_limited:uiMessage('account.you_have_used_all_of_today_s_requests_try'),capacity_reason_required:uiMessage('account.describe_why_you_need_additional_storage')} as Record<string,MessageDescriptor>)[e instanceof Error?e.message:'']||uiMessage('account.your_request_has_not_been_confirmed_retry_shortly');}
 finally{busy.value=false;}
}
function visibility(){if(!document.hidden){void refresh();void refreshLoad();}}
onMounted(()=>{visibility();timer=setInterval(()=>void refresh(),30000);loadTimer=setInterval(()=>void refreshLoad(),1000);document.addEventListener('visibilitychange',visibility);});
onUnmounted(()=>{disposed=true;abort.abort();clearInterval(timer);clearInterval(loadTimer);document.removeEventListener('visibilitychange',visibility);});
</script>
<template>
 <aside class="web-capacity" :aria-label="tr('account.online_users_and_your_storage')">
  <span :title="tr('account.unique_accounts_with_a_heartbeat_in_the_last_90')">{{presence?tr('account.value_users_online',{v0:(presence.online_accounts)}):tr('account.online_count_unavailable')}}</span>
  <span v-if="usage">{{tr('account.server_storage_remaining')}} {{mb(usage.available_bytes)}} / {{mb(usage.quota_bytes)}} <small>{{tr('account.used_223')}} {{mb(usage.used_bytes)}} {{tr('account.task_reservations')}} {{mb(usage.reserved_bytes)}}</small></span>
  <button @click="expanded=true">{{tr('account.storage_and_quota_requests')}}</button>
  <button @click="loadExpanded=true">{{server?tr('account.computing_value_value_waiting_value',{v0:(server.running_tasks),v1:(server.task_slots),v2:(server.waiting_tasks)}):tr('account.server_status')}}</button>
  <span v-if="server && (server.waiting_tasks>0 || loadExpanded && ((server.cpu_percent??0)>=85 || (server.memory_percent??0)>=85))" role="status">{{tr('account.the_server_is_busy_submitted_tasks_are_waiting_there')}}</span>
  <span v-if="usage && usage.available_bytes<usage.quota_bytes*.2">{{tr('account.your_storage_is_nearly_full_download_needed_files_and')}}</span>
  <span v-if="usage?.server_space_low" role="status">{{tr('account.server_storage_is_running_low_download_results_and_remove')}}</span>
 </aside>
 <ModalDialog v-if="loadExpanded" class="capacity-dialog" :title="tr('account.server_status')" @close="loadExpanded=false">
  <template v-if="server">
   <div class="load-grid"><div><strong>{{server.running_tasks}} / {{server.task_slots}}</strong><span>{{tr('account.running_tasks_configured_slots')}}</span></div><div><strong>{{server.waiting_tasks}}</strong><span>{{tr('account.waiting_tasks')}}</span></div><div><strong>{{server.cpu_percent===null?tr('account.temporarily_unavailable'):server.cpu_percent+'%'}}</strong><span>{{tr('account.server_cpu_usage')}}</span></div><div><strong>{{server.memory_percent===null?tr('account.temporarily_unavailable'):server.memory_percent+'%'}}</strong><span>{{tr('account.server_memory_usage')}}</span></div></div>
   <p>{{tr('account.task_slots_are_the_configured_limit_on_simultaneous_background')}}</p>
   <p v-if="server.running_tasks>=server.task_slots || server.waiting_tasks>0">{{tr('account.computations_take_time_submitted_tasks_wait_for_scheduling_keep')}}</p>
   <p class="capacity-hint">{{tr('account.updated_every_second_cpu_usage_is_measured_between_samples')}}{{new Date(server.measured_at*1000).toLocaleTimeString('zh-CN')}}。</p>
  </template><p v-else role="status">{{tr('account.server_status_is_temporarily_unavailable_check_again_later')}}</p>
 </ModalDialog>
 <ModalDialog v-if="expanded" class="capacity-dialog" :title="tr('account.your_storage')" @close="expanded=false">
  <p v-if="usage" class="capacity-total">{{mb(usage.available_bytes)}} <small>{{tr('account.available_total_quota')}} {{mb(usage.quota_bytes)}}</small></p>
  <p>{{tr('account.server_storage_is_limited_download_results_you_need_and')}}</p>
  <p>{{tr('account.remove_temporary_server_files_under_project_files_server_copies')}}</p>
  <p class="capacity-hint">{{tr('account.only_server_inputs_results_temporary_files_and_task_reservations')}}</p>
  <p v-if="application">{{tr('account.latest_request')}}{{mb(application.requested_bytes)}} · {{durations[application.duration]}} · {{application.expires_at && Date.parse(application.expires_at)<=Date.now()?tr('account.expired'):({pending:tr('account.awaiting_review'),approved:tr('account.approved'),declined:tr('account.not_approved')})[application.state]}}<span v-if="application.granted_bytes">{{tr('account.approved_quota')}} {{mb(application.granted_bytes)}}</span><span v-if="application.granted_duration">{{tr('account.approved_duration')}} {{durations[application.granted_duration]}}</span><span v-if="application.expires_at">，{{new Date(application.expires_at).toLocaleString('zh-CN')}} {{tr('account.expires')}}</span><span v-if="application.decision_note"> {{tr('account.administrator_reply')}}{{application.decision_note}}</span></p>
  <form v-if="application?.state!=='pending'" @submit.prevent="submit">
   <label>{{tr('account.requested_total_quota_mb')}}<input v-model.number="amount" type="number" :min="usage?Math.floor(usage.quota_bytes/1_000_000)+1:1" max="10000" required /></label>
   <label>{{tr('account.requested_duration')}}<select v-model="duration"><option value="today">{{tr('account.today')}}</option><option value="7d">{{tr('account.7_days')}}</option><option value="30d">{{tr('account.30_days')}}</option></select></label>
   <label class="capacity-reason">{{tr('account.reason_for_additional_storage')}}<textarea v-model="reason" maxlength="500" required /></label>
   <p class="capacity-hint">{{tr('account.today_ends_at_midnight_beijing_time_on_the_approval')}}</p>
   <button :disabled="busy || !reason.trim()">{{tr('account.submit_storage_request')}}</button>
  </form>
  <p v-if="notice" role="status">{{notice}}</p>
 </ModalDialog>
</template>
<style scoped>
.web-capacity{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem 1.2rem;padding:.45rem 1rem;background:var(--panel);border-bottom:1px solid var(--border);font-size:.85rem}
small{color:var(--muted);margin-left:.5rem}.capacity-dialog{width:min(640px,calc(100vw / var(--page-scale,1) - 32px));border-radius:16px}.capacity-dialog :deep(.dialog-body){line-height:1.75;overflow-wrap:anywhere}.capacity-total{font-size:1.5rem;font-weight:600}.capacity-total small{font-size:.85rem;font-weight:400}.capacity-hint{font-size:.85rem;color:var(--muted)}form,label{display:grid;gap:.5rem}form{grid-template-columns:1fr 1fr;gap:1rem}form>p,form>button,.capacity-reason{grid-column:1/-1}input,textarea,select{width:100%;min-width:0;box-sizing:border-box}textarea{min-height:4rem;resize:vertical}
@media(max-width:440px){form{grid-template-columns:1fr}.capacity-total small{display:block;margin:0}}
.load-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:6px 0 20px}.load-grid>div{display:grid;gap:8px;padding:16px;background:var(--selected);border-radius:10px}.load-grid strong{font-size:1.45rem}.load-grid span{font-size:.85rem;color:var(--muted)}
</style>
