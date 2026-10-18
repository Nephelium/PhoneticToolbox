<script setup lang="ts">
import {onMounted,onUnmounted,ref,watch} from 'vue';
import type {components} from '../../../contracts/generated/api';
import ModalDialog from '../components/ModalDialog.vue';
const props=defineProps<{ownerId:string;csrfToken:string}>();
const usage=ref<components['schemas']['StorageUsage']|null>(null);
const presence=ref<components['schemas']['PresenceView']|null>(null);
const server=ref<components['schemas']['ServerLoadView']|null>(null);
const application=ref<components['schemas']['CapacityRequestView']|null>(null);
const expanded=ref(false),loadExpanded=ref(false),amount=ref(1000),reason=ref(''),notice=ref(''),busy=ref(false);
const duration=ref<'today'|'7d'|'30d'>('today');
const durations={today:'当日', '7d':'7天', '30d':'30天'};
let timer:ReturnType<typeof setInterval>|undefined,loadTimer:ReturnType<typeof setInterval>|undefined,loading=false,loadLoading=false,disposed=false;
const abort=new AbortController();
const mb=(n:number)=>(n/1_000_000).toLocaleString('zh-CN',{maximumFractionDigits:1})+' MB';
async function request<T>(path:string,body?:unknown):Promise<T>{
 const r=await fetch('/api/v1/'+path,{method:body?'POST':'GET',credentials:'same-origin',cache:'no-store',signal:abort.signal,
  headers:{'X-PTB-Account':props.ownerId,...(body?{'Content-Type':'application/json','X-CSRF-Token':props.csrfToken}:{})},body:body?JSON.stringify(body):undefined});
 const value=await r.json();if(!r.ok)throw Error(value.detail);return value;
}
async function refresh(){
 if(loading||disposed)return;loading=true;
 try{const [u,p,a]=await Promise.all([request<components['schemas']['StorageUsage']>('storage/usage'),request<components['schemas']['PresenceView']>('workspaces/presence?include_load=false'),request<components['schemas']['CapacityRequestView']|null>('workspaces/capacity-request')]);
  if(!disposed){usage.value=u;presence.value=p;application.value=a;if(a && a.state!=='pending')notice.value='';}
 }catch{if(!disposed){presence.value=null;usage.value=null;}}finally{loading=false;}
}
async function refreshLoad(){
 if(loadLoading||disposed)return;loadLoading=true;const details=loadExpanded.value;
 try{const value=await request<components['schemas']['ServerLoadView']>('workspaces/server-load?details='+details);if(!disposed&&details===loadExpanded.value)server.value=value;}
 catch{if(!disposed&&details===loadExpanded.value)server.value=null;}
 finally{loadLoading=false;if(!disposed&&details!==loadExpanded.value)void refreshLoad();}
}
watch(loadExpanded,()=>void refreshLoad());
async function submit(){
 if(busy.value)return;busy.value=true;notice.value='';
 try{application.value=await request('workspaces/capacity-request',{requested_bytes:Math.round(amount.value*1_000_000),reason:reason.value.trim(),duration:duration.value});notice.value='申请已提交，等待管理员审核。';}
 catch(e){notice.value=({capacity_not_increase:'申请额度应高于当前额度。',capacity_request_limited:'今天的申请次数已用完，请明天再试。',capacity_reason_required:'请填写扩容用途。'} as Record<string,string>)[e instanceof Error?e.message:'']||'申请未确认，请稍后重试。';}
 finally{busy.value=false;}
}
onMounted(()=>{void refresh();void refreshLoad();timer=setInterval(()=>void refresh(),30000);loadTimer=setInterval(()=>void refreshLoad(),1000);});
onUnmounted(()=>{disposed=true;abort.abort();clearInterval(timer);clearInterval(loadTimer);});
</script>
<template>
 <aside class="web-capacity" aria-label="在线人数与个人容量">
  <span :title="'按最近90秒有心跳的账号去重，每30秒更新'">{{presence?`当前在线 ${presence.online_accounts} 人`:'在线人数暂不可用'}}</span>
  <span v-if="usage">服务器剩余 {{mb(usage.available_bytes)}} / {{mb(usage.quota_bytes)}} <small>已用 {{mb(usage.used_bytes)}} · 任务预留 {{mb(usage.reserved_bytes)}}</small></span>
  <button @click="expanded=true">容量与扩容</button>
  <button @click="loadExpanded=true">{{server?`计算 ${server.running_tasks} / ${server.task_slots} · 等待 ${server.waiting_tasks}`:'服务器状态'}}</button>
  <span v-if="server && (server.waiting_tasks>0 || loadExpanded && ((server.cpu_percent??0)>=85 || (server.memory_percent??0)>=85))" role="status">服务器繁忙，已提交的任务请等待，无需重复提交。</span>
  <span v-if="usage && usage.available_bytes<usage.quota_bytes*.2">个人空间即将用满，请及时下载并清理文件。</span>
  <span v-if="usage?.server_space_low" role="status">服务器可用空间较少，请及时下载结果并清理不再使用的文件。</span>
 </aside>
 <ModalDialog v-if="loadExpanded" class="capacity-dialog" title="服务器状态" @close="loadExpanded=false">
  <template v-if="server">
   <div class="load-grid"><div><strong>{{server.running_tasks}} / {{server.task_slots}}</strong><span>运行任务 / 配置槽位</span></div><div><strong>{{server.waiting_tasks}}</strong><span>等待任务</span></div><div><strong>{{server.cpu_percent===null?'暂不可用':server.cpu_percent+'%'}}</strong><span>服务器 CPU 使用率</span></div><div><strong>{{server.memory_percent===null?'暂不可用':server.memory_percent+'%'}}</strong><span>服务器内存使用率</span></div></div>
   <p>计算槽位是服务器同时执行后台任务的配置上限。在线人数和计算任务数不同，目前还没有确定可承诺的最高在线人数。</p>
   <p v-if="server.running_tasks>=server.task_slots || server.waiting_tasks>0">正在计算的任务需要一些时间。已提交的任务会等待调度，请保持网页连接，无需重复提交。</p>
   <p class="capacity-hint">每秒更新。CPU为两次采样间整机使用率，内存按整机可用内存计算。最近采样：{{new Date(server.measured_at*1000).toLocaleTimeString('zh-CN')}}。</p>
  </template><p v-else role="status">服务器状态暂不可用，请稍后查看。</p>
 </ModalDialog>
 <ModalDialog v-if="expanded" class="capacity-dialog" title="个人容量" @close="expanded=false">
  <p v-if="usage" class="capacity-total">{{mb(usage.available_bytes)}} <small>可用 / 总额度 {{mb(usage.quota_bytes)}}</small></p>
  <p>服务器存储空间有限，请及时下载需要保留的结果，清理不再使用的文件。</p>
  <p>服务器临时文件可在项目文件中清理。退出登录后清理服务器副本，关页或断线保留30分钟。本机文件继续保留。</p>
  <p class="capacity-hint">额度仅计算服务器上的输入、结果、临时文件和任务预留。本机文件使用浏览器存储。扩容由管理员审核，单文件限制与保留时间保持不变。</p>
  <p v-if="application">最近申请：{{mb(application.requested_bytes)}} · {{durations[application.duration]}} · {{application.expires_at && Date.parse(application.expires_at)<=Date.now()?'已到期':({pending:'待审核',approved:'已批准',declined:'未批准'})[application.state]}}<span v-if="application.granted_bytes">，批准额度 {{mb(application.granted_bytes)}}</span><span v-if="application.granted_duration">，批准期限 {{durations[application.granted_duration]}}</span><span v-if="application.expires_at">，{{new Date(application.expires_at).toLocaleString('zh-CN')}} 到期</span><span v-if="application.decision_note"> · 管理员回复：{{application.decision_note}}</span></p>
  <form v-if="application?.state!=='pending'" @submit.prevent="submit">
   <label>申请总额度（MB）<input v-model.number="amount" type="number" :min="usage?Math.floor(usage.quota_bytes/1_000_000)+1:1" max="10000" required /></label>
   <label>申请期限<select v-model="duration"><option value="today">当日</option><option value="7d">7天</option><option value="30d">30天</option></select></label>
   <label class="capacity-reason">扩容用途<textarea v-model="reason" maxlength="500" required /></label>
   <p class="capacity-hint">当日截至审批当天北京时间24点，7天或30天从批准时起算。到期恢复常规额度，超额时暂停新增写入，已有文件按工作区期限保留。</p>
   <button :disabled="busy || !reason.trim()">提交扩容申请</button>
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
