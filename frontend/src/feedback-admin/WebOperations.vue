<script setup lang="ts">
import {computed,onMounted,onUnmounted,ref,watch} from 'vue';
import {feedbackCall} from '../platform/feedback.ts';
import ModalDialog from '../components/ModalDialog.vue';
const props=defineProps<{active:boolean}>();
type Duration='today'|'7d'|'30d';
type State='pending'|'approved'|'declined';
interface Application{id:string;owner_id:string;username:string;reason:string;requested_bytes:number;quota_bytes:number;used_bytes:number;reserved_bytes:number;duration:Duration;state:State;created_at:string;granted_bytes:number|null;granted_duration:Duration|null;decision_note:string|null;expires_at:string|null}
interface Status{measured_at:number;online_accounts:number;pending_requests:number;default_quota_bytes:number;disk_total_bytes:number;disk_free_bytes:number;disk_safety_bytes:number;used_bytes:number;reserved_bytes:number;cleanup_pending:number;cleanup_failed:number;load:{running_tasks:number;waiting_tasks:number;task_slots:number;cpu_percent:number|null;memory_percent:number|null}|null}
const status=ref<Status|null>(null),rows=ref<Application[]>([]),next=ref<string|null>(null),filter=ref<State>('pending'),selected=ref(''),error=ref(''),notice=ref(''),loading=ref(false),busy=ref(false),amount=ref(2000),duration=ref<Duration>('7d'),note=ref(''),quota=ref(1000),quotaScope=ref('default'),now=ref(Date.now());
const confirmation=ref<'approve'|'decline'|'quota'|null>(null);
const requestsLoaded=ref(false);
const current=computed(()=>rows.value.find(r=>r.id===selected.value));
const stale=computed(()=>!status.value||now.value-status.value.measured_at*1000>60000);
const durations={today:'当日', '7d':'7天','30d':'30天'},states={pending:'待审批',approved:'已批准',declined:'已拒绝'};
const gb=(n:number)=>(n/1e9).toLocaleString('zh-CN',{maximumFractionDigits:2})+' GB';
const percent=(n:number|null|undefined)=>n==null?'暂不可用':n.toFixed(1)+'%';
let timer:ReturnType<typeof setInterval>,disposed=false,epoch=0;
function choose(row:Application){selected.value=row.id;amount.value=row.requested_bytes/1e6;duration.value=row.duration;note.value='';quotaScope.value='default';}
async function requests(append=false){
 const version=++epoch,state=filter.value;
 const result=await feedbackCall<{requests:Application[];next_cursor:string|null}>('webRequests',{state,before:append?next.value:null});
 if(disposed||version!==epoch)return;
 rows.value=append?[...rows.value,...result.requests]:result.requests;next.value=result.next_cursor;
 requestsLoaded.value=true;
 if(!rows.value.some(r=>r.id===selected.value)){selected.value='';if(rows.value[0])choose(rows.value[0]);}
}
async function refresh(){
 if(loading.value||busy.value||confirmation.value||disposed)return;loading.value=true;
 try{const snapshot=await feedbackCall<Status>('webStatus');if(disposed)return;status.value=snapshot;await requests();error.value='';}
 catch(e){if(!disposed)error.value=(e as Error).message;}
 finally{loading.value=false;}
}
async function more(){if(loading.value||!next.value)return;loading.value=true;try{await requests(true);}catch(e){error.value=(e as Error).message;}finally{loading.value=false;}}
async function decide(){
 const action=confirmation.value,row=current.value;if(!action||busy.value)return;
 busy.value=true;error.value='';notice.value='';
 try{
  if(action==='quota'){
   if(quotaScope.value==='account'&&!row)throw Error('请先选择一个账号的申请。');
   await feedbackCall('webQuota',{quota_bytes:Math.round(quota.value*1e6),owner_id:quotaScope.value==='account'?row!.owner_id:null});
   notice.value='常规额度已更新，已有文件保留。';
  }else{
   if(!row||row.state!=='pending')throw Error('申请状态已变化，请刷新。');
   await feedbackCall('webDecision',{request_id:row.id,granted_bytes:action==='approve'?Math.round(amount.value*1e6):null,duration:duration.value,note:note.value.trim()});
   notice.value=action==='approve'?'已批准扩容，回复已同步给用户。':'已拒绝申请，回复已同步给用户。';
  }
  confirmation.value=null;
 }catch(e){error.value=(e as Error).message;confirmation.value=null;}
 finally{const failure=error.value;busy.value=false;await refresh();if(failure)error.value=failure;}
}
watch(()=>props.active,value=>{if(value)void refresh();});watch(filter,()=>{selected.value='';rows.value=[];requestsLoaded.value=false;void refresh();});
onMounted(()=>{if(props.active)void refresh();timer=setInterval(()=>{now.value=Date.now();if(props.active&&!document.hidden)void refresh();},30000);});
onUnmounted(()=>{disposed=true;epoch++;clearInterval(timer);});
</script>
<template>
<section class="web-operations" aria-label="网页审批与服务器监控">
 <header class="ops-heading"><div><h2>网页运行情况</h2><p>{{status?'最近采样 '+new Date(status.measured_at*1000).toLocaleTimeString('zh-CN'):'等待服务器数据'}} · 每30秒更新</p></div><button :disabled="loading||busy" @click="refresh">刷新状态</button></header>
 <p v-if="error" class="ops-error" role="alert">{{error}}</p><p v-if="notice" class="ops-notice" role="status">{{notice}}</p>
 <p v-if="status && (stale||error)" role="status">数据可能已过期，请恢复连接后再审批。</p>
 <div v-if="status" class="ops-metrics">
  <article><span>CPU / 内存</span><strong>{{percent(status.load?.cpu_percent)}} / {{percent(status.load?.memory_percent)}}</strong></article>
  <article><span>磁盘可用 / 总量</span><strong>{{gb(status.disk_free_bytes)}} / {{gb(status.disk_total_bytes)}}</strong><small>安全预留 {{gb(status.disk_safety_bytes)}}</small></article>
  <article><span>当前在线</span><strong>{{status.online_accounts}} 人</strong><small>按账号去重</small></article>
  <article><span>运行 / 可同时处理 · 排队</span><strong>{{status.load?`${status.load.running_tasks} / ${status.load.task_slots} · ${status.load.waiting_tasks}`:'暂不可用'}}</strong></article>
  <article><span>用户已用 / 任务预留</span><strong>{{gb(status.used_bytes)}} / {{gb(status.reserved_bytes)}}</strong></article>
  <article><span>待清理 / 清理失败</span><strong>{{status.cleanup_pending}} / {{status.cleanup_failed}}</strong><small>失败项目由服务重试</small></article>
 </div>
 <p v-if="status && status.disk_free_bytes-status.reserved_bytes<=status.disk_safety_bytes+1e9" class="ops-error">服务器空间偏低，请先处理存储余量，再批准扩容。</p>
 <div class="ops-approval"><aside><h2>扩容申请 <small v-if="status">{{status.pending_requests}} 待审批</small></h2><label>申请状态<select v-model="filter" :disabled="busy||loading"><option value="pending">待审批</option><option value="approved">已批准</option><option value="declined">已拒绝</option></select></label>
  <div class="ops-requests"><button v-for="row in rows" :key="row.id" :class="{chosen:selected===row.id}" @click="choose(row)" :disabled="busy"><strong>{{row.username}}</strong><span>{{gb(row.requested_bytes)}} · {{durations[row.duration]}}</span></button><p v-if="!rows.length&&!loading&&requestsLoaded&&!error">暂无{{states[filter]}}申请。</p><p v-else-if="!rows.length&&!loading">申请列表尚未加载，请连接后刷新。</p><button v-if="next" :disabled="loading" @click="more">加载更多</button></div>
 </aside><section v-if="current" class="ops-detail"><h3>{{current.username}}</h3><p class="ops-reason">{{current.reason}}</p><p>当前额度 {{gb(current.quota_bytes)}} · 已用 {{gb(current.used_bytes)}} · 预留 {{gb(current.reserved_bytes)}}</p><p>申请 {{gb(current.requested_bytes)}} · {{durations[current.duration]}} · {{new Date(current.created_at).toLocaleString('zh-CN')}}</p>
  <form v-if="current.state==='pending'" @submit.prevent="confirmation='approve'"><label>批准总额度（MB）<input v-model.number="amount" type="number" min="100" max="10000" required :disabled="busy"/></label><label>批准期限<select v-model="duration" :disabled="busy"><option value="today">当日</option><option value="7d">7天</option><option value="30d">30天</option></select></label><label class="ops-note">给用户的回复<textarea v-model="note" maxlength="500" :disabled="busy"/></label><small class="ops-note">当日截至审批当天北京时间24点，7天或30天从批准时起算。扩容不会延长文件保留时间。</small><div class="ops-actions"><button type="submit" class="primary" :disabled="busy||stale||!!error">批准扩容</button><button type="button" :disabled="busy||stale||!!error" @click="confirmation='decline'">拒绝申请</button></div></form>
  <template v-else><p>{{states[current.state]}} <span v-if="current.granted_bytes">· {{gb(current.granted_bytes)}} · {{current.granted_duration?durations[current.granted_duration]:''}}</span></p><p v-if="current.expires_at">到期：{{new Date(current.expires_at).toLocaleString('zh-CN')}}</p><p class="ops-reason">回复：{{current.decision_note||'无留言'}}</p></template>
 </section><section v-else class="ops-detail"><p>选择申请，查看用途并回复用户。</p></section></div>
 <details v-if="status" class="ops-quota"><summary>调整常规额度</summary><p>当前默认 {{status?gb(status.default_quota_bytes):'暂不可用'}}。调低后保留已有文件，超额时暂停新增写入。个人常规额度会替代该账号的限时扩容。</p><form @submit.prevent="confirmation='quota'"><label>适用范围<select v-model="quotaScope"><option value="default">全站默认</option><option v-if="current" value="account">{{current.username}} 的常规额度</option></select></label><label>常规总额度（MB）<input v-model.number="quota" type="number" min="100" max="10000" required/></label><button :disabled="busy||stale||!!error">保存常规额度</button></form></details>
</section>
<ModalDialog v-if="confirmation" title="确认处理" variant="confirm" :close-disabled="busy" @close="!busy&&(confirmation=null)"><p v-if="confirmation==='quota'">将{{quotaScope==='default'?'全站默认':current?.username+'的常规'}}额度设置为 {{gb(quota*1e6)}}？</p><p v-else-if="confirmation==='approve'">批准 {{current?.username}} 的总额度为 {{gb(amount*1e6)}}，期限{{durations[duration]}}？</p><p v-else>拒绝 {{current?.username}} 的这次扩容申请？</p><p v-if="confirmation!=='quota'" class="ops-reason">回复：{{note||'无留言'}}</p><template #footer><button :disabled="busy" @click="confirmation=null">返回修改</button><button class="primary" :disabled="busy" @click="decide">确认提交</button></template></ModalDialog>
</template>
<style scoped>
.web-operations{overflow:auto;min-height:0;padding:18px 22px;flex:1;scrollbar-gutter:stable}.ops-notice{padding:8px 12px;border-radius:8px;background:var(--selected);color:var(--accent);margin:10px 0;font-size:.85rem}.ops-heading{display:flex;gap:12px;align-items:center;justify-content:space-between}.ops-heading h2,.ops-approval h2{font-size:1.1rem;margin:0 0 6px}.ops-heading p,small{color:var(--muted);font-size:.8rem}.ops-error{color:var(--danger)}.ops-metrics{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin:14px 0 18px}.ops-metrics article{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:7px;border-top:2px solid color-mix(in srgb,var(--teal) 35%,var(--border))}.ops-metrics span{color:var(--muted);font-size:.72rem;line-height:1.5}.ops-metrics strong{font-size:1rem;font-variant-numeric:tabular-nums;white-space:nowrap}.ops-metrics small{font-size:.68rem}.ops-approval{display:grid;grid-template-columns:220px minmax(0,1fr);gap:20px;background:var(--panel);padding:18px;border:1px solid var(--border);border-radius:14px}.ops-requests{display:grid;gap:8px;margin-top:14px;max-height:420px;overflow:auto}.ops-requests button{display:grid;gap:6px;text-align:left;overflow-wrap:anywhere;white-space:normal;padding:12px}.chosen{background:var(--selected);border-color:var(--accent)}.ops-detail{min-width:0;border-left:1px solid var(--border);padding-left:20px}.ops-detail h3{margin:0;overflow-wrap:anywhere}.ops-detail p{font-size:.83rem;line-height:1.7;margin:6px 0}.ops-detail form{margin-top:14px}.ops-reason{white-space:pre-wrap;overflow-wrap:anywhere}.ops-detail form,.ops-quota form{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{display:grid;gap:7px;font-size:.86rem}input,select,textarea{min-width:0;width:100%;box-sizing:border-box}.ops-note,.ops-actions{grid-column:1/-1}textarea{min-height:76px;resize:vertical;border:1px solid var(--border);border-radius:8px;background:var(--app);color:var(--text);font:inherit;padding:10px}.ops-actions{display:flex;gap:10px}.ops-quota{margin-top:20px}.ops-quota summary{cursor:pointer}.ops-quota p{font-size:.85rem;line-height:1.7}.ops-quota form{max-width:620px}.ops-quota button{align-self:end}
@media(max-width:1150px){.ops-metrics{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:700px){.web-operations{padding:14px}.ops-detail{padding:14px 0 0;border-left:0;border-top:1px solid var(--border)}.ops-metrics{grid-template-columns:1fr 1fr;gap:8px}.ops-metrics article{padding:12px}.ops-metrics strong{font-size:.95rem}.ops-approval{grid-template-columns:1fr;padding:14px}.ops-requests{max-height:180px}.ops-detail form,.ops-quota form{grid-template-columns:1fr}.ops-heading h2{font-size:1rem}}
</style>
