<script setup lang="ts">
import { ref, shallowRef, watch, onMounted, onUnmounted } from 'vue';
import type { components } from '../../../contracts/generated/api';
import logo from '../assets/k2.png';
import MethodReferences from '../components/MethodReferences.vue';
import ModalDialog from '../components/ModalDialog.vue';
import ProjectJobs from './ProjectJobs.vue';
import ProjectStorage from './ProjectStorage.vue';
import PublicAccountForm from './PublicAccountForm.vue';
import WebCapacityStatus from './WebCapacityStatus.vue';
import {accountLink,type PublicAccountFlow} from './public-account.ts';
import AppShell from '../app/AppShell.vue';
import {serverFiles,type ResearchContext} from '../platform/research.ts';
import {clearM01Owner} from '../modules/parameter-estimation/store.ts';
import {stop} from '../state/audio.ts';
import {selectFontOwner} from '../state/fonts.ts';
import {clearWorkspaceOwner} from '../state/workspace.ts';
import {openAccountPreferences,closeAccountPreferences} from '../platform/account-preferences.ts';
import {flushWebSettings} from '../state/web-settings.ts';
const research=shallowRef<ResearchContext|null>(null);
const defaultProject=ref('');
const localProject=(id:string)=>publicOptions.value.ephemeral_workspaces&&id===defaultProject.value?'default':id;
const managing=ref(false),filesTab=ref<'files'|'tasks'>('files');
function leaveResearch(){managing.value=false;stop();research.value?.files.dispose();research.value=null;}
function enterResearch(){if(!session.value||!selected.value)return;leaveResearch();const owner=session.value.user.id,project=selected.value;research.value={key:'server:'+owner+':'+project.id+':M01',ownerId:owner,label:project.name,files:serverFiles(owner,project.id,clearAccount,()=>session.value?.csrf_token??'',undefined,localProject(project.id))};}

type Session = components['schemas']['SessionView'];
type Project = components['schemas']['ProjectView'];
const session = ref<Session | null>(null), projects = ref<Project[]>([]);
watch(()=>session.value?.user.id,id=>{void selectFontOwner(id);},{immediate:true});
const username = ref(''), password = ref(''), name = ref(''), message = ref('');
const busy = ref(false), checking = ref(true), available = ref(true), references = ref(false);
const workspace=ref<components['schemas']['WorkspaceView']|null>(null),workspaceExpired=ref(false),connectionMessage=ref('');
let heartbeatTimer:ReturnType<typeof setInterval>|undefined,heartbeatBusy=false;
const selected = ref<Project | null>(null), rename = ref('');
const emailLink=accountLink(location.hash);
if(emailLink)history.replaceState(null,'',location.pathname+location.search);
const publicFlow=ref<PublicAccountFlow|null>(emailLink?.flow??null),emailToken=ref(emailLink?.token);
const publicOptions=ref<components['schemas']['PublicAccountOptions']>({registration:false,password_recovery:false,ephemeral_workspaces:false});
function backToLogin(status?:string){publicFlow.value=null;emailToken.value=undefined;password.value='';message.value=status??'';}
function emailLinkChanged(){const link=accountLink(location.hash);if(!link)return;history.replaceState(null,'',location.pathname+location.search);leaveResearch();publicFlow.value=link.flow;emailToken.value=link.token;message.value='';}
function savedTheme() { try { return localStorage.getItem('ptb.v3.account-theme') || 'system'; } catch { return 'system'; } }
const theme = ref(savedTheme());
const systemTheme = matchMedia('(prefers-color-scheme: dark)');
let generation = 0;
const channel = typeof BroadcastChannel !== 'undefined' ? new BroadcastChannel('ptb-v3-accounts') : null;
function applyTheme() { document.documentElement.dataset.theme = theme.value === 'system' ? (systemTheme.matches ? 'dark' : 'light') : theme.value === 'dark' ? 'dark' : 'light'; try { localStorage.setItem('ptb.v3.account-theme', theme.value); } catch { /* Theme still works without persistence. */ } }
function clearAccount() { if(session.value){clearM01Owner(session.value.user.id);clearWorkspaceOwner(session.value.user.id);};leaveResearch();closeAccountPreferences();generation++; session.value = null; workspace.value=null;workspaceExpired.value=false;connectionMessage.value='';projects.value = []; selected.value = null; name.value = ''; rename.value = ''; password.value = ''; }
class ApiError extends Error { constructor(public status: number, public code: string) { super(code); } }
async function api<T>(path: string, method = 'GET', body?: unknown, csrf?: string, owner?: string): Promise<T> {
  const response = await fetch('/api/v1/' + path, { method, credentials: 'same-origin', cache: 'no-store',
    headers: { ...(body ? { 'Content-Type': 'application/json' } : {}), ...(csrf ? { 'X-CSRF-Token': csrf } : {}), ...(owner ? { 'X-PTB-Account': owner } : {}) },
    body: body ? JSON.stringify(body) : undefined });
  if (!response.ok) { const error = await response.json().catch(() => ({})); throw new ApiError(response.status, error.detail || 'service_unavailable'); }
  return response.status === 204 ? undefined as T : response.json();
}
function errorText(error: unknown) {
  if(error instanceof Error&&error.message.startsWith('settings_'))return '账号设置暂时无法同步，请检查网络或稍后重试。';
  if (error instanceof ApiError) return ({ web_server_busy: '服务器繁忙，请稍等5秒后重试。', invalid_credentials: '登录名或密码不正确。', login_rate_limited: '尝试次数过多，请 15 分钟后重试。',
    csrf_rejected: '登录状态已变化，请重新登录。', account_changed: '另一个窗口已切换账号，请重新登录。', authentication_required: '会话已过期，请重新登录。',
    project_limit_reached: '项目数量已达到当前上限（100 个）。', invalid_request: '请检查输入内容和长度。' } as Record<string,string>)[error.code] || '账号服务暂不可用，请稍后重试。';
  return '无法连接账号服务，请检查服务是否启动。';
}
function expireWorkspace(){leaveResearch();workspace.value=null;workspaceExpired.value=true;projects.value=[];selected.value=null;connectionMessage.value='';}
async function attachWorkspace(current:Session,version:number,fresh=false){
  if(!publicOptions.value.ephemeral_workspaces)return true;
  try{
    const result=await api<components['schemas']['WorkspaceView']>('workspaces/open','POST',{fresh},current.csrf_token,current.user.id);
    if(version!==generation)return false;
    workspace.value=result;workspaceExpired.value=false;return true;
  }catch(error){if(version===generation&&error instanceof ApiError&&error.code==='workspace_expired'){expireWorkspace();return false;}throw error;}
}
async function heartbeat(){
  const current=session.value,lease=workspace.value,version=generation;
  if(!current||!lease||heartbeatBusy)return;
  heartbeatBusy=true;
  try{
    const result=await api<components['schemas']['WorkspaceView']>('workspaces/heartbeat','POST',{id:lease.id,generation:lease.generation},current.csrf_token,current.user.id);
    if(version===generation){workspace.value=result;connectionMessage.value='';}
  }catch(error){
    if(version!==generation)return;
    if(error instanceof ApiError&&error.code==='workspace_expired')expireWorkspace();
    else if(error instanceof ApiError&&[401,403,409].includes(error.status)){clearAccount();message.value=errorText(error);}
    else connectionMessage.value='连接暂时中断，正在重连。服务器保留数据至最后一次成功连接后 30 分钟，请及时下载已有结果。';
  }finally{heartbeatBusy=false;}
}
async function startFresh(){
  const current=session.value,version=generation;if(!current||busy.value)return;
  busy.value=true;message.value='';
  try{if(await attachWorkspace(current,version,true))await loadProjects(current,version,true);}
  catch(error){message.value=errorText(error);}finally{busy.value=false;}
}
async function loadProjects(current: Session, version: number, enter=false) {
  const result = await api<components['schemas']['ProjectList']>('projects', 'GET', undefined, undefined, current.user.id);
  if(version!==generation)return;
  projects.value=result.projects;
  if(enter){
    const project=publicOptions.value.ephemeral_workspaces
      ?await api<Project>('workspaces/default-project','POST',{},current.csrf_token,current.user.id)
      :result.projects[0]??await api<Project>('projects','POST',{name:'默认项目'},current.csrf_token,current.user.id);
    if(version!==generation)return;
    defaultProject.value=project.id;
    if(!projects.value.some(p=>p.id===project.id))projects.value.push(project);
    choose(project);enterResearch();
  }
}
async function refresh() {
  const previousOwner=session.value?.user.id;
  const version = ++generation; checking.value = true;leaveResearch();
  // Clear former identity before resolving a potentially changed shared cookie.
  session.value = null; workspace.value=null; projects.value = []; selected.value = null;
  try {
    await loadPublicOptions();
    const current = await api<Session>('auth/me');
    if (version !== generation) return;
    if(previousOwner&&previousOwner!==current.user.id){clearM01Owner(previousOwner);clearWorkspaceOwner(previousOwner);}
    await openAccountPreferences(current.user.id,()=>session.value?.csrf_token??current.csrf_token);
    if(version!==generation)return;
    session.value = current; available.value = true;
    if(await attachWorkspace(current,version)){await heartbeat();if(version===generation&&!workspaceExpired.value)await loadProjects(current, version,true);}
  } catch (error) {
    if (version !== generation) return;
    if(previousOwner){clearM01Owner(previousOwner);clearWorkspaceOwner(previousOwner);}
    session.value = null; projects.value = [];
    available.value = error instanceof ApiError && error.status === 401;
    if (!available.value) message.value = errorText(error);
  } finally { if (version === generation) checking.value = false; }
}
async function loadPublicOptions(){try{publicOptions.value=await api<components['schemas']['PublicAccountOptions']>('auth/options');}catch{publicOptions.value={registration:false,password_recovery:false,ephemeral_workspaces:false};}}
async function login() {
  if (busy.value) return;
  busy.value = true; message.value = ''; const version = ++generation;
  try {
    const challenge = await api<components['schemas']['Challenge']>('auth/challenge');
    if (version !== generation) return;
    const current = await api<Session>('auth/login', 'POST', { username: username.value, password: password.value }, challenge.csrf_token);
    password.value = '';
    if (version !== generation) return;
    await openAccountPreferences(current.user.id,()=>session.value?.csrf_token??current.csrf_token);
    if(version!==generation)return;
    session.value = current; channel?.postMessage('changed'); if(await attachWorkspace(current,version))await loadProjects(current, version,true);
  } catch (error) { if (version === generation) { clearAccount(); message.value = errorText(error); } }
  finally { busy.value = false; }
}
async function logout() {
  const current = session.value; if (!current || busy.value) return;
  busy.value = true;
  try { await flushWebSettings();await api('auth/logout', 'POST', undefined, current.csrf_token, current.user.id); clearAccount(); channel?.postMessage('changed'); message.value = '已安全退出。'; }
  catch (error) { message.value = errorText(error); if (error instanceof ApiError && [401,403,409].includes(error.status)) clearAccount(); }
  finally { busy.value = false; }
}
async function saveProject(renaming: boolean) {
  const current = session.value; if (!current || busy.value) return;
  const version = generation; busy.value = true; message.value = '';
  try {
    const project = await api<Project>(renaming ? 'projects/' + selected.value!.id : 'projects', renaming ? 'PATCH' : 'POST',
      { name: renaming ? rename.value : name.value }, current.csrf_token, current.user.id);
    if (version !== generation) return;
    selected.value = project; rename.value = project.name; name.value = ''; await loadProjects(current, version);
  } catch (error) { if (version === generation) { message.value = errorText(error); if (error instanceof ApiError && [401,403,409].includes(error.status) && error.code !== 'project_limit_reached') clearAccount(); } }
  finally { busy.value = false; }
}
function choose(project: Project) { selected.value = project; rename.value = project.name; }
function openFiles(){
 const current=projects.value.find(p=>research.value?.key==='server:'+session.value?.user.id+':'+p.id+':M01');
 if(current)choose(current);filesTab.value='files';managing.value=true;
}
function visibility() { if (!document.hidden) void heartbeat(); }
function online(){void heartbeat();void flushWebSettings().catch(()=>{});}
function saveSettingsOnLeave(){void flushWebSettings(true).catch(()=>{});}
onMounted(()=>window.addEventListener('pagehide',saveSettingsOnLeave));
onUnmounted(()=>{window.removeEventListener('pagehide',saveSettingsOnLeave);void flushWebSettings(true).catch(()=>{}).finally(closeAccountPreferences);});
onMounted(() => { systemTheme.addEventListener('change', applyTheme); applyTheme(); void refresh();heartbeatTimer=setInterval(()=>void heartbeat(),30000);window.addEventListener('online',online);window.addEventListener('hashchange',emailLinkChanged); document.addEventListener('visibilitychange', visibility); if (channel) channel.onmessage = () => { clearAccount(); void refresh(); }; });
onUnmounted(() => {clearInterval(heartbeatTimer);window.removeEventListener('online',online);window.removeEventListener('hashchange',emailLinkChanged);leaveResearch();if(session.value){clearM01Owner(session.value.user.id);clearWorkspaceOwner(session.value.user.id);};systemTheme.removeEventListener('change', applyTheme); generation++; channel?.close(); document.removeEventListener('visibilitychange', visibility); });
</script>
<template>
 <div class="web-server-surface" :class="{'account-surface':!research,'research-open':!!research}">
  <WebCapacityStatus v-if="session && workspace" :key="session.user.id" :owner-id="session.user.id" :csrf-token="session.csrf_token"/>
  <p v-if="connectionMessage" role="status" class="account-message">{{connectionMessage}}</p>
  <AppShell v-if="research" :research="research" @leave-project="openFiles"/>
  <div v-else class="account-page">
    <header class="account-header"><div class="account-brand"><img :src="logo" alt="PhoneticToolbox 波形团子"/><div><strong>PhoneticToolbox</strong><small>语音研究工作台 · 网页版</small></div></div>
      <select v-model="theme" aria-label="配色主题" @change="applyTheme"><option value="light">浅色</option><option value="dark">深色</option><option value="system">跟随系统</option></select></header>
    <main class="account-main">
      <p v-if="message" role="status" class="account-message">{{ message }}</p>
      <div v-if="checking" class="account-card"><h1>正在恢复会话</h1><p class="muted">确认你的登录状态…</p></div>
      <div v-else-if="publicFlow || !session" class="entry-layout">
       <aside class="entry-story">
        <p class="eyebrow">PHONETIC TOOLBOX · ONLINE</p>
        <h1>从一段声音，<br/>开始一次<span>探索。</span></h1>
        <div class="entry-note" aria-label="网页版使用说明">
         <p>抱歉大家，受预算限制，暂时还租不起性能很强的服务器，所以……遇上高并发或者计算量很大的任务，这台小服务器就有些吃力了（挠头</p>
         <p>网页版主要是方便大家在决定是否下载之前，先来里面逛逛~或者随手查查、临时用用、应个急什么的，正式做研究时，大家就在本地运行叭~</p>
        </div>
        <div class="entry-links"><a href="https://www.phonetictoolbox.com/" target="_blank" rel="noopener">了解 PhoneticToolbox ↗</a><span>本机推理与完整工具，请使用桌面版</span></div>
       </aside>
       <PublicAccountForm v-if="publicFlow" :key="publicFlow" :flow="publicFlow" :token="emailToken" @back="backToLogin"/>
       <section v-else class="account-card login-card" aria-labelledby="login-title">
        <p class="eyebrow">欢迎回来</p><h2 id="login-title">登录研究工作台</h2><p class="muted">登录后直接进入工作台，选择工具开始研究。</p>
        <p v-if="!available" class="account-message">账号服务尚未就绪。<button @click="refresh">重新连接</button></p>
        <form @submit.prevent="login">
          <label>登录名<input v-model="username" name="username" autocomplete="username" minlength="3" maxlength="64" pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{2,63}" required :disabled="busy || !available"/></label>
          <label>密码<input v-model="password" name="password" type="password" autocomplete="current-password" maxlength="1024" required :disabled="busy || !available"/></label>
          <button class="primary" type="submit" :disabled="busy || !available">{{ busy ? '正在登录…' : '登录' }}</button>
        </form><div class="account-actions"><button v-if="publicOptions.registration" @click="publicFlow='register'">注册账号</button><button v-if="publicOptions.password_recovery" @click="publicFlow='forgot'">忘记密码</button></div><p class="hint">桌面版可直接使用，无需网页账号。</p>
       </section>
      </div>
      <section v-else class="account-message" role="status">
        <template v-if="workspaceExpired"><h2>临时工作区已结束</h2><p>距离上次连接已超过30分钟，请开始新的工作区。</p><button class="primary" :disabled="busy" @click="startFresh">开始新的工作区</button></template>
        <template v-else><p>工作台尚未打开。</p><button :disabled="busy" @click="refresh">重新连接</button></template>
        <button :disabled="busy" @click="logout">退出登录</button>
      </section>
    </main><footer class="account-footer"><span>PhoneticToolbox · 为语音研究而作</span><button @click="references=true">开源与学术致谢</button></footer>
    <ModalDialog v-if="references" title="开源与学术致谢" wide @close="references=false"><MethodReferences/></ModalDialog>
  </div>
  <ModalDialog v-if="managing && session && selected && research" class="web-files-dialog" title="文件与空间" wide @close="managing=false">
   <div class="files-toolbar"><span>当前文件组 · <strong>{{selected.name}}</strong></span><button :disabled="busy" @click="logout">退出登录</button></div>
   <p class="files-retention">模块设置随账号保存，换浏览器或设备登录也会恢复。文件与本机结果保存在当前浏览器，刷新和重新登录可继续使用。服务器计算所用的临时副本在退出后清理，关页或断线保留30分钟。</p>
   <p v-if="message" role="status">{{message}}</p>
   <div class="files-tabs" role="group" aria-label="文件与任务">
    <button :aria-pressed="filesTab==='files'" @click="filesTab='files'">文件管理</button>
    <button :aria-pressed="filesTab==='tasks'" @click="filesTab='tasks'">任务记录</button>
   </div>
   <ProjectStorage v-if="filesTab==='files'" :key="'storage-'+session.user.id+selected.id" compact :project-id="selected.id" :local-project="localProject(selected.id)" :owner-id="session.user.id" :csrf-token="session.csrf_token" @session-invalid="clearAccount"/>
   <ProjectJobs v-else :key="session.user.id+selected.id" :project-id="selected.id" :owner-id="session.user.id" :csrf-token="session.csrf_token" @session-invalid="clearAccount"/>
   <details class="files-advanced"><summary>高级管理 · 分组整理不同研究</summary>
    <p class="hint">文件、任务和草稿按项目分组。日常使用默认项目即可。</p>
    <label>查看项目<select :value="selected.id" @change="choose(projects.find(p=>p.id===($event.target as HTMLSelectElement).value)!)"><option v-for="p in projects" :key="p.id" :value="p.id">{{p.name}}</option></select></label>
    <button v-if="research.key!=='server:'+session.user.id+':'+selected.id+':M01'" class="primary" @click="enterResearch">切换到此项目工作台</button>
    <form @submit.prevent="saveProject(true)"><label>项目名称<input v-model="rename" maxlength="120" required :disabled="busy"/></label><button :disabled="busy || !rename.trim()">保存名称</button></form>
    <form @submit.prevent="saveProject(false)"><label>新项目名称<input v-model="name" maxlength="120" required :disabled="busy" placeholder="例如：元音与发声类型"/></label><button :disabled="busy || !name.trim()">新建项目</button></form>
   </details>
  </ModalDialog>
 </div>
</template>
<style scoped src="./account-web.css"></style>

<style scoped>
.research-open{height:calc(100dvh / var(--page-scale,1));display:flex;flex-direction:column;overflow:hidden}.research-open :deep(.app-shell){height:auto;flex:1;min-height:0}.research-open>:deep(.web-capacity){flex:none}
.web-files-dialog{width:min(960px,calc(100vw / var(--page-scale,1) - 32px));border-radius:16px}
.web-files-dialog :deep(button),.web-files-dialog :deep(input),.web-files-dialog :deep(select){font-family:inherit;border-radius:8px}
.web-files-dialog :deep(.dialog-header){padding:16px 22px}.web-files-dialog :deep(.dialog-body){padding:14px 22px 20px}
.files-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:.9rem}.files-toolbar strong{overflow-wrap:anywhere}
.files-retention{padding:10px 12px;background:var(--selected);border-radius:8px;color:var(--muted);font-size:.82rem;line-height:1.65;margin:12px 0}
.files-tabs{display:flex;gap:6px;padding:4px;background:var(--rail);border-radius:10px;width:fit-content;margin-top:14px}.files-tabs button{border-color:transparent}.files-tabs button[aria-pressed=true]{font-weight:600}.files-tabs button[aria-pressed=true]{background:var(--selected);border-color:var(--accent)}
.web-files-dialog :deep(.project-storage),.web-files-dialog :deep(.project-jobs){margin-top:12px;padding-top:12px}.web-files-dialog :deep(.storage-meter){padding:16px;background:var(--rail);border:1px solid var(--border);border-radius:12px;margin:12px 0}.web-files-dialog :deep(.storage-meter strong){font-size:1.25rem}.web-files-dialog :deep(.storage-meter small){display:block;line-height:1.7}
.web-files-dialog :deep(.storage-import){margin:12px 0;padding:14px 16px;border:1px dashed var(--border);border-radius:12px;background:var(--panel)}.web-files-dialog :deep(summary){cursor:pointer;color:var(--accent);font-size:.86rem}
.web-files-dialog :deep(.storage-upload){margin:12px 0;align-items:end}.web-files-dialog :deep(.storage-upload input[type=file]){border:0;background:transparent;padding:0;font-size:.86rem}.web-files-dialog :deep(input[type=file]::file-selector-button){font:inherit;padding:8px 12px;border:1px solid var(--border);border-radius:8px;background:var(--selected);color:var(--text);margin-right:10px}.web-files-dialog :deep(.storage-import .hint){font-size:.8rem;line-height:1.65;color:var(--muted)}.web-files-dialog :deep(.storage-sort){margin:12px 0}
.files-advanced{border-top:1px solid var(--border);margin-top:16px;padding-top:12px}.files-advanced form{display:flex;align-items:end;gap:12px;margin-top:12px}.files-advanced label{display:grid;gap:6px;flex:1}.files-advanced input,.files-advanced select{min-width:0;width:100%}.files-advanced>label{margin:12px 0}
@media(max-width:600px){.web-files-dialog :deep(.dialog-body){padding:12px 16px}.files-advanced form{flex-wrap:wrap}.files-advanced label{min-width:0}.files-toolbar{align-items:start}}
</style>
