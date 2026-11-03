<script setup lang="ts">
import {messageRef,message as uiMessage,type MessageDescriptor} from '../i18n/index.ts';
import {tr} from '../i18n/index.ts';
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
import LanguageSwitch from '../components/LanguageSwitch.vue';
import {initializeLocale} from '../i18n/index.ts';
import {accountLocaleStore} from '../platform/locale-preferences.ts';
const research=shallowRef<ResearchContext|null>(null);
const defaultProject=ref('');
const localProject=(id:string)=>publicOptions.value.ephemeral_workspaces&&id===defaultProject.value?'default':id;
const managing=ref(false),filesTab=ref<'files'|'tasks'>('files');
function leaveResearch(){managing.value=false;stop();research.value?.files.dispose();research.value=null;}
function enterResearch(){if(!session.value||!selected.value)return;leaveResearch();const owner=session.value.user.id,project=selected.value,version=generation;const invalidate=()=>{if(version===generation&&session.value?.user.id===owner)clearAccount();};research.value={key:'server:'+owner+':'+project.id+':M01',ownerId:owner,label:project.name,files:serverFiles(owner,project.id,invalidate,()=>session.value?.csrf_token??'',undefined,localProject(project.id))};}

type Session = components['schemas']['SessionView'];
type Project = components['schemas']['ProjectView'];
const session = ref<Session | null>(null), projects = ref<Project[]>([]);
watch(()=>session.value?.user.id,id=>{void selectFontOwner(id);},{immediate:true});
const username = ref(''), password = ref(''), name = ref(''), message = messageRef();
const busy = ref(false), checking = ref(true), available = ref(true), references = ref(false);
const workspace=ref<components['schemas']['WorkspaceView']|null>(null),workspaceExpired=ref(false),connectionMessage=messageRef();
let heartbeatTimer:ReturnType<typeof setInterval>|undefined,heartbeatBusy=false;
const selected = ref<Project | null>(null), rename = ref('');
const emailLink=accountLink(location.hash);
if(emailLink)history.replaceState(null,'',location.pathname+location.search);
const publicFlow=ref<PublicAccountFlow|null>(emailLink?.flow??null),emailToken=ref(emailLink?.token);
const publicOptions=ref<components['schemas']['PublicAccountOptions']>({registration:false,password_recovery:false,ephemeral_workspaces:false});
function backToLogin(status?:string|MessageDescriptor){publicFlow.value=null;emailToken.value=undefined;password.value='';message.value=status??'';}
function emailLinkChanged(){const link=accountLink(location.hash);if(!link)return;history.replaceState(null,'',location.pathname+location.search);leaveResearch();publicFlow.value=link.flow;emailToken.value=link.token;message.value='';}
function savedTheme() { try { return localStorage.getItem('ptb.v3.account-theme') || 'system'; } catch { return 'system'; } }
const theme = ref(savedTheme());
const systemTheme = matchMedia('(prefers-color-scheme: dark)');
let generation = 0;
const channel = typeof BroadcastChannel !== 'undefined' ? new BroadcastChannel('ptb-v3-accounts') : null;
function applyTheme() { document.documentElement.dataset.theme = theme.value === 'system' ? (systemTheme.matches ? 'dark' : 'light') : theme.value === 'dark' ? 'dark' : 'light'; try { localStorage.setItem('ptb.v3.account-theme', theme.value); } catch { /* Theme still works without persistence. */ } }
function clearAccount() { if(session.value){clearM01Owner(session.value.user.id);clearWorkspaceOwner(session.value.user.id);};leaveResearch();closeAccountPreferences();void initializeLocale();generation++; session.value = null; workspace.value=null;workspaceExpired.value=false;connectionMessage.value='';projects.value = []; selected.value = null; name.value = ''; rename.value = ''; password.value = ''; }
class ApiError extends Error { constructor(public status: number, public code: string) { super(code); } }
async function api<T>(path: string, method = 'GET', body?: unknown, csrf?: string, owner?: string): Promise<T> {
  const response = await fetch('/api/v1/' + path, { method, credentials: 'same-origin', cache: 'no-store',
    headers: { ...(body ? { 'Content-Type': 'application/json' } : {}), ...(csrf ? { 'X-CSRF-Token': csrf } : {}), ...(owner ? { 'X-PTB-Account': owner } : {}) },
    body: body ? JSON.stringify(body) : undefined });
  if (!response.ok) { const error = await response.json().catch(() => ({})); throw new ApiError(response.status, error.detail || 'service_unavailable'); }
  return response.status === 204 ? undefined as T : response.json();
}
function errorText(error: unknown) {
  if(error instanceof Error&&error.message.startsWith('settings_'))return uiMessage('account.account_settings_could_not_be_synchronized_check_the_network');
  if (error instanceof ApiError) return ({ web_server_busy: uiMessage('account.the_server_is_busy_wait_5_seconds_and_retry'), invalid_credentials: uiMessage('account.the_login_name_or_password_is_incorrect'), login_rate_limited: uiMessage('account.too_many_attempts_retry_in_15_minutes'),
    csrf_rejected: uiMessage('account.your_login_state_has_changed_log_in_again'), account_changed: uiMessage('account.another_window_changed_accounts_log_in_again'), authentication_required: uiMessage('account.your_session_has_expired_log_in_again'),
    project_limit_reached: uiMessage('account.the_current_project_limit_of_100_has_been_reached'), invalid_request: uiMessage('account.check_the_input_and_its_length') } as Record<string,MessageDescriptor>)[error.code] || uiMessage('account.the_account_service_is_temporarily_unavailable_retry_later');
  return uiMessage('account.the_account_service_cannot_be_reached_check_that_it');
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
    else connectionMessage.value=uiMessage('account.the_connection_was_interrupted_reconnecting_server_data_is_retained');
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
  closeAccountPreferences();void initializeLocale();
  // Clear former identity before resolving a potentially changed shared cookie.
  session.value = null; workspace.value=null; projects.value = []; selected.value = null;
  try {
    await loadPublicOptions();
    const current = await api<Session>('auth/me');
    if (version !== generation) return;
    if(previousOwner&&previousOwner!==current.user.id){clearM01Owner(previousOwner);clearWorkspaceOwner(previousOwner);}
    const preferences=await openAccountPreferences(current.user.id,()=>session.value?.csrf_token??current.csrf_token);
    if(version!==generation)return;
    await initializeLocale(accountLocaleStore(preferences));
    if(version!==generation)return;
    session.value = current; available.value = true;
    if(await attachWorkspace(current,version)){await heartbeat();if(version===generation&&!workspaceExpired.value)await loadProjects(current, version,true);}
  } catch (error) {
    if (version !== generation) return;
    if(previousOwner){clearM01Owner(previousOwner);clearWorkspaceOwner(previousOwner);}
    session.value = null; projects.value = [];
    closeAccountPreferences();void initializeLocale();
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
    const preferences=await openAccountPreferences(current.user.id,()=>session.value?.csrf_token??current.csrf_token);
    if(version!==generation)return;
    await initializeLocale(accountLocaleStore(preferences));
    if(version!==generation)return;
    session.value = current; channel?.postMessage('changed'); if(await attachWorkspace(current,version))await loadProjects(current, version,true);
  } catch (error) { if (version === generation) { clearAccount(); message.value = errorText(error); } }
  finally { busy.value = false; }
}
async function logout() {
  const current = session.value; if (!current || busy.value) return;
  busy.value = true;
  try { await flushWebSettings();await api('auth/logout', 'POST', undefined, current.csrf_token, current.user.id); clearAccount(); channel?.postMessage('changed'); message.value = uiMessage('account.logged_out_safely'); }
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
    <header class="account-header"><div class="account-brand"><img :src="logo" :alt="tr('account.phonetictoolbox_waveform_mascot')"/><div><strong>PhoneticToolbox</strong><small>{{tr('account.speech_research_workspace_web')}}</small></div></div>
      <LanguageSwitch/><select v-model="theme" :aria-label="tr('account.color_theme')" @change="applyTheme"><option value="light">{{tr('account.light')}}</option><option value="dark">{{tr('account.dark')}}</option><option value="system">{{tr('account.follow_system')}}</option></select></header>
    <main class="account-main">
      <p v-if="message" role="status" class="account-message">{{ message }}</p>
      <div v-if="checking" class="account-card"><h1>{{tr('account.restoring_your_session')}}</h1><p class="muted">{{tr('account.checking_your_login_state')}}</p></div>
      <div v-else-if="publicFlow || !session" class="entry-layout">
       <aside class="entry-story">
        <p class="eyebrow">PHONETIC TOOLBOX · ONLINE</p>
        <h1>{{tr('account.start_with_a_sound')}}<br/>{{tr('account.begin_an')}}<span>{{tr('account.exploration')}}</span></h1>
        <div class="entry-note" :aria-label="tr('account.about_the_web_version')">
         <p>{{tr('account.sorry_everyone_the_current_budget_only_stretches_to_a')}}</p>
         <p>{{tr('account.the_web_version_is_mainly_here_so_you_can')}}</p>
        </div>
        <div class="entry-links"><a href="https://www.phonetictoolbox.com/" target="_blank" rel="noopener">{{tr('account.explore_phonetictoolbox')}}</a><span>{{tr('account.use_the_desktop_version_for_local_inference_and_the')}}</span></div>
       </aside>
       <PublicAccountForm v-if="publicFlow" :key="publicFlow" :flow="publicFlow" :token="emailToken" @back="backToLogin"/>
       <section v-else class="account-card login-card" aria-labelledby="login-title">
        <p class="eyebrow">{{tr('account.welcome_back')}}</p><h2 id="login-title">{{tr('account.log_in_to_the_research_workspace')}}</h2><p class="muted">{{tr('account.after_login_enter_the_workspace_and_choose_a_tool')}}</p>
        <p v-if="!available" class="account-message">{{tr('account.the_account_service_is_not_ready_yet')}}<button @click="refresh">{{tr('account.reconnect')}}</button></p>
        <form @submit.prevent="login">
          <label>{{tr('account.login_name')}}<input v-model="username" name="username" autocomplete="username" minlength="3" maxlength="64" pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{2,63}" required :disabled="busy || !available"/></label>
          <label>{{tr('account.password')}}<input v-model="password" name="password" type="password" autocomplete="current-password" maxlength="1024" required :disabled="busy || !available"/></label>
          <button class="primary" type="submit" :disabled="busy || !available">{{ busy ? tr('account.logging_in') : tr('account.log_in') }}</button>
        </form><div class="account-actions"><button v-if="publicOptions.registration" @click="publicFlow='register'">{{tr('account.create_account')}}</button><button v-if="publicOptions.password_recovery" @click="publicFlow='forgot'">{{tr('account.forgot_password')}}</button></div><p class="hint">{{tr('account.the_desktop_version_works_directly_without_a_web_account')}}</p>
       </section>
      </div>
      <section v-else class="account-message" role="status">
        <template v-if="workspaceExpired"><h2>{{tr('account.the_temporary_workspace_has_ended')}}</h2><p>{{tr('account.more_than_30_minutes_have_passed_since_the_last')}}</p><button class="primary" :disabled="busy" @click="startFresh">{{tr('account.start_a_new_workspace')}}</button></template>
        <template v-else><p>{{tr('account.the_workspace_is_not_open_yet')}}</p><button :disabled="busy" @click="refresh">{{tr('account.reconnect')}}</button></template>
        <button :disabled="busy" @click="logout">{{tr('account.log_out')}}</button>
      </section>
    </main><footer class="account-footer"><span>{{tr('account.phonetictoolbox_made_for_speech_research')}}</span><button @click="references=true">{{tr('account.software_and_academic_acknowledgements')}}</button></footer>
    <ModalDialog v-if="references" :title="tr('account.software_and_academic_acknowledgements')" wide @close="references=false"><MethodReferences/></ModalDialog>
  </div>
  <ModalDialog v-if="managing && session && selected && research" class="web-files-dialog" :title="tr('account.files_and_storage')" wide @close="managing=false">
   <div class="files-toolbar"><span>{{tr('account.current_file_group')}} <strong>{{selected.name}}</strong></span><button :disabled="busy" @click="logout">{{tr('account.log_out')}}</button></div>
   <p class="files-retention">{{tr('account.module_settings_are_saved_to_your_account_and_restored')}}</p>
   <p v-if="message" role="status">{{message}}</p>
   <div class="files-tabs" role="group" :aria-label="tr('account.files_and_tasks')">
    <button :aria-pressed="filesTab==='files'" @click="filesTab='files'">{{tr('account.file_management')}}</button>
    <button :aria-pressed="filesTab==='tasks'" @click="filesTab='tasks'">{{tr('account.task_history')}}</button>
   </div>
   <ProjectStorage v-if="filesTab==='files'" :key="'storage-'+session.user.id+selected.id" compact :project-id="selected.id" :local-project="localProject(selected.id)" :owner-id="session.user.id" :csrf-token="session.csrf_token" @session-invalid="clearAccount"/>
   <ProjectJobs v-else :key="session.user.id+selected.id" :project-id="selected.id" :owner-id="session.user.id" :csrf-token="session.csrf_token" @session-invalid="clearAccount"/>
   <details class="files-advanced"><summary>{{tr('account.advanced_management_organize_research_in_groups')}}</summary>
    <p class="hint">{{tr('account.files_tasks_and_drafts_are_grouped_by_project_the')}}</p>
    <label>{{tr('account.view_project')}}<select :value="selected.id" @change="choose(projects.find(p=>p.id===($event.target as HTMLSelectElement).value)!)"><option v-for="p in projects" :key="p.id" :value="p.id">{{p.name}}</option></select></label>
    <button v-if="research.key!=='server:'+session.user.id+':'+selected.id+':M01'" class="primary" @click="enterResearch">{{tr('account.switch_to_this_project_s_workspace')}}</button>
    <form @submit.prevent="saveProject(true)"><label>{{tr('account.project_name')}}<input v-model="rename" maxlength="120" required :disabled="busy"/></label><button :disabled="busy || !rename.trim()">{{tr('account.save_name')}}</button></form>
    <form @submit.prevent="saveProject(false)"><label>{{tr('account.new_project_name')}}<input v-model="name" maxlength="120" required :disabled="busy" :placeholder="tr('account.for_example_vowels_and_phonation')"/></label><button :disabled="busy || !name.trim()">{{tr('account.create_project')}}</button></form>
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
