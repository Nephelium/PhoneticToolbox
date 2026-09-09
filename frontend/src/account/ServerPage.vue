<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import type { components } from '../../../contracts/generated/api';
import logo from '../assets/k2.png';
import MethodReferences from '../components/MethodReferences.vue';
import ModalDialog from '../components/ModalDialog.vue';

type Session = components['schemas']['SessionView'];
type Project = components['schemas']['ProjectView'];
const session = ref<Session | null>(null), projects = ref<Project[]>([]);
const username = ref(''), password = ref(''), name = ref(''), message = ref('');
const busy = ref(false), checking = ref(true), available = ref(true), references = ref(false);
const selected = ref<Project | null>(null), rename = ref('');
function savedTheme() { try { return localStorage.getItem('ptb.v3.account-theme') || 'system'; } catch { return 'system'; } }
const theme = ref(savedTheme());
const systemTheme = matchMedia('(prefers-color-scheme: dark)');
let generation = 0;
const channel = typeof BroadcastChannel !== 'undefined' ? new BroadcastChannel('ptb-v3-accounts') : null;
function applyTheme() { document.documentElement.dataset.theme = theme.value === 'system' ? (systemTheme.matches ? 'dark' : 'light') : theme.value === 'dark' ? 'dark' : 'light'; try { localStorage.setItem('ptb.v3.account-theme', theme.value); } catch { /* Theme still works without persistence. */ } }
function clearAccount() { generation++; session.value = null; projects.value = []; selected.value = null; name.value = ''; rename.value = ''; password.value = ''; }
class ApiError extends Error { constructor(public status: number, public code: string) { super(code); } }
async function api<T>(path: string, method = 'GET', body?: unknown, csrf?: string, owner?: string): Promise<T> {
  const response = await fetch('/api/v1/' + path, { method, credentials: 'same-origin', cache: 'no-store',
    headers: { ...(body ? { 'Content-Type': 'application/json' } : {}), ...(csrf ? { 'X-CSRF-Token': csrf } : {}), ...(owner ? { 'X-PTB-Account': owner } : {}) },
    body: body ? JSON.stringify(body) : undefined });
  if (!response.ok) { const error = await response.json().catch(() => ({})); throw new ApiError(response.status, error.detail || 'service_unavailable'); }
  return response.status === 204 ? undefined as T : response.json();
}
function errorText(error: unknown) {
  if (error instanceof ApiError) return ({ invalid_credentials: '登录名或密码不正确。', login_rate_limited: '尝试次数过多，请 15 分钟后重试。',
    csrf_rejected: '登录状态已变化，请重新登录。', account_changed: '另一个窗口已切换账号，请重新登录。', authentication_required: '会话已过期，请重新登录。',
    project_limit_reached: '项目数量已达到当前上限（100 个）。', invalid_request: '请检查输入内容和长度。' } as Record<string,string>)[error.code] || '账号服务暂不可用，请稍后重试。';
  return '无法连接账号服务，请检查服务是否启动。';
}
async function loadProjects(current: Session, version: number) {
  const result = await api<components['schemas']['ProjectList']>('projects', 'GET', undefined, undefined, current.user.id);
  if (version === generation) projects.value = result.projects;
}
async function refresh() {
  const version = ++generation; checking.value = true;
  // Clear former identity before resolving a potentially changed shared cookie.
  session.value = null; projects.value = []; selected.value = null;
  try {
    const current = await api<Session>('auth/me');
    if (version !== generation) return;
    session.value = current; available.value = true;
    await loadProjects(current, version);
  } catch (error) {
    if (version !== generation) return;
    session.value = null; projects.value = [];
    available.value = error instanceof ApiError && error.status === 401;
    if (!available.value) message.value = errorText(error);
  } finally { if (version === generation) checking.value = false; }
}
async function login() {
  if (busy.value) return;
  busy.value = true; message.value = ''; const version = ++generation;
  try {
    const challenge = await api<components['schemas']['Challenge']>('auth/challenge');
    if (version !== generation) return;
    const current = await api<Session>('auth/login', 'POST', { username: username.value, password: password.value }, challenge.csrf_token);
    password.value = '';
    if (version !== generation) return;
    session.value = current; channel?.postMessage('changed'); await loadProjects(current, version);
  } catch (error) { if (version === generation) { clearAccount(); message.value = errorText(error); } }
  finally { busy.value = false; }
}
async function logout() {
  const current = session.value; if (!current || busy.value) return;
  busy.value = true;
  try { await api('auth/logout', 'POST', undefined, current.csrf_token, current.user.id); clearAccount(); channel?.postMessage('changed'); message.value = '已安全退出。'; }
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
function visibility() { if (!document.hidden && !busy.value && session.value) void refresh(); }
onMounted(() => { systemTheme.addEventListener('change', applyTheme); applyTheme(); void refresh(); document.addEventListener('visibilitychange', visibility); if (channel) channel.onmessage = () => { clearAccount(); void refresh(); }; });
onUnmounted(() => { systemTheme.removeEventListener('change', applyTheme); generation++; channel?.close(); document.removeEventListener('visibilitychange', visibility); });
</script>
<template>
  <div class="account-page">
    <header class="account-header"><div class="account-brand"><img :src="logo" alt="PhoneticToolbox 波形团子"/><div><strong>PhoneticToolbox</strong><small>语音研究工作台 · 网页版</small></div></div>
      <select v-model="theme" aria-label="配色主题" @change="applyTheme"><option value="light">浅色</option><option value="dark">深色</option><option value="system">跟随系统</option></select></header>
    <main class="account-main">
      <p v-if="message" role="status" class="account-message">{{ message }}</p>
      <div v-if="checking" class="account-card"><h1>正在恢复会话</h1><p class="muted">确认你的登录状态…</p></div>
      <section v-else-if="!session" class="account-card login-card" aria-labelledby="login-title">
        <p class="eyebrow">RESEARCH WORKSPACE</p><h1 id="login-title">登录研究工作台</h1><p class="muted">使用研究者账号，管理自己的项目。</p>
        <p v-if="!available" class="account-message">账号服务尚未就绪。<button @click="refresh">重新连接</button></p>
        <form @submit.prevent="login">
          <label>登录名<input v-model="username" name="username" autocomplete="username" minlength="3" maxlength="64" pattern="[A-Za-z0-9][A-Za-z0-9_.-]{2,63}" required :disabled="busy || !available"/></label>
          <label>密码<input v-model="password" name="password" type="password" autocomplete="current-password" maxlength="1024" required :disabled="busy || !available"/></label>
          <button class="primary" type="submit" :disabled="busy || !available">{{ busy ? '正在登录…' : '登录' }}</button>
        </form><p class="hint">账号由项目管理员创建。桌面版可直接使用，无需网页账号。</p>
      </section>
      <section v-else class="account-workspace">
        <div class="account-title"><div><p class="eyebrow">{{ session.user.username }}</p><h1>你的研究项目</h1></div><button :disabled="busy" @click="logout">退出登录</button></div>
        <p class="muted">项目用于组织研究内容。创建项目不会自动上传或分析音频。</p>
        <form class="project-create" @submit.prevent="saveProject(false)"><label>项目名称<input v-model="name" maxlength="120" required :disabled="busy" placeholder="例如：元音与发声类型"/></label><button class="primary" :disabled="busy || !name.trim()">新建项目</button></form>
        <div class="project-columns"><section class="project-list" aria-label="项目列表"><h2>项目 · {{ projects.length }}</h2><p v-if="!projects.length" class="empty-small">还没有项目。从一个研究主题开始。</p>
          <button v-for="project in projects" :key="project.id" :aria-pressed="selected?.id === project.id" @click="choose(project)"><strong>{{ project.name }}</strong><small>{{ new Date(project.created_at).toLocaleDateString('zh-CN') }}</small></button></section>
          <section class="project-detail"><template v-if="selected"><h2>{{ selected.name }}</h2><form @submit.prevent="saveProject(true)"><label>项目名称<input v-model="rename" maxlength="120" required :disabled="busy"/></label><button :disabled="busy || !rename.trim()">保存名称</button></form><p class="empty-small">项目已建立。文件、分析任务与结果将在后续阶段接入。</p></template><p v-else class="empty-small">选择项目查看详情。</p></section></div>
      </section>
    </main><footer class="account-footer"><span>PhoneticToolbox 3.0 · 账号与项目试用</span><button @click="references=true">开源与学术致谢</button></footer>
    <ModalDialog v-if="references" title="开源与学术致谢" wide @close="references=false"><MethodReferences/></ModalDialog>
  </div>
</template>
