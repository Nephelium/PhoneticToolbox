<script setup lang="ts">
import {computed,onBeforeUnmount,onMounted,ref,watch} from 'vue';
import {updater,updatesAvailable,onUpdatesAvailability,checkMessage,sourceLabel,type UpdateCheck,type UpdatePreferences,type DownloadProgress,type VerifiedDownload} from '../platform/updates';
const emit=defineEmits<{checked:[result:UpdateCheck];downloaded:[download:VerifiedDownload];applied:[]}>();
const props=defineProps<{initialResult?:UpdateCheck}>();
const available=ref(updatesAvailable()),prefs=ref<UpdatePreferences>(),result=ref<UpdateCheck>(),error=ref(''),busy=ref(false),progress=ref<DownloadProgress>(),download=ref<VerifiedDownload>();
const confirmDownload=ref(false),confirmApply=ref(false),loading=ref(false);
watch(()=>props.initialResult,value=>{if(value&&!busy.value)result.value=value;},{immediate:true});
let abort:AbortController|undefined;
const unsubscribe=onUpdatesAvailability(()=>{available.value=updatesAvailable();if(available.value)void load();});
const candidate=computed(()=>result.value?.candidate),packageInfo=computed(()=>candidate.value?.packages.find(item=>item.kind===prefs.value?.packageKind));
const percentage=computed(()=>progress.value?Math.round(100*progress.value.received/progress.value.total):0);
async function load(){loading.value=true;try{prefs.value=await updater.preferences();}catch(cause){error.value=(cause as Error).message;}finally{loading.value=false;}}
async function configure(){if(!prefs.value)return;error.value='';try{prefs.value=await updater.configure({source:prefs.value.source,channel:prefs.value.channel,autoCheck:prefs.value.autoCheck});result.value=undefined;download.value=undefined;}catch(cause){error.value=(cause as Error).message;}}
async function check(){if(busy.value)return;busy.value=true;error.value='';confirmDownload.value=false;download.value=undefined;progress.value=undefined;abort=new AbortController();try{result.value=await updater.check({manual:true},abort.signal);emit('checked',result.value);}catch(cause){error.value=(cause as Error).message;}finally{busy.value=false;abort=undefined;}}
async function startDownload(){if(!candidate.value||!prefs.value||busy.value)return;busy.value=true;error.value='';confirmDownload.value=false;progress.value=undefined;abort=new AbortController();try{download.value=await updater.download(candidate.value.id,prefs.value.packageKind,true,{signal:abort.signal,progress:value=>{progress.value=value;}});emit('downloaded',download.value);}catch(cause){error.value=(cause as Error).message;}finally{busy.value=false;abort=undefined;}}
async function apply(){if(!download.value||busy.value)return;busy.value=true;error.value='';confirmApply.value=false;try{const response=await updater.apply(download.value.downloadId,true);if(response.started)emit('applied');else error.value=response.message||'暂未开始换版，请重试。';}catch(cause){error.value=(cause as Error).message;}finally{busy.value=false;}}
onMounted(()=>{if(available.value)void load();});
onBeforeUnmount(()=>{abort?.abort();unsubscribe();});
</script>
<template>
  <section class="update-panel" aria-labelledby="update-title">
    <header><div><h1 id="update-title">检查更新</h1><p class="muted">保留设置和用户文件，确认后下载新版本。</p></div><span v-if="prefs" class="version">{{prefs.currentVersion}}</span></header>
    <p v-if="!available" class="notice">在线更新在桌面版提供。网页工具可继续离线使用。</p>
    <p v-else-if="loading" role="status">正在读取更新设置…</p>
    <template v-else-if="prefs">
      <div class="update-settings">
        <label>下载来源<select v-model="prefs.source" :disabled="busy" @change="configure"><option value="auto">自动选择</option><option value="server">国内服务器</option><option value="github">GitHub</option></select></label>
        <label>版本通道<select v-model="prefs.channel" :disabled="busy" @change="configure"><option value="preview">Preview（含预发布版）</option><option value="stable">稳定版</option></select></label>
        <label class="auto-check"><input v-model="prefs.autoCheck" type="checkbox" :disabled="busy" @change="configure">启动时检查更新</label>
      </div>
      <p class="hint">自动检查间隔 {{prefs.checkIntervalHours}} 小时，同一版本一天内提示一次。手动检查可随时重试。{{prefs.packageKind==='installer'?'当前为安装版。':'当前为免安装版。'}}</p>
      <div class="update-actions"><button class="primary" :disabled="busy" @click="check">{{busy&&!progress?'正在检查…':'检查更新'}}</button><button v-if="busy&&abort" @click="abort?.abort()">取消</button></div>
      <div v-if="result" class="update-result" role="status">
        <h2>{{checkMessage(result)}}</h2>
        <p v-if="result.region" class="hint">{{prefs.source==='auto'?'自动选择':'手动选择'}} · {{result.region.label}}{{result.region.country?`（${result.region.country}）`:''}} · 优先{{sourceLabel(result.preferredSource||'server')}}</p>
        <div class="sources"><article v-for="name in (['server','github'] as const)" :key="name"><strong>{{sourceLabel(name)}}</strong><span :class="{'source-error':result.sources[name]?.status==='error'}">{{result.sources[name]?.status==='error'?result.sources[name]?.message:result.sources[name]?.status==='no-releases'?'尚无可用发布':result.sources[name]?.status==='checked'?`已检查 · ${result.sources[name]?.version}`:'尚未检查'}}</span></article></div>
        <article v-if="candidate" class="release"><header><h2>{{candidate.version}}</h2><span class="hint">{{sourceLabel(candidate.source)}}</span></header><p v-if="candidate.notes" class="release-notes">{{candidate.notes}}</p><p v-else class="hint">此版本未提供更新说明。</p><p v-if="packageInfo" class="hint">{{packageInfo.name}} · {{(packageInfo.size/1024/1024).toFixed(1)}} MB</p><button v-if="!download&&!busy" class="primary" :disabled="!packageInfo" @click="confirmDownload=true">下载更新</button></article>
      </div>
      <div v-if="progress" class="download-progress" role="status"><label>{{progress.phase==='verified'?'已完整校验':progress.phase==='fallback'?'正在尝试另一来源':`正在从${sourceLabel(progress.source)}下载`}} · {{percentage}}%<progress :value="progress.received" :max="progress.total"></progress></label></div>
      <div v-if="download" class="download-done"><h2>更新包已下载并通过校验</h2><p>{{download.name}}</p><p class="hint">{{download.applyAvailable?'保存当前工作后，可以退出并更新。':'当前运行方式尚未接入换版，已下载包由桌面工具保留。'}}</p><button v-if="download.applyAvailable" class="primary" :disabled="busy" @click="confirmApply=true">退出并更新</button></div>
    </template>
    <p v-if="error" class="update-error" role="alert">{{error}}</p>
    <div v-if="confirmDownload||confirmApply" class="update-confirm-overlay"><section role="dialog" aria-modal="true" aria-labelledby="update-confirm-title" class="update-confirm"><h2 id="update-confirm-title">{{confirmDownload?'下载新版本？':'退出并更新？'}}</h2><p>{{confirmDownload?`确认下载 ${candidate?.version}。下载完成后会检查大小和 SHA-256。`:'请先保存正在进行的工作。当前软件将退出，由桌面换版工具完成更新。'}}</p><div><button autofocus @click="confirmDownload=false;confirmApply=false">暂不{{confirmDownload?'下载':'更新'}}</button><button class="primary" @click="confirmDownload?startDownload():apply()">{{confirmDownload?'确认下载':'确认退出并更新'}}</button></div></section></div>
  </section>
</template>
<style scoped>
.update-panel{padding:24px;max-width:960px;margin:0 auto;display:grid;gap:18px}.update-panel>header,.release>header{display:flex;justify-content:space-between;align-items:center;gap:16px}.update-panel header p{margin-top:6px}.version{font-variant-numeric:tabular-nums;white-space:nowrap;background:var(--selected);padding:7px 12px;border-radius:var(--radius);color:var(--accent)}.update-settings{display:flex;gap:18px;align-items:end;flex-wrap:wrap}.update-settings label{display:grid;gap:7px;min-width:180px}.update-settings .auto-check{display:flex;align-items:center;gap:8px;min-height:34px}.update-actions{display:flex;gap:10px}.update-result,.download-done{display:grid;gap:12px;border:1px solid var(--border);border-radius:var(--radius);padding:18px}.sources{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.sources article{padding:12px;background:var(--app);border-radius:6px;display:grid;gap:6px}.sources span{color:var(--muted);font-size:.928571rem;overflow-wrap:anywhere}.sources .source-error,.update-error{color:var(--danger)}.release{display:grid;gap:12px;border-top:1px solid var(--border);padding-top:16px}.release-notes{white-space:pre-wrap;overflow-wrap:anywhere;max-height:260px;overflow:auto}.release button,.download-done button{justify-self:start}.download-progress label{display:grid;gap:8px}.download-progress progress{width:100%;accent-color:var(--accent)}.download-done h2{color:var(--success)}.update-confirm-overlay{position:fixed;inset:0;background:var(--overlay);z-index:150;display:grid;place-items:center;padding:20px}.update-confirm{background:var(--panel);border:1px solid var(--border);box-shadow:var(--shadow);border-radius:var(--radius);padding:24px;max-width:500px;display:grid;gap:18px}.update-confirm>div{display:flex;justify-content:flex-end;flex-wrap:wrap;gap:10px}@media(max-width:620px){.update-panel{padding:16px}.update-panel>header{align-items:start;flex-direction:column}.sources{grid-template-columns:1fr}.update-settings label{min-width:0;width:100%}}
</style>
