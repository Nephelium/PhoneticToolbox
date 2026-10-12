<script setup lang="ts">
import {computed,onBeforeUnmount,onMounted,ref,watch} from 'vue';
import {updater,updatesAvailable,onUpdatesAvailability,checkMessage,sourceLabel,type UpdateCheck,type UpdatePreferences,type DownloadProgress,type VerifiedDownload} from '../platform/updates';
import {renderReleaseNotes} from '../platform/release-notes';
import ModalDialog from './ModalDialog.vue';
const emit=defineEmits<{checked:[result:UpdateCheck|undefined];downloaded:[download:VerifiedDownload];applied:[]}>();
const props=defineProps<{initialResult?:UpdateCheck}>();
const available=ref(updatesAvailable()),prefs=ref<UpdatePreferences>(),result=ref<UpdateCheck>(),error=ref(''),busy=ref(false),progress=ref<DownloadProgress>(),download=ref<VerifiedDownload>();
const confirmDownload=ref(false),confirmApply=ref(false),loading=ref(false);
const settingsOpen=ref(true),operation=ref<'check'|'download'|'apply'>('check');
watch(()=>props.initialResult,value=>{if(value&&!busy.value)result.value=value;},{immediate:true});
let abort:AbortController|undefined;
const unsubscribe=onUpdatesAvailability(()=>{available.value=updatesAvailable();if(available.value)void load();});
const candidate=computed(()=>result.value?.candidate),packageInfo=computed(()=>candidate.value?.packages.find(item=>item.kind===prefs.value?.packageKind));
const notes=computed(()=>renderReleaseNotes(candidate.value?.notes||''));
watch(candidate,value=>{settingsOpen.value=!value;},{immediate:true});
const percentage=computed(()=>progress.value?Math.round(100*progress.value.received/progress.value.total):0);
const progressMessage=computed(()=>download.value?'更新包已下载并通过校验':progress.value?.phase==='verified'?'正在完成校验…':progress.value?.phase==='fallback'?'正在尝试另一来源…':progress.value?`正在从${sourceLabel(progress.value.source)}下载更新 ${percentage.value}%…`:operation.value==='download'?'正在准备下载…':'正在检查更新…');
function versionLabel(value:string){return `v${value.replace(/^v/i,'')}`;}
async function load(){loading.value=true;try{prefs.value=await updater.preferences();}catch(cause){error.value=(cause as Error).message;}finally{loading.value=false;}}
async function configure(){if(!prefs.value)return;error.value='';try{prefs.value=await updater.configure({source:prefs.value.source,channel:prefs.value.channel,autoCheck:prefs.value.autoCheck});result.value=undefined;download.value=undefined;progress.value=undefined;emit('checked',undefined);}catch(cause){error.value=(cause as Error).message;}}
async function check(){if(busy.value)return;busy.value=true;operation.value='check';error.value='';confirmDownload.value=false;confirmApply.value=false;download.value=undefined;progress.value=undefined;abort=new AbortController();try{result.value=await updater.check({manual:true},abort.signal);emit('checked',result.value);}catch(cause){error.value=(cause as Error).message;}finally{busy.value=false;abort=undefined;}}
async function startDownload(){if(!candidate.value||!prefs.value||busy.value)return;busy.value=true;operation.value='download';error.value='';confirmDownload.value=false;progress.value=undefined;abort=new AbortController();try{download.value=await updater.download(candidate.value.id,prefs.value.packageKind,true,{signal:abort.signal,progress:value=>{progress.value=value;}});emit('downloaded',download.value);}catch(cause){error.value=(cause as Error).message;progress.value=undefined;}finally{busy.value=false;abort=undefined;}}
async function apply(){if(!download.value||busy.value)return;busy.value=true;operation.value='apply';error.value='';confirmApply.value=false;try{const response=await updater.apply(download.value.downloadId,true);if(response.started)emit('applied');else error.value=response.message||'暂未开始换版，请重试。';}catch(cause){error.value=(cause as Error).message;}finally{busy.value=false;}}
onMounted(()=>{if(available.value)void load();});
onBeforeUnmount(()=>{abort?.abort();unsubscribe();});
</script>
<template>
  <section class="update-panel" aria-label="检查更新">
    <p v-if="!available" class="notice">在线更新在桌面版提供。网页工具可继续离线使用。</p>
    <p v-else-if="loading" role="status">正在读取更新设置…</p>
    <template v-else-if="prefs">
      <p class="update-versions">当前版本：{{versionLabel(prefs.currentVersion)}}<template v-if="candidate"><span class="version-arrow">→</span>新版本：{{versionLabel(candidate.version)}}</template></p>
      <div v-if="candidate" class="release-notes" tabindex="0" role="region" aria-label="更新说明">
        <div v-if="notes" class="release-markdown" v-html="notes"></div>
        <p v-else class="hint">此版本未提供更新说明。</p>
      </div>
      <div v-else-if="result" class="update-result" role="status"><h3>{{checkMessage(result)}}</h3></div>
      <div class="update-footer">
        <div v-if="progress||download" class="download-progress" role="status">
          <progress :value="download?1:progress?.received" :max="download?1:progress?.total" aria-label="更新下载进度"></progress>
          <h3 v-if="download" class="download-verified">{{progressMessage}}</h3><p v-else>{{progressMessage}}</p>
        </div>
        <p v-else-if="busy" role="status">{{progressMessage}}</p>
        <p v-if="packageInfo&&!download&&!busy" class="hint">{{sourceLabel(candidate!.source)}} · {{(packageInfo.size/1024/1024).toFixed(1)}} MB</p>
        <p v-if="download" class="hint">{{download.applyAvailable?'保存当前工作后，可以退出并更新。':'更新包已保存在本机，请使用安装包或更新工具完成更新。'}}</p>
        <div class="update-actions">
          <button v-if="candidate&&!download&&!busy" class="primary" :disabled="!packageInfo" @click="confirmDownload=true">下载更新</button>
          <button v-if="!candidate" class="primary" :disabled="busy" @click="check">{{busy?'正在检查…':'检查更新'}}</button>
          <button v-if="download?.applyAvailable" class="primary" :disabled="busy" @click="confirmApply=true">{{busy?'正在准备退出…':'退出并更新'}}</button>
          <button v-if="busy&&abort" @click="abort?.abort()">取消</button>
        </div>
        <p v-if="candidate&&!packageInfo" class="hint">此版本暂未提供适合当前安装方式的更新包。</p>
      </div>
      <details class="update-options" :open="settingsOpen" @toggle="settingsOpen=($event.target as HTMLDetailsElement).open">
        <summary>更新设置与来源</summary>
        <div class="update-options-body">
          <div class="update-settings">
            <label>下载来源<select v-model="prefs.source" :disabled="busy" @change="configure"><option value="auto">自动选择</option><option value="server">国内服务器</option><option value="github">GitHub</option></select></label>
            <label>版本通道<select v-model="prefs.channel" :disabled="busy" @change="configure"><option value="preview">Preview（含预发布版）</option><option value="stable">稳定版</option></select></label>
            <label class="auto-check"><input v-model="prefs.autoCheck" type="checkbox" :disabled="busy" @change="configure">启动时检查更新</label>
          </div>
          <p class="hint">自动检查间隔 {{prefs.checkIntervalHours}} 小时，同一版本一天内提示一次。{{prefs.packageKind==='installer'?'当前为安装版。':'当前为免安装版。'}}</p>
          <div v-if="result" class="source-results">
            <p v-if="result.region" class="hint">{{prefs.source==='auto'?'自动选择':'手动选择'}} · {{result.region.label}}{{result.region.country?`（${result.region.country}）`:''}} · 优先{{sourceLabel(result.preferredSource||'server')}}</p>
            <div class="sources"><article v-for="name in (['server','github'] as const)" :key="name"><strong>{{sourceLabel(name)}}</strong><span :class="{'source-error':result.sources[name]?.status==='error'}">{{result.sources[name]?.status==='error'?result.sources[name]?.message:result.sources[name]?.status==='no-releases'?'尚无可用发布':result.sources[name]?.status==='checked'?`已检查 · ${result.sources[name]?.version}`:'尚未检查'}}</span></article></div>
          </div>
          <p v-if="packageInfo" class="hint package-name">{{packageInfo.name}}</p>
          <button v-if="candidate" :disabled="busy" @click="check">检查更新</button>
        </div>
      </details>
    </template>
    <p v-if="error" class="update-error" role="alert">{{error}}</p>
    <ModalDialog v-if="confirmDownload||confirmApply" :title="confirmDownload?'下载新版本？':'退出并更新？'" @close="confirmDownload=false;confirmApply=false">
      <p>{{confirmDownload?`确认下载 ${versionLabel(candidate?.version||'')}。下载完成后会校验更新包。`:'请先保存正在进行的工作。确认后软件将退出并完成更新。'}}</p>
      <template #footer><button autofocus @click="confirmDownload=false;confirmApply=false">暂不{{confirmDownload?'下载':'更新'}}</button><button class="primary" @click="confirmDownload?startDownload():apply()">{{confirmDownload?'确认下载':'确认退出并更新'}}</button></template>
    </ModalDialog>
  </section>
</template>
<style scoped>
.update-panel{display:grid;gap:20px;min-width:0}
.update-versions{color:var(--muted);font-variant-numeric:tabular-nums;overflow-wrap:anywhere;line-height:1.6}
.version-arrow{padding:0 6px}
.release-notes{padding:22px 24px;border:1px solid var(--border);border-radius:14px;background:var(--app);max-height:clamp(112px,calc(100dvh / var(--page-scale,1) - 350px),420px);overflow:auto;overscroll-behavior:contain;overflow-wrap:anywhere;scrollbar-gutter:stable}
.release-markdown{color:var(--muted);line-height:1.85}
.release-markdown :deep(ul),.release-markdown :deep(ol){margin:0 0 12px;padding-left:1.3em}
.release-markdown :deep(li){padding-left:4px;margin:8px 0}
.release-markdown :deep(li::marker){color:var(--muted)}
.release-markdown :deep(strong){color:var(--text);font-weight:700}
.release-markdown :deep(p){margin:0 0 10px;line-height:inherit}
.release-markdown :deep(h1),.release-markdown :deep(h2),.release-markdown :deep(h3),.release-markdown :deep(h4),.release-markdown :deep(h5),.release-markdown :deep(h6){color:var(--text);font-size:1.071429rem;line-height:1.5;margin:16px 0 8px}
.release-markdown :deep(a){color:var(--accent);text-decoration:underline;text-underline-offset:3px}
.release-markdown :deep(blockquote){margin:12px 0;padding:4px 14px;border-left:3px solid var(--accent);background:var(--selected)}
.release-markdown :deep(blockquote>p:last-child){margin-bottom:0}
.release-markdown :deep(code){font-family:var(--font-mono);font-size:.928571rem;background:var(--selected);color:var(--text);border-radius:4px;padding:2px 4px}
.release-markdown :deep(pre){margin:12px 0;padding:12px;background:var(--selected);border-radius:6px;overflow:auto}
.release-markdown :deep(pre code){padding:0;background:transparent;white-space:pre;overflow-wrap:normal}
.release-markdown :deep(table){width:100%;table-layout:fixed;border-collapse:collapse;margin:12px 0}
.release-markdown :deep(th),.release-markdown :deep(td){border:1px solid var(--border);padding:7px 10px;overflow-wrap:anywhere}
.release-markdown :deep(th){color:var(--text);font-weight:700;background:var(--selected)}
.release-markdown :deep(hr){margin:16px 0}
.release-markdown :deep(>:first-child){margin-top:0}
.release-markdown :deep(>:last-child){margin-bottom:0}
.update-footer{display:grid;gap:10px;text-align:center}
.update-actions{display:flex;justify-content:center;gap:10px;flex-wrap:wrap}
.update-actions button{min-height:38px;padding:8px 18px}
.download-progress{display:grid;gap:10px;color:var(--muted)}
.download-progress progress{display:block;width:100%;height:10px;border:0;border-radius:6px;overflow:hidden;accent-color:var(--accent);background:var(--border)}
.download-progress progress::-webkit-progress-bar{background:var(--border);border-radius:6px}
.download-progress progress::-webkit-progress-value{background:var(--accent);border-radius:6px}
.download-verified{color:var(--success);font-size:1rem}
.update-options{border-top:1px solid var(--border);padding-top:12px}
.update-options summary{cursor:pointer;color:var(--muted);font-size:.928571rem;width:fit-content}
.update-options-body{display:grid;gap:12px;padding-top:16px}
.update-options-body>button{justify-self:start}
.update-settings{display:flex;gap:14px;align-items:end;flex-wrap:wrap}
.update-settings label{display:grid;gap:7px;flex:1;min-width:160px}
.update-settings .auto-check{display:flex;align-items:center;gap:8px;min-height:34px;flex-basis:100%}
.source-results{display:grid;gap:10px}
.sources{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.sources article{padding:12px;background:var(--app);border-radius:6px;display:grid;gap:6px}
.sources span{color:var(--muted);font-size:.928571rem;overflow-wrap:anywhere}
.sources .source-error,.update-error{color:var(--danger)}
.package-name{overflow-wrap:anywhere}
@media(max-width:620px){.update-panel{gap:16px}.release-notes{padding:16px}.sources{grid-template-columns:1fr}.update-settings label{min-width:0;flex-basis:100%}}
</style>
