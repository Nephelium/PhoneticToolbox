<script setup lang="ts">
import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {computed,onBeforeUnmount,onMounted,ref,watch} from 'vue';
import {updater,updatesAvailable,onUpdatesAvailability,checkMessage,sourceLabel,updateDisplayMessage,type UpdateCheck,type UpdatePreferences,type DownloadProgress,type VerifiedDownload} from '../platform/updates';
import {renderReleaseNotes} from '../platform/release-notes';
import ModalDialog from './ModalDialog.vue';
const emit=defineEmits<{checked:[result:UpdateCheck|undefined];downloaded:[download:VerifiedDownload];applied:[]}>();
const props=defineProps<{initialResult?:UpdateCheck}>();
const available=ref(updatesAvailable()),prefs=ref<UpdatePreferences>(),result=ref<UpdateCheck>(),error=messageRef(),busy=ref(false),progress=ref<DownloadProgress>(),download=ref<VerifiedDownload>();
const confirmDownload=ref(false),confirmApply=ref(false),loading=ref(false);
const settingsOpen=ref(true),operation=ref<'check'|'download'|'apply'>('check');
watch(()=>props.initialResult,value=>{if(value&&!busy.value)result.value=value;},{immediate:true});
let abort:AbortController|undefined;
const unsubscribe=onUpdatesAvailability(()=>{available.value=updatesAvailable();if(available.value)void load();});
const candidate=computed(()=>result.value?.candidate),packageInfo=computed(()=>candidate.value?.packages.find(item=>item.kind===prefs.value?.packageKind));
const notes=computed(()=>renderReleaseNotes(candidate.value?.notes||''));
watch(candidate,value=>{settingsOpen.value=!value;},{immediate:true});
const percentage=computed(()=>progress.value?Math.round(100*progress.value.received/progress.value.total):0);
const progressMessage=computed(()=>download.value?tr('components.the_update_was_downloaded_and_verified'):progress.value?.phase==='verified'?tr('components.finishing_verification'):progress.value?.phase==='fallback'?tr('components.trying_another_source'):progress.value?tr('components.downloading_update_from_value_value',{v0:(sourceLabel(progress.value.source)),v1:(percentage.value)}):operation.value==='download'?tr('components.preparing_download'):tr('components.checking_for_updates'));
function versionLabel(value:string){return `v${value.replace(/^v/i,'')}`;}
async function load(){loading.value=true;try{prefs.value=await updater.preferences();}catch(cause){error.value=uiErrorMessage((cause as Error));}finally{loading.value=false;}}
async function configure(){if(!prefs.value)return;error.value='';try{prefs.value=await updater.configure({source:prefs.value.source,channel:prefs.value.channel,autoCheck:prefs.value.autoCheck});result.value=undefined;download.value=undefined;progress.value=undefined;emit('checked',undefined);}catch(cause){error.value=uiErrorMessage((cause as Error));}}
async function check(){if(busy.value)return;busy.value=true;operation.value='check';error.value='';confirmDownload.value=false;confirmApply.value=false;download.value=undefined;progress.value=undefined;abort=new AbortController();try{result.value=await updater.check({manual:true},abort.signal);emit('checked',result.value);}catch(cause){error.value=uiErrorMessage((cause as Error));}finally{busy.value=false;abort=undefined;}}
async function startDownload(){if(!candidate.value||!prefs.value||busy.value)return;busy.value=true;operation.value='download';error.value='';confirmDownload.value=false;progress.value=undefined;abort=new AbortController();try{download.value=await updater.download(candidate.value.id,prefs.value.packageKind,true,{signal:abort.signal,progress:value=>{progress.value=value;}});emit('downloaded',download.value);}catch(cause){error.value=uiErrorMessage((cause as Error));progress.value=undefined;}finally{busy.value=false;abort=undefined;}}
async function apply(){if(!download.value||busy.value)return;busy.value=true;operation.value='apply';error.value='';confirmApply.value=false;try{const response=await updater.apply(download.value.downloadId,true);if(response.started)emit('applied');else error.value=response.message?uiErrorMessage(uiError(response.message)):uiMessage('components.the_version_change_has_not_started_try_again');}catch(cause){error.value=uiErrorMessage((cause as Error));}finally{busy.value=false;}}
onMounted(()=>{if(available.value)void load();});
onBeforeUnmount(()=>{abort?.abort();unsubscribe();});
</script>
<template>
  <section class="update-panel" :aria-label="tr('components.check_for_updates')">
    <p v-if="!available" class="notice">{{tr('components.online_updates_are_available_in_the_desktop_version_web')}}</p>
    <p v-else-if="loading" role="status">{{tr('components.loading_update_settings')}}</p>
    <template v-else-if="prefs">
      <p class="update-versions">{{tr('components.current_version')}}{{versionLabel(prefs.currentVersion)}}<template v-if="candidate"><span class="version-arrow">→</span>{{tr('components.new_version_477')}}{{versionLabel(candidate.version)}}</template></p>
      <div v-if="candidate" class="release-notes" tabindex="0" role="region" :aria-label="tr('components.release_notes')">
        <div v-if="notes" class="release-markdown" v-html="notes"></div>
        <p v-else class="hint">{{tr('components.no_release_notes_were_provided_for_this_version')}}</p>
      </div>
      <div v-else-if="result" class="update-result" role="status"><h3>{{checkMessage(result)}}</h3></div>
      <div class="update-footer">
        <div v-if="progress||download" class="download-progress" role="status">
          <progress :value="download?1:progress?.received" :max="download?1:progress?.total" :aria-label="tr('components.update_download_progress')"></progress>
          <h3 v-if="download" class="download-verified">{{progressMessage}}</h3><p v-else>{{progressMessage}}</p>
        </div>
        <p v-else-if="busy" role="status">{{progressMessage}}</p>
        <p v-if="packageInfo&&!download&&!busy" class="hint">{{sourceLabel(candidate!.source)}} · {{(packageInfo.size/1024/1024).toFixed(1)}} MB</p>
        <p v-if="download" class="hint">{{download.applyAvailable?tr('components.save_your_work_then_exit_to_update'):tr('components.the_update_package_is_saved_on_this_device_use')}}</p>
        <div class="update-actions">
          <button v-if="candidate&&!download&&!busy" class="primary" :disabled="!packageInfo" @click="confirmDownload=true">{{tr('components.download_update')}}</button>
          <button v-if="!candidate" class="primary" :disabled="busy" @click="check">{{busy?tr('components.checking'):tr('components.check_for_updates')}}</button>
          <button v-if="download?.applyAvailable" class="primary" :disabled="busy" @click="confirmApply=true">{{busy?tr('components.preparing_to_exit'):tr('components.exit_and_update')}}</button>
          <button v-if="busy&&abort" @click="abort?.abort()">{{tr('components.cancel')}}</button>
        </div>
        <p v-if="candidate&&!packageInfo" class="hint">{{tr('components.no_update_package_is_currently_available_for_this_installation')}}</p>
      </div>
      <details class="update-options" :open="settingsOpen" @toggle="settingsOpen=($event.target as HTMLDetailsElement).open">
        <summary>{{tr('components.update_settings_and_sources')}}</summary>
        <div class="update-options-body">
          <div class="update-settings">
            <label>{{tr('components.download_source')}}<select v-model="prefs.source" :disabled="busy" @change="configure"><option value="auto">{{tr('components.choose_automatically')}}</option><option value="server">{{tr('components.china_server')}}</option><option value="github">GitHub</option></select></label>
            <label>{{tr('components.release_channel')}}<select v-model="prefs.channel" :disabled="busy" @change="configure"><option value="preview">{{tr('components.preview_includes_prereleases')}}</option><option value="stable">{{tr('components.stable')}}</option></select></label>
            <label class="auto-check"><input v-model="prefs.autoCheck" type="checkbox" :disabled="busy" @change="configure">{{tr('components.check_for_updates_at_startup')}}</label>
          </div>
          <p class="hint">{{tr('components.automatic_check_interval')}} {{prefs.checkIntervalHours}} {{tr('components.hours_each_version_is_shown_at_most_once_per')}}{{prefs.packageKind==='installer'?tr('components.this_is_an_installed_version'):tr('components.this_is_a_portable_version')}}</p>
          <div v-if="result" class="source-results">
            <p v-if="result.region" class="hint">{{prefs.source==='auto'?tr('components.choose_automatically'):tr('components.manual_choice')}} · {{updateDisplayMessage(result.region.label)}}{{result.region.country?`（${result.region.country}）`:''}} {{tr('components.preferred')}}{{sourceLabel(result.preferredSource||'server')}}</p>
            <div class="sources"><article v-for="name in (['server','github'] as const)" :key="name"><strong>{{sourceLabel(name)}}</strong><span :class="{'source-error':result.sources[name]?.status==='error'}">{{result.sources[name]?.status==='error'?updateDisplayMessage(result.sources[name]?.message,result.sources[name]?.display):result.sources[name]?.status==='no-releases'?tr('components.no_release_available_yet'):result.sources[name]?.status==='checked'?tr('components.checked_value',{v0:(result.sources[name]?.version)}):tr('components.not_checked_yet')}}</span></article></div>
          </div>
          <p v-if="packageInfo" class="hint package-name">{{packageInfo.name}}</p>
          <button v-if="candidate" :disabled="busy" @click="check">{{tr('components.check_for_updates')}}</button>
        </div>
      </details>
    </template>
    <p v-if="error" class="update-error" role="alert">{{error}}</p>
    <ModalDialog v-if="confirmDownload||confirmApply" :title="confirmDownload?tr('components.download_the_new_version'):tr('components.exit_and_update_505')" @close="confirmDownload=false;confirmApply=false">
      <p>{{confirmDownload?tr('components.confirm_downloading_value_the_update_package_will_be_verified',{v0:(versionLabel(candidate?.version||''))}):tr('components.save_your_current_work_first_after_confirmation_the_app')}}</p>
      <template #footer><button autofocus @click="confirmDownload=false;confirmApply=false">{{tr('components.not_now')}}{{confirmDownload?tr('components.download'):tr('components.update')}}</button><button class="primary" @click="confirmDownload?startDownload():apply()">{{confirmDownload?tr('components.confirm_download'):tr('components.confirm_exit_and_update')}}</button></template>
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
