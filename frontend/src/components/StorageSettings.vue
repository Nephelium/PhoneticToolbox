<script setup lang="ts">
import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {onMounted,ref} from 'vue';
import {localStoragePort,cacheCandidateLabel,cacheCandidateReason,cacheCleanupMessage,type StorageStatus,type CacheLocation,type CacheCandidate} from '../platform/storage.ts';
const state=ref<StorageStatus>(),enabled=ref(true),days=ref('3'),exportedDays=ref('1');
const cleanUnusedInputs=ref(false);
const busy=ref(false),message=messageRef(),error=messageRef();
const confirmAll=ref(false);
const location=ref<CacheLocation>(),locationError=messageRef();
const candidates=ref<CacheCandidate[]>([]),scanning=ref(false),scanDone=ref(false),scanError=messageRef(),candidateMessage=messageRef();
const pendingCandidate=ref(''),checkedCandidate=ref(false);
async function loadLocation(){const read=localStoragePort().cacheLocation;if(!read)return;try{location.value=await read();locationError.value='';}catch(e){locationError.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.cache_locations_could_not_be_read');}}
async function openCache(legacy=false){try{const open=localStoragePort().openCacheDirectory;if(!open)return;await open(legacy);locationError.value='';}catch(e){locationError.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.the_cache_folder_could_not_be_opened_copy_the');}}
async function scanCandidates(){const scan=localStoragePort().cacheCandidates;if(!scan)return;scanning.value=true;scanError.value='';try{candidates.value=(await scan()).candidates;scanDone.value=true;}catch(e){scanError.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.the_old_cache_check_did_not_finish_try_again');}finally{scanning.value=false;}}
async function openCandidate(id:string){try{await localStoragePort().openCacheCandidate?.(id);scanError.value='';}catch(e){scanError.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.the_folder_could_not_be_opened_copy_the_path');}}
function askClean(id:string){pendingCandidate.value=id;checkedCandidate.value=false;candidateMessage.value='';}
async function cleanCandidate(id:string){if(pendingCandidate.value!==id||!checkedCandidate.value)return;const port=localStoragePort(),item=candidates.value.find(item=>item.id===id);busy.value=true;scanError.value='';try{if(item?.clean_on_exit){const answer=await port.cleanGraphicsCandidate?.(id);candidateMessage.value=answer?.started?uiMessage('components.checking_unsaved_work_the_selected_graphics_cache_will_be'):uiMessage('components.the_exit_request_did_not_start_try_again_shortly');}else{if(!port.cleanCacheCandidate)return;candidateMessage.value=cacheCleanupMessage(await port.cleanCacheCandidate(id));pendingCandidate.value='';checkedCandidate.value=false;await scanCandidates();}}catch(e){scanError.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.cache_cleanup_did_not_finish_the_original_files_are');}finally{busy.value=false;}}
async function clearAll(){busy.value=true;error.value='';message.value='';try{const answer=await localStoragePort().clearAll();confirmAll.value=false;message.value=answer.started?uiMessage('components.checking_unsaved_work_the_cache_will_be_cleared_after'):uiMessage('components.the_exit_request_did_not_start_try_again_shortly');}catch(e){error.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.cleanup_did_not_start_try_again_shortly');}finally{busy.value=false;}}
function size(bytes:number){return bytes>=1024**3?`${(bytes/1024**3).toFixed(2)} GB`:`${(bytes/1024**2).toFixed(1)} MB`;}
function assign(value:StorageStatus){state.value=value;enabled.value=value.enabled;days.value=String(value.days);exportedDays.value=String(value.exported_cache_days);cleanUnusedInputs.value=value.clean_unused_inputs??false;}
async function load(){busy.value=true;error.value='';try{assign(await localStoragePort().status());}catch(e){error.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.cache_settings_could_not_be_read');}finally{busy.value=false;}}
async function save(){
  const a=Number(days.value),b=Number(exportedDays.value);
  if(!Number.isInteger(a)||a<1||a>3650||!Number.isInteger(b)||b<1||b>3650){error.value=uiMessage('components.enter_an_integer_retention_period_from_1_to_3650');return;}
  busy.value=true;error.value='';message.value='';
  try{assign(await localStoragePort().configure({enabled:enabled.value,days:a,exported_cache_days:b,clean_unused_inputs:cleanUnusedInputs.value}));message.value=uiMessage('components.cache_settings_saved_shorter_retention_periods_apply_at_the');}
  catch(e){error.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.cache_settings_could_not_be_saved_the_original_settings');}finally{busy.value=false;}
}
async function clean(){busy.value=true;error.value='';message.value='';
  try{const answer=await localStoragePort().clean();assign(await localStoragePort().status());message.value=answer.skipped?uiMessage('components.automatic_cleanup_is_disabled_all_files_are_retained'):uiMessage('components.cleaned_value_cached_files_freeing_value_value_value',{v0:answer.count,v1:size(answer.bytes),v2:answer.protected_count?uiMessage('components.another_value_files_are_still_used_by_tasks_and',{v0:answer.protected_count}):'',v3:answer.failed_count?uiMessage('components.another_value_files_could_not_be_cleaned_up_and',{v0:answer.failed_count}):''});}
  catch(e){error.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.cleanup_failed_files_not_confirmed_safe_to_clean_are');}finally{busy.value=false;}
}
onMounted(()=>{void load();void loadLocation();void scanCandidates();});
</script>
<template><section class="storage-settings" aria-labelledby="storage-title">
  <h2 id="storage-title">{{tr('components.local_storage_and_automatic_cleanup')}}</h2>
  <p>{{tr('components.updates_continue_to_use_your_existing_settings_you_manage')}}</p>
  <div v-if="location" class="cache-location"><h3>{{tr('components.cache_folder')}}</h3>
    <div class="cache-path"><code>{{location.path}}</code><button :disabled="busy" @click="openCache()">{{tr('components.open_cache_folder')}}</button></div>
    <p v-if="location.custom" class="hint">{{tr('components.this_folder_stores_runtime_files_downloaded_updates_and_ordinary')}}</p>
  </div><p v-if="locationError" class="error-banner" role="alert">{{locationError}}</p>
  <div v-if="location" class="old-caches"><div class="old-caches-title"><h3>{{tr('components.old_caches_and_graphics_caches')}}</h3><button :disabled="busy||scanning" @click="scanCandidates">{{scanning?tr('components.checking'):tr('components.check_cache_locations_again')}}</button></div>
    <p class="hint">{{tr('components.checks_known_historical_locations_and_lists_separately_stored_graphics')}}</p>
    <p v-if="scanDone&&!candidates.length&&!scanning" role="status">{{tr('components.known_locations_checked_no_caches_need_your_attention')}}</p>
    <div v-for="item in candidates" :key="item.id" :data-kind="item.kind" class="old-cache-item"><strong>{{cacheCandidateLabel(item)}}</strong><span class="hint"> {{tr('components.total_file_size')}}{{item.size_complete?'':tr('components.at_least')}} {{size(item.bytes)}}</span>
      <div class="cache-path"><code>{{item.path}}</code><button :disabled="busy" @click="openCandidate(item.id)">{{tr('components.open_folder_to_inspect')}}</button><button :disabled="busy||scanning||!item.can_clean" @click="askClean(item.id)">{{item.clean_on_exit?tr('components.exit_and_clean_up_after_inspection'):tr('components.clean_this_cache')}}</button></div>
      <p v-if="item.reason||item.reason_code" class="hint">{{cacheCandidateReason(item)}}</p>
      <div v-if="pendingCandidate===item.id" role="alertdialog" :aria-label="tr('components.confirm_cache_cleanup')" class="cache-confirm"><p>{{tr('components.open_the_folder_above_and_confirm_that_it_contains')}}</p><p v-if="item.clean_on_exit">{{tr('components.after_confirmation_unsaved_work_will_be_checked_the_app')}}</p><label><input v-model="checkedCandidate" type="checkbox" :disabled="busy"/>{{tr('components.i_have_inspected_this_folder_and_confirm_its_app')}}</label><div class="storage-actions"><button :disabled="busy||!checkedCandidate" @click="cleanCandidate(item.id)">{{item.clean_on_exit?tr('components.confirm_exit_and_clean_up'):tr('components.confirm_cleanup_of_this_cache')}}</button><button :disabled="busy" @click="pendingCandidate='';checkedCandidate=false">{{tr('components.cancel')}}</button></div></div>
    </div><p v-if="candidateMessage" role="status">{{candidateMessage}}</p><p v-if="scanError" class="error-banner" role="alert">{{scanError}}</p>
  </div>
  <div class="settings-columns"><div class="settings-column">
  <label><input v-model="enabled" type="checkbox" :disabled="busy"/>{{tr('components.automatically_clean_expired_result_caches')}}</label>
  <label><input v-model="cleanUnusedInputs" type="checkbox" :disabled="busy"/>{{tr('components.clean_unused_import_copies')}}</label>
  <p class="hint">{{tr('components.clean_unused_import_copies_hint')}}</p>
  <div class="storage-fields"><label>{{tr('components.retain_unexported_results_for')}} <input v-model="days" type="number" min="1" max="3650" step="1" :disabled="busy"/> {{tr('components.days')}}</label>
  <label>{{tr('components.retain_caches_of_exported_copies_for')}} <input v-model="exportedDays" type="number" min="1" max="3650" step="1" :disabled="busy"/> {{tr('components.days')}}</label></div>
  <p class="hint">{{tr('components.retention_is_measured_from_generation_or_the_most_recent')}}</p>
  <p class="hint">{{tr('components.cleanup_after_export_affects_only_the_app_s_internal')}}</p>
  <p class="hint">{{tr('components.mfa_diagnostic_temporary_folders_are_retained_for_7_days')}}</p>
  </div><div class="settings-column">
  <dl v-if="state"><div><dt>{{tr('components.result_cache')}}</dt><dd>{{size(state.result_bytes)}} · {{state.result_count}} {{tr('components.files')}}</dd></div><div><dt>{{tr('components.protected_input_copies')}}</dt><dd>{{size(state.input_bytes)}}</dd></div><div v-if="state.diagnostic_bytes!==undefined"><dt>{{tr('components.mfa_diagnostic_temporary_files')}}</dt><dd>{{size(state.diagnostic_bytes)}} · {{state.diagnostic_count??0}} {{tr('components.folders')}}</dd></div></dl>
  <p v-if="message" role="status">{{message}}</p><p v-if="error" class="error-banner" role="alert">{{error}}</p>
  <div class="storage-actions"><button class="primary" :disabled="busy" @click="save">{{tr('components.save_cleanup_settings')}}</button><button :disabled="busy||!state||!state.enabled" @click="clean">{{tr('components.clean_expired_caches')}}</button><button :disabled="busy" @click="load">{{tr('components.refresh_storage_usage')}}</button></div>
  <div class="all-caches"><h3>{{tr('components.clear_all_caches')}}</h3><p class="hint">{{tr('components.cleans_runtime_files_downloaded_updates_and_eligible_internal_result')}}</p><p class="hint">{{tr('components.portable_users_can_clean_caches_here_before_deleting_the')}}</p>
  <p v-if="location?.exit_cleanup_supported===false" role="status" class="hint">{{tr('native.cache.qaExitCleanupUnavailable')}}</p>
  <button v-if="!confirmAll" :disabled="busy||location?.exit_cleanup_supported===false" @click="confirmAll=true">{{tr('components.clear_all_caches_434')}}</button>
  <div v-else role="alertdialog" :aria-label="tr('components.confirm_clearing_all_caches')"><p>{{tr('components.save_any_results_you_need_first_after_confirmation_unsaved')}}</p><div class="storage-actions"><button :disabled="busy||location?.exit_cleanup_supported===false" @click="clearAll">{{tr('components.clear_all_caches_and_exit')}}</button><button :disabled="busy" @click="confirmAll=false">{{tr('components.cancel')}}</button></div></div></div>
</div></div>
</section></template>
<style scoped>
.storage-settings{margin-top:1.5rem;padding:1.25rem;border:1px solid var(--border);border-radius:var(--radius);background:var(--panel)}
.all-caches{margin-top:.5rem;border-top:1px solid var(--border);padding-top:.8rem}.all-caches h3{font-size:1rem;margin:.2rem 0}
.cache-location{padding:.7rem 0 1rem;border-bottom:1px solid var(--border);margin-bottom:1rem}.cache-location h3{font-size:1rem;margin:.2rem 0 .6rem}.cache-path{display:flex;align-items:center;gap:.7rem;flex-wrap:wrap}.cache-path code{flex:1 1 230px;min-width:0;overflow-wrap:anywhere;user-select:text}.cache-path button{flex:0 0 auto}.legacy-cache{margin-top:.5rem}
.old-caches{padding:0 0 1rem;margin-bottom:1rem;border-bottom:1px solid var(--border)}.old-caches-title{display:flex;align-items:center;justify-content:space-between;gap:.6rem;flex-wrap:wrap}.old-caches h3{font-size:1rem;margin:.2rem 0}.old-cache-item{padding:.7rem 0}.old-cache-item+.old-cache-item{border-top:1px solid var(--border)}.cache-confirm{margin-top:.7rem;padding:.7rem;border:1px solid var(--border);border-radius:var(--radius)}.cache-confirm p{margin-top:0}
h2{font-size:1.15rem;margin:0 0 .8rem}p{line-height:1.75}.storage-fields{display:flex;flex-wrap:wrap;gap:1rem;margin:.9rem 0}.storage-fields input{width:5.5rem;margin:0 .25rem}.storage-actions{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:.8rem}dl{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:0}dt{color:var(--muted)}dd{margin:.4rem 0}.hint{color:var(--muted)}
</style>
