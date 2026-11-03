<script setup lang="ts">
import {messageRef,message as uiMessage,type MessageDescriptor} from '../i18n/index.ts';
import {tr} from '../i18n/index.ts';
import { computed, onMounted, onUnmounted, ref } from 'vue';
import type { components } from '../../../contracts/generated/api';
import ModalDialog from '../components/ModalDialog.vue';
import {LocalAssets,LocalStorageError} from '../platform/local-assets.ts';
import {LocalAssetTransport} from '../platform/local-asset-transport.ts';
import {zipStore} from '../platform/zip-store.ts';
import {storageErrorMessage as message} from './storage-error.ts';

type Asset = components['schemas']['AssetView'] & {storage?:'browser'};
// Additive response fields until the shared M08/P07 contract generation gate.
type Usage = components['schemas']['StorageUsage'] & { policy_version?: 1 | 2; retention_seconds?: number; over_quota?: boolean };
const props = defineProps<{ projectId: string; localProject?:string; ownerId: string; csrfToken: string; compact?:boolean }>();
const emit = defineEmits<{ sessionInvalid: [] }>();
const usage = ref<Usage | null>(null), assets = ref<Asset[]>([]), notice = messageRef();
const policyReady = computed(() => usage.value?.policy_version === 2);
const overQuota = computed(() => !!usage.value && usage.value.used_bytes + usage.value.reserved_bytes > usage.value.quota_bytes);
const canWrite = computed(() => available.value && policyReady.value && usage.value?.ready && !usage.value?.frozen && !overQuota.value);
const available = ref(false), busy = ref(false), progress = ref(0), order = ref('expires');
const removal = ref<Asset | null>(null), fileInput = ref<HTMLInputElement>();
const pending = ref<{ file: File; key: string; assetId?: string } | null>(null);
const fileJobs = ref(true), serverFileJobs=ref(false), selectedIds = ref<string[]>([]), outputMegabytes = ref(100);
const activeReferences = ref<string[]>([]);
const taskKeys = new Map<string,string>();
const abort = new AbortController();
const local=new LocalAssets(props.ownerId,props.localProject??props.projectId);
const transport=new LocalAssetTransport(local,(path,init)=>fetch('/api/v1/'+path,{...init,credentials:'same-origin',headers:{'X-PTB-Account':props.ownerId,...(init?.method&&init.method!=='GET'?{'X-CSRF-Token':props.csrfToken}:{}),...(init?.body?{'Content-Type':'application/json'}:{}),...init?.headers}}),abort.signal,props.projectId);
let disposed = false, timer: ReturnType<typeof setInterval> | undefined, loading = false;
async function request<T>(path: string, method = 'GET', body?: unknown, binary = false): Promise<T> {
  const response = await transport.fetch(path, {
    method, credentials: 'same-origin', signal: abort.signal,
    headers: { 'Content-Type': binary ? 'application/octet-stream' : 'application/json',
      'X-PTB-Account': props.ownerId, ...(method !== 'GET' ? { 'X-CSRF-Token': props.csrfToken } : {}) },
    body: body === undefined ? undefined : binary ? body as Blob : JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok) {
    if (!disposed && !abort.signal.aborted && (response.status === 401 || value.detail === 'account_changed')) {
      assets.value = []; usage.value = null; pending.value = null; emit('sessionInvalid');
    }
    throw new Error(value.detail || 'unavailable');
  }
  return value;
}
async function refresh() {
  if (loading || disposed) return;
  loading = true;
  try {
    assets.value=(await local.list()).map(f=>transport.view(f)) as unknown as Asset[];
    const [space, files] = await Promise.all([
      request<Usage>('storage/usage'),
      request<components['schemas']['AssetList']>(`assets?project_id=${props.projectId}&order=${order.value}`),
    ]);
    if (!disposed) { usage.value = space; assets.value = files.assets; available.value = true; }
  } catch (error) { if (!disposed) { available.value = false; notice.value = message(error); } }
  finally { loading = false; }
}
function choose(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file && (!file.size || file.size > 100_000_000)) {pending.value=null;notice.value=uiMessage('account.choose_a_non_empty_file_no_larger_than_100');return;}
  if (file) { pending.value = { file, key: crypto.randomUUID() }; progress.value = 0; notice.value = ''; }
}
async function upload() {
  const selected = pending.value;
  if (!selected || busy.value) return;
  busy.value = true; notice.value = '';
  try {
    let asset = await request<Asset>('uploads', 'POST', {
      project_id: props.projectId, name: selected.file.name,
      expected_bytes: selected.file.size, idempotency_key: selected.key,
    });
    selected.assetId = asset.id;
    if (asset.state !== 'uploading' && asset.state !== 'ready') throw new Error('upload_closed');
    while (asset.state === 'uploading' && asset.size_bytes < selected.file.size && !disposed) {
      const offset = asset.size_bytes;
      asset = await request<Asset>(`uploads/${asset.id}/blocks?offset=${offset}`, 'PUT',
        selected.file.slice(offset, offset + 256 * 1024), true);
      progress.value = selected.file.size ? asset.size_bytes / selected.file.size : 1;
    }
    if (disposed) return;
    // The server computes SHA-256 incrementally; keep browser memory bounded.
    await request<Asset>(`uploads/${asset.id}/finalize`, 'POST', {});
    pending.value = null; progress.value = 1; notice.value = uiMessage('account.saved_on_this_device_and_available_to_the_tools');
    if (fileInput.value) fileInput.value.value = '';
    await refresh();
  } catch (error) { if (!disposed) notice.value = message(error); }
  finally { busy.value = false; }
}
async function remove() {
  if (!removal.value || busy.value) return;
  busy.value = true;
  try {
    const current=removal.value;
    const item=current.storage==='browser'?(await local.remove(current.id),{...current,state:'deleted' as const}):await request<Asset>('assets/' + current.id, 'DELETE');
    if (!disposed) {
      notice.value = item.state === 'deleted' ? uiMessage('account.the_file_was_deleted_and_its_storage_was_released') : uiMessage('account.physical_deletion_has_not_finished_so_the_storage_is');
      if (item.id === pending.value?.assetId && item.state === 'deleted') pending.value = null;
      removal.value = null;
      await refresh();
    }
  } catch (error) { if (!disposed) notice.value = message(error); }
  finally { busy.value = false; }
}
async function confirmRemoval(item: Asset) {
  if (busy.value) return;
  busy.value = true;
  try {
    if(item.storage==='browser'){activeReferences.value=[];removal.value=item;return;}
    const impact = await request<components['schemas']['DeleteImpact']>(`assets/${item.id}/delete-impact`);
    if (!disposed) { activeReferences.value = impact.active_jobs; removal.value = item; }
  } catch(error) { if (!disposed) notice.value = message(error); }
  finally { busy.value = false; }
}
async function fileTask(operation: 'storage_check'|'archive_zip'|'extract_zip', inputs: string[] = []) {
  if (busy.value) return;
  busy.value = true; notice.value = '';
  const limit = Math.floor(outputMegabytes.value * 1_000_000);
  const slot = JSON.stringify([operation, inputs, limit]);
  if (!taskKeys.has(slot)) taskKeys.set(slot, crypto.randomUUID());
  try {
    if(operation==='archive_zip'){
      if(!inputs.length||inputs.length>16||!Number.isSafeInteger(limit)||limit<1)throw new LocalStorageError(uiMessage('account.select_1_16_files_and_set_a_valid_output'));
      const packed:{name:string;raw:ArrayBuffer}[]=[];let size=0;
      for(const id of inputs){const item=assets.value.find(a=>a.id===id);if(!item||!readable(item))throw new LocalStorageError(uiMessage('account.the_selected_files_have_changed_refresh_the_list'));size+=item.size_bytes+new TextEncoder().encode(item.name).length*2+76;if(size>Math.min(limit,100_000_000)-22)throw new LocalStorageError(uiMessage('account.the_archive_exceeds_this_batch_limit_or_the_100'));const response=await transport.fetch('assets/'+encodeURIComponent(id)+'/content',{signal:abort.signal});if(!response.ok)throw Error('unavailable');packed.push({name:item.name,raw:await response.arrayBuffer()});}
      await local.save('项目文件-'+new Date().toISOString().replace(/[:.]/g,'-')+'.zip',new Blob([zipStore(packed)]));notice.value=uiMessage('account.the_archive_was_created_locally_and_can_be_exported');selectedIds.value=[];await refresh();return;
    }
    if (!canWrite.value || !usage.value || !Number.isSafeInteger(limit) || limit < 1 || limit > usage.value.quota_bytes || inputs.length > 16) throw new Error('invalid_request');
    const job = await request<components['schemas']['JobView']>('jobs','POST', {
      project_id: props.projectId, operation, idempotency_key: taskKeys.get(slot),
      config: { inputs, max_output_bytes: limit },
    });
    if (!disposed) { taskKeys.delete(slot); notice.value = uiMessage('account.task_value_was_submitted_track_it_in_task_history',{v0:(job.id.slice(0,8))}); selectedIds.value = []; }
  } catch(error) { if (!disposed) notice.value = message(error); }
  finally { busy.value = false; }
}
function bytes(value: number) {
  return value < 1_000 ? `${value} B` : value < 1_000_000 ? `${(value / 1_000).toFixed(1)} KB`
    : value < 1_000_000_000 ? `${(value / 1_000_000).toFixed(1)} MB` : `${(value / 1_000_000_000).toFixed(2)} GB`;
}
function date(value: number) { return new Date(value * 1_000).toLocaleString('zh-CN'); }
async function download(item:Asset){try{if(item.storage==='browser')await local.download(item.id,item.name);else{const response=await transport.fetch('assets/'+encodeURIComponent(item.id)+'/content',{signal:abort.signal});if(!response.ok)throw Error('unavailable');const blob=await response.blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=item.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);await refresh();}}catch(error){notice.value=message(error);}}
function readable(item: Asset) { return item.state === 'ready' && (item.storage==='browser'||item.expires_at * 1_000 > Date.now()); }
function label(item: Asset) {
  if(item.storage==='browser')return tr('account.saved_locally');
  if (item.expires_at * 1_000 <= Date.now() && item.state !== 'delete_failed') return tr('account.expired_awaiting_cleanup');
  return ({ uploading: tr('account.upload_incomplete'), ready: tr('account.available_to_download'), deleting: tr('account.deleting'), delete_failed: tr('account.deletion_incomplete'), deleted: tr('account.deleted') })[item.state];
}
function refreshVisible(){if(!document.hidden)void refresh();}
onMounted(() => {
  refreshVisible(); timer = setInterval(refreshVisible, 5000);document.addEventListener('visibilitychange',refreshVisible);
  void request<components['schemas']['Capabilities']>('capabilities').then(c => { if (!disposed) serverFileJobs.value = c.task_operations.includes('archive_zip'); }).catch(() => {});
});
onUnmounted(() => { disposed = true; abort.abort(); clearInterval(timer);document.removeEventListener('visibilitychange',refreshVisible);pending.value = null; });
</script>

<template>
  <section class="project-storage" aria-labelledby="storage-heading">
    <div class="storage-heading"><h3 id="storage-heading">{{tr('account.project_files_70')}}</h3><button :disabled="busy" @click="refresh">{{tr('account.refresh_storage')}}</button></div>
    <p class="muted">{{tr('account.files_and_locally_computed_results_are_stored_in_this')}}</p>
    <p v-if="!compact && usage?.workspace_policy_version===1" class="muted">{{tr('account.temporary_storage_per_account')}} {{bytes(usage.quota_bytes)}}{{tr('account.cleared_after_logout_or_after_the_page_is_closed')}}</p>
    <p v-else-if="!compact && usage" class="muted">{{tr('account.new_policy_1_gb_1_000_000_000_bytes')}}</p>
    <p v-if="!usage && !notice" role="status" class="muted">{{tr('account.loading_files_and_storage')}}</p>
    <div v-if="usage" class="storage-meter">
      <div>{{tr('account.temporary_server_storage')}}<strong>{{ bytes(usage.used_bytes) }}</strong> {{tr('account.used')}} {{ bytes(usage.reserved_bytes) }} {{tr('account.reserved')}}</div>
      <progress :value="usage.used_bytes + usage.reserved_bytes" :max="usage.quota_bytes" :aria-label="tr('account.account_file_storage')" />
      <small>{{tr('account.available')}} {{ bytes(usage.available_bytes) }} / {{ bytes(usage.quota_bytes) }} {{tr('account.includes_uploads_results_caches_temporary_files_and_reservations_across')}}</small>
      <p v-if="!policyReady" class="hint">{{tr('account.the_service_has_not_confirmed_the_new_policy_the')}}</p>
      <p v-if="overQuota" class="hint">{{tr('account.current_usage_and_reservations_exceed_the_quota_so_new')}}</p>
      <p v-if="usage.frozen || !usage.ready" class="hint">{{tr('account.new_writes_are_temporarily_unavailable_while_storage_is_checked')}}</p>
    </div>
    <p v-if="notice" role="status" class="storage-notice">{{ notice }}</p>
    <component :is="compact?'details':'div'" class="storage-import">
    <summary v-if="compact">{{tr('account.import_local_files')}}</summary>
    <div class="storage-upload">
      <label>{{tr('account.choose_local_files')}}<input ref="fileInput" type="file" :disabled="busy" @change="choose" /></label>
      <button class="primary" :disabled="busy || !pending" @click="upload">{{ busy ? tr('account.processing') : pending?.assetId ? tr('account.continue_importing') : tr('account.import_files') }}</button>
      <progress v-if="busy && pending" :value="progress" :max="1" :aria-label="tr('account.import_progress')" />
    </div>
    <p class="hint">{{tr('account.local_imports_are_limited_to_100_mb_per_file')}}</p>
    <p v-if="usage?.workspace_policy_version===1" class="hint">{{tr('account.while_the_page_remains_connected_heartbeats_extend_the_workspace')}}</p>
    <p v-else-if="usage" class="hint">{{tr('account.temporary_server_files_are_cleaned_up_at_the_expiry')}}</p>
    </component>
    <div v-if="fileJobs" class="file-job-controls">
      <label>{{tr('account.batch_output_limit_mb')}}<input v-model.number="outputMegabytes" type="number" min="1" max="100" :disabled="busy" /></label>
      <div class="file-actions"><button :disabled="busy || !selectedIds.length || selectedIds.length > 16" @click="fileTask('archive_zip', [...selectedIds])">{{tr('account.archive_selected_files_locally')}}{{ selectedIds.length }}）</button><button v-if="serverFileJobs" :disabled="busy || !canWrite" @click="fileTask('storage_check')">{{tr('account.run_storage_workflow_check')}}</button></div>
      <p class="hint">{{tr('account.zip_archives_support_up_to_16_entries_zip64_encryption')}}</p>
    </div>
    <label class="storage-sort">{{tr('account.sort')}}<select v-model="order" @change="refresh"><option value="expires">{{tr('account.earliest_expiry')}}</option><option value="size">{{tr('account.largest_first')}}</option><option value="created">{{tr('account.newest_uploads')}}</option></select></label>
    <p v-if="available && !assets.length" class="muted">{{tr('account.this_project_has_no_files_yet')}}</p>
    <ul v-else class="storage-files">
      <li v-for="item in assets" :key="item.id">
        <input v-if="fileJobs && readable(item)" v-model="selectedIds" type="checkbox" :value="item.id" :aria-label="tr('account.select_value',{v0:(item.name)})" :disabled="busy" />
        <div class="file-details"><strong>{{ item.name }}</strong><small>{{ item.storage==='browser'?tr('account.local_file'):({input:tr('account.server_input'),result:tr('account.generated_result'),archive:tr('account.archived_resource'),temporary:tr('account.temporary_file')})[item.kind] }} · {{ bytes(item.size_bytes) }} · {{ label(item) }}</small><small v-if="item.storage!=='browser'">{{tr('account.server')}} {{ date(item.expires_at) }} {{tr('account.expires')}}</small></div>
        <div class="file-actions"><button v-if="readable(item)" :disabled="busy" @click="download(item)">{{tr('account.export')}}</button><button v-if="serverFileJobs && readable(item) && item.name.toLowerCase().endsWith('.zip')" :disabled="busy || !canWrite" @click="fileTask('extract_zip', [item.id])">{{tr('account.extract_zip')}}</button><button :disabled="busy" @click="confirmRemoval(item)">{{ item.state === 'delete_failed' ? tr('account.retry_deletion') : tr('account.delete') }}</button></div>
      </li>
    </ul>
    <ModalDialog v-if="removal" :variant="usage?.workspace_policy_version===1?'confirm':undefined" :title="tr('account.delete_this_file')" @close="removal = null">
      <p class="remove-name">{{ removal.name }}</p><p v-if="removal.storage==='browser'">{{tr('account.this_file_will_be_removed_from_the_current_browser')}}</p><p v-else>{{tr('account.this_will_delete_1_resource_containing')}} {{ bytes(removal.size_bytes) }}{{tr('account.it_is_currently_referenced_by')}} {{ activeReferences.length }} {{tr('account.active_tasks_deletion_will_request_that_these_tasks_stop')}}</p>
      <p v-if="removal.storage!=='browser'" class="hint">{{tr('account.other_successful_results_keep_their_own_expiry_tasks_that')}}</p>
      <p>{{tr('account.the_file_cannot_be_downloaded_after_deletion_downloading_first')}}</p>
      <template #footer><button :disabled="busy" @click="removal = null">{{tr('account.keep_file')}}</button><button :disabled="busy" @click="remove">{{tr('account.confirm_deletion')}}</button></template>
    </ModalDialog>
  </section>
</template>

<style scoped>
.project-storage { margin-top: 1.5rem; border-top: 1px solid var(--border); padding-top: 1rem; min-width: 0; }
.storage-heading,.storage-upload,.file-actions { display: flex; align-items: center; flex-wrap: wrap; gap: .7rem; }
.storage-heading { justify-content: space-between; }
.storage-meter { padding: 1rem 0; }
progress { display: block; width: 100%; accent-color: var(--accent); height: .7rem; margin: .5rem 0; }
.storage-sort { display: flex; gap: .7rem; align-items: center; margin: 1rem 0; }
.storage-files { list-style: none; padding: 0; margin: 0; }
.storage-files li { padding: 1rem 0; border-top: 1px solid var(--border); display: flex; flex-wrap: wrap; align-items: center; gap: .8rem; }
.file-details { flex: 1; min-width: 12rem; display: grid; gap: .35rem; }
.file-details strong,.remove-name { overflow-wrap: anywhere; }
.file-details small,.storage-meter small { color: var(--muted); }
.storage-upload input { width: 100%; max-width: 22rem; }
.storage-notice { line-height: 1.6; }
.file-job-controls { margin: 1rem 0; display: grid; gap: .7rem; }
.file-job-controls input { max-width: 8rem; margin-left: .5rem; }
.storage-files input[type="checkbox"] { width: 1rem; flex: 0 0 1rem; }
</style>
