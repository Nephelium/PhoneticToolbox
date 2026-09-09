<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue';
import type { components } from '../../../contracts/generated/api';
import ModalDialog from '../components/ModalDialog.vue';

type Asset = components['schemas']['AssetView'];
type Usage = components['schemas']['StorageUsage'];
const props = defineProps<{ projectId: string; ownerId: string; csrfToken: string }>();
const emit = defineEmits<{ sessionInvalid: [] }>();
const usage = ref<Usage | null>(null), assets = ref<Asset[]>([]), notice = ref('');
const available = ref(false), busy = ref(false), progress = ref(0), order = ref('expires');
const removal = ref<Asset | null>(null), fileInput = ref<HTMLInputElement>();
const pending = ref<{ file: File; key: string; assetId?: string } | null>(null);
const abort = new AbortController();
let disposed = false, timer: ReturnType<typeof setInterval> | undefined, loading = false;
const errors: Record<string, string> = {
  quota_exceeded: '空间不足。可以先下载或直接删除不再需要的文件，再继续上传。',
  disk_space_low: '服务器磁盘空间不足，现有文件仍可下载和清理。',
  storage_service_unavailable: '文件服务尚未启用或暂时无法连接。',
  storage_recovery_required: '文件服务正在核对存储，请稍后再试。',
  storage_busy: '文件服务正在处理其他请求，请稍后重试。',
  upload_closed: '这次上传已关闭或到期，请重新选择文件。',
  asset_expired: '文件已到期，不能继续下载。',
  asset_limit_reached: '文件记录数量已达上限，请联系管理员。',
};
function message(error: unknown) {
  const code = error instanceof Error ? error.message : '';
  return errors[code] || '操作未确认，请稍后重试。当前上传保留请求标识，不会重复创建文件。';
}
async function request<T>(path: string, method = 'GET', body?: unknown, binary = false): Promise<T> {
  const response = await fetch('/api/v1/' + path, {
    method, credentials: 'same-origin', signal: abort.signal,
    headers: { 'Content-Type': binary ? 'application/octet-stream' : 'application/json',
      'X-PTB-Account': props.ownerId, ...(method !== 'GET' ? { 'X-CSRF-Token': props.csrfToken } : {}) },
    body: body === undefined ? undefined : binary ? body as Blob : JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok) {
    if (response.status === 401 || value.detail === 'account_changed') {
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
    // The server computes SHA-256 incrementally. Never load a 5 GB file into JS memory.
    await request<Asset>(`uploads/${asset.id}/finalize`, 'POST', {});
    pending.value = null; progress.value = 1; notice.value = '上传完成。文件保留 7 天，下载不会延长到期时间。';
    if (fileInput.value) fileInput.value.value = '';
    await refresh();
  } catch (error) { if (!disposed) notice.value = message(error); }
  finally { busy.value = false; }
}
async function remove() {
  if (!removal.value || busy.value) return;
  busy.value = true;
  try {
    const item = await request<Asset>('assets/' + removal.value.id, 'DELETE');
    if (!disposed) {
      notice.value = item.state === 'deleted' ? '文件已删除，占用空间已释放。' : '物理删除未完成，空间仍计入占用。可稍后重试。';
      if (item.id === pending.value?.assetId && item.state === 'deleted') pending.value = null;
      removal.value = null;
      await refresh();
    }
  } catch (error) { if (!disposed) notice.value = message(error); }
  finally { busy.value = false; }
}
function bytes(value: number) {
  return value < 1_000 ? `${value} B` : value < 1_000_000 ? `${(value / 1_000).toFixed(1)} KB`
    : value < 1_000_000_000 ? `${(value / 1_000_000).toFixed(1)} MB` : `${(value / 1_000_000_000).toFixed(2)} GB`;
}
function date(value: number) { return new Date(value * 1_000).toLocaleString('zh-CN'); }
function readable(item: Asset) { return item.state === 'ready' && item.expires_at * 1_000 > Date.now(); }
function label(item: Asset) {
  if (item.expires_at * 1_000 <= Date.now() && item.state !== 'delete_failed') return '已到期 · 等待清理';
  return ({ uploading: '上传未完成', ready: '可下载', deleting: '正在删除', delete_failed: '删除未完成', deleted: '已删除' })[item.state];
}
onMounted(() => { void refresh(); timer = setInterval(() => { void refresh(); }, 5000); });
onUnmounted(() => { disposed = true; abort.abort(); clearInterval(timer); pending.value = null; });
</script>

<template>
  <section class="project-storage" aria-labelledby="storage-heading">
    <div class="storage-heading"><h3 id="storage-heading">项目文件</h3><button :disabled="busy" @click="refresh">刷新空间</button></div>
    <p class="muted">每个账号共享 5 GB 空间。文件最多保留 7 天，可先下载，也可直接删除。</p>
    <div v-if="usage" class="storage-meter">
      <div><strong>{{ bytes(usage.used_bytes) }}</strong> 已占用 · {{ bytes(usage.reserved_bytes) }} 已预留</div>
      <progress :value="usage.used_bytes + usage.reserved_bytes" :max="usage.quota_bytes" aria-label="账号文件空间" />
      <small>可用 {{ bytes(usage.available_bytes) }} / 5 GB · 包含所有项目和未完成上传</small>
      <p v-if="usage.frozen || !usage.ready" class="hint">新增写入暂不可用，正在等待存储核对。</p>
    </div>
    <p v-if="notice" role="status" class="storage-notice">{{ notice }}</p>
    <div class="storage-upload">
      <label>选择上传文件<input ref="fileInput" type="file" :disabled="busy || !available || usage?.frozen || !usage?.ready" @change="choose" /></label>
      <button class="primary" :disabled="busy || !pending || !available || usage?.frozen || !usage?.ready" @click="upload">{{ busy ? '正在处理…' : pending?.assetId ? '继续上传' : '上传文件' }}</button>
      <progress v-if="busy && pending" :value="progress" :max="1" aria-label="上传进度" />
    </div>
    <p class="hint">未完成上传最多保留 24 小时；关闭页面不会删除电脑上的原文件。当前上传用于文件管理，尚未接入语音分析。</p>
    <label class="storage-sort">排序<select v-model="order" @change="refresh"><option value="expires">最早到期</option><option value="size">占用最大</option><option value="created">最新上传</option></select></label>
    <p v-if="available && !assets.length" class="muted">此项目还没有文件。</p>
    <ul v-else class="storage-files">
      <li v-for="item in assets" :key="item.id">
        <div class="file-details"><strong>{{ item.name }}</strong><small>{{ bytes(item.size_bytes) }} · {{ label(item) }}</small><small>{{ date(item.expires_at) }} 到期</small></div>
        <div class="file-actions"><a v-if="readable(item)" :href="`/api/v1/assets/${item.id}/content?expected_account=${ownerId}`" download>下载</a><button :disabled="busy" @click="removal = item">{{ item.state === 'delete_failed' ? '重试删除' : '删除' }}</button></div>
      </li>
    </ul>
    <ModalDialog v-if="removal" title="删除这个文件？" @close="removal = null">
      <p class="remove-name">{{ removal.name }}</p><p>将删除 1 个上传资源，已写入 {{ bytes(removal.size_bytes) }}。本阶段尚未关联分析任务或结果。</p>
      <p>删除后不能再下载。无需先下载；如果需要留存，请先保存到自己的电脑。</p>
      <template #footer><button :disabled="busy" @click="removal = null">保留文件</button><button :disabled="busy" @click="remove">确认删除</button></template>
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
</style>
