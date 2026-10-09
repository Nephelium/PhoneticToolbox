<script setup lang="ts">
import {onMounted,ref} from 'vue';
import {localStoragePort,type StorageStatus} from '../platform/storage.ts';
const state=ref<StorageStatus>(),enabled=ref(true),days=ref('30'),exportedDays=ref('7');
const busy=ref(false),message=ref(''),error=ref('');
const confirmAll=ref(false);
async function clearAll(){busy.value=true;error.value='';message.value='';try{const answer=await localStoragePort().clearAll();confirmAll.value=false;message.value=answer.started?'正在检查未保存内容，安全退出后清除缓存。其他窗口仍打开时，运行文件会等最后一个窗口退出后清理。':'退出请求未开始，请稍后重试。';}catch(e){error.value=e instanceof Error?e.message:'清理未开始，请稍后重试。';}finally{busy.value=false;}}
function size(bytes:number){return bytes>=1024**3?`${(bytes/1024**3).toFixed(2)} GB`:`${(bytes/1024**2).toFixed(1)} MB`;}
function assign(value:StorageStatus){state.value=value;enabled.value=value.enabled;days.value=String(value.days);exportedDays.value=String(value.exported_cache_days);}
async function load(){busy.value=true;error.value='';try{assign(await localStoragePort().status());}catch(e){error.value=e instanceof Error?e.message:'读取缓存设置失败。';}finally{busy.value=false;}}
async function save(){
  const a=Number(days.value),b=Number(exportedDays.value);
  if(!Number.isInteger(a)||a<1||a>3650||!Number.isInteger(b)||b<1||b>3650){error.value='保留期限请输入 1–3650 天的整数。';return;}
  busy.value=true;error.value='';message.value='';
  try{assign(await localStoragePort().configure({enabled:enabled.value,days:a,exported_cache_days:b}));message.value='缓存设置已保存。期限缩短后，下一次清理会应用新期限。';}
  catch(e){error.value=e instanceof Error?e.message:'缓存设置保存失败，原设置保留。';}finally{busy.value=false;}
}
async function clean(){busy.value=true;error.value='';message.value='';
  try{const answer=await localStoragePort().clean();assign(await localStoragePort().status());message.value=answer.skipped?'自动清理已关闭，文件均保留。':`本次清理 ${answer.count} 个缓存文件，释放 ${size(answer.bytes)}。${answer.protected_count?`另有 ${answer.protected_count} 个文件仍被任务使用，已保留。`:''}${answer.failed_count?`另有 ${answer.failed_count} 个文件暂时无法清理，已保留，可稍后重试。`:''}`;}
  catch(e){error.value=e instanceof Error?e.message:'清理失败，未确认可清理的文件保留。';}finally{busy.value=false;}
}
onMounted(load);
</script>
<template><section class="storage-settings" aria-labelledby="storage-title">
  <h2 id="storage-title">本机存储与自动清理</h2>
  <p>程序更新后会继续读取原有设置。录音工程、实验项目、TextGrid 标注和正式导出文件由你管理。</p>
  <label><input v-model="enabled" type="checkbox" :disabled="busy"/>自动清理过期的处理结果缓存</label>
  <div class="storage-fields"><label>未导出结果保留 <input v-model="days" type="number" min="1" max="3650" step="1" :disabled="busy"/> 天</label>
  <label>已导出副本的缓存保留 <input v-model="exportedDays" type="number" min="1" max="3650" step="1" :disabled="busy"/> 天</label></div>
  <p class="hint">保留期限从生成或最近读取结果开始计算。旧结果缺少可靠时间时，从本次首次登记开始计时。进行中的任务及仍依赖的上游结果会保留，配套文件整组清理。唇形录制结果受额外保护。</p>
  <p class="hint">已导出后的清理仅涉及应用内部副本，不会打开或删除你的导出目录。未导出结果到期清理后，需要重新计算才能恢复。</p>
  <p class="hint">MFA 对齐的诊断临时目录完成后保留 7 天，再随此开关定期清理。正在运行、尚未完成或归属无法确认的目录保留，已登记的模型和词典保留。</p>
  <dl v-if="state"><div><dt>处理结果缓存</dt><dd>{{size(state.result_bytes)}} · {{state.result_count}} 个文件</dd></div><div><dt>受保护的输入副本</dt><dd>{{size(state.input_bytes)}}</dd></div><div v-if="state.diagnostic_bytes!==undefined"><dt>MFA 诊断临时文件</dt><dd>{{size(state.diagnostic_bytes)}} · {{state.diagnostic_count??0}} 个目录</dd></div></dl>
  <p v-if="message" role="status">{{message}}</p><p v-if="error" class="error-banner" role="alert">{{error}}</p>
  <div class="storage-actions"><button class="primary" :disabled="busy" @click="save">保存清理设置</button><button :disabled="busy||!state||!state.enabled" @click="clean">清理已到期缓存</button><button :disabled="busy" @click="load">刷新占用</button></div>
  <div class="all-caches"><h3>清除全部缓存</h3><p class="hint">清理程序运行文件、下载的更新包及可清理的内部处理结果副本，并安全退出。未导出的处理结果清理后需要重新计算。设置、草稿、原始输入、录音、工程及正式导出文件保留。</p><p class="hint">免安装版删掉 EXE 前可先在这里清理。再次打开时会重新准备运行文件，需要多等一会儿。安装版卸载时也会清理运行缓存。</p>
  <button v-if="!confirmAll" :disabled="busy" @click="confirmAll=true">清除全部缓存…</button>
  <div v-else role="alertdialog" aria-label="确认清除全部缓存"><p>请先保存需要的处理结果。确认后将检查未保存内容，退出并清理缓存。</p><div class="storage-actions"><button :disabled="busy" @click="clearAll">清除全部缓存并退出</button><button :disabled="busy" @click="confirmAll=false">取消</button></div></div></div>
</section></template>
<style scoped>
.storage-settings{margin-top:1.5rem;padding:1.25rem;border:1px solid var(--border);border-radius:var(--radius);background:var(--panel)}
.all-caches{margin-top:1.2rem;border-top:1px solid var(--border);padding-top:.8rem}.all-caches h3{font-size:1rem;margin:.2rem 0}
h2{font-size:1.15rem;margin:0 0 .8rem}p{line-height:1.75}.storage-fields{display:flex;flex-wrap:wrap;gap:1rem;margin:.9rem 0}.storage-fields input{width:5.5rem;margin:0 .25rem}.storage-actions{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:.8rem}dl{display:flex;gap:2rem;flex-wrap:wrap}dt{color:var(--muted)}dd{margin:.4rem 0}.hint{color:var(--muted)}
</style>
