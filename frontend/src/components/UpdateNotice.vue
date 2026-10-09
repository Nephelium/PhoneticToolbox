<script setup lang="ts">
import {onMounted,onBeforeUnmount,ref} from 'vue';
import {updater,updatesAvailable,onUpdatesAvailability,type UpdateCheck,type UpdateRelease} from '../platform/updates';
const props=withDefaults(defineProps<{enabled?:boolean}>(),{enabled:true});
const emit=defineEmits<{open:[release:UpdateRelease];checked:[result:UpdateCheck]}>();
const release=ref<UpdateRelease>(),dismissed=ref(false);
let checked=false,abort:AbortController|undefined;
async function checkStartup(){
  if(checked||!props.enabled||!updatesAvailable())return;
  checked=true;abort=new AbortController();
  try{const result=await updater.check({manual:false},abort.signal);emit('checked',result);if(result.shouldPrompt&&result.candidate){release.value=result.candidate;await updater.acknowledge(result.candidate.id);}}
  catch{/* Startup network failures stay available in manual check; no disruptive dialog. */}
  finally{abort=undefined;}
}
const unsubscribe=onUpdatesAvailability(()=>{void checkStartup();});
onMounted(()=>{void checkStartup();});
onBeforeUnmount(()=>{abort?.abort();unsubscribe();});
</script>
<template><aside v-if="release&&!dismissed" class="update-notice" role="status" aria-label="新版本提示"><div><strong>新版本 {{release.version}} 已发布</strong><p>查看更新说明，确认后下载。</p></div><button class="primary" @click="emit('open',release);dismissed=true">查看更新</button><button class="icon-button" aria-label="今天稍后再说" @click="dismissed=true">×</button></aside></template>
<style scoped>
.update-notice{position:fixed;right:20px;bottom:42px;z-index:110;display:flex;align-items:center;gap:14px;max-width:min(620px,calc(100vw - 40px));padding:15px 16px;background:var(--panel);border:1px solid var(--accent);border-radius:var(--radius);box-shadow:var(--shadow)}.update-notice strong{color:var(--accent)}.update-notice p{color:var(--muted);font-size:.928571rem;margin-top:4px}.update-notice .icon-button{font-size:1.5rem}@media(max-width:540px){.update-notice{flex-wrap:wrap;bottom:20px;right:12px;max-width:calc(100vw - 24px)}.update-notice>div{width:calc(100% - 44px)}.update-notice .icon-button{position:absolute;right:10px;top:10px}}
</style>
