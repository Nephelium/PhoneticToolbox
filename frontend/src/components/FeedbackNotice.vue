<script setup lang="ts">
import {ref,onMounted,onUnmounted} from 'vue';
import {feedbackCall} from '../platform/feedback.ts';
const emit=defineEmits<{open:[]}>();
const unread=ref(0);let timer:ReturnType<typeof setInterval>|undefined,alive=true,checking=false;
async function check(){if(!alive||checking||document.hidden)return;checking=true;try{const result=await feedbackCall<{unread:number}>('unread');if(alive)unread.value=result.unread;}catch{/* Network failure keeps the last known unread state. */}finally{checking=false;}}
onMounted(()=>{void check();timer=setInterval(()=>void check(),60000);document.addEventListener('visibilitychange',check);});
onUnmounted(()=>{alive=false;clearInterval(timer);document.removeEventListener('visibilitychange',check);});
</script>
<template><button v-if="unread" class="feedback-notice" @click="emit('open')">维护者回复了 {{unread}} 条消息 · 查看反馈</button></template>
<style scoped>.feedback-notice{position:fixed;right:20px;bottom:22px;z-index:40;border-color:var(--accent);background:var(--selected);box-shadow:var(--shadow);max-width:calc(100vw - 40px);white-space:normal}</style>
