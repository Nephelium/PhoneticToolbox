<script setup lang="ts">
import {ref,onUnmounted} from 'vue';
import type {ResearchFiles} from '../platform/research.ts';
import AppIcon from './AppIcon.vue';
const props=defineProps<{files:ResearchFiles;accept:string;folders?:boolean}>();
const emit=defineEmits<{uploaded:[]}>();
const picker=ref<HTMLInputElement>(),folder=ref<HTMLInputElement>(),busy=ref(false),progress=ref(0),error=ref(''),notice=ref('');
let pending:File[]=[],disposed=false;
async function choose(event:Event){const input=event.target as HTMLInputElement;pending=Array.from(input.files??[]);input.value='';if(pending.length)await upload();}
async function upload(){
 if(busy.value||!props.files.upload)return;busy.value=true;error.value='';notice.value='';progress.value=0;
 try{await props.files.upload(pending,value=>{if(!disposed)progress.value=value;});if(!disposed){notice.value=`已导入 ${pending.length} 个文件`;pending=[];}}
 catch(e){if(!disposed)error.value=e instanceof Error?e.message:'导入未完成，请重试。';}
 finally{if(!disposed){busy.value=false;emit('uploaded');}}
}
onUnmounted(()=>{disposed=true;});
</script>
<template><div v-if="files.upload" class="research-upload" :aria-busy="busy">
 <input ref="picker" type="file" :accept="accept" multiple hidden aria-label="选择要导入的文件" @change="choose"/>
 <button class="primary" :disabled="busy" @click="picker?.click()"><AppIcon name="folder"/>{{busy?'正在导入 '+Math.round(progress*100)+'%':'导入文件'}}</button>
 <template v-if="folders"><input ref="folder" type="file" multiple webkitdirectory hidden @change="choose"/><button :disabled="busy" @click="folder?.click()">导入文件夹</button></template>
 <span v-if="notice" role="status">{{notice}}</span><span v-if="error" role="alert" class="upload-error">{{error}} <button :disabled="busy" @click="upload">重试导入</button></span>
</div></template>
<style scoped>
.research-upload{display:flex;gap:8px;align-items:center;flex-wrap:wrap;min-width:0}.research-upload>span{font-size:.82rem;color:var(--muted);overflow-wrap:anywhere}.research-upload>.upload-error{color:var(--danger)}.research-upload button{display:inline-flex;gap:7px;align-items:center}
</style>
