<script setup lang="ts">
import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {ref,onUnmounted} from 'vue';
import type {ResearchFiles} from '../platform/research.ts';
import AppIcon from './AppIcon.vue';
const props=defineProps<{files:ResearchFiles;accept:string;folders?:boolean}>();
const emit=defineEmits<{uploaded:[]}>();
const picker=ref<HTMLInputElement>(),folder=ref<HTMLInputElement>(),busy=ref(false),progress=ref(0),error=messageRef(),notice=messageRef();
let pending:File[]=[],disposed=false;
async function choose(event:Event){const input=event.target as HTMLInputElement;pending=Array.from(input.files??[]);input.value='';if(pending.length)await upload();}
async function upload(){
 if(busy.value||!props.files.upload)return;busy.value=true;error.value='';notice.value='';progress.value=0;
 try{await props.files.upload(pending,value=>{if(!disposed)progress.value=value;});if(!disposed){notice.value=uiMessage('components.imported_value_files',{v0:(pending.length)});pending=[];}}
 catch(e){if(!disposed)error.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.import_did_not_finish_try_again');}
 finally{if(!disposed){busy.value=false;emit('uploaded');}}
}
onUnmounted(()=>{disposed=true;});
</script>
<template><div v-if="files.upload" class="research-upload" :aria-busy="busy">
 <input ref="picker" type="file" :accept="accept" multiple hidden :aria-label="tr('components.choose_files_to_import')" @change="choose"/>
 <button class="primary" :disabled="busy" @click="picker?.click()"><AppIcon name="folder"/>{{busy?tr('components.importing')+Math.round(progress*100)+'%':tr('components.import_files')}}</button>
 <template v-if="folders"><input ref="folder" type="file" multiple webkitdirectory hidden @change="choose"/><button :disabled="busy" @click="folder?.click()">{{tr('components.import_folder')}}</button></template>
 <span v-if="notice" role="status">{{notice}}</span><span v-if="error" role="alert" class="upload-error">{{error}} <button :disabled="busy" @click="upload">{{tr('components.retry_import')}}</button></span>
</div></template>
<style scoped>
.research-upload{display:flex;gap:8px;align-items:center;flex-wrap:wrap;min-width:0}.research-upload>span{font-size:.82rem;color:var(--muted);overflow-wrap:anywhere}.research-upload>.upload-error{color:var(--danger)}.research-upload button{display:inline-flex;gap:7px;align-items:center}
</style>
