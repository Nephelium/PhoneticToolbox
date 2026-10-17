<script setup lang="ts">
import { ref,onMounted,onBeforeUnmount,onUnmounted,useId } from 'vue';
import AppIcon from './AppIcon.vue';
defineProps<{title:string;wide?:boolean;closeDisabled?:boolean;variant?:'update'|'confirm'|'feedback'}>();const emit=defineEmits<{close:[]}>();const dialog=ref<HTMLDialogElement>();const titleId=useId();let previous:HTMLElement|null=null;
onMounted(()=>{previous=document.activeElement as HTMLElement;dialog.value?.showModal();});
onBeforeUnmount(()=>dialog.value?.close());
onUnmounted(()=>{if(previous?.isConnected)previous.focus();});
</script>
<template>
<dialog ref="dialog" :class="{wide,'update-dialog':variant==='update','confirm-dialog':variant==='confirm','feedback-dialog':variant==='feedback'}" :aria-labelledby="titleId" @cancel.prevent="!closeDisabled&&emit('close')">
<header class="dialog-header">
<h2 :id="titleId">{{title}}</h2>
<button class="icon-button" aria-label="关闭对话框" :disabled="closeDisabled" @click="emit('close')">
<AppIcon name="close"/>
</button>
</header>
<div class="dialog-body">
<slot/>
</div>
<footer v-if="$slots.footer" class="dialog-actions">
<slot name="footer"/>
</footer>
</dialog>
</template>
<style scoped>
dialog.feedback-dialog{width:min(720px,calc(100vw / var(--page-scale,1) - 32px));height:fit-content;max-height:calc(90dvh / var(--page-scale,1));border-radius:20px;box-shadow:0 24px 90px #0004}
.feedback-dialog>.dialog-header{padding:20px 24px 14px;border-bottom:1px solid var(--border)}
.feedback-dialog>.dialog-header h2{font-size:1.2rem}
.feedback-dialog>.dialog-body{padding:24px;overflow:auto}
@media(max-width:600px){.feedback-dialog>.dialog-body{padding:18px}.feedback-dialog>.dialog-header{padding:16px 18px}}
dialog.confirm-dialog{width:min(440px,calc(100vw / var(--page-scale,1) - 32px));border-radius:16px;box-sizing:border-box}
.confirm-dialog>.dialog-body{padding:20px 24px;line-height:1.8;overflow-wrap:anywhere}
.confirm-dialog>.dialog-header h2{font-size:1.2rem;line-height:1.5;min-width:0}
.confirm-dialog>.dialog-actions{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));padding:16px 24px;gap:10px}
.confirm-dialog>.dialog-actions :deep(button){white-space:normal;overflow-wrap:anywhere;min-width:0;min-height:38px}
.confirm-dialog>.dialog-actions :deep(button.primary){grid-column:1/-1}
dialog.update-dialog{width:min(780px,calc(100vw / var(--page-scale,1) - 24px));border-radius:16px}
.update-dialog>.dialog-header{padding:24px 28px 8px;border-bottom:0}
.update-dialog>.dialog-header h2{font-size:1.5rem;font-weight:700}
.update-dialog>.dialog-body{padding:0 28px 24px;scrollbar-gutter:auto}
@media(max-width:620px){.update-dialog>.dialog-header{padding:18px 18px 8px}.update-dialog>.dialog-body{padding:0 18px 18px}}
</style>
