<script setup lang="ts">
import { ref,onMounted,onBeforeUnmount,onUnmounted,useId } from 'vue';
import AppIcon from './AppIcon.vue';
defineProps<{title:string;wide?:boolean;closeDisabled?:boolean;variant?:'update'}>();const emit=defineEmits<{close:[]}>();const dialog=ref<HTMLDialogElement>();const titleId=useId();let previous:HTMLElement|null=null;
onMounted(()=>{previous=document.activeElement as HTMLElement;dialog.value?.showModal();});
onBeforeUnmount(()=>dialog.value?.close());
onUnmounted(()=>{if(previous?.isConnected)previous.focus();});
</script>
<template>
<dialog ref="dialog" :class="{wide,'update-dialog':variant==='update'}" :aria-labelledby="titleId" @cancel.prevent="!closeDisabled&&emit('close')">
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
dialog.update-dialog{width:min(780px,calc(100vw / var(--page-scale,1) - 24px));border-radius:16px}
.update-dialog>.dialog-header{padding:24px 28px 8px;border-bottom:0}
.update-dialog>.dialog-header h2{font-size:1.5rem;font-weight:700}
.update-dialog>.dialog-body{padding:0 28px 24px;scrollbar-gutter:auto}
@media(max-width:620px){.update-dialog>.dialog-header{padding:18px 18px 8px}.update-dialog>.dialog-body{padding:0 18px 18px}}
</style>
