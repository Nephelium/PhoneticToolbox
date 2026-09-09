<script setup lang="ts">
import { ref,onMounted,onBeforeUnmount,onUnmounted,useId } from 'vue';
import AppIcon from './AppIcon.vue';
defineProps<{title:string;wide?:boolean}>();const emit=defineEmits<{close:[]}>();const dialog=ref<HTMLDialogElement>();const titleId=useId();let previous:HTMLElement|null=null;
onMounted(()=>{previous=document.activeElement as HTMLElement;dialog.value?.showModal();});
onBeforeUnmount(()=>dialog.value?.close());
onUnmounted(()=>{if(previous?.isConnected)previous.focus();});
</script>
<template>
<dialog ref="dialog" :class="{wide}" :aria-labelledby="titleId" @cancel.prevent="emit('close')">
<header class="dialog-header">
<h2 :id="titleId">{{title}}</h2>
<button class="icon-button" aria-label="关闭对话框" @click="emit('close')">
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
