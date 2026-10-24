<script setup lang="ts">
import {useId} from 'vue';
import ModalDialog from '../../components/ModalDialog.vue';
import type {PaperAnnotation} from '../../platform/papers.ts';
defineProps<{page:number;color:PaperAnnotation['color'];modelValue:string;saving:boolean;error:string}>();
const emit=defineEmits<{'update:modelValue':[string];save:[];close:[];reload:[]}>();
const inputId=useId();
</script>
<template>
 <ModalDialog class="annotation-note-dialog" :class="'note-'+color" :title="`第 ${page} 页 · 便签`" :close-disabled="saving" @close="emit('close')">
  <label :for="inputId" class="note-input-label">批注内容</label>
  <textarea :id="inputId" :value="modelValue" aria-label="批注内容" autofocus rows="7" maxlength="5000" placeholder="记下一点想法…" @input="emit('update:modelValue',($event.target as HTMLTextAreaElement).value)"/>
  <p v-if="error" class="note-error" role="alert">{{error}} <button :disabled="saving" @click="emit('reload')">重新载入批注</button></p>
  <template #footer><button :disabled="saving" @click="emit('close')">取消</button><button class="primary" :disabled="saving||!modelValue.trim()" @click="emit('save')">{{saving?'正在保存…':'保存批注'}}</button></template>
 </ModalDialog>
</template>
<style scoped>
dialog.annotation-note-dialog{--note-paper:#fff4c5;--note-rule:#d7bf6d;--note-ink:#514021;width:min(500px,calc(100vw / var(--page-scale,1) - 32px));background:var(--note-paper);color:var(--note-ink);border:1px solid var(--note-rule);border-radius:5px 18px 5px 5px;box-shadow:0 18px 55px #34260b38;padding:0;overflow:hidden}
dialog.note-green{--note-paper:#e2f3d9;--note-rule:#a7c79a;--note-ink:#29442b}
dialog.note-pink{--note-paper:#fbe3e9;--note-rule:#d7b0bd;--note-ink:#643e4d}
.annotation-note-dialog :deep(.dialog-header){padding:18px 22px 8px;border:0;background:transparent;gap:16px}
.annotation-note-dialog :deep(.dialog-header h2){font-size:.93rem;font-weight:500;color:inherit;letter-spacing:.04em}
.annotation-note-dialog :deep(.dialog-header .icon-button){color:inherit;background:transparent;border-color:transparent;min-width:28px;min-height:28px}
.annotation-note-dialog :deep(.dialog-body){padding:12px 24px 18px}
.annotation-note-dialog :deep(.dialog-actions){padding:12px 22px 16px;border-top:1px solid var(--note-rule);background:transparent;gap:8px}
.annotation-note-dialog :deep(.dialog-actions button:not(.primary)){background:transparent;color:var(--note-ink);border-color:var(--note-rule)}
.annotation-note-dialog :deep(.dialog-actions .primary){background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.note-input-label{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
textarea{display:block;width:100%;min-height:160px;max-height:45vh;resize:vertical;border:0;border-radius:3px;background:transparent;color:inherit;font:inherit;line-height:1.95;padding:4px 2px;box-shadow:none}
textarea::placeholder{color:inherit;opacity:.5}textarea:focus-visible{outline:1px solid var(--note-rule);outline-offset:6px}
.note-error{font-size:.86rem;line-height:1.6}.note-error button{font:inherit}
[data-theme=dark] dialog.annotation-note-dialog{--note-paper:#393425;--note-rule:#796a3b;--note-ink:#f4e6b8;box-shadow:0 20px 65px #0006}
[data-theme=dark] dialog.note-green{--note-paper:#29372b;--note-rule:#5c7857;--note-ink:#dfefd5}
[data-theme=dark] dialog.note-pink{--note-paper:#3e2e36;--note-rule:#805e70;--note-ink:#f6dce6}
@media(prefers-reduced-motion:no-preference){dialog.annotation-note-dialog[open]{animation:note-appear .13s ease-out}@keyframes note-appear{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}}
</style>
