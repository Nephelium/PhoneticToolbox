<script setup lang="ts">
import ModalDialog from './ModalDialog.vue';
import type {Announcement} from '../platform/announcements.ts';
defineProps<{items:Announcement[];automatic?:boolean;unreadIds:string[];loading?:boolean}>();
const emit=defineEmits<{close:[];action:[id:string]}>();
</script>
<template>
<ModalDialog title="公告" @close="emit('close')">
<div class="announcements-panel">
<p v-if="loading&&!items.length" class="muted" role="status">正在读取公告…</p>
<p v-else-if="!items.length" class="muted">暂时没有公告。</p>
<article v-for="item in items" :key="item.id" class="announcement-item">
<header><h3>{{item.title}}</h3><span v-if="unreadIds.includes(item.id)" class="announcement-unread">新</span></header>
<time :datetime="item.publishedAt">{{item.publishedAt.slice(0,10)}}</time>
<p class="announcement-body">{{item.body}}</p>
<a v-if="item.action&&!automatic" :href="item.action.url" target="_blank" rel="noopener noreferrer" class="announcement-link" @click="emit('action',item.id)">{{item.action.label}} ↗</a>
</article>
</div>
<template #footer>
<button type="button" @click="emit('close')">知道了</button>
<a v-if="automatic&&items[0]?.action" :href="items[0].action.url" target="_blank" rel="noopener noreferrer" class="primary announcement-primary" @click="emit('action',items[0].id)">{{items[0].action.label}} ↗</a>
</template>
</ModalDialog>
</template>
<style scoped>
.announcements-panel{display:grid;gap:22px;min-width:0;overflow-wrap:anywhere}.announcement-item+article{border-top:1px solid var(--border);padding-top:22px}.announcement-item header{display:flex;align-items:baseline;gap:10px}.announcement-item h3{font-size:1.15rem;line-height:1.5;margin:0}.announcement-item time{display:block;margin-top:6px;color:var(--muted);font-size:.86rem}.announcement-body{white-space:pre-wrap;line-height:1.9;margin:16px 0 0}.announcement-unread{flex:none;padding:1px 7px;border-radius:12px;background:var(--selected);color:var(--accent);font-size:.8rem}.announcement-link{display:inline-block;margin-top:14px;color:var(--accent)}.announcement-primary{display:inline-flex;align-items:center;justify-content:center;min-height:34px;border:1px solid var(--accent);border-radius:6px;padding:6px 12px;background:var(--accent);color:var(--on-accent);text-decoration:none;white-space:normal;text-align:center;overflow-wrap:anywhere}.announcement-primary:hover{filter:brightness(.94)}
:global(html[data-button-style=plain]) .announcement-primary{background:var(--panel);color:var(--text);border-color:var(--border)}
@media(max-width:540px){.announcement-item h3{font-size:1.05rem}.announcement-body{line-height:1.75}.announcement-primary{max-width:100%;box-sizing:border-box}}
</style>
