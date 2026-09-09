<script setup lang="ts">
import { computed,ref } from 'vue';import sources from '../generated/sources.json';
const props=defineProps<{moduleId?:string}>();const query=ref(''),message=ref('');
const matches=computed(()=>sources.filter(s=>(!props.moduleId||s.modules.includes(props.moduleId))&&(s.title+' '+s.authors).toLowerCase().includes(query.value.toLowerCase())));
async function copy(text:string){try{await navigator.clipboard.writeText(text);message.value='引用已复制';}catch{message.value='无法访问剪贴板，请选中引用文字复制。';}}
</script>
<template>
<p class="muted">来源与版本来自统一登记。模块尚未迁入时，这里展示迁移来源，不表示当前已经运行该方法。</p>
<input v-model="query" aria-label="搜索来源" placeholder="搜索项目、方法或作者"/>
<p role="status">{{message}}</p>
<article v-for="s in matches" :key="s.id" class="reference-row">
<h3>{{s.title}}</h3>
<p class="selectable">{{s.authors}} · {{s.title}}</p>
<small>{{s.kind}} · {{s.version}}</small>
<small>{{s.license}}</small>
<div class="reference-links">
<a v-for="(url,label) in s.urls" :key="label" :href="url" target="_blank" rel="noopener noreferrer">{{label}}</a>
<button @click="copy(s.authors+' · '+s.title)">复制引用</button>
</div>
</article>
<p v-if="!matches.length" class="empty-small">没有匹配的来源记录</p>
</template>
