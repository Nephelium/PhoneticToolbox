<script setup lang="ts">
import { ref,computed } from 'vue';import parameters from '../generated/parameters.json';import ModalDialog from './ModalDialog.vue';
const props=defineProps<{selected:string[]}>();const emit=defineEmits<{close:[];apply:[keys:string[]]}>();
const query=ref(''),draft=ref([...props.selected]);const filtered=computed(()=>parameters.filter(p=>(p.label+' '+p.key).toLowerCase().includes(query.value.toLowerCase())));
</script>
<template>
<ModalDialog title="输出参数" wide @close="emit('close')">
<p class="muted">调整下一次分析的参数草稿。已有结果不会随草稿变化。</p>
<div class="parameter-toolbar">
<input v-model="query" aria-label="搜索参数" placeholder="搜索参数名或参数键"/>
<button @click="draft=parameters.map(p=>p.key)">全选</button>
<button @click="draft=[]">全不选</button>
<span class="mono">{{draft.length}} / 80 项</span>
</div>
<div class="parameter-grid">
<label v-for="p in filtered" :key="p.key">
<input v-model="draft" type="checkbox" :value="p.key"/>
<span>{{p.label}}<small>{{p.key}}</small>
</span>
</label>
</div>
<p v-if="!filtered.length" class="empty-small">没有匹配的参数</p>
<template #footer>
<button @click="emit('close')">取消</button>
<button class="primary" @click="emit('apply',draft)">应用到草稿</button>
</template>
</ModalDialog>
</template>
