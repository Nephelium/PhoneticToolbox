<script setup lang="ts">
import { computed, nextTick, ref, useId } from 'vue';
import sources from '../generated/sources.json';
const props = defineProps<{ moduleId?: string }>();
const query = ref(''), message = ref('');
const active = ref('academic');
const id = useId();
const groups = [
  { id: 'academic', title: '语音学与语言学学术来源', description: '论文、研究方法、语言规则与研究数据。' },
  { id: 'software', title: '软件与代码来源', description: '软件工具、代码实现、依赖库与工程素材。' },
];
const matches = computed(() => sources.filter(s =>
  (!props.moduleId || s.modules.includes(props.moduleId)) &&
  (s.title + ' ' + s.authors).toLowerCase().includes(query.value.trim().toLowerCase())));
const visible = computed(() => matches.value.filter(s => s.acknowledgement_group === active.value));
function groupKey(event: KeyboardEvent) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  active.value = event.key === 'Home' ? 'academic' : event.key === 'End' ? 'software' : active.value === 'academic' ? 'software' : 'academic';
  void nextTick(() => document.getElementById(id + '-' + active.value)?.focus());
}
async function copy(text: string) {
  try { await navigator.clipboard.writeText(text); message.value = '引用已复制'; }
  catch { message.value = '无法访问剪贴板，请选中引用文字复制。'; }
}
</script>
<template>
  <p class="muted">学术来源与软件来源分别列示。尚未迁入的模块展示迁移来源；待核实的作者、版本和许可会保留标记。</p>
  <div class="reference-groups" role="tablist" aria-label="致谢分组" @keydown="groupKey">
    <button v-for="group in groups" :id="id + '-' + group.id" :key="group.id" role="tab"
      :aria-selected="active === group.id" :aria-controls="id + '-panel'" :tabindex="active === group.id ? 0 : -1"
      @click="active = group.id">
      {{ group.title }} <span class="mono">{{ matches.filter(s => s.acknowledgement_group === group.id).length }}</span>
    </button>
  </div>
  <input v-model="query" aria-label="搜索来源" placeholder="搜索项目、方法或作者"/>
  <p role="status">{{ message }}</p>
  <section :id="id + '-panel'" role="tabpanel" :aria-labelledby="id + '-' + active" tabindex="0">
    <p class="hint">{{ groups.find(group => group.id === active)?.description }}</p>
    <article v-for="s in visible" :key="s.id" class="reference-row">
      <h3>{{ s.title }}</h3>
      <p class="selectable">{{ s.authors }} · {{ s.title }}</p>
      <small>{{ s.kind }} · {{ s.version }}</small>
      <small>{{ s.license }}</small>
      <div class="reference-links">
        <a v-for="(url,label) in s.urls" :key="label" :href="url" target="_blank" rel="noopener noreferrer">{{ label }}</a>
        <button @click="copy(s.authors + ' · ' + s.title)">复制引用</button>
      </div>
    </article>
    <p v-if="!visible.length" class="empty-small">本组没有匹配的来源记录，可切换另一组或调整搜索词。</p>
  </section>
</template>
