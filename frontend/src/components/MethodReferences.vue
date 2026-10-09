<script setup lang="ts">
import { computed, nextTick, ref, useId } from 'vue';
import sources from '../generated/sources.json';
import { modules } from '../app/registry.ts';
import { copyText } from '../platform/clipboard.ts';
const props = defineProps<{ moduleId?: string }>();
const query = ref(''), message = ref('');
const selectedModule = ref('');
const active = ref('academic');
const id = useId();
const groups = [
  { id: 'academic', title: '语音学与语言学学术来源', description: '论文、研究方法、语言规则与研究数据。' },
  { id: 'software', title: '软件与代码来源', description: '软件工具、代码实现、依赖库与工程素材。' },
];
const matches = computed(() => sources.filter(s =>
  (!(props.moduleId || selectedModule.value) || s.modules.includes(props.moduleId || selectedModule.value)) &&
  [s.title, s.authors, s.citation || '', ...modules.filter(m => s.modules.includes(m.id)).map(m => m.title)].join(' ').toLowerCase().includes(query.value.trim().toLowerCase())));
const visible = computed(() => matches.value.filter(s => s.acknowledgement_group === active.value));
const moduleNames = (ids: string[]) => modules.filter(module => ids.includes(module.id)).map(module => module.title).join('、') || '公共组件';
function groupKey(event: KeyboardEvent) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  active.value = event.key === 'Home' ? 'academic' : event.key === 'End' ? 'software' : active.value === 'academic' ? 'software' : 'academic';
  void nextTick(() => document.getElementById(id + '-' + active.value)?.focus());
}
async function copy(text: string) {
  try { await copyText(text); message.value = '引用已复制'; }
  catch { message.value = '无法访问剪贴板，请选中引用文字复制。'; }
}
const citation = (source: { authors: string; title: string; citation?: string }) => source.citation || [source.authors, source.title].filter(Boolean).join(' · ');
</script>
<template>
  <p class="muted">学术文献保留引用与来源。实际使用的第三方代码、库与资源另列许可说明，已记录的版本一并列出。</p>
  <div v-if="!moduleId" class="reference-scope">
    <label>查看范围 <select v-model="selectedModule" aria-label="按模块查看致谢">
      <option value="">全部模块与公共组件</option>
      <option v-for="module in modules" :key="module.id" :value="module.id">{{ module.title }}</option>
    </select></label>
    <p class="hint">汇总各模块的学术、代码、模型、素材与字体来源。同一来源集中列出，保留所用模块及原有署名和授权说明。</p>
  </div>
  <div class="reference-groups" role="tablist" aria-label="致谢分组" @keydown="groupKey">
    <button v-for="group in groups" :id="id + '-' + group.id" :key="group.id" role="tab"
      :aria-selected="active === group.id" :aria-controls="id + '-panel'" :tabindex="active === group.id ? 0 : -1"
      @click="active = group.id">
      {{ group.title }} <span class="mono">{{ matches.filter(s => s.acknowledgement_group === group.id).length }}</span>
    </button>
  </div>
  <input v-model="query" aria-label="搜索来源" placeholder="搜索项目、方法、作者或模块"/>
  <p role="status">{{ message }}</p>
  <section :id="id + '-panel'" role="tabpanel" :aria-labelledby="id + '-' + active" tabindex="0">
    <p class="hint">{{ groups.find(group => group.id === active)?.description }}</p>
    <article v-for="s in visible" :key="s.id" class="reference-row">
      <h3>{{ s.title }}</h3>
      <small class="reference-modules">用于：{{ moduleNames(s.modules) }}</small>
      <p class="selectable">{{ citation(s) }}</p>
      <small>{{ [s.kind, s.version].filter(Boolean).join(' · ') }}</small>
      <small v-if="s.license">{{ s.license }}</small>
      <p v-if="s.attribution" class="selectable">{{ s.attribution }}<span v-if="s.permission_date">（{{ s.permission_date }} 作者邮件许可）</span></p>
      <details v-if="s.adaptation_note" class="reference-adaptation"><summary>改写说明</summary>
        <p>{{ s.adaptation_note }}</p><p v-if="s.adaptation_caution">{{ s.adaptation_caution }}</p>
      </details>
      <div class="reference-links">
        <a v-for="(url,label) in s.urls" :key="label" :href="url" target="_blank" rel="noopener noreferrer">{{ label }}</a>
        <button @click="copy(citation(s))">复制引用</button>
      </div>
    </article>
    <p v-if="!visible.length" class="empty-small">本组没有匹配的来源记录，可切换另一组或调整搜索词。</p>
  </section>
</template>
<style scoped>
.reference-scope label{display:flex;align-items:center;flex-wrap:wrap;gap:8px}
.reference-scope select{max-width:100%}.reference-modules{color:var(--muted)}
.reference-adaptation{margin:6px 0}.reference-adaptation p{overflow-wrap:anywhere}
</style>
