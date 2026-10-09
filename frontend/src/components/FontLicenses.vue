<script setup lang="ts">
import { computed, nextTick, ref, useId } from 'vue';
import doulosLicense from '../assets/Doulos-OFL.txt?raw';
import monoLicense from '../assets/JetBrainsMono-OFL.txt?raw';
import ipaLicense from '../assets/ipa-plus/OFL-PTBIPAPlus.txt?raw';
import notoLicense from '../assets/ipa-plus/OFL-Noto.txt?raw';

const fonts = [
  { id: 'doulos', name: 'Doulos SIL', version: '7.000', usage: '国际音标字体', license: doulosLicense },
  { id: 'mono', name: 'JetBrains Mono', version: '2.304', usage: '代码与等宽字体', license: monoLicense },
  { id: 'ipa', name: 'PTB IPA Plus', version: '1.000', usage: '国际音标表 Plus 字体，基于 Doulos SIL，含 Noto Sans Math 的两个字形', license: ipaLicense + '\n\n' + notoLicense },
];
const active = ref('doulos'), id = useId();
const selected = computed(() => fonts.find(font => font.id === active.value)!);
const panel = ref<HTMLElement>();
function select(key: string) {
  active.value = key;
  void nextTick(() => panel.value?.scrollTo({ top: 0 }));
}
function navigate(event: KeyboardEvent) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const index = fonts.findIndex(font => font.id === active.value);
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? fonts.length - 1 :
    (index + (event.key === 'ArrowRight' ? 1 : -1) + fonts.length) % fonts.length;
  select(fonts[next].id);
  void nextTick(() => document.getElementById(id + '-' + active.value)?.focus());
}
</script>

<template>
  <div class="font-licenses">
    <div class="font-license-tabs" role="tablist" aria-label="内置字体" @keydown="navigate">
      <button v-for="font in fonts" :id="id + '-' + font.id" :key="font.id" role="tab"
        :aria-selected="active === font.id" :aria-controls="id + '-license'"
        :tabindex="active === font.id ? 0 : -1" @click="select(font.id)">{{ font.name }}</button>
    </div>
    <section :id="id + '-license'" ref="panel" class="font-license-panel" role="tabpanel"
      :aria-labelledby="id + '-' + active" tabindex="0">
      <h3>{{ selected.name }} {{ selected.version }}</h3>
      <p class="muted">{{ selected.usage }} · SIL Open Font License 1.1</p>
      <pre class="license-text">{{ selected.license }}</pre>
    </section>
  </div>
</template>

<style scoped>
.font-licenses{display:flex;flex-direction:column;gap:14px;min-height:0}
.font-license-tabs{display:flex;flex-wrap:wrap;gap:8px}
.font-license-tabs button{min-width:0;white-space:normal}
.font-license-tabs button[aria-selected=true]{background:var(--selected);border-color:var(--accent);color:var(--accent)}
.font-license-panel{max-height:54dvh;overflow:auto;overscroll-behavior:contain;scrollbar-width:thin}
.font-license-panel h3{margin-bottom:6px}
.font-license-panel pre{margin-top:14px}
</style>
