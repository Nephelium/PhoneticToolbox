<script setup lang="ts">
import type {SymbolEntry} from './types.ts';
defineProps<{entry:SymbolEntry;selected?:boolean;named?:boolean}>();
const emit=defineEmits<{insert:[entry:SymbolEntry];inspect:[entry:SymbolEntry];hover:[entry:SymbolEntry|null]}>();
</script>
<template>
  <button type="button" class="m17-symbol" :class="{'m17-example':entry.isExample,'m17-selected':selected,'m17-named':named}" :data-symbol-id="entry.id" :aria-label="`${entry.nameZh}：${entry.display}，点击输入；Alt＋Enter 查看介绍`" @pointerdown.prevent @click="emit('insert',entry)" @mouseenter="emit('hover',entry)" @mouseleave="emit('hover',null)" @focus="emit('hover',entry)" @blur="emit('hover',null)" @keydown.alt.enter.prevent.stop="emit('inspect',entry)" @contextmenu.prevent="emit('inspect',entry)">
    <span class="m17-ipa">{{entry.display}}</span><span v-if="named" class="m17-symbol-name">{{entry.nameZh}}</span>
    <span v-if="entry.isExample" class="visually-hidden">完整示例</span>
  </button>
</template>
<style scoped>
.m17-symbol{min-width:23px;min-height:24px;padding:0 2px;border:1px solid transparent;border-radius:4px;background:transparent;gap:4px;line-height:1;white-space:nowrap;vertical-align:middle;color:var(--text)}
.m17-symbol>.m17-ipa{font-size:22px;line-height:1;padding:0}
.m17-symbol.m17-example>.m17-ipa{font-size:20px;color:var(--muted)}
.m17-symbol.m17-selected{border-color:var(--accent);background:var(--selected)}
.m17-named{justify-content:flex-start;width:100%;text-align:left;min-height:26px}
.m17-named>.m17-ipa{min-width:45px;text-align:center}
.m17-symbol-name{font-size:12px;white-space:normal;line-height:1.25}
</style>
