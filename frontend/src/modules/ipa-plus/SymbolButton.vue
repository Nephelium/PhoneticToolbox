<script setup lang="ts">
import type {SymbolEntry} from './types.ts';
import type {HoverTarget} from './hover-position.ts';
const props=defineProps<{entry:SymbolEntry;selected?:boolean;named?:boolean;playOnClick?:boolean}>();
const emit=defineEmits<{insert:[entry:SymbolEntry];inspect:[entry:SymbolEntry];hover:[value:HoverTarget|null]}>();
function hover(event:MouseEvent|FocusEvent){const target=event.currentTarget as HTMLElement,rect=target.getBoundingClientRect();emit('hover',{entry:props.entry,target,pointer:event instanceof MouseEvent?{x:event.clientX,y:event.clientY}:{x:(rect.left+rect.right)/2,y:(rect.top+rect.bottom)/2}});}
</script>
<template>
  <button type="button" class="m17-symbol" :class="{'m17-example':entry.isExample,'m17-selected':selected,'m17-named':named}" :data-symbol-id="entry.id" :aria-label="`${entry.nameZh}：${entry.display}，点击${playOnClick?'播放':'输入'}；Alt＋Enter 查看介绍`" @pointerdown.prevent @click="emit('insert',entry)" @mouseenter="hover" @mousemove="hover" @mouseleave="emit('hover',null)" @focus="hover" @blur="emit('hover',null)" @keydown.alt.enter.prevent.stop="emit('inspect',entry)" @contextmenu.prevent="emit('inspect',entry)">
    <span class="m17-ipa">{{entry.display}}</span><span v-if="named" class="m17-symbol-name">{{entry.nameZh}}</span>
    <span v-if="entry.isExample" class="visually-hidden">完整示例</span>
  </button>
</template>
<style scoped>
.m17-symbol{min-width:23px;min-height:24px;padding:0 2px;border:1px solid transparent;border-radius:4px;background:transparent;gap:4px;line-height:1;white-space:nowrap;vertical-align:middle;scroll-margin:48px 8px 8px;color:var(--text)}
.m17-symbol>.m17-ipa{font-size:calc(var(--m17-font-size,26px) - 6px);line-height:1;padding:0}
.m17-symbol.m17-example>.m17-ipa{font-size:calc(var(--m17-font-size,26px) - 8px);color:var(--muted)}
.m17-symbol.m17-selected{border-color:var(--accent);background:var(--selected)}
.m17-named{justify-content:flex-start;width:100%;text-align:left;min-height:26px}
.m17-named>.m17-ipa{min-width:45px;text-align:center}
.m17-symbol-name{font-size:0.857143rem;white-space:normal;line-height:1.25}
</style>
