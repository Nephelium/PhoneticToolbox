<script setup lang="ts">
import {officialWebsite} from '../platform/feedback-profile.ts';
defineProps<{sealed:boolean;compact?:boolean;title:string;reason:string}>();
defineEmits<{help:[]}>();
</script>
<template>
 <section class="module-seal" :class="{sealed,compact}" :aria-label="title">
  <div class="module-seal-content" :inert="sealed" :aria-hidden="sealed ? true : undefined"><slot/></div>
  <div v-if="sealed" class="module-seal-cover">
   <div class="module-seal-note" role="status">
    <span class="seal-mark" aria-hidden="true"><svg viewBox="0 0 48 48" fill="none"><rect x="6" y="8" width="36" height="26" rx="5"/><path d="M16 41h16M24 34v7M13 21h5l3-7 6 14 3-7h5"/></svg></span>
    <small>DESKTOP</small><h2>{{title}}</h2>
    <p>{{reason}}</p>
    <div class="seal-actions"><a class="button primary" :href="officialWebsite" target="_blank" rel="noopener noreferrer">下载桌面版</a><button @click="$emit('help')">使用说明</button></div>
   </div>
  </div>
 </section>
</template>
<style scoped>
.module-seal{position:relative;width:100%;height:100%;min-height:0;min-width:0;isolation:isolate}
.module-seal-content{height:100%;min-height:0}.sealed{overflow:hidden}.sealed .module-seal-content{pointer-events:none;user-select:none}
.module-seal-cover{position:absolute;inset:0;z-index:5;display:grid;place-items:center;padding:24px;overflow:auto;background:color-mix(in srgb,var(--app) 62%,transparent);backdrop-filter:blur(1.5px);border:1px solid color-mix(in srgb,var(--accent) 35%,var(--border));border-radius:12px}
.module-seal-note{width:min(440px,100%);box-sizing:border-box;padding:28px;border-radius:20px;border:1px solid var(--border);background:var(--panel);box-shadow:0 20px 70px #0003;color:var(--text);text-align:left}
.seal-mark{display:grid;place-items:center;width:52px;height:52px;border-radius:14px;background:var(--selected);color:var(--accent)}.seal-mark svg{width:32px;height:32px;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}.module-seal-note small{display:block;margin-top:20px;color:var(--muted);font-size:.7rem;letter-spacing:.12em}.module-seal-note h2{margin:7px 0 14px;font-size:1.5rem}.module-seal-note p{margin:0;line-height:1.85;color:var(--muted)}.seal-actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:24px;padding-top:20px;border-top:1px solid var(--border)}.seal-actions a{text-decoration:none;display:inline-flex;align-items:center;justify-content:center;padding:8px 14px;border:1px solid var(--accent);border-radius:var(--radius);background:var(--accent);color:var(--on-accent);font-size:inherit;line-height:1.4;box-sizing:border-box}.seal-actions>*{height:40px;min-height:40px;box-sizing:border-box}
@media(max-width:600px){.module-seal-cover{padding:12px}.module-seal-note{padding:20px}}
</style>

<style scoped>
.module-seal:not(.sealed),.module-seal:not(.sealed)>.module-seal-content{display:contents}.module-seal.compact{height:auto;min-height:48px}.compact .module-seal-cover{padding:6px;border-radius:6px}.compact .module-seal-note{width:auto;padding:4px 12px;border:0;box-shadow:none;background:transparent}.compact .seal-mark,.compact small,.compact h2,.compact .seal-actions,.compact p+p{display:none}.compact p{margin:0;font-size:var(--control-size);line-height:1.6}
</style>
