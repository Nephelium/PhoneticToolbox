<script setup lang="ts">
import {locale,pendingLocale,localeError,supportedLocales,isAppLocale,setLocale,tr} from '../i18n/index.ts';
defineProps<{expanded?:boolean}>();
function change(event:Event){const code=(event.target as HTMLSelectElement).value;if(isAppLocale(code))void setLocale(code);}
</script>
<template>
<div class="language-switch" :class="{expanded}" @pointerdown.stop @dblclick.stop>
 <label>
  <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/></svg>
  <span v-if="expanded">{{tr('common.language.label')}}</span>
  <select :value="locale" :aria-label="tr('common.language.choose')" :disabled="!!pendingLocale" @change="change"><option v-for="language in supportedLocales" :key="language.code" :value="language.code" :lang="language.code">{{language.nativeName}}</option></select>
 </label>
 <span v-if="pendingLocale" class="language-status" role="status">{{tr('common.language.loading')}}</span>
 <p v-if="expanded&&localeError" class="error" role="alert">{{tr(localeError)}}</p>
</div>
</template>
<style scoped>
.language-switch{flex:none;min-width:0}.language-switch label{display:flex;align-items:center;gap:6px;white-space:nowrap}.language-switch svg{width:17px;height:17px;fill:none;stroke:currentColor;stroke-width:1.5;flex:none}.language-switch select{max-width:150px;min-width:0;font-size:.9rem;padding:4px 6px}.language-status{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}.expanded{display:grid;gap:8px}.expanded label{justify-content:space-between}.expanded select{min-width:150px}.expanded .error{margin:0;line-height:1.5}
@media(max-width:600px){.language-switch select{max-width:105px;padding:4px 2px;font-size:.8rem}.language-switch svg{display:none}.expanded select{max-width:none}.expanded svg{display:block}}
</style>
