<script setup lang="ts">
import LanguageSwitch from './LanguageSwitch.vue';
import {tr} from '../i18n/index.ts';
</script>
<template><section class="settings-card language-settings"><h3>{{tr('common.language.label')}}</h3><LanguageSwitch expanded/><p class="settings-note">{{tr('common.language.description')}}</p></section></template>
<style scoped>.language-settings{max-width:680px;display:grid;gap:16px}.language-settings h3,.language-settings p{margin:0}</style>
