<script setup lang="ts">
import {tr} from '../i18n/index.ts';
import ModuleFrame from '../components/ModuleFrame.vue';
import {officialWebsite} from '../platform/feedback-profile.ts';
</script>
<template>
  <ModuleFrame :label="tr('account.mfa_forced_alignment')">
    <div class="web-mfa-notice">
      <p>{{tr('account.online_mfa_alignment_is_not_supported_because_of_server')}}</p>
      <a :href="officialWebsite" target="_blank" rel="noopener noreferrer">{{tr('account.download_desktop_version')}}</a>
      <p class="muted">{{tr('account.existing_textgrid_files_can_still_be_edited_in_the')}}</p>
    </div>
  </ModuleFrame>
</template>
<style scoped>
.web-mfa-notice{max-width:680px;padding:20px;line-height:1.8}
.web-mfa-notice a{color:var(--accent);text-underline-offset:3px}
.muted{color:var(--muted)}
</style>
