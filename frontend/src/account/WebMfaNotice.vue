<script setup lang="ts">
import ModuleFrame from '../components/ModuleFrame.vue';
import {officialWebsite} from '../platform/feedback-profile.ts';
</script>
<template>
  <ModuleFrame label="MFA自动标注">
    <div class="web-mfa-notice">
      <p>由于服务器性能限制，网页版暂不支持MFA在线对齐。请下载桌面应用，在本机完成对齐。</p>
      <a :href="officialWebsite" target="_blank" rel="noopener noreferrer">下载桌面版</a>
      <p class="muted">已有的 TextGrid 文件可以在标注工具中继续编辑。</p>
    </div>
  </ModuleFrame>
</template>
<style scoped>
.web-mfa-notice{max-width:680px;padding:20px;line-height:1.8}
.web-mfa-notice a{color:var(--accent);text-underline-offset:3px}
.muted{color:var(--muted)}
</style>
