<script setup lang="ts">
import version from './version.json';
</script>

<template>
  <main>
    <p class="eyebrow">PHONETIC TOOLBOX · {{ version.stage }}</p>
    <h1>语音研究工作台</h1>
    <p class="status">工程入口已建立</p>
    <p>研究模块尚未迁入。正式导航、播放器和浅深主题将在 P04 接入。</p>
    <footer>版本 {{ version.frontend_version }} · 数据协议 {{ version.api_version }}</footer>
  </main>
</template>

<style>
:root { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color: #263448; background: #f3f5f8; }
body { margin: 0; }
main { max-width: 760px; margin: 12vh auto; padding: 32px; background: #fff; border: 1px solid #dce3eb; border-radius: 12px; }
h1 { margin: 12px 0 24px; font-size: 28px; }
p { line-height: 1.7; }
.eyebrow, footer { color: #5c6f85; font-size: 13px; }
.status { color: #245c93; font-weight: 600; }
footer { margin-top: 48px; }
@media (max-width: 700px) { main { margin: 24px 12px; padding: 24px; } }
</style>
