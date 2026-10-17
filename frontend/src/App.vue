<script setup lang="ts">
import { defineAsyncComponent } from 'vue';
import './design/tokens.css';
const server = location.protocol !== 'ptbapp:' && ['/server/','/app/'].some(prefix=>location.pathname.startsWith(prefix));
const AppShell = defineAsyncComponent(() => import('./app/AppShell.vue'));
const ServerPage = defineAsyncComponent(() => import('./account/ServerPage.vue'));
</script>
<template><ServerPage v-if="server"/><AppShell v-else/></template>
