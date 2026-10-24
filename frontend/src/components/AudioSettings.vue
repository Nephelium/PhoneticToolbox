<script setup lang="ts">
import {automaticAudioGain,auditionSettingsError,setAutomaticAudioGain} from '../state/audition.ts';
</script>
<template><section class="settings-card"><h2>音频播放</h2><label><input type="checkbox" :checked="automaticAudioGain" aria-label="音频音量自动增益" @change="setAutomaticAudioGain(($event.target as HTMLInputElement).checked)"/>音频音量自动增益</label><p>自动提高偏小的试听音量，音量滑条仍可调整。仅影响播放，原始文件与导出文件保持原样。</p><p>感知实验的正式刺激使用实验内单独设置的播放音量。</p><p v-if="auditionSettingsError" role="alert">{{auditionSettingsError}} <button @click="setAutomaticAudioGain(automaticAudioGain)">重试保存</button></p></section></template>
