<script setup lang="ts">
import {tr} from '../i18n/index.ts';
import {automaticAudioGain,auditionSettingsError,setAutomaticAudioGain} from '../state/audition.ts';
</script>
<template><section class="settings-card"><h2>{{tr('components.audio_playback')}}</h2><label><input type="checkbox" :checked="automaticAudioGain" :aria-label="tr('components.automatic_playback_gain')" @change="setAutomaticAudioGain(($event.target as HTMLInputElement).checked)"/>{{tr('components.automatic_playback_gain')}}</label><p>{{tr('components.automatically_raises_quiet_preview_audio_you_can_still_adjust')}}</p><p>{{tr('components.formal_stimuli_in_perception_experiments_use_the_playback_volume')}}</p><p v-if="auditionSettingsError" role="alert">{{auditionSettingsError}} <button @click="setAutomaticAudioGain(automaticAudioGain)">{{tr('components.retry_saving')}}</button></p></section></template>
