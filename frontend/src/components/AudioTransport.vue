<script setup lang="ts">
import { onMounted,onUnmounted } from 'vue';import type { Workspace } from '../state/workspace.ts';import {playback,play,pause,stop,volume} from '../state/audio.ts';import { selection } from '../platform/wav.ts';import AppIcon from './AppIcon.vue';
const props=defineProps<{state:Workspace;active:boolean}>();
function range(){[props.state.start,props.state.end]=selection(props.state.start,props.state.end,props.state.asset?.duration||0);stop();}
function toggle(){if(playback.playing)pause();else if(props.state.asset)void play(props.state.asset,playback.position>=props.state.start&&playback.position<props.state.end?playback.position:props.state.start,props.state.end,props.state.channel);}
function key(e:KeyboardEvent){if(!props.active||e.code!=='Space'||e.repeat||document.querySelector('dialog[open]'))return;if((e.target as HTMLElement).closest('input,textarea,select,button,a,[contenteditable]'))return;e.preventDefault();toggle();}
onMounted(()=>window.addEventListener('keydown',key));onUnmounted(()=>window.removeEventListener('keydown',key));
</script>
<template>
<div class="selection-controls">
<strong>时间选区</strong>
<label>起点 <input v-model.number="state.start" type="number" min="0" :max="state.asset?.duration||0" step="0.001" :disabled="!state.asset" @change="range"/> s</label>
<span>—</span>
<label>终点 <input v-model.number="state.end" type="number" min="0" :max="state.asset?.duration||0" step="0.001" :disabled="!state.asset" @change="range"/> s</label>
<button :disabled="!state.asset" @click="state.start=0;state.end=state.asset!.duration;stop()">全部</button>
</div>
<div class="audio-transport">
<button class="primary" :disabled="!state.asset||state.end<=state.start" @click="toggle">
<AppIcon :name="playback.playing?'pause':'play'"/>{{playback.playing?'暂停':'播放选区'}}</button>
<button :disabled="!state.asset" @click="stop">
<AppIcon name="stop"/>停止</button>
<span class="mono">{{playback.position.toFixed(3)}} / {{(state.asset?.duration||0).toFixed(3)}} s</span>
<label class="volume">
<AppIcon name="speaker"/>
<input type="range" aria-label="播放音量" min="0" max="1" step="0.05" :value="playback.volume" @input="volume(Number(($event.target as HTMLInputElement).value))"/>
<small>{{Math.round(playback.volume*100)}}%</small>
</label>
</div>
<p v-if="playback.error" role="alert" class="error-text">{{playback.error}}</p>
</template>
