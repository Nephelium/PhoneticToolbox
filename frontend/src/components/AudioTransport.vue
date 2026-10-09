<script setup lang="ts">
import {vTimePrecision} from '../design/time-precision.ts';
import {useAudioSelection} from '../state/audio-selection.ts';
import { ref,computed,watch,onMounted,onUnmounted } from 'vue';import type { Workspace } from '../state/workspace.ts';import {playback,play,pause,stop,volume,seek,isCurrentAudio} from '../state/audio.ts';import { selection } from '../platform/wav.ts';import AppIcon from './AppIcon.vue';
const props=defineProps<{state:Workspace;selectionState?:Workspace;active:boolean;compact?:boolean;externalPlayback?:boolean}>();
const emit=defineEmits<{'selection-play':[];stop:[]}>();
// EGG previews play normalized audio while selection edits own the source
// workspace. Other modules use their existing single state unchanged.
const rangeState=computed(()=>props.selectionState??props.state);
const selectionGroup=useAudioSelection();const unregister=selectionGroup?.register(()=>rangeState.value);onUnmounted(()=>unregister?.());
function select(){selectionGroup?.select(rangeState.value);}
const scrub=ref<number|null>(null);
const owns=computed(()=>!props.externalPlayback&&isCurrentAudio(props.state.asset,props.state.channel));
const playing=computed(()=>owns.value&&playback.playing),position=computed(()=>owns.value?playback.position:props.state.start);
watch(()=>[props.state.asset,props.state.start,props.state.end],()=>scrub.value=null);
function seekTo(event:Event){if(props.state.asset)seek(props.state.asset,Number((event.target as HTMLInputElement).value),props.state.start,props.state.end,props.state.channel);scrub.value=null;}
function range(){select();const state=rangeState.value;[state.start,state.end]=selection(state.start,state.end,state.asset?.duration||0);stop();}
function toggle(){const wasPlaying=playing.value,start=position.value>=props.state.start&&position.value<props.state.end?position.value:props.state.start;select();emit('selection-play');if(wasPlaying)pause();else if(props.state.asset&&props.state.end>props.state.start)void play(props.state.asset,start,props.state.end,props.state.channel);}
function stopPlayback(){emit('stop');stop();}
// Explicit full-length choice for result-list actions. Keep the shared selection
// owner in sync so Space and the transport operate on the same audio afterward.
async function selectAll(autoplay=false){if(!props.state.asset)return;select();props.state.start=0;props.state.end=props.state.asset.duration;stop();scrub.value=null;if(autoplay&&props.active)await play(props.state.asset,0,props.state.asset.duration,props.state.channel);}
defineExpose({selectAll});
function key(e:KeyboardEvent){if(e.defaultPrevented||!props.active||e.code!=='Space'||e.repeat||document.querySelector('dialog[open]'))return;if((e.target as HTMLElement).closest('input,textarea,select,button,a,[contenteditable]'))return;if(selectionGroup&&!selectionGroup.keyboard(rangeState.value))return;if(!props.state.asset||props.state.end<=props.state.start)return;e.preventDefault();if(playing.value)stopPlayback();else toggle();}
onMounted(()=>window.addEventListener('keydown',key));onUnmounted(()=>window.removeEventListener('keydown',key));
</script>
<template>
<div class="transport-controls" :class="{'transport-compact':compact}">
<div v-if="!compact" class="selection-controls">
<strong>时间选区</strong>
<label>起点 <input v-time-precision="'s'" v-model.number="rangeState.start" type="number" min="0" :max="rangeState.asset?.duration||0" step="0.001" :disabled="!rangeState.asset" @change="range"/> s</label>
<span>—</span>
<label>终点 <input v-time-precision="'s'" v-model.number="rangeState.end" type="number" min="0" :max="rangeState.asset?.duration||0" step="0.001" :disabled="!rangeState.asset" @change="range"/> s</label>
<button :disabled="!rangeState.asset" @click="select();rangeState.start=0;rangeState.end=rangeState.asset!.duration;stop()">全部</button>
</div>
<div class="audio-transport">
<button :class="{primary:!compact}" :disabled="!state.asset||state.end<=state.start" @click="toggle">
<AppIcon :name="playing?'pause':'play'"/>{{playing?'暂停':'播放选区'}}</button>
<button :disabled="!state.asset" @click="stopPlayback">
<AppIcon name="stop"/>停止</button>
<span class="mono">{{position.toFixed(3)}} / {{(state.asset?.duration||0).toFixed(3)}} s</span>
<label class="playback-seek"><span class="visually-hidden">播放进度（当前选区）</span><input type="range" :min="state.start" :max="state.end" step="0.001" :value="scrub??Math.max(state.start,Math.min(position,state.end))" :disabled="!state.asset||state.end<=state.start" :aria-valuetext="(scrub??Math.max(state.start,position)).toFixed(3)+' 秒'" @input="scrub=Number(($event.target as HTMLInputElement).value)" @change="seekTo" @pointercancel="scrub=null" @blur="scrub=null"/></label>
<label class="volume">
<AppIcon name="speaker"/>
<input type="range" aria-label="播放音量" min="0" max="1" step="0.05" :value="playback.volume" @input="volume(Number(($event.target as HTMLInputElement).value))"/>
<small>{{Math.round(playback.volume*100)}}%</small>
</label>
</div>
</div>
<p v-if="playback.error" role="alert" class="error-text">{{playback.error}}</p>
</template>
<style scoped>
.transport-compact{padding:0;gap:6px;flex:1;min-width:220px}.transport-compact .audio-transport{gap:6px;flex-wrap:wrap;margin:0}.transport-compact .playback-seek,.transport-compact .volume{display:none}
</style>
