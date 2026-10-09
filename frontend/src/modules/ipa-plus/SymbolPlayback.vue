<script setup lang="ts">
import {computed,nextTick,onBeforeUnmount,onMounted,ref} from 'vue';
import type {SymbolEntry} from './types.ts';
import {contentFor,mediaUrl} from './content.ts';
import {symbolAnimation} from './playback.ts';
const props=defineProps<{entry:SymbolEntry;request:number}>();
const content=computed(()=>contentFor(props.entry)),media=computed(()=>content.value.media);
const audio=ref<HTMLAudioElement>(),video=ref<HTMLVideoElement>(),animation=ref<HTMLElement>(),message=ref('');
const abort=new AbortController();let stop:(()=>void)|undefined;
function failure(){message.value='演示素材无法播放，请检查素材文件。';}
onMounted(async()=>{
 try{
  await nextTick();
  const spec=media.value?.animation;
  if(spec){const renderer=symbolAnimation(spec.renderer);if(!renderer)message.value='此项实时动画尚未接入。';else{
   const cleanup=await renderer({entry:props.entry,version:spec.version,config:spec.config,container:animation.value!,signal:abort.signal});
   if(abort.signal.aborted){if(typeof cleanup==='function')cleanup();return;}if(typeof cleanup==='function')stop=cleanup;
  }}
  // A video supplies its own sound unless a separate audio file is configured.
  if(abort.signal.aborted)return;
  const starts:Promise<void>[]=[];if(audio.value)starts.push(audio.value.play());if(video.value)starts.push(video.value.play());await Promise.all(starts);
  if(!media.value?.audio&&!media.value?.video&&!spec)message.value='此音标尚未添加演示内容。';
 }catch{if(!abort.signal.aborted)failure();}
});
onBeforeUnmount(()=>{abort.abort();audio.value?.pause();video.value?.pause();stop?.();});
</script>
<template>
 <div class="m17-playback" :data-symbol-playback="entry.id">
  <video v-if="media?.video" ref="video" :src="mediaUrl(media.video)" :muted="!!media.audio" controls playsinline preload="none" @error="failure"/>
  <audio v-if="media?.audio" ref="audio" :src="mediaUrl(media.audio)" controls preload="none" @error="failure"/>
  <div v-if="media?.animation" ref="animation" class="m17-animation" aria-label="实时发音演示"></div>
  <p v-if="message" role="status">{{message}}</p>
 </div>
</template>
<style scoped>
.m17-playback{display:grid;gap:10px;margin:10px 0}.m17-playback video{width:100%;max-height:280px;background:var(--app);border-radius:6px}.m17-playback audio{width:100%}.m17-animation{min-height:120px}.m17-playback p{font-size:.928571rem;color:var(--muted)}
</style>
