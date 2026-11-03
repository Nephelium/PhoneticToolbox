<script setup lang="ts">
import {tr,message,messageRef} from '../i18n/index.ts';
import {computed,inject,onBeforeUnmount,onDeactivated,ref,watch} from 'vue';
import type {AssetResolver,ManualAsset} from './types.ts';
import {assetUrl,safeMediaSource,unnumberedCaption} from './content.ts';
import {manualPlayback,manualPlaybackAllowed} from './playback.ts';
import {mediaWidth} from './styles.ts';
import {useMediaAudition} from '../state/media-audition.ts';
const props=withDefaults(defineProps<{kind:'image'|'audio'|'video';assetId:string;assets:ManualAsset[];assetBaseUrl?:string;assetResolver?:AssetResolver;caption?:string;displayNumber?:string;alt?:string;width?:unknown;align?:unknown;playbackAllowed?:boolean}>(),{assetBaseUrl:'./manual/',playbackAllowed:true});
const readerPlaybackAllowed=inject(manualPlaybackAllowed,undefined);
const playbackAllowed=computed(()=>props.playbackAllowed&&(readerPlaybackAllowed?.value??true));
const media=ref<HTMLMediaElement>(),failed=ref(false),playError=messageRef(),zoomed=ref(false);
const gainMessage=useMediaAudition(media);
const asset=computed(()=>props.assets.find(item=>item.id===props.assetId&&item.kind===props.kind));
const url=computed(()=>{const value=asset.value;if(!value)return null;return safeMediaSource(props.assetResolver?props.assetResolver(value):assetUrl(props.assetBaseUrl,value.path));});
const caption=computed(()=>unnumberedCaption(props.caption||asset.value?.caption||asset.value?.title||''));
const numberedCaption=computed(()=>caption.value&&props.displayNumber?tr('manual.reader.'+(props.kind==='image'?'imageNumber':props.kind==='audio'?'audioNumber':'videoNumber'),{number:props.displayNumber,caption:caption.value}):caption.value);
const sourceType=computed(()=>{const value=asset.value?.sourceType||'';return ({'v3 实际处理':tr('manual.reader.processedAudio'),'v3 实际处理音频':tr('manual.reader.processedAudio'),'V3 实际处理音频':tr('manual.reader.processedAudio'),'实际应用合成输出':tr('manual.reader.synthesizedAudio'),'实际原生参数模型合成':tr('manual.reader.modelExample'),'合成信号实际操作示例':tr('manual.reader.signalExample'),'作者提供的实际应用屏幕录制':tr('manual.reader.screenRecording')} as Record<string,string>)[value]||value.replace(/^V3\s+/i,'');});
const unavailable=computed(()=>!asset.value||!url.value||failed.value);
const pending=computed(()=>!!asset.value&&!!props.assetResolver?.isLoading?.(asset.value)&&!failed.value);
watch(url,()=>{failed.value=false;playError.value='';zoomed.value=false;});
watch(playbackAllowed,allowed=>{if(!allowed)stop();});
function stop(){if(media.value){media.value.pause();manualPlayback.release(media.value);}}
function onPlay(event:Event){const element=event.currentTarget as HTMLMediaElement;try{manualPlayback.activate(element,playbackAllowed.value);playError.value='';}catch(error){const key=error instanceof Error?(error as Error&{manualMessageKey?:string}).manualMessageKey:undefined;playError.value=message(key??'manual.reader.playbackError');}}
function failure(){failed.value=true;stop();}
function release(event:Event){manualPlayback.release(event.currentTarget as HTMLMediaElement);}
onBeforeUnmount(stop);onDeactivated(stop);
</script>
<template>
  <figure class="manual-media" :class="['manual-media-'+kind,{unavailable}]" :style="{textAlign:align==='center'?'center':align==='right'?'right':'left'}" :data-asset-id="assetId">
    <div v-if="pending" class="manual-loading" role="status">{{tr('manual.reader.loadingImage')}}</div>
    <div v-else-if="unavailable" class="manual-media-error" role="status">{{tr('manual.reader.'+(kind==='image'?'imageError':kind==='audio'?'audioError':'videoError'))}}</div>
    <template v-else>
      <button v-if="kind==='image'" class="manual-image-button" type="button" :aria-label="tr('manual.reader.zoomImage',{label:alt||asset?.alt||caption})" @click="zoomed=true">
        <img :src="url!" :alt="alt||asset?.alt||caption" :width="asset?.width" :height="asset?.height" :style="{width:mediaWidth(width)}" loading="lazy" decoding="async" @error="failure"/>
      </button>
      <audio v-else-if="kind==='audio'" :key="url!" ref="media" :src="url!" controls preload="none" :aria-label="caption||tr('manual.reader.audioExample')" @play="onPlay" @pause="release" @ended="release" @error="failure"/>
      <video v-else :key="url!" ref="media" :src="url!" controls preload="none" playsinline :aria-label="caption||tr('manual.reader.videoExample')" @play="onPlay" @pause="release" @ended="release" @error="failure"/>
      <a v-if="kind!=='image'" class="manual-download" :href="url!" :download="asset?.path.split('/').at(-1)">{{tr('manual.reader.saveExample')}}</a>
    </template>
    <figcaption v-if="numberedCaption">{{numberedCaption}}</figcaption>
    <p v-if="kind!=='image'&&asset" class="manual-media-meta"><span v-if="sourceType">{{sourceType}}</span><span v-if="asset.duration!==undefined">{{tr('manual.reader.seconds',{duration:asset.duration.toFixed(3)})}}</span><span v-if="asset.sampleRate">{{asset.sampleRate}} Hz</span><span v-if="asset.channels">{{tr('manual.reader.channels',{count:asset.channels})}}</span></p>
    <p v-if="playError" role="status" class="manual-media-error">{{playError}}</p>
    <p v-if="gainMessage" role="status">{{gainMessage}}</p>
    <div v-if="zoomed&&!unavailable" class="manual-image-overlay" role="dialog" aria-modal="true" :aria-label="numberedCaption||tr('manual.reader.imagePreview')" tabindex="-1" @click.self="zoomed=false" @keydown.esc="zoomed=false">
      <button type="button" autofocus @click="zoomed=false">{{tr('manual.reader.closeImage')}}</button><img :src="url!" :alt="alt||asset?.alt||caption"/><p>{{numberedCaption}}</p>
    </div>
  </figure>
</template>
