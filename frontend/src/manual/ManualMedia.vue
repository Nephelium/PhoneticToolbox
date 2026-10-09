<script setup lang="ts">
import {computed,onBeforeUnmount,onDeactivated,ref,watch} from 'vue';
import type {AssetResolver,ManualAsset} from './types.ts';
import {assetUrl,safeMediaSource,unnumberedCaption} from './content.ts';
import {manualPlayback} from './playback.ts';
import {mediaWidth} from './styles.ts';
const props=withDefaults(defineProps<{kind:'image'|'audio'|'video';assetId:string;assets:ManualAsset[];assetBaseUrl?:string;assetResolver?:AssetResolver;caption?:string;displayNumber?:string;alt?:string;width?:unknown;align?:unknown;playbackAllowed?:boolean}>(),{assetBaseUrl:'./manual/',playbackAllowed:true});
const media=ref<HTMLMediaElement>(),failed=ref(false),playError=ref(''),zoomed=ref(false);
const asset=computed(()=>props.assets.find(item=>item.id===props.assetId&&item.kind===props.kind));
const url=computed(()=>{const value=asset.value;if(!value)return null;return safeMediaSource(props.assetResolver?props.assetResolver(value):assetUrl(props.assetBaseUrl,value.path));});
const caption=computed(()=>unnumberedCaption(props.caption||asset.value?.caption||asset.value?.title||''));
const numberedCaption=computed(()=>caption.value&&props.displayNumber?(props.kind==='image'?'图':props.kind==='audio'?'例音':'视频')+' '+props.displayNumber+'：'+caption.value:caption.value);
const source=computed(()=>{const value=asset.value?.source;return typeof value==='string'?value:typeof value?.label==='string'?value.label:'';});
const unavailable=computed(()=>!asset.value||!url.value||failed.value);
watch(url,()=>{failed.value=false;playError.value='';zoomed.value=false;});
watch(()=>props.playbackAllowed,allowed=>{if(!allowed)stop();});
function stop(){if(media.value){media.value.pause();manualPlayback.release(media.value);}}
function onPlay(event:Event){const element=event.currentTarget as HTMLMediaElement;try{manualPlayback.activate(element,props.playbackAllowed);playError.value='';}catch(error){playError.value=error instanceof Error?error.message:'暂时无法播放说明书音频。';}}
function failure(){failed.value=true;stop();}
function release(event:Event){manualPlayback.release(event.currentTarget as HTMLMediaElement);}
onBeforeUnmount(stop);onDeactivated(stop);
</script>
<template>
  <figure class="manual-media" :class="['manual-media-'+kind,{unavailable}]" :style="{textAlign:align==='center'?'center':align==='right'?'right':'left'}" :data-asset-id="assetId">
    <div v-if="unavailable" class="manual-media-error" role="status">{{kind==='image'?'图片':kind==='audio'?'音频':'视频'}}素材暂不可用。请检查说明书素材包。<span class="manual-media-id">素材标识：{{assetId}}</span></div>
    <template v-else>
      <button v-if="kind==='image'" class="manual-image-button" type="button" :aria-label="'放大图片：'+(alt||asset?.alt||caption)" @click="zoomed=true">
        <img :src="url!" :alt="alt||asset?.alt||caption" :width="asset?.width" :height="asset?.height" :style="{width:mediaWidth(width)}" loading="lazy" decoding="async" @error="failure"/>
      </button>
      <audio v-else-if="kind==='audio'" ref="media" :src="url!" controls preload="none" :aria-label="caption||'说明书示例音频'" @play="onPlay" @pause="release" @ended="release" @error="failure"/>
      <video v-else ref="media" :src="url!" controls preload="none" playsinline :aria-label="caption||'说明书示例视频'" @play="onPlay" @pause="release" @ended="release" @error="failure"/>
      <a v-if="kind!=='image'" class="manual-download" :href="url!" :download="asset?.path.split('/').at(-1)">保存示例文件</a>
    </template>
    <figcaption v-if="numberedCaption">{{numberedCaption}}</figcaption>
    <p v-if="kind!=='image'&&asset" class="manual-media-meta"><span v-if="asset.sourceType">{{asset.sourceType}}</span><span v-if="asset.duration!==undefined">{{asset.duration.toFixed(3)}} 秒</span><span v-if="asset.sampleRate">{{asset.sampleRate}} Hz</span><span v-if="asset.channels">{{asset.channels}} 声道</span><span v-if="source">{{source}}</span></p>
    <p v-if="playError" role="status" class="manual-media-error">{{playError}}</p>
    <div v-if="zoomed&&!unavailable" class="manual-image-overlay" role="dialog" aria-modal="true" :aria-label="numberedCaption||'图片预览'" tabindex="-1" @click.self="zoomed=false" @keydown.esc="zoomed=false">
      <button type="button" autofocus @click="zoomed=false">关闭图片</button><img :src="url!" :alt="alt||asset?.alt||caption"/><p>{{numberedCaption}}</p>
    </div>
  </figure>
</template>
