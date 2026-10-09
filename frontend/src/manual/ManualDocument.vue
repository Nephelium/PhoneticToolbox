<script setup lang="ts">
import {computed} from 'vue';
import type {AssetResolver,ManualAsset,ManualChapter,ManualReference,ManualTarget} from './types.ts';
import {chapterDisplayNumbers,manualAnchor} from './content.ts';
import ManualNode from './ManualNode.vue';
import './manual.css';
const props=withDefaults(defineProps<{chapter:ManualChapter;chapterNumber?:number;assets:ManualAsset[];references?:ManualReference[];assetBaseUrl?:string;assetResolver?:AssetResolver;playbackAllowed?:boolean}>(),{chapterNumber:1,references:()=>[],assetBaseUrl:'./manual/',playbackAllowed:true});
const numbering=computed(()=>chapterDisplayNumbers(props.chapter,props.chapterNumber));
const emit=defineEmits<{navigate:[target:ManualTarget]}>();
</script>
<template>
  <article class="manual-document" :id="manualAnchor(chapter.id)" :data-manual-chapter="chapter.id">
    <header class="manual-chapter-header"><h1><span class="manual-chapter-number">{{chapterNumber}}</span> {{chapter.title}}</h1><p v-if="chapter.summary">{{chapter.summary}}</p></header>
    <ManualNode :node="chapter.body" :chapter-id="chapter.id" :numbering="numbering" :assets="assets" :references="references" :asset-base-url="assetBaseUrl" :asset-resolver="assetResolver" :playback-allowed="playbackAllowed" @navigate="emit('navigate',$event)"/>
    <p v-if="!chapter.body.content?.length" class="manual-empty" role="status">本章正文尚未提供。</p>
  </article>
</template>
