<script setup lang="ts">
import {tr} from '../../i18n/index.ts';
import {ref,computed,watch,onMounted,onBeforeUnmount,nextTick} from 'vue';
const props=defineProps<{src:string;title:string;page:number}>();
const zoom=ref(0),width=ref(3509),failed=ref(false),attempt=ref(0);
const viewport=ref<HTMLDivElement>(),fitWidth=ref(0);let observer:ResizeObserver|undefined;
const style=computed(()=>({width:zoom.value?width.value*zoom.value/100+'px':fitWidth.value?fitWidth.value+'px':'100%'}));
onMounted(()=>{observer=new ResizeObserver(()=>{fitWidth.value=viewport.value?.clientWidth??0;});if(viewport.value)observer.observe(viewport.value);});
onBeforeUnmount(()=>observer?.disconnect());
watch(zoom,async n=>{if(n===0){await nextTick();viewport.value?.scrollTo(0,0);}});
watch(()=>props.src,()=>{failed.value=false;});
</script>
<template><section class="original-map" :aria-label="title+tr('modules.geolinguistics.original_atlas_map')">
 <div class="original-tools"><label>{{tr('modules.geolinguistics.original_page_zoom')}} <select v-model.number="zoom" :aria-label="tr('modules.geolinguistics.original_atlas_map_zoom')"><option :value="0">{{tr('modules.geolinguistics.fit_width')}}</option><option v-for="n in [35,50,75,100,150]" :key="n" :value="n">{{n}}%</option></select></label><small>{{tr('modules.geolinguistics.pdfPageNumber',{page})}}</small></div>
 <div ref="viewport" class="original-scroll" tabindex="0" :aria-label="tr('modules.geolinguistics.complete_original_map_scroll_to_view_legends_and_footnotes')"><p v-if="failed" role="alert">{{tr('modules.geolinguistics.failed_to_read_original_atlas_map')}}<button @click="failed=false;attempt++">{{tr('modules.geolinguistics.reload')}}</button></p><img v-else :key="src+attempt" :src="src" :alt="title+tr('modules.geolinguistics.complete_original_map_and_legend')" :style="style" draggable="false" @load="width=($event.target as HTMLImageElement).naturalWidth" @error="failed=true"/></div>
</section></template>
<style scoped>
.original-map{display:flex;flex-direction:column;flex:1;min-width:0;min-height:0;background:var(--panel)}.original-tools{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:5px 8px;flex-wrap:wrap;border-bottom:1px solid var(--border)}label{display:flex;align-items:center;gap:6px}small{color:var(--muted)}select,button{font:inherit;min-height:var(--control-height)}.original-scroll{flex:1;overflow:auto;min-height:160px;background:#e9e8e3;scrollbar-gutter:stable}.original-scroll img{display:block;max-width:none;height:auto;margin-inline:auto}.original-scroll p{padding:16px;color:#203445}
</style>
