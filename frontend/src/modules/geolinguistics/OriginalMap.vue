<script setup lang="ts">
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
<template><section class="original-map" :aria-label="title+' · 原书地图'">
 <div class="original-tools"><label>原页缩放 <select v-model.number="zoom" aria-label="原书地图缩放"><option :value="0">适合宽度</option><option v-for="n in [35,50,75,100,150]" :key="n" :value="n">{{n}}%</option></select></label><small>PDF 第 {{page}} 页</small></div>
 <div ref="viewport" class="original-scroll" tabindex="0" aria-label="完整原书地图，可滚动查看图例与脚注"><p v-if="failed" role="alert">原书地图读取失败。<button @click="failed=false;attempt++">重新载入</button></p><img v-else :key="src+attempt" :src="src" :alt="title+'，完整原书地图和图例'" :style="style" draggable="false" @load="width=($event.target as HTMLImageElement).naturalWidth" @error="failed=true"/></div>
</section></template>
<style scoped>
.original-map{display:flex;flex-direction:column;flex:1;min-width:0;min-height:0;background:var(--panel)}.original-tools{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:5px 8px;flex-wrap:wrap;border-bottom:1px solid var(--border)}label{display:flex;align-items:center;gap:6px}small{color:var(--muted)}select,button{font:inherit;min-height:var(--control-height)}.original-scroll{flex:1;overflow:auto;min-height:160px;background:#e9e8e3;scrollbar-gutter:stable}.original-scroll img{display:block;max-width:none;height:auto;margin-inline:auto}.original-scroll p{padding:16px;color:#203445}
</style>
