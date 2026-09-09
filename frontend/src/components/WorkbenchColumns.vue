<script setup lang="ts">
import {ref,onMounted,onUnmounted} from 'vue';
const emit=defineEmits<{'files-keydown':[event:KeyboardEvent]}>();
const grid=ref<HTMLElement>(),shared=ref(false);let observer:ResizeObserver,frame=0;
function measure(){cancelAnimationFrame(frame);frame=requestAnimationFrame(()=>{
 const el=grid.value;if(!el)return;
 const contents=[...el.querySelectorAll<HTMLElement>(':scope > .workbench-pane > .pane-content')];
 shared.value=matchMedia('(min-width:901px)').matches&&contents.length===3&&contents.every(x=>x.getBoundingClientRect().height>el.clientHeight-24);
});}
onMounted(()=>{observer=new ResizeObserver(measure);if(grid.value){observer.observe(grid.value);grid.value.querySelectorAll('.pane-content').forEach(x=>observer.observe(x));}window.addEventListener('resize',measure);});
onUnmounted(()=>{observer?.disconnect();cancelAnimationFrame(frame);window.removeEventListener('resize',measure);});
</script>
<template><div ref="grid" class="workbench-grid workbench-columns" :class="{'shared-scroll':shared}">
<aside class="file-panel workbench-pane" tabindex="0" aria-label="音频列表滚动区" @keydown="emit('files-keydown',$event)"><div class="pane-content"><slot name="files"/></div></aside>
<div class="signal-panel workbench-pane" tabindex="0" aria-label="音频与标注滚动区"><div class="pane-content"><slot/></div></div>
<aside class="parameter-summary workbench-pane" tabindex="0" aria-label="参数与结果滚动区"><div class="pane-content"><slot name="settings"/></div></aside>
</div></template>
