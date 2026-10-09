<script setup lang="ts">
import {onBeforeUnmount,onMounted,ref,shallowRef} from 'vue';
import ManualReader from '../manual/ManualReader.vue';
import {parseManualProject} from '../manual/content.ts';
import type {ManualProject,ManualTarget,ManualLocation} from '../manual/types.ts';
import {host} from '../state/workspace.ts';
withDefaults(defineProps<{active?:boolean;target?:ManualTarget;requestKey?:number;playbackAllowed?:boolean;returnLabel?:string}>(),{active:true,playbackAllowed:true});
const emit=defineEmits<{returnTool:[]}>();
const project=shallowRef<ManualProject>(),error=ref(''),loading=ref(false);
const assetBaseUrl=import.meta.env.BASE_URL+'manual/';
const abort=new AbortController();
const saved=host.projects.read<ManualLocation|undefined>('manual-location',undefined);
const initialLocation=saved&&typeof saved.chapterId==='string'&&Number.isFinite(saved.scrollTop)?saved:undefined;
async function load(){
  loading.value=true;error.value='';
  try{const response=await fetch(assetBaseUrl+'project.json',{signal:abort.signal,credentials:'same-origin'});if(!response.ok)throw Error('missing');project.value=parseManualProject(await response.json());}
  catch(reason){if(!abort.signal.aborted)error.value='使用说明资源未能加载，请检查软件资源是否完整，或重新打开此标签。';}
  finally{loading.value=false;}
}
onMounted(load);onBeforeUnmount(()=>abort.abort());
</script>
<template>
  <ManualReader v-if="project" :project="project" :active="active" :target="target" :request-key="requestKey" :asset-base-url="assetBaseUrl" :playback-allowed="playbackAllowed" :return-label="returnLabel" :initial-location="initialLocation" @location="host.projects.write('manual-location',$event)" @return-tool="emit('returnTool')"/>
  <section v-else class="manual-start-state" aria-label="使用说明"><p v-if="loading" role="status">正在载入使用说明…</p><template v-else><p role="alert">{{error}}</p><button type="button" @click="load">重新加载</button></template></section>
</template>
<style scoped>
.manual-start-state{padding:24px;color:var(--text);font-size:var(--body-size)}
</style>
