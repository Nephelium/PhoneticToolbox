<script setup lang="ts">
import {computed,onBeforeUnmount,onMounted,ref,shallowRef,watch} from 'vue';
import {locale,tr,supportedLocales,messageRef,message} from '../i18n/index.ts';
import ManualReader from '../manual/ManualReader.vue';
import {manualPackError} from '../manual/errors.ts';
import {ManualImageCache} from '../manual/image-cache.ts';
import {ManualSharedCache} from '../manual/shared-cache.ts';
import {assetUrl,sharedAssetUrl,parseManualProject} from '../manual/content.ts';
import type {AssetResolver,ChapterLoader,ManualProject,ManualTarget,ManualLocation} from '../manual/types.ts';
import {manualPackStore,fetchManualPackDownloads,inspectLocalManualPack,type InstalledManualPack,type ManualPackLocale,type ManualPackSpec} from '../platform/manual-packs.ts';
import {host} from '../state/workspace.ts';
import {modules} from './registry.ts';
import {manualForHost} from './manual-project.ts';
const chapterIcons=Object.fromEntries(modules.map(item=>[item.id.toLowerCase(),item.icon]));
withDefaults(defineProps<{active?:boolean;target?:ManualTarget;requestKey?:number;playbackAllowed?:boolean;returnLabel?:string}>(),{active:true,playbackAllowed:true});
const emit=defineEmits<{returnTool:[]}>();
const project=shallowRef<ManualProject>(),baseProject=shallowRef<ManualProject>(),error=messageRef(''),loading=ref(false),installing=ref(false),packError=messageRef('');
const desiredLocale=ref(locale.value),readingLocale=ref<'zh-CN'|ManualPackLocale>('zh-CN'),installed=shallowRef<InstalledManualPack>(),download=shallowRef<ManualPackSpec>();
const packInput=ref<HTMLInputElement>();
const packState=ref<'none'|'missing'|'outdated'|'installed'|'unpublished'|'unavailable'>('none');
const assetBaseUrl=import.meta.env.BASE_URL+'manual/';
const abort=new AbortController();let generation=0;
const saved=host.projects.read<ManualLocation|undefined>('manual-location',undefined);
const initialLocation=ref(saved&&typeof saved.chapterId==='string'&&Number.isFinite(saved.scrollTop)?saved:undefined);
const indexes=new Map<string,ManualProject>();
const images=new ManualImageCache(path=>manualPackStore().readAsset(readingLocale.value as ManualPackLocale,path));
let sharedCache:ManualSharedCache|undefined;
const sharedReady=ref(host.kind==='desktop');
const sharedMedia=new ManualImageCache(async digest=>{const blob=await sharedCache?.read(digest)??null;if(!blob)sharedReady.value=false;return blob;});
let visibleAssets:string[]=[];
const packRevision=ref(0);const installRevisions=new Map<string,number>();
const repositoryScope=computed(()=>[installed.value?.locale??'zh-CN',installed.value?.version??'',installed.value?.installedAt??'',packRevision.value].join(':'));
const foreign=computed(()=>desiredLocale.value!=='zh-CN');
const sharedAssets=computed(()=>new Map((baseProject.value?.assets??[]).map(asset=>[asset.id,asset])));
function invalidateIndex(language:ManualPackLocale){for(const key of indexes.keys())if(key.startsWith(language+':'))indexes.delete(key);}
function releaseUnusedAssets(identities:string[]){
 visibleAssets=identities;
 images.retain(identities.flatMap(id=>installed.value?.assetOverrides[id]?[installed.value.assetOverrides[id]]:[]));
 sharedMedia.retain(identities.flatMap(id=>!installed.value?.assetOverrides[id]&&sharedAssets.value.get(id)?.sha256?[sharedAssets.value.get(id)!.sha256!]:[]));
}
const assetResolver:AssetResolver=asset=>{
 const path=installed.value?.assetOverrides[asset.id];
 if(readingLocale.value==='zh-CN')return assetUrl(assetBaseUrl,asset.path);
 if(path)return images.resolve(path);
 const digest=sharedAssets.value.get(asset.id)?.sha256;
 return sharedCache&&sharedReady.value&&digest?sharedMedia.resolve(digest):sharedAssetUrl(assetBaseUrl,asset,sharedAssets.value);
};
assetResolver.isLoading=asset=>{const path=installed.value?.assetOverrides[asset.id],digest=sharedAssets.value.get(asset.id)?.sha256;return path?images.isLoading(path):!!(sharedCache&&sharedReady.value&&digest&&sharedMedia.isLoading(digest));};
const chapterLoader:ChapterLoader=async(descriptor,signal)=>{
  if(signal.aborted)throw new DOMException('Aborted','AbortError');
  const language=readingLocale.value;
  if(language!=='zh-CN')return await manualPackStore().readJson(language,descriptor.path) as Awaited<ReturnType<ChapterLoader>>;
  const response=await fetch(assetBaseUrl+descriptor.path,{signal,credentials:'same-origin'});if(!response.ok)throw Error('manual.pack.missingFile');return await response.json();
};
async function ensureBase(){
  if(baseProject.value)return baseProject.value;
  try{
   const response=await fetch(assetBaseUrl+'project.json',{signal:abort.signal,credentials:'same-origin'});if(!response.ok)throw Error('missing');
   baseProject.value=parseManualProject(await response.json());
  }catch(reason){
   if(abort.signal.aborted||host.kind!=='browser'||typeof caches==='undefined')throw reason;
   const packs=await manualPackStore().list();packs.sort((a,b)=>Number(b.locale===desiredLocale.value)-Number(a.locale===desiredLocale.value));
   const cached=await ManualSharedCache.restore(packs);if(!cached)throw reason;baseProject.value=cached;
  }
  if(host.kind==='browser'&&typeof caches!=='undefined')sharedCache=new ManualSharedCache(baseProject.value,assetBaseUrl);
  return baseProject.value;
}
async function load(){
  const current=++generation,requested=desiredLocale.value;
  loading.value=true;error.value='';download.value=undefined;
  try{
    const base=await ensureBase();if(current!==generation)return;
    let next=base,pack:InstalledManualPack|undefined;
    if(requested!=='zh-CN'){
      pack=(await manualPackStore().list()).find(item=>item.locale===requested);
      if(current!==generation)return;
      if(pack&&pack.projectId===base.id&&!!base.sourceDigest&&pack.sourceDigest===base.sourceDigest){
        const key=[requested,pack.version,pack.sourceDigest].join(':');
        next=indexes.get(key)??parseManualProject(await manualPackStore().readJson(requested,'project.json'));
        indexes.set(key,next);while(indexes.size>2)indexes.delete(indexes.keys().next().value!);
        packState.value='installed';
        sharedReady.value=host.kind==='desktop'||!!(sharedCache&&await sharedCache.ready(pack.assetOverrides,new Set(next.assets.map(asset=>asset.id))).catch(()=>false));
      }else{
        packState.value=pack?'outdated':'missing';pack=undefined;
        try{
          const manifest=await fetchManualPackDownloads(assetBaseUrl+'language-packs.json');if(current!==generation)return;
          download.value=manifest.packs.find(item=>item.locale===requested&&item.sourceDigest===base.sourceDigest);
          if(!download.value)packState.value='unpublished';
        }catch{if(current!==generation)return;packState.value='unavailable';}
      }
    }else packState.value='none';
    if(current!==generation)return;
    images.configure([pack?.locale??'zh-CN',pack?.version??'',pack?.sourceDigest??'',pack?.installedAt??'',installRevisions.get(requested)??0].join(':'));
    sharedMedia.configure(base.sourceDigest??'');if(!pack)sharedMedia.retain([]);
    packRevision.value=installRevisions.get(requested)??0;installed.value=pack;readingLocale.value=pack?requested:'zh-CN';project.value=manualForHost(next,host.kind);
  }catch(reason){if(current===generation&&!abort.signal.aborted)error.value=message('manual.reader.resourceError');}
  finally{if(current===generation)loading.value=false;}
}
async function install(){
  const spec=download.value;if(!spec||installing.value)return;
  installing.value=true;packError.value='';
  try{const pack=await manualPackStore().install(spec);invalidateIndex(spec.locale);installRevisions.set(spec.locale,(installRevisions.get(spec.locale)??0)+1);try{await cacheShared(pack);sharedMedia.clear();releaseUnusedAssets(visibleAssets);}finally{await load();}}
  catch(reason){packError.value=manualPackError(reason);}
  finally{installing.value=false;}
}
async function importFile(event:Event){
  const input=event.target as HTMLInputElement,file=input.files?.[0],language=desiredLocale.value;input.value='';
  if(!file||language==='zh-CN'||installing.value)return;
  const store=manualPackStore();if(!store.importArchive){packError.value=message('manual.reader.pack.importUnavailable');return;}
  installing.value=true;packError.value='';
  try{
    const base=await ensureBase();
    const spec=await inspectLocalManualPack(file,{locale:language,projectId:base.id,sourceDigest:base.sourceDigest??''});
    const pack=await store.importArchive(spec,file);invalidateIndex(spec.locale);installRevisions.set(spec.locale,(installRevisions.get(spec.locale)??0)+1);try{await cacheShared(pack);sharedMedia.clear();releaseUnusedAssets(visibleAssets);}finally{await load();}
  }
  catch(reason){packError.value=manualPackError(reason);}
  finally{installing.value=false;}
}
async function cacheShared(pack:InstalledManualPack){
 if(host.kind==='desktop')return;
 if(!sharedCache)throw Error('manual.pack.sharedUnavailable');
 const key=[pack.locale,pack.version,pack.sourceDigest].join(':');
 const index=indexes.get(key)??parseManualProject(await manualPackStore().readJson(pack.locale,'project.json'));
 indexes.set(key,index);while(indexes.size>2)indexes.delete(indexes.keys().next().value!);
 await sharedCache.install(pack.assetOverrides,new Set(index.assets.map(asset=>asset.id)));
}
async function completeOffline(){
 const pack=installed.value;if(!pack||installing.value)return;
 installing.value=true;packError.value='';
 try{await cacheShared(pack);sharedMedia.clear();releaseUnusedAssets(visibleAssets);sharedReady.value=true;}
 catch(reason){packError.value=manualPackError(reason);}
 finally{installing.value=false;}
}
function saveLocation(location:ManualLocation){initialLocation.value=location;host.projects.write('manual-location',location);}
watch(locale,value=>{desiredLocale.value=value;});
watch(desiredLocale,()=>void load());
onMounted(load);onBeforeUnmount(()=>{generation++;abort.abort();images.dispose();sharedMedia.dispose();});
</script>
<template>
  <section class="manual-help-page" :aria-label="tr('manual.reader.title')">
    <div class="manual-language-toolbar">
      <label>{{tr('manual.reader.pack.language')}} <select v-model="desiredLocale" :disabled="installing" :aria-label="tr('manual.reader.pack.language')"><option v-for="language in supportedLocales" :key="language.code" :value="language.code">{{language.nativeName}}</option></select></label>
      <template v-if="foreign&&packState==='installed'&&!loading"><span>{{tr(sharedReady?'manual.reader.pack.offline':'manual.reader.pack.sharedMissing',{version:installed?.version||''})}}</span><button v-if="!sharedReady" type="button" :disabled="installing" @click="completeOffline">{{tr(installing?'manual.reader.pack.installing':'manual.reader.pack.cacheShared')}}</button></template>
      <template v-else-if="foreign&&!loading"><span role="status">{{tr('manual.reader.pack.'+(packState==='unpublished'?'unpublished':packState==='unavailable'?'unavailable':packState==='outdated'?'outdated':'missing'))}}</span><button v-if="download" type="button" :disabled="installing" @click="install">{{tr('manual.reader.pack.'+(installing?'installing':'download'))}}</button></template>
      <template v-if="foreign&&!loading"><button type="button" :disabled="installing" @click="packInput?.click()">{{tr('manual.reader.pack.import')}}</button><input ref="packInput" type="file" accept=".zip,application/zip" hidden :disabled="installing" :aria-label="tr('manual.reader.pack.import')" @change="importFile"/></template>
      <span v-if="packError" class="manual-pack-error" role="alert">{{packError}}</span>
    </div>
    <p v-if="loading" class="manual-start-state" role="status">{{tr('manual.reader.loading')}}</p>
    <template v-else-if="error"><p class="manual-start-state" role="alert">{{error}}</p><button type="button" @click="load">{{tr('manual.reader.reload')}}</button></template>
    <ManualReader v-if="project" :project="project" :repository-scope="repositoryScope" :chapter-icons="chapterIcons" :active="active" :target="target" :request-key="requestKey" :asset-base-url="assetBaseUrl" :chapter-loader="chapterLoader" :asset-resolver="assetResolver" :playback-allowed="playbackAllowed" :return-label="returnLabel" :initial-location="initialLocation" @chapter-assets="releaseUnusedAssets" @location="saveLocation" @return-tool="emit('returnTool')"/>
  </section>
</template>
<style scoped>
.manual-help-page{height:100%;min-height:0;display:flex;flex-direction:column}.manual-help-page>.manual-reader{flex:1;min-height:0}.manual-language-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:8px 14px;padding:8px 16px;border-bottom:1px solid var(--line);font-size:var(--small-size);color:var(--text)}.manual-language-toolbar select{margin-inline-start:6px}.manual-pack-error{color:var(--danger)}.manual-start-state{padding:24px;color:var(--text);font-size:var(--body-size)}
</style>
