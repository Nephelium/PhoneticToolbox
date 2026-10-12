<script setup lang="ts">
import {ref} from 'vue';
import {applicationIcon,builtinIcons,iconPreferences,iconBusy,iconError,changeIcon,prepareIconSource,homeIconScale,setHomeIconScale} from '../state/app-icon.ts';
import {nativeIcons} from '../platform/icons.ts';
import type {CustomIcon} from '../platform/icons.ts';
import {saveIconSource,removeIconSource,originalIconSource} from '../platform/icon-sources.ts';
import IconCropDialog from './IconCropDialog.vue';
import {onUnmounted} from 'vue';
const input=ref<HTMLInputElement>(),loading=ref(false),message=ref(''),importError=ref('');
const draft=ref<{url:string;name:string;source:Blob;id?:string}>();
function closeCrop(){if(loading.value||iconBusy.value)return;if(draft.value)URL.revokeObjectURL(draft.value.url);draft.value=undefined;importError.value='';iconError.value='';}
onUnmounted(()=>{if(draft.value)URL.revokeObjectURL(draft.value.url);});
async function upload(event:Event){
 const field=event.target as HTMLInputElement,file=field.files?.[0];if(!file)return;
 loading.value=true;message.value='';importError.value='';
 try{const url=await prepareIconSource(file,file.name,true),id=draft.value?.id;if(draft.value)URL.revokeObjectURL(draft.value.url);draft.value={url,name:file.name,source:file,id};}
 catch(error){importError.value=error instanceof Error?error.message:'图片导入失败。';}
 finally{loading.value=false;field.value='';}
}
async function recrop(item:CustomIcon){loading.value=true;message.value='';importError.value='';iconError.value='';try{const source=await originalIconSource(item.id,item.dataUrl),url=await prepareIconSource(source,item.name);draft.value={url,name:item.name,source,id:item.id};}catch(error){importError.value=error instanceof Error?error.message:'原图无法读取。';}finally{loading.value=false;}}
async function applyCrop(dataUrl:string){const current=draft.value;if(!current)return;loading.value=true;importError.value='';
 try{if(!await changeIcon({action:current.id?'replace':'import',id:current.id,name:current.name,dataUrl}))return;
  let sourceSaved=true;try{await saveIconSource(iconPreferences.value.selected,current.source);if(current.id&&current.id!==iconPreferences.value.selected)await removeIconSource(current.id);}catch{sourceSaved=false;}
  URL.revokeObjectURL(current.url);draft.value=undefined;message.value=sourceSaved?'裁剪已应用并设为当前图标，原图保存在本机。':'图标已保存，原图未能保存；重新裁剪时可重新选择图片。';
 }finally{loading.value=false;}
}
async function choose(id:string){message.value='';importError.value='';if(await changeIcon({action:'select',id}))message.value=id==='default'?'已恢复默认图标。':'图标已切换。';}
async function remove(id:string){message.value='';if(await changeIcon({action:'remove',id})){message.value='已移除这张自定义图标，原始图片文件保留。';try{await removeIconSource(id);}catch{message.value+=' 本机原图副本未能清理。';}}}
</script>
<template><section class="icon-settings settings-card" aria-label="应用图标设置">
 <div class="settings-heading"><h3>应用图标</h3><span class="muted">选择后生效</span></div>
 <div class="settings-columns">
  <div class="settings-column icon-library">
   <div class="icon-options builtin-icon-options" aria-label="内置图标"><button v-for="item in builtinIcons" :key="item.id" class="icon-choice" :data-icon-id="item.id" :data-icon-group="item.group" :aria-label="'应用图标 '+item.number+(item.id==='default'?'（默认）':'')" :aria-pressed="iconPreferences.selected===item.id" :disabled="iconBusy||loading" @click="choose(item.id)"><img :src="item.url" alt=""/></button></div>
   <div class="settings-heading"><h4>自己的图片</h4><button :disabled="loading||iconBusy||iconPreferences.custom.length>=8" @click="input?.click()">{{loading?'处理图片…':'导入图片'}}</button></div>
   <input ref="input" class="icon-file-input" type="file" accept=".png,.jpg,.jpeg,.webp,.ico" aria-label="导入自定义图标" :disabled="loading||iconBusy" @change="upload"/>
   <p class="settings-note">支持 PNG、JPG、WebP、ICO，单张不超过 10 MB。导入后先裁剪，默认方形，也可选择其他比例、移除纯色背景。原图保存在本机，最多保存 8 张图标。</p>
   <div v-if="iconPreferences.custom.length" class="icon-options"><div v-for="item in iconPreferences.custom" :key="item.id" class="custom-icon-item"><button class="icon-choice" :aria-label="item.name" :aria-pressed="iconPreferences.selected===item.id" :disabled="iconBusy||loading" @click="choose(item.id)"><img :src="item.dataUrl" alt=""/></button><button class="remove-icon" :aria-label="'重新裁剪图标 '+item.name" :disabled="iconBusy||loading" @click="recrop(item)">重新裁剪</button><button class="remove-icon" :aria-label="'移除图标 '+item.name" :disabled="iconBusy||loading" @click="remove(item.id)">移除</button></div></div>
   <p v-else class="settings-note icon-empty">导入喜欢的图片，做成自己的应用图标。</p>
  </div>
  <div class="settings-column icon-preview-column">
   <h4>效果预览</h4>
   <div class="home-icon-size-setting"><label for="home-icon-size">首页图标大小 <output>{{homeIconScale}}%</output></label><input id="home-icon-size" type="range" min="50" max="300" step="5" :value="homeIconScale" @input="setHomeIconScale(Number(($event.target as HTMLInputElement).value))"/><button :disabled="homeIconScale===100" @click="setHomeIconScale(100)">恢复默认大小</button><p class="settings-note">调整首页左上角的图标，拖动后即时生效。</p></div>
   <div class="icon-preview-pair"><div class="icon-stage stage-light"><img :src="applicationIcon" alt="浅色背景图标预览"/><span>浅色背景</span></div><div class="icon-stage stage-dark"><img :src="applicationIcon" alt="深色背景图标预览"/><span>深色背景</span></div></div>
   <div class="icon-size-preview"><img v-for="size in [16,24,32,48]" :key="size" :src="applicationIcon" :width="size" :height="size" :alt="size+' 像素图标'"/></div>
   <p class="settings-note">{{nativeIcons()?'应用内与窗口图标同步更新，选择保存在本机，升级后继续保留。':'图标用于当前浏览器中的工作台，选择保存在此浏览器。'}}</p>
   <p v-if="nativeIcons()" class="settings-note">Windows 已固定的任务栏入口与桌面快捷方式使用独立图标。</p>
   <button class="icon-reset" :disabled="iconBusy||loading||iconPreferences.selected==='default'" @click="choose('default')">恢复默认图标</button>
  </div>
 </div>
 <p v-if="message" class="settings-note" role="status">{{message}}</p>
 <p v-if="importError||iconError" class="error-banner" role="alert">{{importError||iconError}}</p>
</section><IconCropDialog v-if="draft" :key="draft.url" :url="draft.url" :name="draft.name" :busy="loading||iconBusy" :save-error="importError||iconError" @close="closeCrop" @pick="input?.click()" @apply="applyCrop"/></template>
<style scoped>
.icon-settings{display:flex;flex-direction:column;gap:18px}.icon-settings h4{margin:0;font-size:1rem}.icon-options{display:grid;grid-template-columns:repeat(auto-fill,minmax(105px,1fr));gap:12px}.icon-choice{display:flex;flex-direction:column;align-items:center;gap:7px;min-width:0;width:100%;padding:12px 8px;background:var(--app)}.icon-choice[aria-pressed=true]{border-color:var(--accent);background:var(--selected)}.icon-choice img{width:64px;height:64px;object-fit:contain}.icon-choice span{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.icon-choice small{font-size:.75rem;color:var(--accent)}.icon-library>.icon-options:first-of-type{grid-template-columns:repeat(auto-fill,minmax(105px,135px))}.icon-file-input{position:absolute;width:1px;height:1px;clip-path:inset(50%);overflow:hidden}.custom-icon-item{display:flex;flex-direction:column;gap:3px;min-width:0}.remove-icon{align-self:center;font-size:.8rem;padding:3px 9px}.icon-preview-pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.icon-stage{display:flex;align-items:center;justify-content:center;flex-direction:column;gap:14px;min-height:190px;border-radius:10px;border:1px solid var(--border);padding:18px}.stage-light{background:#f5f3ef;color:#333}.stage-dark{background:#202630;color:#eee}.icon-stage img{width:100px;height:100px;object-fit:contain}.icon-stage span{font-size:.85rem}.icon-size-preview{display:flex;gap:22px;align-items:center;min-height:72px;padding:12px;background:var(--app);border-radius:8px}.icon-size-preview img{object-fit:contain;flex-shrink:0}.icon-reset{align-self:flex-start}.icon-empty{padding:16px;border:1px dashed var(--border);border-radius:8px}
@container module (max-width:480px){.icon-stage img{width:64px;height:64px}.icon-stage{min-height:140px;padding:10px}.icon-size-preview{gap:14px}}
.icon-group{display:flex;flex-direction:column;gap:10px}.icon-group .icon-options,.icon-library>.icon-options{grid-template-columns:repeat(auto-fill,minmax(70px,1fr));gap:8px}.icon-choice{padding:7px;border:2px solid var(--border);border-radius:14px;aspect-ratio:1;justify-content:center}.icon-choice img{width:100%;max-width:72px;height:auto;aspect-ratio:1}.icon-choice[aria-pressed=true]{outline:2px solid var(--accent);outline-offset:1px}.icon-preview-column{position:sticky;top:80px}
.icon-library>.icon-options.builtin-icon-options{grid-template-columns:repeat(auto-fill,minmax(70px,1fr));gap:8px}
@container module (min-width:960px){.icon-settings>.settings-columns{grid-template-columns:1.6fr 1fr}}
@container module (max-width:959px){.icon-preview-column{position:static}}
.home-icon-size-setting{display:grid;gap:8px}.home-icon-size-setting label{display:flex;align-items:center;justify-content:space-between;gap:12px}.home-icon-size-setting input{width:100%;min-width:0;accent-color:var(--accent)}.home-icon-size-setting>button{justify-self:start}.home-icon-size-setting p{margin:0}
</style>
