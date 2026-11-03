<script setup lang="ts">
import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized,joinMessages} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {computed,ref,watch} from 'vue';
import {applicationIcon,builtinIcons,iconPreferences,iconBusy,iconError,changeIcon,prepareIconSource,homeIconScale,setHomeIconScale,loadBuiltinIcons} from '../state/app-icon.ts';
import {nativeIcons} from '../platform/icons.ts';
import type {CustomIcon} from '../platform/icons.ts';
import {saveIconSource,removeIconSource,originalIconSource} from '../platform/icon-sources.ts';
import IconCropDialog from './IconCropDialog.vue';
import {onUnmounted} from 'vue';
const props=withDefaults(defineProps<{active?:boolean}>(),{active:true});
watch(()=>props.active,active=>{if(active)void loadBuiltinIcons();},{immediate:true});
const input=ref<HTMLInputElement>(),loading=ref(false),message=messageRef(),importError=messageRef();
const draft=ref<{url:string;name:string;source:Blob;id?:string}>();
const dockDraft=ref(100);
watch(()=>[iconPreferences.value.selected,iconPreferences.value.dockSize?.value],()=>{dockDraft.value=iconPreferences.value.dockSize?.value??100;},{immediate:true});
const dockPreviewStyle=computed(()=>nativeIcons()&&iconPreferences.value.dockSize?{transform:`scale(${dockDraft.value/100})`}:undefined);
async function resizeDock(size:number){message.value='';const ok=await changeIcon({action:'resize',id:iconPreferences.value.selected,size});if(ok)message.value=uiMessage('components.dock_icon_size_saved');else dockDraft.value=iconPreferences.value.dockSize?.value??100;}
function closeCrop(){if(loading.value||iconBusy.value)return;if(draft.value)URL.revokeObjectURL(draft.value.url);draft.value=undefined;importError.value='';iconError.value='';}
onUnmounted(()=>{if(draft.value)URL.revokeObjectURL(draft.value.url);});
async function upload(event:Event){
 const field=event.target as HTMLInputElement,file=field.files?.[0];if(!file)return;
 loading.value=true;message.value='';importError.value='';
 try{const url=await prepareIconSource(file,file.name,true),id=draft.value?.id;if(draft.value)URL.revokeObjectURL(draft.value.url);draft.value={url,name:file.name,source:file,id};}
 catch(error){importError.value=error instanceof Error?uiErrorMessage(error):uiMessage('components.image_import_failed');}
 finally{loading.value=false;field.value='';}
}
async function recrop(item:CustomIcon){loading.value=true;message.value='';importError.value='';iconError.value='';try{const source=await originalIconSource(item.id,item.dataUrl),url=await prepareIconSource(source,item.name);draft.value={url,name:item.name,source,id:item.id};}catch(error){importError.value=error instanceof Error?uiErrorMessage(error):uiMessage('components.the_original_image_could_not_be_read');}finally{loading.value=false;}}
async function applyCrop(dataUrl:string){const current=draft.value;if(!current)return;loading.value=true;importError.value='';
 try{if(!await changeIcon({action:current.id?'replace':'import',id:current.id,name:current.name,dataUrl}))return;
  let sourceSaved=true;try{await saveIconSource(iconPreferences.value.selected,current.source);if(current.id&&current.id!==iconPreferences.value.selected)await removeIconSource(current.id);}catch{sourceSaved=false;}
  URL.revokeObjectURL(current.url);draft.value=undefined;message.value=sourceSaved?uiMessage('components.the_crop_was_applied_and_selected_as_the_current'):uiMessage('components.the_icon_was_saved_but_the_original_image_could');
 }finally{loading.value=false;}
}
async function choose(id:string){message.value='';importError.value='';if(await changeIcon({action:'select',id}))message.value=id==='default'?uiMessage('components.default_icon_restored'):uiMessage('components.icon_changed');}
async function remove(id:string){message.value='';if(await changeIcon({action:'remove',id})){message.value=uiMessage('components.the_custom_icon_was_removed_the_original_image_file');try{await removeIconSource(id);}catch{message.value=joinMessages(uiMessage('components.the_custom_icon_was_removed_the_original_image_file'),uiMessage('components.the_local_copy_of_the_original_image_could_not'));}}}
</script>
<template><section class="icon-settings settings-card" :aria-label="tr('components.application_icon_settings')">
 <div class="settings-heading"><h3>{{tr('components.application_icon')}}</h3><span class="muted">{{tr('components.applied_when_selected')}}</span></div>
 <div class="settings-columns">
  <div class="settings-column icon-library">
   <div class="icon-options builtin-icon-options" :aria-label="tr('components.built_in_icons')"><button v-for="item in builtinIcons" :key="item.id" class="icon-choice" :data-icon-id="item.id" :data-icon-group="item.group" :aria-label="tr('components.application_icon_273')+item.number+(item.id==='default'?tr('components.default'):'')" :aria-pressed="iconPreferences.selected===item.id" :disabled="iconBusy||loading" @click="choose(item.id)"><img :src="item.url" alt=""/></button></div>
   <div class="settings-heading"><h4>{{tr('components.your_own_images')}}</h4><button :disabled="loading||iconBusy||iconPreferences.custom.length>=8" @click="input?.click()">{{loading?tr('components.processing_image'):tr('components.import_image')}}</button></div>
   <input ref="input" class="icon-file-input" type="file" accept=".png,.jpg,.jpeg,.webp,.ico" :aria-label="tr('components.import_custom_icon')" :disabled="loading||iconBusy" @change="upload"/>
   <p class="settings-note">{{tr('components.png_jpg_webp_and_ico_are_supported_up_to')}}</p>
   <div v-if="iconPreferences.custom.length" class="icon-options"><div v-for="item in iconPreferences.custom" :key="item.id" class="custom-icon-item"><button class="icon-choice" :aria-label="item.name" :aria-pressed="iconPreferences.selected===item.id" :disabled="iconBusy||loading" @click="choose(item.id)"><img :src="item.dataUrl" alt=""/></button><button class="remove-icon" :aria-label="tr('components.recrop_icon')+item.name" :disabled="iconBusy||loading" @click="recrop(item)">{{tr('components.recrop')}}</button><button class="remove-icon" :aria-label="tr('components.remove_icon')+item.name" :disabled="iconBusy||loading" @click="remove(item.id)">{{tr('components.remove')}}</button></div></div>
   <p v-else class="settings-note icon-empty">{{tr('components.import_an_image_you_like_to_make_your_own')}}</p>
  </div>
  <div class="settings-column icon-preview-column">
   <h4>{{tr('components.appearance_preview')}}</h4>
   <div v-if="nativeIcons()&&iconPreferences.dockSize" class="dock-icon-size-setting"><label for="dock-icon-size">{{tr('components.dock_icon_size')}} <output>{{dockDraft}}%</output></label><input id="dock-icon-size" v-model.number="dockDraft" type="range" min="60" max="100" step="1" :disabled="iconBusy||loading" @change="resizeDock(dockDraft)"/><button :disabled="iconBusy||loading||dockDraft===iconPreferences.dockSize.default" @click="resizeDock(iconPreferences.dockSize.default)">{{tr('components.restore_default_size')}}</button><p class="settings-note">{{tr('components.applied_to_the_dock_when_you_release_the_slider')}}</p></div>
   <div class="home-icon-size-setting"><label for="home-icon-size">{{tr('components.home_icon_size')}} <output>{{homeIconScale}}%</output></label><input id="home-icon-size" type="range" min="50" max="300" step="5" :value="homeIconScale" @input="setHomeIconScale(Number(($event.target as HTMLInputElement).value))"/><button :disabled="homeIconScale===100" @click="setHomeIconScale(100)">{{tr('components.restore_default_size')}}</button><p class="settings-note">{{tr('components.adjust_the_icon_in_the_top_left_corner_of')}}</p></div>
   <div class="icon-preview-pair"><div class="icon-stage stage-light"><img :src="applicationIcon" :style="dockPreviewStyle" :alt="tr('components.icon_preview_on_a_light_background')"/><span>{{tr('components.light_background')}}</span></div><div class="icon-stage stage-dark"><img :src="applicationIcon" :style="dockPreviewStyle" :alt="tr('components.icon_preview_on_a_dark_background')"/><span>{{tr('components.dark_background')}}</span></div></div>
   <div class="icon-size-preview"><img v-for="size in [16,24,32,48]" :key="size" :src="applicationIcon" :width="size" :height="size" :alt="size+tr('components.pixel_icon')"/></div>
   <p class="settings-note">{{nativeIcons()?tr('components.the_in_app_and_window_icons_update_together_your'):tr('components.the_icon_is_used_for_the_workspace_in_this')}}</p>
   <p v-if="nativeIcons()&&!iconPreferences.dockSize" class="settings-note">{{tr('components.pinned_windows_taskbar_entries_and_desktop_shortcuts_use_separate')}}</p>
   <button class="icon-reset" :disabled="iconBusy||loading||iconPreferences.selected==='default'" @click="choose('default')">{{tr('components.restore_default_icon')}}</button>
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
.dock-icon-size-setting{display:grid;gap:8px;margin-bottom:18px}.dock-icon-size-setting label{display:flex;align-items:center;justify-content:space-between;gap:12px}.dock-icon-size-setting input{width:100%;min-width:0;accent-color:var(--accent)}.dock-icon-size-setting>button{justify-self:start}.dock-icon-size-setting p{margin:0}
</style>
