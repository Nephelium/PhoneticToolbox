<script setup lang="ts">
import {tr} from '../i18n/index.ts';
import {syncNumberInput} from '../design/numeric-input.ts';
import {ref,computed} from 'vue';
import {backgroundAppearance,backgroundUrl,backgroundName,backgroundError,savedBackgrounds,selectedBackground,MAX_BACKGROUNDS,changeBackground,selectBackground,removeBackground,setBackgroundAppearance} from '../state/background.ts';
const field=ref<HTMLInputElement>(),busy=ref(false);
const blurInput=computed({get:()=>backgroundAppearance.value.blur,set:(value:number|string)=>setBackgroundAppearance({blur:Number(value)})});
const opacityInput=computed({get:()=>backgroundAppearance.value.opacity,set:(value:number|string)=>setBackgroundAppearance({opacity:Number(value)})});
async function upload(event:Event){const input=event.target as HTMLInputElement,file=input.files?.[0];if(!file)return;busy.value=true;try{await changeBackground(file);}finally{busy.value=false;input.value='';}}
async function remove(){busy.value=true;try{await changeBackground();}finally{busy.value=false;}}
async function select(id:string){busy.value=true;try{await selectBackground(id);}finally{busy.value=false;}}
async function discard(id:string){busy.value=true;try{await removeBackground(id);}finally{busy.value=false;}}
</script>
<template><section class="background-settings" :aria-label="tr('components.background_image_settings')">
 <div class="settings-heading"><h4>{{tr('components.background_image')}}</h4><div class="background-actions"><button :disabled="busy||savedBackgrounds.length>=MAX_BACKGROUNDS" @click="field?.click()">{{busy?tr('components.processing_image'):tr('components.upload_background_image')}}</button><button v-if="backgroundUrl" :disabled="busy" @click="remove">{{tr('components.no_background_image')}}</button></div></div>
 <input ref="field" class="visually-hidden" type="file" accept=".png,.jpg,.jpeg,.webp" :aria-label="tr('components.upload_background_image')" :disabled="busy" @change="upload"/>
 <p class="settings-note">{{tr('components.png_jpg_and_webp_images_are_supported_up_to')}}</p>
 <div v-if="savedBackgrounds.length" class="background-library" :aria-label="tr('components.saved_background_images')"><div v-for="image in savedBackgrounds" :key="image.id" class="background-item">
  <button class="background-thumbnail" :aria-label="tr('components.use_background_image')+image.name" :aria-pressed="selectedBackground===image.id" :disabled="busy" @click="select(image.id)"><img :src="image.url" :alt="image.name"/></button>
  <button :aria-label="tr('components.remove_background_image')+image.name" :disabled="busy" @click="discard(image.id)">{{tr('components.remove')}}</button>
 </div></div>
 <p v-if="backgroundName" class="background-name settings-note">{{backgroundName}}</p>
 <div class="background-control"><label for="background-blur">{{tr('components.gaussian_blur')}}</label><div><input id="background-blur" type="range" min="0" max="40" step="1" :value="backgroundAppearance.blur" @input="setBackgroundAppearance({blur:Number(($event.target as HTMLInputElement).value)})"/><input :aria-label="tr('components.background_blur_amount')" type="number" min="0" max="40" v-model.lazy.number="blurInput" @change="syncNumberInput($event,blurInput)"/><span>px</span></div></div>
 <div class="background-control"><label for="background-opacity">{{tr('components.opacity')}}</label><div><input id="background-opacity" type="range" min="0" max="100" step="1" :value="backgroundAppearance.opacity" @input="setBackgroundAppearance({opacity:Number(($event.target as HTMLInputElement).value)})"/><input :aria-label="tr('components.background_image_opacity')" type="number" min="0" max="100" v-model.lazy.number="opacityInput" @change="syncNumberInput($event,opacityInput)"/><span>%</span></div></div>
 <p v-if="backgroundError" class="settings-note" role="alert">{{backgroundError}}</p>
</section></template>
<style scoped>
.background-library{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:10px}.background-item{display:flex;flex-direction:column;align-items:center;gap:7px;min-width:0}.background-thumbnail{width:100%;padding:5px;aspect-ratio:4/3;border-radius:10px;overflow:hidden}.background-thumbnail img{display:block;width:100%;height:100%;object-fit:cover;border-radius:6px}.background-thumbnail[aria-pressed=true]{outline:2px solid var(--accent);outline-offset:1px}
.background-settings{display:grid;gap:12px}.background-settings h4{margin:0;font-size:1rem}.background-actions{display:flex;gap:8px;flex-wrap:wrap}.background-settings>.settings-heading{flex-wrap:wrap;gap:10px}.background-control{display:grid;gap:7px;font-size:var(--control-size)}.background-control>div{display:grid;grid-template-columns:minmax(0,1fr) 70px 22px;gap:10px;align-items:center}.background-control input{width:100%;min-width:0}.background-name{overflow-wrap:anywhere}
</style>
