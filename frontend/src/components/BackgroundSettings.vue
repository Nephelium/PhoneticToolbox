<script setup lang="ts">
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
<template><section class="background-settings" aria-label="背景图设置">
 <div class="settings-heading"><h4>背景图</h4><div class="background-actions"><button :disabled="busy||savedBackgrounds.length>=MAX_BACKGROUNDS" @click="field?.click()">{{busy?'处理图片…':'上传背景图'}}</button><button v-if="backgroundUrl" :disabled="busy" @click="remove">不使用背景图</button></div></div>
 <input ref="field" class="visually-hidden" type="file" accept=".png,.jpg,.jpeg,.webp" aria-label="上传背景图" :disabled="busy" @change="upload"/>
 <p class="settings-note">支持 PNG、JPG、WebP，单张不超过 20 MB、4000 万像素。最多保存 8 张背景图，点击图片即可切换，重新打开后继续沿用。图片等比例铺满所有页面，标签和文本框透出背景，图窗保留底色。上传后自动启用磨砂玻璃按钮。</p>
 <div v-if="savedBackgrounds.length" class="background-library" aria-label="已保存背景图"><div v-for="image in savedBackgrounds" :key="image.id" class="background-item">
  <button class="background-thumbnail" :aria-label="'使用背景图：'+image.name" :aria-pressed="selectedBackground===image.id" :disabled="busy" @click="select(image.id)"><img :src="image.url" :alt="image.name"/></button>
  <button :aria-label="'移除背景图：'+image.name" :disabled="busy" @click="discard(image.id)">移除</button>
 </div></div>
 <p v-if="backgroundName" class="background-name settings-note">{{backgroundName}}</p>
 <div class="background-control"><label for="background-blur">高斯模糊</label><div><input id="background-blur" type="range" min="0" max="40" step="1" :value="backgroundAppearance.blur" @input="setBackgroundAppearance({blur:Number(($event.target as HTMLInputElement).value)})"/><input aria-label="背景图高斯模糊值" type="number" min="0" max="40" v-model.lazy.number="blurInput" @change="syncNumberInput($event,blurInput)"/><span>px</span></div></div>
 <div class="background-control"><label for="background-opacity">不透明度</label><div><input id="background-opacity" type="range" min="0" max="100" step="1" :value="backgroundAppearance.opacity" @input="setBackgroundAppearance({opacity:Number(($event.target as HTMLInputElement).value)})"/><input aria-label="背景图不透明度值" type="number" min="0" max="100" v-model.lazy.number="opacityInput" @change="syncNumberInput($event,opacityInput)"/><span>%</span></div></div>
 <p v-if="backgroundError" class="settings-note" role="alert">{{backgroundError}}</p>
</section></template>
<style scoped>
.background-library{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:10px}.background-item{display:flex;flex-direction:column;align-items:center;gap:7px;min-width:0}.background-thumbnail{width:100%;padding:5px;aspect-ratio:4/3;border-radius:10px;overflow:hidden}.background-thumbnail img{display:block;width:100%;height:100%;object-fit:cover;border-radius:6px}.background-thumbnail[aria-pressed=true]{outline:2px solid var(--accent);outline-offset:1px}
.background-settings{display:grid;gap:12px}.background-settings h4{margin:0;font-size:1rem}.background-actions{display:flex;gap:8px;flex-wrap:wrap}.background-settings>.settings-heading{flex-wrap:wrap;gap:10px}.background-control{display:grid;gap:7px;font-size:var(--control-size)}.background-control>div{display:grid;grid-template-columns:minmax(0,1fr) 70px 22px;gap:10px;align-items:center}.background-control input{width:100%;min-width:0}.background-name{overflow-wrap:anywhere}
</style>
