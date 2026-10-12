<script setup lang="ts">
import {ref} from 'vue';
import {backgroundAppearance,backgroundUrl,backgroundName,backgroundError,changeBackground,setBackgroundAppearance} from '../state/background.ts';
const field=ref<HTMLInputElement>(),busy=ref(false);
async function upload(event:Event){const input=event.target as HTMLInputElement,file=input.files?.[0];if(!file)return;busy.value=true;try{await changeBackground(file);}finally{busy.value=false;input.value='';}}
async function remove(){busy.value=true;try{await changeBackground();}finally{busy.value=false;}}
</script>
<template><section class="background-settings" aria-label="背景图设置">
 <div class="settings-heading"><h4>背景图</h4><div class="background-actions"><button :disabled="busy" @click="field?.click()">{{busy?'处理图片…':backgroundUrl?'更换背景图':'上传背景图'}}</button><button v-if="backgroundUrl" :disabled="busy" @click="remove">移除背景图</button></div></div>
 <input ref="field" class="visually-hidden" type="file" accept=".png,.jpg,.jpeg,.webp" aria-label="上传背景图" :disabled="busy" @change="upload"/>
 <p class="settings-note">支持 PNG、JPG、WebP，单张不超过 20 MB、4000 万像素。图片等比例铺满所有页面，标签和文本框透出背景，图窗保留底色。上传后自动启用磨砂玻璃按钮。图片保存在本机。</p>
 <p v-if="backgroundName" class="background-name settings-note">{{backgroundName}}</p>
 <div class="background-control"><label for="background-blur">高斯模糊</label><div><input id="background-blur" type="range" min="0" max="40" step="1" :value="backgroundAppearance.blur" @input="setBackgroundAppearance({blur:Number(($event.target as HTMLInputElement).value)})"/><input aria-label="背景图高斯模糊值" type="number" min="0" max="40" :value="backgroundAppearance.blur" @change="setBackgroundAppearance({blur:Number(($event.target as HTMLInputElement).value)})"/><span>px</span></div></div>
 <div class="background-control"><label for="background-opacity">不透明度</label><div><input id="background-opacity" type="range" min="0" max="100" step="1" :value="backgroundAppearance.opacity" @input="setBackgroundAppearance({opacity:Number(($event.target as HTMLInputElement).value)})"/><input aria-label="背景图不透明度值" type="number" min="0" max="100" :value="backgroundAppearance.opacity" @change="setBackgroundAppearance({opacity:Number(($event.target as HTMLInputElement).value)})"/><span>%</span></div></div>
 <p v-if="backgroundError" class="settings-note" role="alert">{{backgroundError}}</p>
</section></template>
<style scoped>
.background-settings{display:grid;gap:12px}.background-settings h4{margin:0;font-size:1rem}.background-actions{display:flex;gap:8px;flex-wrap:wrap}.background-settings>.settings-heading{flex-wrap:wrap;gap:10px}.background-control{display:grid;gap:7px;font-size:var(--control-size)}.background-control>div{display:grid;grid-template-columns:minmax(0,1fr) 70px 22px;gap:10px;align-items:center}.background-control input{width:100%;min-width:0}.background-name{overflow-wrap:anywhere}
</style>
