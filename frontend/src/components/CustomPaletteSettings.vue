<script setup lang="ts">
import {ref,watch} from 'vue';
import {customPaletteColor,setCustomPaletteColor} from '../state/appearance.ts';
import {normalizeHex} from '../design/waveform-color.ts';
import {screenPickerAvailable,pickScreenColor} from '../platform/color-picker.ts';
import {backgroundUrl} from '../state/background.ts';
const emit=defineEmits<{select:[]}>();
const draft=ref(customPaletteColor.value),error=ref(''),busy=ref(false),sampling=ref(false);
watch(customPaletteColor,value=>draft.value=value);
watch(backgroundUrl,()=>sampling.value=false);
function change(value:string){const hex=normalizeHex(value);if(!hex){error.value='请输入有效的 HEX 颜色，如 #657b35 或 #abc。';return;}error.value=setCustomPaletteColor(hex)?'':'自定义色系已生效，但设置未能保存。';draft.value=hex;emit('select');}
async function screen(){busy.value=true;error.value='';try{const color=await pickScreenColor(customPaletteColor.value);if(color)change(color);}catch(e){error.value=e instanceof Error?e.message:'取色失败，请重试。';}finally{busy.value=false;}}
function sample(event:MouseEvent){if(!sampling.value)return;const image=event.currentTarget as HTMLImageElement,rect=image.getBoundingClientRect();const canvas=document.createElement('canvas');canvas.width=canvas.height=1;const ctx=canvas.getContext('2d');if(!ctx){error.value='当前环境无法读取图片颜色。';return;}
 const x=Math.min(image.naturalWidth-1,Math.max(0,Math.floor((event.clientX-rect.left)/rect.width*image.naturalWidth))),y=Math.min(image.naturalHeight-1,Math.max(0,Math.floor((event.clientY-rect.top)/rect.height*image.naturalHeight)));
 ctx.drawImage(image,x,y,1,1,0,0,1,1);const pixel=ctx.getImageData(0,0,1,1).data;
 if(!pixel[3]){error.value='这里是透明区域，请选择有颜色的位置。';return;}change('#'+Array.from(pixel.slice(0,3)).map(v=>v.toString(16).padStart(2,'0')).join(''));sampling.value=false;
}
</script>
<template><section class="custom-palette-settings" aria-label="自定义色系">
 <label class="custom-color-label" for="custom-palette-hex">自定义色系</label>
 <div class="custom-color-controls"><input type="color" aria-label="自定义色系色盘" :value="customPaletteColor" @input="change(($event.target as HTMLInputElement).value)"/><input id="custom-palette-hex" v-model="draft" aria-label="自定义色系 HEX 颜色" maxlength="7" spellcheck="false" placeholder="#657b35" @change="change(draft)"/><button v-if="screenPickerAvailable()" :disabled="busy" @click="screen">{{busy?'正在取色…':'吸管取色'}}</button></div>
 <p class="settings-note">选定一种颜色，即时生成同色系的浅色和深色配色。可用色盘、吸管或 HEX 代码设定。</p>
 <template v-if="backgroundUrl"><button :aria-pressed="sampling" @click="sampling=!sampling">{{sampling?'取消背景图取色':'从背景图取色'}}</button><p v-if="sampling" class="settings-note">点击下方原图取色，Esc 取消。</p><img v-if="sampling" class="background-sampler" :src="backgroundUrl" alt="点击背景图取样颜色" tabindex="0" @click="sample" @keydown.esc="sampling=false"/></template>
 <p v-if="error" class="settings-note" role="alert">{{error}}</p>
</section></template>
<style scoped>
.custom-palette-settings{display:grid;gap:9px}.custom-color-label{font-weight:550}.custom-color-controls{display:flex;gap:8px;flex-wrap:wrap}.custom-color-controls input[type=color]{width:44px;height:38px;padding:3px;flex:none}.custom-color-controls input:not([type=color]){width:100px;min-width:0;flex:1}.custom-palette-settings>button{justify-self:start}.background-sampler{display:block;width:100%;height:auto;border-radius:7px;border:1px solid var(--border);cursor:crosshair}
</style>
