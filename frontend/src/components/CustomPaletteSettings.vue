<script setup lang="ts">
import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {ref,watch} from 'vue';
import {customPaletteColor,setCustomPaletteColor} from '../state/appearance.ts';
import {normalizeHex} from '../design/waveform-color.ts';
import {screenPickerAvailable,pickScreenColor} from '../platform/color-picker.ts';
import {backgroundUrl} from '../state/background.ts';
const emit=defineEmits<{select:[]}>();
const draft=ref(customPaletteColor.value),error=messageRef(),busy=ref(false),sampling=ref(false);
watch(customPaletteColor,value=>draft.value=value);
watch(backgroundUrl,()=>sampling.value=false);
function change(value:string){const hex=normalizeHex(value);if(!hex){error.value=uiMessage('components.enter_a_valid_hex_color_such_as_657b35_or');return;}error.value=setCustomPaletteColor(hex)?'':uiMessage('components.the_custom_palette_was_applied_but_its_settings_could');draft.value=hex;emit('select');}
async function screen(){busy.value=true;error.value='';try{const color=await pickScreenColor(customPaletteColor.value);if(color)change(color);}catch(e){error.value=e instanceof Error?uiErrorMessage(e):uiMessage('components.color_sampling_failed_try_again');}finally{busy.value=false;}}
function sample(event:MouseEvent){if(!sampling.value)return;const image=event.currentTarget as HTMLImageElement,rect=image.getBoundingClientRect();const canvas=document.createElement('canvas');canvas.width=canvas.height=1;const ctx=canvas.getContext('2d');if(!ctx){error.value=uiMessage('components.image_colors_cannot_be_read_in_this_environment');return;}
 const x=Math.min(image.naturalWidth-1,Math.max(0,Math.floor((event.clientX-rect.left)/rect.width*image.naturalWidth))),y=Math.min(image.naturalHeight-1,Math.max(0,Math.floor((event.clientY-rect.top)/rect.height*image.naturalHeight)));
 ctx.drawImage(image,x,y,1,1,0,0,1,1);const pixel=ctx.getImageData(0,0,1,1).data;
 if(!pixel[3]){error.value=uiMessage('components.this_area_is_transparent_choose_a_colored_area');return;}change('#'+Array.from(pixel.slice(0,3)).map(v=>v.toString(16).padStart(2,'0')).join(''));sampling.value=false;
}
</script>
<template><section class="custom-palette-settings" :aria-label="tr('components.custom_palette')">
 <label class="custom-color-label" for="custom-palette-hex">{{tr('components.custom_palette')}}</label>
 <div class="custom-color-controls"><input type="color" :aria-label="tr('components.custom_palette_color_picker')" :value="customPaletteColor" @input="change(($event.target as HTMLInputElement).value)"/><input id="custom-palette-hex" v-model="draft" :aria-label="tr('components.custom_palette_hex_color')" maxlength="7" spellcheck="false" placeholder="#657b35" @change="change(draft)"/><button v-if="screenPickerAvailable()" :disabled="busy" @click="screen">{{busy?tr('components.sampling_color'):tr('components.eyedropper')}}</button></div>
 <p class="settings-note">{{tr('components.choose_a_color_to_generate_matching_light_and_dark')}}</p>
 <template v-if="backgroundUrl"><button :aria-pressed="sampling" @click="sampling=!sampling">{{sampling?tr('components.cancel_sampling_from_background'):tr('components.sample_from_background_image')}}</button><p v-if="sampling" class="settings-note">{{tr('components.click_the_original_image_below_to_sample_a_color')}}</p><img v-if="sampling" class="background-sampler" :src="backgroundUrl" :alt="tr('components.sample_a_color_from_the_background_image')" tabindex="0" @click="sample" @keydown.esc="sampling=false"/></template>
 <p v-if="error" class="settings-note" role="alert">{{error}}</p>
</section></template>
<style scoped>
.custom-palette-settings{display:grid;gap:9px}.custom-color-label{font-weight:550}.custom-color-controls{display:flex;gap:8px;flex-wrap:wrap}.custom-color-controls input[type=color]{width:44px;height:38px;padding:3px;flex:none}.custom-color-controls input:not([type=color]){width:100px;min-width:0;flex:1}.custom-palette-settings>button{justify-self:start}.background-sampler{display:block;width:100%;height:auto;border-radius:7px;border:1px solid var(--border);cursor:crosshair}
</style>
