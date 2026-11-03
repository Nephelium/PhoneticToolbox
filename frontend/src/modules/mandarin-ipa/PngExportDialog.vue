<script setup lang="ts">
import {uiErrorMessage as displayErrorValue} from '../module-localization.ts';
import {tr,messageRef} from '../../i18n/index.ts';
import {onBeforeUnmount,reactive,ref,shallowRef,watch} from 'vue';
import ModalDialog from '../../components/ModalDialog.vue';
import {backgroundAppearance,backgroundName,backgroundUrl,prepareBackground} from '../../state/background.ts';
import {downloadMandarinIpa,renderMandarinIpaPng,type MandarinIpaExportSnapshot} from './export.ts';
import {pngBackgroundDefaults} from './export-background.ts';
const props=defineProps<{snapshot:MandarinIpaExportSnapshot}>();
const emit=defineEmits<{close:[];busy:[value:boolean]}>();
const current=()=>pngBackgroundDefaults({url:backgroundUrl.value,name:backgroundName.value,color:getComputedStyle(document.documentElement).getPropertyValue('--app'),...backgroundAppearance.value});
const background=reactive(current()),previewUrl=ref(''),rendering=ref(false),reading=ref(false),previewError=messageRef(),fileError=messageRef();
const previewBlob=shallowRef<Blob>();
let customUrl='',revision=0,timer:ReturnType<typeof setTimeout>|undefined;
function useCurrent(){fileError.value='';Object.assign(background,current());if(customUrl){URL.revokeObjectURL(customUrl);customUrl='';}}
async function chooseImage(event:Event){
  const input=event.target as HTMLInputElement,file=input.files?.[0];if(!file)return;
  reading.value=true;fileError.value='';
  try{const prepared=await prepareBackground(file),url=URL.createObjectURL(prepared.blob);const previous=customUrl;customUrl=url;background.imageUrl=url;background.imageName=prepared.name;background.mode='image';if(previous)URL.revokeObjectURL(previous);}
  catch(cause){fileError.value=displayErrorValue(cause);}
  finally{reading.value=false;input.value='';}
}
async function render(ticket:number){
  try{
    const blob=await renderMandarinIpaPng({...props.snapshot,background:{...background}});
    if(ticket!==revision)return;
    const old=previewUrl.value;previewBlob.value=blob;previewUrl.value=URL.createObjectURL(blob);if(old)URL.revokeObjectURL(old);
  }catch(cause){if(ticket===revision){previewError.value=displayErrorValue(cause);previewBlob.value=undefined;}}
  finally{if(ticket===revision){rendering.value=false;emit('busy',false);}}
}
function schedule(){const ticket=++revision;clearTimeout(timer);rendering.value=true;previewBlob.value=undefined;previewError.value='';emit('busy',true);timer=setTimeout(()=>void render(ticket),180);}
watch(background,schedule,{deep:true,immediate:true});
watch(()=>background.mode,value=>{if(value==='color')fileError.value='';});
function save(){if(!previewBlob.value||rendering.value||reading.value||fileError.value)return;downloadMandarinIpa(previewBlob.value);emit('close');}
onBeforeUnmount(()=>{++revision;clearTimeout(timer);emit('busy',false);if(previewUrl.value)URL.revokeObjectURL(previewUrl.value);if(customUrl)URL.revokeObjectURL(customUrl);});
</script>
<template>
  <ModalDialog :title="tr('modules.mandarin-ipa.text.20513987b013')" wide :close-disabled="reading" @close="emit('close')">
    <div class="png-export">
      <div class="background-heading"><p>{{tr('modules.mandarin-ipa.text.7df448cd8559')}}</p><button :disabled="reading" @click="useCurrent">{{tr('modules.mandarin-ipa.text.a2b25bf9c4e4')}}</button></div>
      <fieldset :disabled="reading"><legend>{{tr('modules.mandarin-ipa.text.6b8025600539')}}</legend><div class="background-modes"><label><input v-model="background.mode" type="radio" value="image" name="png-background"/>{{tr('modules.mandarin-ipa.text.36887d9b2ad3')}}</label><label><input v-model="background.mode" type="radio" value="color" name="png-background"/>{{tr('modules.mandarin-ipa.text.882fe2e84241')}}</label></div>
        <template v-if="background.mode==='image'">
          <p>{{background.imageName||tr('modules.mandarin-ipa.text.095e09a0a40c')}}</p>
          <label>{{tr('modules.mandarin-ipa.text.797192e369af')}}<input type="file" :aria-label="tr('modules.mandarin-ipa.text.7a2a6da29f9e')" accept=".png,.jpg,.jpeg,.webp" @change="chooseImage"/></label>
          <p class="hint">{{tr('modules.mandarin-ipa.text.9122676e4092')}}</p>
          <label>{{tr('modules.mandarin-ipa.text.0e134f5a85ab')}} <span>{{background.blur}} px</span><input v-model.number="background.blur" :aria-label="tr('modules.mandarin-ipa.text.f5b3e5f3c9b8')" type="range" min="0" max="40" step="1"/></label>
          <label>{{tr('modules.mandarin-ipa.text.04b30e78ca4d')}} <span>{{background.opacity}}%</span><input v-model.number="background.opacity" :aria-label="tr('modules.mandarin-ipa.text.5b13f85ff266')" type="range" min="0" max="100" step="1"/></label>
        </template>
        <label v-else class="color-field">{{tr('modules.mandarin-ipa.text.882fe2e84241')}}<input v-model="background.color" type="color" :aria-label="tr('modules.mandarin-ipa.text.0774dd27b6d9')"/><span>{{background.color}}</span></label>
      </fieldset>
      <p v-if="reading" role="status">{{tr('modules.mandarin-ipa.text.4f1415758a10')}}</p><p v-else-if="rendering" role="status">{{tr('modules.mandarin-ipa.text.8123c6be61a3')}}</p>
      <p v-if="fileError||previewError" role="alert">{{fileError||previewError}}</p>
      <figure v-if="previewUrl&&!previewError" class="png-preview" :aria-busy="rendering"><figcaption>{{tr('modules.mandarin-ipa.text.7f9b00e62e2c')}}</figcaption><img :src="previewUrl" :alt="tr('modules.mandarin-ipa.text.f9da44e634c7')"/></figure>
    </div>
    <template #footer><button :disabled="reading" @click="emit('close')">{{tr('modules.mandarin-ipa.text.2cd0f3be8738')}}</button><button class="primary" :disabled="rendering||reading||!previewBlob||!!fileError" @click="save">{{tr('modules.mandarin-ipa.text.8c52149aae45')}}</button></template>
  </ModalDialog>
</template>
<style scoped>
.png-export{display:grid;gap:14px;min-width:0}.png-export p{margin:0;line-height:1.6}.background-heading{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.background-heading p{flex:1;min-width:180px}.png-export fieldset{display:grid;gap:12px;min-width:0}.background-modes{display:flex;gap:20px}.background-modes label{display:flex;gap:6px;align-items:center}.png-export fieldset>label{display:grid;gap:5px}.png-export input:not([type=radio],[type=color]){width:100%;min-width:0}.png-export fieldset>.color-field{display:flex;align-items:center;flex-wrap:wrap;gap:10px}.color-field input{width:50px;min-height:34px}.hint{color:var(--muted);font-size:var(--support-size)}.png-preview{display:grid;gap:8px;margin:0}.png-preview img{display:block;max-width:100%;max-height:300px;object-fit:contain;object-position:left top;border:1px solid var(--border)}.png-preview[aria-busy=true]{opacity:.5}[role=alert]{color:var(--danger)}
</style>
