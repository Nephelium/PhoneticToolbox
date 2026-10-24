<script setup lang="ts">
import {onBeforeUnmount,reactive,ref,shallowRef,watch} from 'vue';
import ModalDialog from '../../components/ModalDialog.vue';
import {backgroundAppearance,backgroundName,backgroundUrl,prepareBackground} from '../../state/background.ts';
import {downloadMandarinIpa,renderMandarinIpaPng,type MandarinIpaExportSnapshot} from './export.ts';
import {pngBackgroundDefaults} from './export-background.ts';
const props=defineProps<{snapshot:MandarinIpaExportSnapshot}>();
const emit=defineEmits<{close:[];busy:[value:boolean]}>();
const current=()=>pngBackgroundDefaults({url:backgroundUrl.value,name:backgroundName.value,color:getComputedStyle(document.documentElement).getPropertyValue('--app'),...backgroundAppearance.value});
const background=reactive(current()),previewUrl=ref(''),rendering=ref(false),reading=ref(false),previewError=ref(''),fileError=ref('');
const previewBlob=shallowRef<Blob>();
let customUrl='',revision=0,timer:ReturnType<typeof setTimeout>|undefined;
function useCurrent(){fileError.value='';Object.assign(background,current());if(customUrl){URL.revokeObjectURL(customUrl);customUrl='';}}
async function chooseImage(event:Event){
  const input=event.target as HTMLInputElement,file=input.files?.[0];if(!file)return;
  reading.value=true;fileError.value='';
  try{const prepared=await prepareBackground(file),url=URL.createObjectURL(prepared.blob);const previous=customUrl;customUrl=url;background.imageUrl=url;background.imageName=prepared.name;background.mode='image';if(previous)URL.revokeObjectURL(previous);}
  catch(cause){fileError.value=(cause as Error).message;}
  finally{reading.value=false;input.value='';}
}
async function render(ticket:number){
  try{
    const blob=await renderMandarinIpaPng({...props.snapshot,background:{...background}});
    if(ticket!==revision)return;
    const old=previewUrl.value;previewBlob.value=blob;previewUrl.value=URL.createObjectURL(blob);if(old)URL.revokeObjectURL(old);
  }catch(cause){if(ticket===revision){previewError.value=(cause as Error).message;previewBlob.value=undefined;}}
  finally{if(ticket===revision){rendering.value=false;emit('busy',false);}}
}
function schedule(){const ticket=++revision;clearTimeout(timer);rendering.value=true;previewBlob.value=undefined;previewError.value='';emit('busy',true);timer=setTimeout(()=>void render(ticket),180);}
watch(background,schedule,{deep:true,immediate:true});
watch(()=>background.mode,value=>{if(value==='color')fileError.value='';});
function save(){if(!previewBlob.value||rendering.value||reading.value||fileError.value)return;downloadMandarinIpa(previewBlob.value);emit('close');}
onBeforeUnmount(()=>{++revision;clearTimeout(timer);emit('busy',false);if(previewUrl.value)URL.revokeObjectURL(previewUrl.value);if(customUrl)URL.revokeObjectURL(customUrl);});
</script>
<template>
  <ModalDialog title="PNG 导出设置" wide :close-disabled="reading" @close="emit('close')">
    <div class="png-export">
      <div class="background-heading"><p>导出背景可单独调整，工作台外观保持原设置。</p><button :disabled="reading" @click="useCurrent">使用当前背景</button></div>
      <fieldset :disabled="reading"><legend>导出背景</legend><div class="background-modes"><label><input v-model="background.mode" type="radio" value="image" name="png-background"/>背景图片</label><label><input v-model="background.mode" type="radio" value="color" name="png-background"/>背景颜色</label></div>
        <template v-if="background.mode==='image'">
          <p>{{background.imageName||'请选择背景图片'}}</p>
          <label>替换背景图<input type="file" aria-label="PNG 背景图片" accept=".png,.jpg,.jpeg,.webp" @change="chooseImage"/></label>
          <p class="hint">PNG、JPG 或 WebP，不超过 20 MB、4000 万像素。图片居中铺满，超出部分裁切。</p>
          <label>高斯模糊 <span>{{background.blur}} px</span><input v-model.number="background.blur" aria-label="PNG 高斯模糊" type="range" min="0" max="40" step="1"/></label>
          <label>图片不透明度 <span>{{background.opacity}}%</span><input v-model.number="background.opacity" aria-label="PNG 图片不透明度" type="range" min="0" max="100" step="1"/></label>
        </template>
        <label v-else class="color-field">背景颜色<input v-model="background.color" type="color" aria-label="PNG 背景颜色"/><span>{{background.color}}</span></label>
      </fieldset>
      <p v-if="reading" role="status">正在读取背景图片…</p><p v-else-if="rendering" role="status">正在生成导出预览…</p>
      <p v-if="fileError||previewError" role="alert">{{fileError||previewError}}</p>
      <figure v-if="previewUrl&&!previewError" class="png-preview" :aria-busy="rendering"><figcaption>导出预览</figcaption><img :src="previewUrl" alt="国际音标 PNG 导出预览"/></figure>
    </div>
    <template #footer><button :disabled="reading" @click="emit('close')">取消</button><button class="primary" :disabled="rendering||reading||!previewBlob||!!fileError" @click="save">导出 PNG</button></template>
  </ModalDialog>
</template>
<style scoped>
.png-export{display:grid;gap:14px;min-width:0}.png-export p{margin:0;line-height:1.6}.background-heading{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.background-heading p{flex:1;min-width:180px}.png-export fieldset{display:grid;gap:12px;min-width:0}.background-modes{display:flex;gap:20px}.background-modes label{display:flex;gap:6px;align-items:center}.png-export fieldset>label{display:grid;gap:5px}.png-export input:not([type=radio],[type=color]){width:100%;min-width:0}.png-export fieldset>.color-field{display:flex;align-items:center;flex-wrap:wrap;gap:10px}.color-field input{width:50px;min-height:34px}.hint{color:var(--muted);font-size:var(--support-size)}.png-preview{display:grid;gap:8px;margin:0}.png-preview img{display:block;max-width:100%;max-height:300px;object-fit:contain;object-position:left top;border:1px solid var(--border)}.png-preview[aria-busy=true]{opacity:.5}[role=alert]{color:var(--danger)}
</style>
