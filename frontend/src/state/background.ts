import {ref} from 'vue';
import {projects} from '../platform/browser.ts';
import {normalizeBackground,type BackgroundAppearance} from '../design/background.ts';
import {buttonAppearance,installButtonAppearance,setButtonAppearance} from './buttons.ts';
import {openImageDatabase} from '../platform/image-storage.ts';
export const backgroundAppearance=ref(normalizeBackground(null)),backgroundUrl=ref(''),backgroundName=ref(''),backgroundError=ref('');
let initialized:Promise<void>|undefined,layer:HTMLDivElement;
interface SavedImage {blob:Blob;name:string}
async function imageStore(value?:SavedImage|null):Promise<SavedImage|undefined>{
 const db=await openImageDatabase('ptb-background','image');
 try{return await new Promise((resolve,reject)=>{
  const transaction=db.transaction('image',value===undefined?'readonly':'readwrite'),store=transaction.objectStore('image');
  const request=value===undefined?store.get('current'):value===null?store.delete('current'):store.put(value,'current');
  transaction.oncomplete=()=>resolve(value===undefined?request.result:undefined);
  transaction.onabort=()=>reject(transaction.error);transaction.onerror=()=>reject(transaction.error);
 });}finally{db.close();}
}
function apply(){
 document.documentElement.dataset.backgroundImage=backgroundUrl.value?'on':'off';
 if(!layer)return;
 const {blur,opacity}=backgroundAppearance.value;
 layer.style.backgroundImage=backgroundUrl.value?`url("${backgroundUrl.value}")`:'none';
 layer.style.filter=`blur(${blur}px)`;layer.style.opacity=String(opacity/100);
 layer.style.inset=`-${Math.ceil(blur*3)}px`;
}
function show(image?:SavedImage){
 const old=backgroundUrl.value;
 backgroundUrl.value=image?URL.createObjectURL(image.blob):'';backgroundName.value=image?.name??'';apply();
 if(image){
  installButtonAppearance();
  if(buttonAppearance.value.material!=='glass'&&!setButtonAppearance({material:'glass'}))backgroundError.value='背景图已保存，磨砂玻璃按钮已生效，但按钮设置未能保存。';
 }
 if(old)URL.revokeObjectURL(old);
}
export function initializeBackground(){return initialized??=(async()=>{
 layer=document.createElement('div');layer.className='app-background-layer';layer.setAttribute('aria-hidden','true');document.body.prepend(layer);
 backgroundAppearance.value=normalizeBackground(projects.read('backgroundAppearance',null));apply();
 try{const image=await imageStore();if(image?.blob instanceof Blob)show(image);}
 catch(error){backgroundError.value=error instanceof Error?error.message:'背景图未能读取，请重新选择图片。';}
})();}
export function setBackgroundAppearance(value:Partial<BackgroundAppearance>){
 backgroundAppearance.value=normalizeBackground({...backgroundAppearance.value,...value});apply();
 backgroundError.value=projects.write('backgroundAppearance',backgroundAppearance.value)?'':'背景效果已生效，但设置未能保存。';
}
export async function prepareBackground(file:File):Promise<SavedImage>{
 if(file.size>20*1024*1024)throw Error('请选择不超过 20 MB 的图片。');
 if(!/\.(png|jpe?g|webp)$/i.test(file.name))throw Error('支持 PNG、JPG 和 WebP 图片。');
 const url=URL.createObjectURL(file),image=new Image();
 try{
  image.src=url;await image.decode();
  if(!image.naturalWidth||!image.naturalHeight||image.naturalWidth*image.naturalHeight>40_000_000)throw Error('请选择不超过 4000 万像素的图片。');
  const scale=Math.min(1,2560/Math.max(image.naturalWidth,image.naturalHeight)),canvas=document.createElement('canvas');
  canvas.width=Math.max(1,Math.round(image.naturalWidth*scale));canvas.height=Math.max(1,Math.round(image.naturalHeight*scale));
  const context=canvas.getContext('2d');if(!context)throw Error('当前环境无法处理图片。');
  context.drawImage(image,0,0,canvas.width,canvas.height);
  const blob=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(value=>value?resolve(value):reject(Error('图片处理失败。')),'image/webp',.9));
  return {blob,name:file.name};
 }catch(error){if(error instanceof Error&&error.message.startsWith('请选择'))throw error;throw Error('无法读取这张图片，请选择有效的 PNG、JPG 或 WebP 文件。');}
 finally{URL.revokeObjectURL(url);}
}
export async function changeBackground(file?:File){
 await initializeBackground();backgroundError.value='';
 try{const image=file?await prepareBackground(file):undefined;await imageStore(image??null);show(image);return true;}
 catch(error){backgroundError.value=error instanceof Error?error.message:'背景图未能保存，请重试。';return false;}
}
/** Frames need a local image for backdrop-filter to sample across their own document.
 * Align it with the outer wallpaper and composite once over the same theme base. */
export function syncBackgroundFrame(frame:HTMLIFrameElement){
 const doc=frame.contentDocument;if(!doc||!layer)return;
 let image=doc.querySelector<HTMLDivElement>('.app-background-layer');
 if(!image){image=doc.createElement('div');image.className='app-background-layer';image.setAttribute('aria-hidden','true');doc.body.prepend(image);}
 image.style.cssText=layer.style.cssText;
 image.style.display=backgroundUrl.value?'block':'none';
 const scale=parseFloat(document.documentElement.style.zoom)||1,outer=layer.getBoundingClientRect(),bounds=frame.getBoundingClientRect();
 image.style.inset='auto';image.style.left=(outer.left-bounds.left)/scale+'px';image.style.top=(outer.top-bounds.top)/scale+'px';image.style.width=outer.width/scale+'px';image.style.height=outer.height/scale+'px';
}
