import {ref,watch} from 'vue';
import {projects} from '../platform/browser.ts';
import {normalizeBackground,type BackgroundAppearance} from '../design/background.ts';
import {buttonAppearance,installButtonAppearance,setButtonAppearance} from './buttons.ts';
import {openImageDatabase} from '../platform/image-storage.ts';
import {refreshGlassWallpaper,syncGlassWallpaperFrame} from './glass-wallpaper.ts';
export const backgroundAppearance=ref(normalizeBackground(null)),backgroundUrl=ref(''),backgroundName=ref(''),backgroundError=ref('');
export const savedBackgrounds=ref<{id:string;name:string;url:string}[]>([]),selectedBackground=ref<string|null>(null);
export const MAX_BACKGROUNDS=8;
let initialized:Promise<void>|undefined,layer:HTMLDivElement;
interface SavedImage {blob:Blob;name:string}
interface SavedBackground extends SavedImage {id:string}
interface BackgroundLibrary {items:SavedBackground[];selected:string|null}
let library:BackgroundLibrary={items:[],selected:null};
let libraryLoaded=false;
async function imageStore(value?:BackgroundLibrary):Promise<BackgroundLibrary>{
 const db=await openImageDatabase('ptb-background','image');
 try{return await new Promise((resolve,reject)=>{
  const transaction=db.transaction('image',value===undefined?'readonly':'readwrite'),store=transaction.objectStore('image');
  if(value){store.put(value,'library');store.delete('current');transaction.oncomplete=()=>resolve(value);}
  else{
   const request=store.get('library'),legacy=store.get('current');
   transaction.oncomplete=()=>{
    if(request.result){resolve(request.result);return;}
    const image=legacy.result as SavedImage|undefined;
    resolve(image?.blob instanceof Blob?{items:[{...image,id:'legacy'}],selected:'legacy'}:{items:[],selected:null});
   };
  }
  transaction.onabort=()=>reject(transaction.error);transaction.onerror=()=>reject(transaction.error);
 });}finally{db.close();}
}
function displayLibrary(value:BackgroundLibrary){
 const old=savedBackgrounds.value;
 const next=value.items.map(image=>({id:image.id,name:image.name,url:URL.createObjectURL(image.blob)}));
 library=value;libraryLoaded=true;savedBackgrounds.value=next;selectedBackground.value=value.selected;
 show(value.items.find(image=>image.id===value.selected));
 for(const image of old)URL.revokeObjectURL(image.url);
}
function apply(){
 document.documentElement.dataset.backgroundImage=backgroundUrl.value?'on':'off';
 if(!layer)return;
 const {blur,opacity}=backgroundAppearance.value;
 layer.style.backgroundImage=backgroundUrl.value?`url("${backgroundUrl.value}")`:'none';
 layer.style.filter=`blur(${blur}px)`;layer.style.opacity=String(opacity/100);
 layer.style.inset=`-${Math.ceil(blur*3)}px`;
 refreshGlassWallpaper({url:backgroundUrl.value,blur,opacity,glassBlur:buttonAppearance.value.glassBlur});
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
export function installBackgroundAppearance(){backgroundAppearance.value=normalizeBackground(projects.read('backgroundAppearance',null));apply();}
export function initializeBackground(){return initialized??=(async()=>{
 layer=document.createElement('div');layer.className='app-background-layer';layer.setAttribute('aria-hidden','true');document.body.prepend(layer);
 installBackgroundAppearance();
 watch(()=>buttonAppearance.value.glassBlur,apply);
 try{const value=await imageStore();displayLibrary(value);}
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
 try{
  if(!libraryLoaded)displayLibrary(await imageStore());
  if(!file){await imageStore({...library,selected:null});displayLibrary({...library,selected:null});return true;}
  const image=await prepareBackground(file),digest=await crypto.subtle.digest('SHA-256',await image.blob.arrayBuffer());
  const id=Array.from(new Uint8Array(digest),byte=>byte.toString(16).padStart(2,'0')).join('');
  const existing=library.items.find(item=>item.id===id||item.id==='legacy'&&item.blob.size===image.blob.size&&item.blob.type===image.blob.type);
  // Compare legacy bytes before treating an upload as the already saved image.
  let match=existing;
  if(match?.id==='legacy'){
   const a=new Uint8Array(await match.blob.arrayBuffer()),b=new Uint8Array(await image.blob.arrayBuffer());
   if(!a.every((byte,index)=>byte===b[index]))match=undefined;
  }
  if(!match&&library.items.length>=MAX_BACKGROUNDS)throw Error('最多保存 8 张背景图，请先移除一张再上传。');
  const value={items:match?library.items:[...library.items,{...image,id}],selected:match?.id??id};
  await imageStore(value);displayLibrary(value);return true;
 }
 catch(error){backgroundError.value=error instanceof Error?error.message:'背景图未能保存，请重试。';return false;}
}
export async function selectBackground(id:string){
 await initializeBackground();backgroundError.value='';
 try{if(!libraryLoaded)displayLibrary(await imageStore());if(!library.items.some(image=>image.id===id))throw Error('这张背景图已移除，请选择其他图片。');const value={...library,selected:id};await imageStore(value);displayLibrary(value);return true;}
 catch(error){backgroundError.value=error instanceof Error?error.message:'背景图未能切换，请重试。';return false;}
}
export async function removeBackground(id:string){
 await initializeBackground();backgroundError.value='';
 try{if(!libraryLoaded)displayLibrary(await imageStore());const value={items:library.items.filter(image=>image.id!==id),selected:library.selected===id?null:library.selected};await imageStore(value);displayLibrary(value);return true;}
 catch(error){backgroundError.value=error instanceof Error?error.message:'背景图未能移除，请重试。';return false;}
}
/** Align the frame wallpaper and shared glass bitmap with the outer document. */
export function syncBackgroundFrame(frame:HTMLIFrameElement){
 const doc=frame.contentDocument;if(!doc?.body||!layer)return;
 let image=doc.querySelector<HTMLDivElement>('.app-background-layer');
 if(!image){image=doc.createElement('div');image.className='app-background-layer';image.setAttribute('aria-hidden','true');doc.body.prepend(image);}
 image.style.cssText=layer.style.cssText;
 image.style.display=backgroundUrl.value?'block':'none';
 const scale=parseFloat(document.documentElement.style.zoom)||1,outer=layer.getBoundingClientRect(),bounds=frame.getBoundingClientRect();
 image.style.inset='auto';image.style.left=(outer.left-bounds.left)/scale+'px';image.style.top=(outer.top-bounds.top)/scale+'px';image.style.width=outer.width/scale+'px';image.style.height=outer.height/scale+'px';
 syncGlassWallpaperFrame(frame);
}
