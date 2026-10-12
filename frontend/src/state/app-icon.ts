import {computed,ref} from 'vue';
import defaultLogo from '../assets/k2.png';
import {emptyIcons,iconRequest,type IconRequest} from '../platform/icons.ts';
import {iconCatalog} from '../platform/icon-catalog.ts';
import {projects} from '../platform/browser.ts';
export const homeIconScale=ref(100);
export function setHomeIconScale(percent:number){
 if(!Number.isFinite(percent))return false;
 const value=Math.max(50,Math.min(300,Math.round(percent)));
 if(!projects.write('homeIconScale',value)){iconError.value='图标大小未能保存，请重试。';return false;}
 homeIconScale.value=value;iconError.value='';return true;
}
export const iconPreferences=ref(emptyIcons()),iconError=ref(''),iconBusy=ref(false);
export const builtinIcons=ref(iconCatalog.icons.map(item=>({...item,url:''})));
export const iconGroups=iconCatalog.groups;
export const applicationIcon=computed(()=>iconPreferences.value.custom.find(icon=>icon.id===iconPreferences.value.selected)?.dataUrl||builtinIcons.value.find(icon=>icon.id===iconPreferences.value.selected)?.url||defaultLogo);
let initialized:Promise<void>|undefined;
function favicon(){let link=document.querySelector<HTMLLinkElement>('link[rel="icon"]');if(!link){link=document.createElement('link');link.rel='icon';document.head.append(link);}link.type='image/png';link.href=applicationIcon.value;}
export async function changeIcon(request:IconRequest){
 if(iconBusy.value)return false;
 iconBusy.value=true;iconError.value='';
 try{iconPreferences.value=await iconRequest(request);favicon();return true;}
 catch(error){iconError.value=error instanceof Error?error.message:'图标更新失败，原选择保留。';return false;}
 finally{iconBusy.value=false;}
}
export function initializeIcons(){return initialized??=(async()=>{
 const saved=projects.read<unknown>('homeIconScale',100);
 homeIconScale.value=typeof saved==='number'&&Number.isFinite(saved)?Math.max(50,Math.min(300,Math.round(saved))):100;
 const sheets=new Map<string,Promise<HTMLImageElement>>();
 const rendered=await Promise.all(iconCatalog.icons.map(async item=>{
  let promise=sheets.get(item.sheet);
  if(!promise){const image=new Image();image.src=import.meta.env.BASE_URL+'app-icons/sheets/'+item.sheet;promise=image.decode().then(()=>image);sheets.set(item.sheet,promise);}
  const image=await promise,[x,y,w,h]=item.rect as [number,number,number,number];
  if(x<0||y<0||x+w>image.naturalWidth||y+h>image.naturalHeight)throw Error('内置图标素材无法读取。');
  const canvas=document.createElement('canvas');canvas.width=canvas.height=256;
  const context=canvas.getContext('2d');if(!context)throw Error('当前环境无法读取内置图标。');
  const scale=256/Math.max(w,h),width=w*scale,height=h*scale,dx=(256-width)/2,dy=(256-height)/2;
  context.beginPath();context.roundRect(dx,dy,width,height,Math.min(width,height)*item.radius);context.clip();
  context.drawImage(image,x,y,w,h,dx,dy,width,height);
  return {...item,url:canvas.toDataURL('image/png')};
 }));
 builtinIcons.value=rendered;
 await changeIcon({action:'status'});
})().catch(error=>{iconError.value=error instanceof Error?error.message:'内置图标加载失败。';});}

export async function prepareIcon(file:File):Promise<string>{
 if(file.size>10*1024*1024)throw Error('请选择不超过 10 MB 的图片。');
 if(!/\.(png|jpe?g|webp|ico)$/i.test(file.name))throw Error('支持 PNG、JPG、WebP 和 ICO 图片。');
 const url=URL.createObjectURL(file),image=new Image();
 try{
  image.src=url;await image.decode();
  if(!image.naturalWidth||!image.naturalHeight||image.naturalWidth*image.naturalHeight>16_777_216||Math.max(image.naturalWidth,image.naturalHeight)>8192)throw Error('图片尺寸过大，请选择不超过 1600 万像素的图片。');
  const canvas=document.createElement('canvas');canvas.width=canvas.height=256;
  const context=canvas.getContext('2d');if(!context)throw Error('当前环境无法处理图片。');
  const scale=256/Math.max(image.naturalWidth,image.naturalHeight),w=image.naturalWidth*scale,h=image.naturalHeight*scale;
  context.drawImage(image,(256-w)/2,(256-h)/2,w,h);
  return canvas.toDataURL('image/png');
 }catch(error){if(error instanceof Error&&error.message.startsWith('图片尺寸'))throw error;throw Error('无法读取这张图片，请选择有效的 PNG、JPG、WebP 或 ICO 文件。');}
 finally{URL.revokeObjectURL(url);}
}
export async function prepareIconSource(blob:Blob,name:string,checkExtension=false){
 if(blob.size>10*1024*1024)throw Error('请选择不超过 10 MB 的图片。');
 if(checkExtension&&!/\.(png|jpe?g|webp|ico)$/i.test(name))throw Error('支持 PNG、JPG、WebP 和 ICO 图片。');
 const url=URL.createObjectURL(blob),image=new Image();
 try{image.src=url;await image.decode();if(!image.naturalWidth||!image.naturalHeight||image.naturalWidth*image.naturalHeight>16_777_216||Math.max(image.naturalWidth,image.naturalHeight)>8192)throw Error('图片尺寸过大，请选择不超过 1600 万像素的图片。');return url;}
 catch(error){URL.revokeObjectURL(url);if(error instanceof Error&&error.message.startsWith('图片尺寸'))throw error;throw Error('无法读取这张图片，请选择有效的 PNG、JPG、WebP 或 ICO 文件。');}
}
