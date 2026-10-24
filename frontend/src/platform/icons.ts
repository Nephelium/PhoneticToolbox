import {builtinIconIds} from './icon-catalog.ts';
export interface CustomIcon {id:string;name:string;dataUrl:string}
export interface DockIconSize {value:number;default:number;min:60;max:100}
export interface IconPreferences {version:1;selected:string;custom:CustomIcon[];dockSize?:DockIconSize}
export type IconRequest={action:'status'|'select'|'import'|'replace'|'remove'|'resize';id?:string;name?:string;dataUrl?:string;size?:number};
export type IconPort=(request:IconRequest)=>Promise<IconPreferences>;
const key='ptb-application-icons-v1';
export const emptyIcons=():IconPreferences=>({version:1,selected:'default',custom:[]});
export function validateIcons(value:unknown):IconPreferences{
 const state=value as IconPreferences;
 if(!state||state.version!==1||!Array.isArray(state.custom)||state.custom.length>8)throw Error('图标设置无法读取。');
 // Earlier W1 and retired K1-A/X3 selections follow the current default.
 if(['builtin-001','builtin-002','builtin-008'].includes(state.selected))state.selected='default';
 const ids=new Set(builtinIconIds);
 for(const icon of state.custom){
  if(!icon||typeof icon.id!=='string'||!/^custom-[a-f0-9]{64}$/.test(icon.id)||ids.has(icon.id)||typeof icon.name!=='string'||icon.name.length>80||typeof icon.dataUrl!=='string'||icon.dataUrl.length>540000||!/^data:image\/png;base64,[A-Za-z0-9+/]+=*$/.test(icon.dataUrl))throw Error('图标设置无法读取。');
  ids.add(icon.id);
 }
 if(!ids.has(state.selected))throw Error('所选图标不存在。');
 if(state.dockSize){const s=state.dockSize;if(s.min!==60||s.max!==100||![s.value,s.default].every(v=>Number.isInteger(v)&&v>=60&&v<=100))throw Error('Dock 图标大小无法读取。');}
 return state;
}
let native:IconPort|undefined;
export function installIconPort(port:IconPort){native=port;}
export function nativeIcons(){return !!native;}
export async function iconRequest(request:IconRequest):Promise<IconPreferences>{
 if(native)return validateIcons(await native(request));
 if(request.action==='resize')throw Error('Dock 图标大小仅适用于 macOS 桌面版。');
 const raw=localStorage.getItem(key),state=raw?validateIcons(JSON.parse(raw)):emptyIcons();
 if(request.action==='status')return state;
 if(request.action==='import'||request.action==='replace'){
  const previous=request.action==='replace'?state.custom.find(icon=>icon.id===request.id):undefined;
  if(request.action==='replace'&&!previous)throw Error('待裁剪的自定义图标不存在。');
  if(!request.dataUrl||request.dataUrl.length>540000||!/^data:image\/png;base64,[A-Za-z0-9+/]+=*$/.test(request.dataUrl))throw Error('图标图片无效。');
  const binary=atob(request.dataUrl.split(',')[1]!),bytes=Uint8Array.from(binary,c=>c.charCodeAt(0));
  const hash=await crypto.subtle.digest('SHA-256',bytes),id='custom-'+Array.from(new Uint8Array(hash),n=>n.toString(16).padStart(2,'0')).join('');
  const entry={id,name:(request.name||previous?.name||'自定义图标').slice(0,80),dataUrl:request.dataUrl};
  if(previous){
   if(id===previous.id)state.custom=state.custom.map(icon=>icon.id===previous.id?entry:icon);
   else if(state.custom.some(icon=>icon.id===id))state.custom=state.custom.filter(icon=>icon.id!==previous.id);
   else state.custom=state.custom.map(icon=>icon.id===previous.id?entry:icon);
  }else if(!state.custom.some(icon=>icon.id===id)){
   if(state.custom.length>=8)throw Error('最多保留 8 张自定义图标，请先移除不需要的图标。');
   state.custom.push(entry);
  }
  state.selected=id;
 }else{
  if(![...builtinIconIds,...state.custom.map(icon=>icon.id)].includes(request.id||''))throw Error('所选图标不存在。');
  if(request.action==='select')state.selected=request.id!;
  else if(request.action==='remove'){
   if(builtinIconIds.has(request.id!))throw Error('内置图标不能移除。');
   state.custom=state.custom.filter(icon=>icon.id!==request.id);
   if(state.selected===request.id)state.selected='default';
  }else throw Error('不支持的图标操作。');
 }
 validateIcons(state);
 try{localStorage.setItem(key,JSON.stringify(state));}catch{throw Error('图标设置保存失败，原选择保留。请检查浏览器存储空间。');}
 return state;
}
