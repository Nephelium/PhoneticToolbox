import { browser } from './browser.ts';
import type { HostCapabilities } from './types.ts';
import type { DirectoryGrant,ResearchFiles,Spectrogram } from './research.ts';
type Bridge={invoke:(body:string,callback:(value:string)=>void)=>void;preview:(id:string,body:string)=>void;previewReady:{connect:(fn:(id:string,body:string)=>void)=>void}};
declare global {interface Window {qt?:{webChannelTransport:unknown};QWebChannel?:new(transport:unknown,callback:(channel:{objects:{files:Bridge}})=>void)=>unknown}}
let active:HostCapabilities=browser;
export let desktopFiles:ResearchFiles|undefined;
export let desktopSession='';
export async function initializePlatform(){
  if(!window.qt?.webChannelTransport||!window.QWebChannel)return;
  const bridge=await new Promise<Bridge>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('桌面握手超时。')),15000);new window.QWebChannel!(window.qt!.webChannelTransport,channel=>{clearTimeout(timer);resolve(channel.objects.files);});});
  const call=<T>(op:string,extra:Record<string,string>={}):Promise<T>=>new Promise((resolve,reject)=>bridge.invoke(JSON.stringify({op,...extra}),text=>{try{const result=JSON.parse(text);if(!result.ok)throw Error(result.error);resolve(result.value);}catch(e){reject(e);}}));
  const hello=await call<{kind:string;session:string;api_version:string}>('hello');
  if(hello.kind!=='desktop'||hello.api_version!=='1.1.0'||!hello.session)throw Error('桌面接口版本不匹配。');
  desktopSession=hello.session;
  const pending=new Map<string,{resolve:(data:Spectrogram)=>void;reject:(error:Error)=>void;timer:ReturnType<typeof setTimeout>}>();
  bridge.previewReady.connect((id,text)=>{const request=pending.get(id);if(!request)return;pending.delete(id);clearTimeout(request.timer);try{const data=JSON.parse(text);if(!data.ok)throw Error(data.error);request.resolve(data.value);}catch(e){request.reject(e instanceof Error?e:Error('语谱图读取失败。'));}});
  desktopFiles={kind:'desktop',choose:(purpose:DirectoryGrant['purpose'])=>call('choose',{purpose}),
    list:(id='')=>call('list',{id}),async read(file){const v=await call<{base64:string;sha256:string}>('read',{id:file.id});const text=atob(v.base64),bytes=new Uint8Array(text.length);for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);return {buffer:bytes.buffer,sha256:v.sha256};},
    textgrid:file=>call('textgrid',{id:file.id}),
    spectrogram:(file,view)=>new Promise((resolve,reject)=>{const id=crypto.randomUUID();const timer=setTimeout(()=>{pending.delete(id);reject(Error('preview_timeout'));},35000);pending.set(id,{resolve,reject,timer});bridge.preview(id,JSON.stringify({id:file.id,...view}));}),dispose(){}};
  active={...browser,kind:'desktop'};
}
export function platform():HostCapabilities{return active;}
