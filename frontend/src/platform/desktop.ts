import { browser } from './browser.ts';
import {serialRequests} from './serial-requests.ts';
import type { HostCapabilities } from './types.ts';
import type { DirectoryGrant,ResearchFiles,Spectrogram,ResearchTasks } from './research.ts';
type Bridge={invoke:(body:string,callback:(value:string)=>void)=>void;preview:(id:string,body:string)=>void;previewReady:{connect:(fn:(id:string,body:string)=>void)=>void};task:(id:string,body:string)=>void;taskReady:{connect:(fn:(id:string,body:string)=>void)=>void}};
declare global {interface Window {qt?:{webChannelTransport:unknown};QWebChannel?:new(transport:unknown,callback:(channel:{objects:{files:Bridge}})=>void)=>unknown}}
let active:HostCapabilities=browser;
export let desktopFiles:ResearchFiles|undefined;
export let desktopSession='';
export let desktopFontFamilies:(()=>Promise<string[]>)|undefined;
export let vocalRequest:((op:string,body?:unknown)=>Promise<unknown>)|undefined;
export async function initializePlatform(){
  if(!window.qt?.webChannelTransport||!window.QWebChannel)return;
  const bridge=await new Promise<Bridge>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('桌面握手超时。')),15000);new window.QWebChannel!(window.qt!.webChannelTransport,channel=>{clearTimeout(timer);resolve(channel.objects.files);});});
  const call=<T>(op:string,extra:Record<string,string>={}):Promise<T>=>new Promise((resolve,reject)=>bridge.invoke(JSON.stringify({op,...extra}),text=>{try{const result=JSON.parse(text);if(!result.ok)throw Error(result.error);resolve(result.value);}catch(e){reject(e);}}));
  const hello=await call<{kind:string;session:string;api_version:string;tasks?:boolean}>('hello');
  if(hello.kind!=='desktop'||hello.api_version!=='1.1.0'||!hello.session)throw Error('桌面接口版本不匹配。');
  desktopSession=hello.session;
  desktopFontFamilies=()=>call<string[]>('fonts');
  const vocalBridge=bridge as Bridge & {vocal?:(id:string,body:string)=>void;vocalReady?:{connect:(callback:(id:string,raw:string)=>void)=>void}};
  if(vocalBridge.vocal&&vocalBridge.vocalReady){
    const requests=new Map<string,{resolve:(value:unknown)=>void;reject:(error:Error)=>void;timer:ReturnType<typeof setTimeout>}>();
    vocalBridge.vocalReady.connect((id,raw)=>{const request=requests.get(id);if(!request)return;requests.delete(id);clearTimeout(request.timer);try{const result=JSON.parse(raw);if(!result.ok)throw Error(result.error);request.resolve(result.value);}catch(e){request.reject(e instanceof Error?e:Error('声道请求失败'));}});
    vocalRequest=(op,body={})=>new Promise((resolve,reject)=>{const id=crypto.randomUUID(),timer=setTimeout(()=>{requests.delete(id);reject(Error('声道请求超时，请重新打开模块。'));},160000);requests.set(id,{resolve,reject,timer});vocalBridge.vocal!(id,JSON.stringify({op,body}));});
  }
  const pending=new Map<string,{resolve:(data:Spectrogram)=>void;reject:(error:Error)=>void;timer:ReturnType<typeof setTimeout>}>();
  bridge.previewReady.connect((id,text)=>{const request=pending.get(id);if(!request)return;pending.delete(id);clearTimeout(request.timer);try{const data=JSON.parse(text);if(!data.ok)throw Error(data.error);request.resolve(data.value);}catch(e){request.reject(e instanceof Error?e:Error('语谱图读取失败。'));}});
  const taskPending=new Map<string,{resolve:(value:any)=>void;reject:(error:Error)=>void;timer:ReturnType<typeof setTimeout>}>();
  bridge.taskReady?.connect((id,text)=>{const request=taskPending.get(id);if(!request)return;taskPending.delete(id);clearTimeout(request.timer);try{const data=JSON.parse(text);if(!data.ok)throw Error(data.error);request.resolve(data.value);}catch(e){request.reject(e instanceof Error?e:Error('任务读取失败。'));}});
  const task=serialRequests(<T>(body:unknown):Promise<T>=>new Promise((resolve,reject)=>{const id=crypto.randomUUID(),timer=setTimeout(()=>{taskPending.delete(id);reject(Error('任务操作超时，刷新批次可核对实际状态。'));},300000);taskPending.set(id,{resolve,reject,timer});bridge.task(id,JSON.stringify(body));}));
  const tasks:ResearchTasks={parent:file=>task({op:'parent',id:file.id}),submit:(operation,inputs,config,layer,key)=>task({op:'submit',operation,inputs:inputs.map(item=>Object.fromEntries(Object.entries(item).filter(([,v])=>v!=null).map(([role,file])=>[role,role==='parent_result'?file:(file as {id:string}).id]))),config,layer,idempotency_key:key}),
    reconstruct:(file,config,key)=>task({op:'reconstruct',id:file.id,config,key}),reconstructions:()=>task({op:'reconstructions'}),cancelJob:id=>task({op:'cancel_job',id}),saveJob:(id,directory)=>task({op:'save_job',id,directory}),
    egg:async(file,config,key)=>{const {exportFontSnapshot}=await import('../state/fonts.ts');return task({op:'egg',id:file.id,config:{...config,font:config.font??exportFontSnapshot()},key});},eggJobs:()=>task({op:'egg_jobs'}),
    async result(job,id,sha){const value=await task<{base64:string;sha256:string}>({op:'result',job,id});if(value.sha256!==sha)throw Error('结果来源已变化。');const text=atob(value.base64);return Uint8Array.from(text,c=>c.charCodeAt(0)).buffer;},
    list:()=>task({op:'list'}),get:id=>task({op:'get',id}),cancel:id=>task({op:'cancel',id}),job:id=>task({op:'job',id}),
    retry:(id,key)=>task({op:'retry',id,key}),save:(id,directory)=>task({op:'save',id,directory})};
  desktopFiles={kind:'desktop',tasks:hello.tasks?tasks:undefined,choose:(purpose:DirectoryGrant['purpose'])=>call('choose',{purpose}),
    parameters:file=>task({op:'parameters',id:file.id}),capture:()=>call('capture'),
    convertLip:hello.tasks?file=>task({op:'convert_lip',id:file.id}):undefined,
    list:(id='')=>call('list',{id}),async read(file){const v=await call<{base64:string;sha256:string}>('read',{id:file.id});const text=atob(v.base64),bytes=new Uint8Array(text.length);for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);return {buffer:bytes.buffer,sha256:v.sha256};},
    textgrid:file=>call('textgrid',{id:file.id}),
    spectrogram:(file,view)=>new Promise((resolve,reject)=>{const id=crypto.randomUUID();const timer=setTimeout(()=>{pending.delete(id);reject(Error('preview_timeout'));},35000);pending.set(id,{resolve,reject,timer});bridge.preview(id,JSON.stringify({id:file.id,...view}));}),dispose(){}};
  active={...browser,kind:'desktop'};
}
export function platform():HostCapabilities{return active;}
