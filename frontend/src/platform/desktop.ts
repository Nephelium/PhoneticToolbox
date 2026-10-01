import { browser } from './browser.ts';
import {serialRequests} from './serial-requests.ts';
import {m07Port} from './m07.ts';
import {m06Port} from './m06.ts';
import {m08Port} from './m08.ts';
import {m14Port} from './m14.ts';
import {m11Port} from './m11.ts';
import {desktopM05} from './m05.ts';
import type {Grant} from '../modules/mfa/port.ts';
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
  const task=serialRequests(<T>(body:unknown):Promise<T>=>new Promise((resolve,reject)=>{const id=crypto.randomUUID(),timer=setTimeout(()=>{taskPending.delete(id);reject(Error('任务操作超时，刷新批次可核对实际状态。'));},(body as {op?:string})?.op==='m11_component'?930000:300000);taskPending.set(id,{resolve,reject,timer});bridge.task(id,JSON.stringify(body));}));
  const tasks:ResearchTasks={parent:file=>task({op:'parent',id:file.id}),submit:(operation,inputs,config,layer,key)=>task({op:'submit',operation,inputs:inputs.map(item=>Object.fromEntries(Object.entries(item).filter(([,v])=>v!=null).map(([role,file])=>[role,role==='parent_result'?file:(file as {id:string}).id]))),config,layer,idempotency_key:key}),
    reconstruct:(file,config,key)=>task({op:'reconstruct',id:file.id,config,key}),reconstructions:()=>task({op:'reconstructions'}),cancelJob:id=>task({op:'cancel_job',id}),saveJob:(id,directory)=>task({op:'save_job',id,directory}),
    egg:async(file,config,key)=>{const {exportFontSnapshot}=await import('../state/fonts.ts');return task({op:'egg',id:file.id,config:{...config,font:config.font??exportFontSnapshot()},key});},eggJobs:()=>task({op:'egg_jobs'}),eggFonts:font=>task({op:'egg_fonts',font}),
    lpc:(file,textgrid,config,key)=>task({op:'lpc',id:file.id,textgrid:textgrid?.id,config,key}),lpcJobs:()=>task({op:'lpc_jobs'}),lpcFonts:font=>task({op:'lpc_fonts',font}),
    async result(job,id,sha){const value=await task<{base64:string;sha256:string}>({op:'result',job,id});if(value.sha256!==sha)throw Error('结果来源已变化。');const text=atob(value.base64);return Uint8Array.from(text,c=>c.charCodeAt(0)).buffer;},
    list:()=>task({op:'list'}),get:id=>task({op:'get',id}),cancel:id=>task({op:'cancel',id}),job:id=>task({op:'job',id}),
    retry:(id,key)=>task({op:'retry',id,key}),save:(id,directory,besideSources=false)=>task({op:'save',id,directory,beside_sources:besideSources})};
  desktopFiles={kind:'desktop',tasks:hello.tasks?tasks:undefined,choose:(purpose:DirectoryGrant['purpose'])=>call('choose',{purpose}),
    annotation:hello.tasks?{async audio(file){const value=await task<{base64:string;sha256:string;sourceDuration:number;previewNote:string}>({op:'annotation_audio',id:file.id});const text=atob(value.base64),bytes=new Uint8Array(text.length);for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);return {buffer:bytes.buffer,sha256:value.sha256,sourceDuration:value.sourceDuration,previewNote:value.previewNote};},scan:directory=>task({op:'annotation_scan',directory}),lip:file=>task({op:'annotation_lip',id:file.id}),target:(file,role,suffix)=>task({op:'annotation_target',id:file.id,role,suffix}),save:body=>task({op:'annotation_save',...body})}:undefined,
    parameters:file=>task({op:'parameters',id:file.id}),capture:()=>call('capture'),
    eggPreview:hello.tasks?{open:file=>task({op:'egg_preview_open',id:file.id}),
      update:(_file,session,config)=>task({op:'egg_preview_update',session,config}),
      close:(_file,session)=>task({op:'egg_preview_close',session})}:undefined,
    convertLip:hello.tasks?file=>task({op:'convert_lip',id:file.id}):undefined,
    list:(id='',recursive=false)=>recursive?task({op:'research_scan',id}):call('list',{id}),
    async previewAudio(file){const v=await task<{base64:string;sha256:string;sourceDuration:number;previewNote:string}>({op:'research_audio',id:file.id});const text=atob(v.base64),bytes=new Uint8Array(text.length);for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);return {buffer:bytes.buffer,sha256:v.sha256,sourceDuration:v.sourceDuration,previewNote:v.previewNote};},
    async read(file){const v=await call<{base64:string;sha256:string}>('read',{id:file.id});const text=atob(v.base64),bytes=new Uint8Array(text.length);for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);return {buffer:bytes.buffer,sha256:v.sha256};},
    textgrid:file=>call('textgrid',{id:file.id}),
    spectrogram:(file,view)=>new Promise((resolve,reject)=>{const id=crypto.randomUUID();const timer=setTimeout(()=>{pending.delete(id);reject(Error('preview_timeout'));},35000);pending.set(id,{resolve,reject,timer});bridge.preview(id,JSON.stringify({id:file.id,...view}));}),dispose(){}};
  if(hello.tasks){let outputDirectory:string|undefined;
    desktopFiles.m07=m07Port({project:'00000000-0000-4000-8000-000000000001',source:file=>task({op:'m07_source',id:file.id,sha256:file.sha256}),create:body=>task({op:'m07_create',body}),job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!,retry:tasks.retry,list:()=>task({op:'m07_list'}),save:(job,directory)=>task({op:'m07_save',job,directory})});
    desktopFiles.m06=m06Port({parameters:text=>task({op:'m06_parameters',text}),project:'00000000-0000-4000-8000-000000000001',source:file=>task({op:'m06_source',id:file.id,sha256:file.sha256}),create:body=>task({op:'m06_create',body}),job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!,async save(job){const grant=await call<DirectoryGrant|null>('choose',{purpose:'output'});if(!grant)return false;await task({op:'save_job',id:job,directory:grant.id});return true;}});
    desktopFiles.m05=await desktopM05(task,()=>call<DirectoryGrant|null>('choose',{purpose:'output'}));
      desktopFiles.m05.prepareCapture=async()=>{await call('m05_media');};
      desktopFiles.m05.captureError=async message=>{const status=await call<{decision:string}>('m05_media_status');return status.decision==='denied_user'?'未允许本次摄像头和麦克风采集。录制未开始，可点击开始后选择允许。':status.decision==='denied_gate'?'本次媒体权限请求已失效，请重新点击开始采集。':message;};
    desktopFiles.m11=m11Port({local:true,log:id=>task({op:'m11_log',id}),project:'00000000-0000-4000-8000-000000000001',
      catalog:()=>task({op:'m11_catalog'}),history:()=>task({op:'m11_jobs'}),events:id=>task({op:'m11_events',id}),
      async upload(file,role){const bytes=new Uint8Array(await file.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=32768)binary+=String.fromCharCode(...bytes.subarray(i,i+32768));return task({op:'m11_import',name:file.name,role,base64:btoa(binary)});},
      create:body=>task({op:'m11_create',body}),job:tasks.job,cancel:tasks.cancelJob!,read:tasks.result!,
      pick:purpose=>call<Grant|null>('m11_pick',{purpose}),corpus:grant=>task({op:'m11_corpus',id:grant.id}),component:body=>task({op:'m11_component',body}),
      chooseOutput:()=>call<Grant|null>('choose',{purpose:'output'}),save:(job,grant)=>task({op:'m11_save',job:job.id,directory:grant.id})});
    desktopFiles.m14=m14Port({project:'00000000-0000-4000-8000-000000000001',
      async upload(file){const bytes=new Uint8Array(await file.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=32768)binary+=String.fromCharCode(...bytes.subarray(i,i+32768));return task({op:'m14_import',name:file.name,base64:btoa(binary)});},
      create:body=>task({op:'m14_create',body}),job:tasks.job,cancel:tasks.cancelJob!,read:tasks.result!,
      async save(result){const grant=await call<DirectoryGrant|null>('choose',{purpose:'output'});if(!grant)return false;await task({op:'m14_save',job:result.job,directory:grant.id});return true;}});
    desktopFiles.m08=m08Port({project:'00000000-0000-4000-8000-000000000001',
      source:file=>task({op:'m08_source',id:file.id,sha256:file.sha256}),
      request:(action,body)=>task({op:'m08_call',action,body}),job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!,
      async export(result){if(!outputDirectory){const grant=await call<DirectoryGrant|null>('choose',{purpose:'output'});if(!grant)throw Error('未选择输出目录，受管结果仍保留，可重试导出。');outputDirectory=grant.id;}
        return (await task<{name:string}>({op:'m08_export',job:result.job_id,id:result.id,directory:outputDirectory})).name;}});
  }
  active={...browser,kind:'desktop'};
}
export function platform():HostCapabilities{return active;}
