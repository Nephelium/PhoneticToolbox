// Updates are an explicit native capability. No browser fetch or network fallback.
export type UpdateSource='auto'|'server'|'github';
export type ReleaseChannel='preview'|'stable';
export type PackageKind='portable'|'installer';
export interface UpdatePreferences{source:UpdateSource;channel:ReleaseChannel;autoCheck:boolean;currentVersion:string;packageKind:PackageKind;checkIntervalHours:number;applyAvailable:boolean}
export interface UpdatePackage{kind:PackageKind;name:string;size:number;sha256:string;source:'server'|'github'}
export interface UpdateRelease{id:string;version:string;source:'server'|'github';notes:string;publishedAt:string;packages:UpdatePackage[]}
export interface SourceResult{status:'checked'|'no-releases'|'error';count?:number;version?:string|null;code?:string;message?:string}
export interface UpdateCheck{status:'available'|'up-to-date'|'incomplete'|'deferred';reason?:'disabled'|'interval';currentVersion:string;channel?:ReleaseChannel;checkedAt?:number;region?:{country:string|null;source:'network'|'system'|'unknown'|'manual';label:string;preferredSource:'server'|'github'};preferredSource?:'server'|'github';sources:Partial<Record<'server'|'github',SourceResult>>;candidate:UpdateRelease|null;shouldPrompt:boolean;packageKind?:PackageKind;packageUnavailable?:boolean}
export interface DownloadProgress{phase:'downloading'|'fallback'|'verified';source:'server'|'github';received:number;total:number;message?:string}
export interface VerifiedDownload{downloadId:string;name:string;size:number;sha256:string;source:'server'|'github';kind:PackageKind;verified:true;applyAvailable:boolean}
interface NativeSignal{connect(callback:(id:string,payload:string)=>void):void;disconnect?(callback:(id:string,payload:string)=>void):void}
interface CloseSignal{connect(callback:(token:string)=>void):void;disconnect?(callback:(token:string)=>void):void}
export interface UpdatesChannel{request(id:string,payload:string):void;cancel(id:string):void;ready:NativeSignal;progress:NativeSignal;prepareClose?:CloseSignal;closeCancelled?:CloseSignal;closeReply?(token:string,allowed:boolean,message:string):void}

export class UpdatesError extends Error{
  readonly code:string;
  constructor(code:string,message:string){super(message);this.code=code;this.name='UpdatesError';}
}
type Pending={resolve:(value:unknown)=>void;reject:(reason:unknown)=>void;progress?:(value:DownloadProgress)=>void;timer:ReturnType<typeof setTimeout>;cleanup:()=>void};
let channel:UpdatesChannel|undefined;
const pending=new Map<string,Pending>();
let counter=0;
const listeners=new Set<()=>void>();
const closeListeners=new Set<(token:string)=>void>(),cancelListeners=new Set<(token:string)=>void>();
function onPrepareClose(token:string){for(const listener of closeListeners)listener(token);}
function onCloseCancelled(token:string){for(const listener of cancelListeners)listener(token);}
export function onUpdatePrepareClose(callback:(token:string)=>void){closeListeners.add(callback);return()=>closeListeners.delete(callback);}
export function onUpdateCloseCancelled(callback:(token:string)=>void){cancelListeners.add(callback);return()=>cancelListeners.delete(callback);}
export function replyUpdateClose(token:string,allowed:boolean,message=''){channel?.closeReply?.(token,allowed,message);}
function onReady(id:string,payload:string){
  const request=pending.get(id);if(!request)return;
  clearTimeout(request.timer);request.cleanup();pending.delete(id);
  try{const response=JSON.parse(payload);if(response?.ok===true)request.resolve(response.value);else request.reject(new UpdatesError(response?.error?.code||'UPDATE_FAILED',response?.error?.message||'更新操作失败。'));}
  catch{request.reject(new UpdatesError('RESPONSE_INVALID','更新响应无法解析。'));}
}
function onProgress(id:string,payload:string){
  try{const value=JSON.parse(payload);if(['downloading','fallback','verified'].includes(value?.phase)&&Number.isFinite(value.received)&&Number.isFinite(value.total)&&value.total>0&&value.received>=0&&value.received<=value.total)pending.get(id)?.progress?.(value);}
  catch{/* A malformed progress packet never completes or changes the verified download. */}
}
export function installUpdatesChannel(next:UpdatesChannel|undefined){
  if(channel===next)return;
  channel?.ready.disconnect?.(onReady);channel?.progress.disconnect?.(onProgress);
  channel?.prepareClose?.disconnect?.(onPrepareClose);channel?.closeCancelled?.disconnect?.(onCloseCancelled);
  for(const [id,request]of pending){channel?.cancel(id);clearTimeout(request.timer);request.cleanup();request.reject(new UpdatesError('DISCONNECTED','更新服务已断开。'));}
  pending.clear();channel=next;channel?.ready.connect(onReady);channel?.progress.connect(onProgress);
  channel?.prepareClose?.connect(onPrepareClose);channel?.closeCancelled?.connect(onCloseCancelled);
  for(const listener of listeners)listener();
}
export function updatesAvailable(){return !!channel;}
export function onUpdatesAvailability(callback:()=>void){listeners.add(callback);return()=>listeners.delete(callback);}
function invoke<T>(operation:string,args:Record<string,unknown>={},options:{signal?:AbortSignal;progress?:(value:DownloadProgress)=>void;timeout?:number}={}):Promise<T>{
  const current=channel;if(!current)return Promise.reject(new UpdatesError('UNAVAILABLE','在线更新仅在桌面版提供。'));
  if(options.signal?.aborted)return Promise.reject(new UpdatesError('CANCELLED','更新操作已取消。'));
  const id=`update_${Date.now().toString(36)}_${++counter}`;
  return new Promise<T>((resolve,reject)=>{
    const stop=(code:string,message:string)=>{const request=pending.get(id);if(!request)return;pending.delete(id);clearTimeout(request.timer);request.cleanup();current.cancel(id);reject(new UpdatesError(code,message));};
    const abort=()=>stop('CANCELLED','更新操作已取消。');
    const timer=setTimeout(()=>stop('TIMEOUT','更新操作超时，请重试。'),options.timeout??90000);
    const cleanup=()=>options.signal?.removeEventListener('abort',abort);
    pending.set(id,{resolve:resolve as(value:unknown)=>void,reject,progress:options.progress,timer,cleanup});
    options.signal?.addEventListener('abort',abort,{once:true});
    try{current.request(id,JSON.stringify({operation,args}));}catch{stop('REQUEST_FAILED','无法向更新服务提交请求。');}
  });
}
export const updater={
  preferences:()=>invoke<UpdatePreferences>('preferences'),
  configure:(value:Partial<Pick<UpdatePreferences,'source'|'channel'|'autoCheck'>>)=>invoke<UpdatePreferences>('configure',value),
  check:(value:{manual?:boolean;source?:UpdateSource;channel?:ReleaseChannel}={},signal?:AbortSignal)=>invoke<UpdateCheck>('check',value,{signal}),
  acknowledge:(releaseId:string)=>invoke<{acknowledged:true}>('acknowledge',{releaseId}),
  download:(releaseId:string,packageKind:PackageKind,confirmed:boolean,options:{signal?:AbortSignal;progress?:(value:DownloadProgress)=>void}={})=>invoke<VerifiedDownload>('download',{releaseId,packageKind,confirmed},{...options,timeout:31*60*1000}),
  apply:(downloadId:string,confirmed:boolean)=>invoke<{started:boolean;message?:string}>('apply',{downloadId,confirmed},{timeout:6*60*1000}),
};
export function sourceLabel(source:'server'|'github'){return source==='server'?'国内服务器':'GitHub';}
export function checkMessage(result:UpdateCheck){
  if(result.status==='available')return`发现新版本 ${result.candidate?.version??''}`;
  if(result.status==='up-to-date')return'当前已是最新版本';
  if(result.status==='deferred')return result.reason==='disabled'?'启动时自动检查已关闭':'最近已自动检查，稍后再检查';
  return result.packageUnavailable?'发现更新，但尚无匹配的更新包':'检查未完整，请查看各来源状态';
}
