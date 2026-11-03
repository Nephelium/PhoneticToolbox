import {tr,messageText,type MessageDescriptor,type AppLocale} from '../i18n/index.ts';
import {ModuleDisplayError,uiMessage,responseError,uiErrorMessage} from '../modules/module-localization.ts';
// Updates are an explicit native capability. No browser fetch or network fallback.
export type UpdateSource='auto'|'server'|'github';
export type ReleaseChannel='preview'|'stable';
export type PackageKind='portable'|'installer';
export interface UpdatePreferences{source:UpdateSource;channel:ReleaseChannel;autoCheck:boolean;currentVersion:string;packageKind:PackageKind;checkIntervalHours:number;applyAvailable:boolean}
export interface UpdatePackage{kind:PackageKind;name:string;size:number;sha256:string;source:'server'|'github'}
export interface UpdateRelease{id:string;version:string;source:'server'|'github';notes:string;notesByLocale?:Partial<Record<AppLocale,string>>;publishedAt:string;packages:UpdatePackage[]}
/** Use the selected UI language; older manifests retain their original notes. */
export function localizedReleaseNotes(release:Pick<UpdateRelease,'notes'|'notesByLocale'>|null|undefined,language:AppLocale):string{
  if(!release)return '';
  const translated=release.notesByLocale?.[language];
  return typeof translated==='string'&&translated.trim()?translated:release.notes;
}
export interface SourceResult{status:'checked'|'no-releases'|'error';count?:number;version?:string|null;code?:string;message?:string;display?:MessageDescriptor}
export interface UpdateCheck{status:'available'|'up-to-date'|'incomplete'|'deferred';reason?:'disabled'|'interval';currentVersion:string;channel?:ReleaseChannel;checkedAt?:number;region?:{country:string|null;source:'network'|'system'|'unknown'|'manual';label:string;preferredSource:'server'|'github'};preferredSource?:'server'|'github';sources:Partial<Record<'server'|'github',SourceResult>>;candidate:UpdateRelease|null;shouldPrompt:boolean;packageKind?:PackageKind;packageUnavailable?:boolean}
export interface DownloadProgress{phase:'downloading'|'fallback'|'verified';source:'server'|'github';received:number;total:number;message?:string}
export interface VerifiedDownload{downloadId:string;name:string;size:number;sha256:string;source:'server'|'github';kind:PackageKind;verified:true;applyAvailable:boolean}
interface NativeSignal{connect(callback:(id:string,payload:string)=>void):void;disconnect?(callback:(id:string,payload:string)=>void):void}
interface CloseSignal{connect(callback:(token:string)=>void):void;disconnect?(callback:(token:string)=>void):void}
export interface UpdatesChannel{request(id:string,payload:string):void;cancel(id:string):void;ready:NativeSignal;progress:NativeSignal;prepareClose?:CloseSignal;closeCancelled?:CloseSignal;closeReply?(token:string,allowed:boolean,message:string):void}

export class UpdatesError extends ModuleDisplayError{
  readonly code:string;
  constructor(code:string,message:string|MessageDescriptor){super(message);this.code=code;this.name='UpdatesError';}
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
  try{const response=JSON.parse(payload);if(response?.ok===true)request.resolve(response.value);else request.reject(new UpdatesError(response?.error?.code||'UPDATE_FAILED',uiErrorMessage(responseError(response?.error?.message||'platform.the_update_operation_failed',response?.error?.display))));}
  catch{request.reject(new UpdatesError('RESPONSE_INVALID',uiMessage('platform.the_update_response_could_not_be_parsed')));}
}
function onProgress(id:string,payload:string){
  try{const value=JSON.parse(payload);if(['downloading','fallback','verified'].includes(value?.phase)&&Number.isFinite(value.received)&&Number.isFinite(value.total)&&value.total>0&&value.received>=0&&value.received<=value.total)pending.get(id)?.progress?.(value);}
  catch{/* A malformed progress packet never completes or changes the verified download. */}
}
export function installUpdatesChannel(next:UpdatesChannel|undefined){
  if(channel===next)return;
  channel?.ready.disconnect?.(onReady);channel?.progress.disconnect?.(onProgress);
  channel?.prepareClose?.disconnect?.(onPrepareClose);channel?.closeCancelled?.disconnect?.(onCloseCancelled);
  for(const [id,request]of pending){channel?.cancel(id);clearTimeout(request.timer);request.cleanup();request.reject(new UpdatesError('DISCONNECTED',uiMessage('platform.the_update_service_disconnected')));}
  pending.clear();channel=next;channel?.ready.connect(onReady);channel?.progress.connect(onProgress);
  channel?.prepareClose?.connect(onPrepareClose);channel?.closeCancelled?.connect(onCloseCancelled);
  for(const listener of listeners)listener();
}
export function updatesAvailable(){return !!channel;}
export function onUpdatesAvailability(callback:()=>void){listeners.add(callback);return()=>listeners.delete(callback);}
function invoke<T>(operation:string,args:Record<string,unknown>={},options:{signal?:AbortSignal;progress?:(value:DownloadProgress)=>void;timeout?:number}={}):Promise<T>{
  const current=channel;if(!current)return Promise.reject(new UpdatesError('UNAVAILABLE',uiMessage('platform.online_updates_are_available_only_in_the_desktop_version')));
  if(options.signal?.aborted)return Promise.reject(new UpdatesError('CANCELLED',uiMessage('platform.the_update_operation_was_cancelled')));
  const id=`update_${Date.now().toString(36)}_${++counter}`;
  return new Promise<T>((resolve,reject)=>{
    const stop=(code:string,message:MessageDescriptor)=>{const request=pending.get(id);if(!request)return;pending.delete(id);clearTimeout(request.timer);request.cleanup();current.cancel(id);reject(new UpdatesError(code,message));};
    const abort=()=>stop('CANCELLED',uiMessage('platform.the_update_operation_was_cancelled'));
    const timer=setTimeout(()=>stop('TIMEOUT',uiMessage('platform.the_update_operation_timed_out_try_again')),options.timeout??90000);
    const cleanup=()=>options.signal?.removeEventListener('abort',abort);
    pending.set(id,{resolve:resolve as(value:unknown)=>void,reject,progress:options.progress,timer,cleanup});
    options.signal?.addEventListener('abort',abort,{once:true});
    try{current.request(id,JSON.stringify({operation,args}));}catch{stop('REQUEST_FAILED',uiMessage('platform.the_request_could_not_be_submitted_to_the_update'));}
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
export function sourceLabel(source:'server'|'github'){return source==='server'?tr('platform.china_server'):'GitHub';}
export function updateDisplayMessage(raw:string|undefined,display?:MessageDescriptor){return raw?messageText(uiErrorMessage(responseError(raw,display))):'';}
export function checkMessage(result:UpdateCheck){
  if(result.status==='available')return tr('platform.new_version_found_value',{v0:(result.candidate?.version??'')});
  if(result.status==='up-to-date')return tr('platform.you_are_using_the_latest_version');
  if(result.status==='deferred')return result.reason==='disabled'?tr('platform.automatic_checks_at_startup_are_disabled'):tr('platform.an_automatic_check_was_performed_recently_check_again_later');
  return result.packageUnavailable?tr('platform.an_update_was_found_but_no_matching_package_is'):tr('platform.the_check_was_incomplete_review_each_source_s_status');
}
