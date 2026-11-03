import {uiMessage,uiError,responseError,localized} from '../modules/module-localization.ts';
import {tr,bindNativeLocale,isAppLocale} from '../i18n/index.ts';
import {installPapersChannel,type PapersChannel} from './papers.ts';
import {ref} from 'vue';
import {installFeedbackChannel,type FeedbackChannel} from './feedback.ts';
import { browser } from './browser.ts';
import {installClipboardWriter} from './clipboard.ts';
import {serialRequests} from './serial-requests.ts';
import {installManualPackStore,type InstalledManualPack,type ManualPackSpec} from './manual-packs.ts';
import {coalescedRequests,desktopReadKey} from './coalesced-requests.ts';
import {m07Port} from './m07.ts';
import {m06Port} from './m06.ts';
import {m08Port} from './m08.ts';
import {m14Port} from './m14.ts';
import {m11Port} from './m11.ts';
import {desktopM05} from './m05.ts';
import {installStoragePort} from './storage.ts';
import {installIconPort} from './icons.ts';
import {installColorPicker} from './color-picker.ts';
import {installUpdatesChannel,type UpdatesChannel} from './updates.ts';
import {recordingPort,installRecordingPort} from './recording.ts';
import {recordingRequests,type RecordingChannel} from './recording-requests.ts';
import type {Grant} from '../modules/mfa/port.ts';
import type { HostCapabilities } from './types.ts';
import type { DirectoryGrant,ResearchFiles,Spectrogram,ResearchTasks } from './research.ts';
type Bridge={invoke:(body:string,callback:(value:string)=>void)=>void;preview:(id:string,body:string)=>void;previewReady:{connect:(fn:(id:string,body:string)=>void)=>void};task:(id:string,body:string)=>void;taskReady:{connect:(fn:(id:string,body:string)=>void)=>void}};
declare global {interface Window {qt?:{webChannelTransport:unknown};QWebChannel?:new(transport:unknown,callback:(channel:{objects:{files:Bridge;updates?:UpdatesChannel;papers?:PapersChannel;feedback?:FeedbackChannel}})=>void)=>unknown}}
let active:HostCapabilities=browser;
export let desktopFiles:ResearchFiles|undefined;
export let desktopSession='';
export const integratedTitlebar=ref(false),desktopMaximized=ref(false);
let windowAction:((action:string)=>void)|undefined;
export function desktopWindowAction(action:'move'|'minimize'|'maximize'|'close'){windowAction?.(action);}
export let desktopFontFamilies:(()=>Promise<string[]>)|undefined;
export let vocalRequest:((op:string,body?:unknown)=>Promise<unknown>)|undefined;
export let manualPacksRequest:((action:string,payload:Record<string,unknown>)=>Promise<unknown>)|undefined;
export async function initializePlatform(){
  if(!window.qt?.webChannelTransport||!window.QWebChannel)return;
  const bridge=await new Promise<Bridge>((resolve,reject)=>{const timer=setTimeout(()=>reject(uiError(uiMessage('platform.the_desktop_connection_timed_out'))),15000);new window.QWebChannel!(window.qt!.webChannelTransport,channel=>{clearTimeout(timer);installUpdatesChannel(channel.objects.updates);installPapersChannel(channel.objects.papers);installFeedbackChannel(channel.objects.feedback);resolve(channel.objects.files);});});
  const call=<T>(op:string,extra:Record<string,string>={}):Promise<T>=>new Promise((resolve,reject)=>bridge.invoke(JSON.stringify({op,...extra}),text=>{try{const result=JSON.parse(text);if(!result.ok)throw responseError(result.error,result.display);resolve(result.value);}catch(e){reject(e);}}));
  const hello=await call<{kind:string;session:string;api_version:string;tasks?:boolean;integrated_titlebar?:boolean;window_maximized?:boolean;ui_locale?:unknown}>('hello');
  if(hello.kind!=='desktop'||hello.api_version!=='1.1.0'||!hello.session)throw uiError(uiMessage('platform.the_desktop_interface_version_does_not_match'));
  desktopSession=hello.session;
  if(isAppLocale(hello.ui_locale))bindNativeLocale(async code=>(await call<{saved:boolean}>('locale',{purpose:code})).saved,hello.ui_locale);
  const windowBridge=bridge as Bridge & {windowAction?:(action:string)=>void;windowStateChanged?:{connect:(callback:(maximized:boolean)=>void)=>void}};
  integratedTitlebar.value=!!hello.integrated_titlebar&&!!windowBridge.windowAction;
  desktopMaximized.value=!!hello.window_maximized;
  windowAction=windowBridge.windowAction;
  windowBridge.windowStateChanged?.connect(maximized=>{desktopMaximized.value=maximized;});
  const colorBridge=bridge as Bridge & {pickColor?:(initial:string,callback:(value:string)=>void)=>void};
  if(colorBridge.pickColor)installColorPicker(initial=>new Promise((resolve,reject)=>colorBridge.pickColor!(initial,raw=>{try{const result=JSON.parse(raw);if(!result.ok)throw responseError(result.error,result.display);resolve(result.color??null);}catch(error){reject(error);}})));
  const iconBridge=bridge as Bridge & {applicationIcon?:(body:string,callback:(value:string)=>void)=>void};
  if(iconBridge.applicationIcon)installIconPort(request=>new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(uiError(uiMessage('platform.the_icon_settings_response_timed_out_reopen_settings_to'))),15000);
    iconBridge.applicationIcon!(JSON.stringify(request),raw=>{clearTimeout(timer);try{const result=JSON.parse(raw);if(!result.ok)throw responseError(result.error,result.display);resolve(result.value);}catch(error){reject(error);}});
  }));
  const surfaceBridge=bridge as Bridge & {setSurfaceColor?:(color:string)=>void;setCaptionColors?:(background:string,foreground:string)=>void};
  if(surfaceBridge.setSurfaceColor){
    const syncSurface=()=>{
      const styles=getComputedStyle(document.documentElement),color=styles.getPropertyValue('--app').trim();
      if(/^#[0-9a-f]{6}$/i.test(color))surfaceBridge.setSurfaceColor!(color);
      const background=styles.getPropertyValue('--sidebar').trim(),foreground=styles.getPropertyValue('--text').trim();
      if(/^#[0-9a-f]{6}$/i.test(background)&&/^#[0-9a-f]{6}$/i.test(foreground))surfaceBridge.setCaptionColors?.(background,foreground);
    };
    new MutationObserver(syncSurface).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme','data-palette','data-custom-palette']});syncSurface();
  }
  const clipboardBridge=bridge as Bridge & {writeClipboard?:(text:string,callback:(result:string)=>void)=>void};
  if(clipboardBridge.writeClipboard){
    installClipboardWriter(text=>new Promise<void>((resolve,reject)=>{
      const timer=setTimeout(()=>reject(uiError(uiMessage('platform.copy_timed_out_try_again'))),5000);
      clipboardBridge.writeClipboard!(text,raw=>{clearTimeout(timer);try{const result=JSON.parse(raw);if(!result.ok)throw responseError(result.error,result.display);resolve();}catch(error){reject(error);}});
    }));
  }
  desktopFontFamilies=()=>call<string[]>('fonts');
  const recordingBridge=bridge as Bridge & Partial<RecordingChannel>;
  if(recordingBridge.recording&&recordingBridge.recordingReady){
    installRecordingPort(recordingPort(recordingRequests(recordingBridge as RecordingChannel),purpose=>call('m16_choose',{purpose})));
  }
  const vocalBridge=bridge as Bridge & {vocal?:(id:string,body:string)=>void;vocalReady?:{connect:(callback:(id:string,raw:string)=>void)=>void}};
  if(vocalBridge.vocal&&vocalBridge.vocalReady){
    const requests=new Map<string,{resolve:(value:unknown)=>void;reject:(error:Error)=>void;timer:ReturnType<typeof setTimeout>}>();
    vocalBridge.vocalReady.connect((id,raw)=>{const request=requests.get(id);if(!request)return;requests.delete(id);clearTimeout(request.timer);try{const result=JSON.parse(raw);if(!result.ok)throw responseError(result.error,result.display);request.resolve(result.value);}catch(e){request.reject(e instanceof Error?e:uiError(uiMessage('platform.vocal_tract_request_failed')));}});
    vocalRequest=(op,body={})=>new Promise((resolve,reject)=>{const id=crypto.randomUUID(),timer=setTimeout(()=>{requests.delete(id);reject(uiError(uiMessage('platform.the_vocal_tract_request_timed_out_reopen_the_module')));},160000);requests.set(id,{resolve,reject,timer});vocalBridge.vocal!(id,JSON.stringify({op,body}));});
  }
  const pending=new Map<string,{resolve:(data:Spectrogram)=>void;reject:(error:Error)=>void;timer:ReturnType<typeof setTimeout>}>();
  bridge.previewReady.connect((id,text)=>{const request=pending.get(id);if(!request)return;pending.delete(id);clearTimeout(request.timer);try{const data=JSON.parse(text);if(!data.ok)throw responseError(data.error,data.display);request.resolve(data.value);}catch(e){request.reject(e instanceof Error?e:uiError(uiMessage('platform.the_spectrogram_could_not_be_read')));}});
  const taskPending=new Map<string,{resolve:(value:any)=>void;reject:(error:Error)=>void;timer?:ReturnType<typeof setTimeout>}>();
  bridge.taskReady?.connect((id,text)=>{const request=taskPending.get(id);if(!request)return;taskPending.delete(id);if(request.timer!==undefined)clearTimeout(request.timer);try{const data=JSON.parse(text);if(!data.ok)throw responseError(data.error,data.display);request.resolve(data.value);}catch(e){request.reject(e instanceof Error?e:uiError(uiMessage('platform.failed_to_read_tasks')));}});
  const nativeTask=<T>(body:unknown):Promise<T>=>new Promise((resolve,reject)=>{
    const id=crypto.randomUUID(),op=(body as {op?:string})?.op;
    // Cleanup must receive the native completion before releasing the task queue.
    const timer=op==='local_storage_cleanup'?undefined:setTimeout(()=>{taskPending.delete(id);reject(uiError(uiMessage('platform.the_task_operation_timed_out_refresh_the_batch_to')));},op==='m11_component'||op==='m09_vocoder_download'?7230000:op==='manual_packs'?650000:op==='m05_save_finish'?650000:op==='parameters'?350000:300000);
    taskPending.set(id,{resolve,reject,timer});bridge.task(id,JSON.stringify(body));
  });
  const task=coalescedRequests(serialRequests(nativeTask),desktopReadKey);
  // Manual transfers have their own queue and native semaphore. Installing a
  // language pack must not block scientific tasks or reads of an installed pack.
  const manualReads=serialRequests(nativeTask);
  manualPacksRequest=(action,payload)=>['install','importFinish'].includes(action)?nativeTask({op:'manual_packs',action,...payload}):manualReads({op:'manual_packs',action,...payload});
  installManualPackStore({
    list:()=>manualPacksRequest!('list',{}) as Promise<InstalledManualPack[]>,
    install:spec=>manualPacksRequest!('install',{...spec}) as Promise<InstalledManualPack>,
    readJson:(code,path)=>manualPacksRequest!('readJson',{locale:code,path}),
    readAsset:async(code,path)=>{
      const value=await manualPacksRequest!('readAsset',{locale:code,path}) as {mime:string;base64:string}|null;
      return value?new Blob([Uint8Array.from(atob(value.base64),char=>char.charCodeAt(0))],{type:value.mime}):null;
    },
    importArchive:async(spec:Omit<ManualPackSpec,'url'>,file:Blob)=>{
      const transfer=await manualPacksRequest!('importBegin',{spec,bytes:file.size}) as {id:string};
      try{
        for(let offset=0;offset<file.size;offset+=1048576){
          const bytes=new Uint8Array(await file.slice(offset,offset+1048576).arrayBuffer());let text='';
          for(let start=0;start<bytes.length;start+=32768)text+=String.fromCharCode(...bytes.subarray(start,start+32768));
          await manualPacksRequest!('importChunk',{id:transfer.id,offset,base64:btoa(text)});
        }
        return await manualPacksRequest!('importFinish',{id:transfer.id}) as InstalledManualPack;
      }catch(error){await manualPacksRequest!('importCancel',{id:transfer.id}).catch(()=>{});throw error;}
    },
  });
  const preview=coalescedRequests(<T>(body:unknown):Promise<T>=>new Promise<Spectrogram>((resolve,reject)=>{
    const id=crypto.randomUUID();const timer=setTimeout(()=>{pending.delete(id);reject(Error('preview_timeout'));},35000);
    pending.set(id,{resolve,reject,timer});bridge.preview(id,JSON.stringify(body));
  }).then(value=>value as T),body=>JSON.stringify(body));
  const tasks:ResearchTasks={parent:file=>task({op:'parent',id:file.id}),submit:(operation,inputs,config,layer,key)=>task({op:'submit',operation,inputs:inputs.map(item=>Object.fromEntries(Object.entries(item).filter(([,v])=>v!=null).map(([role,file])=>[role,role==='parent_result'?file:(file as {id:string}).id]))),config,layer,idempotency_key:key}),
    vocoderStatus:()=>task({op:'m09_vocoder_status'}),downloadVocoder:()=>nativeTask({op:'m09_vocoder_download'}),
    chooseInferencePython:()=>call('m09_pick_python'),checkInferencePython:grant=>task({op:'m09_check_python',grant}),
    reconstruct:(file,config,key)=>task({op:'reconstruct',id:file.id,config,key}),reconstructionPreview:(file,config)=>task({op:'reconstruction_preview',id:file.id,config}),reconstructions:()=>task({op:'reconstructions'}),cancelJob:id=>task({op:'cancel_job',id}),saveJob:(id,directory)=>task({op:'save_job',id,directory}),
    egg:async(file,config,key)=>{const {exportFontSnapshot}=await import('../state/fonts.ts');return task({op:'egg',id:file.id,config:{...config,font:config.font??exportFontSnapshot()},key});},eggJobs:()=>task({op:'egg_jobs'}),eggFonts:font=>task({op:'egg_fonts',font}),
    lpc:(file,textgrid,config,key)=>task({op:'lpc',id:file.id,textgrid:textgrid?.id,config,key}),lpcJobs:()=>task({op:'lpc_jobs'}),lpcFonts:font=>task({op:'lpc_fonts',font}),
    async result(job,id,sha){const value=await task<{base64:string;sha256:string}>({op:'result',job,id});if(value.sha256!==sha)throw uiError(uiMessage('platform.the_result_source_has_changed'));const text=atob(value.base64);return Uint8Array.from(text,c=>c.charCodeAt(0)).buffer;},
    list:()=>task({op:'list'}),get:id=>task({op:'get',id}),cancel:id=>task({op:'cancel',id}),job:id=>task({op:'job',id}),
    retry:(id,key)=>task({op:'retry',id,key}),save:(id,directory,besideSources=false)=>task({op:'save',id,directory,beside_sources:besideSources})};
  desktopFiles={kind:'desktop',tasks:hello.tasks?tasks:undefined,choose:(purpose:DirectoryGrant['purpose'])=>call('choose',{purpose}),
    annotation:hello.tasks?{async audio(file){const value=await task<{base64:string;sha256:string;sourceDuration:number;previewNote:string}>({op:'annotation_audio',id:file.id});const text=atob(value.base64),bytes=new Uint8Array(text.length);for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);return {buffer:bytes.buffer,sha256:value.sha256,sourceDuration:value.sourceDuration,previewNote:value.previewNote};},scan:directory=>task({op:'annotation_scan',directory}),lip:file=>task({op:'annotation_lip',id:file.id}),target:(file,role,suffix)=>task({op:'annotation_target',id:file.id,role,suffix}),save:body=>task({op:'annotation_save',...body})}:undefined,
    parameters:(file,view)=>task({op:'parameters',id:file.id,view}),capture:()=>call('capture'),
    eggPreview:hello.tasks?{open:file=>task({op:'egg_preview_open',id:file.id}),
      update:(_file,session,config)=>task({op:'egg_preview_update',session,config}),
      close:(_file,session)=>task({op:'egg_preview_close',session})}:undefined,
    convertLip:hello.tasks?file=>task({op:'convert_lip',id:file.id}):undefined,
    list:(id='',recursive=false)=>!id?Promise.resolve([]):task({op:recursive?'research_scan':'file_list',id}),
    async previewAudio(file){const v=await task<{base64:string;sha256:string;sourceDuration:number;previewNote:string}>({op:'research_audio',id:file.id});const text=atob(v.base64),bytes=new Uint8Array(text.length);for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);return {buffer:bytes.buffer,sha256:v.sha256,sourceDuration:v.sourceDuration,previewNote:v.previewNote};},
    async read(file,signal){signal?.throwIfAborted();const v=await task<{base64:string;sha256:string}>({op:'file_read',id:file.id});signal?.throwIfAborted();const text=atob(v.base64),bytes=new Uint8Array(text.length);for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);return {buffer:bytes.buffer,sha256:v.sha256};},
    textgrid:file=>task({op:'file_textgrid',id:file.id}),
    spectrogram:(file,view)=>preview({id:file.id,...view}),dispose(){}};
  if(hello.tasks){let outputDirectory:string|undefined;
    installStoragePort({status:()=>task({op:'local_storage_status'}),configure:policy=>task({op:'local_storage_policy',policy}),clean:()=>task({op:'local_storage_cleanup'}),clearAll:()=>call('clear_all_caches',{purpose:'confirmed'}),cacheLocation:()=>call('cache_location'),openCacheDirectory:(legacy=false)=>call('open_cache_directory',{purpose:legacy?'legacy':'current'}),cacheCandidates:()=>task({op:'cache_candidates'}),openCacheCandidate:id=>call('open_cache_candidate',{id}),cleanCacheCandidate:id=>task({op:'clean_cache_candidate',id,confirmed:true}),cleanGraphicsCandidate:id=>call('clear_graphics_cache',{id,purpose:'confirmed'})});
    desktopFiles.m07=m07Port({f0:job=>task({op:'m07_f0',job}),project:'00000000-0000-4000-8000-000000000001',source:file=>task({op:'m07_source',id:file.id,sha256:file.sha256}),create:body=>task({op:'m07_create',body}),job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!,retry:tasks.retry,list:()=>task({op:'m07_list'}),save:(job,directory,name)=>task({op:'m07_save',job,directory,name})});
    desktopFiles.m06=m06Port({parameters:text=>task({op:'m06_parameters',text}),project:'00000000-0000-4000-8000-000000000001',source:file=>task({op:'m06_source',id:file.id,sha256:file.sha256}),create:body=>task({op:'m06_create',body}),job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!,async save(job){const grant=await call<DirectoryGrant|null>('choose',{purpose:'output'});if(!grant)return false;await task({op:'save_job',id:job,directory:grant.id});return true;}});
    desktopFiles.m05=await desktopM05(task,()=>call<DirectoryGrant|null>('choose',{purpose:'output'}));
      desktopFiles.m05.prepareCapture=async purpose=>{await call('m05_media',{purpose:purpose??'capture'});};
      desktopFiles.m05.captureError=async(message,purpose)=>{const status=await call<{decision:string}>('m05_media_status');return status.decision==='denied_user'?(purpose==='devices'?tr('platform.microphone_access_was_not_granted_click_authorize_and_refresh'):tr('platform.camera_and_microphone_capture_was_not_allowed_recording_has')):status.decision==='denied_gate'?(purpose==='devices'?tr('platform.this_audio_permission_request_has_expired_click_authorize_and'):tr('platform.this_media_permission_request_has_expired_click_start_capture')):message;};
    desktopFiles.m11=m11Port({local:true,log:id=>task({op:'m11_log',id}),project:'00000000-0000-4000-8000-000000000001',
      catalog:()=>task({op:'m11_catalog'}),history:()=>task({op:'m11_jobs'}),events:id=>task({op:'m11_events',id}),
      async upload(file,role){const bytes=new Uint8Array(await file.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=32768)binary+=String.fromCharCode(...bytes.subarray(i,i+32768));return task({op:'m11_import',name:file.name,role,base64:btoa(binary)});},
      create:body=>task({op:'m11_create',body}),job:tasks.job,cancel:tasks.cancelJob!,read:tasks.result!,
      pick:purpose=>call<Grant|null>('m11_pick',{purpose}),dictionaryFromGrant:grant=>task({op:'m11_dictionary',id:grant.id}),corpus:(grant,source='auto')=>task({op:'m11_corpus',id:grant.id,transcript_source:source}),component:body=>nativeTask({op:'m11_component',body}),
      chooseOutput:()=>call<Grant|null>('choose',{purpose:'output'}),save:(job,grant)=>task({op:'m11_save',job:job.id,directory:grant.id})});
    desktopFiles.m14=m14Port({project:'00000000-0000-4000-8000-000000000001',
      async upload(file){const bytes=new Uint8Array(await file.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=32768)binary+=String.fromCharCode(...bytes.subarray(i,i+32768));return task({op:'m14_import',name:file.name,base64:btoa(binary)});},
      create:body=>task({op:'m14_create',body}),job:tasks.job,cancel:tasks.cancelJob!,read:tasks.result!,
      async save(result){const grant=await call<DirectoryGrant|null>('choose',{purpose:'output'});if(!grant)return false;await task({op:'m14_save',job:result.job,directory:grant.id});return true;}});
    desktopFiles.m08=m08Port({project:'00000000-0000-4000-8000-000000000001',
      source:file=>task({op:'m08_source',id:file.id,sha256:file.sha256}),
      request:(action,body)=>task({op:'m08_call',action,body}),job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!,
      async exportMany(results){
        const grant=await call<DirectoryGrant|null>('choose',{purpose:'output'});
        const saved:{id:string;name:string}[]=[],failed:{id:string;error:string}[]=[];let directory:string|undefined;
        if(!grant)return {saved,failed,cancelled:true};
        for(const result of results){try{
          const value=await task<{name:string;directory?:string}>({op:'m08_export',job:result.job_id,id:result.id,directory:grant.id,direct:true,name:result.name});
          directory=value.directory??directory;
          saved.push({id:result.id,name:value.name});
        }catch(e){failed.push({id:result.id,error:e instanceof Error?e.message:String(e)});}}
        return {saved,failed,directory};
      },
      async export(result){if(!outputDirectory){const grant=await call<DirectoryGrant|null>('choose',{purpose:'output'});if(!grant)throw uiError(uiMessage('platform.no_output_folder_was_selected_managed_results_are_retained'));outputDirectory=grant.id;}
        return (await task<{name:string}>({op:'m08_export',job:result.job_id,id:result.id,directory:outputDirectory})).name;}});
  }
  active={...browser,kind:'desktop'};
}
export function platform():HostCapabilities{return active;}
