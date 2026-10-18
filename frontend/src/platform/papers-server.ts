import type {AssetFetch} from './local-asset-transport.ts';
import {createPapers,type PapersClient,type PaperProgress,type PaperStatus} from './papers.ts';
import {readWebSettings,writeWebSettings} from '../state/web-settings.ts';

const messages:Record<string,string>={
 paper_service_busy:'论文服务繁忙，请稍后重试。',paper_cancelled:'操作已取消，已保存的论文保留。',paper_timeout:'论文处理超时，请重试。',
 paper_runtime_unavailable:'PDF 阅读服务暂不可用，请稍后重试。',paper_state_conflict:'论文库已在另一窗口更改，请刷新后重试。',
 quota_exceeded:'空间不足，请清理文件或申请扩容后重试。',upload_size_limit:'请选择 100 MB 以内的 PDF。',
 workspace_expired:'工作区已到期，请返回重新开始。',account_changed:'账号已切换，请重新进入工作台。',authentication_required:'登录已过期，请重新登录。',
 paper_duplicate_name:'所选文件夹内有重复文件，请检查文件名。',paper_input_invalid:'请选择当前项目内的有效 PDF。',invalid_request:'论文操作参数无效，请刷新后重试。',
};
type Imported={asset_id:string;relative:string};
export function serverPapers(owner:string,project:string,csrf:()=>string,onInvalid:()=>void,lifetime:AbortSignal,assetFetch:AssetFetch=(path,init)=>fetch('/api/v1/'+path,init)):PapersClient{
 const settingsScope='server:'+owner+':'+project+':M18';let preferencesApplied=false;
 const listeners=new Set<(s:PaperStatus)=>void>();let queue:Promise<unknown>=Promise.resolve();
 let scheduleSettings:((s:PaperStatus)=>void)|undefined,automatic:AbortController|undefined;
 const emit=(s:PaperStatus)=>{for(const listener of listeners)listener(s);};
 async function request(path:string,method='GET',body?:BodyInit,signal?:AbortSignal,binary=false){
  const combined=signal?AbortSignal.any([lifetime,signal]):lifetime;
  let response:Response;
  for(let attempt=0;;attempt++){
  response=await assetFetch(path,{method,credentials:'same-origin',cache:'no-store',signal:combined,
   headers:{'X-PTB-Account':owner,...(method!=='GET'?{'X-CSRF-Token':csrf()}:{}),...(body?{'Content-Type':binary?'application/octet-stream':'application/json'}:{})},body});
  if(response.status!==429||!path.startsWith('papers/')||attempt>=3)break;
  await response.body?.cancel();await new Promise<void>((resolve,reject)=>{if(combined.aborted){reject(Error('操作已取消。'));return;}const abort=()=>{clearTimeout(timer);reject(Error('操作已取消。'));};const timer=setTimeout(()=>{combined.removeEventListener('abort',abort);resolve();},250*(attempt+1));combined.addEventListener('abort',abort,{once:true});});
  }
  if(!response.ok){const e=await response.json().catch(()=>({}));if(response.status===401||e.detail==='account_changed')onInvalid();throw Error(messages[e.detail]??'论文操作未完成，请检查连接与剩余空间后重试。');}return response;
 }
 function pick(directory=false):Promise<File[]>{return new Promise((resolve)=>{
  const input=document.createElement('input');input.type='file';input.accept='.pdf,application/pdf';input.multiple=true;if(directory)input.setAttribute('webkitdirectory','');
  input.addEventListener('change',()=>{resolve([...(input.files??[])].filter(f=>f.name.toLowerCase().endsWith('.pdf')));input.remove();},{once:true});
  input.addEventListener('cancel',()=>{resolve([]);input.remove();},{once:true});input.click();
 });}
 async function upload(files:File[],signal?:AbortSignal):Promise<Imported[]>{
  if(files.length>1000)throw Error('每次请选择不超过 1000 份 PDF。');
  for(const file of files)if(!file.size||file.size>100_000_000)throw Error(file.name+'：请选择 100 MB 以内的 PDF。');
  const imported:Imported[]=[];
  for(const file of files){
   let asset=await(await request('uploads','POST',JSON.stringify({project_id:project,name:file.name,expected_bytes:file.size,idempotency_key:crypto.randomUUID()}),signal)).json();
   for(let offset=0;offset<file.size;offset+=262144)await request('uploads/'+asset.id+'/blocks?offset='+offset,'PUT',file.slice(offset,offset+262144),signal,true);
   asset=await(await request('uploads/'+asset.id+'/finalize','POST','{}',signal)).json();
   const relative=file.webkitRelativePath?.split('/').slice(1).join('/')||file.name;imported.push({asset_id:asset.id,relative});
  }return imported;
 }
 async function execute<T>(operation:string,args:Record<string,unknown>={},signal?:AbortSignal,onProgress?:(p:PaperProgress)=>void):Promise<T>{
  if(signal?.aborted||lifetime.aborted)throw Error('操作已取消。');
  operation=operation.replace(/^__prepared__/, '');
  const response=await request('papers/'+project+'/request','POST',JSON.stringify({operation,args}),signal);
  if(!response.body)throw Error('论文响应不完整，请重试。');
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',value:T|undefined;
  try{while(true){const chunk=await reader.read();buffer+=decoder.decode(chunk.value,{stream:!chunk.done});
   let at:number;while((at=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,at);buffer=buffer.slice(at+1);if(!line.trim())continue;const item=JSON.parse(line);if(item.progress)onProgress?.(item.progress);else if(item.ok)value=item.value;else throw Error(messages[item.error]??item.error);}
   if(chunk.done)break;
  }}finally{await reader.cancel();reader.releaseLock();}
  if(value===undefined)throw Error('论文响应中断，请重试。');
  if(operation==='autoReceive'||operation==='interval')scheduleSettings?.(value as PaperStatus);
  if(operation==='status'&&!preferencesApplied){
   preferencesApplied=true;const saved=readWebSettings(settingsScope);
   try{
    if(typeof saved.paperAutoReceive==='boolean')value=await execute<T>('autoReceive',{enabled:saved.paperAutoReceive},signal);
    if(typeof saved.paperInterval==='number'&&Number.isFinite(saved.paperInterval)&&saved.paperInterval>=0.25&&saved.paperInterval<=720)value=await execute<T>('interval',{hours:saved.paperInterval},signal);
   }catch(error){preferencesApplied=false;throw error;}
  }
  if(operation==='autoReceive')writeWebSettings(settingsScope,{paperAutoReceive:args.enabled});
  if(operation==='interval')writeWebSettings(settingsScope,{paperInterval:args.hours});
  if(operation==='export'){
   const exported=value as {asset_id:string;name:string};
   const file=await request('assets/'+exported.asset_id+'/content','GET',undefined,signal),blob=await file.blob();
   const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=exported.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);
  }return value;
 }
 function invoke<T>(operation:string,args:Record<string,unknown>={},signal?:AbortSignal,onProgress?:(p:PaperProgress)=>void):Promise<T>{
  if(operation==='autoReceive'&&args.enabled===false)automatic?.abort();
  // Open the picker within the originating user gesture, before queued PDF work.
  const selected=operation==='chooseLibrary'||operation==='importLocal'?pick(operation==='chooseLibrary'):undefined;
  const result=queue.catch(()=>{}).then(async()=>{
   if(selected){const files=await selected;if(!files.length)return {cancelled:true} as T;const imported=await upload(files,signal);const next=operation==='chooseLibrary'?{name:files[0].webkitRelativePath.split('/')[0]||'我的 PDF',files:imported}:{folder:args.folder,files:imported};return execute<T>('__prepared__'+operation,next,signal,onProgress);}
   return execute<T>(operation,args,signal,onProgress);
  }).catch(e=>{if(signal?.aborted||lifetime.aborted)throw Error('操作已取消。');throw e;});queue=result;return result;
 }
 const port:PapersClient={...createPapers(invoke),web:true,onStatus(listener){listeners.add(listener);return()=>listeners.delete(listener);},
  trash:()=>invoke('trash'),restoreLocal:id=>invoke('restoreLocal',{id}),maintain(){
   let stopped=false,timer:ReturnType<typeof setTimeout>|undefined,generation=0;
   function schedule(status:PaperStatus){generation++;clearTimeout(timer);if(!stopped&&status.autoReceive!==false)timer=setTimeout(()=>void check(),Math.min((status.checkIntervalHours??6)*3600000,2147483647));}
   scheduleSettings=schedule;
   async function check(startup=false){const version=generation;automatic=new AbortController();const signal=automatic.signal;let status:PaperStatus|undefined;try{status=await invoke<PaperStatus>('status',{},signal);
    if(status.autoReceive!==false||startup){status=await invoke<PaperStatus>('refresh',{},signal);if(status.autoReceive!==false)status=await invoke<PaperStatus>('download',{since:status.firstLaunch},signal);if(!stopped)emit(status);}
   }catch(e){if(!stopped&&!signal.aborted&&status)emit({...status,warning:(e as Error).message});}
   finally{if(!stopped&&version===generation&&status)schedule(status);}}
   timer=setTimeout(()=>void check(true),1200);return()=>{stopped=true;clearTimeout(timer);automatic?.abort();scheduleSettings=undefined;};
  }};
 return port;
}
