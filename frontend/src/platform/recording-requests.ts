import {uiMessage,uiError,responseError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {serialRequests} from './serial-requests.ts';
export interface RecordingChannel {
  recording:(id:string,body:string)=>void;
  recordingReady:{connect:(listener:(id:string,raw:string)=>void)=>void};
}
/** Independent of scientific jobs: neither polling nor stop waits on the task queue. */
export function recordingRequests(channel:RecordingChannel,timeoutMs=45000){
  const pending=new Map<string,{resolve:(value:any)=>void;reject:(error:Error)=>void;timer:ReturnType<typeof setTimeout>}>();
  channel.recordingReady.connect((id,raw)=>{
    const request=pending.get(id);if(!request)return;
    pending.delete(id);clearTimeout(request.timer);
    try{const result=JSON.parse(raw);if(!result.ok)throw responseError(result.error,result.display);request.resolve(result.value);}
    catch(error){request.reject(error instanceof Error?error:uiError(uiMessage('platform.invalid_recording_response_format')));}
  });
  return serialRequests(<T>(body:unknown):Promise<T>=>new Promise((resolve,reject)=>{
    const id=crypto.randomUUID();
    const timer=setTimeout(()=>{pending.delete(id);reject(uiError(uiMessage('platform.the_recording_response_timed_out_check_the_recording_state')));},timeoutMs);
    pending.set(id,{resolve,reject,timer});
    try{channel.recording(id,JSON.stringify(body));}
    catch(error){pending.delete(id);clearTimeout(timer);reject(error);}
  }));
}
