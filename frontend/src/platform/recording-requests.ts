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
    try{const result=JSON.parse(raw);if(!result.ok)throw Error(result.error||'录音操作失败。');request.resolve(result.value);}
    catch(error){request.reject(error instanceof Error?error:Error('录音响应格式错误。'));}
  });
  return serialRequests(<T>(body:unknown):Promise<T>=>new Promise((resolve,reject)=>{
    const id=crypto.randomUUID();
    const timer=setTimeout(()=>{pending.delete(id);reject(Error('录音操作响应超时，请先核对录音状态，再重试保存或停止。'));},timeoutMs);
    pending.set(id,{resolve,reject,timer});
    try{channel.recording(id,JSON.stringify(body));}
    catch(error){pending.delete(id);clearTimeout(timer);reject(error);}
  }));
}
