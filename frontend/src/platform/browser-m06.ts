import {exportParams} from '../modules/speech-synthesis/state.ts';
import type {Action,M06Port,Result} from '../modules/speech-synthesis/port.ts';
import type {Config} from '../modules/speech-synthesis/state.ts';
import type {JobView} from './research.ts';
type Output={files:{name:string;data:string}[]};
type Ports={project:string;compute(args:unknown,signal:AbortSignal):Promise<Output|null>;save(name:string,raw:ArrayBuffer,signal:AbortSignal):Promise<{id:string;sha256:string;size_bytes:number}>};
/** Generate/synthesize reuse the original Klatt child; extraction stays native. */
export function browserM06(remote:M06Port,ports:Ports):M06Port {
 const running=new Map<string,AbortController>();
 const localIds=new Set<string>();
 type Prepared={job:JobView;action:Action;config:Config;files:{name:string;raw:ArrayBuffer}[];result:Result;cursor:number};
 const pending=new Map<string,Prepared>();
 function remember(id:string){localIds.add(id);for(const old of localIds){if(localIds.size<=64)break;if(!running.has(old)&&!pending.has(old))localIds.delete(old);}}
 function failed(job:JobView,signal:AbortSignal,error:unknown,progress:(job:JobView)=>void){
  const cancelled=signal.aborted||error instanceof DOMException&&error.name==='AbortError';
  // Computed bytes remain recoverable until saved or explicitly discarded.
  progress({...job,state:cancelled?'cancelled':'failed',updated_at:Date.now()/1000,error_code:cancelled?'cancelled':pending.has(job.id)?'结果保存失败，可重试保存：'+(error instanceof Error?error.message:String(error)):error instanceof Error?error.message:String(error)});
 }
 async function save(prepared:Prepared,signal:AbortSignal,progress:(job:JobView)=>void){
  while(prepared.cursor<prepared.files.length){signal.throwIfAborted();const {name,raw}=prepared.files[prepared.cursor],asset=await ports.save(prepared.job.id+'-'+name,raw,signal);signal.throwIfAborted();prepared.result.files.push({...asset,name});prepared.cursor++;}
  signal.throwIfAborted();pending.delete(prepared.job.id);progress({...prepared.job,state:'succeeded',progress:1,updated_at:Date.now()/1000});return prepared.result;
 }
 return {
  async run(action,config,file,signal,progress){
   if(pending.size)throw Error('还有待保存的合成结果，请先重试保存或确认放弃后关闭标签。');
   if(running.size)throw Error('请等待当前任务完成或取消。');
   if(action==='extract'||action==='synthesize'&&config.duration>3)return remote.run(action,config,file,signal,progress);
   signal.throwIfAborted();const parameters=exportParams(config),submitted=structuredClone(config);if(new TextEncoder().encode(parameters).byteLength>2_000_000)throw Error('完整参数超过2 MB。');
   const id=crypto.randomUUID(),controller=new AbortController(),current=AbortSignal.any([signal,controller.signal]),now=Date.now()/1000;
   const job:JobView={id,project_id:ports.project,operation:'speech_synthesis',execution:'browser',state:'running',created_at:now,updated_at:now,progress:0,generation:1,error_code:null,result_manifest:null};
   running.set(id,controller);remember(id);progress(job);let fallback=false;
   try{
    const seed=crypto.getRandomValues(new Uint32Array(1))[0],output=await ports.compute({action,parameters,seed},current);current.throwIfAborted();
    if(!output){fallback=true;return await remote.run(action,config,file,current,progress);}
    const expected=action==='synthesize'?['synthesis.wav','m06.ptb.json','parameters.csv']:['m06.ptb.json','parameters.csv'];
    if(JSON.stringify(output.files.map(f=>f.name))!==JSON.stringify(expected))throw Error('合成结果清单不完整。');
    const prepared:Prepared={job,action,config:submitted,files:[],result:{job:id,files:[],metadata:undefined!},cursor:0};let size=0;
    for(const item of output.files){current.throwIfAborted();const raw=Uint8Array.from(atob(item.data),c=>c.charCodeAt(0));size+=raw.byteLength;if(size>24_000_000)throw Error('合成结果超过24 MB。');prepared.files.push({name:item.name,raw:raw.buffer});if(item.name==='m06.ptb.json')prepared.result.metadata=JSON.parse(new TextDecoder().decode(raw));if(item.name==='synthesis.wav')prepared.result.wav=raw.buffer;}
    pending.set(id,prepared);progress({...job,progress:.95,updated_at:Date.now()/1000});return await save(prepared,current,progress);
   }catch(error){if(!fallback)failed(job,current,error,progress);throw error;}finally{running.delete(id);}
  },
  canRetrySave:id=>pending.has(id),
  pendingSave(){const prepared=pending.values().next().value;return prepared?{job:{...prepared.job,state:'failed',error_code:'合成结果尚未保存，请重试保存。'},action:prepared.action,config:structuredClone(prepared.config)}:undefined;},
  discardSave(id){if(running.has(id))throw Error('结果正在保存，请先取消任务。');pending.delete(id);},
  async retrySave(id,signal,progress){
   const prepared=pending.get(id);if(!prepared)throw Error('没有待保存的合成结果，请重新合成。');if(running.has(id))throw Error('结果正在保存，请稍候。');
   const controller=new AbortController(),current=AbortSignal.any([signal,controller.signal]);running.set(id,controller);progress({...prepared.job,state:'running',updated_at:Date.now()/1000});
   try{return await save(prepared,current,progress);}catch(error){failed(prepared.job,current,error,progress);throw error;}finally{running.delete(id);}
  },
  download:(result,name)=>remote.download(result,name),
  async cancel(id){const controller=running.get(id);if(controller){controller.abort();return;}if(localIds.has(id))return;return remote.cancel(id);}
 };
}
