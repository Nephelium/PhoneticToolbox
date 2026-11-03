import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
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
  progress({...job,state:cancelled?'cancelled':'failed',updated_at:Date.now()/1000,error_code:cancelled?'cancelled':pending.has(job.id)?tr('platform.results_could_not_be_saved_retry_saving')+(error instanceof Error?error.message:String(error)):error instanceof Error?error.message:String(error)});
 }
 async function save(prepared:Prepared,signal:AbortSignal,progress:(job:JobView)=>void){
  while(prepared.cursor<prepared.files.length){signal.throwIfAborted();const {name,raw}=prepared.files[prepared.cursor],asset=await ports.save(prepared.job.id+'-'+name,raw,signal);signal.throwIfAborted();prepared.result.files.push({...asset,name});prepared.cursor++;}
  signal.throwIfAborted();pending.delete(prepared.job.id);progress({...prepared.job,state:'succeeded',progress:1,updated_at:Date.now()/1000});return prepared.result;
 }
 return {
  async run(action,config,file,signal,progress){
   if(pending.size)throw uiError(uiMessage('platform.synthesis_results_await_saving_retry_saving_or_confirm_discard'));
   if(running.size)throw uiError(uiMessage('platform.wait_for_the_current_task_to_finish_or_cancel'));
   if(action==='extract'||action==='synthesize'&&config.duration>3)return remote.run(action,config,file,signal,progress);
   signal.throwIfAborted();const parameters=exportParams(config),submitted=structuredClone(config);if(new TextEncoder().encode(parameters).byteLength>2_000_000)throw uiError(uiMessage('platform.the_full_parameters_exceed_2_mb'));
   const id=crypto.randomUUID(),controller=new AbortController(),current=AbortSignal.any([signal,controller.signal]),now=Date.now()/1000;
   const job:JobView={id,project_id:ports.project,operation:'speech_synthesis',execution:'browser',state:'running',created_at:now,updated_at:now,progress:0,generation:1,error_code:null,result_manifest:null};
   running.set(id,controller);remember(id);progress(job);let fallback=false;
   try{
    const seed=crypto.getRandomValues(new Uint32Array(1))[0],output=await ports.compute({action,parameters,seed},current);current.throwIfAborted();
    if(!output){fallback=true;return await remote.run(action,config,file,current,progress);}
    const expected=action==='synthesize'?['synthesis.wav','m06.ptb.json','parameters.csv']:['m06.ptb.json','parameters.csv'];
    if(JSON.stringify(output.files.map(f=>f.name))!==JSON.stringify(expected))throw uiError(uiMessage('platform.the_synthesis_result_manifest_is_incomplete'));
    const prepared:Prepared={job,action,config:submitted,files:[],result:{job:id,files:[],metadata:undefined!},cursor:0};let size=0;
    for(const item of output.files){current.throwIfAborted();const raw=Uint8Array.from(atob(item.data),c=>c.charCodeAt(0));size+=raw.byteLength;if(size>24_000_000)throw uiError(uiMessage('platform.synthesis_results_exceed_24_mb'));prepared.files.push({name:item.name,raw:raw.buffer});if(item.name==='m06.ptb.json')prepared.result.metadata=JSON.parse(new TextDecoder().decode(raw));if(item.name==='synthesis.wav')prepared.result.wav=raw.buffer;}
    pending.set(id,prepared);progress({...job,progress:.95,updated_at:Date.now()/1000});return await save(prepared,current,progress);
   }catch(error){if(!fallback)failed(job,current,error,progress);throw error;}finally{running.delete(id);}
  },
  canRetrySave:id=>pending.has(id),
  pendingSave(){const prepared=pending.values().next().value;return prepared?{job:{...prepared.job,state:'failed',error_code:tr('platform.synthesis_results_have_not_been_saved_retry_saving')},action:prepared.action,config:structuredClone(prepared.config)}:undefined;},
  discardSave(id){if(running.has(id))throw uiError(uiMessage('platform.results_are_being_saved_cancel_the_task_first'));pending.delete(id);},
  async retrySave(id,signal,progress){
   const prepared=pending.get(id);if(!prepared)throw uiError(uiMessage('platform.there_are_no_synthesis_results_awaiting_saving_run_synthesis'));if(running.has(id))throw uiError(uiMessage('platform.results_are_being_saved_please_wait'));
   const controller=new AbortController(),current=AbortSignal.any([signal,controller.signal]);running.set(id,controller);progress({...prepared.job,state:'running',updated_at:Date.now()/1000});
   try{return await save(prepared,current,progress);}catch(error){failed(prepared.job,current,error,progress);throw error;}finally{running.delete(id);}
  },
  download:(result,name)=>remote.download(result,name),
  async cancel(id){const controller=running.get(id);if(controller){controller.abort();return;}if(localIds.has(id))return;return remote.cancel(id);}
 };
}
