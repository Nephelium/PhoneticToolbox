import {exportParams} from '../modules/speech-synthesis/state.ts';
import type {M06Port,Result} from '../modules/speech-synthesis/port.ts';
import type {JobView} from './research.ts';
type Output={files:{name:string;data:string}[]};
type Ports={project:string;compute(args:unknown,signal:AbortSignal):Promise<Output|null>;save(name:string,raw:ArrayBuffer,signal:AbortSignal):Promise<{id:string;sha256:string;size_bytes:number}>};
/** Generate/synthesize reuse the original Klatt child; extraction stays native. */
export function browserM06(remote:M06Port,ports:Ports):M06Port {
 const running=new Map<string,AbortController>();
 return {
  async run(action,config,file,signal,progress){
   if(action==='extract'||action==='synthesize'&&config.duration>3)return remote.run(action,config,file,signal,progress);
   signal.throwIfAborted();const parameters=exportParams(config);if(new TextEncoder().encode(parameters).byteLength>2_000_000)throw Error('完整参数超过2 MB。');
   const id=crypto.randomUUID(),controller=new AbortController(),current=AbortSignal.any([signal,controller.signal]),now=Date.now()/1000;
   const job:JobView={id,project_id:ports.project,operation:'speech_synthesis',execution:'browser',state:'running',created_at:now,updated_at:now,progress:0,generation:1,error_code:null,result_manifest:null};
   running.set(id,controller);progress(job);
   try{
    const seed=crypto.getRandomValues(new Uint32Array(1))[0],output=await ports.compute({action,parameters,seed},current);current.throwIfAborted();
    if(!output)return await remote.run(action,config,file,current,progress);
    const expected=action==='synthesize'?['synthesis.wav','m06.ptb.json','parameters.csv']:['m06.ptb.json','parameters.csv'];
    if(JSON.stringify(output.files.map(f=>f.name))!==JSON.stringify(expected))throw Error('合成结果清单不完整。');
    const result:Result={job:id,files:[],metadata:undefined!};let size=0;
    for(const item of output.files){current.throwIfAborted();const raw=Uint8Array.from(atob(item.data),c=>c.charCodeAt(0));size+=raw.byteLength;if(size>24_000_000)throw Error('合成结果超过24 MB。');const asset=await ports.save(id+'-'+item.name,raw.buffer,current);result.files.push({...asset,name:item.name});if(item.name==='m06.ptb.json')result.metadata=JSON.parse(new TextDecoder().decode(raw));if(item.name==='synthesis.wav')result.wav=raw.buffer;}
    current.throwIfAborted();progress({...job,state:'succeeded',progress:1,updated_at:Date.now()/1000});return result;
   }finally{running.delete(id);}
  },
  download:(result,name)=>remote.download(result,name),
  async cancel(id){const controller=running.get(id);if(controller){controller.abort();return;}return remote.cancel(id);}
 };
}
