import {exportParams} from '../modules/speech-synthesis/state.ts';
import type {JobView,ResearchFile} from './research.ts';
import {sha256} from './research.ts';
import type {M06Port,Result} from '../modules/speech-synthesis/port.ts';
export interface M06Transport {project:string;parameters(text:string,signal:AbortSignal):Promise<{asset_id:string;sha256:string}>;source(file:ResearchFile):Promise<{asset_id:string;sha256:string}>;create(body:unknown):Promise<JobView>;job(id:string):Promise<JobView>;cancel(id:string):Promise<unknown>;read(job:string,id:string,sha:string):Promise<ArrayBuffer>;save?(job:string):Promise<boolean>}
export function downloadBytes(bytes:BlobPart,name:string){const url=URL.createObjectURL(new Blob([bytes]));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
export function m06Port(t:M06Transport):M06Port {
 async function read(r:Pick<Result,'job'|'files'>,name:string){const f=r.files.find(f=>f.name===name);if(!f)throw Error('合成结果缺少 '+name);const raw=await t.read(r.job,f.id,f.sha256);if(raw.byteLength!==f.size_bytes||await sha256(raw)!==f.sha256)throw Error('合成工件校验失败');return raw;}
 return {cancel:t.cancel,
 async run(action,config,file,signal,progress){
  const audio=action==='extract'&&file?await t.source(file):undefined;if(signal.aborted)throw Error('操作已取消');
  const text=exportParams(config);if(new TextEncoder().encode(text).length>2_000_000)throw Error('完整参数超过 2 MB');
  const parameters=await t.parameters(text,signal);if(signal.aborted)throw Error('操作已取消');
  let job=await t.create({project_id:t.project,idempotency_key:crypto.randomUUID(),action,parameters,audio});progress(job);
  const deadline=Date.now()+180000;
  while(['queued','running','cancel_requested'].includes(job.state)){
   if(signal.aborted||Date.now()>deadline){await t.cancel(job.id);throw Error(signal.aborted?'操作已取消':'等待超时，已请求取消');}
   await new Promise(resolve=>setTimeout(resolve,120));job=await t.job(job.id);progress(job);
  }
  if(signal.aborted)throw Error('操作已取消');
  if(job.state!=='succeeded')throw Error(job.error_code??'合成任务失败');
  if(!job.result_manifest||!('files' in job.result_manifest))throw Error('结果清单不完整');
  const base={job:job.id,files:job.result_manifest.files};
  const result:Result={...base,metadata:JSON.parse(new TextDecoder().decode(await read(base,'m06.ptb.json')))};
  if(action==='synthesize')result.wav=await read(result,'synthesis.wav');return result;
 },async download(result,name){if(t.save&&name==='synthesis.wav'){if(!await t.save(result.job))throw Error('未保存，合成工件仍保留');return;}downloadBytes(await read(result,name),name);}};
}
