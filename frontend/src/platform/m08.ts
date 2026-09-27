import type {ResearchFile,JobView} from './research.ts';
import type {M08Port,Result,Job,Config} from '../modules/pitch-manipulation/port.ts';
import {sha256} from './research.ts';

type Ref={asset_id:string;sha256:string};
type Managed=Result&{job_id:string;sha256:string;saved:boolean};
export interface M08Transport {
 project:string;
 source(file:ResearchFile):Promise<Ref>;
 request<T>(action:string,body?:unknown):Promise<T>;
 job(id:string):Promise<JobView>;
 read(job:string,id:string,sha:string):Promise<ArrayBuffer>;
 cancel(id:string):Promise<unknown>;
 export?(result:Managed):Promise<string>;
}
/** Both production adapters use these exact task/ID contracts. No local compute. */
export function m08Port(transport:M08Transport):M08Port {
 const refs=new Map<string,Ref>(),localIds=new Map<string,string>(),results=new Map<string,Managed>();
 async function source(file:ResearchFile){const ref=await transport.source(file);refs.set(file.id,ref);localIds.set(ref.asset_id,file.id);return ref;}
 const mapped=(r:Managed):Managed=>{results.set(r.id,r);return {...r,source_id:localIds.get(r.source_id)??r.source_id};};
 async function jobs(){const values=await transport.request<Job[]>('list');return values.map(j=>({...j,source_id:localIds.get(j.source_id)??j.source_id,results:j.results.map(r=>mapped(r as Managed))}));}
 async function wait(job:JobView,signal?:AbortSignal){
  const until=Date.now()+150000;
  while(['queued','running','cancel_requested'].includes(job.state)){
   if(signal?.aborted){await transport.cancel(job.id);throw new DOMException('操作已取消','AbortError');}
   if(Date.now()>until)throw Error('任务仍在处理，可在任务列表查询或取消。');
   await new Promise(resolve=>setTimeout(resolve,150));job=await transport.job(job.id);
  }
  if(job.state!=='succeeded')throw Error(job.error_code??'M08 任务未完成');return job;
 }
 async function submit(file:ResearchFile,config:Config,key:string){
  const ref=await source(file);const j=await transport.request<JobView>('create',{project_id:transport.project,audio:ref,config,idempotency_key:key});
  return {id:j.id,state:j.state,source_id:file.id,results:[],error:j.error_code??undefined} as Job;
 }
 async function scope(ids:string[]){
  const values=ids.map(id=>results.get(id));if(!values.length||values.some(v=>!v))throw Error('结果已变化，请刷新。');
  const sourceId=values[0]!.source_id;if(values.some(v=>v!.source_id!==sourceId))throw Error('只能管理同一源音频的明确结果。');
  const ref=[...refs.values()].find(v=>v.asset_id===sourceId);if(!ref)throw Error('源文件授权已失效，请重新选择。');
  return {project_id:transport.project,source:ref,ids};
 }
 async function audio(r:Result){const value=results.get(r.id);if(!value)throw Error('结果已失效，请刷新。');const raw=await transport.read(value.job_id,value.id,value.sha256);if(await sha256(raw)!==value.sha256)throw Error('结果校验失败');return raw;}
 return {
  async preview(file,signal){
   const ref=await source(file);
   const created=await transport.request<JobView>('create',{project_id:transport.project,audio:ref,config:{action:'preview'},idempotency_key:'m08-preview-'+crypto.randomUUID()});
   const job=await wait(created,signal);const files=job.result_manifest&&'files' in job.result_manifest?job.result_manifest.files:[];
   const metadata=files.find(f=>f.name==='m08.ptb.json'),wav=files.find(f=>f.name==='m08-preview.wav');
   if(!metadata||!wav)throw Error('M08 预览产物不完整');
   const data=JSON.parse(new TextDecoder().decode(await transport.read(job.id,metadata.id,metadata.sha256)));
   if(data.audio_sha256!==ref.sha256)throw Error('源音频校验不一致');
   return {...data,sha256:ref.sha256,wav:await transport.read(job.id,wav.id,wav.sha256)};
  },submit,jobs,cancel:async id=>{await transport.cancel(id);},audio,
  async history(file){const ref=await source(file);return (await transport.request<Managed[]>('history',{project_id:transport.project,source:ref})).map(mapped);},
  async save(r){const copy=await wait(await transport.request<JobView>('save',await scope([r.id])));const job=(await jobs()).find(j=>j.id===copy.id);if(!job?.results[0])throw Error('保存结果未发布');const saved=job.results[0];const name=await transport.export?.(results.get(saved.id)!);
   if(name&&name!==saved.name){await transport.request('rename',{...await scope([saved.id]),names:[name]});return mapped({...results.get(saved.id)!,name});}return saved;},
  async remove(ids){return transport.request('remove',await scope(ids));},
  async rename(changes){await transport.request('rename',{...await scope(changes.map(c=>c.id)),names:changes.map(c=>c.name)});return (await jobs()).flatMap(j=>j.results).filter(r=>changes.some(c=>c.id===r.id));},
  async download(r){const bytes=await audio(r),url=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));const a=document.createElement('a');a.href=url;a.download=r.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);},
 };
}
