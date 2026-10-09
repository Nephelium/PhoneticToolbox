import {m07Archive} from './m07Archive.ts';
import type {components} from '../../../contracts/generated/api';
import type {ResearchFile,JobView} from './research.ts';
import {sha256} from './research.ts';
import type {Request,Analysis,Generation,Points} from '../modules/phonation-synthesis/state.ts';
export interface Metadata {action:'analyze'|'apply'|'generate';analysis:Analysis;generation:Generation;alignment:'normalize'|'onset';point_count:number;controls:Points;source_f0:number[];target_f0:number[];source_samples:number;target_samples:number;continuum_type:1|2|3;reverse_direction:boolean;batch_id:string|null;batch_group_index:number;batch_group_count:1|6;analysis_job_id:string|null;inputs:Record<'source'|'target',{asset_id:string;sha256:string}>;runtime:Record<string,string>;algorithm:string}
export interface M07Result {job:string;files:components['schemas']['M07Manifest']['files'];metadata:Metadata}
export interface M07F0Display {kind:'synthesis-control-f0';alignment:'normalize'|'onset';curves:{name:string;axis:number[];values:number[]}[]}
export interface M07Transport {project:string;source(file:ResearchFile):Promise<{asset_id:string;sha256:string}>;create(body:Request):Promise<JobView>;job(id:string):Promise<JobView>;list():Promise<JobView[]>;cancel(id:string):Promise<unknown>;retry(id:string,key:string):Promise<JobView>;read(job:string,id:string,sha:string):Promise<ArrayBuffer>;f0?(job:string):Promise<M07F0Display>;save?(job:string,directory:string):Promise<unknown>}
export type Settings=Omit<Request,'project_id'|'idempotency_key'|'source'|'target'>;
export interface M07Port {run(source:ResearchFile,target:ResearchFile,settings:Settings,signal:AbortSignal,progress:(job:JobView)=>void):Promise<M07Result>;result(job:JobView):Promise<M07Result>;list():Promise<JobView[]>;cancel(id:string):Promise<unknown>;retry(id:string):Promise<JobView>;read(result:M07Result,name:string):Promise<ArrayBuffer>;f0?(result:M07Result):Promise<M07F0Display>;save(result:M07Result,directory:string):Promise<void>}
export const messages:Record<string,string>={m07_analysis_stale:'分析已失效，请重新提取 F0',m07_reaper_unavailable:'REAPER 不可用，未切换后端',m07_platform_unverified:'当前平台尚未通过 M07 科学与资源准入',m07_no_voiced_region:'没有有效有声区，请检查录音与 F0 范围',m07_insufficient_pulses:'有效脉冲不足，请检查录音或脉冲参数',m07_f0_analysis_failed:'F0 分析失败，输入可能过短或 F0 范围不适合',m07_input_budget:'输入超过 10 秒、480,000 帧或 8 MB 预算',m07_invalid_audio:'音频为空、损坏或包含非有限数值',m07_timeout:'计算超时，已回收工作进程',m07_execution_failed:'科学计算失败，当前组未发布',cancelled:'任务已取消，已完成组保留',quota_exceeded:'剩余空间不足，已完成组保留',input_unavailable:'输入已失效或临近到期，请重新选择',m07_invalid_f0:'F0 控制点无效，请检查数值'};
export function downloadM07(raw:ArrayBuffer,name:string){const url=URL.createObjectURL(new Blob([raw]));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
export function m07Port(t:M07Transport):M07Port{
 async function read(result:Pick<M07Result,'job'|'files'>,name:string){const f=result.files.find(f=>f.name===name);if(!f)throw Error('结果文件缺失：'+name);const raw=await t.read(result.job,f.id,f.sha256);if(raw.byteLength!==f.size_bytes||await sha256(raw)!==f.sha256)throw Error('结果文件校验失败');return raw;}
 async function result(job:JobView):Promise<M07Result>{if(job.state!=='succeeded'||job.result_manifest?.kind!=='managed_m07_files')throw Error(messages[job.error_code??'']??'当前组尚未完整成功');const base={job:job.id,files:job.result_manifest.files};return {...base,metadata:JSON.parse(new TextDecoder().decode(await read(base,'m07.ptb.json')))};}
 return {read,result,f0:t.f0?r=>t.f0!(r.job):undefined,list:t.list,cancel:t.cancel,retry:id=>t.retry(id,crypto.randomUUID()),async save(r,directory){if(t.save){await t.save(r.job,directory);return;}const files=[];for(const f of r.files)files.push({name:f.name,raw:await read(r,f.name)});downloadM07(m07Archive(files),'M07-'+r.job+'.zip');},async run(source,target,settings,signal,progress){
  const refs=await Promise.all([t.source(source),t.source(target)]);if(signal.aborted)throw Error('操作已取消');
  let job=await t.create({...settings,project_id:t.project,idempotency_key:crypto.randomUUID(),source:refs[0],target:refs[1]});progress(job);
  const deadline=Date.now()+300000;
  while(['queued','running','cancel_requested'].includes(job.state)){
   if(signal.aborted||Date.now()>deadline){await t.cancel(job.id);throw Error('操作已取消，已完成组保留');}
   await new Promise(resolve=>setTimeout(resolve,150));job=await t.job(job.id);progress(job);
  }
  if(signal.aborted)throw Error('操作已取消，已完成组保留');return result(job);
 }};
}
