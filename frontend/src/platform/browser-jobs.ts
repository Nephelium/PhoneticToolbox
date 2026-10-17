import type {JobView,ResearchFile,ReconstructionConfig} from './research.ts';

type RecordView<C>={schema:'ptb-browser-job/1';owner:string;project:string;file:ResearchFile;config:C;job:JobView};
type Ports<C>={list:()=>Promise<any[]>;read:(asset:any)=>Promise<ArrayBuffer>;upload:(name:string,raw:ArrayBuffer)=>Promise<any>;compute:(file:ResearchFile,config:C,signal:AbortSignal)=>Promise<{files:{name:string;data:string}[]}>};
type JobSpec={prefix:string;operation:'spectrogram_to_audio'|'lpc_analysis';kind:'managed_spec2wav_files'|'managed_lpc_files';names:string[]};
const reconstruction:JobSpec={prefix:'重建任务-',operation:'spectrogram_to_audio',kind:'managed_spec2wav_files',names:['calibrated.png','reconstructed.png','reconstructed.wav','reconstruction.ptb.json']};
/** Local execution, managed result/history assets; no counterfeit server queue rows. */
export class BrowserJobs<C=ReconstructionConfig> {
 private entries=new Map<string,RecordView<C>>();private running=new Map<string,AbortController>();private loaded=new Map<string,string>();private pendingWrites=new Map<string,RecordView<C>>();private closed=false;
 private unload=(event:BeforeUnloadEvent)=>{if(this.running.size||this.pendingWrites.size){event.preventDefault();event.returnValue='';}};
 constructor(privateOwner:string,privateProject:string,ports:Ports<C>,spec:JobSpec=reconstruction){this.owner=privateOwner;this.project=privateProject;this.ports=ports;this.spec=spec;if(typeof window!=='undefined')window.addEventListener('beforeunload',this.unload);}
 private owner:string;private project:string;private ports:Ports<C>;private spec:JobSpec;
 has(id:string){return this.entries.has(id);}
 get(id:string){return this.entries.get(id)?.job;}
 source(id:string){const record=this.entries.get(id);return record?{file:record.file,config:record.config}:undefined;}
 async persist(record:RecordView<C>){if(this.closed)return;const previous=record.job.error_code;if(previous==='结果保存尚未完成，请保持页面打开后重试。')record.job.error_code=null;try{await this.ports.upload(this.spec.prefix+record.job.id+'.ptb-browser.json',new TextEncoder().encode(JSON.stringify(record)).buffer);this.pendingWrites.delete(record.job.id);}catch(error){record.job.error_code=previous;this.pendingWrites.set(record.job.id,record);throw error;}}
 async start(file:ResearchFile,config:C){
  if(this.closed)throw Error('项目会话已关闭');if(this.running.size)throw Error('请先等待或取消当前浏览器计算。');
  const id=crypto.randomUUID(),now=Date.now()/1000,job:JobView={id,project_id:this.project,operation:this.spec.operation,state:'running',progress:0,generation:1,created_at:now,updated_at:now,error_code:null,result_manifest:null,execution:'browser'};
  const record:RecordView<C>={schema:'ptb-browser-job/1',owner:this.owner,project:this.project,file:JSON.parse(JSON.stringify(file)),config:JSON.parse(JSON.stringify(config)),job};this.entries.set(id,record);const abort=new AbortController();this.running.set(id,abort);
  try{await this.persist(record);}catch(error){this.running.delete(id);this.entries.delete(id);this.pendingWrites.delete(id);throw error;}
  void (async()=>{
   try{
    const output=await this.ports.compute(file,config,abort.signal);if(this.closed||abort.signal.aborted)throw new DOMException('Cancelled','AbortError');
    const names=this.spec.names;if(output.files.length!==names.length||!output.files.every(f=>names.includes(f.name))||new Set(output.files.map(f=>f.name)).size!==names.length)throw Error('计算结果清单不完整');
    const files=[];let coreVersion='',policyVersion:1|2=1;
    for(const item of output.files){if(this.closed||abort.signal.aborted)throw new DOMException('Cancelled','AbortError');const raw=Uint8Array.from(atob(item.data),c=>c.charCodeAt(0));if(item.name.endsWith('.json')){const meta=JSON.parse(new TextDecoder().decode(raw));coreVersion=meta.core_version??meta.runtime?.versions?.['phonetic-core'];}const asset=await this.ports.upload(id+'-'+item.name,raw.buffer);policyVersion=asset.policyVersion??1;files.push({id:asset.id,name:item.name,kind:'result' as const,size_bytes:asset.size_bytes??asset.size,expires_at:asset.expires_at??asset.expiresAt??null,sha256:asset.sha256});}
    if(this.closed||abort.signal.aborted)throw new DOMException('Cancelled','AbortError');
    job.result_manifest={kind:this.spec.kind,complete:true,operation:this.spec.operation,core_version:coreVersion,policy_version:policyVersion,files} as JobView['result_manifest'];job.state='succeeded';job.progress=1;
   }catch(error){job.state=abort.signal.aborted?'cancelled':'failed';if(!abort.signal.aborted)console.warn('Browser computation failed:',error);const known=String(error).match(/(?:ValueError|AcousticFailure): ([a-z0-9_]{1,80})/);job.error_code=abort.signal.aborted?'cancelled':known?.[1]??'浏览器计算未完成，请点重试改用服务器。';}
   finally{job.updated_at=Date.now()/1000;this.running.delete(id);await this.persist(record).catch(()=>{job.error_code='结果保存尚未完成，请保持页面打开后重试。';});}
  })();return structuredClone(job);
 }
 async list(){
  for(const record of this.pendingWrites.values())await this.persist(record).catch(()=>{});
  const prefix=this.spec.prefix,assets=(await this.ports.list()).filter(a=>a.name.startsWith(prefix)&&a.name.endsWith('.ptb-browser.json')&&a.size_bytes<=32000).sort((a,b)=>b.created_at-a.created_at);const latest=new Set<string>();
  for(const asset of assets.slice(0,1000)){
   const id=asset.name.slice(prefix.length,-'.ptb-browser.json'.length);if(latest.has(id))continue;latest.add(id);if(this.running.has(id)||this.loaded.get(id)===asset.sha256)continue;
   let record:RecordView<C>;try{record=JSON.parse(new TextDecoder().decode(await this.ports.read(asset))) as RecordView<C>;}catch{continue;}
   if(record?.schema!=='ptb-browser-job/1'||record.owner!==this.owner||record.project!==this.project||record.job?.id!==id||record.job.project_id!==this.project||record.job.operation!==this.spec.operation||!Number.isFinite(record.job.created_at)||!Number.isFinite(record.job.updated_at))continue;
   if((this.entries.get(id)?.job.updated_at??0)>record.job.updated_at)continue;
   if(['running','queued'].includes(record.job.state)&&Date.now()/1000-record.job.created_at>180){record.job.state='interrupted';record.job.error_code='浏览器计算已中断，请重试。';}
   this.entries.set(id,record);this.loaded.set(id,asset.sha256);
  }
  for(const [id,record] of this.entries)if(!this.running.has(id)&&['running','queued'].includes(record.job.state)&&Date.now()/1000-record.job.created_at>180){record.job.state='interrupted';record.job.error_code='浏览器计算已中断，请重试。';}
  return [...this.entries.values()].map(v=>structuredClone(v.job)).sort((a,b)=>b.created_at-a.created_at);
 }
 async cancel(id:string){const record=this.entries.get(id);if(!record)throw Error('重建任务不存在');if(!['running','queued'].includes(record.job.state))return structuredClone(record.job);this.running.get(id)?.abort();record.job.state='cancelled';record.job.error_code='cancelled';record.job.updated_at=Date.now()/1000;await this.persist(record);return structuredClone(record.job);}
 async retry(id:string){const record=this.entries.get(id);if(!record)throw Error('重建任务不存在');return this.start(record.file,record.config);}
 close(){this.closed=true;if(typeof window!=='undefined')window.removeEventListener('beforeunload',this.unload);for(const abort of this.running.values())abort.abort();this.running.clear();this.entries.clear();this.loaded.clear();this.pendingWrites.clear();}
}
