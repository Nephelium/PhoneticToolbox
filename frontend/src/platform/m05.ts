import type {LipPort} from '../modules/lip-extraction/port.ts';
import type {LipResult,OfflineConfig} from '../modules/lip-extraction/port.ts';
import type {JobView} from './research.ts';
import {sha256} from './research.ts';
export async function saveM05Local(name:string,blob:Blob):Promise<'saved'|'download_requested'|'cancelled'>{
 const picker=(window as any).showSaveFilePicker;
 if(picker){
  let handle:any;
  try{handle=await picker({suggestedName:name});}catch(error){if((error as Error).name==='AbortError')return 'cancelled';throw error;}
  const writer=await handle.createWritable();
  try{await writer.write(blob);await writer.close();const saved=await handle.getFile();if(saved.size!==blob.size)throw Error('保存后文件大小不一致');return 'saved';}
  catch(error){await writer.abort().catch(()=>{});throw error;}
 }
 const link=document.createElement('a'),url=URL.createObjectURL(blob);link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),60000);
 return 'download_requested';
}
export const unavailableLipPort:LipPort={kind:'browser',available:false,reason:'当前宿主尚未开放 legacy 离线任务，请使用本地 M05 分析入口。候选预览不能替代正式结果。',
 async analyze(){throw Error(this.reason);},async save(){throw Error(this.reason);}};

type Task=<T=any>(body:unknown)=>Promise<T>;
function encoded(bytes:ArrayBuffer){const data=new Uint8Array(bytes);let binary='';for(let i=0;i<data.length;i+=32768)binary+=String.fromCharCode(...data.subarray(i,i+32768));return btoa(binary);}
const errorMessages:Record<string,string>={m05_runtime_not_admitted:"当前 M05 科学运行环境尚未通过准入。",m05_memory_budget:"唇形任务超过本地内存预算，未发布部分结果。",m05_timeout:"唇形任务超过 30 分钟期限，已回收计算进程。",m05_output_budget:"结果超过 512 MB 预算，请分段分析并保留原始时间信息。",m05_execution_failed:"唇形任务未完成，请检查本地任务诊断及视频编码。",m05_legacy_runtime_version_mismatch:"科学依赖版本与冻结版本不一致，已拒绝生成正式结果。",m05_legacy_model_hash_mismatch:"FaceMesh 模型哈希不匹配，已拒绝运行。",m05_nonmonotonic_decoded_pts:"视频时间戳不递增，无法生成正式时间轴。",m05_missing_decoded_pts:"视频缺少实际解码时间戳，不能用标称帧率替代。",m05_unsupported_display_rotation:"此视频的旋转元数据尚未通过迁移验证。",m05_unsupported_display_reflection:"此视频含镜像显示矩阵，暂不进行正式测量。"};
export async function desktopM05(transport:Task,choose:()=>Promise<{id:string}|null>):Promise<LipPort>{
 const task:Task=async body=>{try{return await transport(body);}catch(e){throw Error(errorMessages[(e as Error).message]??(e as Error).message);}};
 const support=await task({op:'m05_catalog'});
 async function run(input:{asset_id:string;sha256:string},name:string,config:OfflineConfig,signal:AbortSignal,progress:(s:string)=>void,extra:Record<string,unknown>={},existing?:JobView):Promise<LipResult>{
  if(signal.aborted)throw Error('任务已取消');
  let job=existing??await task<JobView>({op:'m05_create',body:{schema_version:'m05/1',project_id:'00000000-0000-4000-8000-000000000001',idempotency_key:crypto.randomUUID(),video:input,config:{...config,...extra}}});
  const cancel=()=>{void task({op:'cancel_job',id:job.id}).catch(()=>{});};signal.addEventListener('abort',cancel,{once:true});
  try{
   if(signal.aborted){cancel();throw Error('任务已取消');}
   while(!['succeeded','failed','cancelled','interrupted'].includes(job.state)){
    if(signal.aborted)throw Error('任务已取消，后台正在回收本任务进程');
    progress(job.state==='running'?'按解码 PTS 逐帧分析；较长视频可慢于实时':'等待本地任务');
    await new Promise(resolve=>setTimeout(resolve,300));job=await task<JobView>({op:'job',id:job.id});
   }
   if(job.state!=='succeeded'||!job.result_manifest||!('files'in job.result_manifest))throw Error(errorMessages[job.error_code??'']??job.error_code??'分析未完成');
   const files=job.result_manifest.files,preview=files.find(f=>f.name==='preview.json');if(!preview||preview.size_bytes>16_000_000)throw Error('结果预览不完整');
   const value=await task<{base64:string;sha256:string}>({op:'result',job:job.id,id:preview.id}),bytes=Uint8Array.from(atob(value.base64),c=>c.charCodeAt(0));
   if(await sha256(bytes.buffer)!==preview.sha256)throw Error('预览校验失败');
   const data=JSON.parse(new TextDecoder().decode(bytes));
   return {id:job.id,name,backend:data.backend,rows:data.rows,metadata:data.metadata,files:files.map(f=>({id:f.id,name:f.name,bytes:f.size_bytes,sha256:f.sha256}))};
  }finally{signal.removeEventListener('abort',cancel);}
 }
 const port:LipPort={kind:'desktop',available:support.available,reason:support.available?undefined:'M05 专用运行环境尚未准入，请从 Start-M05-Workbench.ps1 启动。',
  async analyze(file,config,signal,progress){
   if(file.size<=0||file.size>128_000_000)throw Error('视频须为 1 字节至 128 MB；更长输入可使用流式本地 CLI。');
   const upload=await task({op:'m05_upload_begin',name:file.name,size:file.size});
   let input:{asset_id:string;sha256:string};
   try{
   for(let offset=0;offset<file.size;offset+=262144){if(signal.aborted)throw Error('输入复制已取消，未提交分析任务');progress(`本地复制 ${Math.round(offset/file.size*100)}%`);await task({op:'m05_upload_block',id:upload.id,offset,base64:encoded(await file.slice(offset,offset+262144).arrayBuffer())});}
   input=await task({op:'m05_upload_finish',id:upload.id});
   }catch(error){await task({op:'m05_upload_abort',id:upload.id}).catch(()=>{});throw error;}
   return run(input,file.name,config,signal,progress);
  },
  async save(result,offset,action){const grant=await choose();if(!grant)return false;const resultSave=await task({op:'m05_save',job:result.id,directory:grant.id,offset,action});return resultSave.saved===true;},
  async history(){return (await task<JobView[]>({op:'m05_history'})).map(j=>({id:j.id,name:new Date(j.created_at*1000).toLocaleString()+' · '+j.id.slice(0,8)}));},
  async load(id){const job=await task<JobView>({op:'job',id});return run({asset_id:'',sha256:''},id.slice(0,8),{filter_enabled:true,cutoff_hz:15},new AbortController().signal,()=>{},{},job);},
  async exportAnimation(result,format,quality,signal){const config={filter_enabled:result.metadata.config.filter_enabled,cutoff_hz:result.metadata.config.cutoff_hz,animation:format,quality,offset:result.metadata.timing?.lip_manual_offset??0};
   const job=await task<JobView>({op:'m05_repeat',job:result.id,config});const rendered=await run({asset_id:'',sha256:''},result.name,config,signal,()=>{},{},job);return port.save(rendered,config.offset,'apply');},
  async saveLocal(name,blob){const grant=await choose();if(!grant)return false;const session=await task({op:'m05_save_begin',directory:grant.id,name,size:blob.size});for(let offset=0;offset<blob.size;offset+=262144)await task({op:'m05_save_block',id:session.id,offset,base64:encoded(await blob.slice(offset,offset+262144).arrayBuffer())});return (await task({op:'m05_save_finish',id:session.id})).saved===true;}
 };return port;
}
