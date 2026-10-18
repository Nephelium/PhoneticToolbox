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
export const unavailableLipPort:LipPort={kind:'browser',available:false,reason:'当前环境支持实时预览，离线分析请使用桌面版。',
 async analyze(){throw Error(this.reason);},async save(){throw Error(this.reason);}};

type Task=<T=any>(body:unknown)=>Promise<T>;
function encoded(bytes:ArrayBuffer){const data=new Uint8Array(bytes);let binary='';for(let i=0;i<data.length;i+=32768)binary+=String.fromCharCode(...data.subarray(i,i+32768));return btoa(binary);}
const errorMessages:Record<string,string>={m05_runtime_not_admitted:"唇形分析暂时无法运行。可在关于页反馈此问题。",m05_memory_budget:"唇形任务超过本地内存预算，未发布部分结果。",m05_timeout:"唇形任务超过 30 分钟期限，已回收计算进程。",m05_output_budget:"结果超过 512 MB 预算，请分段分析并保留原始时间信息。",m05_execution_failed:"唇形分析未完成，请换用其他视频重试。若仍有问题，可在关于页反馈。",m05_legacy_runtime_version_mismatch:"唇形分析暂时无法运行。可在关于页反馈此问题。",m05_legacy_model_hash_mismatch:"唇形分析暂时无法运行。可在关于页反馈此问题。",m05_nonmonotonic_decoded_pts:"视频时间戳不递增，无法生成正式时间轴。",m05_missing_decoded_pts:"视频缺少逐帧时间戳，请换用保留完整时间信息的视频。",m05_unsupported_display_rotation:"暂不支持此视频的旋转信息，请先转为正常方向的视频。",m05_unsupported_display_reflection:"此视频含镜像显示矩阵，暂不进行正式测量。"};
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
   return {id:job.id,name:data.metadata.input_name??name,backend:data.backend,rows:data.rows,metadata:data.metadata,files:files.map(f=>({id:f.id,name:f.name,bytes:f.size_bytes,sha256:f.sha256}))};
  }finally{signal.removeEventListener('abort',cancel);}
 }
 const port:LipPort={kind:'desktop',available:support.available,reason:support.available?undefined:'唇形分析暂时无法运行。可在关于页反馈此问题。',
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
  async save(result,offset,action){const grant=await choose();if(!grant)return false;const response=await task({op:'m05_save',job:result.id,directory:grant.id,offset,action});return response.saved===true;},
  async saveBatch(items,action,options,onSaved){const grant=await choose();if(!grant)return false;for(const item of items){const response=await task({op:'m05_save',job:item.result.id,directory:grant.id,offset:item.offset,action,options});if(!response.saved)return false;onSaved(item.result.id);}return true;},
  async history(){return task({op:'m05_history'});},
  async load(id){const job=await task<JobView>({op:'job',id});return run({asset_id:'',sha256:''},id.slice(0,8),{filter_enabled:true,cutoff_hz:15},new AbortController().signal,()=>{},{},job);},
  async sourceVideo(result){const info=await task({op:'m05_source_info',job:result.id});const parts:Uint8Array<ArrayBuffer>[]=[];for(let offset=0;offset<info.size_bytes;offset+=262144){const response=await task({op:'m05_source_block',job:result.id,offset});parts.push(Uint8Array.from(atob(response.base64),c=>c.charCodeAt(0)));}const blob=new Blob(parts,{type:info.name.toLowerCase().endsWith('.webm')?'video/webm':'video/mp4'});if(blob.size!==info.size_bytes||await sha256(await blob.arrayBuffer())!==info.sha256)throw Error('历史视频校验失败');return blob;},
  async playbackVideo(result){
   const info=await task({op:'m05_playback_info',job:result.id});
   try{
    if(!Number.isSafeInteger(info.size_bytes)||info.size_bytes<=0||info.size_bytes>128_000_000||!Number.isFinite(info.origin_s))throw Error('回放视频信息无效');
    const parts:Uint8Array<ArrayBuffer>[]=[];
    for(let offset=0;offset<info.size_bytes;offset+=262144){const response=await task({op:'m05_playback_block',token:info.token,offset});const part=Uint8Array.from(atob(response.base64),c=>c.charCodeAt(0));if(part.length!==Math.min(262144,info.size_bytes-offset))throw Error('回放视频读取不完整');parts.push(part);}
    const blob=new Blob(parts,{type:'video/webm'});if(await sha256(await blob.arrayBuffer())!==info.sha256)throw Error('回放视频校验失败');
    return {blob,origin_s:info.origin_s};
   }finally{await task({op:'m05_playback_release',token:info.token});}
  },
  replay(result,start){return task({op:'m05_replay',job:result.id,start});},
  audioPreview(result){return task({op:'m05_audio_preview',job:result.id});},
  async inspectRecording(blob){
   const session=await task({op:'m05_inspect_begin',size:blob.size});
   try{for(let offset=0;offset<blob.size;offset+=262144)await task({op:'m05_save_block',id:session.id,offset,base64:encoded(await blob.slice(offset,offset+262144).arrayBuffer())});return (await task({op:'m05_save_finish',id:session.id})).inspection;}
   catch(e){await task({op:'m05_save_abort',id:session.id}).catch(()=>{});throw e;}
  },
  async analyzeRecording(token,config,signal,progress){if(signal.aborted)throw Error('任务已取消');progress('读取已核对的 MP4');const input=await task({op:'m05_recording_input',token});return run(input,'本地原始录制.mp4',config,signal,progress);},
  async saveRecording(name,blob,metadata){
   const grant=await choose();if(!grant)return {saved:false};
   const session=await task({op:'m05_save_begin',directory:grant.id,name,size:blob.size,recording:true,metadata_size:metadata.size});
   try{
    for(const [id,data] of [[session.metadata_id,metadata],[session.id,blob]] as [string,Blob][]){
     for(let offset=0;offset<data.size;offset+=262144)await task({op:'m05_save_block',id,offset,base64:encoded(await data.slice(offset,offset+262144).arrayBuffer())});
     const response=await task({op:'m05_save_finish',id});if(id===session.id)return response;
    }
    throw Error('录制保存未完成');
   }catch(e){await task({op:'m05_save_abort',id:session.id}).catch(()=>{});await task({op:'m05_save_abort',id:session.metadata_id}).catch(()=>{});throw e;}
  },
  async exportAnimation(result,format,quality,signal){if(signal.aborted)throw Error('动画导出已取消');const grant=await choose();if(!grant||signal.aborted)return false;const exportId=crypto.randomUUID();const cancel=()=>{void task({op:'m05_cancel_export',export_id:exportId}).catch(()=>{});};signal.addEventListener('abort',cancel,{once:true});try{const response=await task({op:'m05_export_animation',export_id:exportId,job:result.id,directory:grant.id,format,quality,offset:result.metadata.timing?.lip_manual_offset??0});return response.saved===true;}finally{signal.removeEventListener('abort',cancel);}},
  async saveLocal(name,blob){const grant=await choose();if(!grant)return false;const session=await task({op:'m05_save_begin',directory:grant.id,name,size:blob.size});try{for(let offset=0;offset<blob.size;offset+=262144)await task({op:'m05_save_block',id:session.id,offset,base64:encoded(await blob.slice(offset,offset+262144).arrayBuffer())});return (await task({op:'m05_save_finish',id:session.id})).saved===true;}catch(e){await task({op:'m05_save_abort',id:session.id}).catch(()=>{});throw e;}}
 };return port;
}
