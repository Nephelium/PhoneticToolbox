import type {M14Port,Source,Preview,Result,Options,Inspection} from '../modules/phonology-induction/port.ts';
import type {JobView} from './research.ts';
import {sha256} from './research.ts';
interface Transport {project:string;upload(file:File,signal:AbortSignal):Promise<Source>;create(body:unknown):Promise<JobView>;job(id:string):Promise<JobView>;cancel(id:string):Promise<unknown>;read(job:string,id:string,sha:string):Promise<ArrayBuffer>;save?(result:Result):Promise<boolean>}
const errors:Record<string,string>={m14_timeout:'导出超过 60 秒执行期限，任务已停止；请分表处理后重试。',m14_unsupported_format:'仅支持 XLSX、XLS、CSV、TXT 和 TSV。',m14_cell_output_budget:'同音字矩阵有单格超过 Excel 的 32,767 字符上限，请分表处理。',m14_invalid_cell:'单元格过长或含非法控制字符。',m14_expanded_budget:'Excel 展开内容超过读取预算。',m14_column_budget:'调查表列数超过 64 列，请保留所需列。',m14_output_budget:'三份结果超过 16 MB 输出预算。',m14_execution_failed:'文档生成失败，原编辑与结果保留，请检查输入后重试。',m14_decode_failed:'文件无法解析，请核对格式及 UTF-8 编码。',m14_no_valid_rows:'没有有效记录，请检查字头、音标两列及跳首行设置。',m14_empty_input:'文件为空。',m14_missing_columns:'调查字表至少需要字头、音标两列。',m14_formula_input:'调查字表含公式，请转换为明确文本值后导入。',m14_runtime_unavailable:'当前平台尚未通过 M14 任务运行资格检查。',m14_input_budget:'文件超过 2 MB 读取预算。',m14_row_budget:'调查字表超过 10,000 行预算。',m14_output_shape_budget:'声韵矩阵超过 20,000 格，请分表处理。',input_unavailable:'源文件已失效、发生变化或临近到期，请重新导入。',cancelled:'操作已取消，原编辑与结果保留。',execution_failed:'任务执行失败，可检查输入后重试。',quota_exceeded:'项目剩余空间不足，请清理后重试。'};
Object.assign(errors,{m14_unsupported_format:'支持 XLSX、XLS、CSV、TSV、TXT 和 DOCX 表格。',m14_column_selection:'字头、IPA 与备注请选择不同的有效列。',m14_start_row:'开始行须为 1–10,001 的整数。',m14_table_index:'没有此工作表或表格，请重新选择。',m14_encoding:'请选择支持的文本编码。',m14_delimiter:'请选择支持的分隔符。',m14_start_inside_record:'开始行位于引号内的跨行记录中，请选择该记录的起始行。',m14_decode_failed:'文件无法解析，请核对格式、编码和分隔符。',m14_no_valid_rows:'没有有效记录，请核对列号、开始行及 IPA。'});
export const m14Error=(code:string)=>errors[code]??code;
function download(raw:ArrayBuffer,name:string){const url=URL.createObjectURL(new Blob([raw]));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
function zip(files:{name:string;raw:ArrayBuffer}[]):ArrayBuffer {
 // ZIP STORE with UTF-8 filenames; bounded to exactly the verified result set.
 const parts:Uint8Array[]=[],central:Uint8Array[]=[];let offset=0;
 const crc=(v:Uint8Array)=>{let c=0xffffffff;for(const b of v){c^=b;for(let i=0;i<8;i++)c=(c>>>1)^((c&1)?0xedb88320:0);}return (c^0xffffffff)>>>0;};
 for(const f of files){const name=new TextEncoder().encode(f.name),bytes=new Uint8Array(f.raw),hash=crc(bytes),local=new Uint8Array(30+name.length),v=new DataView(local.buffer);v.setUint32(0,0x04034b50,true);v.setUint16(4,20,true);v.setUint16(6,0x800,true);v.setUint32(14,hash,true);v.setUint32(18,bytes.length,true);v.setUint32(22,bytes.length,true);v.setUint16(26,name.length,true);local.set(name,30);parts.push(local,bytes);const c=new Uint8Array(46+name.length),d=new DataView(c.buffer);d.setUint32(0,0x02014b50,true);d.setUint16(4,20,true);d.setUint16(6,20,true);d.setUint16(8,0x800,true);d.setUint32(16,hash,true);d.setUint32(20,bytes.length,true);d.setUint32(24,bytes.length,true);d.setUint16(28,name.length,true);d.setUint32(42,offset,true);c.set(name,46);central.push(c);offset+=local.length+bytes.length;}
 const size=central.reduce((n,c)=>n+c.length,0),end=new Uint8Array(22),e=new DataView(end.buffer);e.setUint32(0,0x06054b50,true);e.setUint16(8,files.length,true);e.setUint16(10,files.length,true);e.setUint32(12,size,true);e.setUint32(16,offset,true);const all=new Uint8Array(offset+size+22);let at=0;for(const p of [...parts,...central,end]){all.set(p,at);at+=p.length;}return all.buffer;
}
export function m14Port(t:Transport):M14Port {
 async function wait(job:JobView,signal:AbortSignal){
  const deadline=Date.now()+180000;
  while(['queued','running','cancel_requested'].includes(job.state)){
   if(signal.aborted){await t.cancel(job.id);throw Error(m14Error('cancelled'));}
   if(Date.now()>deadline){await t.cancel(job.id);throw Error('等待任务超时，已请求取消。');}
   await new Promise(r=>setTimeout(r,120));job=await t.job(job.id);
  }
  if(signal.aborted)throw Error(m14Error('cancelled'));
  if(job.state!=='succeeded')throw Error(m14Error(job.error_code??'execution_failed'));
  if(!job.result_manifest||!('files' in job.result_manifest))throw Error('结果清单不完整。');return {job:job.id,files:job.result_manifest.files} as Result;
 }
 async function read(r:Result,id:string){const f=r.files.find(f=>f.id===id);if(!f)throw Error('结果已失效。');const raw=await t.read(r.job,id,f.sha256);if(raw.byteLength!==f.size_bytes||await sha256(raw)!==f.sha256)throw Error('文件完整性校验失败。');return raw;}
 async function upload(file:File,signal:AbortSignal){if(!file.size)throw Error(m14Error('m14_empty_input'));if(file.size>2_000_000)throw Error(m14Error('m14_input_budget'));const source=await t.upload(file,signal);if(signal.aborted)throw Error(m14Error('cancelled'));return source;}
 async function json(source:Source,options:Options,action:'inspect'|'preview',signal:AbortSignal){const r=await wait(await t.create({project_id:t.project,idempotency_key:crypto.randomUUID(),table:{asset_id:source.asset_id,sha256:source.sha256},config:{action,...options}}),signal);return JSON.parse(new TextDecoder().decode(await read(r,r.files[0].id)));}
 async function analyze(source:Source,options:Options,signal:AbortSignal){const preview=await json(source,options,'preview',signal) as Preview;if(preview.input_sha256!==source.sha256)throw Error('输入校验值不一致。');return preview;}
 return {
  async inspect(file,options,signal){const source=await upload(file,signal);return {source,inspection:await json(source,options,'inspect',signal) as Inspection};},
  async inspectSource(source,options,signal){return await json(source,options,'inspect',signal) as Inspection;},
  analyze,
  async import(file,options,signal){if(file.size===0)throw Error(m14Error('m14_empty_input'));if(file.size>2_000_000)throw Error(m14Error('m14_input_budget'));const source=await t.upload(file,signal);if(signal.aborted)throw Error(m14Error('cancelled'));const r=await wait(await t.create({project_id:t.project,idempotency_key:crypto.randomUUID(),table:{asset_id:source.asset_id,sha256:source.sha256},config:{action:'preview',...options}}),signal);const preview=JSON.parse(new TextDecoder().decode(await read(r,r.files[0].id))) as Preview;if(preview.input_sha256!==source.sha256)throw Error('输入校验值不一致。');return {source,preview};},
  async generate(source,options,settings,font,signal){return wait(await t.create({project_id:t.project,idempotency_key:crypto.randomUUID(),table:{asset_id:source.asset_id,sha256:source.sha256},config:{action:'export',...options,settings,font}}),signal);},
  async save(result){if(t.save)return t.save(result);const files=[];for(const f of result.files)files.push({name:f.name,raw:await read(result,f.id)});download(zip(files),'同音字表_完整结果.zip');return true;},
  async download(result,id){const f=result.files.find(f=>f.id===id)!;download(await read(result,id),f.name);}
 };
}
export function serverM14(owner:string,project:string,csrf:()=>string,onInvalid:()=>void,signal:AbortSignal):M14Port {
 async function request(path:string,method='GET',body?:unknown,raw?:ArrayBuffer){const r=await fetch('/api/v1/'+path,{method,signal,credentials:'same-origin',cache:'no-store',headers:{'X-PTB-Account':owner,...(method!=='GET'?{'X-CSRF-Token':csrf()}:{}),...(raw?{'Content-Type':'application/octet-stream'}:body?{'Content-Type':'application/json'}:{})},body:raw??(body?JSON.stringify(body):undefined)});if(!r.ok){if(r.status===401)onInvalid();const e=await r.json().catch(()=>({}));throw Error(m14Error(typeof e.detail==='string'?e.detail:'请求失败，请检查输入及项目状态。'));}return r;}
 return m14Port({project,
  async upload(file,abort){const bytes=await file.arrayBuffer(),hash=await sha256(bytes),a=await(await request('uploads','POST',{project_id:project,name:file.name,expected_bytes:bytes.byteLength,idempotency_key:'m14-'+crypto.randomUUID()})).json();for(let offset=0;offset<bytes.byteLength;offset+=262144){if(abort.aborted)throw Error(m14Error('cancelled'));await request(`uploads/${a.id}/blocks?offset=${offset}`,'PUT',undefined,bytes.slice(offset,offset+262144));}const ready=await(await request('uploads/'+a.id+'/finalize','POST',{sha256:hash})).json();return {asset_id:ready.id,sha256:ready.sha256,name:file.name};},
  async create(body){return(await request('jobs/m14/create','POST',body)).json();},async job(id){return(await request('jobs/'+id)).json();},async cancel(id){return(await request('jobs/'+id+'/cancel','POST')).json();},async read(_job,id){return(await request('assets/'+id+'/content')).arrayBuffer();}
 });
}
