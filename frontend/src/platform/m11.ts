import type {M11Port,CorpusItem,Catalog,Grant} from '../modules/mfa/port.ts';
import type {JobView} from './research.ts';
import {sha256} from './research.ts';
import {pairFiles} from '../modules/mfa/state.ts';

const messages:Record<string,string>={
 m11_runtime_missing:'运行环境已不存在，请重新选择并检查。',m11_model_missing:'登记模型或词典已不存在，请重新检查并导入资源。',m11_version_mismatch:'当前仅验证 MFA 3.3.8，请选择该固定版本，不会自动升级环境。',m11_bootstrap_missing:'主程序缺少 MFA 启动文件，请使用本轮开发入口。',
 m11_install_path_too_long:'组件安装路径过长，Windows 原生库无法可靠加载。请使用说明中的较短便携组件目录，原可用版本保留。',
 m11_transcript_tiers:'TextGrid 缺少可用转写层，或有多份 words 层无法确定。请提供独立转写区间层、唯一 words 层或同名 LAB / TXT。phones 层不能作为词转写。',
 m11_no_alignments:'当前模型与默认搜索参数未能得到有效对齐。请检查转写与音频，不会自动修改 Beam、模型或截断音频。',
 m11_oov_words:'转写中有词典未收录的词。请核对汉字 / 拼音与声调格式，并在下方 MFA 日志查看具体词；任务未发布完整结果。',
 m11_model_mismatch:'声学模型与词典音素集不匹配。',m11_dictionary_mismatch:'词典格式或内容不适用于当前模型。',
 m11_waiting_verified_node:'等待已通过 MFA 版本、模型和资源验证的计算节点；当前服务器回退未开放。',
 m11_download_not_published:'固定版本组件下载清单尚未发布。可导入可信离线包，或检查已有 auto_alignment 环境。',
 m11_timeout:'对齐达到 180 秒期限，已停止本任务进程树。本机部分输出与诊断保留，未作为完整结果发布。',
 m11_component_not_registered:'所选组件或模型尚未登记，请先检查运行环境与模型。',
 m11_temp_budget:'临时文件达到当前 512 MB 预算，任务已停止；请使用已验证的较小输入范围。',
 m11_audio_budget:'当前单文件支持上限为 120 秒、单/双声道。超范围输入未截断。',
 m11_self_test_required:'此组件需要完成实际小任务检查。',m11_feature_failed:'MFA 特征提取失败，请检查运行环境和音频格式。',
 m11_execution_failed:'MFA 执行失败。输入和配置保留，可检查组件后重试。',
 m11_probe_word_unavailable:'当前公开合成检查支持词典中的 a、啊 或 a1。此词典未包含这些探针词条，未登记为可执行。',
 m11_process_crashed:'MFA 子进程异常退出，已回收本任务进程树。',m11_runtime_changed:'运行环境发生变化，请重新检查。',
 m11_model_changed:'模型或词典发生变化，请重新检查。',m11_missing_transcript:'音频缺少已有转写。',
 m11_manifest_hash_mismatch:'离线清单摘要与独立可信来源不一致，未安装。',
 m11_archive_hash_mismatch:'组件包摘要不匹配，当前版本保留。',m11_component_failed:'组件检查或安装失败，原可用版本保留。',
 progress:'任务阶段更新',failed:'任务失败',
 cancelled:'任务已取消。本机部分输出保留，完整结果未发布。',
 m11_preparing:'准备已授权输入',m11_runtime_checked:'外部运行环境检查通过',m11_aligning:'MFA 正在对齐',m11_exporting:'正在生成 TextGrid',m11_complete:'TextGrid 回读完成',queued:'已进入任务队列',running:'任务运行中',succeeded:'完整结果已发布',cancel_requested:'正在停止任务进程树',
};
export const m11Message=(code:string)=>messages[code]??code;
interface Transport {
 log?:M11Port['log'];
 local:boolean;project:string;catalog():Promise<Catalog>;history():Promise<JobView[]>;
 upload(file:File,role:string,signal:AbortSignal):Promise<{asset_id:string;sha256:string}>;
 create(body:unknown):Promise<JobView>;job(id:string):Promise<JobView>;cancel(id:string):Promise<unknown>;
 events(id:string):ReturnType<M11Port['events']>;read(job:string,id:string,sha:string):Promise<ArrayBuffer>;
 save?:M11Port['save'];chooseOutput?:M11Port['chooseOutput'];pick?:M11Port['pick'];corpus?:M11Port['corpus'];component?:M11Port['component'];dictionaryFromGrant?:M11Port['dictionaryFromGrant'];
}
export function m11Port(t:Transport):M11Port {
 return {local:t.local,log:t.log,catalog:t.catalog,history:t.history,job:t.job,cancel:t.cancel,events:t.events,
  async import(files,signal,source){const pairs=pairFiles(files,source),items:CorpusItem[]=[];for(const p of pairs){if(signal.aborted)throw Error('cancelled');const audio=await t.upload(p.audio,'audio',signal),transcript=await t.upload(p.transcript,'transcript',signal);items.push({name:p.name,audio,transcript,transcript_format:/\.textgrid$/i.test(p.transcript.name)?'.TextGrid':/\.txt$/i.test(p.transcript.name)?'.txt':'.lab'});}return items;},
  dictionary:(file,signal)=>t.upload(file,'dictionary',signal),dictionaryFromGrant:t.dictionaryFromGrant,
  create:body=>t.create({...body,project_id:t.project,idempotency_key:crypto.randomUUID(),schema_version:'m11/1'}),
  async download(job,id){const m=job.result_manifest;if(!m||!('files'in m))throw Error('结果不可用。');const f=m.files.find(f=>f.id===id);if(!f)throw Error('结果不可用。');const raw=await t.read(job.id,id,f.sha256);if(raw.byteLength!==f.size_bytes||await sha256(raw)!==f.sha256)throw Error('结果完整性检查失败。');const url=URL.createObjectURL(new Blob([raw]));const a=document.createElement('a');a.href=url;a.download=f.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);},
  save:t.save,chooseOutput:t.chooseOutput,pick:t.pick,corpus:t.corpus,component:t.component};
}
export function serverM11(owner:string,project:string,csrf:()=>string,onInvalid:()=>void,signal:AbortSignal):M11Port {
 async function request(path:string,method='GET',body?:unknown,raw?:ArrayBuffer){const r=await fetch('/api/v1/'+path,{method,signal,credentials:'same-origin',cache:'no-store',headers:{'X-PTB-Account':owner,...(method!=='GET'?{'X-CSRF-Token':csrf()}:{}),...(raw?{'Content-Type':'application/octet-stream'}:body?{'Content-Type':'application/json'}:{})},body:raw??(body?JSON.stringify(body):undefined)});if(!r.ok){if(r.status===401)onInvalid();const e=await r.json().catch(()=>({}));throw Error(m11Message(typeof e.detail==='string'?e.detail:'请求失败。'));}return r;}
 return m11Port({local:false,project,catalog:async()=> (await request('jobs/m11/catalog')).json(),history:async()=> (await (await request('jobs?project_id='+project)).json()).jobs.filter((j:JobView)=>j.operation==='mfa_alignment'),
  async upload(file,_role,abort){const bytes=await file.arrayBuffer(),hash=await sha256(bytes),a=await(await request('uploads','POST',{project_id:project,name:file.name,expected_bytes:bytes.byteLength,idempotency_key:'m11-'+crypto.randomUUID()})).json();for(let offset=0;offset<bytes.byteLength;offset+=262144){if(abort.aborted)throw Error('cancelled');await request(`uploads/${a.id}/blocks?offset=${offset}`,'PUT',undefined,bytes.slice(offset,offset+262144));}const ready=await(await request('uploads/'+a.id+'/finalize','POST',{sha256:hash})).json();return {asset_id:ready.id,sha256:ready.sha256};},
  create:async body=>(await request('jobs/m11/create','POST',body)).json(),job:async id=>(await request('jobs/'+id)).json(),cancel:async id=>(await request('jobs/'+id+'/cancel','POST')).json(),events:async id=>(await request('jobs/'+id+'/events')).json(),read:async(_job,id)=>(await request('assets/'+id+'/content')).arrayBuffer()});
}
