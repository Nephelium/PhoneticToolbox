import type {M11Port,CorpusItem,Catalog,Grant} from '../modules/mfa/port.ts';
import type {JobView} from './research.ts';
import {sha256} from './research.ts';
import {pairFiles} from '../modules/mfa/state.ts';

const messages:Record<string,string>={
 m11_download_not_uploaded:'服务器尚未提供此版本组件。可从发布者网盘下载后选择组件压缩包安装。',
 m11_download_failed:'组件下载失败，请检查网络后重试；也可从网盘下载并导入压缩包。',
 m11_download_incomplete:'组件尚未下载完整，请重试以继续下载。',
 m11_download_resume_rejected:'服务器不支持从上次位置续传。请从网盘下载完整组件压缩包后导入。',
 m11_delivery_folder_required:'请选择组件压缩包解压后的根文件夹，其中应包含 env 和 models。旧版环境请从高级选项导入。',
 m11_folder_files_changed:'组件文件夹有缺失或额外文件，请重新解压原始组件包后选择文件夹。',
 m11_file_hash_mismatch:'组件文件已损坏或被修改，请重新下载原始组件包。',
 m11_component_busy:'正在安装另一个 MFA 组件，请等待完成。',
 m11_component_disk_full:'安装空间不足，请释放磁盘空间后重试。当前组件保留。',
 m11_runtime_missing:'运行环境已不存在，请重新选择并检查。',m11_model_missing:'登记模型或词典已不存在，请重新检查并导入资源。',m11_version_mismatch:'当前支持 MFA 3.3.8，请选择此版本。',m11_bootstrap_missing:'MFA 自动标注暂时无法启动。可在关于页反馈此问题。',
 m11_install_path_too_long:'组件安装路径过长，Windows 原生库无法可靠加载。请使用说明中的较短便携组件目录，原可用版本保留。',
 m11_transcript_tiers:'TextGrid 缺少可用转写层，或有多份 words 层无法确定。请提供独立转写区间层、唯一 words 层或同名 LAB / TXT。phones 层不能作为词转写。',
 m11_no_alignments:'当前模型与默认搜索参数未能得到有效对齐。请检查转写与音频，不会自动修改 Beam、模型或截断音频。',
 m11_oov_words:'转写中有词典未收录的词。请核对汉字 / 拼音与声调格式，并在下方 MFA 日志查看具体词；任务未发布完整结果。',
 m11_model_mismatch:'声学模型与词典音素集不匹配。',m11_dictionary_mismatch:'词典格式或内容不适用于当前模型。',
 m11_waiting_verified_node:'正在等待可用的计算服务。',
 m11_web_disabled:'网页版暂不支持MFA在线对齐，请下载桌面应用。',
 m11_download_not_published:'组件暂时无法在线下载，可导入离线组件包。',
 m11_timeout:'对齐达到 180 秒期限，已停止本任务进程树。本机部分输出与诊断保留，未作为完整结果发布。',
 m11_component_not_registered:'所选组件或模型尚未登记，请先检查运行环境与模型。',
 m11_temp_budget:'临时文件达到当前 512 MB 预算，任务已停止；请使用已验证的较小输入范围。',
 m11_audio_budget:'当前单文件支持上限为 120 秒、单/双声道。超范围输入未截断。',
 m11_self_test_required:'此组件需要完成实际小任务检查。',m11_feature_failed:'MFA 特征提取失败，请检查运行环境和音频格式。',
 m11_execution_failed:'MFA 执行失败。输入和配置保留，可检查组件后重试。',
 m11_probe_word_unavailable:'此词典缺少组件检查所需的 a、啊 或 a1 词条，请使用配套词典。',
 m11_process_crashed:'MFA 子进程异常退出，已回收本任务进程树。',m11_runtime_changed:'运行环境发生变化，请重新检查。',
 m11_model_changed:'模型或词典发生变化，请重新检查。',m11_missing_transcript:'音频缺少已有转写。',
 m11_manifest_hash_mismatch:'离线清单摘要与独立可信来源不一致，未安装。',
 m11_archive_hash_mismatch:'组件包摘要不匹配，当前版本保留。',m11_component_failed:'组件检查或安装失败，原可用版本保留。',
 progress:'任务阶段更新',failed:'任务失败',
 cancelled:'任务已取消。本机部分输出保留，完整结果未发布。',
 m11_preparing:'正在准备音频与转写',m11_runtime_checked:'外部运行环境检查通过',m11_aligning:'MFA 正在对齐',m11_exporting:'正在生成 TextGrid',m11_complete:'TextGrid 已生成',queued:'已进入任务队列',running:'任务运行中',succeeded:'完整结果已发布',cancel_requested:'正在停止任务',
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
