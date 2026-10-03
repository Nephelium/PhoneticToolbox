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
  const audio=['extract','resynthesize'].includes(action)&&file?await t.source(file):undefined;if(signal.aborted)throw Error('操作已取消');
  const text=exportParams(config);if(new TextEncoder().encode(text).length>2_000_000)throw Error('完整参数超过 2 MB');
  const parameters=await t.parameters(text,signal);if(signal.aborted)throw Error('操作已取消');
  let job=await t.create({project_id:t.project,idempotency_key:crypto.randomUUID(),action,parameters,audio});progress(job);
  const deadline=Date.now()+180000;
  while(['queued','running','cancel_requested'].includes(job.state)){
   if(signal.aborted||Date.now()>deadline){await t.cancel(job.id);throw Error(signal.aborted?'操作已取消':'等待超时，已请求取消');}
   await new Promise(resolve=>setTimeout(resolve,120));job=await t.job(job.id);progress(job);
  }
  if(signal.aborted)throw Error('操作已取消');
  if(job.state!=='succeeded')throw Error(m06Error(job.error_code));
  if(!job.result_manifest||!('files' in job.result_manifest))throw Error('结果清单不完整');
  const base={job:job.id,files:job.result_manifest.files};
  const result:Result={...base,metadata:JSON.parse(new TextDecoder().decode(await read(base,'m06.ptb.json')))};
  if(['synthesize','resynthesize'].includes(action))result.wav=await read(result,'synthesis.wav');return result;
 },async download(result,name){if(t.save&&name==='synthesis.wav'){if(!await t.save(result.job))throw Error('未保存，合成工件仍保留');return;}downloadBytes(await read(result,name),name);}};
}
function m06Error(code?:string|null){const messages:Record<string,string>={
 m06_reaper_unavailable:'当前环境没有可用的已验证 REAPER，请选择 Praat 或检查本机运行环境。',
 m06_reaper_failed:'REAPER 提取失败，原参数保留。可调整 F0 范围或选择 Praat 后重试。',
 m06_world_unavailable:'当前运行环境未安装 WORLD 组件。可暂用 Klatt 或 PSOLA。',
 m06_world_version:'WORLD 组件版本不匹配，需要本项目锁定的 PyWORLD 0.3.5。',
 m06_world_input_range:'WORLD / Harvest 支持 16–48 kHz 录音，F0 范围须在 40–1000 Hz 内。请调整范围或选择其他方法。',
 m06_psola_f0_range:'PSOLA 的 F0 分析范围须在 40–1000 Hz 内。',
 m06_psola_no_voiced:'所选算法未检出有声帧，PSOLA 无法应用编辑音高。请调整 F0 范围或算法，或选择保留原 F0。',
 m06_resynthesis_pitch_range:'重合成的有声 F0 须在 40–1000 Hz 内，请调整 F0 曲线。',
 m06_resynthesis_duration:'重合成目标时长应为原录音的 0.5–2 倍，且不超过 10 秒 / 480,000 样本。',
 m06_world_matrix_budget:'当前录音和 F0 下限所需谱矩阵超过预算，请缩短录音或提高合理的 F0 下限。',
 m06_resynthesis_source_mismatch:'源音频与参数记录不一致，请重新关联原录音或提取参数。',
 m06_method_action_mismatch:'当前合成方法与操作不匹配，请重新选择合成方法。',
 };return code&&messages[code]||code||'合成任务失败';}
