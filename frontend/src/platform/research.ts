import {sharedJobReads,waitForTaskPoll} from './task-poll.ts';
import {BrowserScience} from './browser-science.ts';
import {BrowserJobs} from './browser-jobs.ts';
import {serverPapers} from './papers-server.ts';
import {webVocalPort,type VocalPort} from './vocal.ts';
import type {PapersClient} from './papers.ts';
import type { components } from '../../../contracts/generated/api';
import { decodeWav } from './decode.ts';
import {portableAnnotation,type AnnotationPort} from './annotation.ts';
import {m07Port,type M07Port} from './m07.ts';
import {m06Port} from './m06.ts';
import type {M06Port} from '../modules/speech-synthesis/port.ts';
import {m08Port} from './m08.ts';
import {serverM14} from './m14.ts';
import {browserM14} from './browser-m14.ts';
import {browserM06} from './browser-m06.ts';
import {serverM11} from './m11.ts';
import type {LipPort} from '../modules/lip-extraction/port.ts';
import type {M11Port} from '../modules/mfa/port.ts';
import type {M14Port} from '../modules/phonology-induction/port.ts';
import type {M08Port} from '../modules/pitch-manipulation/port.ts';
import {recordingSpectrum,type RecordingServer,type RecordingAsset} from './recording-server.ts';
export type Tier=components['schemas']['TextGridPreview']['tiers'][number];
export type Spectrogram=components['schemas']['SpectrogramPreview'];
export type ParameterTable=components['schemas']['ParameterTable'] & {streamed?:boolean;duration_s?:number;tracks?:Record<string,[number,number|string|null][]>;stats?:Record<string,{count:number;mean:number;min:number|null;max:number|null}>;metadata?:Record<string,unknown>};
export interface ParameterView {start:number;end:number;width:number;parameters:string[]}
export type ReconstructionConfig=components['schemas']['Spec2WavConfig'];
export type ReconstructionPreview=components['schemas']['Spec2WavPreview'];
export type EggTaskConfig=components['schemas']['EggTaskConfig'];
export type LpcTaskConfig=components['schemas']['LpcTaskConfig'];
export type LpcSpectrumData=components['schemas']['LpcSpectrumData'];
export type EggPreviewData=components['schemas']['EggPreviewData'];
export type FigureFontSnapshot=components['schemas']['FigureFontSnapshot'];
export type FontPreflight=components['schemas']['FontPreflight'];
export interface SpectrogramView {channel:number;start:number;end:number;width:number}
export interface ResearchFile { id:string; name:string; kind:'audio'|'textgrid'|'lip'|'lip_pickle'|'parameter'|'image'|'lab'; size:number; sha256?:string; expiresAt?:number }
export interface DirectoryGrant { id:string; label:string; purpose:'input'|'output'|'association' }
export type BatchView=components['schemas']['BatchView'];
export type JobView=components['schemas']['JobView'] & {execution?:'browser'};
export type BatchConfig=components['schemas']['AcousticConfigSnapshot'];
export interface BatchSelection {audio:ResearchFile;textgrid?:ResearchFile|null;lip?:ResearchFile|null;legacy_result?:ResearchFile|null;parent_result?:{asset_id:string;sha256:string}}
export interface ResearchTasks {
  parent(file:ResearchFile):Promise<{asset_id:string;sha256:string}|null>;
  submit(operation:'acoustic_analysis'|'textgrid_segment',inputs:BatchSelection[],config:BatchConfig|null,layer:string|null,key:string):Promise<BatchView>;
  list():Promise<BatchView[]>;get(id:string):Promise<BatchView>;cancel(id:string):Promise<BatchView>;job(id:string):Promise<JobView>;
  retry(id:string,key:string):Promise<JobView>;
  save?(id:string,directory:string,besideSources?:boolean):Promise<{count:number;saved:string[]}>;
  download?(id:string,name:string):Promise<void>;
  reconstruct?(file:ResearchFile,config:ReconstructionConfig,key:string):Promise<JobView>;
  reconstructionPreview?(file:ResearchFile,config:ReconstructionConfig):Promise<ReconstructionPreview>;
  vocoderStatus?():Promise<{ready:boolean;pipeline_ready?:boolean;pipeline_available?:boolean;pipeline_download_bytes?:number;available:boolean;model:string;download_bytes:number;installed_bytes:number;gpu_ready?:boolean;gpu_name?:string|null;python_name?:string|null;progress:{stage:string;completed:number;total:number}}>;
  chooseInferencePython?():Promise<{id:string;name:string}|null>;
  checkInferencePython?(grant:string):Promise<{cuda:boolean;gpu_name?:string;model_checked?:boolean}>;
  downloadVocoder?():Promise<unknown>;
  egg?(file:ResearchFile,config:EggTaskConfig,key:string):Promise<JobView>;
  eggJobs?():Promise<JobView[]>;
  eggFonts?(font:FigureFontSnapshot):Promise<FontPreflight>;
  lpc?(file:ResearchFile,textgrid:ResearchFile|null,config:LpcTaskConfig,key:string):Promise<JobView>;
  lpcJobs?():Promise<JobView[]>;
  lpcFonts?(font:FigureFontSnapshot):Promise<FontPreflight>;
  reconstructions?():Promise<JobView[]>;
  cancelJob?(id:string):Promise<JobView>;
  saveJob?(id:string,directory:string):Promise<{count:number;saved:string[]}>;
  result?(job:string,id:string,sha256:string):Promise<ArrayBuffer>;
}
export interface ResearchFiles {
  feedbackCsrf?:()=>string;
  associatedParameters?(file:ResearchFile):Promise<ResearchFile|null>;
  vocal?:(key:string)=>VocalPort;
  papers?:PapersClient;
  kind:'desktop'|'server'|'preview';
  tasks?:ResearchTasks;
  annotation?:AnnotationPort;
  m06?:M06Port;
  m07?:M07Port;
  m08?:M08Port;
  m14?:M14Port;
  m11?:M11Port;
  m05?:LipPort;
  recording?:RecordingServer;
  choose?(purpose:DirectoryGrant['purpose']):Promise<DirectoryGrant|null>;
  add?(files:File[]):void;
  upload?(files:File[],progress:(value:number)=>void):Promise<void>;
  list(directory?:string,recursive?:boolean):Promise<ResearchFile[]>;
  previewAudio?(file:ResearchFile):Promise<{buffer:ArrayBuffer;sha256:string;sourceDuration:number;previewNote:string}>;
  read(file:ResearchFile,signal?:AbortSignal):Promise<{buffer:ArrayBuffer;sha256:string}>;
  textgrid(file:ResearchFile):Promise<{sha256:string;tiers:Tier[]}>;
  spectrogram?(file:ResearchFile,view:SpectrogramView):Promise<Spectrogram>;
  parameters?(file:ResearchFile,view?:ParameterView):Promise<ParameterTable>;
  eggPreview?:{
    open(file:ResearchFile):Promise<components['schemas']['EggPreviewSession']>;
    update(file:ResearchFile,session:string,config:EggTaskConfig,options?:{autoDb:boolean;needAudio?:boolean}):Promise<components['schemas']['EggInteractiveResult']>;
    close(file:ResearchFile,session:string):Promise<unknown>;
  };
  capture?():Promise<(ResearchFile & {corners?:{x:number;y:number}[]})|null>;
  convertLip?(file:ResearchFile):Promise<{file:ResearchFile;companion_found:boolean}>;
  dispose():void;
}
export interface ResearchContext { key:string; label:string; files:ResearchFiles; ownerId?:string }
export const fileKind=(name:string):ResearchFile['kind']|null=>/\.(wav|mp3|flac)$/i.test(name)?'audio':/\.textgrid$/i.test(name)?'textgrid':/\.lip\.json$/i.test(name)?'lip':/\.lab$/i.test(name)?'lab':/\.(xlsx|ptb\.sqlite3?)$/i.test(name)?'parameter':/\.(png|jpg|jpeg|bmp)$/i.test(name)?'image':null;
export async function audioPreview(files:ResearchFiles,file:ResearchFile,signal?:AbortSignal) {
  const data=await files.read(file,signal);return {asset:await decodeWav(data.buffer,file.name,signal),sha256:data.sha256};
}
// Explicit opt-in for M01/M02; scientific readers keep their original inputs.
export async function researchAudio(files:ResearchFiles,file:ResearchFile,signal?:AbortSignal) {
  if(!files.previewAudio||!file.name.toLowerCase().endsWith('.wav'))return {...await audioPreview(files,file,signal),previewNote:''};
  signal?.throwIfAborted();
  const data=await files.previewAudio(file);signal?.throwIfAborted();
  const asset=await decodeWav(data.buffer,file.name,signal);
  if(!Number.isFinite(data.sourceDuration)||data.sourceDuration<=0||Math.abs(asset.duration-data.sourceDuration)>1/asset.sampleRate+1e-9)throw Error('预览与原音频时间范围不一致。');
  asset.duration=data.sourceDuration;
  return {asset,sha256:data.sha256,previewNote:data.previewNote};
}
export async function sha256(buffer:ArrayBuffer) {return [...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))].map(n=>n.toString(16).padStart(2,'0')).join('');}

export function previewFiles():ResearchFiles {
  const files=new Map<string,File>();
  return {kind:'preview',add(values){for(const f of values){if(!fileKind(f.name))continue;files.set(crypto.randomUUID(),f);}},
    async list(){return [...files].map(([id,f])=>({id,name:f.name,size:f.size,kind:fileKind(f.name)!}));},
    async read(file){const f=files.get(file.id);if(!f||f.size>64_000_000)throw Error('预览限64 MB，请先截取较短音频。');const buffer=await f.arrayBuffer();return {buffer,sha256:await sha256(buffer)};},
    async textgrid(){throw Error('TextGrid关联预览请使用桌面版或登录项目。当前页面仅提供本地WAV预览。');},dispose(){files.clear();}};
}

export function serverFiles(owner:string,project:string,onInvalid:()=>void,csrf?:()=>string):ResearchFiles {
  const science=new BrowserScience(),batchScience=new BrowserScience(),browserSources=new Map<string,ArrayBuffer>();
  const browserFailures=new Map<string,number>();
  const browserEggs=new Map<string,{remote?:string}>();
    async function upload(name:string,buffer:ArrayBuffer,key:string,label='标注'){
      const hash=await sha256(buffer);
      key=key+'_'+hash.slice(0,32);
      // The idempotent upload endpoint returns both partial and ready uploads.
      // Public asset metadata deliberately rejects files that are still uploading.
      const current:components['schemas']['AssetView']=await(await request('uploads',undefined,'POST',{project_id:project,name,expected_bytes:buffer.byteLength,idempotency_key:key})).json();
      const id=current.id;
      if(current.state!=='ready'){
        for(let offset=current.size_bytes;offset<buffer.byteLength;offset+=262144){
          if(disposed)throw Error('项目会话已关闭。');
          const response=await fetch('/api/v1/uploads/'+id+'/blocks?offset='+offset,{method:'PUT',credentials:'same-origin',signal:abort.signal,headers:{'X-PTB-Account':owner,'X-CSRF-Token':csrf!(),'Content-Type':'application/octet-stream'},body:buffer.slice(offset,offset+262144)});
          if(!response.ok){if(response.status===401)onInvalid();throw Error(label+'上传失败，编辑仍保留。请检查登录和剩余空间后重试。');}
        }
      }
      const asset:components['schemas']['AssetView']=await(await request('uploads/'+id+'/finalize',undefined,'POST',{sha256:hash})).json();
      if(asset.sha256!==hash)throw Error(label+'保存校验失败。');
      return {id:asset.id,name:asset.name,kind:fileKind(asset.name)!,size:asset.size_bytes,sha256:hash,expiresAt:asset.expires_at,policyVersion:asset.policy_version};
    }
  const localJobs=new BrowserJobs(owner,project,{
    list:async()=>((await(await request('assets?project_id='+encodeURIComponent(project)+'&order=created')).json()).assets),
    read:async asset=>{const value=await readManaged({id:asset.id,name:asset.name,kind:'parameter',size:asset.size_bytes,sha256:asset.sha256},undefined,32000);return value.buffer;},
    upload:async(name,raw)=>upload(name,raw,'browser-'+(await sha256(new TextEncoder().encode(name).buffer)).slice(0,32),'浏览器重建'),
    compute:async(file,config,signal)=>{try{await verify(file);const result=await batchScience.run<{files:{name:string;data:string}[]}>('reconstruction',await browserRaw(file,16_000_000),config,AbortSignal.any([abort.signal,signal]));await verify(file);return result;}catch(error){if(!signal.aborted)browserFailures.set('reconstruction',Date.now());throw error;}}
  });
  const encodeBase64=(raw:ArrayBuffer)=>{const bytes=new Uint8Array(raw);let value='';for(let i=0;i<bytes.length;i+=8192)value+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(value);};
  const localLpcJobs=new BrowserJobs<{analysis:LpcTaskConfig;textgrid:ResearchFile|null}>(owner,project,{
    list:async()=>((await(await request('assets?project_id='+encodeURIComponent(project)+'&order=created')).json()).assets),
    read:async asset=>(await readManaged({id:asset.id,name:asset.name,kind:'parameter',size:asset.size_bytes,sha256:asset.sha256},undefined,32000)).buffer,
    upload:async(name,raw)=>upload(name,raw,'browser-'+(await sha256(new TextEncoder().encode(name).buffer)).slice(0,32),'浏览器LPC'),
    compute:async(file,config,signal)=>{try{const grid=config.textgrid;await verify(file);if(grid)await verify(grid);const args={name:file.name,config:config.analysis,textgrid:grid?encodeBase64(await browserRaw(grid,2_000_000)):null};const output=await batchScience.run<any>('lpc',await browserRaw(file),args,AbortSignal.any([abort.signal,signal]));await verify(file);if(grid)await verify(grid);return output;}catch(error){if(!signal.aborted)browserFailures.set('lpc',Date.now());throw error;}}
  },{prefix:'LPC任务-',operation:'lpc_analysis',kind:'managed_lpc_files',names:['lpc.ptb.json','lpc_SPECTRUM.png','lpc_AUDIO.wav']});
  async function browserTry<T>(operation:string,work:()=>Promise<T>,fallback:()=>Promise<T>):Promise<T>{
    if(typeof Worker==='undefined'||Date.now()-(browserFailures.get(operation)??0)<30000)return fallback();
    try{return await work();}catch(error){if(disposed||abort.signal.aborted||error instanceof DOMException&&error.name==='AbortError')throw error;browserFailures.set(operation,Date.now());console.warn('Browser compute unavailable; using server:',operation);return fallback();}
  }
  const vocalPorts=new Set<VocalPort>();
  const abort=new AbortController();let disposed=false,listedAtCapacity=false;
  async function request(path:string,signal?:AbortSignal,method='GET',body?:unknown,binary=false) {
    if(disposed)throw Error('项目会话已关闭。');
    const jobMutation=method!=='GET'&&path.startsWith('jobs/');if(jobMutation){jobLists.clear();jobDetails.clear();}
    const r=await fetch('/api/v1/'+path,{method,body:body?(binary?body as Blob:JSON.stringify(body)):undefined,credentials:'same-origin',cache:'no-store',signal:signal?AbortSignal.any([abort.signal,signal]):abort.signal,headers:{'X-PTB-Account':owner,...(body?{'Content-Type':binary?'application/octet-stream':'application/json'}:{}),...(method!=='GET'?{'X-CSRF-Token':csrf?.()??''}:{})}});
    if(!r.ok){const error=await r.json().catch(()=>({}));if(r.status===401||error.detail==='account_changed')onInvalid();if(r.status===422&&path==='jobs/spec2wav/create')throw Error('标定参数无效：请核对时间、频率、dB范围及迭代次数。');if(typeof error.detail==='string'&&(error.detail.startsWith('egg_')||error.detail.startsWith('m09_')||error.detail.startsWith('m08_')||error.detail.startsWith('m06_')||error.detail.startsWith('preview_')||error.detail.startsWith('m07_')||['invalid_spectrogram_input','invalid_spectrogram_config','invalid_spectrogram_image','invalid_image_corners','spectrogram_budget'].includes(error.detail)))throw Error(({m06_platform_unverified:'服务器上的声学参数合成暂不可用，请稍后重试。',m08_platform_unverified:'服务器上的变速变调暂不可用，请稍后重试。',m06_runtime_unavailable:'服务器合成服务暂不可用，请稍后重试。',m08_runtime_unavailable:'服务器变调服务暂不可用，请稍后重试。'} as Record<string,string>)[error.detail]??error.detail);throw Error(({m06_platform_unverified:'服务器上的声学参数合成暂不可用，请稍后重试。',m08_platform_unverified:'服务器上的变速变调暂不可用，请稍后重试。',upload_size_limit:'文件超过单文件上传上限。',disk_space_low:'服务器空间不足，请清理文件或稍后重试。',upload_closed:'上传已关闭，请重新选择文件。',storage_busy:'文件服务忙，请稍后重试。',web_server_busy:'服务器繁忙，请稍等5秒后重试。',account_queue_full:'你的待处理任务已达上限（8 个），请等待完成或取消部分任务后重试。',server_queue_full:'服务器排队已满，请稍后重试。',workspace_expired:'本次工作区已到期，请返回项目页面开始新的工作。',input_unavailable:'源文件已失效或临近到期，请重新上传。',quota_exceeded:'文件空间不足，请清理不需要的文件后重试。',invalid_parameter_table:'参数表包含不支持的内容、公式或损坏结构。原文件未修改。',parameter_read_failed:'参数表超过读取预算或读取失败。',parameter_read_timeout:'参数表读取超时，请缩小数据。',asset_expired:'文件已到期，请重新上传。',invalid_textgrid:'TextGrid格式不受支持或不完整。',unsupported_textgrid:'TextGrid超过2 MB或类型不正确。'} as Record<string,string>)[error.detail]||(r.status>=500?'服务器暂时无法完成请求，请稍后重试。':'请求未完成，请刷新后重试。'));}
    if(jobMutation){jobLists.clear();jobDetails.clear();}
    return r;
  }
  async function verify(file:ResearchFile){const r=await request('assets/'+encodeURIComponent(file.id));const asset:components['schemas']['AssetView']=await r.json();if(asset.project_id!==project||asset.state!=='ready'||asset.sha256!==file.sha256)throw Error('文件已变化，请刷新项目列表。');return asset;}
  async function readManaged(file:ResearchFile,signal:AbortSignal|undefined,limit:number){const verified=await verify(file),expected=file.size??verified.size_bytes;if(expected>limit)throw Error('文件超过当前预览上限。');const response=await request('assets/'+encodeURIComponent(file.id)+'/content',signal);const reader=response.body?.getReader();if(!reader)throw Error('文件读取失败。');const chunks:Uint8Array[]=[];let size=0;try{while(true){const {done,value}=await reader.read();if(done)break;size+=value.length;if(size>limit||size>expected)throw Error('文件长度已变化。');chunks.push(value);}}finally{await reader.cancel();}if(size!==expected)throw Error('文件下载不完整。');const bytes=new Uint8Array(size);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}const hash=await sha256(bytes.buffer);if(hash!==file.sha256)throw Error('文件校验失败，请重新加载。');return {buffer:bytes.buffer,sha256:hash};}
  const jobLists=sharedJobReads<JobView[]>(async()=>((await(await request('jobs?project_id='+encodeURIComponent(project))).json()) as components['schemas']['JobList']).jobs);
  const jobDetails=sharedJobReads<JobView>(async id=>(await request('jobs/'+encodeURIComponent(id))).json());
  const tasks:ResearchTasks={async parent(file){if(!file.sha256)return null;return (await request('jobs/parents/latest?'+new URLSearchParams({project_id:project,sha256:file.sha256}))).json();},async submit(operation,inputs,config,layer,key){const mapped=inputs.map(item=>Object.fromEntries(Object.entries(item).filter(([,v])=>v!=null).map(([role,value])=>{if(role==='parent_result')return [role,value];const file=value as ResearchFile;if(!file.sha256)throw Error('文件缺少校验值，请刷新。');return [role,{asset_id:file.id,sha256:file.sha256}];})));return (await request('jobs/batches/create',undefined,'POST',{project_id:project,operation,inputs:mapped,config,layer,idempotency_key:key})).json();},
    async reconstruct(file,config,key){if(!file.sha256)throw Error('图片缺少校验值。');if(typeof Worker!=='undefined'&&(navigator.hardwareConcurrency??4)>=4&&((navigator as Navigator&{deviceMemory?:number}).deviceMemory??4)>=4&&Date.now()-(browserFailures.get('reconstruction')??0)>=30000&&(config.mode!=='audio_draw'||file.name.toLowerCase().endsWith('.wav'))&&JSON.stringify(config).length<=16000&&config.algorithm==='griffin_lim'&&config.vocoder==='none'&&config.time_end-config.time_start<=5&&config.n_iter<=64&&file.size<=(config.mode==='audio_draw'?4_000_000:1_000_000)){await verify(file);return localJobs.start(file,config);}return (await request('jobs/spec2wav/create',undefined,'POST',{project_id:project,idempotency_key:key,image:{asset_id:file.id,sha256:file.sha256},config})).json();},
    async reconstructionPreview(file,config){if(!file.sha256)throw Error('输入缺少校验值。');await verify(file);const value=await browserTry('prediction',async()=>science.run<any>('prediction',await browserRaw(file,16_000_000),config,abort.signal),async()=>(await request('jobs/spec2wav/preview',undefined,'POST',{project_id:project,idempotency_key:crypto.randomUUID(),image:{asset_id:file.id,sha256:file.sha256},config})).json());await verify(file);return value;},
    async egg(file,config,key){if(!file.sha256)throw Error('音频缺少校验值。');const {exportFontSnapshot}=await import('../state/fonts.ts');return (await request('jobs/egg/create',undefined,'POST',{project_id:project,idempotency_key:key,audio:{asset_id:file.id,sha256:file.sha256},config:{...config,font:config.font??exportFontSnapshot()}})).json();},
    async eggJobs(){return (await jobLists.read('project')).filter(j=>j.operation==='egg_analysis');},
    async eggFonts(font){return (await request('jobs/egg/fonts',undefined,'POST',font)).json();},
    async lpc(file,textgrid,config,key){if(!file.sha256||(textgrid&&!textgrid.sha256))throw Error('文件缺少校验值，请刷新。');if(typeof Worker!=='undefined'&&Date.now()-(browserFailures.get('lpc')??0)>=30000&&file.size<=64_000_000){await verify(file);if(textgrid)await verify(textgrid);return localLpcJobs.start(file,{analysis:config,textgrid});}return (await request('jobs/lpc/create',undefined,'POST',{project_id:project,idempotency_key:key,audio:{asset_id:file.id,sha256:file.sha256},textgrid:textgrid?{asset_id:textgrid.id,sha256:textgrid.sha256}:null,config})).json();},
    async lpcJobs(){const [server,local]=await Promise.all([jobLists.read('project'),localLpcJobs.list()]);return [...local,...server.filter(j=>j.operation==='lpc_analysis')].sort((a,b)=>b.created_at-a.created_at);},
    async lpcFonts(font){return browserTry('lpc',async()=>{const value=await batchScience.run<FontPreflight>('lpc_fonts',new ArrayBuffer(0),font,abort.signal);if(!value.available)throw Error('Requested browser export font is unavailable');return value;},async()=>(await request('jobs/lpc/fonts',undefined,'POST',font)).json());},
    async reconstructions(){const [server,local]=await Promise.all([jobLists.read('project'),localJobs.list()]);return [...local,...server.filter(j=>j.operation==='spectrogram_to_audio')].sort((a,b)=>b.created_at-a.created_at);},
    async cancelJob(id){if(localJobs.has(id))return localJobs.cancel(id);if(localLpcJobs.has(id))return localLpcJobs.cancel(id);return (await request('jobs/'+encodeURIComponent(id)+'/cancel',undefined,'POST')).json();},
    async result(_job,id,sha){const response=await request('assets/'+encodeURIComponent(id)+'/content');const raw=await response.arrayBuffer();if(raw.byteLength>64_000_000||await sha256(raw)!==sha)throw Error('结果校验失败。');return raw;},
    async list(){return (await(await request('jobs/batches/list?project_id='+encodeURIComponent(project))).json()).batches;},
    async get(id){return (await request('jobs/batches/'+encodeURIComponent(id))).json();},
    async cancel(id){return (await request('jobs/batches/'+encodeURIComponent(id)+'/cancel',undefined,'POST')).json();},
    async job(id){return localJobs.get(id)??localLpcJobs.get(id)??jobDetails.read(id);},
    async retry(id,key){const lpc=localLpcJobs.source(id);if(lpc){if(localLpcJobs.get(id)?.state!=='failed')return tasks.lpc!(lpc.file,lpc.config.textgrid,lpc.config.analysis,key);await verify(lpc.file);if(lpc.config.textgrid)await verify(lpc.config.textgrid);return(await request('jobs/lpc/create',undefined,'POST',{project_id:project,idempotency_key:key,audio:{asset_id:lpc.file.id,sha256:lpc.file.sha256},textgrid:lpc.config.textgrid?{asset_id:lpc.config.textgrid.id,sha256:lpc.config.textgrid.sha256}:null,config:lpc.config.analysis})).json();}const source=localJobs.source(id);if(source){if(localJobs.get(id)?.state!=='failed')return tasks.reconstruct!(source.file,source.config,key);await verify(source.file);return(await request('jobs/spec2wav/create',undefined,'POST',{project_id:project,idempotency_key:key,image:{asset_id:source.file.id,sha256:source.file.sha256},config:source.config})).json();}return (await request('jobs/'+encodeURIComponent(id)+'/retry',undefined,'POST',{idempotency_key:key})).json();},
    async download(id,name){const response=await request('assets/'+encodeURIComponent(id)+'/content');const blob=await response.blob();if(disposed)return;const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}};
  async function browserRaw(file:ResearchFile,limit=64_000_000){
    await verify(file);const key=file.id+':'+file.sha256;let raw=browserSources.get(key);
    if(!raw){raw=(await readManaged(file,undefined,limit)).buffer;for(const id of browserSources.keys()){if([...browserSources.values()].reduce((n,b)=>n+b.byteLength,0)+raw.byteLength<=64_000_000)break;browserSources.delete(id);}if(raw.byteLength<=64_000_000)browserSources.set(key,raw);}
    if(raw.byteLength>limit)throw Error('文件超过当前预览上限。');return raw.slice(0);
  }
  const adapter:ResearchFiles={kind:'server',feedbackCsrf:csrf,tasks:csrf?tasks:undefined,async list(){const data:components['schemas']['AssetList']=await(await request('assets?project_id='+encodeURIComponent(project)+'&order=created')).json();listedAtCapacity=data.assets.length>=1000;return data.assets.filter(a=>a.project_id===project&&a.state==='ready'&&fileKind(a.name)).sort((a,b)=>b.created_at-a.created_at).map(a=>({id:a.id,name:a.name,kind:fileKind(a.name)!,size:a.size_bytes,sha256:a.sha256??undefined,expiresAt:a.expires_at}));},
    async associatedParameters(file){
      if(!file.sha256)return null;
      const table:{asset_id:string;name:string;sha256:string}|null=await(await request('jobs/parents/latest?'+new URLSearchParams({project_id:project,sha256:file.sha256,table:'true'}))).json();
      if(!table)return null;
      const associated:ResearchFile={id:table.asset_id,name:table.name,kind:'parameter',sha256:table.sha256,size:0};
      const asset=await verify(associated);associated.size=asset.size_bytes;associated.expiresAt=asset.expires_at;return associated;
    },
    async read(file,signal){if(signal?.aborted)throw new DOMException('Cancelled','AbortError');if(['audio','image','parameter'].includes(file.kind)){const buffer=await browserRaw(file,file.kind==='audio'?64_000_000:16_000_000);if(signal?.aborted)throw new DOMException('Cancelled','AbortError');return {buffer,sha256:file.sha256!};}return readManaged(file,signal,2_000_000);},
    async textgrid(file){await verify(file);const data:components['schemas']['TextGridPreview']=await(await request('assets/'+encodeURIComponent(file.id)+'/textgrid')).json();if(data.sha256!==file.sha256)throw Error('关联文件已变化，请刷新。');return data;},
    async spectrogram(file,view){await verify(file);const data:Spectrogram=await browserTry('spectrogram',()=>browserRaw(file).then(raw=>science.run<Spectrogram>('spectrogram',raw,view,abort.signal)),async()=>{const query=new URLSearchParams(Object.entries(view).map(([k,v])=>[k,String(v)]));return(await request('assets/'+encodeURIComponent(file.id)+'/spectrogram?'+query)).json();});await verify(file);if(data.sha256!==file.sha256)throw Error('音频已变化，请刷新。');return data;},
    async parameters(file){await verify(file);const data:ParameterTable=await browserTry('parameters',()=>browserRaw(file,16_000_000).then(raw=>science.run<ParameterTable>('parameters',raw,{name:file.name},abort.signal)),async()=>(await request('assets/'+encodeURIComponent(file.id)+'/parameters')).json());await verify(file);if(data.sha256!==file.sha256)throw Error('参数文件已变化，请刷新。');return data;},
    eggPreview:{
      async open(file){await verify(file);return browserTry('egg',async()=>{const id=crypto.randomUUID(),value=await science.run<any>('egg_open',await browserRaw(file),{id},abort.signal);await verify(file);browserEggs.set(id,{});return value;},async()=>(await request('assets/'+encodeURIComponent(file.id)+'/egg-preview',undefined,'POST')).json());},
      async update(file,session,config,options){
        await verify(file);const local=browserEggs.get(session),query=new URLSearchParams();if(options?.autoDb)query.set('auto_db','true');if(options?.needAudio)query.set('need_audio','true');
        const remote=async()=>{if(local&&!local.remote)local.remote=(await(await request('assets/'+encodeURIComponent(file.id)+'/egg-preview',undefined,'POST')).json()).session_id;return(await request('assets/'+encodeURIComponent(file.id)+'/egg-preview/'+encodeURIComponent(local?.remote??session)+(query.size?'?'+query:''),undefined,'POST',config)).json();};
        if(!local||config.keep_reaper_f0)return remote();
        const value=await browserTry('egg',async()=>{const args={id:session,config,...options};try{return await science.run<any>('egg_update',new ArrayBuffer(0),args,abort.signal);}catch(error){if(!String(error).includes('egg_preview_expired'))throw error;await science.run('egg_open',await browserRaw(file),{id:session},abort.signal);return science.run<any>('egg_update',new ArrayBuffer(0),args,abort.signal);}},remote);await verify(file);return value;
      },
      async close(file,session){const local=browserEggs.get(session);browserEggs.delete(session);if(local){await science.run('egg_close',new ArrayBuffer(0),{id:session},abort.signal).catch(()=>{});if(local.remote)return(await request('assets/'+encodeURIComponent(file.id)+'/egg-preview/'+encodeURIComponent(local.remote),undefined,'DELETE')).json();return {closed:true};}return(await request('assets/'+encodeURIComponent(file.id)+'/egg-preview/'+encodeURIComponent(session),undefined,'DELETE')).json();}
    },
    dispose(){jobLists.clear();jobDetails.clear();localJobs.close();localLpcJobs.close();science.close();batchScience.close();browserSources.clear();browserEggs.clear();for(const port of vocalPorts)void port.close(true);vocalPorts.clear();disposed=true;abort.abort();}};
  if(csrf){
    adapter.vocal=key=>{
      const port=webVocalPort(async(method,path,body)=>{
        const r=await fetch('/api/v1/'+path,{method,credentials:'same-origin',cache:'no-store',headers:{'X-PTB-Account':owner,'X-CSRF-Token':csrf(),'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined,keepalive:method==='DELETE'});
        if(!r.ok){const e=await r.json().catch(()=>({})),code=e.detail;if(r.status===401||code==='account_changed')onInvalid();throw Error(({vocal_unavailable:'服务器声道引擎尚未启动，请稍后重试。',vocal_capacity_full:'声道计算席位正在使用，请稍后重新打开。',vocal_busy:'服务器正在计算，请稍后重试。',vocal_cancelled:'生成已取消。',vocal_session_expired:'声道会话已结束，请重新打开生理参数合成。',vocal_session_not_found:'声道会话已结束，请重新打开生理参数合成。',vocal_timeout:'声道计算超时，请重新打开并缩短动作序列。',vocal_runtime_unavailable:'声道计算进程已退出，请重新打开。',workspace_expired:'本次工作区已到期，请返回项目页面。'} as Record<string,string>)[code]||code||'声道请求失败。');}
        return r.json();
      },project,key);vocalPorts.add(port);return port;
    };
    let recordingSave:{raw:string;asset:any;expected:string|null}|undefined;
    const recordingAsset=(a:any):RecordingAsset=>({id:a.id,name:a.name,kind:'audio',size:a.size_bytes??a.size,sha256:a.sha256,expiresAt:a.expires_at??a.expiresAt});
    adapter.recording={
      async compute(config,signal){
        const current=signal?AbortSignal.any([abort.signal,signal]):abort.signal;
        return browserTry('recording_compute',async()=>{
          const inputs=await Promise.all((config.inputs as string[]).map(async id=>{const asset=await(await request('assets/'+encodeURIComponent(id))).json();return {id,name:asset.name,kind:'audio' as const,size:asset.size_bytes,sha256:(config.hashes as Record<string,string>)[id]};}));
          const buffers=await Promise.all(inputs.map(file=>browserRaw(file,100_000_000)));if(buffers.reduce((n,b)=>n+b.byteLength,0)>64_000_000)throw Error('Browser recording budget');
          const encoded=(raw:ArrayBuffer)=>{const bytes=new Uint8Array(raw);let text='';for(let i=0;i<bytes.length;i+=8192)text+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(text);};
          const result=await science.run<{audio_base64:string|null;metadata:Record<string,unknown>}>('recording_compute',buffers[0],{config,extra:buffers.slice(1).map(encoded)},current);
          for(const file of inputs)await verify(file);if(current.aborted)throw new DOMException('Cancelled','AbortError');
          const asset=result.audio_base64?await adapter.recording!.upload(new File([Uint8Array.from(atob(result.audio_base64),c=>c.charCodeAt(0))],'录音-'+config.action+'.wav',{type:'audio/wav'})):undefined;
          return {asset,metadata:result.metadata};
        },async()=>{
          let job=await adapter.recording!.create(config,crypto.randomUUID());
          try{while(['queued','running','cancel_requested'].includes(job.state)){if(current.aborted)throw new DOMException('Cancelled','AbortError');await waitForTaskPoll(job.state,current);if(current.aborted)throw new DOMException('Cancelled','AbortError');job=await adapter.recording!.job(job.id);}if(job.state!=='succeeded')throw Error(job.error_code||'录音处理失败');return adapter.recording!.result(job);}
          catch(error){if(current.aborted)await adapter.recording!.cancel(job.id).catch(()=>{});throw error;}
        });
      },
      async upload(file){let key=selectedUploads.get(file);if(!key){key='m16-'+crypto.randomUUID();selectedUploads.set(file,key);}return recordingAsset(await upload(file.name,await file.arrayBuffer(),key,'录音'));},
      async load(){for(let attempt=0;;attempt++){const {asset}=await(await request('jobs/m16/project?project_id='+project)).json();if(!asset)return null;try{const raw=await(await request('assets/'+asset.id+'/content')).arrayBuffer();return {snapshot:JSON.parse(new TextDecoder().decode(raw)),id:asset.id};}catch(error){if(disposed||abort.signal.aborted||attempt===2)throw error;await new Promise(resolve=>setTimeout(resolve,75));}}},
      async save(snapshot,expected){const raw=JSON.stringify(snapshot);const commit=async()=>{const saved=await(await request('jobs/m16/project',undefined,'POST',{project_id:project,asset_id:recordingSave!.asset.id,expected:recordingSave!.expected})).json();recordingSave=undefined;return saved.id as string;};if(recordingSave){const same=recordingSave.raw===raw;expected=await commit();if(same)return expected;}const key=crypto.randomUUID();recordingSave={raw,asset:await upload(key+'.m16-draft.json',new TextEncoder().encode(raw).buffer,'m16-project-'+key,'录音工程'),expected};return commit();},
      async create(config,key){return (await request('jobs/m16/create',undefined,'POST',{project_id:project,idempotency_key:key,config})).json();},
      job:tasks.job,cancel:tasks.cancelJob!,
      async read(asset){return (await readManaged(asset,undefined,100_000_000)).buffer;},
      async remove(asset){await verify(asset);await request('assets/'+asset.id,undefined,'DELETE');},
      async spectrum(asset,start,end,channel){const value:any=await browserTry('recording_spectrum',async()=>science.run('recording_spectrum',await browserRaw(asset,100_000_000),{start,end,channel,width:640},abort.signal),async()=>{const value=await(await request('assets/'+asset.id+'/recording-preview?'+new URLSearchParams({start:String(start),end:String(end),channel:String(channel)}))).json();if(value.sha256!==asset.sha256)throw Error('录音已变化，请刷新。');return value;});await verify(asset);return recordingSpectrum(value);},
      async live(file,frames,channel){return recordingSpectrum(await browserTry('recording_spectrum',async()=>science.run('recording_spectrum',await file.arrayBuffer(),{start:0,end:frames,channel,width:640},abort.signal),async()=>(await request('jobs/m16/live-preview?'+new URLSearchParams({start:'0',end:String(frames),channel:String(channel)}),undefined,'POST',file,true)).json()));},
      async result(job){const manifest=job.result_manifest as {files?:any[]}|null,files=manifest?.files;if(!files)throw Error('录音处理结果不完整。');const meta=files.find(f=>f.name.endsWith('.json'));if(!meta)throw Error('录音处理清单缺失。');const metadata=JSON.parse(new TextDecoder().decode(await(await request('assets/'+meta.id+'/content')).arrayBuffer()));const wav=files.find(f=>f.name.endsWith('.wav'));return {asset:wav?recordingAsset(wav):undefined,metadata};}
    };
    // Keep each selected File's key across retry, including a lost block/finalize reply.
    const selectedUploads=new WeakMap<File,string>();
    adapter.upload=async(values,progress)=>{
      for(const file of values){if(!fileKind(file.name))throw Error('不支持的文件类型：'+file.name);if(file.size>100_000_000)throw Error(file.name+' 超过单文件100 MB上限。');if(!file.size)throw Error(file.name+' 是空文件。');}
      const total=values.reduce((sum,f)=>sum+f.size,0);let completed=0;
      for(const file of values){
        let key=selectedUploads.get(file);if(!key){key=crypto.randomUUID();selectedUploads.set(file,key);}
        let asset:components['schemas']['AssetView']=await(await request('uploads',undefined,'POST',{project_id:project,name:file.name,expected_bytes:file.size,idempotency_key:key})).json();
        if(!['uploading','ready'].includes(asset.state))throw Error('上传已关闭，请重新选择文件。');
        while(asset.state==='uploading'&&asset.size_bytes<file.size){
          const offset=asset.size_bytes;
          asset=await(await request('uploads/'+asset.id+'/blocks?offset='+offset,undefined,'PUT',file.slice(offset,offset+262144),true)).json();
          if(asset.size_bytes<=offset||asset.size_bytes>file.size)throw Error('上传进度异常，请重试。');
          progress((completed+asset.size_bytes)/total);
        }
        await request('uploads/'+asset.id+'/finalize',undefined,'POST',{});
        completed+=file.size;progress(completed/total);
      }
    };
    adapter.m07=m07Port({project,async source(file){await verify(file);return {asset_id:file.id,sha256:file.sha256!};},async create(body){return (await request('jobs/m07/create',undefined,'POST',body)).json();},job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!,retry:tasks.retry,async list(){return (await jobLists.read('project')).filter(j=>j.operation==='phonation_synthesis');}});
    const saveBrowserOutput=async(name:string,raw:ArrayBuffer,signal:AbortSignal)=>{signal.throwIfAborted();const asset=await upload(name,raw,'browser-'+(await sha256(new TextEncoder().encode(name).buffer)).slice(0,32),'浏览器结果');signal.throwIfAborted();return {id:asset.id,sha256:asset.sha256,size_bytes:asset.size};};
    adapter.m06=browserM06(m06Port({project,async parameters(text,signal){const a=await upload('m06-parameters.csv',new TextEncoder().encode(text).buffer,'m06-'+crypto.randomUUID(),'合成参数');if(signal.aborted)throw Error('操作已取消');return {asset_id:a.id,sha256:a.sha256!};},async source(file){await verify(file);return {asset_id:file.id,sha256:file.sha256!};},
      async create(body){return (await request('jobs/m06/create',undefined,'POST',body)).json();},job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!}),{project,
      compute:(args,signal)=>browserTry('speech_synthesis',()=>batchScience.run<any>('speech_synthesis',new ArrayBuffer(0),args,AbortSignal.any([abort.signal,signal])),async()=>null),save:saveBrowserOutput});
    adapter.papers=serverPapers(owner,project,csrf!,onInvalid,abort.signal);
    adapter.m11=serverM11(owner,project,csrf,onInvalid,abort.signal);
    const phonologyFile=(source:{asset_id:string;sha256:string;name:string}):ResearchFile=>({id:source.asset_id,sha256:source.sha256,name:source.name,kind:'parameter',size:0});
    adapter.m14=browserM14(serverM14(owner,project,csrf,onInvalid,abort.signal),{
      async upload(file,signal){signal.throwIfAborted();const asset=await upload(file.name,await file.arrayBuffer(),'m14-browser-'+crypto.randomUUID(),'调查字表');signal.throwIfAborted();return {asset_id:asset.id,sha256:asset.sha256,name:file.name};},
      async read(source,signal){signal.throwIfAborted();const asset=await verify(phonologyFile(source));const raw=await browserRaw({...phonologyFile(source),size:asset.size_bytes},2_000_000);signal.throwIfAborted();return raw;},
      async verify(source,signal){signal.throwIfAborted();await verify(phonologyFile(source));signal.throwIfAborted();},
      compute:(raw,args,signal)=>browserTry('phonology',()=>batchScience.run<any>('phonology',raw,args,AbortSignal.any([abort.signal,signal])),async()=>null),save:saveBrowserOutput
    });
    adapter.m08=m08Port({project,async source(file){await verify(file);return {asset_id:file.id,sha256:file.sha256!};},
      async request(action,body){return (await request('jobs/m08/'+(action==='list'?'list/'+encodeURIComponent(project):action),undefined,action==='list'?'GET':'POST',body)).json();},
      job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!});

    const annotation=portableAnnotation(adapter,upload);
    adapter.annotation={...annotation,async scan(){const values=await adapter.list();if(listedAtCapacity)throw Error('项目资源达到当前 1000 条列表预算，请先在文件管理清理旧版本。');const seen=new Set<string>();return values.filter(f=>{const key=f.name.toLowerCase();if(seen.has(key))return false;seen.add(key);return true;});}};
  }
  return adapter;
}
