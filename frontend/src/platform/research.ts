import type { components } from '../../../contracts/generated/api';
import { decodeWav } from './decode.ts';
import {portableAnnotation,type AnnotationPort} from './annotation.ts';
import {m07Port,type M07Port} from './m07.ts';
import {m06Port} from './m06.ts';
import type {M06Port} from '../modules/speech-synthesis/port.ts';
import {m08Port} from './m08.ts';
import {serverM14} from './m14.ts';
import {serverM11} from './m11.ts';
import type {LipPort} from '../modules/lip-extraction/port.ts';
import type {M11Port} from '../modules/mfa/port.ts';
import type {M14Port} from '../modules/phonology-induction/port.ts';
import type {M08Port} from '../modules/pitch-manipulation/port.ts';
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
export type JobView=components['schemas']['JobView'];
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
  kind:'desktop'|'server'|'preview';
  tasks?:ResearchTasks;
  annotation?:AnnotationPort;
  m06?:M06Port;
  m07?:M07Port;
  m08?:M08Port;
  m14?:M14Port;
  m11?:M11Port;
  m05?:LipPort;
  choose?(purpose:DirectoryGrant['purpose']):Promise<DirectoryGrant|null>;
  add?(files:File[]):void;
  list(directory?:string,recursive?:boolean):Promise<ResearchFile[]>;
  previewAudio?(file:ResearchFile):Promise<{buffer:ArrayBuffer;sha256:string;sourceDuration:number;previewNote:string}>;
  read(file:ResearchFile,signal?:AbortSignal):Promise<{buffer:ArrayBuffer;sha256:string}>;
  textgrid(file:ResearchFile):Promise<{sha256:string;tiers:Tier[]}>;
  spectrogram?(file:ResearchFile,view:SpectrogramView):Promise<Spectrogram>;
  parameters?(file:ResearchFile,view?:ParameterView):Promise<ParameterTable>;
  eggPreview?:{
    open(file:ResearchFile):Promise<components['schemas']['EggPreviewSession']>;
    update(file:ResearchFile,session:string,config:EggTaskConfig):Promise<components['schemas']['EggInteractiveResult']>;
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
  const abort=new AbortController();let disposed=false,listedAtCapacity=false;
  async function request(path:string,signal?:AbortSignal,method='GET',body?:unknown) {
    if(disposed)throw Error('项目会话已关闭。');
    const r=await fetch('/api/v1/'+path,{method,body:body?JSON.stringify(body):undefined,credentials:'same-origin',cache:'no-store',signal:signal?AbortSignal.any([abort.signal,signal]):abort.signal,headers:{'X-PTB-Account':owner,...(body?{'Content-Type':'application/json'}:{}),...(method!=='GET'?{'X-CSRF-Token':csrf?.()??''}:{})}});
    if(!r.ok){const error=await r.json().catch(()=>({}));if(r.status===401||error.detail==='account_changed')onInvalid();if(r.status===422&&path==='jobs/spec2wav/create')throw Error('标定参数无效：请核对时间、频率、dB范围及迭代次数。');if(typeof error.detail==='string'&&(error.detail.startsWith('egg_')||error.detail.startsWith('preview_')||error.detail.startsWith('m07_')||error.detail==='invalid_spectrogram_input'))throw Error(error.detail);throw Error(({input_unavailable:'源文件已失效或临近到期，请重新上传。',quota_exceeded:'文件空间不足，请清理不需要的文件后重试。',invalid_parameter_table:'参数表包含不支持的内容、公式或损坏结构。原文件未修改。',parameter_read_failed:'参数表超过读取预算或读取失败。',parameter_read_timeout:'参数表读取超时，请缩小数据。',asset_expired:'文件已到期，请重新上传。',invalid_textgrid:'TextGrid格式不受支持或不完整。',unsupported_textgrid:'TextGrid超过2 MB或类型不正确。'} as Record<string,string>)[error.detail]||'项目文件不可访问，请刷新或检查登录状态。');}
    return r;
  }
  async function verify(file:ResearchFile){const r=await request('assets/'+encodeURIComponent(file.id));const asset:components['schemas']['AssetView']=await r.json();if(asset.project_id!==project||asset.state!=='ready'||asset.sha256!==file.sha256)throw Error('文件已变化，请刷新项目列表。');return asset;}
  const tasks:ResearchTasks={async parent(file){if(!file.sha256)return null;return (await request('jobs/parents/latest?'+new URLSearchParams({project_id:project,sha256:file.sha256}))).json();},async submit(operation,inputs,config,layer,key){const mapped=inputs.map(item=>Object.fromEntries(Object.entries(item).filter(([,v])=>v!=null).map(([role,value])=>{if(role==='parent_result')return [role,value];const file=value as ResearchFile;if(!file.sha256)throw Error('文件缺少校验值，请刷新。');return [role,{asset_id:file.id,sha256:file.sha256}];})));return (await request('jobs/batches/create',undefined,'POST',{project_id:project,operation,inputs:mapped,config,layer,idempotency_key:key})).json();},
    async reconstruct(file,config,key){if(!file.sha256)throw Error('图片缺少校验值。');return (await request('jobs/spec2wav/create',undefined,'POST',{project_id:project,idempotency_key:key,image:{asset_id:file.id,sha256:file.sha256},config})).json();},
    async reconstructionPreview(file,config){if(!file.sha256)throw Error('输入缺少校验值。');return (await request('jobs/spec2wav/preview',undefined,'POST',{project_id:project,idempotency_key:crypto.randomUUID(),image:{asset_id:file.id,sha256:file.sha256},config})).json();},
    async egg(file,config,key){if(!file.sha256)throw Error('音频缺少校验值。');const {exportFontSnapshot}=await import('../state/fonts.ts');return (await request('jobs/egg/create',undefined,'POST',{project_id:project,idempotency_key:key,audio:{asset_id:file.id,sha256:file.sha256},config:{...config,font:config.font??exportFontSnapshot()}})).json();},
    async eggJobs(){const data:components['schemas']['JobList']=await(await request('jobs?project_id='+encodeURIComponent(project))).json();return data.jobs.filter(j=>j.operation==='egg_analysis');},
    async eggFonts(font){return (await request('jobs/egg/fonts',undefined,'POST',font)).json();},
    async lpc(file,textgrid,config,key){if(!file.sha256||(textgrid&&!textgrid.sha256))throw Error('文件缺少校验值，请刷新。');return (await request('jobs/lpc/create',undefined,'POST',{project_id:project,idempotency_key:key,audio:{asset_id:file.id,sha256:file.sha256},textgrid:textgrid?{asset_id:textgrid.id,sha256:textgrid.sha256}:null,config})).json();},
    async lpcJobs(){const data:components['schemas']['JobList']=await(await request('jobs?project_id='+encodeURIComponent(project))).json();return data.jobs.filter(j=>j.operation==='lpc_analysis');},
    async lpcFonts(font){return (await request('jobs/lpc/fonts',undefined,'POST',font)).json();},
    async reconstructions(){const data:components['schemas']['JobList']=await(await request('jobs?project_id='+encodeURIComponent(project))).json();return data.jobs.filter(j=>j.operation==='spectrogram_to_audio');},
    async cancelJob(id){return (await request('jobs/'+encodeURIComponent(id)+'/cancel',undefined,'POST')).json();},
    async result(_job,id,sha){const response=await request('assets/'+encodeURIComponent(id)+'/content');const raw=await response.arrayBuffer();if(raw.byteLength>64_000_000||await sha256(raw)!==sha)throw Error('结果校验失败。');return raw;},
    async list(){return (await(await request('jobs/batches/list?project_id='+encodeURIComponent(project))).json()).batches;},
    async get(id){return (await request('jobs/batches/'+encodeURIComponent(id))).json();},
    async cancel(id){return (await request('jobs/batches/'+encodeURIComponent(id)+'/cancel',undefined,'POST')).json();},
    async job(id){return (await request('jobs/'+encodeURIComponent(id))).json();},
    async retry(id,key){return (await request('jobs/'+encodeURIComponent(id)+'/retry',undefined,'POST',{idempotency_key:key})).json();},
    async download(id,name){const response=await request('assets/'+encodeURIComponent(id)+'/content');const blob=await response.blob();if(disposed)return;const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}};
  const adapter:ResearchFiles={kind:'server',tasks:csrf?tasks:undefined,async list(){const data:components['schemas']['AssetList']=await(await request('assets?project_id='+encodeURIComponent(project)+'&order=created')).json();listedAtCapacity=data.assets.length>=1000;return data.assets.filter(a=>a.project_id===project&&a.state==='ready'&&fileKind(a.name)).sort((a,b)=>b.created_at-a.created_at).map(a=>({id:a.id,name:a.name,kind:fileKind(a.name)!,size:a.size_bytes,sha256:a.sha256??undefined,expiresAt:a.expires_at}));},
    async read(file,signal){await verify(file);const limit=file.kind==='audio'?64_000_000:['image','parameter'].includes(file.kind)?16_000_000:2_000_000;if(file.size>limit)throw Error('文件超过当前预览上限。');const response=await request('assets/'+encodeURIComponent(file.id)+'/content',signal);const reader=response.body?.getReader();if(!reader)throw Error('文件读取失败。');const chunks:Uint8Array[]=[];let size=0;try{while(true){const {done,value}=await reader.read();if(done)break;size+=value.length;if(size>limit||size>file.size)throw Error('文件长度已变化。');chunks.push(value);}}finally{await reader.cancel();}if(size!==file.size)throw Error('文件下载不完整。');const bytes=new Uint8Array(size);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}const hash=await sha256(bytes.buffer);if(hash!==file.sha256)throw Error('文件校验失败，请重新加载。');return {buffer:bytes.buffer,sha256:hash};},
    async textgrid(file){await verify(file);const data:components['schemas']['TextGridPreview']=await(await request('assets/'+encodeURIComponent(file.id)+'/textgrid')).json();if(data.sha256!==file.sha256)throw Error('关联文件已变化，请刷新。');return data;},
    async spectrogram(file,view){await verify(file);const query=new URLSearchParams(Object.entries(view).map(([k,v])=>[k,String(v)]));const data:Spectrogram=await(await request('assets/'+encodeURIComponent(file.id)+'/spectrogram?'+query)).json();if(data.sha256!==file.sha256)throw Error('音频已变化，请刷新。');return data;},
    async parameters(file){await verify(file);const data:ParameterTable=await(await request('assets/'+encodeURIComponent(file.id)+'/parameters')).json();if(data.sha256!==file.sha256)throw Error('参数文件已变化，请刷新。');return data;},
    eggPreview:{
      async open(file){await verify(file);return (await request('assets/'+encodeURIComponent(file.id)+'/egg-preview',undefined,'POST')).json();},
      async update(file,session,config){return (await request('assets/'+encodeURIComponent(file.id)+'/egg-preview/'+encodeURIComponent(session),undefined,'POST',config)).json();},
      async close(file,session){return (await request('assets/'+encodeURIComponent(file.id)+'/egg-preview/'+encodeURIComponent(session),undefined,'DELETE')).json();}
    },
    dispose(){disposed=true;abort.abort();}};
  if(csrf){
    adapter.m07=m07Port({project,async source(file){await verify(file);return {asset_id:file.id,sha256:file.sha256!};},async create(body){return (await request('jobs/m07/create',undefined,'POST',body)).json();},job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!,retry:tasks.retry,async list(){return ((await(await request('jobs?project_id='+encodeURIComponent(project))).json()).jobs as JobView[]).filter(j=>j.operation==='phonation_synthesis');}});
    adapter.m06=m06Port({project,async parameters(text,signal){const a=await upload('m06-parameters.csv',new TextEncoder().encode(text).buffer,'m06-'+crypto.randomUUID(),'合成参数');if(signal.aborted)throw Error('操作已取消');return {asset_id:a.id,sha256:a.sha256!};},async source(file){await verify(file);return {asset_id:file.id,sha256:file.sha256!};},
      async create(body){return (await request('jobs/m06/create',undefined,'POST',body)).json();},job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!});
    adapter.m11=serverM11(owner,project,csrf,onInvalid,abort.signal);
    adapter.m14=serverM14(owner,project,csrf,onInvalid,abort.signal);
    adapter.m08=m08Port({project,async source(file){await verify(file);return {asset_id:file.id,sha256:file.sha256!};},
      async request(action,body){return (await request('jobs/m08/'+(action==='list'?'list/'+encodeURIComponent(project):action),undefined,action==='list'?'GET':'POST',body)).json();},
      job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!});
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
      return {id:asset.id,name:asset.name,kind:fileKind(asset.name)!,size:asset.size_bytes,sha256:hash,expiresAt:asset.expires_at};
    }
    const annotation=portableAnnotation(adapter,upload);
    adapter.annotation={...annotation,async scan(){const values=await adapter.list();if(listedAtCapacity)throw Error('项目资源达到当前 1000 条列表预算，请先在文件管理清理旧版本。');const seen=new Set<string>();return values.filter(f=>{const key=f.name.toLowerCase();if(seen.has(key))return false;seen.add(key);return true;});}};
  }
  return adapter;
}
