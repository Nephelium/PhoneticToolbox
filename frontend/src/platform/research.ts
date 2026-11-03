import type {MessageDescriptor} from '../i18n/index.ts';
import {joinMessages} from '../modules/module-localization.ts';
import {uiMessage,uiError,responseError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {LocalAssets,LocalStorageError,type AssetDatabase} from './local-assets.ts';
import {LocalAssetTransport} from './local-asset-transport.ts';
import {decodeText,parseGrid} from '../modules/annotation/format.ts';
import {sharedJobReads,waitForTaskPoll} from './task-poll.ts';
import {BrowserScience} from './browser-science.ts';
import {BrowserJobs} from './browser-jobs.ts';
import {serverPapers} from './papers-server.ts';
import {webVocalPort,type VocalPort} from './vocal.ts';
import {vocalProfileStore} from './vocal-profile.ts';
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
export interface ResearchFile { id:string; name:string; kind:'audio'|'textgrid'|'lip'|'lip_pickle'|'parameter'|'image'|'lab'; size:number; sha256?:string; expiresAt?:number; annotationKey?:string }
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
  annotationScope?:string;
  localFiles?:boolean;
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
  if(!Number.isFinite(data.sourceDuration)||data.sourceDuration<=0||Math.abs(asset.duration-data.sourceDuration)>1/asset.sampleRate+1e-9)throw uiError(uiMessage('platform.the_preview_time_range_does_not_match_the_original'));
  asset.duration=data.sourceDuration;
  return {asset,sha256:data.sha256,previewNote:data.previewNote};
}
export async function sha256(buffer:ArrayBuffer) {return [...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))].map(n=>n.toString(16).padStart(2,'0')).join('');}

export function previewFiles():ResearchFiles {
  const files=new Map<string,File>();
  return {kind:'preview',add(values){for(const f of values){if(!fileKind(f.name))continue;files.set(crypto.randomUUID(),f);}},
    async list(){return [...files].map(([id,f])=>({id,name:f.name,size:f.size,kind:fileKind(f.name)!}));},
    async read(file){const f=files.get(file.id);if(!f||f.size>64_000_000)throw uiError(uiMessage('platform.preview_is_limited_to_64_mb_extract_shorter_audio'));const buffer=await f.arrayBuffer();return {buffer,sha256:await sha256(buffer)};},
    async textgrid(){throw uiError(uiMessage('platform.for_linked_textgrid_preview_use_the_desktop_version_or'));},dispose(){files.clear();}};
}

export function serverFiles(owner:string,project:string,onInvalid:()=>void,csrf?:()=>string,database?:AssetDatabase,localProject=project):ResearchFiles {
  const science=new BrowserScience(),batchScience=new BrowserScience(),browserSources=new Map<string,ArrayBuffer>();
  const browserFailures=new Map<string,number>();
  const browserEggs=new Map<string,{remote?:string}>();
    async function upload(name:string,buffer:ArrayBuffer,key:string,label:string|MessageDescriptor=uiMessage('platform.annotations')){
      const hash=await sha256(buffer);
      key=key+'_'+hash.slice(0,32);
      // The idempotent upload endpoint returns both partial and ready uploads.
      // Public asset metadata deliberately rejects files that are still uploading.
      const current:components['schemas']['AssetView']=await(await request('uploads',undefined,'POST',{project_id:project,name,expected_bytes:buffer.byteLength,idempotency_key:key})).json();
      const id=current.id;
      if(current.state!=='ready'){
        for(let offset=current.size_bytes;offset<buffer.byteLength;offset+=262144){
          if(disposed)throw uiError(uiMessage('platform.the_project_session_is_closed_445'));
          const response=await assetFetch('uploads/'+id+'/blocks?offset='+offset,{method:'PUT',credentials:'same-origin',signal:abort.signal,headers:{'X-PTB-Account':owner,'X-CSRF-Token':csrf!(),'Content-Type':'application/octet-stream'},body:buffer.slice(offset,offset+262144)});
          if(!response.ok){if(response.status===401)onInvalid();throw uiError(joinMessages(label,uiMessage('platform.upload_failed_edits_are_preserved_check_your_login_and')));}
        }
      }
      const asset:components['schemas']['AssetView']=await(await request('uploads/'+id+'/finalize',undefined,'POST',{sha256:hash})).json();
      if(asset.sha256!==hash)throw uiError(joinMessages(label,uiMessage('platform.save_verification_failed')));
      return {id:asset.id,name:asset.name,kind:fileKind(asset.name)!,size:asset.size_bytes,sha256:hash,expiresAt:asset.expires_at,policyVersion:asset.policy_version};
    }
  const localJobs=new BrowserJobs(owner,localProject,{
    list:async()=>((await(await request('assets?project_id='+encodeURIComponent(project)+'&order=created')).json()).assets),
    read:async asset=>{const value=await readManaged({id:asset.id,name:asset.name,kind:'parameter',size:asset.size_bytes,sha256:asset.sha256},undefined,32000);return value.buffer;},
    upload:async(name,raw)=>upload(name,raw,'browser-'+(await sha256(new TextEncoder().encode(name).buffer)).slice(0,32),uiMessage('platform.browser_reconstruction')),
    compute:async(file,config,signal)=>{try{await verify(file);const result=await batchScience.run<{files:{name:string;data:string}[]}>('reconstruction',await browserRaw(file,16_000_000),config,AbortSignal.any([abort.signal,signal]));await verify(file);return result;}catch(error){if(!signal.aborted&&!(error instanceof LocalStorageError))browserFailures.set('reconstruction',Date.now());throw error;}}
  },undefined,project);
  const encodeBase64=(raw:ArrayBuffer)=>{const bytes=new Uint8Array(raw);let value='';for(let i=0;i<bytes.length;i+=8192)value+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(value);};
  const localLpcJobs=new BrowserJobs<{analysis:LpcTaskConfig;textgrid:ResearchFile|null}>(owner,localProject,{
    list:async()=>((await(await request('assets?project_id='+encodeURIComponent(project)+'&order=created')).json()).assets),
    read:async asset=>(await readManaged({id:asset.id,name:asset.name,kind:'parameter',size:asset.size_bytes,sha256:asset.sha256},undefined,32000)).buffer,
    upload:async(name,raw)=>upload(name,raw,'browser-'+(await sha256(new TextEncoder().encode(name).buffer)).slice(0,32),uiMessage('platform.browser_lpc')),
    compute:async(file,config,signal)=>{try{const grid=config.textgrid;await verify(file);if(grid)await verify(grid);const args={name:file.name,config:config.analysis,textgrid:grid?encodeBase64(await browserRaw(grid,2_000_000)):null};const output=await batchScience.run<any>('lpc',await browserRaw(file),args,AbortSignal.any([abort.signal,signal]));await verify(file);if(grid)await verify(grid);return output;}catch(error){if(!signal.aborted&&!(error instanceof LocalStorageError))browserFailures.set('lpc',Date.now());throw error;}}
  },{prefix:'LPC任务-',operation:'lpc_analysis',kind:'managed_lpc_files',names:['lpc.ptb.json','lpc_SPECTRUM.png','lpc_AUDIO.wav']},project);
  async function browserTry<T>(operation:string,work:()=>Promise<T>,fallback:()=>Promise<T>):Promise<T>{
    if(typeof Worker==='undefined'||Date.now()-(browserFailures.get(operation)??0)<30000)return fallback();
    try{return await work();}catch(error){if(error instanceof LocalStorageError||disposed||abort.signal.aborted||error instanceof DOMException&&error.name==='AbortError')throw error;browserFailures.set(operation,Date.now());console.warn('Browser compute unavailable; using server:',operation);return fallback();}
  }
  const vocalPorts=new Set<VocalPort>();
  const abort=new AbortController();let disposed=false,listedAtCapacity=false;
  const notifyInvalid=onInvalid;onInvalid=()=>{if(!disposed&&!abort.signal.aborted)notifyInvalid();};
  const local=new LocalAssets(owner,localProject,database);
  const transport=new LocalAssetTransport(local,async(path,init)=>{const r=await fetch('/api/v1/'+path,{...init,credentials:'same-origin',cache:'no-store',headers:{'X-PTB-Account':owner,...(init?.body?{'Content-Type':'application/json'}:{}),...(init?.method&&init.method!=='GET'?{'X-CSRF-Token':csrf?.()??''}:{}),...init?.headers}});if(r.status===401||r.status===409&&(await r.clone().json().catch(()=>({}))).detail==='account_changed')onInvalid();return r;},abort.signal,project);
  const assetFetch=transport.fetch.bind(transport);
  async function request(path:string,signal?:AbortSignal,method='GET',body?:unknown,binary=false) {
    if(disposed)throw uiError(uiMessage('platform.the_project_session_is_closed_445'));
    const jobMutation=method!=='GET'&&path.startsWith('jobs/');if(jobMutation){jobLists.clear();jobDetails.clear();}
    const r=await assetFetch(path,{method,body:body?(binary?body as Blob:JSON.stringify(body)):undefined,credentials:'same-origin',cache:'no-store',signal:signal?AbortSignal.any([abort.signal,signal]):abort.signal,headers:{'X-PTB-Account':owner,...(body?{'Content-Type':binary?'application/octet-stream':'application/json'}:{}),...(method!=='GET'?{'X-CSRF-Token':csrf?.()??''}:{})}});
    if(!r.ok){const error=await r.json().catch(()=>({}));if(!disposed&&!abort.signal.aborted&&(r.status===401||error.detail==='account_changed'))onInvalid();if(r.status===422&&path==='jobs/spec2wav/create')throw uiError(uiMessage('platform.invalid_calibration_parameters_check_time_frequency_db_ranges_and'));if(typeof error.detail==='string'&&(error.detail.startsWith('egg_')||error.detail.startsWith('m09_')||error.detail.startsWith('m08_')||error.detail.startsWith('m06_')||error.detail.startsWith('preview_')||error.detail.startsWith('m07_')||['invalid_spectrogram_input','invalid_spectrogram_config','invalid_spectrogram_image','invalid_image_corners','spectrogram_budget'].includes(error.detail)))throw uiError(({m06_platform_unverified:uiMessage('platform.server_acoustic_parameter_synthesis_is_temporarily_unavailable_retry_later'),m08_platform_unverified:uiMessage('platform.server_time_and_pitch_processing_is_temporarily_unavailable_retry'),m06_runtime_unavailable:uiMessage('platform.the_server_synthesis_service_is_temporarily_unavailable_retry_later'),m08_runtime_unavailable:uiMessage('platform.the_server_pitch_processing_service_is_temporarily_unavailable_retry')} as Record<string,MessageDescriptor>)[error.detail]??error.detail);throw uiError(({m06_platform_unverified:uiMessage('platform.server_acoustic_parameter_synthesis_is_temporarily_unavailable_retry_later'),m08_platform_unverified:uiMessage('platform.server_time_and_pitch_processing_is_temporarily_unavailable_retry'),upload_size_limit:uiMessage('platform.the_file_exceeds_the_single_file_upload_limit'),disk_space_low:uiMessage('platform.the_server_has_insufficient_space_clean_up_files_or'),upload_closed:uiMessage('platform.the_upload_is_closed_select_the_file_again'),storage_busy:uiMessage('platform.the_file_service_is_busy_retry_later'),web_server_busy:uiMessage('platform.the_server_is_busy_wait_5_seconds_and_retry'),account_queue_full:uiMessage('platform.you_have_reached_the_limit_of_8_pending_tasks'),server_queue_full:uiMessage('platform.the_server_queue_is_full_retry_later'),workspace_expired:uiMessage('platform.this_workspace_has_expired_return_to_the_project_page'),input_unavailable:uiMessage('platform.the_source_file_is_unavailable_or_nearly_expired_upload'),quota_exceeded:uiMessage('platform.file_storage_is_full_remove_unneeded_files_and_retry'),invalid_parameter_table:uiMessage('platform.the_parameter_table_contains_unsupported_content_formulas_or_damaged'),parameter_read_failed:uiMessage('platform.the_parameter_table_exceeds_the_reading_budget_or_could'),parameter_read_timeout:uiMessage('platform.parameter_table_reading_timed_out_reduce_the_data_size'),asset_expired:uiMessage('platform.the_file_has_expired_upload_it_again'),invalid_textgrid:uiMessage('platform.the_textgrid_format_is_unsupported_or_incomplete'),unsupported_textgrid:uiMessage('platform.the_textgrid_exceeds_2_mb_or_has_an_incorrect')} as Record<string,MessageDescriptor>)[error.detail]||(r.status>=500?uiMessage('platform.the_server_cannot_complete_the_request_right_now_retry'):uiMessage('platform.the_request_did_not_finish_refresh_and_retry')));}
    if(jobMutation){jobLists.clear();jobDetails.clear();}
    return r;
  }
  async function verify(file:ResearchFile){const r=await request('assets/'+encodeURIComponent(file.id));const asset:components['schemas']['AssetView']=await r.json();if(asset.project_id!==project||asset.state!=='ready'||asset.sha256!==file.sha256)throw uiError(uiMessage('platform.the_file_has_changed_refresh_the_project_list'));return asset;}
  async function readManaged(file:ResearchFile,signal:AbortSignal|undefined,limit:number){const verified=await verify(file),expected=file.size??verified.size_bytes;if(expected>limit)throw uiError(uiMessage('platform.the_file_exceeds_the_current_preview_limit'));const response=await request('assets/'+encodeURIComponent(file.id)+'/content',signal);const reader=response.body?.getReader();if(!reader)throw uiError(uiMessage('platform.the_file_could_not_be_read'));const chunks:Uint8Array[]=[];let size=0;try{while(true){const {done,value}=await reader.read();if(done)break;size+=value.length;if(size>limit||size>expected)throw uiError(uiMessage('platform.the_file_length_has_changed'));chunks.push(value);}}finally{await reader.cancel();}if(size!==expected)throw uiError(uiMessage('platform.the_file_download_is_incomplete'));const bytes=new Uint8Array(size);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}const hash=await sha256(bytes.buffer);if(hash!==file.sha256)throw uiError(uiMessage('platform.file_verification_failed_reload_it'));return {buffer:bytes.buffer,sha256:hash};}
  const jobLists=sharedJobReads<JobView[]>(async()=>((await(await request('jobs?project_id='+encodeURIComponent(project))).json()) as components['schemas']['JobList']).jobs);
  const jobDetails=sharedJobReads<JobView>(async id=>(await request('jobs/'+encodeURIComponent(id))).json());
  const tasks:ResearchTasks={async parent(file){if(!file.sha256)return null;return (await request('jobs/parents/latest?'+new URLSearchParams({project_id:project,sha256:file.sha256}))).json();},async submit(operation,inputs,config,layer,key){const mapped=inputs.map(item=>Object.fromEntries(Object.entries(item).filter(([,v])=>v!=null).map(([role,value])=>{if(role==='parent_result')return [role,value];const file=value as ResearchFile;if(!file.sha256)throw uiError(uiMessage('platform.the_file_has_no_checksum_refresh_the_list'));return [role,{asset_id:file.id,sha256:file.sha256}];})));return (await request('jobs/batches/create',undefined,'POST',{project_id:project,operation,inputs:mapped,config,layer,idempotency_key:key})).json();},
    async reconstruct(file,config,key){if(!file.sha256)throw uiError(uiMessage('platform.the_image_has_no_checksum'));if(typeof Worker!=='undefined'&&(navigator.hardwareConcurrency??4)>=4&&((navigator as Navigator&{deviceMemory?:number}).deviceMemory??4)>=4&&Date.now()-(browserFailures.get('reconstruction')??0)>=30000&&(config.mode!=='audio_draw'||file.name.toLowerCase().endsWith('.wav'))&&JSON.stringify(config).length<=16000&&config.algorithm==='griffin_lim'&&config.vocoder==='none'&&config.time_end-config.time_start<=5&&config.n_iter<=64&&file.size<=(config.mode==='audio_draw'?4_000_000:1_000_000)){await verify(file);return localJobs.start(file,config);}return (await request('jobs/spec2wav/create',undefined,'POST',{project_id:project,idempotency_key:key,image:{asset_id:file.id,sha256:file.sha256},config})).json();},
    async reconstructionPreview(file,config){if(!file.sha256)throw uiError(uiMessage('platform.the_input_has_no_checksum'));await verify(file);const value=await browserTry('prediction',async()=>science.run<any>('prediction',await browserRaw(file,16_000_000),config,abort.signal),async()=>(await request('jobs/spec2wav/preview',undefined,'POST',{project_id:project,idempotency_key:crypto.randomUUID(),image:{asset_id:file.id,sha256:file.sha256},config})).json());await verify(file);return value;},
    async egg(file,config,key){if(!file.sha256)throw uiError(uiMessage('platform.the_audio_has_no_checksum'));const {exportFontSnapshot}=await import('../state/fonts.ts');return (await request('jobs/egg/create',undefined,'POST',{project_id:project,idempotency_key:key,audio:{asset_id:file.id,sha256:file.sha256},config:{...config,font:config.font??exportFontSnapshot()}})).json();},
    async eggJobs(){return (await jobLists.read('project')).filter(j=>j.operation==='egg_analysis');},
    async eggFonts(font){return (await request('jobs/egg/fonts',undefined,'POST',font)).json();},
    async lpc(file,textgrid,config,key){if(!file.sha256||(textgrid&&!textgrid.sha256))throw uiError(uiMessage('platform.the_file_has_no_checksum_refresh_the_list'));if(typeof Worker!=='undefined'&&Date.now()-(browserFailures.get('lpc')??0)>=30000&&file.size<=64_000_000){await verify(file);if(textgrid)await verify(textgrid);return localLpcJobs.start(file,{analysis:config,textgrid});}return (await request('jobs/lpc/create',undefined,'POST',{project_id:project,idempotency_key:key,audio:{asset_id:file.id,sha256:file.sha256},textgrid:textgrid?{asset_id:textgrid.id,sha256:textgrid.sha256}:null,config})).json();},
    async lpcJobs(){const [server,local]=await Promise.all([jobLists.read('project'),localLpcJobs.list()]);return [...local,...server.filter(j=>j.operation==='lpc_analysis')].sort((a,b)=>b.created_at-a.created_at);},
    async lpcFonts(font){return browserTry('lpc',async()=>{const value=await batchScience.run<FontPreflight>('lpc_fonts',new ArrayBuffer(0),font,abort.signal);if(!value.available)throw Error('Requested browser export font is unavailable');return value;},async()=>(await request('jobs/lpc/fonts',undefined,'POST',font)).json());},
    async reconstructions(){const [server,local]=await Promise.all([jobLists.read('project'),localJobs.list()]);return [...local,...server.filter(j=>j.operation==='spectrogram_to_audio')].sort((a,b)=>b.created_at-a.created_at);},
    async cancelJob(id){if(localJobs.has(id))return localJobs.cancel(id);if(localLpcJobs.has(id))return localLpcJobs.cancel(id);return (await request('jobs/'+encodeURIComponent(id)+'/cancel',undefined,'POST')).json();},
    async result(_job,id,sha){const response=await request('assets/'+encodeURIComponent(id)+'/content');const raw=await response.arrayBuffer();if(raw.byteLength>64_000_000||await sha256(raw)!==sha)throw uiError(uiMessage('platform.result_verification_failed_484'));return raw;},
    async list(){return (await(await request('jobs/batches/list?project_id='+encodeURIComponent(project))).json()).batches;},
    async get(id){return (await request('jobs/batches/'+encodeURIComponent(id))).json();},
    async cancel(id){return (await request('jobs/batches/'+encodeURIComponent(id)+'/cancel',undefined,'POST')).json();},
    async job(id){return localJobs.get(id)??localLpcJobs.get(id)??jobDetails.read(id);},
    async retry(id,key){const lpc=localLpcJobs.source(id);if(lpc){if(localLpcJobs.get(id)?.state!=='failed'||!localLpcJobs.serverRetry(id))return tasks.lpc!(lpc.file,lpc.config.textgrid,lpc.config.analysis,key);await verify(lpc.file);if(lpc.config.textgrid)await verify(lpc.config.textgrid);return(await request('jobs/lpc/create',undefined,'POST',{project_id:project,idempotency_key:key,audio:{asset_id:lpc.file.id,sha256:lpc.file.sha256},textgrid:lpc.config.textgrid?{asset_id:lpc.config.textgrid.id,sha256:lpc.config.textgrid.sha256}:null,config:lpc.config.analysis})).json();}const source=localJobs.source(id);if(source){if(localJobs.get(id)?.state!=='failed'||!localJobs.serverRetry(id))return tasks.reconstruct!(source.file,source.config,key);await verify(source.file);return(await request('jobs/spec2wav/create',undefined,'POST',{project_id:project,idempotency_key:key,image:{asset_id:source.file.id,sha256:source.file.sha256},config:source.config})).json();}return (await request('jobs/'+encodeURIComponent(id)+'/retry',undefined,'POST',{idempotency_key:key})).json();},
    async download(id,name){const response=await request('assets/'+encodeURIComponent(id)+'/content');const blob=await response.blob();if(disposed)return;const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}};
  async function browserRaw(file:ResearchFile,limit=64_000_000){
    await verify(file);const key=file.id+':'+file.sha256;let raw=browserSources.get(key);
    if(!raw){raw=(await readManaged(file,undefined,limit)).buffer;for(const id of browserSources.keys()){if([...browserSources.values()].reduce((n,b)=>n+b.byteLength,0)+raw.byteLength<=64_000_000)break;browserSources.delete(id);}if(raw.byteLength<=64_000_000)browserSources.set(key,raw);}
    if(raw.byteLength>limit)throw uiError(uiMessage('platform.the_file_exceeds_the_current_preview_limit'));return raw.slice(0);
  }
  const adapter:ResearchFiles={kind:'server',localFiles:true,annotationScope:localProject,feedbackCsrf:csrf,tasks:csrf?tasks:undefined,async list(){const shadows=new Set((await local.list()).filter(f=>f.server?.project===project&&f.server.id!==f.id).map(f=>f.server!.id));const data:components['schemas']['AssetList']=await(await request('assets?project_id='+encodeURIComponent(project)+'&order=created')).json();listedAtCapacity=data.assets.length>=1000;return data.assets.filter(a=>!shadows.has(a.id)&&a.project_id===project&&a.state==='ready'&&fileKind(a.name)).sort((a,b)=>b.created_at-a.created_at).map(a=>({id:a.id,name:a.name,kind:fileKind(a.name)!,size:a.size_bytes,sha256:a.sha256??undefined,expiresAt:a.expires_at}));},
    async associatedParameters(file){
      if(!file.sha256)return null;
      const table:{asset_id:string;name:string;sha256:string}|null=await(await request('jobs/parents/latest?'+new URLSearchParams({project_id:project,sha256:file.sha256,table:'true'}))).json();
      if(!table)return null;
      const associated:ResearchFile={id:table.asset_id,name:table.name,kind:'parameter',sha256:table.sha256,size:0};
      const asset=await verify(associated);associated.size=asset.size_bytes;associated.expiresAt=asset.expires_at;return associated;
    },
    async read(file,signal){if(signal?.aborted)throw new DOMException('Cancelled','AbortError');if(['audio','image','parameter'].includes(file.kind)){const buffer=await browserRaw(file,file.kind==='audio'?64_000_000:16_000_000);if(signal?.aborted)throw new DOMException('Cancelled','AbortError');return {buffer,sha256:file.sha256!};}return readManaged(file,signal,2_000_000);},
    async textgrid(file){const raw=await readManaged(file,undefined,2_000_000),grid=parseGrid(decodeText(raw.buffer));return {sha256:raw.sha256,tiers:grid.tiers.filter(t=>t.intervals).map(t=>({name:t.name,intervals:t.intervals!}))};},
    async spectrogram(file,view){await verify(file);const data:Spectrogram=await browserTry('spectrogram',()=>browserRaw(file).then(raw=>science.run<Spectrogram>('spectrogram',raw,view,abort.signal)),async()=>{const query=new URLSearchParams(Object.entries(view).map(([k,v])=>[k,String(v)]));return(await request('assets/'+encodeURIComponent(file.id)+'/spectrogram?'+query)).json();});await verify(file);if(data.sha256!==file.sha256)throw uiError(uiMessage('platform.the_audio_has_changed_refresh_the_list'));return data;},
    async parameters(file){await verify(file);const data:ParameterTable=await browserTry('parameters',()=>browserRaw(file,16_000_000).then(raw=>science.run<ParameterTable>('parameters',raw,{name:file.name},abort.signal)),async()=>(await request('assets/'+encodeURIComponent(file.id)+'/parameters')).json());await verify(file);if(data.sha256!==file.sha256)throw uiError(uiMessage('platform.the_parameter_file_has_changed_refresh_the_list'));return data;},
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
        if(!r.ok){const e=await r.json().catch(()=>({})),code=e.detail;if(r.status===401||code==='account_changed')onInvalid();if(e.display)throw responseError(code,e.display);throw uiError(({vocal_unavailable:uiMessage('platform.the_server_vocal_tract_engine_has_not_started_retry'),vocal_capacity_full:uiMessage('platform.the_vocal_tract_computation_slot_is_in_use_reopen'),vocal_busy:uiMessage('platform.the_server_is_computing_retry_later'),vocal_cancelled:uiMessage('platform.generation_cancelled_490'),vocal_session_expired:uiMessage('platform.the_vocal_tract_session_has_ended_reopen_articulatory_synthesis'),vocal_session_not_found:uiMessage('platform.the_vocal_tract_session_has_ended_reopen_articulatory_synthesis'),vocal_timeout:uiMessage('platform.vocal_tract_computation_timed_out_reopen_it_and_shorten'),vocal_runtime_unavailable:uiMessage('platform.the_vocal_tract_process_exited_reopen_it'),workspace_expired:uiMessage('platform.this_workspace_has_expired_return_to_the_project_page_494')} as Record<string,MessageDescriptor>)[code]||code||uiMessage('platform.the_vocal_tract_request_failed'));}
        return r.json();
      },project,key,undefined,vocalProfileStore(local));vocalPorts.add(port);return port;
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
          try{while(['queued','running','cancel_requested'].includes(job.state)){if(current.aborted)throw new DOMException('Cancelled','AbortError');await waitForTaskPoll(job.state,current);if(current.aborted)throw new DOMException('Cancelled','AbortError');job=await adapter.recording!.job(job.id);}if(job.state!=='succeeded')throw uiError(job.error_code||uiMessage('platform.recording_processing_failed'));return adapter.recording!.result(job);}
          catch(error){if(current.aborted)await adapter.recording!.cancel(job.id).catch(()=>{});throw error;}
        });
      },
      async upload(file){let key=selectedUploads.get(file);if(!key){key='m16-'+crypto.randomUUID();selectedUploads.set(file,key);}return recordingAsset(await upload(file.name,await file.arrayBuffer(),key,uiMessage('platform.recording')));},
      async load(){for(let attempt=0;;attempt++){const {asset}=await(await request('jobs/m16/project?project_id='+project)).json();if(!asset)return null;try{const raw=await(await request('assets/'+asset.id+'/content')).arrayBuffer();return {snapshot:JSON.parse(new TextDecoder().decode(raw)),id:asset.id};}catch(error){if(disposed||abort.signal.aborted||attempt===2)throw error;await new Promise(resolve=>setTimeout(resolve,75));}}},
      async save(snapshot,expected){const raw=JSON.stringify(snapshot);const commit=async()=>{const saved=await(await request('jobs/m16/project',undefined,'POST',{project_id:project,asset_id:recordingSave!.asset.id,expected:recordingSave!.expected})).json();recordingSave=undefined;return saved.id as string;};if(recordingSave){const same=recordingSave.raw===raw;expected=await commit();if(same)return expected;}const key=crypto.randomUUID();recordingSave={raw,asset:await upload(key+'.m16-draft.json',new TextEncoder().encode(raw).buffer,'m16-project-'+key,uiMessage('platform.recording_project_498')),expected};return commit();},
      async create(config,key){return (await request('jobs/m16/create',undefined,'POST',{project_id:project,idempotency_key:key,config})).json();},
      job:tasks.job,cancel:tasks.cancelJob!,
      async read(asset){return (await readManaged(asset,undefined,100_000_000)).buffer;},
      async remove(asset){await verify(asset);await request('assets/'+asset.id,undefined,'DELETE');},
      async spectrum(asset,start,end,channel){const value:any=await browserTry('recording_spectrum',async()=>science.run('recording_spectrum',await browserRaw(asset,100_000_000),{start,end,channel,width:640},abort.signal),async()=>{const value=await(await request('assets/'+asset.id+'/recording-preview?'+new URLSearchParams({start:String(start),end:String(end),channel:String(channel)}))).json();if(value.sha256!==asset.sha256)throw uiError(uiMessage('platform.the_recording_has_changed_refresh_the_list'));return value;});await verify(asset);return recordingSpectrum(value);},
      async live(file,frames,channel){return recordingSpectrum(await browserTry('recording_spectrum',async()=>science.run('recording_spectrum',await file.arrayBuffer(),{start:0,end:frames,channel,width:640},abort.signal),async()=>(await request('jobs/m16/live-preview?'+new URLSearchParams({start:'0',end:String(frames),channel:String(channel)}),undefined,'POST',file,true)).json()));},
      async result(job){const manifest=job.result_manifest as {files?:any[]}|null,files=manifest?.files;if(!files)throw uiError(uiMessage('platform.the_recording_processing_result_is_incomplete'));const meta=files.find(f=>f.name.endsWith('.json'));if(!meta)throw uiError(uiMessage('platform.the_recording_processing_manifest_is_missing'));const metadata=JSON.parse(new TextDecoder().decode(await(await request('assets/'+meta.id+'/content')).arrayBuffer()));const wav=files.find(f=>f.name.endsWith('.wav'));return {asset:wav?recordingAsset(wav):undefined,metadata};}
    };
    // Keep each selected File's key across retry, including a lost block/finalize reply.
    const selectedUploads=new WeakMap<File,string>();
    adapter.upload=async(values,progress)=>{
      for(const file of values){if(!fileKind(file.name))throw uiError(joinMessages(uiMessage('platform.unsupported_file_type'),file.name));if(file.size>100_000_000)throw uiError(joinMessages(file.name,uiMessage('platform.exceeds_the_100_mb_single_file_limit')));if(!file.size)throw uiError(joinMessages(file.name,uiMessage('platform.is_an_empty_file')));}
      const total=values.reduce((sum,f)=>sum+f.size,0);let completed=0;
      for(const file of values){
        let key=selectedUploads.get(file);if(!key){key=crypto.randomUUID();selectedUploads.set(file,key);}
        let asset:components['schemas']['AssetView']=await(await request('uploads',undefined,'POST',{project_id:project,name:file.name,expected_bytes:file.size,idempotency_key:key})).json();
        if(!['uploading','ready'].includes(asset.state))throw uiError(uiMessage('platform.the_upload_is_closed_select_the_file_again'));
        while(asset.state==='uploading'&&asset.size_bytes<file.size){
          const offset=asset.size_bytes;
          asset=await(await request('uploads/'+asset.id+'/blocks?offset='+offset,undefined,'PUT',file.slice(offset,offset+262144),true)).json();
          if(asset.size_bytes<=offset||asset.size_bytes>file.size)throw uiError(uiMessage('platform.upload_progress_is_inconsistent_try_again'));
          progress((completed+asset.size_bytes)/total);
        }
        await request('uploads/'+asset.id+'/finalize',undefined,'POST',{});
        completed+=file.size;progress(completed/total);
      }
    };
    adapter.m07=m07Port({project,async source(file){await verify(file);const saved=await local.get(file.id);return {asset_id:saved?await transport.server(saved):file.id,sha256:file.sha256!};},async create(body){return (await request('jobs/m07/create',undefined,'POST',body)).json();},job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!,retry:tasks.retry,async list(){return (await jobLists.read('project')).filter(j=>j.operation==='phonation_synthesis');}});
    const saveBrowserOutput=async(name:string,raw:ArrayBuffer,signal:AbortSignal)=>{signal.throwIfAborted();const asset=await upload(name,raw,'browser-'+(await sha256(new TextEncoder().encode(name).buffer)).slice(0,32),uiMessage('platform.browser_result'));signal.throwIfAborted();return {id:asset.id,sha256:asset.sha256,size_bytes:asset.size};};
    adapter.m06=browserM06(m06Port({project,async parameters(text,signal){const a=await upload('m06-parameters.csv',new TextEncoder().encode(text).buffer,'m06-'+crypto.randomUUID(),uiMessage('platform.synthesis_parameters'));if(signal.aborted)throw uiError(uiMessage('platform.operation_cancelled'));return {asset_id:a.id,sha256:a.sha256!};},async source(file){await verify(file);return {asset_id:file.id,sha256:file.sha256!};},
      async create(body){return (await request('jobs/m06/create',undefined,'POST',body)).json();},job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!}),{project,
      compute:(args,signal)=>browserTry('speech_synthesis',()=>batchScience.run<any>('speech_synthesis',new ArrayBuffer(0),args,AbortSignal.any([abort.signal,signal])),async()=>null),save:saveBrowserOutput});
    adapter.papers=serverPapers(owner,project,csrf!,onInvalid,abort.signal,assetFetch);
    adapter.m11=serverM11(owner,project,csrf,onInvalid,abort.signal,assetFetch);
    const phonologyFile=(source:{asset_id:string;sha256:string;name:string}):ResearchFile=>({id:source.asset_id,sha256:source.sha256,name:source.name,kind:'parameter',size:0});
    adapter.m14=browserM14(serverM14(owner,project,csrf,onInvalid,abort.signal,assetFetch),{
      async upload(file,signal){signal.throwIfAborted();const asset=await upload(file.name,await file.arrayBuffer(),'m14-browser-'+crypto.randomUUID(),uiMessage('platform.survey_table'));signal.throwIfAborted();return {asset_id:asset.id,sha256:asset.sha256,name:file.name};},
      async read(source,signal){signal.throwIfAborted();const asset=await verify(phonologyFile(source));const raw=await browserRaw({...phonologyFile(source),size:asset.size_bytes},2_000_000);signal.throwIfAborted();return raw;},
      async verify(source,signal){signal.throwIfAborted();await verify(phonologyFile(source));signal.throwIfAborted();},
      compute:(raw,args,signal)=>browserTry('phonology',()=>batchScience.run<any>('phonology',raw,args,AbortSignal.any([abort.signal,signal])),async()=>null),save:saveBrowserOutput
    });
    adapter.m08=m08Port({project,async source(file){await verify(file);const saved=await local.get(file.id);return {asset_id:saved?await transport.server(saved):file.id,sha256:file.sha256!};},
      async request(action,body){return (await request('jobs/m08/'+(action==='list'?'list/'+encodeURIComponent(project):action),undefined,action==='list'?'GET':'POST',body)).json();},
      job:tasks.job,read:tasks.result!,cancel:tasks.cancelJob!});

    const annotation=portableAnnotation(adapter,upload);
    adapter.annotation={...annotation,async scan(){const values=await adapter.list();if(listedAtCapacity)throw uiError(uiMessage('platform.project_resources_reached_the_current_1000_entry_list_limit'));const seen=new Set<string>();return values.filter(f=>{const key=f.name.toLowerCase();if(seen.has(key))return false;seen.add(key);return true;});}};
  }
  return adapter;
}
