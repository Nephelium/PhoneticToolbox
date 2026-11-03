import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {waitForTaskPoll} from './task-poll.ts';
import type {ResearchFile,JobView} from './research.ts';
import type {M08Port,Result,Job,Config,ExportReport} from '../modules/pitch-manipulation/port.ts';
import {m08Archive} from './m08Archive.ts';
import {sha256} from './research.ts';

type Ref={asset_id:string;sha256:string};
type Managed=Result&{job_id:string;sha256:string;saved:boolean};
export interface M08Transport {
 project:string;
 source(file:ResearchFile):Promise<Ref>;
 request<T>(action:string,body?:unknown):Promise<T>;
 job(id:string):Promise<JobView>;
 read(job:string,id:string,sha:string):Promise<ArrayBuffer>;
 cancel(id:string):Promise<unknown>;
 export?(result:Managed):Promise<string>;
 exportMany?(results:Managed[]):Promise<ExportReport>;
}
/** Both production adapters use these exact task/ID contracts. No local compute. */
export function m08Port(transport:M08Transport):M08Port {
 const refs=new Map<string,Ref>(),localIds=new Map<string,string>(),results=new Map<string,Managed>();
 async function source(file:ResearchFile){const ref=await transport.source(file);refs.set(file.id,ref);localIds.set(ref.asset_id,file.id);return ref;}
 const mapped=(r:Managed):Managed=>{results.set(r.id,r);return {...r,source_id:localIds.get(r.source_id)??r.source_id};};
 async function jobs(){const values=await transport.request<Job[]>('list');return values.map(j=>({...j,source_id:localIds.get(j.source_id)??j.source_id,results:j.results.map(r=>mapped(r as Managed))}));}
 async function wait(job:JobView,signal?:AbortSignal){
  const until=Date.now()+150000;
  while(['queued','running','cancel_requested'].includes(job.state)){
   if(signal?.aborted){await transport.cancel(job.id);throw new DOMException(tr('platform.operation_cancelled'),'AbortError');}
   if(Date.now()>until)throw uiError(uiMessage('platform.the_task_is_still_running_check_or_cancel_it'));
   await waitForTaskPoll(job.state,signal);if(signal?.aborted)continue;job=await transport.job(job.id);
  }
  if(job.state!=='succeeded')throw uiError(job.error_code??uiMessage('platform.the_time_and_pitch_task_did_not_finish'));return job;
 }
 async function submit(file:ResearchFile,config:Config,key:string){
  const ref=await source(file);const j=await transport.request<JobView>('create',{project_id:transport.project,audio:ref,config,idempotency_key:key});
  return {id:j.id,state:j.state,source_id:file.id,source_name:file.name,source_sha256:ref.sha256,config,results:[],error:j.error_code??undefined} as Job;
 }
 async function scope(ids:string[]){
  const values=ids.map(id=>results.get(id));if(!values.length||values.some(v=>!v))throw uiError(uiMessage('platform.results_have_changed_refresh_the_list'));
  const sourceId=values[0]!.source_id;if(values.some(v=>v!.source_id!==sourceId))throw uiError(uiMessage('platform.only_explicit_results_from_the_same_source_audio_can'));
  const ref=[...refs.values()].find(v=>v.asset_id===sourceId);if(!ref)throw uiError(uiMessage('platform.source_file_permission_has_expired_select_the_file_again'));
  return {project_id:transport.project,source:ref,ids};
 }
 async function audio(r:Result){const value=results.get(r.id);if(!value)throw uiError(uiMessage('platform.results_are_no_longer_available_refresh_the_list'));const raw=await transport.read(value.job_id,value.id,value.sha256);if(await sha256(raw)!==value.sha256)throw uiError(uiMessage('platform.result_verification_failed'));return raw;}
 return {
  async preview(file,signal){
   const ref=await source(file);
   const created=await transport.request<JobView>('create',{project_id:transport.project,audio:ref,config:{action:'preview'},idempotency_key:'m08-preview-'+crypto.randomUUID()});
   const job=await wait(created,signal);const files=job.result_manifest&&'files' in job.result_manifest?job.result_manifest.files:[];
   const metadata=files.find(f=>f.name==='m08.ptb.json'),wav=files.find(f=>f.name==='m08-preview.wav');
   if(!metadata||!wav)throw uiError(uiMessage('platform.the_time_and_pitch_preview_file_is_incomplete'));
   const data=JSON.parse(new TextDecoder().decode(await transport.read(job.id,metadata.id,metadata.sha256)));
   if(data.audio_sha256!==ref.sha256)throw uiError(uiMessage('platform.source_audio_checksums_do_not_match'));
   return {...data,sha256:ref.sha256,wav:await transport.read(job.id,wav.id,wav.sha256)};
  },submit,jobs,cancel:async id=>{await transport.cancel(id);},audio,
  async history(file){const ref=await source(file);return (await transport.request<Managed[]>('history',{project_id:transport.project,source:ref})).map(mapped);},
  async save(r){const copy=await wait(await transport.request<JobView>('save',await scope([r.id])));const job=(await jobs()).find(j=>j.id===copy.id);if(!job?.results[0])throw uiError(uiMessage('platform.the_saved_result_was_not_published'));const saved=job.results[0];const name=await transport.export?.(results.get(saved.id)!);
   if(name&&name!==saved.name){await transport.request('rename',{...await scope([saved.id]),names:[name]});return mapped({...results.get(saved.id)!,name});}return saved;},
  async saveMany(selected,sequenceNumbers){
   if(!selected.length||selected.length>256||new Set(selected.map(r=>r.id)).size!==selected.length)throw uiError(uiMessage('platform.select_1_256_distinct_results'));
   const values=selected.map(r=>{
    const value=results.get(r.id);if(!value)throw uiError(uiMessage('platform.results_are_no_longer_available_refresh_the_list_296'));
    const number=sequenceNumbers?.[r.id];
    if(number!==undefined&&(!Number.isInteger(number)||number<1||number>256))throw uiError(uiMessage('platform.invalid_audio_index'));
    return {...value,name:number===undefined?value.name:`${number}_${value.name}`};
   });
   if(transport.exportMany)return transport.exportMany(values);
   // Ask while the Save click still has a browser user activation, before reads.
   const picker=(globalThis as any).showSaveFilePicker;
   let handle:any;
   if(picker)try{handle=await picker({suggestedName:values.length===1?values[0].name:'变速变调_批量音频.zip',types:[values.length===1?{description:tr('platform.wav_audio'),accept:{'audio/wav':['.wav']}}:{description:tr('platform.zip_audio_archive'),accept:{'application/zip':['.zip']}}]});}
   catch(e){if(e instanceof Error&&e.name==='AbortError')return {saved:[],failed:[],cancelled:true};throw e;}
   // One download avoids browser multi-download blocking. Keep owner/expiry/hash checks.
   const files:{name:string;raw:ArrayBuffer}[]=[],used=new Set<string>();let total=0;
   const saved=[];
   for(const r of values){
    if(!/\.wav$/i.test(r.name)||/[\x00-\x1f/\\:<>"|?*]/.test(r.name))throw uiError(uiMessage('platform.invalid_audio_filename'));
    const raw=await audio(r);total+=raw.byteLength;if(total>64_000_000)throw uiError(uiMessage('platform.the_selected_audio_exceeds_the_64_mb_download_budget'));
    let name=r.name,n=2;while(used.has(name.toLowerCase()))name=r.name.replace(/\.wav$/i,` (${n++}).wav`);
    used.add(name.toLowerCase());files.push({name,raw});saved.push({id:r.id,name});
   }
   const blob=files.length===1?new Blob([files[0].raw],{type:'audio/wav'}):new Blob([m08Archive(files)],{type:'application/zip'});
   if(handle){const writer=await handle.createWritable();try{await writer.write(blob);await writer.close();}catch(e){await writer.abort().catch(()=>{});throw e;}return {saved,failed:[],destination:handle.name};}
   const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=files.length===1?files[0].name:'变速变调_批量音频.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
   return {saved,failed:[]};
  },
  async remove(ids){return transport.request('remove',await scope(ids));},
  async rename(changes){await transport.request('rename',{...await scope(changes.map(c=>c.id)),names:changes.map(c=>c.name)});return (await jobs()).flatMap(j=>j.results).filter(r=>changes.some(c=>c.id===r.id));},
  async download(r){const bytes=await audio(r),url=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));const a=document.createElement('a');a.href=url;a.download=r.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);},
 };
}
