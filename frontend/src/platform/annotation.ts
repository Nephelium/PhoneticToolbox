import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import type {ResearchFiles,ResearchFile} from './research.ts';
import {sha256,audioPreview} from './research.ts';
import {decodeWav} from './decode.ts';
import {decodeText,parseGrid} from '../modules/annotation/format.ts';
export interface AnnotationTarget {id:string;name:string;sha256:string|null}
export interface AnnotationSaved {file:ResearchFile;name:string;sha256:string}
export interface AnnotationSave {target:string;source:{id:string;sha256:string};text?:string;offset?:number}
export interface AnnotationPort {
 audio?(file:ResearchFile):Promise<{buffer:ArrayBuffer;sha256:string;sourceDuration:number;previewNote:string}>;
 scan(directory?:string):Promise<ResearchFile[]>;
 lip(file:ResearchFile):Promise<{wire:unknown;sha256:string}>;
 target(file:ResearchFile,role:'textgrid'|'lip',suffix?:string):Promise<AnnotationTarget>;
 save(body:AnnotationSave):Promise<AnnotationSaved>;
}
export async function annotationAudio(files:ResearchFiles,file:ResearchFile,signal?:AbortSignal){
 if(!files.annotation?.audio)return {...await audioPreview(files,file,signal),previewNote:''};
 if(signal?.aborted)throw new DOMException('Preview cancelled','AbortError');
 const data=await files.annotation.audio(file),asset=await decodeWav(data.buffer,file.name,signal);
 if(!Number.isFinite(data.sourceDuration)||data.sourceDuration<=0||Math.abs(asset.duration-data.sourceDuration)>1/asset.sampleRate+1e-9)throw uiError(uiMessage('platform.the_preview_time_range_does_not_match_the_original'));
 // A fractional final resampling frame must not stretch or truncate TextGrid time.
 asset.duration=data.sourceDuration;
 return {asset,sha256:data.sha256,previewNote:data.previewNote};
}
export interface LipTrack {wire:any;times:number[];open:(number|null)[];width:(number|null)[];area:(number|null)[];circularity:(number|null)[];offset:number}
export function lipTrack(wire:unknown):LipTrack {
  const doc=wire as {schema?:string;data?:Record<string,any>};
  if(doc?.schema!=='ptb.lip/1'||!doc.data||typeof doc.data!=='object')throw uiError(uiMessage('platform.lip_data_requires_the_safe_ptb_lip_1_format'));
  const d=doc.data,meta=d.metadata??{};
  function vector(key:string):number[]{const v=d[key];if(!v)return [];if(!Array.isArray(v.values)||!Array.isArray(v.nonfinite)||v.values.length!==v.nonfinite.length||v.values.length>100000)throw uiError(uiMessage('platform.invalid_lip_vector_format_or_length'));return v.values.map((n:unknown,i:number)=>{const code=v.nonfinite[i];if(![0,1,2,3].includes(code)||((code===0)&&(typeof n!=='number'||!Number.isFinite(n)))||((code!==0)&&n!==null))throw uiError(uiMessage('platform.the_lip_data_numeric_mask_is_inconsistent'));return code===0?n as number:NaN;});}
  const absolute=vector('absolute_timestamps'),relative=vector('relative_times');
  const anchor=typeof meta.audio_first_frame_time==='number'&&Number.isFinite(meta.audio_first_frame_time)?meta.audio_first_frame_time:null;
  const anchored=anchor!==null&&absolute.length>0;
  let times=anchored?absolute.map(t=>t-anchor):relative;
  const open=vector('open'),width=vector('outer_width'),offset=meta.lip_manual_offset??0;
  if(!Number.isFinite(offset)||Math.abs(offset)>3600||open.length!==times.length)throw uiError(uiMessage('platform.lip_times_and_apertures_do_not_match_or_the'));
  const ordered=times.map((time,index)=>({time,index})).filter(v=>Number.isFinite(v.time)).sort((a,b)=>a.time-b.time);
  // V2 compares adjacent sorted timestamps before selecting the unique mask.
  const indices=ordered.filter((v,i)=>i===0||v.time-ordered[i-1].time>1e-9).map(v=>v.index);
  times=indices.map(i=>times[i]);
  if(times.length<2||indices.filter(i=>Number.isFinite(open[i])).length<2)throw uiError(uiMessage('platform.not_enough_valid_lip_data'));
  if(!anchored&&meta.time_alignment_mode!=='anchored_audio_start'){const first=times[0];times=times.map(t=>t-first);}
  const track=(values:number[])=>values.length===open.length?indices.map(i=>Number.isFinite(values[i])?values[i]:null):[];
  return {wire:structuredClone(wire),times,open:track(open),width:track(width),area:track(vector('area')),circularity:track(vector('circularity')),offset};
}
export function portableAnnotation(files:ResearchFiles,upload?:(name:string,buffer:ArrayBuffer,key:string)=>Promise<ResearchFile>):AnnotationPort {
  const prepared=new Map<string,{target:AnnotationTarget;role:'textgrid'|'lip';file:ResearchFile;key:string}>();
  return {
    scan:directory=>files.list(directory),
    async lip(file){if(file.kind==='lip_pickle')throw uiError(uiMessage('platform.the_web_version_does_not_read_pkl_convert_it'));const read=await files.read(file);const wire=JSON.parse(decodeText(read.buffer));lipTrack(wire);return {wire,sha256:read.sha256};},
    async target(file,role,suffix='_自动保存'){
      if(suffix.length>60||/[\x00-\x1f/\\:*?"<>|]/.test(suffix)||/[. ]$/.test(suffix))throw uiError(uiMessage('platform.invalid_file_extension'));
      const name=role==='textgrid'?file.name.replace(/\.wav$/i,'')+suffix+'.TextGrid':file.name;
      const existing=(await files.list()).find(f=>f.name===name);
      const target={id:crypto.randomUUID(),name,sha256:existing?.sha256??null};
      prepared.set(target.id,{target,role,file,key:crypto.randomUUID()});return target;
    },
    async save(body){
      const p=prepared.get(body.target);if(!p)throw uiError(uiMessage('platform.the_save_destination_is_no_longer_available'));
      const source=(await files.list()).find(f=>f.id===body.source.id);if(!source)throw uiError(uiMessage('platform.the_source_is_no_longer_available_your_edits_are'));
      const read=await files.read(source);if(read.sha256!==body.source.sha256)throw uiError(uiMessage('platform.the_source_has_changed_your_edits_are_preserved'));
      let text=body.text;
      if(p.role==='textgrid'){if(typeof text!=='string')throw uiError(uiMessage('platform.annotation_content_is_missing'));parseGrid(text);}
      else{if(source.kind==='lip_pickle')throw uiError(uiMessage('platform.the_web_version_does_not_write_pkl'));const wire=JSON.parse(decodeText(read.buffer));lipTrack(wire);if(!Number.isFinite(body.offset)||Math.abs(body.offset!)>3600)throw uiError(uiMessage('platform.invalid_lip_offset'));wire.data.metadata??={};wire.data.metadata.lip_manual_offset=body.offset;text=JSON.stringify(wire);}
      const buffer=new TextEncoder().encode(text).buffer;
      if(buffer.byteLength>2_000_000)throw uiError(uiMessage('platform.annotation_output_exceeds_2_mb'));
      const hash=await sha256(buffer);
      if(upload){const file=await upload(p.target.name,buffer,p.key);p.key=crypto.randomUUID();p.target.sha256=hash;return {file,name:file.name,sha256:hash};}
      const blob=new Blob([buffer]),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=p.target.name.split('/').at(-1)!;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
      const before=new Set((await files.list()).map(f=>f.id));files.add?.([new File([buffer],p.target.name)]);
      const stored=(await files.list()).find(f=>!before.has(f.id)&&f.name===p.target.name);
      if(!stored)throw uiError(uiMessage('platform.the_download_has_started_but_the_new_version_cannot'));
      return {file:{...stored,sha256:hash},name:p.target.name,sha256:hash};
    },
  };
}
