import type {ResearchFile,JobView} from './research.ts';
import type {Project,Spectrum,Config,Task} from '../modules/recording/types.ts';
export type RecordingAsset=ResearchFile & {sha256:string};
export type RecordingCapture={config:Config;task:Task|null;started:string;chunks:{asset:RecordingAsset;frames:number}[]};
export type RecordingSnapshot={schema:'m16-web-project/1';project:Project;assets:Record<string,RecordingAsset>;captures?:Record<string,RecordingCapture>};
export interface RecordingServer {
 compute?(config:Record<string,unknown>,signal?:AbortSignal):Promise<{asset?:RecordingAsset;metadata:Record<string,unknown>}>;
 upload(file:File):Promise<RecordingAsset>;
 load():Promise<{snapshot:RecordingSnapshot;id:string}|null>;
 save(snapshot:RecordingSnapshot,expected:string|null):Promise<string>;
 create(config:Record<string,unknown>,key:string):Promise<JobView>;
 job(id:string):Promise<JobView>;
 cancel(id:string):Promise<JobView>;
 read(asset:RecordingAsset):Promise<ArrayBuffer>;
 remove(asset:RecordingAsset):Promise<void>;
 spectrum(asset:RecordingAsset,start:number,end:number,channel:number):Promise<Spectrum>;
 live(file:File,frames:number,channel:number):Promise<Spectrum>;
 result(job:JobView):Promise<{asset?:RecordingAsset;metadata:Record<string,unknown>}>;
}
export function recordingSpectrum(raw:any):Spectrum {
 const bytes=Uint8Array.from(atob(raw.pixels_base64),c=>c.charCodeAt(0)),height=raw.frequencies.length;
 if(bytes.length!==raw.columns*height)throw Error('语谱图尺寸不一致。');
 return {...raw,rows:Array.from({length:raw.columns},()=>[]),pixels:Array.from({length:raw.columns},(_,i)=>Array.from(bytes.slice(i*height,(i+1)*height)))};
}
