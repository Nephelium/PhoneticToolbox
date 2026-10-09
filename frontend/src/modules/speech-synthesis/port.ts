import type {Config} from './state.ts';
import type {JobView,ResearchFile} from '../../platform/research.ts';
export type Action='generate'|'synthesize'|'extract';
export interface Raster {width:number;height:number;pixels:string;t0:number;t1:number;fmax:number;low_db:number;high_db:number}
export interface Result {job:string;files:{id:string;name:string;sha256:string;size_bytes:number}[];metadata:{diagnostics?:{output_gain?:number;actual_f0_backend?:string;voiced_frames?:number;voiced_mask?:boolean[]};config:Config;seed:number;input_sha256:string;action:Action;spectrograms:Record<string,Raster>};wav?:ArrayBuffer}
export interface M06Port {run(action:Action,config:Config,file:ResearchFile|undefined,signal:AbortSignal,progress:(job:JobView)=>void):Promise<Result>;download(result:Result,name:string):Promise<void>;cancel(id:string):Promise<unknown>}
