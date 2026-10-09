import type {ResearchFile} from '../../platform/research.ts';
export type Mode='full'|'order'|'reverse'|'constant';
export interface Point {time:number;freqs:number[];mode:Mode}
export interface Config {action:'preview'|'synthesize'|'transform'|'linear';start:number;end:number|null;speed:number;pitch_ratio:number;pitch_hz:number;modified_f0:number[]|null;points:Point[];offset:boolean}
export interface Track {times:number[];original_f0:number[]}
export interface Preview extends Track {sha256:string;wav:ArrayBuffer}
export interface Result extends Track {id:string;name:string;source_id:string;start:number;end:number;config:Config}
export interface Job {id:string;state:'queued'|'running'|'succeeded'|'failed'|'cancelled'|'cancel_requested'|'interrupted';source_id:string;results:Result[];error?:string}
export interface ExportReport {saved:{id:string;name:string}[];failed:{id:string;error:string}[];cancelled?:boolean;directory?:string}
/** Module-owned adapter contract. All writes require owner, expiry, quota and fencing
 * in the shared host. Missing adapter is explicitly unavailable, never simulated. */
export interface M08Port {
 preview(file:ResearchFile,signal:AbortSignal):Promise<Preview>;
 submit(file:ResearchFile,config:Config,key:string):Promise<Job>;
 jobs():Promise<Job[]>;
 cancel(id:string):Promise<void>;
 audio(result:Result):Promise<ArrayBuffer>;
 save(result:Result):Promise<Result>; // atomic V2 numbering in chosen owner destination
 saveMany?(results:Result[]):Promise<ExportReport>; // export existing WAVs, no copy job
 history(source:ResearchFile):Promise<Result[]>;
 remove(ids:string[]):Promise<{removed:string[];failed:{id:string;error:string}[]}>;
 rename(changes:{id:string;name:string}[]):Promise<Result[]>; // all-name preflight, no overwrite
 download(result:Result):Promise<void>;
}
