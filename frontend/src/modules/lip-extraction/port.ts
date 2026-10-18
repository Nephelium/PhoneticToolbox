export interface OfflineConfig {filter_enabled:boolean;cutoff_hz:number;}
export interface LipRow {index:number;time_s:number;detected:boolean;points:[number,number][]|null;metrics:Record<string,number|null>|null;width?:number;height?:number;input_resolution?:[number,number];}
export interface LipResult {id:string;name:string;backend:string;rows:LipRow[];metadata:Record<string,any>;files:{id:string;name:string;bytes:number;sha256:string}[];}
export interface LipSaveOptions {video:boolean;animation:boolean;quality:'high'|'standard'|'small';}
export interface LipSaveItem {result:LipResult;offset:number;}
export interface LipPort {
 kind:'desktop'|'browser';
 available:boolean;
 prepareCapture?(purpose?:'capture'|'devices'):Promise<void>;
 captureError?(message:string,purpose?:'capture'|'devices'):Promise<string>;
 history?():Promise<{id:string;name:string}[]>;
 load?(id:string):Promise<LipResult>;
 replay?(result:LipResult,start:number):Promise<{rows:LipRow[];complete:boolean}>;
 inspectRecording?(blob:Blob):Promise<Record<string,any>>;
 audioPreview?(result:LipResult):Promise<Record<string,any>>;
 saveRecording?(name:string,blob:Blob,metadata:Blob):Promise<{saved:boolean;recording?:Record<string,any>}>;
 analyzeRecording?(token:string,config:OfflineConfig,signal:AbortSignal,progress:(s:string)=>void):Promise<LipResult>;
 reason?:string;
 analyze(file:File,config:OfflineConfig,signal:AbortSignal,progress:(message:string)=>void):Promise<LipResult>;
 save(result:LipResult,offset:number,action:'apply'|'save_without_offset'):Promise<boolean>;
 saveBatch?(items:LipSaveItem[],action:'apply'|'save_without_offset',options:LipSaveOptions,onSaved:(id:string)=>void):Promise<boolean>;
 sourceVideo?(result:LipResult):Promise<Blob>;
 playbackVideo?(result:LipResult):Promise<{blob:Blob;origin_s:number}>;
 exportAnimation?(result:LipResult,format:'mp4'|'gif',quality:'high'|'standard'|'small',signal:AbortSignal):Promise<boolean>;
 saveLocal?(name:string,blob:Blob):Promise<boolean>;
}
