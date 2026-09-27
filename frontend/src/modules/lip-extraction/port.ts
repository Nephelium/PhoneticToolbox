export interface OfflineConfig {filter_enabled:boolean;cutoff_hz:number;}
export interface LipRow {index:number;time_s:number;detected:boolean;points:[number,number][]|null;metrics:Record<string,number|null>|null;width?:number;height?:number;input_resolution?:[number,number];}
export interface LipResult {id:string;name:string;backend:string;rows:LipRow[];metadata:Record<string,any>;files:{id:string;name:string;bytes:number;sha256:string}[];}
export interface LipPort {
 kind:'desktop'|'browser';
 available:boolean;
 prepareCapture?():Promise<void>;
 captureError?(message:string):Promise<string>;
 history?():Promise<{id:string;name:string}[]>;
 load?(id:string):Promise<LipResult>;
 reason?:string;
 analyze(file:File,config:OfflineConfig,signal:AbortSignal,progress:(message:string)=>void):Promise<LipResult>;
 save(result:LipResult,offset:number,action:'apply'|'save_without_offset'):Promise<boolean>;
 exportAnimation?(result:LipResult,format:'mp4'|'gif',quality:'high'|'standard'|'small',signal:AbortSignal):Promise<boolean>;
 saveLocal?(name:string,blob:Blob):Promise<boolean>;
}
