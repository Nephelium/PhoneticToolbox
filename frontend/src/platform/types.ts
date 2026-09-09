export interface PeakIndex { stride:number; min:Float32Array; max:Float32Array }
export interface AudioAsset { name:string; sampleRate:number; channels:Float32Array[]; peaks?:PeakIndex[]; duration:number; frames:number }
export interface FileProvider { load(file:File):Promise<AudioAsset> }
export interface ProjectStore { read<T>(key:string, fallback:T):T; write(key:string,value:unknown):boolean }
export interface HostCapabilities { kind:'browser'|'desktop'; files:FileProvider; projects:ProjectStore; jobs:false; capture:false }
