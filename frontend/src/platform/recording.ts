import type {Device, Project, RecordingStatus} from '../modules/recording/types.ts';
export type RecordingRequest = <T=unknown>(body:Record<string,unknown>)=>Promise<T>;
export type DirectoryChoice={grant:string;label:string};
export type RecordingPort={request:RecordingRequest;choose:(purpose:'new'|'open'|'export')=>Promise<DirectoryChoice|null>};
let installed:RecordingPort|undefined;
export function installRecordingPort(port:RecordingPort){installed=port;}
export function getRecordingPort(){return installed;}
export function recordingPort(request:RecordingRequest,choose:RecordingPort['choose']):RecordingPort{return {request,choose};}
export type RecordingCapabilities={available:boolean;devices:Device[];device_error:string;project:Project|null;local_only:true};
export type {RecordingStatus};
