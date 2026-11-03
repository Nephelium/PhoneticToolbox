import {uiMessage,uiErrorMessage,responseError} from '../modules/module-localization.ts';
import type {Device, Project, RecordingStatus} from '../modules/recording/types.ts';
export type RecordingRequest = <T=unknown>(body:Record<string,unknown>)=>Promise<T>;
export type DirectoryChoice={grant:string;label:string};
export type RecordingPort={kind?:'browser'|'desktop';request:RecordingRequest;choose:(purpose:'new'|'open'|'export')=>Promise<DirectoryChoice|null>;dispose?:()=>void};
let installed:RecordingPort|undefined;
export function installRecordingPort(port:RecordingPort){installed=port;}
export function getRecordingPort(){return installed;}
export function recordingPort(request:RecordingRequest,choose:RecordingPort['choose']):RecordingPort{return {request,choose};}
export type RecordingCapabilities={available:boolean;devices:Device[];device_error:string;device_error_display?:unknown;project:Project|null;local_only:boolean};
export type {RecordingStatus};

const diagnosticKeys=new Set(['deviceUnavailable','playbackStartCleanup','playbackStreamStatus','playbackAbortFailed','playbackCloseFailed'].map(key=>'native.errors.recording.'+key));
/** Translate product context and registered causes; driver diagnostics stay literal. */
export function recordingErrorMessage(raw:unknown,display?:unknown){
 const value=uiErrorMessage(typeof raw==='string'?responseError(raw,display):raw);
 if(typeof value==='object'&&diagnosticKeys.has(value.key)&&typeof value.values?.reason==='string')return uiMessage(value.key,{reason:uiErrorMessage(responseError(value.values.reason))});
 return value;
}
