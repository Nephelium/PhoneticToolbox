import {uiMessage,uiError} from '../modules/module-localization.ts';
import type {MessageDescriptor} from '../i18n/index.ts';
import {tr} from '../i18n/index.ts';
import {ref} from 'vue';

export interface StorageStatus {
  schema:'ptb-retention/1';enabled:boolean;days:number;exported_cache_days:number;
  clean_unused_inputs?:boolean;
  result_bytes:number;input_bytes:number;result_count:number;
  diagnostic_days?:number;diagnostic_count?:number;diagnostic_bytes?:number;
  last_result:null|{count:number;bytes:number;protected_count:number;failed_count?:number;complete?:boolean;at:number};
}
export interface CleanupResult {skipped:boolean;count:number;bytes:number;protected_count?:number;failed_count?:number;complete?:boolean;at?:number}
export interface CacheLocation {path:string;custom:boolean;legacy_path:string|null;exit_cleanup_supported?:boolean}
export interface CacheCandidate {id:string;kind:string;path:string;label:string;label_key?:string;bytes:number;size_complete:boolean;can_clean:boolean;clean_on_exit?:boolean;reason:string;reason_code?:string}
const candidateKinds=new Set(['runtime','updates','browser','graphics']);
const reasonKeys:Record<string,string>={graphics_exit:'graphicsExit',in_use:'inUse',ownership_unknown:'ownershipUnknown',leases_unknown:'leasesUnknown',inspection_failed:'inspectionFailed',qa_exit_cleanup_disabled:'qaExitCleanupDisabled'};
/** Resolve on display, preserving stable candidate IDs and original paths. */
export function cacheCandidateLabel(item:CacheCandidate):string{return candidateKinds.has(item.kind)?tr('native.cache.candidate.'+item.kind):item.label;}
export function cacheCandidateReason(item:CacheCandidate):string{return item.reason_code&&reasonKeys[item.reason_code]?tr('native.cache.reason.'+reasonKeys[item.reason_code]):item.reason;}
export function cacheCleanupMessage(answer:{complete:boolean;message:string;message_code?:string}):MessageDescriptor|string{return answer.message_code==='complete'||answer.message_code==='partial'?uiMessage('native.cache.cleaned.'+answer.message_code):answer.message;}
export interface StoragePort {
  status():Promise<StorageStatus>;
  configure(policy:{enabled:boolean;days:number;exported_cache_days:number;clean_unused_inputs?:boolean}):Promise<StorageStatus>;
  clean():Promise<CleanupResult>;
  clearAll():Promise<{started:boolean}>;
  cacheLocation?():Promise<CacheLocation>;
  openCacheDirectory?(legacy?:boolean):Promise<{opened:boolean}>;
  cacheCandidates?():Promise<{candidates:CacheCandidate[]}>;
  openCacheCandidate?(id:string):Promise<{opened:boolean}>;
  cleanCacheCandidate?(id:string):Promise<{complete:boolean;message:string;message_code?:string}>;
  cleanGraphicsCandidate?(id:string):Promise<{started:boolean}>;
}
export const storageAvailable=ref(false);
let port:StoragePort|undefined;
export function installStoragePort(value:StoragePort){port=value;storageAvailable.value=true;}
export function localStoragePort(){if(!port)throw uiError(uiMessage('platform.the_local_result_cache_is_not_ready_yet'));return port;}
