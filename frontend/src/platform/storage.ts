import {ref} from 'vue';

export interface StorageStatus {
  schema:'ptb-retention/1';enabled:boolean;days:number;exported_cache_days:number;
  result_bytes:number;input_bytes:number;result_count:number;
  diagnostic_days?:number;diagnostic_count?:number;diagnostic_bytes?:number;
  last_result:null|{count:number;bytes:number;protected_count:number;failed_count?:number;complete?:boolean;at:number};
}
export interface CleanupResult {skipped:boolean;count:number;bytes:number;protected_count?:number;failed_count?:number;complete?:boolean;at?:number}
export interface StoragePort {
  status():Promise<StorageStatus>;
  configure(policy:{enabled:boolean;days:number;exported_cache_days:number}):Promise<StorageStatus>;
  clean():Promise<CleanupResult>;
  clearAll():Promise<{started:boolean}>;
}
export const storageAvailable=ref(false);
let port:StoragePort|undefined;
export function installStoragePort(value:StoragePort){port=value;storageAvailable.value=true;}
export function localStoragePort(){if(!port)throw Error('本机结果缓存尚未就绪。');return port;}
