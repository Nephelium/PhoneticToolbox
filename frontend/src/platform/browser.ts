import { decodeWav } from './decode.ts';
import type { HostCapabilities,ProjectStore } from './types.ts';
import {accountPreferenceTarget} from './account-preferences.ts';
export const projects:ProjectStore = {
  read<T>(key:string,fallback:T):T {const target=accountPreferenceTarget(key);if(target){const values=target.session.read(target.module);return (target.field?values[target.field]:Object.keys(values).length?values:undefined)??fallback;}try{return JSON.parse(localStorage.getItem('ptb.v3.'+key)||'null')??fallback;}catch{return fallback;}},
  write(key,value){const target=accountPreferenceTarget(key);if(target)return target.session.write(target.module,target.field?{[target.field]:value}:value as Record<string,unknown>);try{localStorage.setItem('ptb.v3.'+key,JSON.stringify(value));return true;}catch{return false;}},
};
export const browser:HostCapabilities={kind:'browser',projects,jobs:false,capture:false,files:{async load(file){
  if(file.size>64_000_000) throw Error('请选择 64 MB 以内的 WAV 文件进行预览。');
  return decodeWav(await file.arrayBuffer(),file.name);
}}};
