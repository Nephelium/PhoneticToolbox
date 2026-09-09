import { parseWav } from './wav.ts';
import type { HostCapabilities,ProjectStore } from './types.ts';
export const projects:ProjectStore = {
  read<T>(key:string,fallback:T):T {try{return JSON.parse(localStorage.getItem('ptb.v3.'+key)||'null')??fallback;}catch{return fallback;}},
  write(key,value){try{localStorage.setItem('ptb.v3.'+key,JSON.stringify(value));return true;}catch{return false;}},
};
export const browser:HostCapabilities={kind:'browser',projects,jobs:false,capture:false,files:{async load(file){
  if(file.size>64*1024*1024) throw Error('当前预览支持 64 MB 以内的 WAV；大文件流式读取待接入。');
  return parseWav(await file.arrayBuffer(),file.name);
}}};
