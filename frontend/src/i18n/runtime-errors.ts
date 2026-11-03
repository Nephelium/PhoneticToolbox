// Worker-safe structured errors. This file never imports Vue or language packs.
import {workerFallbackMessages} from './generated/worker-fallback.ts';
import type {MessageDescriptor,MessageValues} from './index.ts';

let format:((value:MessageDescriptor)=>string)|undefined;
export function installErrorFormatter(value:(message:MessageDescriptor)=>string){format=value;}
function fallback(value:MessageDescriptor){
 const text=workerFallbackMessages[value.key]??value.key;
 return text.replace(/\{([A-Za-z_][A-Za-z0-9_]*|[0-9]+)\}/g,(whole,name)=>value.values?.[name]===undefined?whole:String(value.values[name]??''));
}
export class RuntimeDisplayError extends Error {
 readonly display:MessageDescriptor;
 constructor(display:MessageDescriptor,initial?:string){
  super(initial??fallback(display));this.display=display;this.name='RuntimeDisplayError';
  Object.defineProperty(this,'message',{configurable:true,get:()=>format?format(display):initial??fallback(display)});
 }
}
export function runtimeError(key:string,values?:MessageValues){return new RuntimeDisplayError({key,values});}
export function errorDisplay(error:unknown):MessageDescriptor|undefined {
 return error instanceof RuntimeDisplayError?error.display:undefined;
}
