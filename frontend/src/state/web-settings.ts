import {computed,isRef,nextTick,onBeforeUnmount,onMounted,ref,watch,type Ref} from 'vue';
import {projects} from '../platform/browser.ts';
import {accountSettingsError,flushAccountPreferences,mergePreferences,preferenceChanges} from '../platform/account-preferences.ts';

/** Preferences belong to an account/module, never to its expiring project. */
export function webSettingsKey(scope?:string){
 const match=/^server:([^:]+):[^:]+:(M\d{2})$/.exec(scope??'');
 return match?'web-settings.v1.'+encodeURIComponent(match[1]!)+'.'+match[2]:null;
}
export function settingsFields<T extends object,K extends keyof T>(target:T|Ref<T>,keys:readonly K[]){
 return Object.fromEntries(keys.map(key=>[key,computed({get:()=> (isRef(target)?target.value:target)[key],set:value=>{(isRef(target)?target.value:target)[key]=value;}})])) as Record<string,Ref<T[K]>>;
}
const failures=ref<string[]>([]);
export const webSettingsError=computed(()=>accountSettingsError.value||(failures.value.length?'设置暂未保存，请检查存储权限后重试。':''));
const flushers=new Set<()=>void>();
const hydrating=new Map<string,number>();
export function applyingWebSettings(scope?:string){return (hydrating.get(webSettingsKey(scope)??'')??0)>0;}
export async function flushWebSettings(keepalive=false){for(const flush of flushers)flush();await flushAccountPreferences(keepalive);}
export function retryWebSettings(){return flushWebSettings();}
type Fields=Record<string,Ref>;
type Store={read<T>(key:string,fallback:T):T;write(key:string,value:unknown):boolean};
const own=(value:object,key:string)=>Object.prototype.hasOwnProperty.call(value,key);
export function readWebSettings(scope:string|undefined,store:Store=projects):Record<string,unknown>{
 const key=webSettingsKey(scope),value=key?store.read<unknown>(key,null):null;
 return value&&typeof value==='object'&&!Array.isArray(value)?value as Record<string,unknown>:{};
}
export function writeWebSettings(scope:string|undefined,values:Record<string,unknown>,store:Store=projects){
 const key=webSettingsKey(scope);if(!key)return false;
 let ok=false;
 try{const json=JSON.stringify(mergePreferences(readWebSettings(scope,store),values));ok=json.length<=65536&&store.write(key,JSON.parse(json));}catch{}
 failures.value=ok?failures.value.filter(k=>k!==key):[...new Set([...failures.value,key])];return ok;
}

// Restore only the shape declared by the module. Unknown fields cannot turn
// preferences into a file/result cache; malformed records keep current defaults.
function restore(value:unknown,fallback:any):any{
 if(fallback===null||fallback===undefined)return value===null||typeof value==='boolean'||typeof value==='string'||typeof value==='number'&&Number.isFinite(value)?value:fallback;
 if(Array.isArray(fallback)){
  if(!Array.isArray(value)||value.length>2048)return fallback;
  if(!fallback.length)return value.every(v=>typeof v==='string'||typeof v==='number'&&Number.isFinite(v))?value:fallback;
  return value.map(v=>restore(v,fallback[0]));
 }
 if(typeof fallback==='object'){
  if(!value||typeof value!=='object'||Array.isArray(value))return fallback;
  return Object.fromEntries(Object.entries(fallback).map(([k,v])=>[k,own(value,k)?restore((value as Record<string,unknown>)[k],v):v]));
 }
 // Vue's number inputs produce numbers even when the editable draft starts as
 // a string. Preserve the module's declared representation when reopening.
 if(typeof fallback==='string'&&typeof value==='number'&&Number.isFinite(value))return String(value);
 if(typeof fallback==='number'&&typeof value==='string'&&value.trim()&&Number.isFinite(Number(value)))return Number(value);
 return typeof value===typeof fallback&&(typeof value!=='number'||Number.isFinite(value))?value:fallback;
}
export function connectWebSettings(scope:string|undefined,fields:Fields,store:Store=projects,afterSave?:()=>void){
 const key=webSettingsKey(scope);if(!key)return {flush:()=>{},stop:()=>{}};
 const raw=store.read<unknown>(key,null);
 hydrating.set(key,(hydrating.get(key)??0)+1);
 if(raw&&typeof raw==='object'&&!Array.isArray(raw))for(const [name,field] of Object.entries(fields))if(own(raw,name))field.value=restore((raw as Record<string,unknown>)[name],field.value);
 void nextTick(()=>{hydrating.set(key,Math.max(0,(hydrating.get(key)??1)-1));afterSave?.();});
 const snapshot=()=>JSON.parse(JSON.stringify(Object.fromEntries(Object.entries(fields).map(([name,field])=>[name,field.value]))));
 let baseline=snapshot();
 let timer:ReturnType<typeof setTimeout>|undefined,pending=false;
 function flush(){
  clearTimeout(timer);if(!pending)return;pending=false;
  const current=snapshot(),ok=writeWebSettings(scope,preferenceChanges(baseline,current),store);
  if(ok){baseline=current;afterSave?.();}
  if(!ok)pending=true;
 }
 const unwatch=watch(()=>Object.fromEntries(Object.entries(fields).map(([name,field])=>[name,field.value])),()=>{pending=true;clearTimeout(timer);timer=setTimeout(flush,200);},{deep:true,flush:'sync'});
 flushers.add(flush);
 return {flush,stop:()=>{flush();unwatch();clearTimeout(timer);flushers.delete(flush);}};
}
export function useWebSettings(scope:string|undefined,fields:Fields,ready?:()=>boolean,afterSave?:()=>void){
 if(!webSettingsKey(scope))return;
 let connection:ReturnType<typeof connectWebSettings>|undefined,unwatch:(()=>void)|undefined;
 const flush=()=>connection?.flush(),hidden=()=>{if(document.visibilityState==='hidden')flush();};
 onMounted(()=>{
  if(ready)unwatch=watch(ready,value=>{if(value&&!connection)connection=connectWebSettings(scope,fields,projects,afterSave);},{immediate:true,flush:'sync'});
  else connection=connectWebSettings(scope,fields,projects,afterSave);
  window.addEventListener('pagehide',flush);document.addEventListener('visibilitychange',hidden);
 });
 onBeforeUnmount(()=>{unwatch?.();connection?.stop();window.removeEventListener('pagehide',flush);document.removeEventListener('visibilitychange',hidden);});
}
