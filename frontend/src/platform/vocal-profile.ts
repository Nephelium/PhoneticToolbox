/** Durable, account/project-scoped records in the existing IndexedDB values store. */
import {LocalAssets} from './local-assets.ts';
export interface VocalProfileStore {
 read():Promise<string|undefined>;
 write(expected:string|null,value:string):Promise<void>;
}
export function vocalProfileStore(assets:LocalAssets):VocalProfileStore {
 return {
  read:()=>assets.value<string>('vocal-profile-v1'),
  async write(expected,value){
   if(new TextEncoder().encode(value).length>32_000_000)throw Error('声道保存记录超过 32 MB，请先导出并减少关键帧。');
   try{await assets.compare('vocal-profile-v1',expected,value);}
   catch(error){throw Error(error instanceof Error&&error.message.includes('另一窗口')?'构型已在另一窗口更新，请导出当前草稿后重新打开工作区。':'构型未保存成功，请检查本机空间后重试；当前草稿仍可导出。');}
  }
 };
}
export function parseVocalProfile(raw:string):any {
 try{
  const value=JSON.parse(raw);
  if(!value||typeof value!=='object'||Array.isArray(value)||value.speaker&&!['JD2','W02'].includes(value.speaker))throw Error();
  if(value.models&&(typeof value.models!=='object'||Array.isArray(value.models)||!Object.keys(value.models).every(k=>['JD2','W02'].includes(k))||Object.values(value.models).some(m=>!m||typeof m!=='object'||Array.isArray(m))))throw Error();
  return value;
 }catch{throw Error('保存的声道记录无法读取，原数据已保留。请先导出恢复副本。');}
}
