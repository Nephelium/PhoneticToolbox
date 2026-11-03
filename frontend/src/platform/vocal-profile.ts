import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
/** Durable, account/project-scoped records in the existing IndexedDB values store. */
import {LocalAssets,LocalStorageError} from './local-assets.ts';
export interface VocalProfileStore {
 read():Promise<string|undefined>;
 write(expected:string|null,value:string):Promise<void>;
}
export function vocalProfileStore(assets:LocalAssets):VocalProfileStore {
 return {
  read:()=>assets.value<string>('vocal-profile-v1'),
  async write(expected,value){
   if(new TextEncoder().encode(value).length>32_000_000)throw uiError(uiMessage('platform.saved_vocal_tract_data_exceeds_32_mb_export_it'));
   try{await assets.compare('vocal-profile-v1',expected,value);}
   catch(error){throw uiError(error instanceof LocalStorageError&&error.code==='local_record_revision_conflict'?uiMessage('platform.the_configuration_was_updated_in_another_window_export_your'):uiMessage('platform.the_configuration_could_not_be_saved_check_local_free'));}
  }
 };
}
export function parseVocalProfile(raw:string):any {
 try{
  const value=JSON.parse(raw);
  if(!value||typeof value!=='object'||Array.isArray(value)||value.speaker&&!['JD2','W02'].includes(value.speaker))throw Error();
  if(value.models&&(typeof value.models!=='object'||Array.isArray(value.models)||!Object.keys(value.models).every(k=>['JD2','W02'].includes(k))||Object.values(value.models).some(m=>!m||typeof m!=='object'||Array.isArray(m))))throw Error();
  return value;
 }catch{throw uiError(uiMessage('platform.saved_vocal_tract_data_could_not_be_read_original'));}
}
