import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
/** Bounded database opening: a locked profile must not leave upload/startup waiting forever. */
export function openImageDatabase(name:string,store:string,timeout=12000):Promise<IDBDatabase>{
 return new Promise((resolve,reject)=>{
  const request=indexedDB.open(name,1);let settled=false;
  const fail=(error:unknown)=>{if(settled)return;settled=true;clearTimeout(timer);reject(error);};
  const timer=setTimeout(()=>fail(uiError(uiMessage('platform.local_image_storage_is_not_responding_close_other_phonetictoolbox'))),timeout);
  request.onupgradeneeded=()=>request.result.createObjectStore(store);
  request.onblocked=()=>fail(uiError(uiMessage('platform.image_settings_are_in_use_by_another_app_window')));
  request.onerror=()=>fail(uiError(uiMessage('platform.local_image_storage_could_not_be_opened_close_other')));
  request.onsuccess=()=>{if(settled){request.result.close();return;}settled=true;clearTimeout(timer);resolve(request.result);};
 });
}
