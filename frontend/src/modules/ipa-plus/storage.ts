import {uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';
import {tr,type MessageDescriptor} from '../../i18n/index.ts';
import type {Draft} from './types.ts';
import {catalog} from './catalog.ts';
export const DB_NAME='phonetic-toolbox-m17';
export function createDraft(writer:string):Draft{return {version:1,catalogVersion:catalog.version,text:'',start:0,end:0,system:'ipa',introductions:true,textSize:26,textSizeVersion:1,editorHeight:120,revision:0,writer};}
export function restoreDraft(value:unknown,writer:string):{draft:Draft;blocked:boolean;message:string;messageDisplay?:MessageDescriptor}{
  const fallback=createDraft(writer);if(value===undefined)return {draft:fallback,blocked:false,message:''};
  if(!value||typeof value!=='object')return {draft:fallback,blocked:true,message:tr('modules.ipa-plus.text.ed3270fa2210'),messageDisplay:displayMessage('modules.ipa-plus.text.ed3270fa2210')};
  const d=value as Partial<Draft>;const text=typeof d.text==='string'?d.text:'';
  // The former default was 28px; preserve non-default choices and migrate once.
  const textSize=d.textSizeVersion===1||Number(d.textSize)!==28?Number(d.textSize)||26:26;
  const valid=d.version===1&&typeof d.text==='string'&&Number.isInteger(d.revision)&&Number(d.revision)>=0&&['ipa','extipa','voqs'].includes(d.system??'');
  return {draft:{...fallback,text,start:Math.max(0,Math.min(text.length,Number(d.start)||0)),end:Math.max(0,Math.min(text.length,Number(d.end)||0)),system:valid?d.system!:'ipa',introductions:d.introductions!==false,textSize:Math.max(18,Math.min(54,Math.round(textSize))),editorHeight:Math.max(96,Math.min(360,Number(d.editorHeight)||144)),revision:valid?d.revision!:0,writer},blocked:!valid,message:valid?'':tr('modules.ipa-plus.text.7e63187713a9'),...(!valid?{messageDisplay:displayMessage('modules.ipa-plus.text.7e63187713a9')}:{})};
}
let connection:Promise<IDBDatabase>|undefined;
function database(){return connection??=new Promise<IDBDatabase>((resolve,reject)=>{
  if(!globalThis.indexedDB){reject(displayError(displayMessage('modules.ipa-plus.text.b3b2ac2606f1')));return;}
  const request=indexedDB.open(DB_NAME,1);
  request.onupgradeneeded=()=>request.result.createObjectStore('drafts');
  request.onsuccess=()=>{request.result.onversionchange=()=>{request.result.close();connection=undefined;};resolve(request.result);};
  request.onerror=()=>{connection=undefined;reject(displayError(displayMessage('modules.ipa-plus.text.3f523225a286')));};
  request.onblocked=()=>{connection=undefined;reject(displayError(displayMessage('modules.ipa-plus.text.43ec4523a586')));};
});}
export async function loadDraft(key:string):Promise<unknown>{
  const db=await database();return new Promise((resolve,reject)=>{const r=db.transaction('drafts','readonly').objectStore('drafts').get(key);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(displayError(displayMessage('modules.ipa-plus.text.97b07541cf02')));});
}
export async function persistDraft(key:string,draft:Draft,expectedRevision:number):Promise<number>{
  const db=await database();return new Promise((resolve,reject)=>{
    const tx=db.transaction('drafts','readwrite'),store=tx.objectStore('drafts');let failure=displayMessage('modules.ipa-plus.text.82ca8642a75c');
    const r=store.get(key);r.onsuccess=()=>{
      const previous=r.result as Draft|undefined;
      if(previous&&(previous.version!==1||previous.revision!==expectedRevision)){
        failure=displayMessage('modules.ipa-plus.text.7d3ef6f7e3e5');tx.abort();return;
      }
      if(!previous&&expectedRevision!==0){failure=displayMessage('modules.ipa-plus.text.3ab191fcc964');tx.abort();return;}
      try{store.put({...draft,revision:expectedRevision+1},key);}
      catch{tx.abort();}
    };
    tx.oncomplete=()=>resolve(expectedRevision+1);tx.onerror=()=>reject(displayError(failure));tx.onabort=()=>reject(displayError(failure));
  });
}
