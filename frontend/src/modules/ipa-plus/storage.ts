import type {Draft} from './types.ts';
import {catalog} from './catalog.ts';
export const DB_NAME='phonetic-toolbox-m17';
export function createDraft(writer:string):Draft{return {version:1,catalogVersion:catalog.version,text:'',start:0,end:0,system:'ipa',introductions:true,textSize:28,editorHeight:120,revision:0,writer};}
export function restoreDraft(value:unknown,writer:string):{draft:Draft;blocked:boolean;message:string}{
  const fallback=createDraft(writer);if(value===undefined)return {draft:fallback,blocked:false,message:''};
  if(!value||typeof value!=='object')return {draft:fallback,blocked:true,message:'草稿格式无法识别，原记录已保留。请另存独立草稿。'};
  const d=value as Partial<Draft>;const text=typeof d.text==='string'?d.text:'';
  const valid=d.version===1&&typeof d.text==='string'&&Number.isInteger(d.revision)&&Number(d.revision)>=0&&['ipa','extipa','voqs'].includes(d.system??'');
  return {draft:{...fallback,text,start:Math.max(0,Math.min(text.length,Number(d.start)||0)),end:Math.max(0,Math.min(text.length,Number(d.end)||0)),system:valid?d.system!:'ipa',introductions:d.introductions!==false,textSize:Math.max(18,Math.min(54,Number(d.textSize)||28)),editorHeight:Math.max(96,Math.min(360,Number(d.editorHeight)||144)),revision:valid?d.revision!:0,writer},blocked:!valid,message:valid?'':'草稿版本未知，已取回可读文字并保留原记录。请导出文字或另存独立草稿。'};
}
let connection:Promise<IDBDatabase>|undefined;
function database(){return connection??=new Promise<IDBDatabase>((resolve,reject)=>{
  if(!globalThis.indexedDB){reject(Error('当前环境不支持本机草稿存储。'));return;}
  const request=indexedDB.open(DB_NAME,1);
  request.onupgradeneeded=()=>request.result.createObjectStore('drafts');
  request.onsuccess=()=>{request.result.onversionchange=()=>{request.result.close();connection=undefined;};resolve(request.result);};
  request.onerror=()=>{connection=undefined;reject(Error('本机草稿存储无法打开。'));};
  request.onblocked=()=>{connection=undefined;reject(Error('另一窗口正在升级草稿存储，请关闭该窗口后重试。'));};
});}
export async function loadDraft(key:string):Promise<unknown>{
  const db=await database();return new Promise((resolve,reject)=>{const r=db.transaction('drafts','readonly').objectStore('drafts').get(key);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(Error('本机草稿读取失败，未覆盖已有记录。'));});
}
export async function persistDraft(key:string,draft:Draft,expectedRevision:number):Promise<number>{
  const db=await database();return new Promise((resolve,reject)=>{
    const tx=db.transaction('drafts','readwrite'),store=tx.objectStore('drafts');let failure='本机草稿写入失败，文字仍保留在编辑器。';
    const r=store.get(key);r.onsuccess=()=>{
      const previous=r.result as Draft|undefined;
      if(previous&&(previous.version!==1||previous.revision!==expectedRevision)){
        failure='另一窗口已经修改此草稿，已停止覆盖。请导出当前文字或另存独立草稿。';tx.abort();return;
      }
      if(!previous&&expectedRevision!==0){failure='原草稿已被其他窗口移除，未自动覆盖。请另存独立草稿。';tx.abort();return;}
      try{store.put({...draft,revision:expectedRevision+1},key);}
      catch{tx.abort();}
    };
    tx.oncomplete=()=>resolve(expectedRevision+1);tx.onerror=()=>reject(Error(failure));tx.onabort=()=>reject(Error(failure));
  });
}
