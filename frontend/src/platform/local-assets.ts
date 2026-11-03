import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr,errorMessageDescriptor} from '../i18n/index.ts';
import type {MessageDescriptor} from '../i18n/index.ts';
import {RuntimeDisplayError} from '../i18n/runtime-errors.ts';
/** Account/project-scoped research bytes. Settings continue through account preferences. */
export class LocalStorageError extends RuntimeDisplayError {
 readonly code?:string;
 constructor(value:string|MessageDescriptor,code?:string){super(typeof value==='string'?errorMessageDescriptor(value)??{key:'modules.display.composed',values:{text:value}}:value,typeof value==='string'?value:undefined);this.code=code;}
}
export const localAssetId=(id:string)=>id.startsWith('ptb-local:');
export type LocalAsset={id:string;owner:string;project:string;name:string;blob:Blob;sha256:string;created_at:number;key?:string;mirrorKey?:string;server?:{id:string;expires_at:number|null;project:string}};
type Value={key:string;value:unknown};
type Entry=LocalAsset|Value;
type Bucket='files'|'values';
export interface AssetDatabase {
 get<T extends Entry>(bucket:Bucket,key:string):Promise<T|undefined>;
 all<T extends Entry>(bucket:Bucket,scope?:[string,string]):Promise<T[]>;
 put(bucket:Bucket,value:Entry):Promise<void>;
 remove(bucket:Bucket,key:string):Promise<void>;
 compare(key:string,expected:unknown,value:unknown):Promise<void>;
}
export class MemoryAssetDatabase implements AssetDatabase {
 private files=new Map<string,Entry>();private values=new Map<string,Entry>();
 async get<T extends Entry>(bucket:Bucket,key:string){return structuredClone(this[bucket].get(key)) as T|undefined;}
 async all<T extends Entry>(bucket:Bucket,scope?:[string,string]){return structuredClone([...this[bucket].values()].filter(f=>!scope||('owner'in f&&f.owner===scope[0]&&f.project===scope[1]))) as T[];}
 async put(bucket:Bucket,value:Entry){this[bucket].set('id'in value?value.id:value.key,structuredClone(value));}
 async remove(bucket:Bucket,key:string){this[bucket].delete(key);}
 async compare(key:string,expected:unknown,value:unknown){const current=this.values.get(key) as Value|undefined;if(current?.value===value)return;if((current?.value??null)!==expected)throw new LocalStorageError(uiMessage('platform.the_recording_project_was_updated_in_another_window_reload'),'local_record_revision_conflict');this.values.set(key,structuredClone({key,value}));}
}
class IndexedAssetDatabase implements AssetDatabase {
 private database:Promise<IDBDatabase>;
 constructor(){this.database=new Promise((resolve,reject)=>{
  if(typeof indexedDB==='undefined'){reject(new LocalStorageError(uiMessage('platform.the_browser_cannot_save_local_files_allow_site_storage')));return;}
  const request=indexedDB.open('ptb-research-files-v1',1);
  request.onupgradeneeded=()=>{const db=request.result;db.createObjectStore('files',{keyPath:'id'}).createIndex('scope',['owner','project']);db.createObjectStore('values',{keyPath:'key'});};
  request.onsuccess=()=>{request.result.onversionchange=()=>request.result.close();resolve(request.result);};
  request.onerror=()=>reject(new LocalStorageError(uiMessage('platform.the_local_file_database_could_not_be_opened_check')));
  request.onblocked=()=>reject(new LocalStorageError(uiMessage('platform.the_local_file_database_is_in_use_by_another')));
 });void this.database.catch(()=>{});}
 private async transaction<T>(bucket:Bucket,mode:IDBTransactionMode,work:(store:IDBObjectStore,done:(value:T)=>void)=>void){
  const db=await this.database;return new Promise<T>((resolve,reject)=>{const tx=db.transaction(bucket,mode);let result:T;
   tx.oncomplete=()=>resolve(result);tx.onabort=tx.onerror=()=>reject(new LocalStorageError(tx.error?.name==='QuotaExceededError'?uiMessage('platform.browser_storage_is_full_export_or_clean_up_local'):uiMessage('platform.local_file_saving_did_not_finish_keep_this_page')));
   try{work(tx.objectStore(bucket),value=>{result=value;});}catch(error){tx.abort();reject(error);}
  });
 }
 get<T extends Entry>(bucket:Bucket,key:string){return this.transaction<T|undefined>(bucket,'readonly',(store,done)=>{const r=store.get(key);r.onsuccess=()=>done(r.result);});}
 all<T extends Entry>(bucket:Bucket,scope?:[string,string]){return this.transaction<T[]>(bucket,'readonly',(store,done)=>{const r=scope?store.index('scope').getAll(IDBKeyRange.only(scope)):store.getAll();r.onsuccess=()=>done(r.result);});}
 put(bucket:Bucket,value:Entry){return this.transaction<void>(bucket,'readwrite',store=>{store.put(value);});}
 remove(bucket:Bucket,key:string){return this.transaction<void>(bucket,'readwrite',store=>{store.delete(key);});}
 async compare(key:string,expected:unknown,value:unknown){
  const db=await this.database;return new Promise<void>((resolve,reject)=>{const tx=db.transaction('values','readwrite'),store=tx.objectStore('values');let conflict=false;
   tx.oncomplete=()=>resolve();tx.onabort=tx.onerror=()=>reject(new LocalStorageError(conflict?uiMessage('platform.the_recording_project_was_updated_in_another_window_reload'):uiMessage('platform.local_project_saving_did_not_finish_try_again'),conflict?'local_record_revision_conflict':undefined));
   const r=store.get(key);r.onsuccess=()=>{const current=r.result?.value??null;if(current===value)return;if(current!==expected){conflict=true;tx.abort();return;}store.put({key,value});};
  });
 }
}
export async function localDigest(blob:Blob){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',await blob.arrayBuffer()))].map(n=>n.toString(16).padStart(2,'0')).join('');}
export class LocalAssets {
 readonly database:AssetDatabase;private prefix:string;readonly owner:string;readonly project:string;
 constructor(owner:string,project:string,database?:AssetDatabase){this.owner=owner;this.project=project;this.prefix=owner+'\u0000'+project+'\u0000';this.database=database??(typeof window==='undefined'&&typeof indexedDB==='undefined'?new MemoryAssetDatabase():new IndexedAssetDatabase());}
 private valueKey(key:string){return this.prefix+key;}
 private fileKey(id:string){return this.prefix+id;}
 async get(id:string){const saved=await this.database.get<LocalAsset>('files',this.fileKey(id));if(!saved)return;return {...saved,id:saved.id.slice(this.prefix.length)};}
 async list(){return (await this.database.all<LocalAsset>('files',[this.owner,this.project])).map(f=>({...f,id:f.id.slice(this.prefix.length)})).sort((a,b)=>b.created_at-a.created_at);}
 async save(name:string,blob:Blob,key?:string,id?:string){
  if(blob.size>100_000_000)throw new LocalStorageError(uiMessage('platform.the_file_exceeds_the_current_100_mb_local_processing'));
  const digest=await localDigest(blob),previous=key?(await this.list()).find(f=>f.key===key):undefined;
  const file:LocalAsset={id:id??previous?.id??'ptb-local:'+crypto.randomUUID(),owner:this.owner,project:this.project,name,blob,sha256:digest,key,created_at:Date.now()/1000};
  await this.database.put('files',{...file,id:this.fileKey(file.id)});return file;
 }
 async remember(file:LocalAsset){if(file.owner!==this.owner||file.project!==this.project)throw new LocalStorageError(uiMessage('platform.the_file_does_not_belong_to_this_account_or'));await this.database.put('files',{...file,id:this.fileKey(file.id)});}
 async remove(id:string){await this.database.remove('files',this.fileKey(id));}
 async value<T>(key:string){return (await this.database.get<Value>('values',this.valueKey(key)))?.value as T|undefined;}
 async setValue(key:string,value:unknown){await this.database.put('values',{key:this.valueKey(key),value});}
 async compare(key:string,expected:unknown,value:unknown){await this.database.compare(this.valueKey(key),expected,value);}
 async download(id:string,name?:string){const file=await this.get(id);if(!file)throw new LocalStorageError(uiMessage('platform.the_local_file_does_not_exist_import_it_again'));const url=URL.createObjectURL(file.blob),a=document.createElement('a');a.href=url;a.download=name??file.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
}
