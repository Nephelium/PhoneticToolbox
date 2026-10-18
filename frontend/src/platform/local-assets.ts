/** Account/project-scoped research bytes. Settings continue through account preferences. */
export class LocalStorageError extends Error {}
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
 async compare(key:string,expected:unknown,value:unknown){const current=this.values.get(key) as Value|undefined;if(current?.value===value)return;if((current?.value??null)!==expected)throw new LocalStorageError('录音工程已在另一窗口更新，请重新载入后继续。');this.values.set(key,structuredClone({key,value}));}
}
class IndexedAssetDatabase implements AssetDatabase {
 private database:Promise<IDBDatabase>;
 constructor(){this.database=new Promise((resolve,reject)=>{
  if(typeof indexedDB==='undefined'){reject(new LocalStorageError('浏览器无法保存本机文件，请允许站点存储后重试。'));return;}
  const request=indexedDB.open('ptb-research-files-v1',1);
  request.onupgradeneeded=()=>{const db=request.result;db.createObjectStore('files',{keyPath:'id'}).createIndex('scope',['owner','project']);db.createObjectStore('values',{keyPath:'key'});};
  request.onsuccess=()=>{request.result.onversionchange=()=>request.result.close();resolve(request.result);};
  request.onerror=()=>reject(new LocalStorageError('本机文件库打开失败，请检查浏览器的站点存储权限。'));
  request.onblocked=()=>reject(new LocalStorageError('本机文件库正被其他窗口占用，请关闭旧页面后重试。'));
 });void this.database.catch(()=>{});}
 private async transaction<T>(bucket:Bucket,mode:IDBTransactionMode,work:(store:IDBObjectStore,done:(value:T)=>void)=>void){
  const db=await this.database;return new Promise<T>((resolve,reject)=>{const tx=db.transaction(bucket,mode);let result:T;
   tx.oncomplete=()=>resolve(result);tx.onabort=tx.onerror=()=>reject(new LocalStorageError(tx.error?.name==='QuotaExceededError'?'浏览器本机空间不足，请先导出或清理本机文件。':'本机文件保存未完成，请保持页面打开后重试。'));
   try{work(tx.objectStore(bucket),value=>{result=value;});}catch(error){tx.abort();reject(error);}
  });
 }
 get<T extends Entry>(bucket:Bucket,key:string){return this.transaction<T|undefined>(bucket,'readonly',(store,done)=>{const r=store.get(key);r.onsuccess=()=>done(r.result);});}
 all<T extends Entry>(bucket:Bucket,scope?:[string,string]){return this.transaction<T[]>(bucket,'readonly',(store,done)=>{const r=scope?store.index('scope').getAll(IDBKeyRange.only(scope)):store.getAll();r.onsuccess=()=>done(r.result);});}
 put(bucket:Bucket,value:Entry){return this.transaction<void>(bucket,'readwrite',store=>{store.put(value);});}
 remove(bucket:Bucket,key:string){return this.transaction<void>(bucket,'readwrite',store=>{store.delete(key);});}
 async compare(key:string,expected:unknown,value:unknown){
  const db=await this.database;return new Promise<void>((resolve,reject)=>{const tx=db.transaction('values','readwrite'),store=tx.objectStore('values');let conflict=false;
   tx.oncomplete=()=>resolve();tx.onabort=tx.onerror=()=>reject(new LocalStorageError(conflict?'录音工程已在另一窗口更新，请重新载入后继续。':'本机工程保存未完成，请重试。'));
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
  if(blob.size>100_000_000)throw new LocalStorageError('文件超过当前 100 MB 本机处理上限，请缩小材料后重试。');
  const digest=await localDigest(blob),previous=key?(await this.list()).find(f=>f.key===key):undefined;
  const file:LocalAsset={id:id??previous?.id??'ptb-local:'+crypto.randomUUID(),owner:this.owner,project:this.project,name,blob,sha256:digest,key,created_at:Date.now()/1000};
  await this.database.put('files',{...file,id:this.fileKey(file.id)});return file;
 }
 async remember(file:LocalAsset){if(file.owner!==this.owner||file.project!==this.project)throw new LocalStorageError('文件不属于当前账号或项目。');await this.database.put('files',{...file,id:this.fileKey(file.id)});}
 async remove(id:string){await this.database.remove('files',this.fileKey(id));}
 async value<T>(key:string){return (await this.database.get<Value>('values',this.valueKey(key)))?.value as T|undefined;}
 async setValue(key:string,value:unknown){await this.database.put('values',{key:this.valueKey(key),value});}
 async compare(key:string,expected:unknown,value:unknown){await this.database.compare(this.valueKey(key),expected,value);}
 async download(id:string,name?:string){const file=await this.get(id);if(!file)throw new LocalStorageError('本机文件不存在，请重新导入。');const url=URL.createObjectURL(file.blob),a=document.createElement('a');a.href=url;a.download=name??file.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
}
