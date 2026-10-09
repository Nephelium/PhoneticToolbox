import {copy,type Asset,type Project,type Session} from './model.ts';
const NAME='ptb-m15-local-v1';
function request<T>(r:IDBRequest<T>){return new Promise<T>((resolve,reject)=>{r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});}
function done(tx:IDBTransaction){return new Promise<void>((resolve,reject)=>{tx.oncomplete=()=>resolve();tx.onabort=()=>reject(tx.error??Error('本地事务中止'));tx.onerror=()=>reject(tx.error??Error('本地写入失败'));});}
export class LocalStore {
 private db:IDBDatabase|null=null;
 async open(){if(this.db)return this.db;this.db=await new Promise<IDBDatabase>((resolve,reject)=>{const r=indexedDB.open(NAME,1);r.onupgradeneeded=()=>{for(const n of ['projects','assets','sessions'])r.result.createObjectStore(n,{keyPath:'id'});};r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);r.onblocked=()=>reject(Error('本地存储升级被其他标签阻止'));});return this.db;}
 async saveProject(p:Project){const db=await this.open(),tx=db.transaction('projects','readwrite'),completion=done(tx);tx.objectStore('projects').put(copy(p));await completion;}
 async putAssets(assets:{meta:Asset;blob:Blob}[]){const db=await this.open(),tx=db.transaction('assets','readwrite'),completion=done(tx);for(const a of assets)tx.objectStore('assets').put({id:a.meta.hash,blob:a.blob});await completion;}
 async blob(hash:string):Promise<Blob>{const db=await this.open(),r=await request(db.transaction('assets').objectStore('assets').get(hash));if(!r?.blob)throw Error(`本机缺少刺激 ${hash.slice(0,12)}，请重新选择文件并关联`);return r.blob;}
 async list<T>(store:'projects'|'sessions'):Promise<T[]>{const db=await this.open();return request(db.transaction(store).objectStore(store).getAll());}
 async session(id:string):Promise<Session>{const db=await this.open();const s=await request(db.transaction('sessions').objectStore('sessions').get(id));if(!s)throw Error('会话不存在');return s;}
 async commit(s:Session){const db=await this.open(),tx=db.transaction('sessions','readwrite'),completion=done(tx),table=tx.objectStore('sessions');let conflict=false,writeError:unknown;
 const r=table.get(s.id);r.onsuccess=()=>{const old=r.result as Session|undefined;if((old?.revision??0)!==s.revision){conflict=true;tx.abort();return;}try{table.put({...copy(s),revision:s.revision+1});}catch(e){writeError=e;tx.abort();}};
 try{await completion;}catch(e){if(conflict)throw Error('会话已被其他标签修改，停止写入。请导出当前结果，再重新载入。');throw writeError??e;}s.revision++;
 }
 async capacity(){return navigator.storage?.estimate?await navigator.storage.estimate():{};}
 async persist(){return navigator.storage?.persist?await navigator.storage.persist():false;}
 close(){this.db?.close();this.db=null;}
}
export async function lockSession(id:string):Promise<()=>Promise<void>>{
 if(!navigator.locks)throw Error('此浏览器缺少安全会话锁，无法正式运行。请使用支持 Web Locks 的浏览器。');
 return new Promise((resolve,reject)=>{const completion=navigator.locks.request('ptb-m15:'+id,{ifAvailable:true},async lock=>{if(!lock){reject(Error('该会话正在另一个标签运行，请在那里暂停并退出运行视图后再恢复。'));return;}await new Promise<void>(release=>resolve(async()=>{release();await completion;}));});void completion.catch(reject);});
}
