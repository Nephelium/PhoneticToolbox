import {safeRelativePath,parseManualProject,parseManualChapter} from '../manual/content.ts';
import type {ManualProject} from '../manual/types.ts';
import type {AppLocale} from '../i18n/generated/keys.ts';
import languages from '../i18n/languages.json' with {type:'json'};

export type ManualPackLocale=Exclude<AppLocale,'zh-CN'>;
export interface ManualPackSpec {locale:ManualPackLocale;url:string;sha256:string;version:string;sourceDigest:string}
export interface InstalledManualPack {
  locale:ManualPackLocale;version:string;sourceDigest:string;installedAt:string;
  projectId:string;manualVersion?:string;softwareVersion?:string;distribution:'public'|'software';
  assetOverrides:Record<string,string>;
}
export interface ManualPackStore {
  list():Promise<InstalledManualPack[]>;
  install(spec:ManualPackSpec):Promise<InstalledManualPack>;
  readJson(locale:ManualPackLocale,path:string):Promise<unknown>;
  readAsset(locale:ManualPackLocale,path:string):Promise<Blob|null>;
  importArchive?(spec:Omit<ManualPackSpec,'url'>,file:Blob):Promise<InstalledManualPack>;
}
interface PackManifest extends Omit<InstalledManualPack,'installedAt'> {
  schemaVersion:'ptb-manual-language-pack/1';hashes:Record<string,string>;sharedAssets:string[];
}
export interface ManualPackRecord {metadata:InstalledManualPack;files:Record<string,Blob>}
export interface ManualPackStorage {
  list():Promise<InstalledManualPack[]>;
  readFile(locale:ManualPackLocale,path:string):Promise<Blob|undefined>;
  put(record:ManualPackRecord):Promise<void>;
}
const digestPattern=/^[a-f0-9]{64}$/;
const versionPattern=/^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/;
const localeValid=(value:unknown):value is ManualPackLocale=>typeof value==='string'&&value!=='zh-CN'&&languages.some(item=>item.code===value);
const object=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const maxArchive=500_000_000,maxExpanded=1_000_000_000,maxFile=64_000_000,maxFiles=10000;
export async function manualPackSha256(bytes:Uint8Array):Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes as Uint8Array<ArrayBuffer>))].map(byte=>byte.toString(16).padStart(2,'0')).join('');
}
function pathAllowed(path:string):boolean {
  return safeRelativePath(path)&&!path.includes('%')&&(path==='project.json'||path==='language-pack.json'||/^chapters\/[A-Za-z0-9._-]+\.json$/.test(path)||/^assets\/[A-Za-z0-9._/-]+$/.test(path));
}
export function manualPackDownloadAllowed(value:string,pageUrl=typeof location==='undefined'?'':location.href):boolean {
  try{
    const url=new URL(value,pageUrl||undefined);
    if(url.username||url.password||url.hash)return false;
    if(url.protocol==='https:'&&['phonetictoolbox.com','www.phonetictoolbox.com'].includes(url.hostname))return true;
    if(url.protocol==='https:'&&url.hostname==='github.com'&&/^\/Nephelium\/PhoneticToolbox\/releases\/download\/[^/]+\/[^/]+$/i.test(url.pathname))return true;
    const loopback=(hostname:string)=>['localhost','127.0.0.1','[::1]'].includes(hostname);
    return !!pageUrl&&loopback(new URL(pageUrl).hostname)&&loopback(url.hostname)&&['http:','https:'].includes(url.protocol);
  }catch{return false;}
}

/** Bounded ZIP reader: hashes are checked before any file reaches persistent storage. */
export async function unpackManualArchive(bytes:Uint8Array,only?:ReadonlySet<string>):Promise<Map<string,Uint8Array>> {
  if(bytes.length>maxArchive||bytes.length<22)throw Error('manual.pack.invalidArchive');
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),decoder=new TextDecoder('utf-8',{fatal:true});
  let end=-1;
  for(let pos=bytes.length-22;pos>=Math.max(0,bytes.length-65557);pos--){if(view.getUint32(pos,true)===0x06054b50&&pos+22+view.getUint16(pos+20,true)===bytes.length){end=pos;break;}}
  if(end<0||view.getUint16(end+4,true)||view.getUint16(end+6,true))throw Error('manual.pack.invalidArchive');
  const count=view.getUint16(end+10,true),centralSize=view.getUint32(end+12,true),centralOffset=view.getUint32(end+16,true);
  if(!count||count>maxFiles||count!==view.getUint16(end+8,true)||centralOffset+centralSize!==end)throw Error('manual.pack.invalidArchive');
  const files=new Map<string,Uint8Array>(),caseNames=new Set<string>(),ranges:Array<[number,number]>=[];
  let cursor=centralOffset,total=0;
  for(let n=0;n<count;n++){
    if(cursor+46>end||view.getUint32(cursor,true)!==0x02014b50)throw Error('manual.pack.invalidArchive');
    const flags=view.getUint16(cursor+8,true),method=view.getUint16(cursor+10,true),compressed=view.getUint32(cursor+20,true),expanded=view.getUint32(cursor+24,true);
    const nameLength=view.getUint16(cursor+28,true),extra=view.getUint16(cursor+30,true),comment=view.getUint16(cursor+32,true),local=view.getUint32(cursor+42,true),attributes=view.getUint32(cursor+38,true);
    if(cursor+46+nameLength+extra+comment>end)throw Error('manual.pack.invalidArchive');
    const name=decoder.decode(bytes.subarray(cursor+46,cursor+46+nameLength));
    if(!pathAllowed(name)||caseNames.has(name.toLowerCase())||(flags&~0x808)||![0,8].includes(method)||expanded>maxFile||(attributes>>>16&0xf000)===0xa000)throw Error('manual.pack.invalidArchive');
    caseNames.add(name.toLowerCase());
    total+=expanded;if(total>maxExpanded||local+30>centralOffset||view.getUint32(local,true)!==0x04034b50)throw Error('manual.pack.invalidArchive');
    const localName=view.getUint16(local+26,true),localExtra=view.getUint16(local+28,true),start=local+30+localName+localExtra,finish=start+compressed;
    if(finish>centralOffset||decoder.decode(bytes.subarray(local+30,local+30+localName))!==name||view.getUint16(local+8,true)!==method||view.getUint16(local+6,true)!==flags||ranges.some(([a,b])=>local<b&&finish>a))throw Error('manual.pack.invalidArchive');
    ranges.push([local,finish]);
    if(only&&!only.has(name)){cursor+=46+nameLength+extra+comment;continue;}
    const packed=bytes.slice(start,finish);
    let data:Uint8Array;
    if(method===0)data=packed;
    else{
      const stream=new Blob([packed]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
      const reader=stream.getReader(),chunks:Uint8Array[]=[];let size=0;
      try{while(true){const result=await reader.read();if(result.done)break;size+=result.value.length;if(size>expanded||size>maxFile)throw Error('manual.pack.invalidArchive');chunks.push(result.value);}}
      finally{await reader.cancel();reader.releaseLock();}
      data=new Uint8Array(size);let offset=0;for(const chunk of chunks){data.set(chunk,offset);offset+=chunk.length;}
    }
    if(data.length!==expanded)throw Error('manual.pack.invalidArchive');
    files.set(name,data);cursor+=46+nameLength+extra+comment;
  }
  if(cursor!==end)throw Error('manual.pack.invalidArchive');
  return files;
}

function parseJson(bytes:Uint8Array|undefined):unknown {
  if(!bytes)throw Error('manual.pack.missingFile');
  return JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes));
}
/** Explicitly selected local ZIPs can be installed before any public release.
 * Only the manifest is inflated for inspection; the store validates every file
 * on installation. The Chinese project and source digest must match exactly.
 */
export async function inspectLocalManualPack(file:Blob,expected:{locale:ManualPackLocale;projectId:string;sourceDigest:string}):Promise<Omit<ManualPackSpec,'url'>>{
  if(file.size>maxArchive||!digestPattern.test(expected.sourceDigest))throw Error('manual.pack.incompatible');
  const bytes=new Uint8Array(await file.arrayBuffer()),archive=await unpackManualArchive(bytes,new Set(['language-pack.json'])),manifest=parseJson(archive.get('language-pack.json'));
  if(!object(manifest)||manifest.schemaVersion!=='ptb-manual-language-pack/1'||manifest.locale!==expected.locale||manifest.projectId!==expected.projectId||manifest.sourceDigest!==expected.sourceDigest||typeof manifest.version!=='string'||!versionPattern.test(manifest.version))throw Error('manual.pack.incompatible');
  return {locale:expected.locale,version:manifest.version,sourceDigest:expected.sourceDigest,sha256:await manualPackSha256(bytes)};
}
export async function validateManualPack(bytes:Uint8Array,spec:ManualPackSpec):Promise<ManualPackRecord> {
  if(!localeValid(spec.locale)||!versionPattern.test(spec.version)||!digestPattern.test(spec.sha256)||!digestPattern.test(spec.sourceDigest))throw Error('manual.pack.invalidSpec');
  const receivedDigest=await manualPackSha256(bytes);
  if(receivedDigest!==spec.sha256)throw Error('manual.pack.hashMismatch',{cause:{stage:'archive',expected:spec.sha256,received:receivedDigest,bytes:bytes.length}});
  const archive=await unpackManualArchive(bytes),manifest=parseJson(archive.get('language-pack.json'));
  if(!object(manifest)||manifest.schemaVersion!=='ptb-manual-language-pack/1'||manifest.locale!==spec.locale||manifest.version!==spec.version||manifest.sourceDigest!==spec.sourceDigest||!object(manifest.hashes)||!object(manifest.assetOverrides)||!Array.isArray(manifest.sharedAssets)||!['public','software'].includes(String(manifest.distribution)))throw Error('manual.pack.invalidManifest');
  const files:Record<string,Blob>=Object.create(null);
  for(const [path,digest] of Object.entries(manifest.hashes)){
    const data=archive.get(path);
    if(path==='language-pack.json'||!pathAllowed(path)||typeof digest!=='string'||!digestPattern.test(digest)||!data||await manualPackSha256(data)!==digest)throw Error('manual.pack.hashMismatch',{cause:'file:'+path});
    files[path]=new Blob([data as Uint8Array<ArrayBuffer>],{type:path.endsWith('.json')?'application/json':path.endsWith('.png')?'image/png':path.endsWith('.webp')?'image/webp':'application/octet-stream'});
  }
  if(archive.size!==Object.keys(files).length+1)throw Error('manual.pack.invalidManifest');
  const project=parseManualProject(parseJson(archive.get('project.json')));
  if(project.id!==manifest.projectId||project.language!==spec.locale||(project as ManualProject&{sourceDigest?:string}).sourceDigest!==spec.sourceDigest)throw Error('manual.pack.incompatible');
  const expected=new Set(['project.json']);
  for(const chapter of project.chapters){expected.add(chapter.path);parseManualChapter(parseJson(archive.get(chapter.path)),chapter.id);}
  const ids=new Set(project.assets.map(asset=>asset.id)),shared=manifest.sharedAssets,overrides=manifest.assetOverrides;
  if(shared.some(id=>typeof id!=='string'||!ids.has(id))||new Set(shared).size!==shared.length)throw Error('manual.pack.invalidManifest');
  if(project.assets.some(asset=>!shared.includes(asset.id)&&!(asset.id in overrides)))throw Error('manual.pack.invalidManifest');
  for(const [id,path] of Object.entries(overrides)){
    if(!ids.has(id)||typeof path!=='string'||!path.startsWith('assets/')||!files[path]||!project.assets.some(asset=>asset.id===id&&asset.kind==='image'))throw Error('manual.pack.invalidManifest');
    expected.add(path);
  }
  if(Object.keys(files).some(path=>!expected.has(path))||expected.size!==Object.keys(files).length)throw Error('manual.pack.invalidManifest');
  const typed=manifest as unknown as PackManifest;
  return {metadata:{locale:spec.locale,version:spec.version,sourceDigest:spec.sourceDigest,installedAt:new Date().toISOString(),projectId:project.id,manualVersion:typed.manualVersion,softwareVersion:typed.softwareVersion,distribution:typed.distribution,assetOverrides:typed.assetOverrides},files};
}

export class IndexedDbManualPackStorage implements ManualPackStorage {
  private database:Promise<IDBDatabase>|undefined;
  private open():Promise<IDBDatabase>{
    return this.database??=new Promise((resolve,reject)=>{
      const request=indexedDB.open('phonetic-toolbox-manual-packs',1);
      request.onupgradeneeded=()=>{request.result.createObjectStore('packs',{keyPath:'locale'});request.result.createObjectStore('files',{keyPath:'key'});};
      request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);request.onblocked=()=>reject(Error('manual.pack.storageBlocked'));
    });
  }
  private async read<T>(name:string,operation:(store:IDBObjectStore)=>IDBRequest<T>):Promise<T>{
    const db=await this.open();return new Promise((resolve,reject)=>{const transaction=db.transaction(name,'readonly'),request=operation(transaction.objectStore(name));request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});
  }
  list(){return this.read<InstalledManualPack[]>('packs',store=>store.getAll());}
  async readFile(locale:ManualPackLocale,path:string){return (await this.read<{file:Blob}|undefined>('files',store=>store.get([locale,path])))?.file;}
  async put(record:ManualPackRecord):Promise<void>{
    const db=await this.open();await new Promise<void>((resolve,reject)=>{
      const transaction=db.transaction(['packs','files'],'readwrite'),files=transaction.objectStore('files');
      files.delete(IDBKeyRange.bound([record.metadata.locale,''],[record.metadata.locale,'\uffff']));
      for(const [path,file] of Object.entries(record.files))files.put({key:[record.metadata.locale,path],file});
      transaction.objectStore('packs').put(record.metadata);
      transaction.oncomplete=()=>resolve();transaction.onerror=()=>reject(transaction.error);transaction.onabort=()=>reject(transaction.error??Error('manual.pack.storageFailed'));
    });
  }
}

export function createBrowserManualPackStore(options:{storage?:ManualPackStorage;fetcher?:typeof fetch;pageUrl?:string}={}):ManualPackStore {
  const storage=options.storage??new IndexedDbManualPackStorage(),fetcher=options.fetcher??((...args)=>fetch(...args));
  let installing=false;
  return {
    list:()=>storage.list(),
    async install(spec){
      if(installing)throw Error('manual.pack.installBusy');
      if(!manualPackDownloadAllowed(spec.url,options.pageUrl))throw Error('manual.pack.invalidUrl');
      installing=true;
      try{
        const response=await fetcher(spec.url,{credentials:'omit',redirect:'error',cache:'no-store'});
        if(!response.ok||response.status===204||Number(response.headers.get('content-length'))>maxArchive||!response.body)throw Error('manual.pack.downloadFailed');
        const reader=response.body.getReader(),chunks:Uint8Array[]=[];let size=0;
        try{while(true){const result=await reader.read();if(result.done)break;size+=result.value.length;if(size>maxArchive)throw Error('manual.pack.invalidArchive');chunks.push(result.value);}}
        finally{await reader.cancel();reader.releaseLock();}
        if(!size)throw Error('manual.pack.downloadFailed');
        const bytes=new Uint8Array(size);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
        const record=await validateManualPack(bytes,spec);
        await storage.put(record);return record.metadata;
      }finally{installing=false;}
    },
    async importArchive(spec,file){
      if(installing)throw Error('manual.pack.installBusy');
      if(file.size>maxArchive)throw Error('manual.pack.invalidArchive');
      installing=true;
      try{const record=await validateManualPack(new Uint8Array(await file.arrayBuffer()),{...spec,url:''});await storage.put(record);return record.metadata;}finally{installing=false;}
    },
    async readJson(locale,path){if(!safeRelativePath(path)||!path.endsWith('.json'))throw Error('manual.pack.invalidPath');const file=await storage.readFile(locale,path);if(!file)throw Error('manual.pack.missingFile');return JSON.parse(await file.text());},
    async readAsset(locale,path){if(!safeRelativePath(path)||!path.startsWith('assets/'))throw Error('manual.pack.invalidPath');return await storage.readFile(locale,path)??null;}
  };
}

let activeStore:ManualPackStore|undefined;
/** Called by the desktop bridge after handshake; browsers use IndexedDB. */
export function installManualPackStore(store:ManualPackStore){activeStore=store;}
export function manualPackStore():ManualPackStore{return activeStore??=createBrowserManualPackStore();}

export interface ManualPackDownloadManifest {schemaVersion:'ptb-manual-downloads/1';published:boolean;packs:ManualPackSpec[]}
export async function fetchManualPackDownloads(url:string,fetcher:typeof fetch=fetch):Promise<ManualPackDownloadManifest>{
  const response=await fetcher(url,{credentials:'same-origin',cache:'no-cache'});
  if(!response.ok)throw Error('manual.pack.manifestUnavailable');
  const value=await response.json();
  if(!object(value)||value.schemaVersion!=='ptb-manual-downloads/1'||typeof value.published!=='boolean'||!Array.isArray(value.packs))throw Error('manual.pack.invalidManifest');
  for(const item of value.packs)if(!object(item)||!localeValid(item.locale)||typeof item.url!=='string'||!manualPackDownloadAllowed(item.url)||typeof item.version!=='string'||!versionPattern.test(item.version)||typeof item.sha256!=='string'||!digestPattern.test(item.sha256)||typeof item.sourceDigest!=='string'||!digestPattern.test(item.sourceDigest))throw Error('manual.pack.invalidSpec');
  if(!value.published&&value.packs.length)throw Error('manual.pack.invalidManifest');
  if(new Set(value.packs.map(item=>item.locale)).size!==value.packs.length)throw Error('manual.pack.invalidManifest');
  return value as unknown as ManualPackDownloadManifest;
}
