import {assetUrl,parseManualProject} from './content.ts';
import type {ManualAsset,ManualProject} from './types.ts';
import {manualPackSha256,type InstalledManualPack} from '../platform/manual-packs.ts';

const cacheName='phonetic-toolbox-manual-shared-v1';
const hash=/^[a-f0-9]{64}$/;
/** Only an explicit installation downloads shared media. Reading never fetches. */
export class ManualSharedCache{
 private project:ManualProject;
 private base:string;
 private storage:CacheStorage;
 private fetcher:typeof fetch;
 constructor(project:ManualProject,base:string,storage:CacheStorage=caches,fetcher:typeof fetch=(...args)=>fetch(...args)){this.project=project;this.base=base;this.storage=storage;this.fetcher=fetcher;}
 private key(digest:string){return new URL('/__ptb_manual_shared__/sha256/'+digest,location.href).href;}
 private assets(overrides:Record<string,string>,requiredIds?:ReadonlySet<string>){
  const unique=new Map<string,ManualAsset>();
  if(requiredIds){const hostIds=new Set(this.project.assets.map(asset=>asset.id));for(const id of requiredIds)if(!overrides[id]&&!hostIds.has(id))throw Error('manual.pack.missingFile');}
  for(const asset of this.project.assets){
   if(overrides[asset.id]||requiredIds&&!requiredIds.has(asset.id))continue;
   if(!asset.sha256||!hash.test(asset.sha256)||!assetUrl(this.base,asset.path))throw Error('manual.pack.incompatible');
   if(!unique.has(asset.sha256))unique.set(asset.sha256,asset);
  }
  return [...unique.values()];
 }
 async ready(overrides:Record<string,string>,requiredIds?:ReadonlySet<string>){
  const cache=await this.storage.open(cacheName);
  for(const asset of this.assets(overrides,requiredIds))if(!await this.valid(cache,asset.sha256!))return false;
  const saved=await cache.match(await this.projectKey());
  return !!saved&&await saved.text()===JSON.stringify(this.project);
 }
 private async valid(cache:Cache,digest:string){
  const response=await cache.match(this.key(digest));
  return !!response&&await manualPackSha256(new Uint8Array(await response.arrayBuffer()))===digest;
 }
 private async projectKey(){return new URL('/__ptb_manual_shared__/projects/'+encodeURIComponent(this.project.id)+'/'+encodeURIComponent(this.project.sourceDigest??'')+'/'+await manualPackSha256(new TextEncoder().encode(JSON.stringify(this.project))),location.href).href;}
 /** Offline reopening uses only a hash-verified base index matching an installed
  * pack. An online host project is always attempted first by the caller. */
 static async restore(packs:InstalledManualPack[],storage:CacheStorage=caches):Promise<ManualProject|undefined>{
  const cache=await storage.open(cacheName);
  for(const pack of packs){
   const prefix=new URL('/__ptb_manual_shared__/projects/'+encodeURIComponent(pack.projectId)+'/'+encodeURIComponent(pack.sourceDigest)+'/',location.href).href;
   for(const request of await cache.keys()){
    if(!request.url.startsWith(prefix))continue;
    const digest=request.url.slice(prefix.length);if(!hash.test(digest))continue;
    try{
     const response=await cache.match(request);if(!response)continue;
     const bytes=new Uint8Array(await response.arrayBuffer());if(await manualPackSha256(bytes)!==digest)continue;
     const project=parseManualProject(JSON.parse(new TextDecoder().decode(bytes)));
     if(project.id===pack.projectId&&project.sourceDigest===pack.sourceDigest)return project;
    }catch{/* A malformed stale index never replaces a verified one. */}
   }
  }
 }
 async install(overrides:Record<string,string>,requiredIds?:ReadonlySet<string>){
  const cache=await this.storage.open(cacheName);let total=0;
  for(const asset of this.assets(overrides,requiredIds)){
   const key=this.key(asset.sha256!);if(await this.valid(cache,asset.sha256!))continue;
   const url=new URL(assetUrl(this.base,asset.path)!,location.href);
   if(url.origin!==location.origin)throw Error('manual.pack.invalidUrl');
   const response=await this.fetcher(url.href,{credentials:'same-origin',redirect:'error'});
   if(!response.ok)throw Error('manual.pack.sharedUnavailable');
   const declared=Number(response.headers.get('Content-Length'));
   if(declared>64_000_000)throw Error('manual.pack.invalidArchive');
   const reader=response.body?.getReader();if(!reader)throw Error('manual.pack.sharedUnavailable');
   const chunks:Uint8Array[]=[];let size=0;
   try{while(true){const part=await reader.read();if(part.done)break;size+=part.value.length;total+=part.value.length;if(size>64_000_000||total>128_000_000)throw Error('manual.pack.invalidArchive');chunks.push(part.value);}}
   finally{await reader.cancel();reader.releaseLock();}
   const bytes=new Uint8Array(size);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
   if(await manualPackSha256(bytes)!==asset.sha256)throw Error('manual.pack.hashMismatch');
   await cache.put(key,new Response(bytes,{headers:{'Content-Type':asset.mime??response.headers.get('Content-Type')??'application/octet-stream'}}));
  }
  // The current packaged base index is retained with its exact host paths and
  // hashes. The completion marker is written only after every shared payload.
  await cache.put(await this.projectKey(),new Response(JSON.stringify(this.project),{headers:{'Content-Type':'application/json'}}));
 }
 async read(digest:string):Promise<Blob|null>{
  if(!hash.test(digest))return null;
  const response=await(await this.storage.open(cacheName)).match(this.key(digest));
  if(!response)return null;
  const blob=await response.blob();
  return await manualPackSha256(new Uint8Array(await blob.arrayBuffer()))===digest?blob:null;
 }
}
