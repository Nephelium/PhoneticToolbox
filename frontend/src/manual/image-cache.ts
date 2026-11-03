import {shallowReactive} from 'vue';

/** Lazy image blobs are scoped to the installed version and addressed by payload. */
export class ManualImageCache{
 readonly urls=shallowReactive(new Map<string,string>());
 readonly pending=shallowReactive(new Set<string>());
 private failed=new Set<string>();
 private blobs=new Map<string,Blob>();
 private bytes=0;
 private visible=new Set<string>();
 private generation=0;
 private scope='';
 private read:(path:string)=>Promise<Blob|null>;
 private objectUrls:Pick<typeof URL,'createObjectURL'|'revokeObjectURL'>;
 constructor(read:(path:string)=>Promise<Blob|null>,objectUrls:Pick<typeof URL,'createObjectURL'|'revokeObjectURL'>=URL){this.read=read;this.objectUrls=objectUrls;}
 private key(path:string){return this.scope+':'+path;}
 configure(scope:string){if(this.scope!==scope){this.clear();this.scope=scope;}}
 retain(paths:string[]){
  this.visible=new Set(paths.map(path=>this.key(path)));
  for(const [key,url]of this.urls)if(!this.visible.has(key)){this.objectUrls.revokeObjectURL(url);this.urls.delete(key);}
 }
 private cache(key:string,blob:Blob){
  if(blob.size>32_000_000)return;
  const prior=this.blobs.get(key);if(prior)this.bytes-=prior.size;
  this.blobs.delete(key);this.blobs.set(key,blob);this.bytes+=blob.size;
  while(this.blobs.size>64||this.bytes>32_000_000){const oldest=this.blobs.keys().next().value!;this.bytes-=this.blobs.get(oldest)!.size;this.blobs.delete(oldest);}
 }
 resolve(path:string):string|null{
  const key=this.key(path),prior=this.urls.get(key);if(prior)return prior;
  const cached=this.blobs.get(key);
  if(cached){this.cache(key,cached);const url=this.objectUrls.createObjectURL(cached);this.urls.set(key,url);return url;}
  if(!this.pending.has(key)&&!this.failed.has(key)){
   this.pending.add(key);const generation=this.generation;
   void this.read(path).then(blob=>{
    if(generation!==this.generation)return;
    if(blob){this.cache(key,blob);if(this.visible.has(key))this.urls.set(key,this.objectUrls.createObjectURL(blob));}
    else this.failed.add(key);
   }).catch(()=>{if(generation===this.generation)this.failed.add(key);}).finally(()=>{if(generation===this.generation)this.pending.delete(key);});
  }
  return null;
 }
 isLoading(path:string){return this.pending.has(this.key(path));}
 clear(){this.generation++;for(const url of this.urls.values())this.objectUrls.revokeObjectURL(url);this.urls.clear();this.pending.clear();this.failed.clear();this.visible.clear();}
 dispose(){this.clear();this.blobs.clear();this.bytes=0;}
}
