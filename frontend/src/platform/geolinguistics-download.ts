import {parseGeoManifest,readGeoManifestFiles,type GeoFile} from './geolinguistics-directory.ts';

export const geoDownloadRoot='https://www.phonetictoolbox.com/app/geolinguistics/';
export type GeoDownloadProgress={phase:'manifest'|'download'|'validate'|'save';loaded:number;total:number;files:number;totalFiles:number};

/** Download only the fixed runtime allowlist; commit is owned by the resource port. */
export async function downloadGeoData(progress:(value:GeoDownloadProgress)=>void,signal?:AbortSignal):Promise<GeoFile[]>{
 const state:GeoDownloadProgress={phase:'manifest',loaded:0,total:0,files:0,totalFiles:16};
 let reportedAt=0;
 const report=(force=false)=>{const now=performance.now();if(force||now-reportedAt>=100){reportedAt=now;progress({...state});}};report(true);
 async function get(name:string,limit:number,track=false){
  signal?.throwIfAborted();
  const response=await fetch(new URL(name,geoDownloadRoot),{signal,credentials:'omit',redirect:'error',cache:'no-store'});
  if(!response.ok||response.headers.get('content-type')?.includes('text/html'))throw Error('服务器资料暂时无法下载，请稍后重试，也可以选择本地资料夹。');
  if(!response.body)throw Error('资料下载未完成，请重试。');
  const reader=response.body.getReader(),chunks:Uint8Array<ArrayBuffer>[]= [];let size=0;
  try{
   for(;;){signal?.throwIfAborted();const {done,value}=await reader.read();if(done)break;
    size+=value.byteLength;if(size>limit)throw Error('下载的资料大小不符，请重试。');
    chunks.push(value as Uint8Array<ArrayBuffer>);if(track){state.loaded+=value.byteLength;report();}
   }
  }catch(error){await reader.cancel().catch(()=>{});throw error;}finally{reader.releaseLock();}
  signal?.throwIfAborted();return new Blob(chunks);
 }
 const raw=await get('data-manifest.json',100_000),manifest=await parseGeoManifest(raw,signal);
 state.total=manifest.files.reduce((n,f)=>n+f.bytes,0);state.phase='download';report(true);
 const sizes=new Map(manifest.files.map(f=>[f.path,f.bytes]));
 const files=await readGeoManifestFiles(manifest,async name=>{const blob=await get(name,sizes.get(name)!,true);state.files++;report(true);return blob;},signal);
 signal?.throwIfAborted();state.phase='validate';report(true);return files;
}
