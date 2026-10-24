import type {Resources} from '../modules/geolinguistics/model.ts';
import {LocalAssets} from './local-assets.ts';
import {downloadGeoData,type GeoDownloadProgress} from './geolinguistics-download.ts';
export type {GeoDownloadProgress} from './geolinguistics-download.ts';
export class GeoDataUnavailableError extends Error {constructor(){super('地图资料尚未下载。点击下载资料后即可载入调查点与地图。');this.name='GeoDataUnavailableError';}}
import {readGeoDirectory,readGeoManifest,geoDataNames,geoMapNames,geoMatrixNames,validateGeoLayers,type GeoFile} from './geolinguistics-directory.ts';
export interface GeoPort {load(signal?:AbortSignal):Promise<Resources>;map(name:string):Promise<object>;matrix(name:string):Promise<Float64Array>;save(name:string,blob:Blob):Promise<'saved'|'requested'|'cancelled'>;useDirectory(files:File[]):Promise<void>;download(progress:(value:GeoDownloadProgress)=>void,signal?:AbortSignal):Promise<void>}
const root=new URL('./geolinguistics/',document.baseURI);
export function createGeoPort(scope:string):GeoPort{
const storage=new LocalAssets(scope,'geolinguistics-resources');
type Snapshot={resources:Resources}&Awaited<ReturnType<typeof validateGeoLayers>>;
type RestoreRun={controller:AbortController;consumers:Set<symbol>;promise:Promise<void>};
let snapshot:Snapshot|undefined,generation=0,restoring:RestoreRun|undefined,imports:Promise<void>=Promise.resolve();
async function bytes(name:string,signal?:AbortSignal){signal?.throwIfAborted();const unavailable=()=>name==='data-manifest.json'?new GeoDataUnavailableError():Error('语言地图资料不完整，请重新下载资料。');let response:Response;try{response=await fetch(new URL(name,root),{signal});}catch(e){if(signal?.aborted)throw e;throw unavailable();}signal?.throwIfAborted();if(!response.ok||response.headers.get('content-type')?.includes('text/html'))throw unavailable();const blob=await response.blob();signal?.throwIfAborted();return blob;}
const reader=(source:Map<string,Blob>)=>async<T>(name:string,signal?:AbortSignal):Promise<T>=>{signal?.throwIfAborted();const file=source.get(name);if(!file)throw Error('语言地图资料不完整，请重新选择完整资料夹。');const text=await file.text();signal?.throwIfAborted();return JSON.parse(text);};
async function readResources(read:<T>(name:string,signal?:AbortSignal)=>Promise<T>,signal?:AbortSignal):Promise<Resources>{
 const [catalog,sites,observations,classification,admin,paper]=await Promise.all([read<Resources['catalog']>('catalog.json',signal),read<Resources['sites']>('sites.json',signal),read<Resources['observations']>('observations.json',signal),read<Resources['classification']>('classification.json',signal),read<Resources['admin']>('map-index.json',signal),read<Resources['paper']>('paper.json',signal)]);
 if(catalog.version!=='m19-local-20261022'||!Array.isArray(sites)||sites.length!==930||!Array.isArray(catalog.features)||catalog.features.length!==510||new Set(catalog.features.map(f=>f.id)).size!==510||sites.some((s,i)=>s.id!==`S${i+1}`||!Number.isFinite(s.lon)||!Number.isFinite(s.lat))||catalog.features.some(f=>!Array.isArray(observations[f.id]?.values)||observations[f.id].values.length!==930||!Array.isArray(f.categories)||!f.labels)||!Array.isArray(classification.sheets)||!Array.isArray(admin)||paper.points?.length!==930)throw Error('语言地图资料结构不完整，请重新选择完整资料夹。');
 return {catalog,sites,observations,classification,admin,paper};
}
async function commit(checked:GeoFile[],signal?:AbortSignal,beforeSave?:()=>void){const candidate=await validateCandidate(new Map(checked.map(v=>[v.name,v.blob])),signal);signal?.throwIfAborted();beforeSave?.();signal?.throwIfAborted();await storage.setValue('data',checked);snapshot=candidate;generation++;if(restoring){const old=restoring;restoring=undefined;old.controller.abort();}}
async function validateCandidate(candidate:Map<string,Blob>,signal?:AbortSignal):Promise<Snapshot>{signal?.throwIfAborted();const resources=await readResources(reader(candidate),signal);signal?.throwIfAborted();return {resources,...await validateGeoLayers(candidate,signal)};}
function startRestore():RestoreRun{
 const run:RestoreRun={controller:new AbortController(),consumers:new Set(),promise:Promise.resolve()};restoring=run;
 const signal=run.controller.signal;
 run.promise=(async()=>{
   const version=generation;
   try{
    const saved=await storage.value<GeoFile[]>('data');
    signal.throwIfAborted();if(snapshot||version!==generation)return;
    let files:Map<string,Blob>;
    if(saved){
     if(!Array.isArray(saved)||saved.length!==geoDataNames.length||saved.some(v=>!v||!geoDataNames.includes(v.name as typeof geoDataNames[number])||!(v.blob instanceof Blob))||new Set(saved.map(v=>v.name)).size!==geoDataNames.length)throw Error('本机语言地图资料不完整，请重新选择资料夹。');
     files=new Map(saved.map(v=>[v.name,v.blob]));
    }else{
     const checked=await readGeoManifest(await bytes('data-manifest.json',signal),name=>bytes(name,signal),signal);
     files=new Map(checked.map(v=>[v.name,v.blob]));
    }
    const candidate=await validateCandidate(files,signal);signal.throwIfAborted();
    // An explicit import may complete while the old persisted snapshot is being read.
    if(!snapshot&&version===generation&&restoring===run)snapshot=candidate;
   }catch(error){if(!snapshot&&version===generation)throw error;}
  })().finally(()=>{if(restoring===run)restoring=undefined;});
 // A cancelled last consumer can leave the shared request settling after its caller.
 void run.promise.catch(()=>{});return run;
}
async function restore(signal?:AbortSignal){
 signal?.throwIfAborted();if(snapshot)return;
 const run=restoring&&!restoring.controller.signal.aborted?restoring:startRestore(),consumer=Symbol();run.consumers.add(consumer);
 await new Promise<void>((resolve,reject)=>{
  let settled=false;
  const release=()=>{if(settled)return false;settled=true;signal?.removeEventListener('abort',abort);run.consumers.delete(consumer);if(!run.consumers.size&&restoring===run){restoring=undefined;run.controller.abort();}return true;};
  const abort=()=>{if(release())reject(signal!.reason);};
  signal?.addEventListener('abort',abort,{once:true});
  run.promise.then(()=>{if(release())resolve();},error=>{if(release())reject(error);});
  if(signal?.aborted)abort();
 });
}
async function current<T>(read:(value:Snapshot)=>T,signal?:AbortSignal):Promise<T>{
 signal?.throwIfAborted();await restore(signal);signal?.throwIfAborted();return read(snapshot!);
}
return {
 useDirectory(files){
  const selected=[...files];
  const operation=imports.then(async()=>{await commit(await readGeoDirectory(selected));});
  imports=operation.catch(()=>{});return operation;
 },
 download(progress,signal){
  const operation=imports.then(async()=>{const checked=await downloadGeoData(progress,signal),total=checked.reduce((n,f)=>n+f.blob.size,0);await commit(checked,signal,()=>progress({phase:'save',loaded:total,total,files:checked.length,totalFiles:checked.length}));});
  imports=operation.catch(()=>{});return operation;
 },
 load(signal){return current(value=>value.resources,signal);},
 map(name){if(!geoMapNames.includes(name as typeof geoMapNames[number]))throw Error('未知地图图层。');return current(value=>value.maps.get(name)!);},
 matrix(name){if(!geoMatrixNames.includes(name as typeof geoMatrixNames[number]))throw Error('未知距离方案。');return current(value=>value.matrices.get(name)!);},
 async save(name,blob){const picker=document.location.protocol==='ptbapp:'?undefined:(window as any).showSaveFilePicker;if(picker){let handle;try{handle=await picker.call(window,{suggestedName:name});}catch(e){if((e as Error).name==='AbortError')return 'cancelled';throw e;}const writer=await handle.createWritable();try{await writer.write(blob);await writer.close();return 'saved';}catch(e){await writer.abort().catch(()=>{});throw e;}}const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);return 'requested';}
};
}
