/** A user-selected, data-only folder. Validation finishes before replacing the active resource set. */
export const geoDataNames=['catalog.json','sites.json','observations.json','classification.json','map-index.json','paper.json','main.f64','type-0-0.f64','type-0-1.f64','type-1-0.f64','type-1-1.f64','maps/world.json','maps/territory.json','maps/province.json','maps/city.json','maps/district.json'] as const;
export type GeoFile={name:string;blob:Blob};
export type GeoDataManifest={schema:'ptb-geolinguistics-data/1';version:'m19-local-20261022';profile:'interactive-data';files:{path:string;bytes:number;sha256:string}[]};
export const geoMatrixNames=['main','type-0-0','type-0-1','type-1-0','type-1-1'] as const;
export const geoMapNames=['world','territory','province','city','district'] as const;

/** These are the original 930-site distances, not editable or recomputed results. */
export function validateGeoMatrix(buffer:ArrayBuffer):Float64Array{
 const size=930;
 if(buffer.byteLength!==size*size*8)throw Error('距离矩阵不完整。');
 const values=new Float64Array(buffer);
 for(let i=0;i<size;i++)for(let j=0;j<size;j++){
  const value=values[i*size+j];
  if(!Number.isFinite(value)||value<0||(i===j&&value!==0)||(j<i&&value!==values[j*size+i]))throw Error('距离矩阵含无效数值，或不满足对称、对角为零的原始距离格式。');
 }
 return values;
}
export function validateGeoMap(value:unknown):object{
 const record=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==='object'&&!Array.isArray(v);
 const list=(v:unknown,check:(item:unknown)=>boolean,min=1):boolean=>Array.isArray(v)&&v.length>=min&&v.every(check);
 const position=(v:unknown):v is number[]=>Array.isArray(v)&&v.length>=2&&v.every(n=>typeof n==='number'&&Number.isFinite(n))&&Math.abs(v[0])<=180&&Math.abs(v[1])<=90;
 const line=(v:unknown)=>list(v,position,2);
 const ring=(v:unknown)=>list(v,position,4)&&Array.isArray(v)&&v[0].length===v[v.length-1].length&&v[0].every((n:number,i:number)=>n===v[v.length-1][i]);
 const polygon=(v:unknown)=>list(v,ring);
 function geometry(v:unknown,depth=0):boolean{
  if(!record(v)||depth>32)return false;
  switch(v.type){
   case 'LineString':return line(v.coordinates);
   case 'MultiLineString':return list(v.coordinates,line);
   case 'Polygon':return polygon(v.coordinates);
   case 'MultiPolygon':return list(v.coordinates,polygon);
   case 'GeometryCollection':return list(v.geometries,g=>geometry(g,depth+1));
   default:return false;
  }
 }
 if(!record(value)||value.type!=='FeatureCollection'||!list(value.features,f=>record(f)&&f.type==='Feature'&&(f.properties===null||record(f.properties))&&geometry(f.geometry)))throw Error('地图资料不是有效的地理边界集合，请重新选择完整资料夹。');
 return value;
}
/** Validate every delayed layer before a candidate may replace persisted resources. */
export async function validateGeoLayers(files:Map<string,Blob>,signal?:AbortSignal){
 const matrices=new Map<string,Float64Array>(),maps=new Map<string,object>();
 for(const name of geoMatrixNames){signal?.throwIfAborted();const file=files.get(name+'.f64');if(!file)throw Error('距离矩阵不完整。');const buffer=await file.arrayBuffer();signal?.throwIfAborted();matrices.set(name,validateGeoMatrix(buffer));}
 for(const name of geoMapNames){signal?.throwIfAborted();const file=files.get('maps/'+name+'.json');if(!file)throw Error('地图资料不完整。');const text=await file.text();signal?.throwIfAborted();maps.set(name,validateGeoMap(JSON.parse(text)));}
 signal?.throwIfAborted();return {matrices,maps};
}
export async function readGeoDirectory(files:File[]):Promise<GeoFile[]>{
 const candidates=files.filter(f=>f.name==='data-manifest.json');
 if(candidates.length!==1)throw Error('请选择包含 data-manifest.json 的语言地图资料夹。');
 const manifestFile=candidates[0],base=manifestFile.webkitRelativePath.slice(0,-manifestFile.name.length),byPath=new Map(files.map(f=>[f.webkitRelativePath||f.name,f]));
 return readGeoManifest(manifestFile,async path=>byPath.get(base+path));
}
/** Automatic sidecar/web loads and user-selected folders share the exact manifest contract. */
export async function readGeoManifest(manifestFile:Blob,read:(path:string)=>Promise<Blob|undefined>,signal?:AbortSignal):Promise<GeoFile[]>{
 return readGeoManifestFiles(await parseGeoManifest(manifestFile,signal),read,signal);
}
/** The manifest has already passed parseGeoManifest; downloads need its sizes before streaming. */
export async function readGeoManifestFiles(manifest:GeoDataManifest,read:(path:string)=>Promise<Blob|undefined>,signal?:AbortSignal):Promise<GeoFile[]>{
 const result:GeoFile[]=[];
 for(const item of manifest.files){
  signal?.throwIfAborted();
  const file=await read(item.path);
  signal?.throwIfAborted();
  if(!file||file.size!==item.bytes)throw Error('资料文件缺失或大小不一致：'+item.path);
  const buffer=await file.arrayBuffer();signal?.throwIfAborted();
  const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))].map(n=>n.toString(16).padStart(2,'0')).join('');
  signal?.throwIfAborted();
  if(digest!==item.sha256)throw Error('资料校验未通过：'+item.path+'，请重新取得完整资料。');
  result.push({name:item.path,blob:file});
 }
 return result;
}
export async function parseGeoManifest(manifestFile:Blob,signal?:AbortSignal):Promise<GeoDataManifest>{
 signal?.throwIfAborted();
 if(manifestFile.size>100_000)throw Error('资料清单过大。');
 const text=await manifestFile.text();signal?.throwIfAborted();const manifest=JSON.parse(text);
 if(!manifest||manifest.schema!=='ptb-geolinguistics-data/1'||manifest.version!=='m19-local-20261022'||manifest.profile!=='interactive-data'||!Array.isArray(manifest.files)||manifest.files.length!==geoDataNames.length)throw Error('语言地图资料清单不匹配。');
 const seen=new Set<string>();let total=0;
 for(const item of manifest.files){
  if(!item||!geoDataNames.includes(item.path)||seen.has(item.path)||!Number.isSafeInteger(item.bytes)||item.bytes<1||item.bytes>100_000_000||!/^[a-f0-9]{64}$/.test(item.sha256))throw Error('资料清单含无效条目。');
  seen.add(item.path);total+=item.bytes;if(total>200_000_000)throw Error('资料文件总大小超出限制。');
 }
 return manifest;
}
