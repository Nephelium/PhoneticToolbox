import {LocalAssets,LocalStorageError,localAssetId,localDigest,type LocalAsset} from './local-assets.ts';
export type AssetFetch=(path:string,init?:RequestInit)=>Promise<Response>;
/** Only a server computation request materializes its referenced local inputs remotely. */
export class LocalAssetTransport {
 private uploads=new Map<string,{id:string;name:string;size:number;chunks:Blob[];offset:number;key:string}>();
 private mirrors=new Map<string,Promise<string>>();
 readonly files:LocalAssets;private remote:AssetFetch;private lifetime:AbortSignal;readonly project:string;
 constructor(files:LocalAssets,remote:AssetFetch,lifetime:AbortSignal,project=files.project){this.files=files;this.remote=remote;this.lifetime=lifetime;this.project=project;}
 private check(signal?:AbortSignal|null){this.lifetime.throwIfAborted();signal?.throwIfAborted();}
 view(f:LocalAsset){return {id:f.id,project_id:this.project,name:f.name,kind:'input',state:'ready',size_bytes:f.blob.size,sha256:f.sha256,created_at:f.created_at,expires_at:null,policy_version:1,storage:'browser'};}
 private async payload(value:unknown,signal?:AbortSignal|null,role?:string):Promise<unknown>{
  if(typeof value==='string'&&(role==='asset_id'||role==='inputs')){const local=await this.files.get(value);if(!local&&localAssetId(value))throw new LocalStorageError('任务引用的本机文件不存在，请重新导入。');return local?this.server(local,signal):value;}
  if(Array.isArray(value))return Promise.all(value.map(v=>this.payload(v,signal,role)));
  if(value&&typeof value==='object'){const result:Record<string,unknown>={};for(const [key,v]of Object.entries(value)){const record=role==='hashes'?await this.files.get(key):undefined;result[record?await this.server(record,signal):key]=await this.payload(v,signal,key);}return result;}
  return value;
 }
 async server(file:LocalAsset,signal?:AbortSignal|null):Promise<string>{
  this.check(signal);const existing=this.mirrors.get(file.id);if(existing)return existing;
  const work=(async()=>{
   if(file.server&&file.server.project===this.project){const r=await this.remote('assets/'+encodeURIComponent(file.server.id),{signal:signal??this.lifetime});if(r.ok){const value=await r.json();if(value.project_id===this.project&&value.state==='ready'&&value.sha256===file.sha256&&(value.expires_at==null||value.expires_at>Date.now()/1000+300))return file.server.id;}else if(![404,410].includes(r.status))return this.failed(r);file.server=undefined;file.mirrorKey=undefined;}
   this.check(signal);const raw=await file.blob.arrayBuffer();if(await localDigest(file.blob)!==file.sha256)throw new LocalStorageError('本机文件校验失败，请重新导入。');
   const prefix='browser-input-'+this.project+'-';if(!file.mirrorKey?.startsWith(prefix))file.mirrorKey=prefix+crypto.randomUUID();await this.files.remember(file);const key=file.mirrorKey;
   let response=await this.remote('uploads',{method:'POST',signal:signal??this.lifetime,body:JSON.stringify({project_id:this.project,name:file.name,expected_bytes:raw.byteLength,idempotency_key:key})});if(!response.ok)return this.failed(response);
   let asset=await response.json();if(!['uploading','ready'].includes(asset.state))throw Error('服务器输入准备未完成，请重试。');
   while(asset.state==='uploading'&&asset.size_bytes<raw.byteLength){this.check(signal);const offset=asset.size_bytes;response=await this.remote('uploads/'+asset.id+'/blocks?offset='+offset,{method:'PUT',signal:signal??this.lifetime,body:new Blob([raw.slice(offset,offset+262144)]),headers:{'Content-Type':'application/octet-stream'}});if(!response.ok)return this.failed(response);asset=await response.json();if(asset.size_bytes<=offset||asset.size_bytes>raw.byteLength)throw Error('服务器输入传输进度异常，请重试。');}
   this.check(signal);response=await this.remote('uploads/'+asset.id+'/finalize',{method:'POST',signal:signal??this.lifetime,body:JSON.stringify({sha256:file.sha256})});if(!response.ok)return this.failed(response);asset=await response.json();if(asset.sha256!==file.sha256)throw Error('服务器输入校验失败。');
   this.check(signal);file.server={id:asset.id,expires_at:asset.expires_at,project:this.project};await this.files.remember(file);return asset.id as string;
  })();this.mirrors.set(file.id,work);try{return await work;}finally{this.mirrors.delete(file.id);}
 }
 private async failed(response:Response):Promise<never>{const body=await response.json().catch(()=>({}));throw Error(body.detail??'服务器输入准备失败，请重试。');}
 async fetch(path:string,init:RequestInit={}):Promise<Response>{
  const method=init.method??'GET',signal=init.signal;this.check(signal);
  const body=typeof init.body==='string'?JSON.parse(init.body):undefined;
  if(path==='uploads'&&method==='POST'){
   if(!Number.isSafeInteger(body.expected_bytes)||body.expected_bytes<1||body.expected_bytes>100_000_000)throw new LocalStorageError('请选择 100 MB 以内的非空文件。');
   const saved=(await this.files.list()).find(f=>f.key===body.idempotency_key);if(saved)return Response.json(this.view(saved));
   const pending=[...this.uploads.values()].find(f=>f.key===body.idempotency_key);if(pending)return Response.json({id:pending.id,state:'uploading',size_bytes:pending.offset});
   const id='ptb-local:'+crypto.randomUUID();this.uploads.set(id,{id,name:body.name,size:body.expected_bytes,chunks:[],offset:0,key:body.idempotency_key});return Response.json({id,state:'uploading',size_bytes:0});
  }
  const block=/^uploads\/([^/]+)\/blocks\?offset=(\d+)$/.exec(path);
  if(block&&localAssetId(decodeURIComponent(block[1]))){const pending=this.uploads.get(decodeURIComponent(block[1]));if(!pending||pending.offset!==Number(block[2]))throw new LocalStorageError('本机导入进度已变化，请重新选择文件。');const raw=init.body instanceof Blob?init.body:new Blob([init.body as ArrayBuffer]);if(pending.offset+raw.size>pending.size)throw new LocalStorageError('本机导入长度不一致。');pending.chunks.push(raw);pending.offset+=raw.size;return Response.json({id:pending.id,state:'uploading',size_bytes:pending.offset});}
  const finish=/^uploads\/([^/]+)\/finalize$/.exec(path);
  if(finish&&localAssetId(decodeURIComponent(finish[1]))){const id=decodeURIComponent(finish[1]),saved=await this.files.get(id);if(saved){if(body?.sha256&&saved.sha256!==body.sha256)throw new LocalStorageError('本机导入校验失败。');return Response.json(this.view(saved));}const pending=this.uploads.get(id);if(!pending||pending.offset!==pending.size)throw new LocalStorageError('本机导入未完成，请重试。');const blob=new Blob(pending.chunks);if(body?.sha256&&await localDigest(blob)!==body.sha256)throw new LocalStorageError('本机导入校验失败。');this.check(signal);const file=await this.files.save(pending.name,blob,pending.key,pending.id);this.check(signal);this.uploads.delete(pending.id);return Response.json(this.view(file));}
  const asset=/^assets\/([^/?]+)(.*)$/.exec(path);
  if(asset){const id=decodeURIComponent(asset[1]),suffix=asset[2],file=await this.files.get(id);
   if(file&&method==='GET'&&(!suffix||suffix==='/content'))return suffix?new Response(file.blob):Response.json(this.view(file));
   if(file&&method==='DELETE'&&!suffix&&localAssetId(id)){await this.files.remove(id);return Response.json({...this.view(file),state:'deleted'});}
   if(file&&suffix&&suffix!='/content'&&suffix!=='/delete-impact')path='assets/'+encodeURIComponent(await this.server(file,signal))+suffix;
   if(!file&&localAssetId(id))throw new LocalStorageError('本机文件不存在或属于其他账号，请重新导入。');
  }
  if(path==='jobs/m16/project?project_id='+this.project&&method==='GET'){
   const id=await this.files.value<string>('recording-project');if(id){const file=await this.files.get(id);if(!file)throw new LocalStorageError('本机录音工程缺少保存文件。');return Response.json({asset:this.view(file)});}
   // Remember only the legacy pointer, so the first local save preserves its CAS baseline.
   const response=await this.remote(path,init);if(response.ok){const value=await response.clone().json();await this.files.setValue('recording-legacy',value.asset?.id??null);}return response;
  }
  if(path==='jobs/m16/project'&&method==='POST'){
   const file=await this.files.get(body.asset_id);if(!file)throw new LocalStorageError('录音工程文件尚未保存在本机。');let expected=body.expected??null;if(!(await this.files.value('recording-project'))&&expected===(await this.files.value('recording-legacy')))expected=null;await this.files.compare('recording-project',expected,body.asset_id);return Response.json({id:body.asset_id});
  }
  if((path.startsWith('jobs/')||path.startsWith('papers/'))&&method==='POST'&&body){this.check(signal);init={...init,body:JSON.stringify(await this.payload(body,signal))};}
  if(path.startsWith('assets?')&&method==='GET'){
   const local=await this.files.list();let response:Response;try{response=await this.remote(path,init);}catch(error){this.check(signal);if(!local.length)throw error;return Response.json({assets:local.map(f=>this.view(f))});}
   if(!response.ok)return response;const value=await response.json();const merged=new Map((value.assets??[]).map((f:any)=>[f.id,f]));for(const file of local)merged.set(file.id,this.view(file));return Response.json({...value,assets:[...merged.values()]});
  }
  if(asset&&method==='GET'&&asset[2]==='/content'){
   const response=await this.remote(path,init);if(!response.ok)return response;const raw=await response.clone().blob();const metaResponse=await this.remote('assets/'+asset[1],{signal});if(metaResponse.ok){const meta=await metaResponse.json();if(meta.project_id===this.project&&meta.sha256===await localDigest(raw)){const file=await this.files.save(meta.name,raw,undefined,decodeURIComponent(asset[1]));file.server={id:file.id,expires_at:meta.expires_at,project:this.project};await this.files.remember(file);}}return response;
  }
  return this.remote(path,init);
 }
}
