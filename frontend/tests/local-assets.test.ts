import test from 'node:test';
import assert from 'node:assert/strict';
import {LocalAssets,LocalStorageError,MemoryAssetDatabase,localDigest} from '../src/platform/local-assets.ts';
import {LocalAssetTransport} from '../src/platform/local-asset-transport.ts';
import {serverFiles} from '../src/platform/research.ts';

test('local file namespaces, hashes, CAS conflicts and idempotent receipts survive new adapters',async()=>{
 const db=new MemoryAssetDatabase(),a=new LocalAssets('a','p',db),other=new LocalAssets('b','p',db),project=new LocalAssets('a','q',db);
 const f=await a.save('test.wav',new Blob(['samples']),'once');assert.equal(f.sha256,await localDigest(f.blob));assert.equal((await a.save('test.wav',f.blob,'once')).id,f.id);
 assert.equal(await other.get(f.id),undefined);assert.equal((await project.list()).length,0);await assert.rejects(other.remember(f),LocalStorageError);
 await a.compare('recording',null,f.id);await new LocalAssets('a','p',db).compare('recording',null,f.id);await assert.rejects(a.compare('recording',null,'other'),/另一窗口/);
 await a.remove(f.id);assert.equal(await a.get(f.id),undefined);
});

test('server operations upload only referenced files, preserve keyed hashes, resume lost blocks and reuse valid mirrors',async()=>{
 const files=new LocalAssets('owner','local-default'),input=await files.save('input.wav',new Blob([new Uint8Array(600000)]));await files.save('unused.wav',new Blob(['unused']));
 let sequence=0,offset=0,drop=true,expire=false;const uploads:string[]=[],blocks:number[]=[],keys:string[]=[];let job:any;
 const transport=new LocalAssetTransport(files,async(path,init={})=>{
  const body=typeof init.body==='string'?JSON.parse(init.body):null;
  if(path==='uploads'){keys.push(body.idempotency_key);if(!uploads.includes(body.idempotency_key)){uploads.push(body.idempotency_key);sequence++;offset=0;}assert.equal(body.project_id,'remote');assert.equal(body.name,'input.wav');return Response.json({id:'server'+sequence,state:'uploading',size_bytes:offset});}
  if(path.includes('/blocks?')){assert.equal(Number(new URL(path,'http://test').searchParams.get('offset')),offset);const blob=init.body as Blob;assert(blob.size<=262144);blocks.push(blob.size);offset+=blob.size;if(drop){drop=false;throw Error('lost reply');}return Response.json({id:'server'+sequence,state:'uploading',size_bytes:offset});}
  if(path.endsWith('/finalize')){assert.equal(offset,600000);assert.equal(body.sha256,input.sha256);return Response.json({id:'server'+sequence,state:'ready',sha256:input.sha256,expires_at:Date.now()/1000+10000});}
  if(path.startsWith('assets/'))return expire&&path==='assets/server1'?Response.json({detail:'asset_expired'},{status:410}):Response.json({project_id:'remote',state:'ready',sha256:input.sha256,expires_at:Date.now()/1000+10000});
  assert.equal(path,'jobs/m16/create');job=body;return Response.json({id:'job'});
 },new AbortController().signal,'remote');
 const invoke=()=>transport.fetch('jobs/m16/create',{method:'POST',body:JSON.stringify({config:{inputs:[input.id],hashes:{[input.id]:input.sha256}}})});
 await assert.rejects(invoke(),/lost reply/);await invoke();assert.deepEqual(blocks,[262144,262144,75712]);assert.equal(new Set(keys).size,1);assert.deepEqual(job.config.inputs,['server1']);assert.deepEqual(job.config.hashes,{server1:input.sha256});await invoke();assert.equal(uploads.length,1);
 expire=true;await invoke();assert.equal(uploads.length,2);assert.equal((await files.list()).length,2);assert.deepEqual(job.config.inputs,['server2']);
});

test('local protocol validates before commit and never uploads on missing, cancelled or storage-failed input',async()=>{
 const files=new LocalAssets('owner','project'),abort=new AbortController(),t=new LocalAssetTransport(files,async()=>assert.fail('unexpected network'),abort.signal);
 const begin=await(await t.fetch('uploads',{method:'POST',body:JSON.stringify({name:'x.wav',expected_bytes:3,idempotency_key:'key'})})).json();
 await t.fetch('uploads/'+begin.id+'/blocks?offset=0',{method:'PUT',body:new Blob(['abc'])});await assert.rejects(t.fetch('uploads/'+begin.id+'/finalize',{method:'POST',body:JSON.stringify({sha256:'wrong'})}),/校验/);assert.equal((await files.list()).length,0);
 await assert.rejects(t.fetch('jobs/create',{method:'POST',body:JSON.stringify({asset_id:'ptb-local:missing'})}),LocalStorageError);
 abort.abort();await assert.rejects(t.fetch('uploads',{method:'POST',body:'{}'}),{name:'AbortError'});
});

test('TextGrid preview reads local bytes with full text fidelity and makes no content request',async()=>{
 const db=new MemoryAssetDatabase(),original=globalThis.fetch;let calls=0;globalThis.fetch=async()=>{calls++;return Response.json({assets:[]});};const files=serverFiles('owner','project',()=>{},()=>'csrf',db);
 try{const text='File type = "ooTextFile short"\n"TextGrid"\n0\n1\n<exists>\n1\n"IntervalTier"\n"汉字"\n0\n1\n1\n0\n1\n"a"\n';await files.upload!([new File([text],'x.TextGrid')],()=>{});const f=(await files.list())[0],data=await files.textgrid(f);assert.deepEqual(data.tiers,[{name:'汉字',intervals:[{xmin:0,xmax:1,text:'a'}]}]);assert.equal(data.sha256,f.sha256);assert.equal(calls,1);}
 finally{files.dispose();globalThis.fetch=original;}
});

test('local storage failure during browser compute cannot trigger a server fallback upload',async()=>{
 const db=new MemoryAssetDatabase(),original=globalThis.fetch,worker=globalThis.Worker;let fail=false;
 const get=db.get.bind(db);db.get=async(bucket,key)=>{if(fail)throw new LocalStorageError('storage failed');return get(bucket,key) as any;};
 globalThis.fetch=async()=>Response.json({assets:[]});const files=serverFiles('owner','project',()=>{},()=>'csrf',db);
 try{await files.upload!([new File(['x'],'x.xlsx')],()=>{});const f=(await files.list())[0];globalThis.fetch=async()=>assert.fail('must retain storage error');globalThis.Worker=class{} as any;fail=true;await assert.rejects(files.parameters!(f),/storage failed/);}
 finally{files.dispose();globalThis.fetch=original;globalThis.Worker=worker;}
});

test('server result is cached by digest and can be read after its server copy disappears',async()=>{
 const files=new LocalAssets('owner','project'),blob=new Blob(['result']),sha=await localDigest(blob);let online=true,calls=0;
 const t=new LocalAssetTransport(files,async path=>{assert(online);calls++;return path.endsWith('/content')?new Response(blob):Response.json({project_id:'project',name:'result.wav',sha256:sha,expires_at:1});},new AbortController().signal);
 assert.equal(await(await t.fetch('assets/remote/content')).text(),'result');online=false;assert.equal(await(await t.fetch('assets/remote/content')).text(),'result');assert.equal(calls,2);assert.equal((await files.list())[0].sha256,sha);
});

test('renewed server workspace gets a fresh idempotency scope and text labels are never treated as file references',async()=>{
 const files=new LocalAssets('owner','default'),f=await files.save('a.wav',new Blob(['audio']));f.server={id:'old',project:'expired-workspace',expires_at:1};f.mirrorKey='browser-input-expired-workspace-key';await files.remember(f);let key='',body:any,uploads=0;
 const t=new LocalAssetTransport(files,async(path,init={})=>{const value=typeof init.body==='string'?JSON.parse(init.body):null;if(path==='uploads'){uploads++;key=value.idempotency_key;assert.equal(value.project_id,'new-workspace');return Response.json({id:'new',state:'uploading',size_bytes:0});}if(path.includes('/blocks?'))return Response.json({id:'new',state:'uploading',size_bytes:5});if(path.endsWith('/finalize'))return Response.json({id:'new',sha256:f.sha256,expires_at:9999999999});body=value;return Response.json({ok:true});},new AbortController().signal,'new-workspace');
 await t.fetch('jobs/create',{method:'POST',body:JSON.stringify({label:f.id})});assert.equal(uploads,0);assert.equal(body.label,f.id);
 await t.fetch('jobs/create',{method:'POST',body:JSON.stringify({audio:{asset_id:f.id,sha256:f.sha256}})});assert.equal(uploads,1);assert(key.startsWith('browser-input-new-workspace-'));assert.equal(body.audio.asset_id,'new');
});
