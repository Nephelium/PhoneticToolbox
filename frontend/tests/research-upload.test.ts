import test from 'node:test';import assert from 'node:assert/strict';
import {serverFiles} from '../src/platform/research.ts';

test('parameter display associates generic result names only through source provenance',async()=>{
 const original=globalThis.fetch;let parent=true;
 globalThis.fetch=async(url)=>{const path=String(url);
  if(path.includes('parents/latest')){assert.equal(new URL(path,'http://test').searchParams.get('table'),'true');return Response.json(parent?{asset_id:'table',name:'result.ptb.sqlite',sha256:'table-sha'}:null);}
  assert(path.endsWith('/assets/table'));return Response.json({project_id:'project',state:'ready',sha256:'table-sha',size_bytes:4096,expires_at:10000});
 };
 const files=serverFiles('owner','project',()=>assert.fail('identity'),()=>'csrf');
 try{const table=await files.associatedParameters!({id:'audio',name:'sample.wav',kind:'audio',sha256:'audio-sha',size:100});assert.equal(table?.id,'table');assert.equal(table?.size,4096);parent=false;assert.equal(await files.associatedParameters!({id:'audio',name:'sample.wav',kind:'audio',sha256:'audio-sha',size:100}),null);}finally{files.dispose();globalThis.fetch=original;}
});

test('server parameter prediction retains model and runtime errors instead of blaming calibration',async()=>{
 const original=globalThis.fetch,files=serverFiles('owner','project',()=>assert.fail('identity'),()=>'csrf');
 try{for(const detail of ['m09_parameter_model_invalid','m09_runtime_unavailable','preview_platform_unverified','invalid_spectrogram_config']){
  globalThis.fetch=async()=>Response.json({detail},{status:503});
  await assert.rejects(files.tasks!.reconstructionPreview!({id:'image',name:'image.png',kind:'image',size:4,sha256:'0'.repeat(64)},{} as any),{message:detail});
 }}finally{files.dispose();globalThis.fetch=original;}
});

test('module upload streams bounded blocks and retries a lost reply without duplicating data',async()=>{
 const original=globalThis.fetch;let offset=0,create=0,drop=true,finalized=false;const keys:string[]=[],blocks:number[]=[];
 const file=new File([new Uint8Array(600000)],'audio.wav');
 globalThis.fetch=async(url,init={})=>{
  const path=String(url),headers=init.headers as Record<string,string>;
  assert.equal(headers['X-PTB-Account'],'owner');assert.equal(headers['X-CSRF-Token'],'csrf');
  if(path.endsWith('/uploads')){const body=JSON.parse(String(init.body));assert.equal(body.project_id,'project');keys.push(body.idempotency_key);create++;}
  else if(path.includes('/blocks?')){assert.equal(Number(new URL(path,'http://test').searchParams.get('offset')),offset);const blob=init.body as Blob;assert(blob.size<=262144);blocks.push(blob.size);offset+=blob.size;if(drop){drop=false;throw Error('lost reply');}}
  else if(path.endsWith('/finalize')){assert.equal(offset,file.size);finalized=true;}
  else assert.fail(path);
  return new Response(JSON.stringify({id:'same',state:finalized?'ready':'uploading',size_bytes:offset}),{status:200});
 };
 const files=serverFiles('owner','project',()=>assert.fail('identity'),()=>'csrf');
 try{await assert.rejects(files.upload!([file],()=>{}),/lost reply/);let progress=0;await files.upload!([file],n=>progress=n);assert.equal(progress,1);assert.equal(create,2);assert.equal(new Set(keys).size,1);assert.deepEqual(blocks,[262144,262144,75712]);assert(finalized);}
 finally{files.dispose();globalThis.fetch=original;}
});

test('module upload rejects oversize or unsupported batches before reserving space and aborts on logout',async()=>{
 const original=globalThis.fetch;let called=0,invalid=0;
 globalThis.fetch=async(_url,init={})=>{called++;assert(init.signal?.aborted===false);return new Response(JSON.stringify({detail:'account_changed'}),{status:403});};
 const files=serverFiles('owner','project',()=>invalid++,()=>'csrf');
 try{
  const large=new File(['x'],'big.wav');Object.defineProperty(large,'size',{value:100000001});
  await assert.rejects(files.upload!([large],()=>{}),/100 MB/);
  await assert.rejects(files.upload!([new File(['x'],'a.wav'),new File(['x'],'run.exe')],()=>{}),/类型/);assert.equal(called,0);
  await assert.rejects(files.upload!([new File(['x'],'a.wav')],()=>{}));assert.equal(invalid,1);
  files.dispose();await assert.rejects(files.upload!([new File(['x'],'a.wav')],()=>{}),/会话已关闭/);assert.equal(called,1);
 }finally{files.dispose();globalThis.fetch=original;}
});
test('recording recovers a lost snapshot commit before publishing a changed capture snapshot',async()=>{
 const original=globalThis.fetch;let sequence=0,drop=true,current:string|null=null;const assets=new Map<string,any>(),expected:(string|null)[]=[];
 globalThis.fetch=async(url,init={})=>{const path=String(url),body=typeof init.body==='string'?JSON.parse(init.body):null;
  if(path.endsWith('/uploads')){const id='asset'+(++sequence),a={id,name:body.name,state:'uploading',size_bytes:body.expected_bytes,project_id:'project'};assets.set(id,a);return Response.json({...a,size_bytes:0});}
  if(path.includes('/blocks?'))return Response.json({});
  if(path.endsWith('/finalize')){const id=path.split('/').at(-2)!,a=assets.get(id);Object.assign(a,{state:'ready',sha256:body.sha256});return Response.json(a);}
  if(path.endsWith('/jobs/m16/project')){expected.push(body.expected);if(current!==body.asset_id){assert.equal(body.expected,current);current=body.asset_id;}if(drop){drop=false;throw Error('lost commit reply');}return Response.json({id:current});}
  assert.fail(path);
 };
 const files=serverFiles('owner','project',()=>assert.fail('identity'),()=>'csrf');
 try{const first={schema:'m16-web-project/1',project:{revision:1},assets:{}} as any;await assert.rejects(files.recording!.save(first,null),/lost commit reply/);const saved=await files.recording!.save({...first,project:{revision:2}},null);assert.equal(saved,'asset2');assert.deepEqual(expected,[null,null,'asset1']);
  const uploaded=await files.recording!.upload(new File(['original'],'recording.wav'));assert.equal(uploaded.size,8);assert.equal(uploaded.name,'recording.wav');
 }finally{files.dispose();globalThis.fetch=original;}
});
test('recording reader reloads the current manifest when a writer supersedes its first asset',async()=>{
 const original=globalThis.fetch;let reads=0;globalThis.fetch=async url=>{const path=String(url);if(path.includes('m16/project'))return Response.json({asset:{id:++reads===1?'old':'new'}});if(path.includes('/old/'))throw Error('superseded during stream');return Response.json({schema:'m16-web-project/1',project:{revision:2},assets:{}});};const files=serverFiles('owner','project',()=>{},()=>'csrf');try{const loaded=await files.recording!.load();assert.equal(loaded!.id,'new');assert.equal(loaded!.snapshot.project.revision,2);assert.equal(reads,2);}finally{files.dispose();globalThis.fetch=original;}
});
