import test from 'node:test';import assert from 'node:assert/strict';
import {serverFiles,sha256} from '../src/platform/research.ts';
import {serializeGrid} from '../src/modules/annotation/format.ts';

test('M12 server save resumes partial uploads and retries a lost finalize response without duplicate versions',async()=>{
 const original=serializeGrid({xmin:0,xmax:2,tiers:[{name:'words',intervals:[{xmin:0,xmax:2,text:'原始'}]}]}),sourceBytes=new TextEncoder().encode(original).buffer;
 const source={id:'source',project_id:'project',name:'a.TextGrid',state:'ready',size_bytes:sourceBytes.byteLength,sha256:await sha256(sourceBytes),created_at:0,expires_at:9999999999};
 const uploads=new Map<string,any>();let metadataWhileUploading=0,failFinalize=true,blocks=0;
 const json=(v:unknown,status=200)=>new Response(JSON.stringify(v),{status,headers:{'Content-Type':'application/json'}});
 const previous=globalThis.fetch;
 globalThis.fetch=async(input,options)=>{
  const url=String(input),method=options?.method??'GET',body=typeof options?.body==='string'?JSON.parse(options.body):undefined;
  if(url.includes('assets?'))return json({assets:[source,...[...uploads.values()].map(u=>u.asset)]});
  if(url.endsWith('assets/source'))return json(source);
  if(url.endsWith('assets/source/content'))return new Response(sourceBytes);
  if(url.endsWith('/uploads')&&method==='POST'){
   if(!uploads.has(body.idempotency_key))uploads.set(body.idempotency_key,{asset:{...source,id:'saved',name:body.name,state:'uploading',size_bytes:0,sha256:null},bytes:new Uint8Array(body.expected_bytes)});
   return json(uploads.get(body.idempotency_key).asset);
  }
  const upload=[...uploads.values()][0];
  if(url.includes('/blocks?')){blocks++;const offset=Number(new URL(url,'http://owned.test').searchParams.get('offset')),bytes=new Uint8Array(options!.body as ArrayBuffer);assert.equal(offset,upload.asset.size_bytes);upload.bytes.set(bytes,offset);upload.asset.size_bytes+=bytes.length;return json(upload.asset);}
  if(url.endsWith('/finalize')){upload.asset.state='ready';upload.asset.sha256=await sha256(upload.bytes.buffer);if(failFinalize){failFinalize=false;throw Error('injected lost response');}return json(upload.asset);}
  if(url.endsWith('assets/saved')){metadataWhileUploading++;return json({detail:'asset_unavailable'},410);}
  throw Error('Unexpected request '+url);
 };
 const files=serverFiles('owner','project',()=>assert.fail('unexpected account change'),()=> 'csrf');
 try{
  const port=files.annotation!,target=await port.target({id:'wave',name:'a.wav',size:1,kind:'audio'},'textgrid');
  const body={target:target.id,source:{id:source.id,sha256:source.sha256},text:original.replace('原始','网页 æ')};
  await assert.rejects(port.save(body),/lost response/);const result=await port.save(body);
  assert.equal(uploads.size,1);assert.equal(metadataWhileUploading,0);assert.equal(blocks,1);assert.equal(result.sha256,await sha256(new TextEncoder().encode(body.text).buffer));
 }finally{files.dispose();globalThis.fetch=previous;}
});
