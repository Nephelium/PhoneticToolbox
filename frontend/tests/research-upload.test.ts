import test from 'node:test';import assert from 'node:assert/strict';
import {MemoryAssetDatabase,LocalAssets} from '../src/platform/local-assets.ts';
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

test('module import stays local, resumes across adapters and shares files without content requests',async()=>{
 const original=globalThis.fetch,database=new MemoryAssetDatabase();let calls=0;
 globalThis.fetch=async url=>{calls++;assert(String(url).includes('assets?'));return Response.json({assets:[]});};
 const file=new File([new Uint8Array(600000)],'audio.wav');let files=serverFiles('owner','project',()=>assert.fail('identity'),()=>'csrf',database);
 try{let progress=0;await files.upload!([file],n=>progress=n);await files.upload!([file],()=>{});assert.equal(progress,1);assert.equal(calls,0);const list=await files.list();assert.equal(list.length,1);const read=await files.read(list[0]);assert.equal(read.buffer.byteLength,600000);assert.equal(calls,1);files.dispose();files=serverFiles('owner','project',()=>{},()=>'csrf',database);assert.equal((await files.list())[0].id,list[0].id);assert.deepEqual(new Uint8Array((await files.read(list[0])).buffer),new Uint8Array(await file.arrayBuffer()));}
 finally{files.dispose();globalThis.fetch=original;}
});
test('local import validates the complete batch and blocks writes after disposal',async()=>{
 const original=globalThis.fetch;globalThis.fetch=async()=>assert.fail('local import must not contact server');const files=serverFiles('owner','project',()=>assert.fail('identity'),()=>'csrf');
 try{const large=new File(['x'],'big.wav');Object.defineProperty(large,'size',{value:100000001});await assert.rejects(files.upload!([large],()=>{}),/100 MB/);await assert.rejects(files.upload!([new File(['x'],'a.wav'),new File(['x'],'run.exe')],()=>{}),/类型/);files.dispose();await assert.rejects(files.upload!([new File(['x'],'a.wav')],()=>{}),/会话已关闭/);}
 finally{files.dispose();globalThis.fetch=original;}
});
test('recording snapshot retries a lost local transaction receipt and restores after a new server workspace',async()=>{
 class LostReceipt extends MemoryAssetDatabase{drop=true;override async compare(key:string,expected:unknown,value:unknown){await super.compare(key,expected,value);if(this.drop){this.drop=false;throw Error('lost commit reply');}}}
 const original=globalThis.fetch,database=new LostReceipt();globalThis.fetch=async()=>assert.fail('recording save is local');let files=serverFiles('owner','old-server-project',()=>{},()=>'csrf',database,'default');
 try{const first={schema:'m16-web-project/1',project:{revision:1},assets:{}} as any;await assert.rejects(files.recording!.save(first,null),/lost commit reply/);const saved=await files.recording!.save({...first,project:{revision:2}},null);assert(saved.startsWith('ptb-local:'));const recorded=await files.recording!.upload(new File(['original'],'recording.wav'));assert.equal(recorded.size,8);files.dispose();files=serverFiles('owner','new-server-project',()=>{},()=>'csrf',database,'default');const loaded=await files.recording!.load();assert.equal(loaded!.id,saved);assert.equal(loaded!.snapshot.project.revision,2);assert.equal(new TextDecoder().decode(await files.recording!.read(recorded)),'original');const other=new LocalAssets('other','default',database);assert.equal((await other.list()).length,0);}
 finally{files.dispose();globalThis.fetch=original;}
});
test('recording reader reloads the current manifest when a writer supersedes its first asset',async()=>{
 const original=globalThis.fetch;let reads=0;globalThis.fetch=async url=>{const path=String(url);if(path.includes('m16/project'))return Response.json({asset:{id:++reads===1?'old':'new'}});if(path.includes('/old/'))throw Error('superseded during stream');return Response.json({schema:'m16-web-project/1',project:{revision:2},assets:{}});};const files=serverFiles('owner','project',()=>{},()=>'csrf');try{const loaded=await files.recording!.load();assert.equal(loaded!.id,'new');assert.equal(loaded!.snapshot.project.revision,2);assert.equal(reads,2);}finally{files.dispose();globalThis.fetch=original;}
});
