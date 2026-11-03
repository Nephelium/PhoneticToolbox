import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {deflateRawSync} from 'node:zlib';
import {createBrowserManualPackStore,inspectLocalManualPack,manualPackDownloadAllowed,manualPackSha256,unpackManualArchive,validateManualPack,type ManualPackStorage,type ManualPackRecord} from '../src/platform/manual-packs.ts';

// A neutral, tiny fixture exercises installation. It is never a published pack.
function zip(files:Record<string,Uint8Array>,compressed=false):Uint8Array{
  const local:Buffer[]=[],central:Buffer[]=[];let offset=0;
  for(const [path,data] of Object.entries(files)){
    const packed=compressed?deflateRawSync(data):Buffer.from(data),method=compressed?8:0;
    const name=Buffer.from(path),header=Buffer.alloc(30);header.writeUInt32LE(0x04034b50,0);header.writeUInt16LE(20,4);header.writeUInt16LE(method,8);header.writeUInt32LE(packed.length,18);header.writeUInt32LE(data.length,22);header.writeUInt16LE(name.length,26);
    local.push(header,name,packed);const index=Buffer.alloc(46);index.writeUInt32LE(0x02014b50,0);index.writeUInt16LE(20,4);index.writeUInt16LE(20,6);index.writeUInt16LE(method,10);index.writeUInt32LE(packed.length,20);index.writeUInt32LE(data.length,24);index.writeUInt16LE(name.length,28);index.writeUInt32LE(offset,42);central.push(index,name);offset+=30+name.length+packed.length;
  }
  const directory=Buffer.concat(central),end=Buffer.alloc(22);end.writeUInt32LE(0x06054b50,0);end.writeUInt16LE(Object.keys(files).length,8);end.writeUInt16LE(Object.keys(files).length,10);end.writeUInt32LE(directory.length,12);end.writeUInt32LE(offset,16);return new Uint8Array(Buffer.concat([...local,directory,end]));
}
const bytes=(value:unknown)=>new TextEncoder().encode(JSON.stringify(value));
async function fixture(){
  const sourceDigest='a'.repeat(64),project={schemaVersion:'ptb-manual/1',id:'fixture',language:'en',sourceDigest,title:'Neutral manual',chapters:[{id:'one',title:'One',path:'chapters/one.json'}],assets:[{id:'photo',path:'assets/chinese/photo.png',kind:'image',distribution:'public'},{id:'sound',path:'assets/chinese/sound.wav',kind:'audio',distribution:'public'}]};
  const files:Record<string,Uint8Array>={'project.json':bytes(project),'chapters/one.json':bytes({schemaVersion:'ptb-manual-chapter/1',id:'one',title:'One',body:{type:'doc',content:[{type:'paragraph',content:[{type:'text',text:'Neutral example with F0 = 100 Hz.'}]}]}}),'assets/photo.png':new Uint8Array([1,2,3])};
  const hashes=Object.fromEntries(await Promise.all(Object.entries(files).map(async([path,data])=>[path,await manualPackSha256(data)])));
  files['language-pack.json']=bytes({schemaVersion:'ptb-manual-language-pack/1',locale:'en',version:'fixture.1',sourceDigest,projectId:'fixture',distribution:'public',sharedAssets:['photo','sound'],assetOverrides:{photo:'assets/photo.png'},hashes});
  const archive=zip(files),spec={locale:'en' as const,version:'fixture.1',sourceDigest,sha256:await manualPackSha256(archive),url:'http://127.0.0.1/fixture.zip'};
  return {archive,spec,files};
}
class MemoryStorage implements ManualPackStorage{
  records=new Map<string,ManualPackRecord>();reads:string[]=[];
  async list(){return [...this.records.values()].map(record=>record.metadata);}
  async readFile(locale:string,path:string){this.reads.push(path);return this.records.get(locale)?.files[path];}
  async put(record:ManualPackRecord){this.records.set(record.metadata.locale,record);}
}

test('language pack validates download and every file hash, and reuses shared Chinese media',async()=>{
  const {archive,spec}=await fixture(),record=await validateManualPack(archive,spec);
  assert.equal(record.metadata.locale,'en');assert.equal(record.metadata.assetOverrides.photo,'assets/photo.png');
  assert.equal(record.files['assets/chinese/sound.wav'],undefined);
  await assert.rejects(validateManualPack(archive,{...spec,sha256:'b'.repeat(64)}),/hashMismatch/);
  await assert.rejects(validateManualPack(archive,{...spec,sourceDigest:'b'.repeat(64)}),/invalidManifest/);
});

test('deflated ZIPs emitted by the offline pack format decompress and validate',async()=>{
  const {files,spec}=await fixture(),archive=zip(files,true);
  assert.equal((await validateManualPack(archive,{...spec,sha256:await manualPackSha256(archive)})).metadata.locale,'en');
});

test('unsafe paths and altered archive files are rejected before persistence',async()=>{
  const {files,spec}=await fixture();
  const traversal=zip({...files,'../escape.json':bytes({})});await assert.rejects(unpackManualArchive(traversal),/invalidArchive/);
  const changed=zip({...files,'chapters/one.json':bytes({changed:true})});
  await assert.rejects(validateManualPack(changed,{...spec,sha256:await manualPackSha256(changed)}),/hashMismatch/);
});

test('actual loopback HTTP download installs once and serves JSON and images offline on demand',async()=>{
  const {archive,spec}=await fixture(),storage=new MemoryStorage();let requests=0;
  const server=createServer((_request,response)=>{requests++;response.setHeader('Content-Type','application/zip');response.end(archive);});
  await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
  const address=server.address();assert(address&&typeof address!=='string');
  const store=createBrowserManualPackStore({storage,pageUrl:'http://127.0.0.1/'});
  try{
    await store.install({...spec,url:`http://127.0.0.1:${address.port}/fixture.zip`});
    assert.equal(requests,1);assert.deepEqual(storage.reads,[],'installation does not read/decode media for rendering');
  }finally{await new Promise<void>((resolve,reject)=>server.close(error=>error?reject(error):resolve()));}
  const listed=await store.list();assert.equal(listed.length,1);assert.deepEqual(storage.reads,[],'metadata listing does not load chapter or image blobs');
  assert.equal((await store.readJson('en','project.json') as {title:string}).title,'Neutral manual');assert.deepEqual(storage.reads,['project.json']);
  assert((await store.readAsset('en','assets/photo.png')) instanceof Blob);
  assert.equal(await store.readAsset('en','assets/chinese/sound.wav'),null);
  assert.equal(requests,1,'offline reads do not retry the download');
});

test('a failed install preserves the previously installed pack and local ZIP uses the same checks',async()=>{
  const {archive,spec}=await fixture(),storage=new MemoryStorage(),store=createBrowserManualPackStore({storage});
  const original=await store.importArchive!(spec,new Blob([archive]));
  await assert.rejects(store.importArchive!({...spec,sha256:'b'.repeat(64)},new Blob([archive])),/hashMismatch/);
  assert.deepEqual((await store.list())[0],original);
});

test('download hosts are restricted, and loopback requires a loopback page',()=>{
  assert(manualPackDownloadAllowed('https://www.phonetictoolbox.com/manual/en.zip','https://www.phonetictoolbox.com/'));
  assert(manualPackDownloadAllowed('https://github.com/Nephelium/PhoneticToolbox/releases/download/v3/manual-en.zip'));
  assert(!manualPackDownloadAllowed('https://example.org/manual-en.zip'));
  assert(!manualPackDownloadAllowed('https://github.com/other/project/releases/download/v3/manual-en.zip'));
  assert(!manualPackDownloadAllowed('http://127.0.0.1/manual-en.zip','https://www.phonetictoolbox.com/'));
  assert(manualPackDownloadAllowed('http://127.0.0.1/manual-en.zip','http://localhost/'));
});

test('explicit local ZIP inspection works without a published feed and binds language, project, and Chinese source',async()=>{
  const {archive,spec}=await fixture(),file=new Blob([archive]);
  const inspected=await inspectLocalManualPack(file,{locale:'en',projectId:'fixture',sourceDigest:spec.sourceDigest});
  assert.equal(inspected.sha256,spec.sha256);assert.equal(inspected.version,spec.version);assert(!('url' in inspected));
  for(const expected of [{locale:'ja' as const,projectId:'fixture',sourceDigest:spec.sourceDigest},{locale:'en' as const,projectId:'other',sourceDigest:spec.sourceDigest},{locale:'en' as const,projectId:'fixture',sourceDigest:'b'.repeat(64)}])await assert.rejects(inspectLocalManualPack(file,expected),/incompatible/);
});
