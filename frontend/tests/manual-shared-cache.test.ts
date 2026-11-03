import test from 'node:test';
import assert from 'node:assert/strict';
import {ManualSharedCache} from '../src/manual/shared-cache.ts';
import {manualPackSha256} from '../src/platform/manual-packs.ts';
import type {ManualProject} from '../src/manual/types.ts';

test('explicit shared media caching verifies actual host bytes, deduplicates locales and works without any fetch on reads',async()=>{
 Object.defineProperty(globalThis,'location',{configurable:true,value:new URL('http://127.0.0.1:3456/app')});
 const entries=new Map<string,Response>();
 const cache={match:async(key:string|Request)=>entries.get(typeof key==='string'?key:key.url)?.clone(),keys:async()=>[...entries.keys()].map(url=>new Request(url)),put:async(key:string,response:Response)=>{entries.set(key,response.clone());}};
 const storage={open:async()=>cache} as unknown as CacheStorage;
 const bytes=new TextEncoder().encode('actual packed MP3 bytes'),sha=await manualPackSha256(bytes);
 const project:ManualProject={schemaVersion:'ptb-manual/1',id:'book',title:'Book',sourceDigest:'a'.repeat(64),chapters:[],assets:[
  {id:'example-a',path:'assets/packed/tone.mp3',sha256:sha,kind:'audio',distribution:'software-only',mime:'audio/mpeg'},
  {id:'example-copy',path:'assets/packed/tone-copy.mp3',sha256:sha,kind:'audio',distribution:'software-only'},
  {id:'translated-screen',path:'assets/cn.png',kind:'image',distribution:'public'},
 ]};
 let requests=0,offline=false;
 const fetcher=(async(url:string)=>{requests++;assert.equal(url,'http://127.0.0.1:3456/manual/assets/packed/tone.mp3');if(offline)throw Error('offline');return new Response(bytes);}) as typeof fetch;
 const shared=new ManualSharedCache(project,'/manual/',storage,fetcher),overrides={'translated-screen':'assets/en.png'};
 assert.equal(await shared.ready(overrides),false);assert.equal(requests,0,'ordinary readiness check never downloads');
 await shared.install(overrides);assert.equal(requests,1,'same digest is stored once across IDs');
 assert.equal(await shared.ready(overrides),true);
 const pack={locale:'en',version:'1',sourceDigest:project.sourceDigest!,projectId:project.id,installedAt:'',distribution:'software',assetOverrides:overrides} as const;
 assert.equal((await ManualSharedCache.restore([pack],storage))?.sourceDigest,project.sourceDigest);
 assert.equal(await ManualSharedCache.restore([{...pack,sourceDigest:'b'.repeat(64)}],storage),undefined,'offline base must match installed pack identity');
 offline=true;
 await new ManualSharedCache(project,'/manual/',storage,fetcher).install(overrides);assert.equal(requests,1,'second language reuses verified shared media');
 assert.equal(await(await shared.read(sha))!.text(),'actual packed MP3 bytes');assert.equal(requests,1);
 const changed={...project,sourceDigest:'b'.repeat(64)};
 assert.equal(await new ManualSharedCache(changed,'/manual/',storage,fetcher).ready(overrides),false,'different host source has a different completion index');
 entries.set('http://127.0.0.1:3456/__ptb_manual_shared__/sha256/'+sha,new Response('corrupted'));
 assert.equal(await shared.read(sha),null,'corrupt cached bytes are never rendered');
 assert.equal(await shared.ready(overrides),false,'existing corrupt entry cannot claim offline completeness');
 offline=false;await shared.install(overrides);assert.equal(requests,2,'explicit retry replaces only the corrupt payload');
 assert.equal(await shared.ready(overrides),true);assert.equal(await(await shared.read(sha))!.text(),'actual packed MP3 bytes');
 const changedAsset={...project,assets:project.assets.map(asset=>asset.id==='example-a'?{...asset,sha256:'c'.repeat(64)}:asset)};
 assert.equal(await new ManualSharedCache(changedAsset,'/manual/',storage,fetcher).ready(overrides),false,'new host SHA never uses an older cached asset');
 const authorTree={...project,assets:[...project.assets,{id:'unused-old-media',path:'assets/old.wav',kind:'audio' as const,distribution:'software-only' as const}]};
 const scoped=new ManualSharedCache(authorTree,'/manual/',storage,fetcher),required=new Set(project.assets.map(asset=>asset.id));
 await scoped.install(overrides,required);assert.equal(requests,2,'unreferenced author assets are excluded from offline downloads');
 assert.equal(await scoped.ready(overrides,required),true);
 await assert.rejects(scoped.install(overrides,new Set([...required,'unknown-shared-id'])),/manual.pack.missingFile/);
});

test('failed shared media verification never marks an installation complete',async()=>{
 Object.defineProperty(globalThis,'location',{configurable:true,value:new URL('http://127.0.0.1:3456/app')});
 const entries=new Map<string,Response>(),cache={match:async(key:string)=>entries.get(key)?.clone(),put:async(key:string,response:Response)=>{entries.set(key,response.clone());}};
 const storage={open:async()=>cache} as unknown as CacheStorage;
 const project:ManualProject={schemaVersion:'ptb-manual/1',id:'book',title:'Book',chapters:[],assets:[{id:'example',path:'assets/example.wav',kind:'audio',distribution:'software-only',sha256:'a'.repeat(64)}]};
 const shared=new ManualSharedCache(project,'/manual/',storage,(async()=>new Response('wrong bytes')) as typeof fetch);
 await assert.rejects(shared.install({}),/manual.pack.hashMismatch/);assert.equal(await shared.ready({}),false);assert.equal(entries.size,0);
});
