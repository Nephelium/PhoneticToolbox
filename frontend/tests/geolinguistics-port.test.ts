import test,{type TestContext} from 'node:test';
import assert from 'node:assert/strict';
import {LocalAssets} from '../src/platform/local-assets.ts';
import {geoDataNames,type GeoFile} from '../src/platform/geolinguistics-directory.ts';

Object.defineProperty(globalThis,'document',{value:{baseURI:'https://atlas.test/'},configurable:true});
const {createGeoPort}=await import('../src/platform/geolinguistics.ts');
const pause=()=>new Promise<void>(resolve=>setImmediate(resolve));
function deferred(){let resolve!:()=>void;const promise=new Promise<void>(r=>{resolve=r;});return {promise,resolve};}
const matrix=new Float64Array(930*930);for(let i=0;i<930;i++)for(let j=0;j<930;j++)matrix[i*930+j]=i===j?0:.25;
const geometry={type:'Polygon',coordinates:[[[0,0],[1,0],[1,1],[0,0]]]};
async function fixture(tag:string,overrides:Record<string,BlobPart>={}){
 const features=Array.from({length:510},(_,i)=>({id:`F${i+1}_P`,categories:['A'],labels:{A:'甲'}}));
 const values:Record<string,unknown>={
  'catalog.json':{version:'m19-local-20261022',features,survey:tag},
  'sites.json':Array.from({length:930},(_,i)=>({id:`S${i+1}`,name:String(i),lon:100,lat:30,admin:null})),
  'observations.json':Object.fromEntries(features.map(f=>[f.id,{values:Array(930).fill('A'),raw:{}}])),
  'classification.json':{sheets:[],columns:[]},'map-index.json':[],
  'paper.json':{points:Array.from({length:930},()=>({mds:[0,0,0],sharp:[],heat:[]}))},
 };
 const files:File[]=[],entries:{path:string;bytes:number;sha256:string}[]=[];
 for(const name of geoDataNames){const part=overrides[name]??(name.endsWith('.f64')?matrix.buffer:JSON.stringify(values[name]??{type:'FeatureCollection',features:[{type:'Feature',properties:{tag},geometry}]}));const file=new File([part],name.split('/').at(-1)!);Object.defineProperty(file,'webkitRelativePath',{value:'data/'+name});files.push(file);entries.push({path:name,bytes:file.size,sha256:[...new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer()))].map(n=>n.toString(16).padStart(2,'0')).join('')});}
 const manifest=new File([JSON.stringify({schema:'ptb-geolinguistics-data/1',version:'m19-local-20261022',profile:'interactive-data',files:entries})],'data-manifest.json');Object.defineProperty(manifest,'webkitRelativePath',{value:'data/data-manifest.json'});files.push(manifest);return files;
}
function saved(files:File[]):GeoFile[]{return files.filter(f=>f.name!=='data-manifest.json').map(blob=>({name:blob.webkitRelativePath.slice(5),blob}));}
function storage(t:TestContext){
 const disk=new Map<string,GeoFile[]>();let writes=0;
 t.mock.method(LocalAssets.prototype,'value',async function(this:LocalAssets){return disk.get(this.owner);});
 t.mock.method(LocalAssets.prototype,'setValue',async function(this:LocalAssets,_key:string,value:GeoFile[]){writes++;disk.set(this.owner,value);});
 t.mock.method(globalThis,'fetch',async()=>{throw Error('unexpected network');});
 return {disk,get writes(){return writes;}};
}

test('candidate validation precedes storage and preserves active resource caches on failure',async t=>{
 const db=storage(t),port=createGeoPort('account-a');await port.useDirectory(await fixture('old'));const oldMap=await port.map('world'),oldMatrix=await port.matrix('main');
 for(const name of geoDataNames.filter(n=>n.endsWith('.f64'))){await assert.rejects(port.useDirectory(await fixture('bad',{[name]:new Uint8Array([1])})),/矩阵/);}
 assert.equal(db.writes,1);assert.equal((await port.load()).catalog.survey,'old');assert.strictEqual(await port.map('world'),oldMap);assert.strictEqual(await port.matrix('main'),oldMatrix);
});
test('matrix contract rejects nonfinite, negative, nonzero diagonal and asymmetric values without rounding',async t=>{
 const db=storage(t),port=createGeoPort('matrix');
 for(const [index,value] of [[1,NaN],[1,Infinity],[1,-.1],[0,.1],[1,.5]]){const bad=matrix.slice();bad[index]=value;await assert.rejects(port.useDirectory(await fixture('bad',{'main.f64':bad.buffer})),/矩阵/);}
 assert.equal(db.writes,0);await port.useDirectory(await fixture('valid'));assert.equal((await port.matrix('main'))[1],.25);
});
test('every map is parsed and GeoJSON coordinates and ring closure are validated before replacement',async t=>{
 const db=storage(t),port=createGeoPort('maps');
 for(const name of geoDataNames.filter(n=>n.startsWith('maps/'))){await assert.rejects(port.useDirectory(await fixture('bad',{[name]:'{'})));}
 for(const geometry of [{type:'Point',coordinates:[0,0]},{type:'Polygon',coordinates:[[[0,0],[1,0],[1,1],[2,2]]]},{type:'LineString',coordinates:[[0,0],[181,0]]}])await assert.rejects(port.useDirectory(await fixture('bad',{'maps/world.json':JSON.stringify({type:'FeatureCollection',features:[{type:'Feature',properties:{},geometry}]})})),/地图/);
 assert.equal(db.writes,0);
});
test('parallel imports commit in invocation order rather than validation completion order',async t=>{
 const db=storage(t),port=createGeoPort('parallel'),old=await fixture('first'),next=await fixture('second'),entered=deferred(),release=deferred();
 const manifest=old.at(-1)!,read=manifest.text.bind(manifest);manifest.text=async()=>{entered.resolve();await release.promise;return read();};
 let nextStarted=false;const nextManifest=next.at(-1)!,nextRead=nextManifest.text.bind(nextManifest);nextManifest.text=async()=>{nextStarted=true;return nextRead();};
 const first=port.useDirectory(old);await entered.promise;const second=port.useDirectory(next);await pause();const wasQueued=!nextStarted;release.resolve();await Promise.all([first,second]);assert(wasQueued);assert.equal(db.writes,2);assert.equal((await port.load()).catalog.survey,'second');
});
test('late saved-resource restoration cannot overwrite an imported resource set',async t=>{
 storage(t);const old=saved(await fixture('old')),next=await fixture('new'),entered=deferred(),release=deferred();
 t.mock.method(LocalAssets.prototype,'value',async()=>{entered.resolve();await release.promise;return old;});
 const port=createGeoPort('restore'),loading=port.load();await entered.promise;await port.useDirectory(next);release.resolve();assert.equal((await loading).catalog.survey,'new');assert.equal((await port.load()).catalog.survey,'new');
});
test('a failed persistent write retains the old resource set and subsequent imports recover',async t=>{
 const db=storage(t),port=createGeoPort('writes');await port.useDirectory(await fixture('old'));const old=await port.map('world');
 t.mock.method(LocalAssets.prototype,'setValue',async()=>{throw Error('quota');});await assert.rejects(port.useDirectory(await fixture('rejected')),/quota/);assert.equal((await port.load()).catalog.survey,'old');assert.strictEqual(await port.map('world'),old);
 t.mock.method(LocalAssets.prototype,'setValue',async function(this:LocalAssets,_key:string,value:GeoFile[]){db.disk.set(this.owner,value);});await port.useDirectory(await fixture('new'));assert.equal((await port.load()).catalog.survey,'new');
});
test('an in-flight remote load retries the complete resource snapshot after an import',async t=>{
 storage(t);const remote=new Map((await fixture('remote')).map(v=>[v.webkitRelativePath.slice(5),v])),entered=deferred(),release=deferred();
 t.mock.method(globalThis,'fetch',async(url:URL)=>{const name=url.pathname.slice('/geolinguistics/'.length);if(name==='catalog.json'){entered.resolve();await release.promise;}return new Response(remote.get(name));});
 const port=createGeoPort('remote'),loading=port.load();await entered.promise;await port.useDirectory(await fixture('imported'));release.resolve();assert.equal((await loading).catalog.survey,'imported');
});
test('late old-snapshot validation failure cannot replace or evict a newly imported generation',async t=>{
 const db=storage(t),entered=deferred(),release=deferred(),oldMatrix=matrix.slice();oldMatrix[1]=oldMatrix[930]=.75;
 const files=await fixture('old',{'main.f64':oldMatrix.buffer}),port=createGeoPort('cache');db.disk.set('cache',saved(files));
 files.find(f=>f.name==='world.json')!.text=async()=>{entered.resolve();await release.promise;throw Error('old request failed');};
 const oldLoad=port.load();await entered.promise;await port.useDirectory(await fixture('new'));
 const fresh=await port.load(),freshMap=await port.map('world'),freshDistance=await port.matrix('main');release.resolve();
 assert.strictEqual(await oldLoad,fresh);assert.equal(freshDistance[1],.25);assert.strictEqual(await port.map('world'),freshMap);assert.strictEqual(await port.matrix('main'),freshDistance);
});
test('automatic loading verifies the same manifest hashes and complete layer contract as folder import',async t=>{
 const db=storage(t),files=await fixture('sidecar'),remote=new Map(files.map(v=>[v.webkitRelativePath.slice(5),v])),requests:string[]=[];
 t.mock.method(globalThis,'fetch',async(url:URL)=>{const name=url.pathname.slice('/geolinguistics/'.length);requests.push(name);return new Response(remote.get(name));});
 const original=remote.get('catalog.json')!;remote.set('catalog.json',new File([(await original.text()).replace('sidecar','changed')],'catalog.json'));
 const port=createGeoPort('sidecar');await assert.rejects(port.load(),/校验/);assert.equal(db.writes,0);
 remote.set('catalog.json',original);requests.length=0;assert.equal((await port.load()).catalog.survey,'sidecar');assert.equal(requests.length,17);assert.equal(requests[0],'data-manifest.json');assert.equal((await port.matrix('main'))[1],.25);await port.map('world');assert.equal(requests.length,17);
 const invalid=await fixture('invalid',{'main.f64':new Uint8Array([1])});remote.clear();for(const file of invalid)remote.set(file.webkitRelativePath.slice(5),file);await assert.rejects(createGeoPort('invalid-sidecar').load(),/矩阵/);assert.equal(db.writes,0);
});
test('persisted resources remain scope-isolated and direct map reads restore their own scope',async t=>{
 const db=storage(t);await createGeoPort('account-a/project-1').useDirectory(await fixture('a'));await createGeoPort('account-b/project-1').useDirectory(await fixture('b'));
 const a=createGeoPort('account-a/project-1'),b=createGeoPort('account-b/project-1');assert.equal((await a.load()).catalog.survey,'a');assert.equal((await b.load()).catalog.survey,'b');assert.equal(db.disk.size,2);
 const direct=createGeoPort('account-a/project-1');assert.deepEqual(await direct.map('world'),await a.map('world'));
});

test('server download validates the complete set once, persists it and reopens offline',async t=>{
 const db=storage(t),files=await fixture('downloaded'),remote=new Map(files.map(v=>[v.webkitRelativePath.slice(5),v]));
 const requests:string[]=[],progress:any[]=[],text=Blob.prototype.text;let manifestReads=0;
 t.mock.method(Blob.prototype,'text',async function(this:Blob){const value=await text.call(this);if(value.includes('"schema":"ptb-geolinguistics-data/1"'))manifestReads++;return value;});
 t.mock.method(globalThis,'fetch',async(url:URL,options:RequestInit)=>{assert.equal(url.origin,'https://www.phonetictoolbox.com');assert.equal(options.credentials,'omit');assert.equal(options.redirect,'error');requests.push(url.pathname);return new Response(remote.get(url.pathname.slice('/app/geolinguistics/'.length)));});
 const port=createGeoPort('download');await port.download(value=>progress.push(value));
 assert.equal(db.writes,1);assert.equal(requests.length,17);assert.equal(manifestReads,1);assert.equal(progress.at(-1).loaded,progress.at(-1).total);assert.equal(progress.at(-1).files,16);
 for(let i=1;i<progress.length;i++)assert.ok(progress[i].loaded>=progress[i-1].loaded);
 t.mock.method(globalThis,'fetch',async()=>{throw Error('offline');});
 assert.equal((await port.load()).catalog.survey,'downloaded');assert.equal((await createGeoPort('download').load()).catalog.survey,'downloaded');
});

test('a corrupt server download preserves prior resources and permits retry',async t=>{
 const db=storage(t),port=createGeoPort('retry');await port.useDirectory(await fixture('old'));const old=await port.map('world');
 const files=await fixture('new'),remote=new Map(files.map(v=>[v.webkitRelativePath.slice(5),v])),original=remote.get('catalog.json')!;
 remote.set('catalog.json',new File([(await original.text()).replace('new','bad')],'catalog.json'));
 t.mock.method(globalThis,'fetch',async(url:URL)=>new Response(remote.get(url.pathname.slice('/app/geolinguistics/'.length))));
 await assert.rejects(port.download(()=>{}),/校验/);assert.equal(db.writes,1);assert.strictEqual(await port.map('world'),old);assert.equal((await port.load()).catalog.survey,'old');
 remote.set('catalog.json',original);await port.download(()=>{});assert.equal((await port.load()).catalog.survey,'new');
});

test('cancelled downloads and server outages never replace the prior persistent set',async t=>{
 const db=storage(t),port=createGeoPort('cancel');await port.useDirectory(await fixture('old'));
 const files=await fixture('new'),remote=new Map(files.map(v=>[v.webkitRelativePath.slice(5),v])),controller=new AbortController();
 t.mock.method(globalThis,'fetch',async(url:URL)=>new Response(remote.get(url.pathname.slice('/app/geolinguistics/'.length))));
 await assert.rejects(port.download(value=>{if(value.files===1)controller.abort();},controller.signal),{name:'AbortError'});
 assert.equal(db.writes,1);assert.equal((await port.load()).catalog.survey,'old');
 t.mock.method(globalThis,'fetch',async()=>new Response('unavailable',{status:503}));await assert.rejects(port.download(()=>{}),/暂时无法下载/);assert.equal(db.writes,1);
});

test('server manifest cannot request outside the runtime allowlist',async t=>{
 const db=storage(t),manifest={schema:'ptb-geolinguistics-data/1',version:'m19-local-20261022',profile:'interactive-data',files:geoDataNames.map((path,i)=>({path:i===0?'../private.json':path,bytes:1,sha256:'0'.repeat(64)}))};let requests=0;
 t.mock.method(globalThis,'fetch',async()=>{requests++;return new Response(JSON.stringify(manifest));});
 await assert.rejects(createGeoPort('allowlist').download(()=>{}),/无效条目/);assert.equal(requests,1);assert.equal(db.writes,0);
});

test('save phase precedes the atomic write and cannot report a misleading cancellation after commit starts',async t=>{
 const db=storage(t),port=createGeoPort('save-phase'),files=await fixture('new'),remote=new Map(files.map(v=>[v.webkitRelativePath.slice(5),v]));
 t.mock.method(globalThis,'fetch',async(url:URL)=>new Response(remote.get(url.pathname.slice('/app/geolinguistics/'.length))));
 const entered=deferred(),release=deferred(),controller=new AbortController();let phase='';
 t.mock.method(LocalAssets.prototype,'setValue',async function(this:LocalAssets,_key:string,value:GeoFile[]){assert.equal(phase,'save');entered.resolve();await release.promise;db.disk.set(this.owner,value);});
 const action=port.download(value=>{phase=value.phase;},controller.signal);await entered.promise;
 controller.abort();release.resolve();await action;
 assert.equal((await port.load()).catalog.survey,'new');assert.equal((await createGeoPort('save-phase').load()).catalog.survey,'new');
});

test('an already cancelled load performs no storage read or GET',async t=>{
 storage(t);let reads=0,requests=0;t.mock.method(LocalAssets.prototype,'value',async()=>{reads++;});t.mock.method(globalThis,'fetch',async()=>{requests++;throw Error('unexpected GET');});
 const controller=new AbortController();controller.abort();await assert.rejects(createGeoPort('already-cancelled').load(controller.signal),{name:'AbortError'});assert.equal(reads,0);assert.equal(requests,0);
});

test('cancelling the last consumer aborts the actual GET and a new load can retry', {timeout:10000},async t=>{
 storage(t);const entered=deferred(),controller=new AbortController();let actualSignal:AbortSignal|undefined;
 t.mock.method(globalThis,'fetch',async(_url:URL,options:RequestInit)=>{actualSignal=options.signal!;entered.resolve();return new Promise<Response>((_resolve,reject)=>actualSignal!.addEventListener('abort',()=>reject(actualSignal!.reason),{once:true}));});
 const port=createGeoPort('last-consumer'),action=assert.rejects(port.load(controller.signal),{name:'AbortError'});await entered.promise;controller.abort();await action;assert.equal(actualSignal?.aborted,true);assert.notStrictEqual(actualSignal,controller.signal);
 const remote=new Map((await fixture('retry')).map(v=>[v.webkitRelativePath.slice(5),v]));t.mock.method(globalThis,'fetch',async(url:URL)=>new Response(remote.get(url.pathname.slice('/geolinguistics/'.length))));assert.equal((await port.load()).catalog.survey,'retry');
});

test('one cancelled consumer does not abort the shared no-signal map consumer', {timeout:10000},async t=>{
 storage(t);const remote=new Map((await fixture('shared')).map(v=>[v.webkitRelativePath.slice(5),v])),entered=deferred(),release=deferred(),controller=new AbortController();let actualSignal:AbortSignal|undefined,requests=0;
 t.mock.method(globalThis,'fetch',async(url:URL,options:RequestInit)=>{requests++;const name=url.pathname.slice('/geolinguistics/'.length);if(name==='data-manifest.json'){actualSignal=options.signal!;entered.resolve();await release.promise;}options.signal?.throwIfAborted();return new Response(remote.get(name));});
 const port=createGeoPort('shared'),cancelled=assert.rejects(port.load(controller.signal),{name:'AbortError'});await entered.promise;const surviving=port.map('world');controller.abort();await cancelled;assert.equal(actualSignal?.aborted,false);release.resolve();
 assert.strictEqual(await surviving,await port.map('world'));assert.equal((await port.load()).catalog.survey,'shared');assert.equal(requests,17);
});

test('retry starts before a cancelled run settles and old finally cannot clear the new shared run', {timeout:10000},async t=>{
 storage(t);const remote=new Map((await fixture('fresh')).map(v=>[v.webkitRelativePath.slice(5),v])),oldEntered=deferred(),oldRelease=deferred(),newEntered=deferred(),newRelease=deferred(),controller=new AbortController();let manifests=0;
 t.mock.method(globalThis,'fetch',async(url:URL)=>{const name=url.pathname.slice('/geolinguistics/'.length);if(name==='data-manifest.json'&&++manifests===1){oldEntered.resolve();await oldRelease.promise;throw Error('late cancelled request');}if(name==='catalog.json'){newEntered.resolve();await newRelease.promise;}return new Response(remote.get(name));});
 const port=createGeoPort('immediate-retry'),cancelled=assert.rejects(port.load(controller.signal),{name:'AbortError'});await oldEntered.promise;controller.abort();await cancelled;
 const fresh=port.load();await newEntered.promise;oldRelease.resolve();await pause();const shared=port.map('world');newRelease.resolve();assert.equal((await fresh).catalog.survey,'fresh');assert.strictEqual(await shared,await port.map('world'));assert.equal(manifests,2);
});

test('each generation parses every resource and map once and reuses validated matrix buffers', {timeout:10000},async t=>{
 storage(t);const files=await fixture('first'),reads=new Map<string,number>(),buffers:ArrayBuffer[]=[];
 for(const file of files){const name=file.webkitRelativePath.slice(5),text=file.text.bind(file);file.text=async()=>{reads.set(name,(reads.get(name)??0)+1);return text();};if(name==='main.f64'){const arrayBuffer=file.arrayBuffer.bind(file);file.arrayBuffer=async()=>{const buffer=await arrayBuffer();buffers.push(buffer);return buffer;};}}
 const port=createGeoPort('counts');await port.useDirectory(files);const first=await port.load(),world=await port.map('world'),distance=await port.matrix('main'),before=[...reads];
 for(const name of geoDataNames.filter(n=>n.endsWith('.json')))assert.equal(reads.get(name),1,name);
 assert.equal(buffers.length,2);assert.strictEqual(distance.buffer,buffers[1]);
 for(let i=0;i<3;i++){assert.strictEqual(await port.load(),first);for(const name of ['world','territory','province','city','district'])await port.map(name);for(const name of ['main','type-0-0','type-0-1','type-1-0','type-1-1'])await port.matrix(name);}
 assert.deepEqual([...reads],before);assert.strictEqual(await port.map('world'),world);assert.strictEqual(await port.matrix('main'),distance);assert.equal(buffers.length,2);
 await port.useDirectory(await fixture('second'));assert.notStrictEqual(await port.load(),first);assert.notStrictEqual(await port.map('world'),world);assert.notStrictEqual(await port.matrix('main'),distance);assert.equal((await port.load()).catalog.survey,'second');
});

test('cancellation between map blob reading and parsing stops remaining validation layers', {timeout:10000},async t=>{
 const db=storage(t),files=await fixture('cancel-map'),controller=new AbortController();db.disk.set('cancel-map',saved(files));let laterReads=0;
 const world=files.find(f=>f.name==='world.json')!,read=world.text.bind(world);world.text=async()=>{controller.abort();return read();};const territory=files.find(f=>f.name==='territory.json')!,later=territory.text.bind(territory);territory.text=async()=>{laterReads++;return later();};
 await assert.rejects(createGeoPort('cancel-map').load(controller.signal),{name:'AbortError'});await pause();assert.equal(laterReads,0);assert.equal(db.writes,0);
});
