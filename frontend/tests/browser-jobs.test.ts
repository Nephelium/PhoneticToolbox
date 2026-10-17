import test from 'node:test';
import assert from 'node:assert/strict';
import {BrowserJobs} from '../src/platform/browser-jobs.ts';
import type {ResearchFile,ReconstructionConfig} from '../src/platform/research.ts';
const source={id:'input',name:'image.png',kind:'image',size:100,sha256:'a'.repeat(64)} as ResearchFile;
const config={mode:'image'} as ReconstructionConfig;
const encode=(text:string)=>new TextEncoder().encode(text).buffer;
function fixture(){
 const assets:any[]=[],raws=new Map<string,ArrayBuffer>();let next=0;
 const ports={async list(){return assets;},async read(asset:any){return raws.get(asset.id)!;},async upload(name:string,raw:ArrayBuffer){const id=String(++next),asset={id,name,size_bytes:raw.byteLength,sha256:id,created_at:next,expires_at:9999999999,policyVersion:1};assets.push(asset);raws.set(id,raw);return asset;},async compute(){return {files:['calibrated.png','reconstructed.png','reconstructed.wav','reconstruction.ptb.json'].map(name=>({name,data:btoa(name.endsWith('.json')?' {"core_version":"3.0.3"}':'test')}))};}};
 return {ports,assets,raws};
}
async function terminal(jobs:BrowserJobs,id:string){for(let i=0;i<30;i++){const job=jobs.get(id)!;if(job.state!=='running')return job;await new Promise(r=>setTimeout(r,0));}throw Error('Local task did not finish');}
test('browser job persists its four managed outputs and restores history without server queue',async()=>{
 const f=fixture(),a=new BrowserJobs('owner','project',f.ports),started=await a.start(source,config),done=await terminal(a,started.id);assert.equal(done.execution,'browser');assert.equal(done.state,'succeeded');assert.equal(done.result_manifest?.kind,'managed_spec2wav_files');await new Promise(r=>setTimeout(r,0));a.close();const b=new BrowserJobs('owner','project',f.ports);const restored=await b.list();assert.equal(restored[0].state,'succeeded');assert.equal(restored[0].result_manifest?.files?.length,4);assert.equal((await b.cancel(started.id)).state,'succeeded');b.close();
});
test('cancel terminates computation and never publishes late outputs',async()=>{
 const f=fixture();let finish:(value:any)=>void=()=>{};f.ports.compute=()=>new Promise(resolve=>finish=resolve);const jobs=new BrowserJobs('owner','project',f.ports),job=await jobs.start(source,config);await jobs.cancel(job.id);finish({files:[]});await new Promise(r=>setTimeout(r,0));assert.equal(jobs.get(job.id)?.state,'cancelled');assert(f.assets.every(a=>a.name.endsWith('.ptb-browser.json')));jobs.close();
});
test('other-account records and malformed user files do not enter browser task history',async()=>{
 const f=fixture(),a=new BrowserJobs('owner','project',f.ports);const job=await a.start(source,config);await terminal(a,job.id);await new Promise(r=>setTimeout(r,0));a.close();await f.ports.upload('重建任务-malformed.ptb-browser.json',encode('null'));const b=new BrowserJobs('other','project',f.ports);assert.deepEqual(await b.list(),[]);b.close();
});
test('failure to persist admission does not leave a phantom running job',async()=>{
 const f=fixture(),original=f.ports.upload;let fail=true;f.ports.upload=async(...args)=>{if(fail)throw Error('quota');return original(...args);};const jobs=new BrowserJobs('owner','project',f.ports);await assert.rejects(jobs.start(source,config),/quota/);fail=false;const job=await jobs.start(source,config);assert.equal((await terminal(jobs,job.id)).state,'succeeded');jobs.close();
});

test('failed final history save is retried and restores the completed result',async()=>{
 const f=fixture(),upload=f.ports.upload;let records=0;
 f.ports.upload=async(name,raw)=>{if(name.endsWith('.ptb-browser.json')&&++records===2)throw Error('connection lost');return upload(name,raw);};
 const a=new BrowserJobs('owner','project',f.ports),job=await a.start(source,config);await terminal(a,job.id);await new Promise(r=>setTimeout(r,0));
 assert.match(a.get(job.id)?.error_code??'',/保存尚未完成/);assert.equal((await a.list())[0].error_code,null);a.close();
 const b=new BrowserJobs('owner','project',f.ports),restored=await b.list();assert.equal(restored.length,1);assert.equal(restored[0].state,'succeeded');assert.equal(restored[0].result_manifest?.files?.length,4);b.close();
});

test('stale running metadata becomes interrupted after reload without recreating compute',async()=>{
 const f=fixture();let compute=0;f.ports.compute=async()=>{compute++;return new Promise(()=>{});};
 const a=new BrowserJobs('owner','project',f.ports),job=await a.start(source,config);a.close();
 const asset=f.assets[0],record=JSON.parse(new TextDecoder().decode(f.raws.get(asset.id)!));record.job.created_at-=181;record.job.updated_at-=181;f.raws.set(asset.id,encode(JSON.stringify(record)));
 const b=new BrowserJobs('owner','project',f.ports);assert.equal((await b.list())[0].state,'interrupted');assert.equal(compute,1);b.close();
});

test('LPC shares managed history lifecycle while keeping its three-file contract separate',async()=>{
 const f=fixture();f.ports.compute=async()=>({files:['lpc.ptb.json','lpc_SPECTRUM.png','lpc_AUDIO.wav'].map(name=>({name,data:btoa(name.endsWith('.json')?'{"runtime":{"versions":{"phonetic-core":"3.0.3"}}}':'test')}))});
 const spec={prefix:'LPC任务-',operation:'lpc_analysis' as const,kind:'managed_lpc_files' as const,names:['lpc.ptb.json','lpc_SPECTRUM.png','lpc_AUDIO.wav']};
 const a=new BrowserJobs('owner','project',f.ports,spec),job=await a.start(source,config),done=await terminal(a,job.id);assert.equal(done.result_manifest?.kind,'managed_lpc_files');assert.equal(done.result_manifest?.core_version,'3.0.3');assert.equal(done.result_manifest?.files?.length,3);await new Promise(r=>setTimeout(r,0));a.close();
 const b=new BrowserJobs('owner','project',f.ports,spec);assert.equal((await b.list())[0].operation,'lpc_analysis');b.close();const unrelated=new BrowserJobs('owner','project',f.ports);assert.deepEqual(await unrelated.list(),[]);unrelated.close();
});
