import {test} from 'node:test';
import assert from 'node:assert/strict';
import {BrowserRecordingProject} from '../src/platform/browser-recording-project.ts';
import type {Project,Take} from '../src/modules/recording/types.ts';
import type {RecordingServer} from '../src/platform/recording-server.ts';
const project=():Project=>({schema_version:'ptb-recording/1',id:'project',revision:0,label:'synthetic',tasks:[],takes:[],selected:{},recoveries:[]});
const take=():Take=>({id:'take',started_at:'2026-10-16T00:00:00Z',config:{device:'default',sample_rate:48000,channels:1,roles:['microphone'],gain_db:0},task_snapshot:null,versions:[{id:'raw',kind:'raw',frames:10,created_at:'now',metadata:{}}],head:0,undo:[],redo:[],status:'complete',error:'',quality:{raw_clip:[0],near_full_scale:[0],peak:[.1],rms:[.1]},frames:10});
const asset={id:'asset',name:'test.wav',kind:'audio' as const,size:84,sha256:'a'.repeat(64)};
test('failed upload retains identical File; retry saves original, selection and full snapshot',async()=>{
 const p=project(),files:File[]=[],snapshots:any[]=[];let fail=true;
 const server={async upload(file:File){files.push(file);if(fail){fail=false;throw Error('offline');}return asset;},async save(snapshot:any,expected:string|null){snapshots.push({snapshot:structuredClone(snapshot),expected});return 'snapshot';}} as unknown as RecordingServer;
 const m=new BrowserRecordingProject(p,server),file=new File([new Uint8Array(84)],'test.wav');m.add(take(),file,[[new Float32Array(10)]]);
 await assert.rejects(m.save(),/offline/);assert.equal(m.pending.size,1);assert.equal(m.cache.get('raw')?.file,file);
 await m.save();assert.deepEqual(files,[file,file]);assert.equal(m.pending.size,0);assert.equal(snapshots[0].snapshot.assets.raw.sha256,asset.sha256);assert.equal(snapshots[0].snapshot.project.selected.__free__,'take');assert.equal(m.expected,'snapshot');await m.save();assert.equal(snapshots.length,1);
});
test('reload preserves saved versions and undo/redo navigation without rebuilding audio',async()=>{
 const p=project(),t=take();t.versions.push({id:'gain',kind:'gain',frames:10,created_at:'now',metadata:{parent:'raw'}});t.head=1;t.undo=[0];p.takes.push(t);p.selected.__free__=t.id;
 const server={async load(){return {id:'manifest',snapshot:{schema:'m16-web-project/1',project:p,assets:{raw:asset,gain:{...asset,id:'derived'}}}};},async save(){return 'new-manifest';}} as unknown as RecordingServer;
 const restored=project(),m=new BrowserRecordingProject(restored,server);await m.load();assert.equal(restored.takes[0].head,1);await m.navigate(restored.takes[0],'undo');assert.equal(restored.takes[0].head,0);await m.navigate(restored.takes[0],'redo');assert.equal(restored.takes[0].head,1);assert.equal(restored.takes[0].versions[0].id,'raw');
});
test('cancelled processing never publishes a version or replaces the source',async()=>{
 const p=project(),t=take();p.takes.push(t);let cancelled=false;
 const server={async create(){return {id:'job',state:'queued'};},async cancel(){cancelled=true;return {id:'job',state:'cancel_requested'};},async job(){return {id:'job',state:cancelled?'cancelled':'running',progress:.2,error_code:cancelled?'cancelled':null};}} as unknown as RecordingServer;
 const m=new BrowserRecordingProject(p,server);m.assets.raw=asset;await m.process(t,'gain',{channels:[0],gain_db:6});assert.equal((await m.poll())?.state,'running');await m.cancel();assert.equal((await m.poll())?.state,'cancelled');assert.equal(t.versions.length,1);assert.equal(t.head,0);assert.equal(m.assets.raw.id,'asset');
});
test('capture checkpoints survive reload and are removed only after full take snapshot commit',async()=>{
 const p=project(),snapshots:any[]=[],removed:string[]=[];let fail=false;
 const server={async upload(){return asset;},async remove(a:typeof asset){removed.push(a.id);},async save(s:any){if(fail)throw Error('save offline');snapshots.push(structuredClone(s));return 'snapshot';},async load(){return {id:'snapshot',snapshot:snapshots.at(-1)};}} as unknown as RecordingServer;
 const m=new BrowserRecordingProject(p,server),t=take(),file=new File([new Uint8Array(84)],'checkpoint.wav');m.beginCapture('take',t.config,null);await m.checkpoint('take',0,file,10);assert.equal(snapshots[0].captures.take.chunks.length,1);assert.deepEqual(removed,[]);
 const loaded=new BrowserRecordingProject(project(),server);await loaded.load();assert.equal(loaded.project.recoveries[0].frames,10);
 m.add(t,file,[[new Float32Array(10)]]);fail=true;await assert.rejects(m.save(),/offline/);assert(m.captures.take);assert.deepEqual(removed,[]);fail=false;await m.save();assert.equal(m.captures.take,undefined);assert.equal(p.recoveries.length,0);assert.deepEqual(removed,['asset']);assert.deepEqual(snapshots.at(-1).captures,{});
});
test('export scopes select task takes/history and exact frame ranges without changing the project',async()=>{
 const p=project();p.tasks=[{id:'A',title:'A',prompt:'=1+1',filename_stem:'a',group:'',note:'',enabled:true,skipped:false},{id:'B',title:'B',prompt:'B',filename_stem:'b',group:'',note:'',enabled:true,skipped:true}];
 p.takes=[take(),take(),take()];for(const [i,t]of p.takes.entries()){t.id='take'+i;t.task_snapshot=p.tasks[i]??null;t.versions=[{...t.versions[0],id:'raw'+i},{id:'gain'+i,kind:'gain',frames:10,created_at:'now',metadata:{parent:'raw'+i}}];t.head=1;}p.selected={A:'take0',B:'take1',__free__:'take2'};
 const calls:any[]=[],downloads:File[]=[];let source:any;const server={async create(config:any){source=config;calls.push(config);return {id:'job',state:'succeeded'};},async result(){return {asset:{...asset,id:'export'},metadata:{frames:source.end-source.start}};},async read(){return new ArrayBuffer(44);}} as unknown as RecordingServer;
 const m=new BrowserRecordingProject(p,server);m.download=file=>downloads.push(file);for(const t of p.takes)for(const v of t.versions)m.assets[v.id]={...asset,id:v.id};const before=structuredClone(p);
 const scenarios=[{mode:'current',ids:['gain0']},{mode:'selection',ids:['gain0']},{mode:'selected',ids:['gain0']},{mode:'all',ids:['gain0','gain1','gain2']},{mode:'versions',ids:['raw0','gain0','raw1','gain1','raw2','gain2']}];
 for(const s of scenarios){calls.length=0;await m.export({mode:s.mode,start:2,end:7,subtype:'PCM_16'},p.takes[0]);for(let i=0;i<100&&m.exportState?.state==='running';i++)await new Promise(r=>setTimeout(r,0));assert.equal(m.exportState?.state,'complete');assert.deepEqual(calls.map(c=>c.inputs[0]),s.ids);for(const c of calls){assert.equal(c.start,s.mode==='selection'?2:0);assert.equal(c.end,s.mode==='selection'?7:10);assert.equal(c.subtype,'PCM_16');}const bytes=new Uint8Array(await downloads.at(-1)!.arrayBuffer()),view=new DataView(bytes.buffer),entries=new Map<string,Uint8Array>();let at=0;while(view.getUint32(at,true)===0x04034b50){const size=view.getUint32(at+18,true),n=view.getUint16(at+26,true),extra=view.getUint16(at+28,true),name=new TextDecoder().decode(bytes.slice(at+30,at+30+n)),start=at+30+n+extra;entries.set(name,bytes.slice(start,start+size));at=start+size;}const manifest=JSON.parse(new TextDecoder().decode(entries.get('manifest.json')));assert.equal(manifest.items.length,s.ids.length);assert.equal(manifest.task_summary[1].skipped,true);assert(new TextDecoder().decode(entries.get('manifest.csv')).includes("'=1+1"));}
 assert.deepEqual(p,before);
});
