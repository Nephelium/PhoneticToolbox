import test from 'node:test';
import assert from 'node:assert/strict';
import {Runner} from '../src/modules/perception/runner.ts';
import {defaults,newSession,generate,relink,type Asset,type Session} from '../src/modules/perception/model.ts';
import {rowsToTrials} from '../src/modules/perception/formats.ts';
import type {MediaBank} from '../src/modules/perception/media.ts';
import type {LocalStore} from '../src/modules/perception/storage.ts';

function fixture(){
 const p=defaults();p.questionnaire=[];p.config.advanceMode='manual';p.config.useBeep=false;
 p.assets=[{id:'x',hash:'a'.repeat(64),name:'x.wav',path:'x.wav',group:'X',kind:'audio',size:44,mime:'audio/wav',originalSampleRate:48000} as Asset];p.trials=generate(p);
 let fail=false;const saved:Session[]=[];
 const store={async commit(s:Session){if(fail)throw Error('quota');saved.push(structuredClone(s));s.revision++;}} as unknown as LocalStore;
 const r=new Runner(newSession(p,{}),{} as MediaBank,store,()=>{});
 return {r,saved,setFail:(v:boolean)=>fail=v};
}
test('M15-R1 ending an already completed session preserves natural completion and revision',async()=>{
 const {r}=fixture();r.session.status='completed';r.session.nextIndex=1;r.phase='completed';const revision=r.session.revision;
 await r.end();assert.equal(r.session.status,'completed');assert.equal(r.session.revision,revision);assert.equal(r.phase,'completed');
});
test('M15-R1 ending twice is idempotent and cannot resume a terminal session',async()=>{
 const {r,saved}=fixture();await Promise.all([r.end(),r.end()]);assert.equal(r.session.status,'ended');assert.equal(r.phase,'completed');
 assert.equal(saved.length,1);await r.start();assert.equal(r.phase,'completed');
});
test('M15-R1 failed end retains saving-error; retry restores terminal phase',async()=>{
 const {r,setFail}=fixture();setFail(true);await r.end();assert.equal(r.phase,'saving-error');setFail(false);await r.retrySave();assert.equal(r.phase,'completed');assert.equal(r.session.status,'ended');
});
test('M15-R1 failed dispose does not permit dropping the only in-memory response',async()=>{
 const {r,setFail}=fixture();r.phase='paused';setFail(true);const closed=await r.dispose();assert.equal(closed,false);assert.equal(r.phase,'saving-error');
 setFail(false);await r.retrySave();assert.equal(await r.dispose(),true);
});
test('M15-R1 cannot confirm a download that was never requested',async()=>{
 const {r}=fixture();await assert.rejects(()=>r.confirmExport(),/导出/);assert.equal(r.session.exportConfirmedRevision,null);
});
test('M15-R1 a new interruption invalidates the previous download confirmation',async()=>{
 const {r}=fixture();await r.exportRequested();await r.confirmExport();assert.equal(r.session.exportConfirmedRevision,r.session.revision);
 await r.interrupt('test-new-event');await assert.rejects(()=>r.confirmExport(),/导出/);assert.equal(r.session.exportConfirmedRevision,null);
});
test('M15-R1 identical content in A and X relinks and imports by role',()=>{
 const p=defaults();p.paradigm='AX';const shared={hash:'a'.repeat(64),name:'same.wav',path:'same.wav',kind:'audio',size:44,mime:'audio/wav',originalSampleRate:48000};
 p.assets=[{...shared,id:'a',group:'A'},{...shared,id:'x',group:'X'}] as Asset[];p.trials=generate(p);
 p.trials[0].stimuli.A!.id=null;p.trials[0].stimuli.X!.id=null;relink(p);
 assert.equal(p.trials[0].stimuli.A?.id,'a');assert.equal(p.trials[0].stimuli.X?.id,'x');
 const rows=rowsToTrials([['File_A','File_X','SHA256_A','SHA256_X'],['same.wav','same.wav',shared.hash,shared.hash]],p);
 assert.equal(rows[0].stimuli.A?.id,'a');assert.equal(rows[0].stimuli.X?.id,'x');
});
test('M15-R1 ending during preparation waits for the write without a late extra save or playback',async()=>{
 const {r,saved}=fixture();const commit=r.store.commit.bind(r.store);let release!:()=>void,entered!:()=>void;
 const started=new Promise<void>(resolve=>entered=resolve),gate=new Promise<void>(resolve=>release=resolve);let first=true;
 r.store.commit=async s=>{if(first){first=false;entered();await gate;}await commit(s);};
 await r.start();const preparing=r.begin();await started;const ending=r.end();release();await Promise.all([preparing,ending]);
 assert.equal(r.phase,'completed');assert.equal(r.session.status,'ended');assert.equal(r.session.attempts[0].status,'interrupted');assert.equal(saved.length,2);
});
