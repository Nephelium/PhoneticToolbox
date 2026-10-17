import test from 'node:test';
import assert from 'node:assert/strict';
import {BrowserScience} from '../src/platform/browser-science.ts';
import {reactive} from 'vue';

class FakeWorker {
 onmessage:any;onerror:any;stopped=false;sent:any[]=[];
 postMessage(value:any){this.sent.push(structuredClone(value));}
 terminate(){this.stopped=true;}
 result(id:number,result:unknown){this.onmessage?.({data:{id,result}});}
}
test('scientific worker serializes operations and ignores late responses',async()=>{
 const worker=new FakeWorker(),port=new BrowserScience(()=>worker as any);
 const a=port.run('parameters',new ArrayBuffer(8),{}),b=port.run('spectrogram',new ArrayBuffer(8),{});
 assert.equal(worker.sent.length,1);worker.result(99,'stale');assert.equal(worker.sent.length,1);
 worker.result(1,'table');assert.equal(await a,'table');assert.equal(worker.sent.length,2);worker.result(2,'spectrum');assert.equal(await b,'spectrum');port.close();assert(worker.stopped);
});
test('abort terminates active computation and rejects queued work without stale data',async()=>{
 const worker=new FakeWorker(),port=new BrowserScience(()=>worker as any),abort=new AbortController();
 const a=port.run('egg_update',new ArrayBuffer(0),{},abort.signal),b=port.run('parameters',new ArrayBuffer(0),{});
 const done=Promise.all([assert.rejects(a,{name:'AbortError'}),assert.rejects(b,{name:'AbortError'})]);abort.abort();await done;assert(worker.stopped);worker.result(1,'late');port.close();
});
test('queued cancellation preserves the active scientific task',async()=>{
 const worker=new FakeWorker(),port=new BrowserScience(()=>worker as any),abort=new AbortController();
 const a=port.run('parameters',new ArrayBuffer(0),{}),b=port.run('spectrogram',new ArrayBuffer(0),{},abort.signal);
 const done=assert.rejects(b,{name:'AbortError'});abort.abort();await done;assert.equal(worker.stopped,false);worker.result(1,'ok');assert.equal(await a,'ok');assert.equal(worker.sent.length,1);port.close();
});
test('workspace close removes pending work and prevents later submissions',async()=>{
 const worker=new FakeWorker(),port=new BrowserScience(()=>worker as any),a=port.run('parameters',new ArrayBuffer(0),{});const done=assert.rejects(a,{name:'AbortError'});port.close();await done;await assert.rejects(port.run('parameters',new ArrayBuffer(0),{}));
});

test('reactive settings become an immutable serializable worker snapshot',async()=>{
 const worker=new FakeWorker(),port=new BrowserScience(()=>worker as any),settings=reactive({channels:[0],gain_db:3});
 const job=port.run('recording_compute',new ArrayBuffer(0),{config:settings});settings.channels.push(1);settings.gain_db=6;
 assert.deepEqual(worker.sent[0].args,{config:{channels:[0],gain_db:3}});worker.result(1,'done');assert.equal(await job,'done');port.close();
});
