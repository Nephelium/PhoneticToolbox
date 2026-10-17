import {test} from 'node:test';
import assert from 'node:assert/strict';
import {coalescedRequests,desktopReadKey} from '../src/platform/coalesced-requests.ts';
import {serialRequests} from '../src/platform/serial-requests.ts';

test('pending equal reads share work and return independent results; completed reads revalidate',async()=>{
  let sends=0,finish!:(value:unknown)=>void;
  const request=coalescedRequests(<T>()=>{sends++;return new Promise<T>(resolve=>{finish=resolve;});},desktopReadKey);
  const a=request<{rows:number[]}>({op:'parameters',id:'x'}),b=request<{rows:number[]}>({op:'parameters',id:'x'});
  assert.equal(sends,1);finish({rows:[1]});const [one,two]=await Promise.all([a,b]);one.rows[0]=3;assert.deepEqual(two.rows,[1]);
  const again=request({op:'parameters',id:'x'});assert.equal(sends,2);finish({rows:[2]});await again;
});

test('a queued mutation separates reads without clearing the serial queue',async()=>{
  const started:string[]=[],ends:Array<(value:unknown)=>void>=[];
  const send=coalescedRequests(serialRequests(<T>(body:unknown)=>{started.push((body as {op:string}).op);return new Promise<T>(resolve=>ends.push(resolve));}),desktopReadKey);
  const before=send({op:'job',id:'one'}),write=send({op:'cancel_job',id:'one'}),after=send({op:'job',id:'one'});
  await Promise.resolve();assert.deepEqual(started,['job']);ends[0]({state:'running'});assert.deepEqual(await before,{state:'running'});
  await Promise.resolve();ends[1]({state:'cancel_requested'});await write;
  await Promise.resolve();ends[2]({state:'cancelled'});assert.deepEqual(await after,{state:'cancelled'});
  assert.deepEqual(started,['job','cancel_job','job']);
});

test('different views never merge and failed reads can retry',async()=>{
  let calls=0;
  const request=coalescedRequests(async<T>()=>{calls++;throw Error('preview_busy');},desktopReadKey);
  await Promise.all([
    assert.rejects(request({op:'parameters',id:'x',view:{start:0}}),/preview_busy/),
    assert.rejects(request({op:'parameters',id:'x',view:{start:1}}),/preview_busy/),
  ]);assert.equal(calls,2);
  await assert.rejects(request({op:'parameters',id:'x',view:{start:0}}),/preview_busy/);assert.equal(calls,3);
  assert.equal(desktopReadKey({op:'submit'}),null);assert.equal(desktopReadKey({op:'egg_preview_update'}),null);
});
