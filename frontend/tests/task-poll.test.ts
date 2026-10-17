import test from 'node:test';
import assert from 'node:assert/strict';
import {sharedJobReads,taskPollDelay,waitForTaskPoll} from '../src/platform/task-poll.ts';
import {serverFiles} from '../src/platform/research.ts';

test('queued reads back off and cancellation wakes immediately',async()=>{
 assert(taskPollDelay('queued')>taskPollDelay('running'));assert(taskPollDelay('running')>=500);
 const abort=new AbortController(),start=Date.now(),wait=waitForTaskPoll('queued',abort.signal);abort.abort();await wait;assert(Date.now()-start<200);
});
test('server project modules share the job list and creation invalidates it',async()=>{
 const original=globalThis.fetch;let lists=0;
 globalThis.fetch=async(url,options)=>{
  if(String(url).includes('jobs?')){lists++;return Response.json({jobs:[{id:'a',operation:'lpc_analysis'},{id:'b',operation:'egg_analysis'}]});}
  if(String(url).includes('assets?')){assert.equal(options?.method,'GET');return Response.json({assets:[]});}
  assert.equal(options?.method,'POST');return Response.json({id:'created'});
 };
 const port=serverFiles('owner-a','project-a',()=>{},()=>''),other=serverFiles('owner-b','project-a',()=>{},()=>'');
 try{
  const [lpc,egg]=await Promise.all([port.tasks!.lpcJobs!(),port.tasks!.eggJobs!(),port.tasks!.reconstructions!()]);assert.equal(lpc.length,1);assert.equal(egg.length,1);assert.equal(lists,1);
  await port.tasks!.lpc!({id:'audio',sha256:'x'} as any,undefined,{} as any,'key');await port.tasks!.lpcJobs!();assert.equal(lists,2);
  await other.tasks!.lpcJobs!();assert.equal(lists,3);
 }finally{port.dispose();other.dispose();globalThis.fetch=original;}
});
test('project reads coalesce, invalidate after mutation, and errors are retryable',async()=>{
 let reads=0,finish!:(v:number)=>void;const cache=sharedJobReads(()=>{reads++;return new Promise<number>(r=>finish=r);});
 const a=cache.read('project'),b=cache.read('project');assert.equal(reads,1);finish(1);assert.deepEqual(await Promise.all([a,b]),[1,1]);assert.equal(await cache.read('project'),1);
 cache.clear();const c=cache.read('project');assert.equal(reads,2);finish(2);assert.equal(await c,2);
 let failures=0;const broken=sharedJobReads(async()=>{failures++;throw Error('offline');});await assert.rejects(broken.read('a'));await assert.rejects(broken.read('a'));assert.equal(failures,2);
});
