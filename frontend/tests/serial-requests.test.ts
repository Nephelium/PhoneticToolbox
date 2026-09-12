import {test} from 'node:test';import assert from 'node:assert/strict';import {serialRequests} from '../src/platform/serial-requests.ts';
test('Qt concurrent result reads and polls run in order even after a failed request',async()=>{
 let running=0;const started:number[]=[];
 const send=serialRequests(async<T>(body:unknown)=>{assert.equal(running++,0);started.push(body as number);await new Promise(r=>setTimeout(r,2));running--;if(body===2)throw Error('Expired result');return body as T;});
 const results=await Promise.allSettled([send(1),send(2),send(3)]);assert.deepEqual(started,[1,2,3]);assert.deepEqual(results.map(r=>r.status),['fulfilled','rejected','fulfilled']);assert.equal(running,0);
});
