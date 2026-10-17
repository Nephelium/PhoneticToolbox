import test from 'node:test';
import assert from 'node:assert/strict';
import {serverPapers} from '../src/platform/papers-server.ts';
function stream(value:unknown){return new Response(JSON.stringify({progress:{received:3,total:8}})+'\n'+JSON.stringify({ok:true,value})+'\n',{headers:{'Content-Type':'application/x-ndjson'}});}

test('web papers carry the current account, project and CSRF; parse streamed progress',async()=>{
 const original=globalThis.fetch,seen:any[]=[],progress:any[]=[];
 globalThis.fetch=async(url,init)=>{seen.push({url,init});return stream({firstLaunch:'2026-10-16',papers:[]});};
 try{const port=serverPapers('owner-a','project-a',()=> 'csrf-a',()=>assert.fail('unexpected logout'),new AbortController().signal);
 const result=await port.downloadSelected(['one'],undefined,p=>progress.push(p));assert.equal(result.firstLaunch,'2026-10-16');
 assert.deepEqual(progress,[{received:3,total:8}]);assert.equal(seen[0].url,'/api/v1/papers/project-a/request');
 assert.equal(seen[0].init.headers['X-PTB-Account'],'owner-a');assert.equal(seen[0].init.headers['X-CSRF-Token'],'csrf-a');
 assert.deepEqual(JSON.parse(seen[0].init.body),{operation:'downloadSelected',args:{ids:['one']}});
 }finally{globalThis.fetch=original;}
});
test('separate account ports never share requests, state or status listeners',async()=>{
 const original=globalThis.fetch;globalThis.fetch=async(_,init)=>stream({folders:[(init!.headers as Record<string,string>)['X-PTB-Account']]});
 try{const a=serverPapers('a','pa',()=> 'ca',()=>{},new AbortController().signal),b=serverPapers('b','pb',()=> 'cb',()=>{},new AbortController().signal);
 assert.deepEqual((await a.status()).folders,['a']);assert.deepEqual((await b.status()).folders,['b']);
 }finally{globalThis.fetch=original;}
});
test('expired account invalidates its session and incomplete replies never count as success',async()=>{
 const original=globalThis.fetch;let invalid=0;const port=serverPapers('a','p',()=> 'c',()=>invalid++,new AbortController().signal);
 try{globalThis.fetch=async()=>new Response(JSON.stringify({detail:'authentication_required'}),{status:401});
 await assert.rejects(port.status(),/登录已过期/);assert.equal(invalid,1);
 globalThis.fetch=async()=>new Response('{"progress":{"received":0,"total":1}}\n');await assert.rejects(port.status(),/响应中断/);
 }finally{globalThis.fetch=original;}
});
test('cancelled queued work does not reach the server after another request completes',async()=>{
 const original=globalThis.fetch;let release:(()=>void)|undefined,calls=0;
 globalThis.fetch=async()=>{calls++;await new Promise<void>(resolve=>{release=resolve;});return stream({papers:[]});};
 try{const port=serverPapers('a','p',()=> 'c',()=>{},new AbortController().signal),controller=new AbortController();
 const first=port.status(),second=port.refresh(controller.signal);await new Promise(resolve=>setTimeout(resolve,0));controller.abort();release!();await first;await assert.rejects(second,/取消/);assert.equal(calls,1);
 }finally{globalThis.fetch=original;}
});
