import test from 'node:test';
import assert from 'node:assert/strict';
import {eligible,missing,installPapersChannel,papers,type Paper} from '../src/platform/papers.ts';
test('M18 date boundary includes selected day and excludes future, history skips complete papers',()=>{
 const entries=[{publishedAt:'2026-10-06',downloaded:false},{publishedAt:'2026-10-07',downloaded:true},{publishedAt:'2026-10-07',downloaded:false},{publishedAt:'2026-10-08',downloaded:false}] as Paper[];
 assert.equal(eligible(entries,'2026-10-07','2026-10-07').length,2);
 assert.equal(missing(entries,'2026-10-06','2026-10-07').length,2);
});
test('M18 disconnected browser does not silently fetch remote documents',async()=>{
 installPapersChannel(undefined);await assert.rejects(papers.status(),/桌面版/);
});
test('M18 cancellation removes pending request and ignores late native results',async()=>{
 let ready:(id:string,payload:string)=>void=()=>{};let id='';const cancelled:string[]=[];
 installPapersChannel({ready:{connect(fn){ready=fn;}},progress:{connect(){}},request(key){id=key;},cancel(key){cancelled.push(key);}});
 const controller=new AbortController();const promise=papers.download('2026-10-07',controller.signal);controller.abort();
 await assert.rejects(promise,/取消/);assert.deepEqual(cancelled,[id]);ready(id,JSON.stringify({ok:true,value:{}}));installPapersChannel(undefined);
});
